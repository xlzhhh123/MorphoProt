"""Train MorphoProt on the human liver data (GSE278662) with the paper configuration.

Train sections N123 and N126, test sections Sah73 and Sah80. Compared with the
tonsil/spleen recipe, the liver models
  * train for at most 160 epochs with early-stopping patience 30,
  * z-score every image-feature dimension with the mean/std of the inner-training
    spots of each seed, and
  * build the validation neighborhoods from all training-section spots
    (unlabelled RNA context only; validation protein labels are never inputs).
The checkpoint is chosen by validation mean SCC, as for the other tissues.

Usage:
    python morphoprot/train_liver.py --seeds 42 99 7 1024
Outputs: results/liver/seed_<seed>/ (per-seed predictions and metrics) and
results/liver/ensemble_test_pred.npy, ensemble_metrics.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import core as base  # noqa: E402
import data_liver as liver  # noqa: E402
from paths import RESULTS_ROOT  # noqa: E402


CONFIG = dict(
    LR=4e-4, WEIGHT_DECAY=1e-4, DROPOUT=0.15, RNA_HIDDEN_DIM=768, EMB_DIM=256,
    EPOCHS=160, PATIENCE=30, LAMBDA_CCC=0.15, LAMBDA_RANK=0.05,
    BATCH_SIZE_TRAIN=128, SPATIAL_K=8, FEATURE_K=4, ATT_ALPHA_INIT=0.10, GRAPH_BETA_INIT=0.08,
)
SCHEDULER_PATIENCE = 4
SCALES = [16, 32, 64, 96, 128]


def metrics(y, p, names):
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)

    def corr(a, b):
        a = a - a.mean(0)
        b = b - b.mean(0)
        den = np.sqrt((a * a).sum(0) * (b * b).sum(0))
        return np.divide((a * b).sum(0), den, out=np.zeros(a.shape[1]), where=den > 0)

    per = pd.DataFrame(dict(protein=names, scc=corr(rankdata(y, axis=0), rankdata(p, axis=0)),
                            pcc=corr(y, p), rmse=np.sqrt(((y - p) ** 2).mean(0))))
    summary = dict(scc=float(per.scc.mean()), pcc=float(per.pcc.mean()),
                   pooled_rmse=float(np.sqrt(((y - p) ** 2).mean())))
    return summary, per


def configure():
    for key, value in CONFIG.items():
        setattr(base, key, value)
    base.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    base.build_model = lambda name, rna_dim, he_dims, out_dim, A: base.DualGATv2GraphModel(
        rna_dim, he_dims, out_dim, A, graph_input_mode="fuse").to(base.DEVICE)
    base.export_cross_attention_summary = lambda *a, **kw: None
    original = torch.optim.lr_scheduler.ReduceLROnPlateau

    class Scheduler(original):
        def __init__(self, optimizer, **kw):
            kw["patience"] = SCHEDULER_PATIENCE
            super().__init__(optimizer, **kw)

    torch.optim.lr_scheduler.ReduceLROnPlateau = Scheduler


def slice_pack(pack, ix):
    return {k: ([a[ix] for a in v] if k == "x_he_list" else v[ix]) for k, v in pack.items()}


def prepare(seed):
    x, y, xt, yt, ha, ht, c, ct, names = liver.load_all()
    tr, va = train_test_split(np.arange(len(x)), test_size=0.15, random_state=seed)
    ha = [h.copy() for h in ha]
    ht = [h.copy() for h in ht]
    for i in range(len(ha)):  # image-feature z-score fitted on inner-training spots only
        mu = ha[i][tr].mean(0)
        sd = np.maximum(ha[i][tr].std(0), 1e-5)
        ha[i] = (ha[i] - mu) / sd
        ht[i] = (ht[i] - mu) / sd
    train, pca = liver.make_split_pack(x[tr], y[tr], [h[tr] for h in ha], c[tr], seed=seed)
    test, _ = liver.make_split_pack(xt, yt, ht, ct, pca_model=pca, seed=seed)
    # validation spots keep all training-section spots as RNA neighborhood context
    allpack, _ = liver.make_split_pack(x, y, ha, c, pca_model=pca, seed=seed)
    val = slice_pack(allpack, va)
    test_ids = liver.read_lines(liver.GENE_DIR / "test_barcodes.txt")
    return train, val, test, [h.shape[1] for h in ha], names, test_ids


def fit(seed, out_root):
    configure()
    base.seed_everything(seed)
    out = out_root / f"seed_{seed}"
    out.mkdir(parents=True, exist_ok=True)
    train, val, test, dims, names, test_ids = prepare(seed)
    base.MODEL_SPECS["morphoprot"] = dict(kind="dual_gat", gat_block="gatv2", graph_input_mode="fuse")
    result = base.run_one_model("morphoprot", out, train, val, test, dims, names, SCALES)
    np.save(out / "test_pred.npy", result["test_pred"])
    np.save(out / "test_true.npy", result["test_true"])
    summary, per = metrics(result["test_true"], result["test_pred"], names)
    per.to_csv(out / "test_per_protein.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(dict(seed=seed, best_epoch=result["best_epoch"],
                                                      val_scc=float(result["val"]["mean_scc"]), test=summary), indent=2))
    print("TEST", seed, json.dumps(summary), flush=True)
    return result["test_pred"], result["test_true"], names, test_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="*", type=int, default=[42, 99, 7, 1024])
    ap.add_argument("--out-dir", type=str, default=str(RESULTS_ROOT / "liver"))
    args = ap.parse_args()
    out_root = Path(args.out_dir)
    preds = []
    for seed in args.seeds:
        p, y, names, test_ids = fit(seed, out_root)
        preds.append(p)
    ens = np.mean(np.stack(preds, 0), 0)
    np.save(out_root / "ensemble_test_pred.npy", ens)
    np.save(out_root / "ensemble_test_true.npy", y)
    pooled, per = metrics(y, ens, names)
    per.to_csv(out_root / "ensemble_test_per_protein.csv", index=False)
    sections = {}
    for s in sorted({b.split(":")[0] for b in test_ids}):
        ix = np.array([b.split(":")[0] == s for b in test_ids])
        sections[s], _ = metrics(y[ix], ens[ix], names)
    (out_root / "ensemble_metrics.json").write_text(json.dumps(dict(seeds=args.seeds, pooled=pooled, sections=sections), indent=2))
    print("ENSEMBLE", json.dumps(pooled), flush=True)


if __name__ == "__main__":
    main()
