"""CPU smoke test on random data: builds the paper model, trains two epochs through
the same training loop and checks prediction shapes. No real data needed.

    python tests/test_smoke.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "morphoprot"))
import core as base  # noqa: E402
import data_liver as liver  # noqa: E402

base.DEVICE = "cpu"
base.EPOCHS, base.PATIENCE = 2, 2
base.export_cross_attention_summary = lambda *a, **kw: None

rng = np.random.default_rng(0)
G, P, dims = 200, 6, [32] * 5


def fake(n):
    x = rng.normal(size=(n, G)).astype(np.float32)
    y = (x[:, :P] + 0.1 * rng.normal(size=(n, P))).astype(np.float32)
    he = [rng.normal(size=(n, d)).astype(np.float32) for d in dims]
    xy = rng.uniform(0, 100, size=(n, 2)).astype(np.float32)
    return x, y, he, xy


x, y, he, xy = fake(300)
xt, yt, het, xyt = fake(120)
train, pca = liver.make_split_pack(x[:250], y[:250], [h[:250] for h in he], xy[:250], seed=0)
val, _ = liver.make_split_pack(x[250:], y[250:], [h[250:] for h in he], xy[250:], pca_model=pca, seed=0)
test, _ = liver.make_split_pack(xt, yt, het, xyt, pca_model=pca, seed=0)

model = base.DualGATv2GraphModel(G, dims, P, base.build_corr_init(y[:250]), graph_input_mode="fuse")
n_param = sum(p.numel() for p in model.parameters())
print("paper model parameters:", n_param)

base.MODEL_SPECS["smoke"] = dict(kind="dual_gat", gat_block="gatv2", graph_input_mode="fuse")
out = ROOT / "results" / "smoke_test"
out.mkdir(parents=True, exist_ok=True)
res = base.run_one_model("smoke", out, train, val, test, dims, [f"p{i}" for i in range(P)], [16, 32, 64, 96, 128])
assert res["test_pred"].shape == (120, P) and np.isfinite(res["test_pred"]).all()
print("OK: test predictions", res["test_pred"].shape, "val SCC", round(float(res["val"]["mean_scc"]), 3))
