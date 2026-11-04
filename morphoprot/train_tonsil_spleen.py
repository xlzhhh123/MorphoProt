"""Train MorphoProt on tonsil and spleen, with the modality ablations.

Ablations (--ablations): full (MorphoProt), rna (RNA only), rna_image (RNA + histology,
no graphs), rna_graph (RNA + graphs, no histology). Each is trained with every seed in
--seeds; per-seed and four-seed-ensemble predictions and per-protein metrics are written
to results/tonsil_spleen/<dataset>/<ablation>/.

    python morphoprot/train_tonsil_spleen.py --datasets tonsil spleen \
        --ablations full rna rna_image rna_graph --seeds 42 99 7 1024
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


MODEL_DIR = Path(__file__).resolve().parent
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

import core as base
import data_spleen as spleen
import data_liver as liver
from paths import DATA_ROOT, RESULTS_ROOT


OUT_ROOT = RESULTS_ROOT / "tonsil_spleen"

ABLATIONS = {
    "rna": {
        "kind": "no_graph",
        "ablation_kind": "rna_only",
        "description": "RNA only; image and graph branches disabled.",
    },
    "rna_image": {
        "kind": "no_graph",
        "description": "RNA + image cross-attention; graph branch disabled.",
    },
    "rna_graph": {
        "kind": "dual_gat",
        "graph_input_mode": "rna",
        "gat_block": "gatv2",
        "ablation_kind": "rna_graph_only",
        "description": "RNA + spatial/feature GATv2 graph; image branch disabled.",
    },
    "full": {
        "kind": "dual_gat",
        "graph_input_mode": "fuse",
        "gat_block": "gatv2",
        "description": "Full MorphoProt: RNA + image cross-attention + spatial/feature GATv2 graph.",
    },
}


class RNAOnlyModel(nn.Module):
    def __init__(self, rna_dim, he_dims, out_dim, A_init):
        super().__init__()
        self.rna_tower = base.Tower(rna_dim, base.RNA_HIDDEN_DIM, base.EMB_DIM, base.DROPOUT, n_blocks=3)
        self.head = base.ProteinPredictorHead(base.EMB_DIM, out_dim, base.DROPOUT, A_init)
        self.uncertainty = base.UncertaintyLoss(out_dim)

    def forward(self, x_rna, x_he_list):
        return self.head(self.rna_tower(x_rna))

    def compute_loss(self, pred, target):
        return base.compute_multitask_loss(pred, target, self.uncertainty)


class RNAGraphOnlyDualGATv2Model(base.DualGATv2GraphModel):
    def forward(self, x_rna, x_he_list, spatial_neighbors_raw, spatial_edge_feat, feature_neighbors_raw, feature_edge_feat):
        z_rna = self.backbone.encode_rna_only(x_rna)
        graph_center = z_rna

        b, k1, g = spatial_neighbors_raw.shape
        z_spatial_neighbors = self.backbone.encode_rna_only(spatial_neighbors_raw.reshape(b * k1, g)).reshape(b, k1, -1)

        b, k2, g = feature_neighbors_raw.shape
        z_feature_neighbors = self.backbone.encode_rna_only(feature_neighbors_raw.reshape(b * k2, g)).reshape(b, k2, -1)

        z_spatial = self.spatial_gat(graph_center, z_spatial_neighbors, spatial_edge_feat)
        z_feature = self.feature_gat(graph_center, z_feature_neighbors, feature_edge_feat)

        beta_s = torch.sigmoid(self.beta_spatial_logit)
        beta_f = torch.sigmoid(self.beta_feature_logit)
        z_spatial = beta_s * z_spatial
        z_feature = beta_f * z_feature

        g_s = self.spatial_gate(torch.cat([z_rna, z_spatial], dim=1))
        g_f = self.feature_gate(torch.cat([z_rna, z_feature], dim=1))
        z = z_rna + g_s * z_spatial + g_f * z_feature
        return self.head(z)


_ORIG_BUILD_MODEL = base.build_model
_ORIG_EXPORT_ATTENTION = base.export_cross_attention_summary


def patched_build_model(model_name, rna_dim, he_dims, out_dim, A_init):
    spec = base.MODEL_SPECS[model_name]
    if spec.get("ablation_kind") == "rna_only":
        return RNAOnlyModel(rna_dim, he_dims, out_dim, A_init).to(base.DEVICE)
    if spec.get("ablation_kind") == "rna_graph_only":
        return RNAGraphOnlyDualGATv2Model(
            rna_dim, he_dims, out_dim, A_init, graph_input_mode=spec.get("graph_input_mode", "rna")
        ).to(base.DEVICE)
    return _ORIG_BUILD_MODEL(model_name, rna_dim, he_dims, out_dim, A_init)


base.build_model = patched_build_model


def patched_export_cross_attention_summary(model, loader, model_name, patch_sizes, out_dir):
    if base.MODEL_SPECS.get(model_name, {}).get("ablation_kind") == "rna_graph_only":
        return None
    return _ORIG_EXPORT_ATTENTION(model, loader, model_name, patch_sizes, out_dir)


base.export_cross_attention_summary = patched_export_cross_attention_summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", default=["tonsil", "spleen"], choices=["tonsil", "spleen"])
    parser.add_argument("--ablations", nargs="*", default=["full", "rna", "rna_image", "rna_graph"], choices=list(ABLATIONS))
    parser.add_argument("--seeds", nargs="*", type=int, default=[42, 99, 7, 1024])
    parser.add_argument("--out-dir", type=str, default=str(OUT_ROOT))
    parser.add_argument("--liver-processed-dir", type=str, default=str(liver.ROOT))
    parser.add_argument("--liver-train-label", type=str, default="N123,N126")
    parser.add_argument("--liver-test-label", type=str, default="Sah73,Sah80")
    return parser.parse_args()


def zero_he_like(he_list: list[np.ndarray]) -> list[np.ndarray]:
    return [np.zeros_like(x, dtype=np.float32) for x in he_list]


def split_pack_tonsil(seed: int):
    base.configure_tonsil_direction("forward")
    x_all, y_all, x_test_all, y_test_all, proteins = base.load_rna_protein()
    coords_all = base.load_coords(base.TRAIN_COORD_PATH, base.BARCODES["train"])
    coords_test = base.load_coords(base.TEST_COORD_PATH, base.BARCODES["test"])
    he_all, he_test = base.load_multiscale_he(base.PATCH_SIZES_MULTI)

    idx = np.arange(x_all.shape[0])
    train_idx, val_idx = train_test_split(idx, test_size=base.VAL_RATIO, random_state=seed, shuffle=True)
    x_train, x_val, x_test = x_all[train_idx], x_all[val_idx], x_test_all
    y_train, y_val, y_test = y_all[train_idx], y_all[val_idx], y_test_all
    coords_train, coords_val = coords_all[train_idx], coords_all[val_idx]
    he_train = [x[train_idx] for x in he_all]
    he_val = [x[val_idx] for x in he_all]

    def make_pack(x_rna, y, he_list, coords, pca_model=None):
        spatial_knn, spatial_dist = base.build_knn_graph(coords.astype(np.float32), k=base.SPATIAL_K)
        spatial_neighbors = x_rna[spatial_knn]
        spatial_edge = base.build_edge_features_from_base(coords.astype(np.float32), spatial_knn, spatial_dist)
        spatial_mean = base.build_neighbor_mean_feature(x_rna, spatial_knn)
        if pca_model is None:
            pca_dim = max(2, min(base.PCA_DIM, x_rna.shape[0] - 1, x_rna.shape[1]))
            pca_model = PCA(n_components=pca_dim, random_state=seed)
            feat = pca_model.fit_transform(x_rna).astype(np.float32)
        else:
            feat = pca_model.transform(x_rna).astype(np.float32)
        feature_knn, feature_dist = base.build_knn_graph(feat, k=base.FEATURE_K)
        feature_neighbors = x_rna[feature_knn]
        feature_edge = base.build_edge_features_from_base(feat, feature_knn, feature_dist)
        feature_mean = base.build_neighbor_mean_feature(x_rna, feature_knn)
        return base.build_split_pack(
            x_rna.astype(np.float32), [h.astype(np.float32) for h in he_list], y.astype(np.float32),
            spatial_neighbors, spatial_edge, spatial_mean,
            feature_neighbors, feature_edge, feature_mean,
        ), pca_model

    train_pack, pca = make_pack(x_train, y_train, he_train, coords_train)
    val_pack, _ = make_pack(x_val, y_val, he_val, coords_val, pca)
    test_pack, _ = make_pack(x_test, y_test, he_test, coords_test, pca)
    return train_pack, val_pack, test_pack, [h.shape[1] for h in he_train], proteins, base.PATCH_SIZES_MULTI


def split_pack_spleen(seed: int):
    x_all, y_all, x_test_all, y_test_all, he_all, he_test, coords_all, coords_test, proteins = spleen.load_all()
    idx = np.arange(x_all.shape[0])
    train_idx, val_idx = train_test_split(idx, test_size=spleen.VAL_RATIO, random_state=seed, shuffle=True)
    train_pack, pca = spleen.make_split_pack(
        x_all[train_idx], y_all[train_idx], [x[train_idx] for x in he_all], coords_all[train_idx], seed=seed
    )
    val_pack, _ = spleen.make_split_pack(
        x_all[val_idx], y_all[val_idx], [x[val_idx] for x in he_all], coords_all[val_idx], pca_model=pca, seed=seed
    )
    test_pack, _ = spleen.make_split_pack(x_test_all, y_test_all, he_test, coords_test, pca_model=pca, seed=seed)
    return train_pack, val_pack, test_pack, [h.shape[1] for h in he_all], proteins, spleen.HE_SCALES


def split_pack_liver(seed: int, processed_dir: str):
    liver.ROOT = Path(processed_dir)
    liver.GENE_DIR = liver.ROOT / "_scheme2_log1p_zscore_hvg3500_force_marker_in_tissue1"
    liver.PRO_DIR = liver.ROOT / "_protein_scheme_log1p_zscore_in_tissue1"
    liver.HE_DIR = liver.ROOT / "_hires_resnet18_multiscale512"
    x_all, y_all, x_test_all, y_test_all, he_all, he_test, coords_all, coords_test, proteins = liver.load_all()
    idx = np.arange(x_all.shape[0])
    train_idx, val_idx = train_test_split(idx, test_size=liver.VAL_RATIO, random_state=seed, shuffle=True)
    train_pack, pca = liver.make_split_pack(
        x_all[train_idx], y_all[train_idx], [x[train_idx] for x in he_all], coords_all[train_idx], seed=seed
    )
    val_pack, _ = liver.make_split_pack(
        x_all[val_idx], y_all[val_idx], [x[val_idx] for x in he_all], coords_all[val_idx], pca_model=pca, seed=seed
    )
    test_pack, _ = liver.make_split_pack(x_test_all, y_test_all, he_test, coords_test, pca_model=pca, seed=seed)
    return train_pack, val_pack, test_pack, [h.shape[1] for h in he_all], proteins, liver.HE_SCALES


def apply_ablation_to_packs(ablation: str, train_pack: dict, val_pack: dict, test_pack: dict):
    if ablation in {"rna", "rna_graph"}:
        for pack in [train_pack, val_pack, test_pack]:
            pack["x_he_list"] = zero_he_like(pack["x_he_list"])


def run_dataset(dataset: str, ablations: list[str], seeds: list[int], out_root: Path, args) -> list[dict]:
    dataset_out = out_root / dataset
    dataset_out.mkdir(parents=True, exist_ok=True)
    rows = []
    combined_per = []

    for ablation in ablations:
        model_name = f"morphoprot_{dataset}_{ablation}"
        pred_list = []
        true_ref = None
        proteins_ref = None
        patch_sizes_ref = None

        for seed in seeds:
            print("=" * 100)
            print(f"DATASET={dataset} | ABLATION={ablation} | SEED={seed}")
            base.seed_everything(seed)
            if dataset == "tonsil":
                train_pack, val_pack, test_pack, he_dims, proteins, patch_sizes = split_pack_tonsil(seed)
            elif dataset == "spleen":
                train_pack, val_pack, test_pack, he_dims, proteins, patch_sizes = split_pack_spleen(seed)
            elif dataset == "liver":
                train_pack, val_pack, test_pack, he_dims, proteins, patch_sizes = split_pack_liver(seed, args.liver_processed_dir)
            else:
                raise ValueError(dataset)

            base.MODEL_SPECS[model_name] = dict(ABLATIONS[ablation])
            base.MODEL_LIST = list(base.MODEL_SPECS.keys())
            apply_ablation_to_packs(ablation, train_pack, val_pack, test_pack)
            seed_out = dataset_out / ablation / f"seed_{seed}" / model_name
            seed_out.mkdir(parents=True, exist_ok=True)
            pack = base.run_one_model(
                model_name=model_name,
                out_dir=seed_out,
                train_pack=train_pack,
                val_pack=val_pack,
                test_pack=test_pack,
                he_dims=he_dims,
                prot_names=proteins,
                patch_sizes=patch_sizes,
            )
            rows.append({
                "dataset": dataset,
                "ablation": ablation,
                "seed": seed,
                "model": model_name,
                "best_epoch": pack["best_epoch"],
                "val_mean_scc": pack["val"]["mean_scc"],
                "val_mean_pcc": pack["val"]["mean_pcc"],
                "val_top10_scc_mean": pack["val"]["top10_scc_mean"],
                "val_mse": pack["val"]["mse_raw"],
                "val_rmse": pack["val"]["rmse_raw"],
                "test_mean_scc": pack["test"]["mean_scc"],
                "test_mean_pcc": pack["test"]["mean_pcc"],
                "test_top10_scc_mean": pack["test"]["top10_scc_mean"],
                "test_mse": pack["test"]["mse_raw"],
                "test_rmse": pack["test"]["rmse_raw"],
            })
            per = pack["test"]["per_protein_df"].copy()
            per.insert(0, "seed", seed)
            per.insert(0, "ablation", ablation)
            per.insert(0, "dataset", dataset)
            per.to_csv(dataset_out / ablation / f"{model_name}_seed{seed}_test_per_protein.csv", index=False, encoding="utf-8-sig")
            combined_per.append(per)
            pred_list.append(pack["test_pred"].astype(np.float32))
            true_ref = pack["test_true"].astype(np.float32)
            proteins_ref = proteins
            patch_sizes_ref = patch_sizes

            del train_pack, val_pack, test_pack, pack
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if len(pred_list) > 1:
            pred = np.stack(pred_list, axis=0).mean(axis=0).astype(np.float32)
            ens = base.summarize_metrics(true_ref, pred, proteins_ref, {
                "loss_total_scaled": np.nan,
                "loss_mse_scaled": np.nan,
                "loss_ccc_scaled": np.nan,
                "loss_rank_scaled": np.nan,
            })
            per = ens["per_protein_df"].copy()
            per.insert(0, "seed", "ensemble")
            per.insert(0, "ablation", ablation)
            per.insert(0, "dataset", dataset)
            per.to_csv(dataset_out / ablation / f"{model_name}_ensemble_test_per_protein.csv", index=False, encoding="utf-8-sig")
            rows.append({
                "dataset": dataset,
                "ablation": ablation,
                "seed": "ensemble",
                "model": model_name,
                "best_epoch": np.nan,
                "val_mean_scc": np.nan,
                "val_mean_pcc": np.nan,
                "val_top10_scc_mean": np.nan,
                "val_mse": np.nan,
                "val_rmse": np.nan,
                "test_mean_scc": ens["mean_scc"],
                "test_mean_pcc": ens["mean_pcc"],
                "test_top10_scc_mean": ens["top10_scc_mean"],
                "test_mse": ens["mse_raw"],
                "test_rmse": ens["rmse_raw"],
            })

    pd.DataFrame(rows).to_csv(dataset_out / "ablation_summary.csv", index=False, encoding="utf-8-sig")
    if combined_per:
        pd.concat(combined_per, axis=0).to_csv(dataset_out / "ablation_per_protein_all.csv", index=False, encoding="utf-8-sig")
    return rows


def main():
    args = parse_args()
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    start = time.time()
    all_rows = []
    for dataset in args.datasets:
        all_rows.extend(run_dataset(dataset, args.ablations, args.seeds, out_root, args))
    summary = pd.DataFrame(all_rows)
    summary.to_csv(out_root / "all_datasets_ablation_summary.csv", index=False, encoding="utf-8-sig")
    if not summary.empty:
        print(summary[["dataset", "ablation", "seed", "test_mean_scc", "test_mean_pcc", "test_rmse"]].to_string(index=False))
    print(f"DONE -> {out_root} in {(time.time() - start) / 60:.2f} min")


if __name__ == "__main__":
    main()
