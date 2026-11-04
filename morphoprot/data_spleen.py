"""Mouse spleen data loading (train Dataset1_Mouse_Spleen1, test Dataset2_Mouse_Spleen2).

Inputs are read from $MORPHOPROT_DATA (see README.md, "Data").
"""
from __future__ import annotations

import gc
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

MODEL_DIR = Path(__file__).resolve().parent
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

import core as base
from paths import DATA_ROOT, RESULTS_ROOT


ROOT = DATA_ROOT / "spleen"
GENE_DIR = ROOT / "_scheme2_log1p_zscore_hvg3500_force_marker_in_tissue1"
PRO_DIR = ROOT / "_protein_scheme_log1p_zscore_in_tissue1"
HE_DIR = ROOT / "_hires_resnet18_multiscale512"
TRAIN_COORD = ROOT / "Dataset1_Mouse_Spleen1" / "tissue_positions_list_with_hires_coords.csv"
TEST_COORD = ROOT / "Dataset2_Mouse_Spleen2" / "tissue_positions_list_with_hires_coords.csv"

HE_SCALES = [16, 32, 64, 96, 128]
VAL_RATIO = 0.15


def read_lines(path: Path) -> list[str]:
    return [x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def load_coords(csv: Path, barcodes: list[str]) -> np.ndarray:
    df = pd.read_csv(csv)
    coord_map = {
        str(b): (float(x), float(y))
        for b, x, y in zip(df["barcode"], df["pxl_col_in_hires"], df["pxl_row_in_hires"])
    }
    return np.asarray([coord_map[b] for b in barcodes], dtype=np.float32)


def load_all():
    x_train = np.load(GENE_DIR / "X_train_log1p_zscore.npy").astype(np.float32)
    x_test = np.load(GENE_DIR / "X_test_log1p_zscore.npy").astype(np.float32)
    y_train = np.load(PRO_DIR / "Y_train_log1p_zscore.npy").astype(np.float32)
    y_test = np.load(PRO_DIR / "Y_test_log1p_zscore.npy").astype(np.float32)

    train_barcodes = read_lines(GENE_DIR / "train_barcodes.txt")
    test_barcodes = read_lines(GENE_DIR / "test_barcodes.txt")
    if train_barcodes != read_lines(PRO_DIR / "train_barcodes.txt"):
        raise ValueError("Train gene/protein barcode order mismatch")
    if test_barcodes != read_lines(PRO_DIR / "test_barcodes.txt"):
        raise ValueError("Test gene/protein barcode order mismatch")

    he_train = [
        np.load(HE_DIR / f"he_resnet18_ps{s}_train_512.npy").astype(np.float32)
        for s in HE_SCALES
    ]
    he_test = [
        np.load(HE_DIR / f"he_resnet18_ps{s}_test_512.npy").astype(np.float32)
        for s in HE_SCALES
    ]
    he_train = [base.align_to_barcodes(h, read_lines(HE_DIR / "train_barcodes.txt"), train_barcodes, "train image features")
                for h in he_train]
    he_test = [base.align_to_barcodes(h, read_lines(HE_DIR / "test_barcodes.txt"), test_barcodes, "test image features")
               for h in he_test]
    coords_train = load_coords(TRAIN_COORD, train_barcodes)
    coords_test = load_coords(TEST_COORD, test_barcodes)
    proteins = read_lines(PRO_DIR / "protein_names.txt")
    return x_train, y_train, x_test, y_test, he_train, he_test, coords_train, coords_test, proteins


def make_split_pack(x_rna, y, he_list, coords, pca_source=None, pca_model=None, seed=0):
    spatial_knn, spatial_dist = base.build_knn_graph(coords.astype(np.float32), k=base.SPATIAL_K)
    spatial_neighbors_raw = x_rna[spatial_knn]
    spatial_edge = base.build_edge_features_from_base(coords.astype(np.float32), spatial_knn, spatial_dist)
    spatial_mean = base.build_neighbor_mean_feature(x_rna, spatial_knn)

    if pca_model is None:
        pca_dim = max(2, min(base.PCA_DIM, x_rna.shape[0] - 1, x_rna.shape[1]))
        pca_model = PCA(n_components=pca_dim, random_state=seed)
        feat = pca_model.fit_transform(pca_source if pca_source is not None else x_rna).astype(np.float32)
        if pca_source is not None and pca_source is not x_rna:
            feat = pca_model.transform(x_rna).astype(np.float32)
    else:
        feat = pca_model.transform(x_rna).astype(np.float32)

    feature_knn, feature_dist = base.build_knn_graph(feat, k=base.FEATURE_K)
    feature_neighbors_raw = x_rna[feature_knn]
    feature_edge = base.build_edge_features_from_base(feat, feature_knn, feature_dist)
    feature_mean = base.build_neighbor_mean_feature(x_rna, feature_knn)

    pack = base.build_split_pack(
        x_rna.astype(np.float32),
        [x.astype(np.float32) for x in he_list],
        y.astype(np.float32),
        spatial_neighbors_raw,
        spatial_edge,
        spatial_mean,
        feature_neighbors_raw,
        feature_edge,
        feature_mean,
    )
    return pack, pca_model


