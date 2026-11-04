# -*- coding: utf-8 -*-
"""MorphoProt core: model definitions, data loading for the tonsil sections and the
shared training / evaluation loop.

The paper model is ``DualGATv2GraphModel`` with ``graph_input_mode="fuse"``:
an RNA encoder, RNA-queried cross-attention over five H&E field-of-view tokens,
two GATv2 neighborhood blocks (physical kNN, k=8; RNA-PCA kNN, k=4) whose
center uses the fused representation and whose neighbors use RNA-only
representations, gated residual aggregation, and a decoder with a correlation
head. The module also provides reusable components for different neighborhood
aggregation configurations.

Entry points: ``train_tonsil_spleen.py`` (tonsil, spleen) and ``train_liver.py`` (liver).
"""

import gc
import copy
import random
import warnings
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp

from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from scipy.stats import spearmanr, pearsonr

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")


# =========================================================
# 0. Paths and hyperparameters
# =========================================================
MODEL_DIR = Path(__file__).resolve().parent
import sys as _sys
_sys.path.insert(0, str(MODEL_DIR))
from paths import DATA_ROOT, RESULTS_ROOT
ROOT = DATA_ROOT / "tonsil"
SAVE_ROOT = RESULTS_ROOT / "tonsil"

SAMPLE_INFO = {
    "female": {
        "gene": ROOT / "female" / "gene" / "tonsilwoman_train_hvg3500_seurat_gene_zscore_log1p_raw.h5ad",
        "pro": ROOT / "female" / "pro" / "tonsilwoman_pro_filter_log1p_zscore.h5ad",
        "coord": ROOT / "female" / "coord" / "tissue_positions_intersect_ordered.csv",
        "he_prefix": "tonsilwoman_uni2h",
        "he_dir_1": ROOT / "female" / "image_uni2h_patch96_to_1024_step32",
        "he_dir_2": ROOT / "female" / "image_uni2h_patch1024_to_2000_step32",
        "base_dir": ROOT / "female",
    },
    "male": {
        "gene": ROOT / "male" / "gene" / "tonsilman_test_from_tonsilwoman_hvg3500_seurat_gene_zscore_log1p_raw.h5ad",
        "pro": ROOT / "male" / "pro" / "tonsilman_pro_filter_log1p_zscore.h5ad",
        "coord": ROOT / "male" / "coord" / "tissue_positions_intersect_ordered.csv",
        "he_prefix": "tonsilman_uni2h",
        "he_dir_1": ROOT / "male" / "image_uni2h_patch96_to_1024_step32",
        "he_dir_2": ROOT / "male" / "image_uni2h_patch1024_to_2000_step32",
        "base_dir": ROOT / "male",
    },
}

PATCH_SIZES_MULTI = [256, 512, 768, 1024, 1280]

SEEDS = [42, 52, 62, 72, 82]
VAL_RATIO = 0.15

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 0

BATCH_SIZE_TRAIN = 128
BATCH_SIZE_EVAL = 256

EPOCHS = 40
PATIENCE = 8

LR = 4e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 5.0

RNA_HIDDEN_DIM = 768
EMB_DIM = 256
DROPOUT = 0.15

LAMBDA_CCC = 0.15
LAMBDA_RANK = 0.05

ATT_NUM_HEADS = 4
GRAPH_NUM_HEADS = 4
ATT_ALPHA_INIT = 0.10
GRAPH_BETA_INIT = 0.08

SPATIAL_K = 8
FEATURE_K = 4
PCA_DIM = 64
EXCLUDE_PROTEINS = {"mouse_IgG1k", "mouse_IgG2a", "mouse_IgG2bk", "rat_IgG2a"}

# Registry of model configurations by name; entries are added by the training entry points.
MODEL_SPECS = {}
MODEL_LIST = []


def configure_tonsil_direction(direction: str = "forward"):
    """Set the tonsil train/test sections. The paper uses "forward": train on the
    female section, test on the male section."""
    global TRAIN_GENE_PATH, TEST_GENE_PATH, TRAIN_PRO_PATH, TEST_PRO_PATH
    global TRAIN_COORD_PATH, TEST_COORD_PATH, TRAIN_HE_DIR_1, TEST_HE_DIR_1
    global TRAIN_HE_DIR_2, TEST_HE_DIR_2, TRAIN_HE_PREFIX, TEST_HE_PREFIX
    global TRAIN_HE_BASE_DIR, TEST_HE_BASE_DIR

    if direction == "forward":
        train_name, test_name = "female", "male"
    elif direction == "reverse":
        train_name, test_name = "male", "female"
    else:
        raise ValueError(f"Unknown direction: {direction}")

    train = SAMPLE_INFO[train_name]
    test = SAMPLE_INFO[test_name]
    TRAIN_GENE_PATH, TEST_GENE_PATH = train["gene"], test["gene"]
    TRAIN_PRO_PATH, TEST_PRO_PATH = train["pro"], test["pro"]
    TRAIN_COORD_PATH, TEST_COORD_PATH = train["coord"], test["coord"]
    TRAIN_HE_DIR_1, TEST_HE_DIR_1 = train["he_dir_1"], test["he_dir_1"]
    TRAIN_HE_DIR_2, TEST_HE_DIR_2 = train["he_dir_2"], test["he_dir_2"]
    TRAIN_HE_PREFIX, TEST_HE_PREFIX = train["he_prefix"], test["he_prefix"]
    TRAIN_HE_BASE_DIR, TEST_HE_BASE_DIR = train["base_dir"], test["base_dir"]


configure_tonsil_direction("forward")

# =========================================================
# 1. Utilities: metrics, coordinates, kNN graphs
# =========================================================
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def to_dense(x):
    if sp.issparse(x):
        return x.toarray()
    return np.asarray(x)

def safe_spearman(y_true, y_pred):
    vals = []
    for j in range(y_true.shape[1]):
        a = y_true[:, j]
        b = y_pred[:, j]
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            continue
        r, _ = spearmanr(a, b)
        if np.isfinite(r):
            vals.append(r)
    return float(np.mean(vals)) if len(vals) > 0 else np.nan

def safe_pearson(y_true, y_pred):
    vals = []
    for j in range(y_true.shape[1]):
        a = y_true[:, j]
        b = y_pred[:, j]
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            continue
        r, _ = pearsonr(a, b)
        if np.isfinite(r):
            vals.append(r)
    return float(np.mean(vals)) if len(vals) > 0 else np.nan

def per_protein_metrics(y_true, y_pred, prot_names):
    rows = []
    for j in range(y_true.shape[1]):
        yt = y_true[:, j]
        yp = y_pred[:, j]

        scc = np.nan
        pcc = np.nan
        if np.std(yt) >= 1e-12 and np.std(yp) >= 1e-12:
            scc, _ = spearmanr(yt, yp)
            pcc, _ = pearsonr(yt, yp)

        mse = float(np.mean((yt - yp) ** 2))
        rmse = float(np.sqrt(mse))

        rows.append({
            "protein": prot_names[j],
            "scc": scc,
            "pcc": pcc,
            "mse": mse,
            "rmse": rmse,
        })
    return pd.DataFrame(rows)

def topk_scc_mean(per_protein_df, k=10):
    s = per_protein_df["scc"].dropna().sort_values(ascending=False)
    if len(s) == 0:
        return np.nan
    k = min(k, len(s))
    return float(s.iloc[:k].mean())

def summarize_metrics(y_true, y_pred, prot_names, loss_pack):
    mse_raw = float(np.mean((y_pred - y_true) ** 2))
    rmse_raw = float(np.sqrt(mse_raw))
    mean_scc = safe_spearman(y_true, y_pred)
    mean_pcc = safe_pearson(y_true, y_pred)
    per_df = per_protein_metrics(y_true, y_pred, prot_names)
    top10 = topk_scc_mean(per_df, k=10)
    return {
        "loss_total_scaled": loss_pack["loss_total_scaled"],
        "loss_mse_scaled": loss_pack["loss_mse_scaled"],
        "loss_ccc_scaled": loss_pack["loss_ccc_scaled"],
        "loss_rank_scaled": loss_pack["loss_rank_scaled"],
        "mse_raw": mse_raw,
        "rmse_raw": rmse_raw,
        "mean_scc": mean_scc,
        "mean_pcc": mean_pcc,
        "top10_scc_mean": top10,
        "per_protein_df": per_df,
    }

def check_file_exists(path_obj: Path):
    if not path_obj.exists():
        raise FileNotFoundError(f"File not found: {path_obj}")

def resolve_he_path(split: str, patch_size: int) -> Path:
    if split == "train":
        dir1, dir2 = TRAIN_HE_DIR_1, TRAIN_HE_DIR_2
        prefix = TRAIN_HE_PREFIX
        base_dir = TRAIN_HE_BASE_DIR
    else:
        dir1, dir2 = TEST_HE_DIR_1, TEST_HE_DIR_2
        prefix = TEST_HE_PREFIX
        base_dir = TEST_HE_BASE_DIR

    fname = f"{prefix}_patch{patch_size}.npy"
    candidate_dirs = [
        dir1,
        dir2,
        base_dir / "image_uni2h_patch96_to_2016_step32",
        base_dir / "image_uni2h_multiscale",
        base_dir / "image_uni2h_patch256_384",
        base_dir / "image_uni2h_patch736_to_1024_step32",
    ]
    for d in candidate_dirs:
        p = d / fname
        if p.exists():
            return p

    glob_hits = sorted(base_dir.glob(f"image_uni2h*/{fname}"))
    if glob_hits:
        return glob_hits[0]

    p1 = candidate_dirs[0] / fname
    p2 = candidate_dirs[1] / fname

    raise FileNotFoundError(f"Image features for patch{patch_size} not found:\n{p1}\n{p2}")

def find_xy_cols(df: pd.DataFrame):
    cols = {c.lower(): c for c in df.columns}
    candidates_x = ["pxl_col_in_fullres", "pixel_x", "x", "pxl_col", "imagecol"]
    candidates_y = ["pxl_row_in_fullres", "pixel_y", "y", "pxl_row", "imagerow"]

    xcol, ycol = None, None
    for c in candidates_x:
        if c in cols:
            xcol = cols[c]
            break
    for c in candidates_y:
        if c in cols:
            ycol = cols[c]
            break

    if xcol is None or ycol is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) >= 2:
            xcol, ycol = numeric_cols[-2], numeric_cols[-1]
        else:
            raise ValueError(f"No usable x/y columns in coordinate file, columns={df.columns.tolist()}")

    return xcol, ycol

# Spot barcodes of the RNA/protein matrices, filled by load_rna_protein(); every other
# per-spot input (coordinates, image features) is reordered to this order.
BARCODES = {}


def align_to_barcodes(arr, row_barcodes, target, what):
    """Return `arr` with rows reordered to `target`; fail if any target barcode is missing."""
    row_barcodes = [str(b) for b in row_barcodes]
    if len(row_barcodes) != len(arr):
        raise ValueError(f"{what}: {len(arr)} rows but {len(row_barcodes)} barcodes")
    if row_barcodes == list(target):
        return arr
    pos = {b: i for i, b in enumerate(row_barcodes)}
    missing = [b for b in target if b not in pos]
    if missing:
        raise ValueError(f"{what}: {len(missing)} RNA barcodes have no entry, first={missing[:3]}")
    print(f"[align] {what}: rows reordered to the RNA barcode order")
    return arr[[pos[b] for b in target]]


def load_coords(csv_path: Path, barcodes=None):
    """Spot coordinates; with `barcodes`, rows are matched to them by the barcode column."""
    check_file_exists(csv_path)
    df = pd.read_csv(csv_path)
    xcol, ycol = find_xy_cols(df)
    coords = df[[xcol, ycol]].to_numpy(dtype=np.float32)
    if barcodes is not None:
        bc_col = "barcode" if "barcode" in df.columns else df.columns[0]
        coords = align_to_barcodes(coords, df[bc_col].astype(str).tolist(), barcodes, f"coordinates {csv_path.name}")
    return coords

def build_knn_graph(feat: np.ndarray, k: int):
    n_neighbors = min(k + 1, feat.shape[0])
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    nbrs.fit(feat)
    distances, indices = nbrs.kneighbors(feat, return_distance=True)

    if indices.shape[1] > 1:
        neigh_idx = indices[:, 1:]
        neigh_dist = distances[:, 1:]
    else:
        neigh_idx = indices
        neigh_dist = distances

    return neigh_idx.astype(np.int64), neigh_dist.astype(np.float32)

def build_neighbor_mean_feature(x: np.ndarray, neigh_idx: np.ndarray):
    return x[neigh_idx].mean(axis=1).astype(np.float32)

def build_edge_features_from_base(base_feat: np.ndarray, neigh_idx: np.ndarray, neigh_dist: np.ndarray):
    d_use = min(2, base_feat.shape[1])
    src = base_feat[:, None, :d_use]
    nbr = base_feat[neigh_idx][:, :, :d_use]
    delta = nbr - src

    if d_use == 1:
        delta = np.concatenate([delta, np.zeros_like(delta)], axis=-1)
    elif d_use == 0:
        delta = np.zeros((base_feat.shape[0], neigh_idx.shape[1], 2), dtype=np.float32)

    dist = neigh_dist[..., None]
    edge_feat = np.concatenate([delta[:, :, :2], dist], axis=-1).astype(np.float32)
    return edge_feat


# =========================================================
# 2. Loss
# =========================================================
def ccc_loss(pred, target, eps=1e-6):
    ux = pred.mean(0)
    uy = target.mean(0)
    vx = pred.var(0, unbiased=False)
    vy = target.var(0, unbiased=False)
    cov = ((pred - ux) * (target - uy)).mean(0)
    ccc = 2 * cov / (vx + vy + (ux - uy) ** 2 + eps)
    return 1 - ccc.mean()

def rank_loss(pred, target, num_pairs=1024):
    B, P = pred.shape
    if B < 2:
        return pred.new_tensor(0.0)

    i = torch.randint(0, B, (num_pairs,), device=pred.device)
    j = torch.randint(0, B, (num_pairs,), device=pred.device)
    s = torch.sign(target[i] - target[j])
    mask = (s != 0)
    if mask.sum() == 0:
        return pred.new_tensor(0.0)

    diff = pred[i] - pred[j]
    x = -s[mask] * diff[mask]
    return F.softplus(x).mean()

class UncertaintyLoss(nn.Module):
    def __init__(self, n_outputs: int):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(n_outputs))

    def forward(self, mse_per_protein):
        log_vars = torch.clamp(self.log_vars, -5.0, 5.0)
        inv_vars = torch.exp(-log_vars)
        return (inv_vars * mse_per_protein + log_vars).mean()

def compute_multitask_loss(pred, target, uncertainty_module):
    mse_per_protein = ((pred - target) ** 2).mean(dim=0)
    loss_mse = uncertainty_module(mse_per_protein)
    loss_ccc = ccc_loss(pred, target)
    loss_rank = rank_loss(pred, target, num_pairs=1024)
    total = loss_mse + LAMBDA_CCC * loss_ccc + LAMBDA_RANK * loss_rank
    return total, {
        "loss_total": float(total.detach().item()),
        "loss_mse": float(loss_mse.detach().item()),
        "loss_ccc": float(loss_ccc.detach().item()),
        "loss_rank": float(loss_rank.detach().item()),
    }


# =========================================================
# 3. Dataset
# =========================================================
class NoGraphDataset(Dataset):
    def __init__(self, x_rna, x_he_list, y):
        self.x_rna = torch.from_numpy(x_rna.astype(np.float32))
        self.x_he_list = [torch.from_numpy(x.astype(np.float32)) for x in x_he_list]
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self):
        return self.x_rna.shape[0]

    def __getitem__(self, idx):
        he_items = [x[idx] for x in self.x_he_list]
        return self.x_rna[idx], he_items, self.y[idx]


class SingleMeanDataset(Dataset):
    def __init__(self, x_rna, x_he_list, neigh_mean, y):
        self.x_rna = torch.from_numpy(x_rna.astype(np.float32))
        self.x_he_list = [torch.from_numpy(x.astype(np.float32)) for x in x_he_list]
        self.neigh_mean = torch.from_numpy(neigh_mean.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self):
        return self.x_rna.shape[0]

    def __getitem__(self, idx):
        he_items = [x[idx] for x in self.x_he_list]
        return self.x_rna[idx], he_items, self.neigh_mean[idx], self.y[idx]


class DualMeanDataset(Dataset):
    def __init__(self, x_rna, x_he_list, spatial_mean, feature_mean, y):
        self.x_rna = torch.from_numpy(x_rna.astype(np.float32))
        self.x_he_list = [torch.from_numpy(x.astype(np.float32)) for x in x_he_list]
        self.spatial_mean = torch.from_numpy(spatial_mean.astype(np.float32))
        self.feature_mean = torch.from_numpy(feature_mean.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self):
        return self.x_rna.shape[0]

    def __getitem__(self, idx):
        he_items = [x[idx] for x in self.x_he_list]
        return self.x_rna[idx], he_items, self.spatial_mean[idx], self.feature_mean[idx], self.y[idx]


class SingleGATDataset(Dataset):
    def __init__(self, x_rna, x_he_list, neighbors_raw, edge_feat, y):
        self.x_rna = torch.from_numpy(x_rna.astype(np.float32))
        self.x_he_list = [torch.from_numpy(x.astype(np.float32)) for x in x_he_list]
        self.neighbors_raw = torch.from_numpy(neighbors_raw.astype(np.float32))
        self.edge_feat = torch.from_numpy(edge_feat.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self):
        return self.x_rna.shape[0]

    def __getitem__(self, idx):
        he_items = [x[idx] for x in self.x_he_list]
        return self.x_rna[idx], he_items, self.neighbors_raw[idx], self.edge_feat[idx], self.y[idx]


class DualGATDataset(Dataset):
    def __init__(self, x_rna, x_he_list, spatial_neighbors_raw, spatial_edge_feat, feature_neighbors_raw, feature_edge_feat, y):
        self.x_rna = torch.from_numpy(x_rna.astype(np.float32))
        self.x_he_list = [torch.from_numpy(x.astype(np.float32)) for x in x_he_list]
        self.spatial_neighbors_raw = torch.from_numpy(spatial_neighbors_raw.astype(np.float32))
        self.spatial_edge_feat = torch.from_numpy(spatial_edge_feat.astype(np.float32))
        self.feature_neighbors_raw = torch.from_numpy(feature_neighbors_raw.astype(np.float32))
        self.feature_edge_feat = torch.from_numpy(feature_edge_feat.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self):
        return self.x_rna.shape[0]

    def __getitem__(self, idx):
        he_items = [x[idx] for x in self.x_he_list]
        return (
            self.x_rna[idx],
            he_items,
            self.spatial_neighbors_raw[idx],
            self.spatial_edge_feat[idx],
            self.feature_neighbors_raw[idx],
            self.feature_edge_feat[idx],
            self.y[idx],
        )


def collate_nograph(batch):
    rna_list, he_list_all, y_list = zip(*batch)
    x_rna = torch.stack(rna_list, dim=0)
    y = torch.stack(y_list, dim=0)

    num_scales = len(he_list_all[0])
    he_scales = []
    for s in range(num_scales):
        he_scales.append(torch.stack([items[s] for items in he_list_all], dim=0))
    return x_rna, he_scales, y


def collate_single_mean(batch):
    rna_list, he_list_all, neigh_mean_list, y_list = zip(*batch)
    x_rna = torch.stack(rna_list, dim=0)
    neigh_mean = torch.stack(neigh_mean_list, dim=0)
    y = torch.stack(y_list, dim=0)

    num_scales = len(he_list_all[0])
    he_scales = []
    for s in range(num_scales):
        he_scales.append(torch.stack([items[s] for items in he_list_all], dim=0))
    return x_rna, he_scales, neigh_mean, y


def collate_dual_mean(batch):
    rna_list, he_list_all, spatial_mean_list, feature_mean_list, y_list = zip(*batch)
    x_rna = torch.stack(rna_list, dim=0)
    spatial_mean = torch.stack(spatial_mean_list, dim=0)
    feature_mean = torch.stack(feature_mean_list, dim=0)
    y = torch.stack(y_list, dim=0)

    num_scales = len(he_list_all[0])
    he_scales = []
    for s in range(num_scales):
        he_scales.append(torch.stack([items[s] for items in he_list_all], dim=0))
    return x_rna, he_scales, spatial_mean, feature_mean, y


def collate_single_gat(batch):
    rna_list, he_list_all, neighbors_raw_list, edge_feat_list, y_list = zip(*batch)
    x_rna = torch.stack(rna_list, dim=0)
    neighbors_raw = torch.stack(neighbors_raw_list, dim=0)
    edge_feat = torch.stack(edge_feat_list, dim=0)
    y = torch.stack(y_list, dim=0)

    num_scales = len(he_list_all[0])
    he_scales = []
    for s in range(num_scales):
        he_scales.append(torch.stack([items[s] for items in he_list_all], dim=0))
    return x_rna, he_scales, neighbors_raw, edge_feat, y


def collate_dual_gat(batch):
    (
        rna_list,
        he_list_all,
        spatial_nbr_list,
        spatial_edge_list,
        feature_nbr_list,
        feature_edge_list,
        y_list,
    ) = zip(*batch)

    x_rna = torch.stack(rna_list, dim=0)
    spatial_neighbors_raw = torch.stack(spatial_nbr_list, dim=0)
    spatial_edge_feat = torch.stack(spatial_edge_list, dim=0)
    feature_neighbors_raw = torch.stack(feature_nbr_list, dim=0)
    feature_edge_feat = torch.stack(feature_edge_list, dim=0)
    y = torch.stack(y_list, dim=0)

    num_scales = len(he_list_all[0])
    he_scales = []
    for s in range(num_scales):
        he_scales.append(torch.stack([items[s] for items in he_list_all], dim=0))

    return (
        x_rna,
        he_scales,
        spatial_neighbors_raw,
        spatial_edge_feat,
        feature_neighbors_raw,
        feature_edge_feat,
        y,
    )


# =========================================================
# 4. Network modules
# =========================================================
class ResMLPBlock(nn.Module):
    def __init__(self, dim=256, hidden=None, drop=0.15, res_scale=0.5):
        super().__init__()
        hidden = dim if hidden is None else hidden
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)
        self.res_scale = res_scale

    def forward(self, x):
        h = self.bn1(x)
        h = F.gelu(h)
        h = self.fc1(h)
        h = self.bn2(h)
        h = F.gelu(h)
        h = self.drop(h)
        h = self.fc2(h)
        return x + self.res_scale * h


class Tower(nn.Module):
    def __init__(self, in_dim, hidden_dim=768, emb_dim=256, dropout=0.15, n_blocks=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, emb_dim),
            nn.BatchNorm1d(emb_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.Sequential(*[
            ResMLPBlock(dim=emb_dim, hidden=emb_dim, drop=dropout, res_scale=0.5)
            for _ in range(n_blocks)
        ])

    def forward(self, x):
        x = self.net(x)
        x = self.blocks(x)
        return x


class CorrHead(nn.Module):
    def __init__(self, A_init: torch.Tensor, corr_scale=0.15):
        super().__init__()
        self.A = nn.Parameter(A_init.clone())
        self.corr_scale = corr_scale

    def forward(self, y):
        return y + self.corr_scale * (y @ self.A.T)


class ProteinPredictorHead(nn.Module):
    def __init__(self, emb_dim, out_dim, dropout, A_init):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(emb_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            ResMLPBlock(256, 256, dropout * 0.8, 0.5),
            nn.Linear(256, out_dim),
        )
        self.corr_head = CorrHead(A_init=A_init, corr_scale=0.15)

    def forward(self, z):
        y = self.decoder(z)
        y = self.corr_head(y)
        return y


class MultiScaleTokenEncoder(nn.Module):
    def __init__(self, he_dims, emb_dim=256, dropout=0.1):
        super().__init__()
        self.num_scales = len(he_dims)
        self.proj_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d, emb_dim),
                nn.LayerNorm(emb_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            for d in he_dims
        ])
        self.scale_pos = nn.Parameter(torch.randn(1, self.num_scales, emb_dim) * 0.02)

    def forward(self, x_he_list):
        tokens = []
        for x, proj in zip(x_he_list, self.proj_layers):
            tok = proj(x).unsqueeze(1)
            tokens.append(tok)
        tokens = torch.cat(tokens, dim=1)
        tokens = tokens + self.scale_pos
        return tokens


class CrossAttentionBlock(nn.Module):
    def __init__(self, emb_dim=256, num_heads=4, dropout=0.1):
        super().__init__()
        assert emb_dim % num_heads == 0
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(emb_dim, emb_dim)
        self.k_proj = nn.Linear(emb_dim, emb_dim)
        self.v_proj = nn.Linear(emb_dim, emb_dim)
        self.out_proj = nn.Linear(emb_dim, emb_dim)

        self.attn_drop = nn.Dropout(dropout)
        self.out_drop = nn.Dropout(dropout)
        self.norm_q = nn.LayerNorm(emb_dim)
        self.norm_kv = nn.LayerNorm(emb_dim)

    def forward(self, q_vec, kv_tokens):
        B, T, D = kv_tokens.shape

        q = self.norm_q(q_vec).unsqueeze(1)
        kv = self.norm_kv(kv_tokens)

        q = self.q_proj(q).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(kv).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn_logits, dim=-1)
        self.last_attn = attn.detach()
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, 1, D).squeeze(1)
        out = self.out_proj(out)
        out = self.out_drop(out)
        return out, attn


class NeighborGATBlock(nn.Module):
    def __init__(self, in_dim=256, out_dim=256, num_heads=4, edge_dim=3, dropout=0.1):
        super().__init__()
        assert out_dim % num_heads == 0
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads

        self.q_proj = nn.Linear(in_dim, out_dim)
        self.k_proj = nn.Linear(in_dim, out_dim)
        self.v_proj = nn.Linear(in_dim, out_dim)
        self.e_proj = nn.Linear(edge_dim, out_dim)

        self.attn_vec = nn.Parameter(torch.randn(num_heads, self.head_dim) * 0.02)
        self.out_proj = nn.Linear(out_dim, out_dim)
        self.drop = nn.Dropout(dropout)
        self.norm_center = nn.LayerNorm(in_dim)
        self.norm_neighbor = nn.LayerNorm(in_dim)

    def forward(self, z_center, z_neighbors, edge_feat):
        B, K, D = z_neighbors.shape

        center = self.norm_center(z_center)
        neigh = self.norm_neighbor(z_neighbors)

        q = self.q_proj(center).view(B, self.num_heads, self.head_dim)
        k = self.k_proj(neigh).view(B, K, self.num_heads, self.head_dim)
        v = self.v_proj(neigh).view(B, K, self.num_heads, self.head_dim)
        e = self.e_proj(edge_feat).view(B, K, self.num_heads, self.head_dim)

        q_expand = q.unsqueeze(1)
        h = torch.tanh(q_expand + k + e)
        attn_logits = (h * self.attn_vec.unsqueeze(0).unsqueeze(0)).sum(dim=-1)
        attn_logits = attn_logits.transpose(1, 2)

        attn = torch.softmax(attn_logits, dim=-1)
        attn = self.drop(attn)

        v = v.transpose(1, 2)
        out = torch.sum(attn.unsqueeze(-1) * v, dim=2)
        out = out.reshape(B, self.out_dim)
        out = self.out_proj(out)
        out = self.drop(out)
        return out


class NeighborGATv2Block(nn.Module):
    def __init__(self, in_dim=256, out_dim=256, num_heads=4, edge_dim=3, dropout=0.1):
        super().__init__()
        assert out_dim % num_heads == 0
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads

        self.joint_proj = nn.Linear(in_dim * 2 + edge_dim, out_dim)
        self.v_proj = nn.Linear(in_dim, out_dim)
        self.attn_vec = nn.Parameter(torch.randn(num_heads, self.head_dim) * 0.02)
        self.out_proj = nn.Linear(out_dim, out_dim)
        self.drop = nn.Dropout(dropout)
        self.norm_center = nn.LayerNorm(in_dim)
        self.norm_neighbor = nn.LayerNorm(in_dim)
        self.norm_edge = nn.LayerNorm(edge_dim)

    def forward(self, z_center, z_neighbors, edge_feat):
        B, K, _ = z_neighbors.shape
        center = self.norm_center(z_center).unsqueeze(1).expand(-1, K, -1)
        neigh = self.norm_neighbor(z_neighbors)
        edge = self.norm_edge(edge_feat)

        joint = torch.cat([center, neigh, edge], dim=-1)
        h = F.leaky_relu(self.joint_proj(joint), negative_slope=0.2)
        h = h.view(B, K, self.num_heads, self.head_dim)

        attn_logits = (h * self.attn_vec.unsqueeze(0).unsqueeze(0)).sum(dim=-1)
        attn_logits = attn_logits.transpose(1, 2)
        attn = torch.softmax(attn_logits, dim=-1)
        attn = self.drop(attn)

        v = self.v_proj(neigh).view(B, K, self.num_heads, self.head_dim).transpose(1, 2)
        out = torch.sum(attn.unsqueeze(-1) * v, dim=2)
        out = out.reshape(B, self.out_dim)
        out = self.out_proj(out)
        out = self.drop(out)
        return out


class RNAHEBackbone(nn.Module):
    def __init__(self, rna_dim, he_dims, rna_hidden_dim, emb_dim, dropout, att_heads, alpha_init):
        super().__init__()
        self.rna_tower = Tower(rna_dim, rna_hidden_dim, emb_dim, dropout, n_blocks=3)
        self.ms_token_encoder = MultiScaleTokenEncoder(he_dims, emb_dim, dropout * 0.8)
        self.cross_attn = CrossAttentionBlock(emb_dim, att_heads, dropout * 0.8)

        self.alpha_logit = nn.Parameter(torch.tensor(np.log(alpha_init / (1 - alpha_init)), dtype=torch.float32))
        self.post_fuse = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, emb_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.8),
        )

    def encode_rna_only(self, x_rna):
        return self.rna_tower(x_rna)

    def encode(self, x_rna, x_he_list):
        z_rna = self.rna_tower(x_rna)
        he_tokens = self.ms_token_encoder(x_he_list)
        z_cross, _ = self.cross_attn(z_rna, he_tokens)

        alpha = torch.sigmoid(self.alpha_logit)
        z_fuse = z_rna + alpha * z_cross
        z_fuse = z_fuse + self.post_fuse(z_fuse)
        return z_rna, z_fuse


# =========================================================
# 5. Models
# =========================================================
class NoGraphModel(nn.Module):
    def __init__(self, rna_dim, he_dims, out_dim, A_init):
        super().__init__()
        self.backbone = RNAHEBackbone(
            rna_dim, he_dims, RNA_HIDDEN_DIM, EMB_DIM, DROPOUT, ATT_NUM_HEADS, ATT_ALPHA_INIT
        )
        self.head = ProteinPredictorHead(EMB_DIM, out_dim, DROPOUT, A_init)
        self.uncertainty = UncertaintyLoss(out_dim)

    def forward(self, x_rna, x_he_list):
        _, z_fuse = self.backbone.encode(x_rna, x_he_list)
        return self.head(z_fuse)

    def compute_loss(self, pred, target):
        return compute_multitask_loss(pred, target, self.uncertainty)


class SingleMeanGraphModel(nn.Module):
    def __init__(self, rna_dim, he_dims, out_dim, A_init):
        super().__init__()
        self.backbone = RNAHEBackbone(
            rna_dim, he_dims, RNA_HIDDEN_DIM, EMB_DIM, DROPOUT, ATT_NUM_HEADS, ATT_ALPHA_INIT
        )
        self.res_tower = Tower(rna_dim, RNA_HIDDEN_DIM // 2, EMB_DIM, DROPOUT * 0.8, n_blocks=2)
        self.beta_logit = nn.Parameter(torch.tensor(np.log(GRAPH_BETA_INIT / (1 - GRAPH_BETA_INIT)), dtype=torch.float32))
        self.gate = nn.Sequential(
            nn.Linear(EMB_DIM * 2, EMB_DIM),
            nn.GELU(),
            nn.Dropout(DROPOUT * 0.5),
            nn.Linear(EMB_DIM, EMB_DIM),
            nn.Sigmoid(),
        )
        self.head = ProteinPredictorHead(EMB_DIM, out_dim, DROPOUT, A_init)
        self.uncertainty = UncertaintyLoss(out_dim)

    def forward(self, x_rna, x_he_list, neigh_mean):
        _, z_fuse = self.backbone.encode(x_rna, x_he_list)
        z_res = self.res_tower(neigh_mean)
        beta = torch.sigmoid(self.beta_logit)
        z_res = beta * z_res
        g = self.gate(torch.cat([z_fuse, z_res], dim=1))
        z = z_fuse + g * z_res
        return self.head(z)

    def compute_loss(self, pred, target):
        return compute_multitask_loss(pred, target, self.uncertainty)


class DualMeanGraphModel(nn.Module):
    def __init__(self, rna_dim, he_dims, out_dim, A_init):
        super().__init__()
        self.backbone = RNAHEBackbone(
            rna_dim, he_dims, RNA_HIDDEN_DIM, EMB_DIM, DROPOUT, ATT_NUM_HEADS, ATT_ALPHA_INIT
        )
        self.spatial_tower = Tower(rna_dim, RNA_HIDDEN_DIM // 2, EMB_DIM, DROPOUT * 0.8, n_blocks=2)
        self.feature_tower = Tower(rna_dim, RNA_HIDDEN_DIM // 2, EMB_DIM, DROPOUT * 0.8, n_blocks=2)

        self.beta_spatial_logit = nn.Parameter(torch.tensor(np.log(GRAPH_BETA_INIT / (1 - GRAPH_BETA_INIT)), dtype=torch.float32))
        self.beta_feature_logit = nn.Parameter(torch.tensor(np.log(GRAPH_BETA_INIT / (1 - GRAPH_BETA_INIT)), dtype=torch.float32))

        self.spatial_gate = nn.Sequential(
            nn.Linear(EMB_DIM * 2, EMB_DIM),
            nn.GELU(),
            nn.Dropout(DROPOUT * 0.5),
            nn.Linear(EMB_DIM, EMB_DIM),
            nn.Sigmoid(),
        )
        self.feature_gate = nn.Sequential(
            nn.Linear(EMB_DIM * 2, EMB_DIM),
            nn.GELU(),
            nn.Dropout(DROPOUT * 0.5),
            nn.Linear(EMB_DIM, EMB_DIM),
            nn.Sigmoid(),
        )
        self.head = ProteinPredictorHead(EMB_DIM, out_dim, DROPOUT, A_init)
        self.uncertainty = UncertaintyLoss(out_dim)

    def forward(self, x_rna, x_he_list, spatial_mean, feature_mean):
        _, z_fuse = self.backbone.encode(x_rna, x_he_list)

        z_spatial = self.spatial_tower(spatial_mean)
        z_feature = self.feature_tower(feature_mean)

        beta_s = torch.sigmoid(self.beta_spatial_logit)
        beta_f = torch.sigmoid(self.beta_feature_logit)
        z_spatial = beta_s * z_spatial
        z_feature = beta_f * z_feature

        g_s = self.spatial_gate(torch.cat([z_fuse, z_spatial], dim=1))
        g_f = self.feature_gate(torch.cat([z_fuse, z_feature], dim=1))

        z = z_fuse + g_s * z_spatial + g_f * z_feature
        return self.head(z)

    def compute_loss(self, pred, target):
        return compute_multitask_loss(pred, target, self.uncertainty)


class SingleGATGraphModel(nn.Module):
    def __init__(self, rna_dim, he_dims, out_dim, A_init, graph_input_mode="rna"):
        super().__init__()
        assert graph_input_mode in ["rna", "fuse"]
        self.graph_input_mode = graph_input_mode

        self.backbone = RNAHEBackbone(
            rna_dim, he_dims, RNA_HIDDEN_DIM, EMB_DIM, DROPOUT, ATT_NUM_HEADS, ATT_ALPHA_INIT
        )
        self.gat = NeighborGATBlock(
            in_dim=EMB_DIM,
            out_dim=EMB_DIM,
            num_heads=GRAPH_NUM_HEADS,
            edge_dim=3,
            dropout=DROPOUT * 0.8,
        )
        self.beta_logit = nn.Parameter(torch.tensor(np.log(GRAPH_BETA_INIT / (1 - GRAPH_BETA_INIT)), dtype=torch.float32))
        self.gate = nn.Sequential(
            nn.Linear(EMB_DIM * 2, EMB_DIM),
            nn.GELU(),
            nn.Dropout(DROPOUT * 0.5),
            nn.Linear(EMB_DIM, EMB_DIM),
            nn.Sigmoid(),
        )
        self.head = ProteinPredictorHead(EMB_DIM, out_dim, DROPOUT, A_init)
        self.uncertainty = UncertaintyLoss(out_dim)

    def forward(self, x_rna, x_he_list, neighbors_raw, edge_feat):
        z_rna, z_fuse = self.backbone.encode(x_rna, x_he_list)
        graph_center = z_rna if self.graph_input_mode == "rna" else z_fuse

        B, K, G = neighbors_raw.shape
        z_neighbors = self.backbone.encode_rna_only(neighbors_raw.reshape(B * K, G)).reshape(B, K, -1)

        z_res = self.gat(graph_center, z_neighbors, edge_feat)
        beta = torch.sigmoid(self.beta_logit)
        z_res = beta * z_res

        g = self.gate(torch.cat([z_fuse, z_res], dim=1))
        z = z_fuse + g * z_res
        return self.head(z)

    def compute_loss(self, pred, target):
        return compute_multitask_loss(pred, target, self.uncertainty)


class DualGATGraphModel(nn.Module):
    def __init__(self, rna_dim, he_dims, out_dim, A_init, graph_input_mode="rna"):
        super().__init__()
        assert graph_input_mode in ["rna", "fuse"]
        self.graph_input_mode = graph_input_mode

        self.backbone = RNAHEBackbone(
            rna_dim, he_dims, RNA_HIDDEN_DIM, EMB_DIM, DROPOUT, ATT_NUM_HEADS, ATT_ALPHA_INIT
        )
        self.spatial_gat = NeighborGATBlock(
            in_dim=EMB_DIM, out_dim=EMB_DIM, num_heads=GRAPH_NUM_HEADS, edge_dim=3, dropout=DROPOUT * 0.8
        )
        self.feature_gat = NeighborGATBlock(
            in_dim=EMB_DIM, out_dim=EMB_DIM, num_heads=GRAPH_NUM_HEADS, edge_dim=3, dropout=DROPOUT * 0.8
        )

        self.beta_spatial_logit = nn.Parameter(torch.tensor(np.log(GRAPH_BETA_INIT / (1 - GRAPH_BETA_INIT)), dtype=torch.float32))
        self.beta_feature_logit = nn.Parameter(torch.tensor(np.log(GRAPH_BETA_INIT / (1 - GRAPH_BETA_INIT)), dtype=torch.float32))

        self.spatial_gate = nn.Sequential(
            nn.Linear(EMB_DIM * 2, EMB_DIM),
            nn.GELU(),
            nn.Dropout(DROPOUT * 0.5),
            nn.Linear(EMB_DIM, EMB_DIM),
            nn.Sigmoid(),
        )
        self.feature_gate = nn.Sequential(
            nn.Linear(EMB_DIM * 2, EMB_DIM),
            nn.GELU(),
            nn.Dropout(DROPOUT * 0.5),
            nn.Linear(EMB_DIM, EMB_DIM),
            nn.Sigmoid(),
        )
        self.head = ProteinPredictorHead(EMB_DIM, out_dim, DROPOUT, A_init)
        self.uncertainty = UncertaintyLoss(out_dim)

    def forward(self, x_rna, x_he_list, spatial_neighbors_raw, spatial_edge_feat, feature_neighbors_raw, feature_edge_feat):
        z_rna, z_fuse = self.backbone.encode(x_rna, x_he_list)
        graph_center = z_rna if self.graph_input_mode == "rna" else z_fuse

        B, K1, G = spatial_neighbors_raw.shape
        z_spatial_neighbors = self.backbone.encode_rna_only(spatial_neighbors_raw.reshape(B * K1, G)).reshape(B, K1, -1)

        B, K2, G = feature_neighbors_raw.shape
        z_feature_neighbors = self.backbone.encode_rna_only(feature_neighbors_raw.reshape(B * K2, G)).reshape(B, K2, -1)

        z_spatial = self.spatial_gat(graph_center, z_spatial_neighbors, spatial_edge_feat)
        z_feature = self.feature_gat(graph_center, z_feature_neighbors, feature_edge_feat)

        beta_s = torch.sigmoid(self.beta_spatial_logit)
        beta_f = torch.sigmoid(self.beta_feature_logit)
        z_spatial = beta_s * z_spatial
        z_feature = beta_f * z_feature

        g_s = self.spatial_gate(torch.cat([z_fuse, z_spatial], dim=1))
        g_f = self.feature_gate(torch.cat([z_fuse, z_feature], dim=1))

        z = z_fuse + g_s * z_spatial + g_f * z_feature
        return self.head(z)

    def compute_loss(self, pred, target):
        return compute_multitask_loss(pred, target, self.uncertainty)


class DualGATv2GraphModel(DualGATGraphModel):
    def __init__(self, rna_dim, he_dims, out_dim, A_init, graph_input_mode="fuse"):
        super().__init__(rna_dim, he_dims, out_dim, A_init, graph_input_mode=graph_input_mode)
        self.spatial_gat = NeighborGATv2Block(
            in_dim=EMB_DIM, out_dim=EMB_DIM, num_heads=GRAPH_NUM_HEADS, edge_dim=3, dropout=DROPOUT * 0.8
        )
        self.feature_gat = NeighborGATv2Block(
            in_dim=EMB_DIM, out_dim=EMB_DIM, num_heads=GRAPH_NUM_HEADS, edge_dim=3, dropout=DROPOUT * 0.8
        )


# =========================================================
# 6. Data loading (tonsil)
# =========================================================
def load_rna_protein():
    for p in [TRAIN_GENE_PATH, TEST_GENE_PATH, TRAIN_PRO_PATH, TEST_PRO_PATH]:
        check_file_exists(p)

    adata_train_rna = ad.read_h5ad(TRAIN_GENE_PATH)
    adata_test_rna = ad.read_h5ad(TEST_GENE_PATH)
    adata_train_pro = ad.read_h5ad(TRAIN_PRO_PATH)
    adata_test_pro = ad.read_h5ad(TEST_PRO_PATH)

    common_train = adata_train_rna.obs_names.intersection(adata_train_pro.obs_names)
    adata_train_rna = adata_train_rna[common_train].copy()
    adata_train_pro = adata_train_pro[common_train].copy()
    assert np.all(adata_train_rna.obs_names == adata_train_pro.obs_names)

    common_test = adata_test_rna.obs_names.intersection(adata_test_pro.obs_names)
    adata_test_rna = adata_test_rna[common_test].copy()
    adata_test_pro = adata_test_pro[common_test].copy()
    assert np.all(adata_test_rna.obs_names == adata_test_pro.obs_names)

    train_gene_names = adata_train_rna.var_names
    if not np.array_equal(adata_test_rna.var_names, train_gene_names):
        common_genes = train_gene_names.intersection(adata_test_rna.var_names)
        if len(common_genes) != len(train_gene_names):
            raise ValueError(f"Test gene set incomplete: train={len(train_gene_names)}, common={len(common_genes)}")
        adata_test_rna = adata_test_rna[:, train_gene_names].copy()

    train_pro_names = adata_train_pro.var_names
    if not np.array_equal(adata_test_pro.var_names, train_pro_names):
        common_pro = train_pro_names.intersection(adata_test_pro.var_names)
        if len(common_pro) != len(train_pro_names):
            raise ValueError(f"Test protein set incomplete: train={len(train_pro_names)}, common={len(common_pro)}")
        adata_test_pro = adata_test_pro[:, train_pro_names].copy()

    keep_proteins = [p for p in adata_train_pro.var_names if p not in EXCLUDE_PROTEINS]
    removed = [p for p in adata_train_pro.var_names if p in EXCLUDE_PROTEINS]
    if removed:
        print(f"Excluding control antibodies from protein targets: {removed}")
        adata_train_pro = adata_train_pro[:, keep_proteins].copy()
        adata_test_pro = adata_test_pro[:, keep_proteins].copy()

    xr_train_all = to_dense(adata_train_rna.X).astype(np.float32)
    y_train_all = to_dense(adata_train_pro.X).astype(np.float32)
    xr_test_all = to_dense(adata_test_rna.X).astype(np.float32)
    y_test_all = to_dense(adata_test_pro.X).astype(np.float32)
    prot_names = list(adata_train_pro.var_names)
    BARCODES["train"] = [str(b) for b in adata_train_rna.obs_names]
    BARCODES["test"] = [str(b) for b in adata_test_rna.obs_names]
    return xr_train_all, y_train_all, xr_test_all, y_test_all, prot_names


def he_barcodes(feature_path: Path, prefix: str):
    """Barcode order of an image-feature array, from <prefix>_barcodes.csv in the same folder."""
    path = feature_path.parent / f"{prefix}_barcodes.csv"
    if not path.exists():
        raise FileNotFoundError(f"No barcode file next to {feature_path} (expected {path.name})")
    df = pd.read_csv(path)
    return df["barcode" if "barcode" in df.columns else df.columns[0]].astype(str).tolist()


def load_multiscale_he(patch_sizes):
    """Image features per field of view, aligned to the RNA barcodes (call load_rna_protein() first)."""
    if not BARCODES:
        raise RuntimeError("load_rna_protein() must run before load_multiscale_he()")
    train_list = []
    test_list = []
    for p in patch_sizes:
        for split, prefix, out in (("train", TRAIN_HE_PREFIX, train_list), ("test", TEST_HE_PREFIX, test_list)):
            path = resolve_he_path(split, p)
            check_file_exists(path)
            arr = np.load(path).astype(np.float32)
            out.append(align_to_barcodes(arr, he_barcodes(path, prefix), BARCODES[split], f"{split} image features {path.name}"))
    return train_list, test_list


# =========================================================
# 7. Model construction, training and prediction
def build_corr_init(y_train):
    R = np.corrcoef(y_train.T)
    R = np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return torch.tensor(R, dtype=torch.float32)


def build_model(model_name, rna_dim, he_dims, out_dim, A_init):
    spec = MODEL_SPECS[model_name]
    kind = spec["kind"]

    if kind == "no_graph":
        return NoGraphModel(rna_dim, he_dims, out_dim, A_init).to(DEVICE)

    if kind == "single_mean":
        return SingleMeanGraphModel(rna_dim, he_dims, out_dim, A_init).to(DEVICE)

    if kind == "dual_mean":
        return DualMeanGraphModel(rna_dim, he_dims, out_dim, A_init).to(DEVICE)

    if kind == "single_gat":
        return SingleGATGraphModel(
            rna_dim, he_dims, out_dim, A_init, graph_input_mode=spec["graph_input_mode"]
        ).to(DEVICE)

    if kind == "dual_gat":
        model_cls = DualGATv2GraphModel if spec.get("gat_block") == "gatv2" else DualGATGraphModel
        return model_cls(rna_dim, he_dims, out_dim, A_init, graph_input_mode=spec["graph_input_mode"]).to(DEVICE)

    raise ValueError(model_name)


def build_dataset_and_collate(model_name, split_pack):
    spec = MODEL_SPECS[model_name]
    kind = spec["kind"]

    x_rna = split_pack["x_rna"]
    x_he_list = split_pack["x_he_list"]
    y = split_pack["y"]

    if kind == "no_graph":
        ds = NoGraphDataset(x_rna, x_he_list, y)
        return ds, collate_nograph

    if kind == "single_mean":
        graph_name = spec["graph"]
        ds = SingleMeanDataset(x_rna, x_he_list, split_pack[f"{graph_name}_mean"], y)
        return ds, collate_single_mean

    if kind == "dual_mean":
        ds = DualMeanDataset(
            x_rna, x_he_list,
            split_pack["spatial_mean"],
            split_pack["feature_mean"],
            y
        )
        return ds, collate_dual_mean

    if kind == "single_gat":
        graph_name = spec["graph"]
        ds = SingleGATDataset(
            x_rna, x_he_list,
            split_pack[f"{graph_name}_neighbors_raw"],
            split_pack[f"{graph_name}_edge_feat"],
            y
        )
        return ds, collate_single_gat

    if kind == "dual_gat":
        ds = DualGATDataset(
            x_rna, x_he_list,
            split_pack["spatial_neighbors_raw"],
            split_pack["spatial_edge_feat"],
            split_pack["feature_neighbors_raw"],
            split_pack["feature_edge_feat"],
            y
        )
        return ds, collate_dual_gat

    raise ValueError(model_name)


@torch.no_grad()
def predict_model(model, loader, model_name):
    spec = MODEL_SPECS[model_name]
    kind = spec["kind"]

    model.eval()
    preds, trues = [], []
    loss_vals, mse_vals, ccc_vals, rank_vals = [], [], [], []

    for batch in loader:
        if kind == "no_graph":
            xb_rna, xb_he_list, yb = batch
            xb_rna = xb_rna.to(DEVICE).float()
            xb_he_list = [x.to(DEVICE).float() for x in xb_he_list]
            yb = yb.to(DEVICE).float()
            pred = model(xb_rna, xb_he_list)

        elif kind == "single_mean":
            xb_rna, xb_he_list, neigh_mean, yb = batch
            xb_rna = xb_rna.to(DEVICE).float()
            xb_he_list = [x.to(DEVICE).float() for x in xb_he_list]
            neigh_mean = neigh_mean.to(DEVICE).float()
            yb = yb.to(DEVICE).float()
            pred = model(xb_rna, xb_he_list, neigh_mean)

        elif kind == "dual_mean":
            xb_rna, xb_he_list, spatial_mean, feature_mean, yb = batch
            xb_rna = xb_rna.to(DEVICE).float()
            xb_he_list = [x.to(DEVICE).float() for x in xb_he_list]
            spatial_mean = spatial_mean.to(DEVICE).float()
            feature_mean = feature_mean.to(DEVICE).float()
            yb = yb.to(DEVICE).float()
            pred = model(xb_rna, xb_he_list, spatial_mean, feature_mean)

        elif kind == "single_gat":
            xb_rna, xb_he_list, neighbors_raw, edge_feat, yb = batch
            xb_rna = xb_rna.to(DEVICE).float()
            xb_he_list = [x.to(DEVICE).float() for x in xb_he_list]
            neighbors_raw = neighbors_raw.to(DEVICE).float()
            edge_feat = edge_feat.to(DEVICE).float()
            yb = yb.to(DEVICE).float()
            pred = model(xb_rna, xb_he_list, neighbors_raw, edge_feat)

        elif kind == "dual_gat":
            (
                xb_rna,
                xb_he_list,
                spatial_neighbors_raw,
                spatial_edge_feat,
                feature_neighbors_raw,
                feature_edge_feat,
                yb,
            ) = batch
            xb_rna = xb_rna.to(DEVICE).float()
            xb_he_list = [x.to(DEVICE).float() for x in xb_he_list]
            spatial_neighbors_raw = spatial_neighbors_raw.to(DEVICE).float()
            spatial_edge_feat = spatial_edge_feat.to(DEVICE).float()
            feature_neighbors_raw = feature_neighbors_raw.to(DEVICE).float()
            feature_edge_feat = feature_edge_feat.to(DEVICE).float()
            yb = yb.to(DEVICE).float()
            pred = model(
                xb_rna, xb_he_list,
                spatial_neighbors_raw, spatial_edge_feat,
                feature_neighbors_raw, feature_edge_feat
            )
        else:
            raise ValueError(model_name)

        loss, loss_dict = model.compute_loss(pred, yb)

        preds.append(pred.cpu().numpy())
        trues.append(yb.cpu().numpy())
        loss_vals.append(float(loss.item()))
        mse_vals.append(float(loss_dict["loss_mse"]))
        ccc_vals.append(float(loss_dict["loss_ccc"]))
        rank_vals.append(float(loss_dict["loss_rank"]))

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)

    loss_pack = {
        "loss_total_scaled": float(np.mean(loss_vals)),
        "loss_mse_scaled": float(np.mean(mse_vals)),
        "loss_ccc_scaled": float(np.mean(ccc_vals)),
        "loss_rank_scaled": float(np.mean(rank_vals)),
    }
    return preds, trues, loss_pack


@torch.no_grad()
def export_cross_attention_summary(model, loader, model_name, patch_sizes, out_dir):
    if MODEL_SPECS[model_name]["kind"] != "dual_gat":
        return None

    model.eval()
    attn_chunks = []

    for batch in loader:
        (
            xb_rna,
            xb_he_list,
            spatial_neighbors_raw,
            spatial_edge_feat,
            feature_neighbors_raw,
            feature_edge_feat,
            _,
        ) = batch
        xb_rna = xb_rna.to(DEVICE).float()
        xb_he_list = [x.to(DEVICE).float() for x in xb_he_list]
        spatial_neighbors_raw = spatial_neighbors_raw.to(DEVICE).float()
        spatial_edge_feat = spatial_edge_feat.to(DEVICE).float()
        feature_neighbors_raw = feature_neighbors_raw.to(DEVICE).float()
        feature_edge_feat = feature_edge_feat.to(DEVICE).float()

        _ = model(
            xb_rna, xb_he_list,
            spatial_neighbors_raw, spatial_edge_feat,
            feature_neighbors_raw, feature_edge_feat
        )
        attn = model.backbone.cross_attn.last_attn.squeeze(2).detach().cpu().numpy()
        attn_chunks.append(attn)

    attn_all = np.concatenate(attn_chunks, axis=0)  # [N, heads, scales]
    scale_mean = attn_all.mean(axis=(0, 1))
    scale_std = attn_all.reshape(-1, attn_all.shape[-1]).std(axis=0)
    head_mean = attn_all.mean(axis=0)

    rows = []
    for i, patch_size in enumerate(patch_sizes):
        rows.append({
            "patch_size": patch_size,
            "mean_attention": float(scale_mean[i]),
            "std_attention_over_spot_heads": float(scale_std[i]),
            "rank_by_mean_attention": int(np.argsort(-scale_mean).tolist().index(i) + 1),
        })
    scale_df = pd.DataFrame(rows).sort_values("rank_by_mean_attention")
    scale_df.to_csv(out_dir / f"{model_name}_test_cross_attention_by_scale.csv", index=False, encoding="utf-8-sig")

    head_rows = []
    for h in range(head_mean.shape[0]):
        for i, patch_size in enumerate(patch_sizes):
            head_rows.append({
                "head": h,
                "patch_size": patch_size,
                "mean_attention": float(head_mean[h, i]),
            })
    pd.DataFrame(head_rows).to_csv(out_dir / f"{model_name}_test_cross_attention_by_head_scale.csv", index=False, encoding="utf-8-sig")
    np.save(out_dir / f"{model_name}_test_cross_attention_spot_head_scale.npy", attn_all.astype(np.float32))
    return scale_df


def run_one_model(model_name, out_dir, train_pack, val_pack, test_pack, he_dims, prot_names, patch_sizes):
    model = build_model(
        model_name,
        train_pack["x_rna"].shape[1],
        he_dims,
        train_pack["y"].shape[1],
        build_corr_init(train_pack["y"])
    )

    train_ds, train_collate = build_dataset_and_collate(model_name, train_pack)
    val_ds, val_collate = build_dataset_and_collate(model_name, val_pack)
    test_ds, test_collate = build_dataset_and_collate(model_name, test_pack)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE_TRAIN, shuffle=True, drop_last=False,
        num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda"), collate_fn=train_collate
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE_EVAL, shuffle=False, drop_last=False,
        num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda"), collate_fn=val_collate
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE_EVAL, shuffle=False, drop_last=False,
        num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda"), collate_fn=test_collate
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=4
    )

    best_scc = -999.0
    best_epoch = -1
    best_state = None
    patience_count = 0
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss_total, train_loss_mse, train_loss_ccc, train_loss_rank = [], [], [], []

        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)

            spec = MODEL_SPECS[model_name]
            kind = spec["kind"]

            if kind == "no_graph":
                xb_rna, xb_he_list, yb = batch
                xb_rna = xb_rna.to(DEVICE).float()
                xb_he_list = [x.to(DEVICE).float() for x in xb_he_list]
                yb = yb.to(DEVICE).float()
                pred = model(xb_rna, xb_he_list)

            elif kind == "single_mean":
                xb_rna, xb_he_list, neigh_mean, yb = batch
                xb_rna = xb_rna.to(DEVICE).float()
                xb_he_list = [x.to(DEVICE).float() for x in xb_he_list]
                neigh_mean = neigh_mean.to(DEVICE).float()
                yb = yb.to(DEVICE).float()
                pred = model(xb_rna, xb_he_list, neigh_mean)

            elif kind == "dual_mean":
                xb_rna, xb_he_list, spatial_mean, feature_mean, yb = batch
                xb_rna = xb_rna.to(DEVICE).float()
                xb_he_list = [x.to(DEVICE).float() for x in xb_he_list]
                spatial_mean = spatial_mean.to(DEVICE).float()
                feature_mean = feature_mean.to(DEVICE).float()
                yb = yb.to(DEVICE).float()
                pred = model(xb_rna, xb_he_list, spatial_mean, feature_mean)

            elif kind == "single_gat":
                xb_rna, xb_he_list, neighbors_raw, edge_feat, yb = batch
                xb_rna = xb_rna.to(DEVICE).float()
                xb_he_list = [x.to(DEVICE).float() for x in xb_he_list]
                neighbors_raw = neighbors_raw.to(DEVICE).float()
                edge_feat = edge_feat.to(DEVICE).float()
                yb = yb.to(DEVICE).float()
                pred = model(xb_rna, xb_he_list, neighbors_raw, edge_feat)

            elif kind == "dual_gat":
                (
                    xb_rna,
                    xb_he_list,
                    spatial_neighbors_raw,
                    spatial_edge_feat,
                    feature_neighbors_raw,
                    feature_edge_feat,
                    yb,
                ) = batch
                xb_rna = xb_rna.to(DEVICE).float()
                xb_he_list = [x.to(DEVICE).float() for x in xb_he_list]
                spatial_neighbors_raw = spatial_neighbors_raw.to(DEVICE).float()
                spatial_edge_feat = spatial_edge_feat.to(DEVICE).float()
                feature_neighbors_raw = feature_neighbors_raw.to(DEVICE).float()
                feature_edge_feat = feature_edge_feat.to(DEVICE).float()
                yb = yb.to(DEVICE).float()
                pred = model(
                    xb_rna, xb_he_list,
                    spatial_neighbors_raw, spatial_edge_feat,
                    feature_neighbors_raw, feature_edge_feat
                )
            else:
                raise ValueError(model_name)

            loss, loss_dict = model.compute_loss(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
            optimizer.step()

            train_loss_total.append(float(loss.item()))
            train_loss_mse.append(float(loss_dict["loss_mse"]))
            train_loss_ccc.append(float(loss_dict["loss_ccc"]))
            train_loss_rank.append(float(loss_dict["loss_rank"]))

        val_pred, val_true, val_loss_pack = predict_model(model, val_loader, model_name)
        val_res = summarize_metrics(val_true, val_pred, prot_names, val_loss_pack)

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss_total_scaled": float(np.mean(train_loss_total)),
            "train_loss_mse_scaled": float(np.mean(train_loss_mse)),
            "train_loss_ccc_scaled": float(np.mean(train_loss_ccc)),
            "train_loss_rank_scaled": float(np.mean(train_loss_rank)),
            "val_mean_scc": val_res["mean_scc"],
            "val_mean_pcc": val_res["mean_pcc"],
            "val_top10_scc_mean": val_res["top10_scc_mean"],
            "val_rmse_raw": val_res["rmse_raw"],
        }
        history.append(row)

        print(
            f"[{model_name}] Epoch {epoch:02d} | "
            f"train_total={row['train_loss_total_scaled']:.6f} | "
            f"val_scc={row['val_mean_scc']:.6f} | "
            f"val_pcc={row['val_mean_pcc']:.6f} | "
            f"top10={row['val_top10_scc_mean']:.6f}"
        )

        scheduler.step(val_res["mean_scc"] if np.isfinite(val_res["mean_scc"]) else -1.0)

        if np.isfinite(val_res["mean_scc"]) and val_res["mean_scc"] > best_scc:
            best_scc = val_res["mean_scc"]
            best_epoch = epoch
            patience_count = 0
            best_state = {"model": copy.deepcopy(model.state_dict())}
        else:
            patience_count += 1

        if patience_count >= PATIENCE:
            print(f"[{model_name}] Early stopping at epoch {epoch}, best epoch = {best_epoch}")
            break

    pd.DataFrame(history).to_csv(out_dir / f"{model_name}_train_history.csv", index=False, encoding="utf-8-sig")

    if best_state is not None:
        torch.save(best_state, out_dir / f"{model_name}_best_model.pt")
        model.load_state_dict(best_state["model"])

    val_pred, val_true, val_loss_pack = predict_model(model, val_loader, model_name)
    test_pred, test_true, test_loss_pack = predict_model(model, test_loader, model_name)

    val_res = summarize_metrics(val_true, val_pred, prot_names, val_loss_pack)
    test_res = summarize_metrics(test_true, test_pred, prot_names, test_loss_pack)

    val_res["per_protein_df"].to_csv(out_dir / f"{model_name}_val_per_protein.csv", index=False, encoding="utf-8-sig")
    test_res["per_protein_df"].to_csv(out_dir / f"{model_name}_test_per_protein.csv", index=False, encoding="utf-8-sig")

    np.save(out_dir / f"{model_name}_test_pred.npy", test_pred)
    np.save(out_dir / f"{model_name}_test_true.npy", test_true)
    attn_df = export_cross_attention_summary(model, test_loader, model_name, patch_sizes, out_dir)
    if attn_df is not None:
        print(f"[{model_name}] Test cross-attention by scale:")
        print(attn_df.to_string(index=False))

    del model, train_loader, val_loader, test_loader, train_ds, val_ds, test_ds
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "best_epoch": best_epoch,
        "val": val_res,
        "test": test_res,
        "test_pred": test_pred,
        "test_true": test_true,
    }


# =========================================================
# 8. Split packs
# =========================================================
def build_split_pack(
    x_rna,
    x_he_list,
    y,
    spatial_neighbors_raw,
    spatial_edge_feat,
    spatial_mean,
    feature_neighbors_raw,
    feature_edge_feat,
    feature_mean,
):
    return {
        "x_rna": x_rna.astype(np.float32),
        "x_he_list": [x.astype(np.float32) for x in x_he_list],
        "y": y.astype(np.float32),

        "spatial_neighbors_raw": spatial_neighbors_raw.astype(np.float32),
        "spatial_edge_feat": spatial_edge_feat.astype(np.float32),
        "spatial_mean": spatial_mean.astype(np.float32),

        "feature_neighbors_raw": feature_neighbors_raw.astype(np.float32),
        "feature_edge_feat": feature_edge_feat.astype(np.float32),
        "feature_mean": feature_mean.astype(np.float32),
    }
