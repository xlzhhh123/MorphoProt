"""Human liver (GSE278662) RNA, protein and coordinate preprocessing.

Train sections N123, N126; test sections Sah73, Sah80. RNA: genes detected in >= 3 training
spots -> 3,500 genes (genes of the measured proteins first, then training HVGs by normalized
dispersion) -> log1p(counts) -> gene-wise z-score with the training mean/std. Protein:
log1p(counts) -> per-protein z-score with the training mean/std; the four isotype-control
antibodies are removed.

    MORPHOPROT_RAW=raw/GSE278662 python preprocessing/preprocess_liver.py --out-dir data/liver \
        --gene-list gene_lists/liver_genes_3500.txt
"""
# -*- coding: utf-8 -*-
from __future__ import annotations

import gzip
import json
import shutil
import argparse
from collections import Counter
import os
from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from scipy.io import mmread
from scipy import sparse


RAW = Path(os.environ.get("MORPHOPROT_RAW", "raw/GSE278662"))  # downloaded GEO supplementary files
OUT = Path(os.environ.get("MORPHOPROT_DATA", "data")) / "liver"
GENE_DIR = OUT / "_scheme2_log1p_zscore_hvg3500_force_marker_in_tissue1"
PRO_DIR = OUT / "_protein_scheme_log1p_zscore_in_tissue1"
SAMPLE_DIR = OUT / "samples"

SAMPLES = ["N123", "N126", "Sah73", "Sah80"]
TRAIN_SAMPLES = ["N123", "N126"]
TEST_SAMPLES = ["Sah73", "Sah80"]
CONTROL_ANTIBODIES = {"mouse_IgG1k", "mouse_IgG2a", "mouse_IgG2bk", "rat_IgG2a"}
TARGET_GENE_N = 3500
MIN_CELLS = 3
TARGET_SUM = 1e4


def ensure_dirs() -> None:
    for d in [OUT, GENE_DIR, PRO_DIR, SAMPLE_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def read_gzip_lines(path: Path) -> list[str]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [x.rstrip("\n") for x in f]


def make_unique(names: list[str]) -> list[str]:
    seen: Counter[str] = Counter()
    out = []
    for name in names:
        seen[name] += 1
        out.append(name if seen[name] == 1 else f"{name}_{seen[name]}")
    return out


def read_features(sample: str) -> pd.DataFrame:
    rows = []
    with gzip.open(RAW / f"GSE278662_{sample}_features.tsv.gz", "rt", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            rows.append({"feature_id": parts[0], "feature_name": parts[1], "feature_type": parts[2]})
    df = pd.DataFrame(rows)
    df["unique_name"] = make_unique(df["feature_name"].astype(str).tolist())
    antibody = df["feature_type"].eq("Antibody Capture")
    df.loc[antibody, "unique_name"] = df.loc[antibody, "feature_id"].astype(str)
    return df


def copy_spatial_files(sample: str, scale: float) -> None:
    sdir = SAMPLE_DIR / sample
    sdir.mkdir(parents=True, exist_ok=True)
    for suffix, out_name in [
        ("tissue_hires_image.png.gz", "tissue_hires_image.png"),
        ("tissue_lowres_image.png.gz", "tissue_lowres_image.png"),
    ]:
        src = RAW / f"GSE278662_{sample}_{suffix}"
        dst = sdir / out_name
        if not dst.exists():
            with gzip.open(src, "rb") as fin, open(dst, "wb") as fout:
                shutil.copyfileobj(fin, fout)


def load_sample(sample: str, offset_x: float) -> dict:
    features = read_features(sample)
    barcodes = read_gzip_lines(RAW / f"GSE278662_{sample}_barcodes.tsv.gz")
    pos = pd.read_csv(RAW / f"GSE278662_{sample}_tissue_positions.csv.gz")
    scale = json.loads(gzip.open(RAW / f"GSE278662_{sample}_scalefactors_json.json.gz", "rt").read())["tissue_hires_scalef"]
    pos = pos[pos["in_tissue"].astype(int).eq(1)].copy()
    pos["barcode_raw"] = pos["barcode"].astype(str)
    pos["barcode"] = sample + ":" + pos["barcode_raw"]
    pos["pxl_row_in_hires"] = pos["pxl_row_in_fullres"].astype(float) * float(scale)
    pos["pxl_col_in_hires"] = pos["pxl_col_in_fullres"].astype(float) * float(scale)
    pos["pxl_row_in_hires_raw"] = pos["pxl_row_in_hires"]
    pos["pxl_col_in_hires_raw"] = pos["pxl_col_in_hires"]
    pos["pxl_col_in_hires"] = pos["pxl_col_in_hires"] + offset_x
    pos["sample"] = sample
    pos = pos.sort_values("barcode_raw")

    copy_spatial_files(sample, float(scale))
    sample_pos = pos.copy()
    sample_pos["barcode"] = sample_pos["barcode"].astype(str)
    sample_pos["pxl_col_in_hires"] = sample_pos["pxl_col_in_hires_raw"]
    sample_pos["pxl_row_in_hires"] = sample_pos["pxl_row_in_hires_raw"]
    sample_pos.to_csv(SAMPLE_DIR / sample / "tissue_positions_list_with_hires_coords.csv", index=False)

    raw_barcode_to_col = {b: i for i, b in enumerate(barcodes)}
    cols = np.asarray([raw_barcode_to_col[b] for b in pos["barcode_raw"]], dtype=np.int64)
    with gzip.open(RAW / f"GSE278662_{sample}_matrix.mtx.gz", "rb") as f:
        mat = mmread(f).tocsr()
    mat = mat[:, cols].T.tocsr()

    gene_idx = np.where(features["feature_type"].values == "Gene Expression")[0]
    pro_idx = np.where(features["feature_type"].values == "Antibody Capture")[0]
    genes = features.iloc[gene_idx]["unique_name"].astype(str).tolist()
    gene_symbols = features.iloc[gene_idx]["feature_name"].astype(str).tolist()
    proteins = features.iloc[pro_idx]["unique_name"].astype(str).tolist()
    protein_names = features.iloc[pro_idx]["feature_name"].astype(str).tolist()
    # 10x feature IDs and names can differ; drop a control antibody if either field matches
    keep_pro = [i for i, (fid, name) in enumerate(zip(proteins, protein_names))
                if fid not in CONTROL_ANTIBODIES and name not in CONTROL_ANTIBODIES]
    dropped = sorted(set(proteins) - {proteins[i] for i in keep_pro})
    if len(dropped) != len(CONTROL_ANTIBODIES):
        raise ValueError(f"{sample}: expected to drop {sorted(CONTROL_ANTIBODIES)}, dropped {dropped}")

    return {
        "sample": sample,
        "barcodes": pos["barcode"].astype(str).tolist(),
        "coords": pos[["barcode", "sample", "pxl_row_in_hires", "pxl_col_in_hires", "pxl_row_in_hires_raw", "pxl_col_in_hires_raw"]],
        "gene_counts": mat[:, gene_idx].tocsr(),
        "protein_counts": mat[:, pro_idx[keep_pro]].tocsr(),
        "genes": genes,
        "gene_symbols": gene_symbols,
        "proteins": [proteins[i] for i in keep_pro],
        "protein_names": [protein_names[i] for i in keep_pro],
    }


def stack_samples(samples: list[dict], key: str) -> sparse.csr_matrix:
    return sparse.vstack([s[key] for s in samples], format="csr")


def zscore_by_train(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, dtype=np.float64)
    std = train.std(axis=0, dtype=np.float64)
    std[std < 1e-6] = 1.0
    return (
        ((train - mean) / std).astype(np.float32),
        ((test - mean) / std).astype(np.float32),
        mean.astype(np.float32),
        std.astype(np.float32),
    )


def select_train_hvg_with_forced_markers(
    train_counts_min: sparse.csr_matrix,
    kept_genes: np.ndarray,
    kept_symbols: np.ndarray,
    protein_display: list[str],
) -> tuple[list[int], list[int], pd.DataFrame]:
    train_for_hvg = ad.AnnData(train_counts_min.copy())
    train_for_hvg.var_names = kept_genes.astype(str)
    train_for_hvg.var["gene_symbol"] = kept_symbols.astype(str)
    sc.pp.normalize_total(train_for_hvg, target_sum=TARGET_SUM)
    sc.pp.log1p(train_for_hvg)
    sc.pp.highly_variable_genes(
        train_for_hvg,
        n_top_genes=min(TARGET_GENE_N, train_for_hvg.n_vars),
        flavor="seurat",
        subset=False,
        inplace=True,
    )
    hvg_rank = train_for_hvg.var["dispersions_norm"].to_numpy()
    hvg_idx = np.where(train_for_hvg.var["highly_variable"].to_numpy(dtype=bool))[0].tolist()
    hvg_idx = sorted(hvg_idx, key=lambda i: (-float(hvg_rank[i]), str(kept_genes[i])))

    forced_candidates = {
        str(name).split("_")[0].upper()
        for name in protein_display
        if name not in CONTROL_ANTIBODIES
    }
    forced_idx = [
        i for i, symbol in enumerate(kept_symbols)
        if str(symbol).upper() in forced_candidates
    ]

    selected: list[int] = []
    selected_set: set[int] = set()
    for idx in forced_idx + hvg_idx:
        if idx not in selected_set:
            selected.append(idx)
            selected_set.add(idx)
        if len(selected) >= TARGET_GENE_N:
            break
    if len(selected) < TARGET_GENE_N:
        filler = np.argsort(hvg_rank)[::-1]
        for idx in filler:
            idx = int(idx)
            if idx not in selected_set:
                selected.append(idx)
                selected_set.add(idx)
            if len(selected) >= TARGET_GENE_N:
                break

    marker_rows = []
    kept_symbol_upper = {str(x).upper() for x in kept_symbols}
    selected_symbol_upper = {str(kept_symbols[i]).upper() for i in selected}
    for marker in sorted(forced_candidates):
        marker_rows.append({
            "marker": marker,
            "after_min_cells": marker in kept_symbol_upper,
            "in_final_gene_set": marker in selected_symbol_upper,
        })
    return selected, forced_idx, pd.DataFrame(marker_rows)


def main() -> None:
    global OUT, GENE_DIR, PRO_DIR, SAMPLE_DIR, TRAIN_SAMPLES, TEST_SAMPLES
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, default=str(OUT))
    parser.add_argument("--train-samples", type=str, default=",".join(TRAIN_SAMPLES))
    parser.add_argument("--test-samples", type=str, default=",".join(TEST_SAMPLES))
    parser.add_argument("--gene-list", type=Path, default=None,
                        help="fixed 3,500-gene list in model column order (gene_lists/liver_genes_3500.txt); "
                             "skips HVG selection and reproduces the paper inputs exactly")
    args = parser.parse_args()

    OUT = Path(args.out_dir)
    GENE_DIR = OUT / "_scheme2_log1p_zscore_hvg3500_force_marker_in_tissue1"
    PRO_DIR = OUT / "_protein_scheme_log1p_zscore_in_tissue1"
    SAMPLE_DIR = OUT / "samples"
    TRAIN_SAMPLES = [x.strip() for x in args.train_samples.split(",") if x.strip()]
    TEST_SAMPLES = [x.strip() for x in args.test_samples.split(",") if x.strip()]
    requested_samples = list(dict.fromkeys(TRAIN_SAMPLES + TEST_SAMPLES))
    unknown = sorted(set(requested_samples) - set(SAMPLES))
    if unknown:
        raise ValueError(f"Unknown samples: {unknown}")

    ensure_dirs()
    offsets = {sample: i * 20000.0 for i, sample in enumerate(SAMPLES)}
    loaded = {sample: load_sample(sample, offsets[sample]) for sample in requested_samples}

    ref_sample = requested_samples[0]
    gene_names = loaded[ref_sample]["genes"]
    gene_symbols = loaded[ref_sample]["gene_symbols"]
    protein_names = loaded[ref_sample]["proteins"]
    protein_display = loaded[ref_sample]["protein_names"]

    train = [loaded[s] for s in TRAIN_SAMPLES]
    test = [loaded[s] for s in TEST_SAMPLES]
    x_train_counts = stack_samples(train, "gene_counts")
    x_test_counts = stack_samples(test, "gene_counts")
    y_train_counts = stack_samples(train, "protein_counts")
    y_test_counts = stack_samples(test, "protein_counts")

    train_barcodes = sum([s["barcodes"] for s in train], [])
    test_barcodes = sum([s["barcodes"] for s in test], [])
    train_coords = pd.concat([s["coords"] for s in train], ignore_index=True)
    test_coords = pd.concat([s["coords"] for s in test], ignore_index=True)

    gene_nnz = np.asarray((x_train_counts > 0).sum(axis=0)).ravel()
    keep_min = gene_nnz >= MIN_CELLS
    x_train_min = x_train_counts[:, keep_min].astype(np.float32).tocsr()
    x_test_min = x_test_counts[:, keep_min].astype(np.float32).tocsr()
    kept_genes = np.asarray(gene_names, dtype=object)[keep_min]
    kept_symbols = np.asarray(gene_symbols, dtype=object)[keep_min]

    selected_idx, forced_idx, marker_status = select_train_hvg_with_forced_markers(
        x_train_min,
        kept_genes,
        kept_symbols,
        protein_display,
    )
    if args.gene_list is not None:
        fixed = [g for g in args.gene_list.read_text(encoding="utf-8").split() if g]
        pos = {str(g): i for i, g in enumerate(kept_genes)}
        missing = [g for g in fixed if g not in pos]
        if missing:
            raise ValueError(f"{len(missing)} listed genes absent after the min-cells filter, first={missing[:5]}")
        selected_idx = [pos[g] for g in fixed]

    x_train_log = x_train_min.copy()
    x_test_log = x_test_min.copy()
    x_train_log.data = np.log1p(x_train_log.data)
    x_test_log.data = np.log1p(x_test_log.data)
    train_dense = x_train_log.toarray().astype(np.float32)
    x_train_sel = train_dense[:, selected_idx]
    x_test_sel = x_test_log[:, selected_idx].toarray().astype(np.float32)
    x_train_z, x_test_z, x_mean, x_std = zscore_by_train(x_train_sel, x_test_sel)

    y_train_log = y_train_counts.toarray().astype(np.float32)
    y_test_log = y_test_counts.toarray().astype(np.float32)
    y_train_log = np.log1p(y_train_log)
    y_test_log = np.log1p(y_test_log)
    y_train_z, y_test_z, y_mean, y_std = zscore_by_train(y_train_log, y_test_log)

    np.save(GENE_DIR / "X_train_log1p_zscore.npy", x_train_z)
    np.save(GENE_DIR / "X_test_log1p_zscore.npy", x_test_z)
    np.save(GENE_DIR / "X_train_log1p.npy", x_train_sel.astype(np.float32))
    np.save(GENE_DIR / "X_test_log1p.npy", x_test_sel.astype(np.float32))
    np.save(GENE_DIR / "zscore_mean.npy", x_mean)
    np.save(GENE_DIR / "zscore_std.npy", x_std)
    (GENE_DIR / "genes_selected_3500.txt").write_text("\n".join(kept_genes[selected_idx].astype(str)) + "\n", encoding="utf-8")
    (GENE_DIR / "genes_selected_symbols_3500.txt").write_text("\n".join(kept_symbols[selected_idx].astype(str)) + "\n", encoding="utf-8")
    (GENE_DIR / "genes_forced_marker.txt").write_text("\n".join(kept_symbols[forced_idx].astype(str)) + "\n", encoding="utf-8")
    marker_status.to_csv(GENE_DIR / "marker_keep_status.csv", index=False)
    (GENE_DIR / "train_barcodes.txt").write_text("\n".join(train_barcodes) + "\n", encoding="utf-8")
    (GENE_DIR / "test_barcodes.txt").write_text("\n".join(test_barcodes) + "\n", encoding="utf-8")
    train_coords.to_csv(GENE_DIR / "train_tissue_positions_with_hires_coords.csv", index=False)
    test_coords.to_csv(GENE_DIR / "test_tissue_positions_with_hires_coords.csv", index=False)

    np.save(PRO_DIR / "Y_train_log1p_zscore.npy", y_train_z)
    np.save(PRO_DIR / "Y_test_log1p_zscore.npy", y_test_z)
    np.save(PRO_DIR / "Y_train_log1p.npy", y_train_log.astype(np.float32))
    np.save(PRO_DIR / "Y_test_log1p.npy", y_test_log.astype(np.float32))
    np.save(PRO_DIR / "zscore_mean.npy", y_mean)
    np.save(PRO_DIR / "zscore_std.npy", y_std)
    (PRO_DIR / "protein_names.txt").write_text("\n".join(protein_names) + "\n", encoding="utf-8")
    (PRO_DIR / "protein_display_names.txt").write_text("\n".join(protein_display) + "\n", encoding="utf-8")
    (PRO_DIR / "train_barcodes.txt").write_text("\n".join(train_barcodes) + "\n", encoding="utf-8")
    (PRO_DIR / "test_barcodes.txt").write_text("\n".join(test_barcodes) + "\n", encoding="utf-8")

    summary = {
        "raw_dir": str(RAW),
        "train_samples": TRAIN_SAMPLES,
        "test_samples": TEST_SAMPLES,
        "control_antibodies_excluded": sorted(CONTROL_ANTIBODIES),
        "train_spots": len(train_barcodes),
        "test_spots": len(test_barcodes),
        "common_genes_before_filter": len(gene_names),
        "genes_after_min_cells_filter": int(keep_min.sum()),
        "hvg_method": "scanpy.pp.highly_variable_genes(flavor='seurat') on normalize_total+log1p train data",
        "model_gene_matrix": "subset raw counts by train HVG/forced markers, then log1p(raw) and train-fit gene-wise z-score",
        "target_sum_for_hvg": TARGET_SUM,
        "forced_gene_candidates_n": int(marker_status.shape[0]),
        "forced_genes_kept_n": int(marker_status["after_min_cells"].sum()),
        "forced_genes_in_final_n": int(marker_status["in_final_gene_set"].sum()),
        "final_gene_n": int(len(selected_idx)),
        "protein_n": int(len(protein_names)),
        "protein_names": protein_names,
        "protein_display_names": protein_display,
        "train_shape": list(x_train_z.shape),
        "test_shape": list(x_test_z.shape),
    }
    (OUT / "preprocess_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
