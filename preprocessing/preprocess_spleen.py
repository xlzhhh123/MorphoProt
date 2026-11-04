"""Mouse spleen (SPOTS, GSE198353) RNA, protein and coordinate preprocessing.

Training section: Dataset1_Mouse_Spleen1 (replicate 1); test section: Dataset2_Mouse_Spleen2.
Inputs per section folder: adata_RNA.h5ad, adata_ADT.h5ad (raw counts, same spot order),
tissue_positions_list.csv and scalefactors_json.json.

RNA (fitted on the training section only, applied to the test section):
  genes detected in >= 3 training spots -> highly variable genes on log1p(training counts)
  (scanpy, flavor="seurat", n_top_genes=3500) -> final list = the genes of the measured
  proteins (always kept) followed by the HVGs in decreasing normalized dispersion, up to 3,500
  -> log1p(counts) -> gene-wise z-score with the training mean/std.
Protein: log1p(ADT counts) -> per-protein z-score with the training mean/std.
Coordinates: Visium full-resolution pixel positions x tissue_hires_scalef.

The gene list used in the paper is gene_lists/spleen_genes_3500.txt; pass it with
--gene-list to reproduce the paper inputs exactly. Recomputing the HVGs gives the same
3,500 genes, but genes with tied dispersion can come out in a different column order.

    python preprocessing/preprocess_spleen.py --raw raw/spleen --out data/spleen \
        --gene-list gene_lists/spleen_genes_3500.txt
"""
import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

TRAIN, TEST = "Dataset1_Mouse_Spleen1", "Dataset2_Mouse_Spleen2"
MIN_CELLS = 3
N_GENES = 3500
GENE_DIR = "_scheme2_log1p_zscore_hvg3500_force_marker_in_tissue1"
PRO_DIR = "_protein_scheme_log1p_zscore_in_tissue1"

# measured protein -> encoding gene(s), always kept in the RNA input
PROTEIN_GENES = {
    "CD29": ["Itgb1"], "CD3": ["Cd3d", "Cd3e", "Cd3g"], "F4_80": ["Adgre1"], "CD169_siglec": ["Siglec1"],
    "CD4": ["Cd4"], "CD19": ["Cd19"], "CD105": ["Eng"], "CD8": ["Cd8a", "Cd8b1"], "B220_CD45R": ["Ptprc"],
    "Ly6C": ["Ly6c1", "Ly6c2"], "CD31": ["Pecam1"], "CD68": ["Cd68"], "EpCAM": ["Epcam"], "CD163": ["Cd163"],
    "Ly6G": ["Ly6g"], "CD38": ["Cd38"], "IgM": ["Ighm"], "IgD": ["Ighd"], "MadCAM1": ["Madcam1"],
    "CD20": ["Ms4a1"], "CD11b": ["Itgam"],
}


def zscore(train, test):
    mean, std = train.mean(0), train.std(0)
    std[std == 0] = 1.0
    return (train - mean) / std, (test - mean) / std, mean, std


def first_index(names, genes):
    """Column of each gene; for duplicated gene symbols the first occurrence is used."""
    pos = {}
    for i, g in enumerate(names):
        pos.setdefault(g, i)
    return [pos[g] for g in genes]


def select_genes(tr):
    counts = tr.X if sp.issparse(tr.X) else sp.csr_matrix(tr.X)
    keep = np.asarray((counts > 0).sum(0)).ravel() >= MIN_CELLS
    t = tr[:, keep].copy()
    sc.pp.log1p(t)
    sc.pp.highly_variable_genes(t, n_top_genes=N_GENES, flavor="seurat", subset=False, inplace=True)
    forced = sorted({g for genes in PROTEIN_GENES.values() for g in genes if g in set(t.var_names)})
    hv = t.var[t.var["highly_variable"]]
    hv = hv[~hv.index.duplicated()].sort_values("dispersions_norm", ascending=False, kind="stable")
    selected = list(forced)
    for g in hv.index:
        if len(selected) >= N_GENES:
            break
        if g not in selected:
            selected.append(g)
    return selected, forced, int(keep.sum())


def coords(section_dir):
    pos = pd.read_csv(section_dir / "tissue_positions_list.csv", header=None,
                      names=["barcode", "in_tissue", "array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres"])
    if not str(pos.iloc[0, 1]).isdigit():  # file with header line
        pos = pd.read_csv(section_dir / "tissue_positions_list.csv")
    s = json.loads((section_dir / "scalefactors_json.json").read_text())["tissue_hires_scalef"]
    pos["pxl_row_in_hires"] = pos["pxl_row_in_fullres"] * s
    pos["pxl_col_in_hires"] = pos["pxl_col_in_fullres"] * s
    pos["pxl_row_in_hires_round"] = pos["pxl_row_in_hires"].round().astype(int)
    pos["pxl_col_in_hires_round"] = pos["pxl_col_in_hires"].round().astype(int)
    pos["tissue_hires_scalef_used"] = s
    return pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=Path("raw/spleen"))
    ap.add_argument("--out", type=Path, default=Path("data/spleen"))
    ap.add_argument("--gene-list", type=Path, default=None, help="fixed gene list (gene_lists/spleen_genes_3500.txt)")
    args = ap.parse_args()

    rna = {s: ad.read_h5ad(args.raw / s / "adata_RNA.h5ad") for s in (TRAIN, TEST)}
    adt = {s: ad.read_h5ad(args.raw / s / "adata_ADT.h5ad") for s in (TRAIN, TEST)}
    for s in (TRAIN, TEST):
        assert (rna[s].obs_names == adt[s].obs_names).all(), f"{s}: RNA/ADT spot order differs"
        pos = coords(args.raw / s)
        (args.out / s).mkdir(parents=True, exist_ok=True)
        pos.to_csv(args.out / s / "tissue_positions_list_with_hires_coords.csv", index=False)
        in_tissue = set(pos.loc[pos.in_tissue == 1, "barcode"])
        keep = rna[s].obs_names.isin(in_tissue)
        rna[s], adt[s] = rna[s][keep].copy(), adt[s][keep].copy()

    if args.gene_list:
        genes = [g for g in args.gene_list.read_text().split() if g]
        forced = sorted({g for v in PROTEIN_GENES.values() for g in v})
        n_after_filter = None
    else:
        genes, forced, n_after_filter = select_genes(rna[TRAIN])

    gdir, pdir = args.out / GENE_DIR, args.out / PRO_DIR
    gdir.mkdir(parents=True, exist_ok=True)
    pdir.mkdir(parents=True, exist_ok=True)

    idx = first_index(list(rna[TRAIN].var_names), genes)
    xtr = np.log1p(np.ascontiguousarray(rna[TRAIN].X.tocsc()[:, idx].toarray(), dtype=np.float32))
    xte = np.log1p(np.ascontiguousarray(rna[TEST].X.tocsc()[:, idx].toarray(), dtype=np.float32))
    ztr, zte, mean, std = zscore(xtr, xte)
    for name, arr in [("X_train_log1p", xtr), ("X_test_log1p", xte), ("X_train_log1p_zscore", ztr),
                      ("X_test_log1p_zscore", zte), ("zscore_mean", mean), ("zscore_std", std)]:
        np.save(gdir / f"{name}.npy", arr.astype(np.float32))
    (gdir / "genes_selected_3500.txt").write_text("\n".join(genes) + "\n")
    (gdir / "genes_forced_marker.txt").write_text("\n".join(forced) + "\n")

    proteins = list(adt[TRAIN].var_names)
    assert proteins == list(adt[TEST].var_names)
    ytr = np.log1p(np.asarray(adt[TRAIN].X.toarray() if sp.issparse(adt[TRAIN].X) else adt[TRAIN].X, dtype=np.float32))
    yte = np.log1p(np.asarray(adt[TEST].X.toarray() if sp.issparse(adt[TEST].X) else adt[TEST].X, dtype=np.float32))
    ztr, zte, mean, std = zscore(ytr, yte)
    for name, arr in [("Y_train_log1p", ytr), ("Y_test_log1p", yte), ("Y_train_log1p_zscore", ztr),
                      ("Y_test_log1p_zscore", zte), ("zscore_mean", mean), ("zscore_std", std)]:
        np.save(pdir / f"{name}.npy", arr.astype(np.float32))
    (pdir / "protein_names.txt").write_text("\n".join(proteins) + "\n")

    for d in (gdir, pdir):
        (d / "train_barcodes.txt").write_text("\n".join(rna[TRAIN].obs_names) + "\n")
        (d / "test_barcodes.txt").write_text("\n".join(rna[TEST].obs_names) + "\n")
    summary = dict(min_cells=MIN_CELLS, n_genes=len(genes), genes_after_min_cells=n_after_filter,
                   n_forced=len(forced), train_spots=int(rna[TRAIN].n_obs), test_spots=int(rna[TEST].n_obs),
                   n_proteins=len(proteins))
    (gdir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
