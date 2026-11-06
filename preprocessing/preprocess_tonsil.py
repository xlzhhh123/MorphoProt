"""Tonsil RNA and protein preprocessing (training section: female "tonsilwoman";
test section: male "tonsilman").

RNA (fitted on the training section only, applied to the test section):
  genes detected in >= 3 training spots -> normalize_total(1e4) + log1p ->
  scanpy highly_variable_genes(flavor="seurat", n_top_genes=3500) -> the selected
  genes are taken from the raw counts -> log1p -> gene-wise z-score with the
  training mean/std.
Protein: filtered ADT counts -> log1p -> per-protein z-score fitted on the
training section and applied to both sections. Test protein values do not
contribute to the normalization parameters.

Inputs are the barcode-aligned, in-tissue AnnData files of each section
(<name>_gene_filter.h5ad and <name>_pro_filter.h5ad: spots shared by the RNA and
antibody matrices of the 10x Genomics "CytAssist FFPE Human Tonsil (Protein
Expression)" outputs). The expected training gene
list is gene_lists/tonsil_genes_3500.txt.

    python preprocessing/preprocess_tonsil.py --raw raw/tonsil --out data/tonsil \
        --gene-list gene_lists/tonsil_genes_3500.txt
"""
import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc
import scipy.sparse as sp

MIN_CELLS = 3
N_HVG = 3500
TARGET_SUM = 1e4


def to_dense(x):
    return x.toarray() if sp.issparse(x) else np.asarray(x)


def zfit(x):
    mean, std = x.mean(0), x.std(0)
    std[std == 0] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def zapply(x, mean, std):
    return np.nan_to_num((x - mean) / std, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def save_like(template, x, path, note):
    out = ad.AnnData(X=x, obs=template.obs.copy(), var=template.var.copy(), uns=template.uns.copy(),
                     obsm=template.obsm.copy(), varm=template.varm.copy())
    out.uns["preprocess_method"] = note
    path.parent.mkdir(parents=True, exist_ok=True)
    out.write(path)
    print("saved", path, out.shape)


def select_hvg(tr):
    n_cells = np.asarray((tr.X > 0).sum(0)).ravel() if sp.issparse(tr.X) else (np.asarray(tr.X) > 0).sum(0)
    hv = tr[:, n_cells >= MIN_CELLS].copy()
    sc.pp.normalize_total(hv, target_sum=TARGET_SUM)
    sc.pp.log1p(hv)
    sc.pp.highly_variable_genes(hv, n_top_genes=N_HVG, flavor="seurat", subset=False, inplace=True)
    return list(map(str, hv.var_names[hv.var["highly_variable"].values]))


def rna(raw, out, gene_list=None, train="tonsilwoman", test="tonsilman"):
    tr = ad.read_h5ad(raw / train / f"{train}_gene_filter.h5ad")
    te = ad.read_h5ad(raw / test / f"{test}_gene_filter.h5ad")
    if gene_list is not None:
        genes = [g for g in Path(gene_list).read_text(encoding="utf-8").split() if g]
        missing = [g for g in genes if g not in set(map(str, tr.var_names))]
        if missing:
            raise ValueError(f"{len(missing)} listed genes absent from the training matrix, first={missing[:5]}")
    else:
        genes = select_hvg(tr)
    (out / "female" / "gene").mkdir(parents=True, exist_ok=True)
    (out / "female" / "gene" / f"{train}_train_hvg{N_HVG}_gene_list_seurat.txt").write_text("\n".join(genes) + "\n")
    tr_h, te_h = tr[:, genes].copy(), te[:, genes].copy()
    xtr = np.log1p(to_dense(tr_h.X).astype(np.float32))
    xte = np.log1p(to_dense(te_h.X).astype(np.float32))
    mean, std = zfit(xtr)
    note = "log1p(raw) -> gene-wise z-score fitted on the training section"
    save_like(tr_h, zapply(xtr, mean, std),
              out / "female" / "gene" / f"{train}_train_hvg{N_HVG}_seurat_gene_zscore_log1p_raw.h5ad", note)
    save_like(te_h, zapply(xte, mean, std),
              out / "male" / "gene" / f"{test}_test_from_{train}_hvg{N_HVG}_seurat_gene_zscore_log1p_raw.h5ad", note)


def protein(raw, out, train="tonsilwoman", test="tonsilman"):
    """log1p -> per-protein z-score using training-section mean/std for both sections."""
    a_tr = ad.read_h5ad(raw / train / f"{train}_pro_filter.h5ad")
    a_te = ad.read_h5ad(raw / test / f"{test}_pro_filter.h5ad")
    assert list(a_tr.var_names) == list(a_te.var_names), "protein panels differ between sections"
    y_tr = np.log1p(to_dense(a_tr.X).astype(np.float32))
    y_te = np.log1p(to_dense(a_te.X).astype(np.float32))
    mean_tr, std_tr = zfit(y_tr)
    note = "filtered protein: log1p -> per-protein z-score fitted on the training section"
    save_like(a_tr, zapply(y_tr, mean_tr, std_tr), out / "female" / "pro" / f"{train}_pro_filter_log1p_zscore.h5ad", note)
    save_like(a_te, zapply(y_te, mean_tr, std_tr), out / "male" / "pro" / f"{test}_pro_filter_log1p_zscore.h5ad", note)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=Path("raw/tonsil"))
    ap.add_argument("--out", type=Path, default=Path("data/tonsil"))
    ap.add_argument("--gene-list", type=Path, default=None,
                    help="fixed 3,500-gene list in model column order (gene_lists/tonsil_genes_3500.txt); "
                         "skips HVG selection and reproduces the paper inputs exactly")
    args = ap.parse_args()
    rna(args.raw, args.out, args.gene_list)
    protein(args.raw, args.out)
