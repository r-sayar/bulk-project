#!/usr/bin/env python3
"""
Deconvolution of Bulk RNA-seq Using Deep Learning
==================================================

Interactive script for bulk RNA-seq deconvolution with selectable:
  - Data source (GEO download, local pancreatic islet data, synthetic,
    or any GEO accession by GSM/GSE ID)
  - Deconvolution method (NNLS, NMF, Neural W-CLS v3, or all three)

Reference-based deconvolution solves: b ≈ S·p  (signature × proportions)
Reference-free deconvolution factorizes: B^T ≈ W·H  (NMF)

Usage:
    python deconvolution_of_bulk_rna_seq_using_deep_learning.py

Original Colab notebook:
    https://colab.research.google.com/drive/1V1BXQlxys63JcG4VXos3VsgNbfZDC9Lj
"""

import os
import sys
import gzip
import math
import tarfile
import textwrap
import urllib.request
import shutil

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.io as sio
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD, NMF, PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from scipy.optimize import nnls, linear_sum_assignment

np.random.seed(3407)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
REPO_ROOT = os.path.abspath(os.path.join(SRC_ROOT, ".."))


# ═══════════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════════

def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def pearson_corr_flat(a, b):
    a, b = a.ravel(), b.ravel()
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def mae(a, b):
    return float(np.mean(np.abs(a - b)))


def concordance_corr_coef(y_true, y_pred):
    """Lin's concordance correlation coefficient (agreement, not just correlation)."""
    y_true, y_pred = y_true.ravel(), y_pred.ravel()
    mean_t, mean_p = y_true.mean(), y_pred.mean()
    var_t, var_p = y_true.var(), y_pred.var()
    cov = np.mean((y_true - mean_t) * (y_pred - mean_p))
    denom = var_t + var_p + (mean_t - mean_p) ** 2
    return float(2 * cov / denom) if denom > 0 else 0.0


def compute_all_metrics(P_pred, P_true, K, cell_type_names=None):
    """Compute global and per-cell-type evaluation metrics.

    Returns dict with keys: RMSE, MAE, Pearson, CCC (global)
    and per_celltype list of dicts.
    """
    from scipy.stats import spearmanr

    global_metrics = {
        "RMSE": rmse(P_pred, P_true),
        "MAE": mae(P_pred, P_true),
        "Pearson": pearson_corr_flat(P_pred, P_true),
        "CCC": concordance_corr_coef(P_pred, P_true),
    }
    sr, _ = spearmanr(P_pred.ravel(), P_true.ravel())
    global_metrics["Spearman"] = float(sr) if not np.isnan(sr) else 0.0

    per_ct = []
    for k in range(K):
        name = cell_type_names[k] if cell_type_names is not None else f"C{k}"
        pred_k, true_k = P_pred[:, k], P_true[:, k]
        per_ct.append({
            "cell_type": name,
            "RMSE": rmse(pred_k, true_k),
            "MAE": mae(pred_k, true_k),
            "Pearson": pearson_corr_flat(pred_k, true_k),
            "CCC": concordance_corr_coef(pred_k, true_k),
            "bias": float(np.mean(pred_k - true_k)),
        })
    global_metrics["per_celltype"] = per_ct
    return global_metrics


def _download(url, out_path):
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        print(f"  Found existing: {out_path}")
        return
    print(f"  Downloading: {out_path} ...")
    urllib.request.urlretrieve(url, out_path)
    print("    done.")


def read_tsv_gz(path):
    with gzip.open(path, "rt") as f:
        return pd.read_csv(f, sep="\t", header=None)


# ═══════════════════════════════════════════════════════════════════════
# DATA LOADERS
# ═══════════════════════════════════════════════════════════════════════

def load_gsm8696075(data_dir=None):
    """Download and load 10x Matrix Market data from GEO (GSM8696075)."""
    if data_dir is None:
        data_dir = os.path.join(REPO_ROOT, "data", "gsm8696075_10x")
    os.makedirs(data_dir, exist_ok=True)

    urls = {
        "matrix.mtx.gz":   "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSM8696075&format=file&file=GSM8696075%5Fmatrix%2Emtx%2Egz",
        "features.tsv.gz": "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSM8696075&format=file&file=GSM8696075%5Ffeatures%2Etsv%2Egz",
        "barcodes.tsv.gz": "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSM8696075&format=file&file=GSM8696075%5Fbarcodes%2Etsv%2Egz",
    }

    paths = {}
    for fname, url in urls.items():
        p = os.path.join(data_dir, fname)
        _download(url, p)
        paths[fname] = p

    with gzip.open(paths["matrix.mtx.gz"], "rb") as f:
        X = sio.mmread(f).tocsr()

    feat = read_tsv_gz(paths["features.tsv.gz"])
    gene_names = feat.iloc[:, 1 if feat.shape[1] >= 2 else 0].astype(str).values

    bc = read_tsv_gz(paths["barcodes.tsv.gz"])
    barcodes = bc.iloc[:, 0].astype(str).values

    print(f"  Loaded GSM8696075: {X.shape[0]} genes × {X.shape[1]} cells")
    return X, gene_names, barcodes


def load_gse84133(donor="human1"):
    """
    Load pancreatic islet scRNA-seq from GSE84133 (CSV format).
    Available donors: human1, human2, human3, human4, mouse1, mouse2
    """
    raw_dir = os.path.join(REPO_ROOT, "data", "GSE84133_RAW")
    pattern = f"GSM*_{donor}_umifm_counts.csv"
    candidates = [f for f in os.listdir(raw_dir) if donor in f and f.endswith(".csv")]

    if not candidates:
        gz_candidates = [f for f in os.listdir(raw_dir) if donor in f and f.endswith(".csv.gz")]
        if not gz_candidates:
            raise FileNotFoundError(
                f"No file for donor '{donor}' in {raw_dir}.\n"
                f"Available: {os.listdir(raw_dir)}"
            )
        gz_path = os.path.join(raw_dir, gz_candidates[0])
        csv_path = gz_path.rstrip(".gz")
        print(f"  Decompressing {gz_candidates[0]} ...")
        with gzip.open(gz_path, "rb") as fin, open(csv_path, "wb") as fout:
            shutil.copyfileobj(fin, fout)
        candidates = [os.path.basename(csv_path)]

    csv_path = os.path.join(raw_dir, candidates[0])
    print(f"  Reading {candidates[0]} ...")
    df = pd.read_csv(csv_path, index_col=0)

    # GSE84133 CSVs: rows = cells, columns include gene columns + metadata
    # The 'assigned_cluster' column has cell type labels
    meta_cols = [c for c in df.columns if c in (
        "assigned_cluster", "barcode", "cell_id", "well", "plate",
    )]
    gene_cols = [c for c in df.columns if c not in meta_cols]

    cell_labels = df["assigned_cluster"].values if "assigned_cluster" in df.columns else None

    expr = df[gene_cols].values.T.astype(np.float32)  # genes × cells
    X = sp.csr_matrix(expr)
    gene_names = np.array(gene_cols)
    barcodes = np.array(df.index.astype(str))

    print(f"  Loaded GSE84133 ({donor}): {X.shape[0]} genes × {X.shape[1]} cells")
    if cell_labels is not None:
        unique_types = np.unique(cell_labels)
        print(f"  Cell types available: {', '.join(unique_types)}")

    return X, gene_names, barcodes, cell_labels


def build_synthetic_reference(n_genes=2000, K=10, n_cells_per_type=200, seed=42):
    """Generate synthetic scRNA-seq-like reference with K cell types."""
    rng = np.random.default_rng(seed)
    S = rng.exponential(0.5, size=(n_genes, K)).astype(np.float32)

    markers_per_type = n_genes // K
    for k in range(K):
        start = k * markers_per_type
        end = start + markers_per_type // 4
        S[start:end, k] += rng.exponential(3.0, size=(end - start,))

    n_cells = n_cells_per_type * K
    labels = np.repeat(np.arange(K), n_cells_per_type)
    X = np.zeros((n_genes, n_cells), dtype=np.float32)
    for i in range(n_cells):
        rate = S[:, labels[i]] * rng.uniform(0.8, 1.2)
        X[:, i] = rng.poisson(np.maximum(rate, 0.01))

    lib_sizes = X.sum(axis=0, keepdims=True)
    lib_sizes[lib_sizes == 0] = 1
    X_norm = (X / lib_sizes * 1e4).astype(np.float32)

    gene_names = np.array([f"Gene_{i}" for i in range(n_genes)])
    barcodes = np.array([f"Cell_{i}" for i in range(n_cells)])

    print(f"  Built synthetic reference: {n_genes} genes × {n_cells} cells, K={K} types")
    return sp.csr_matrix(X_norm), gene_names, barcodes, labels


# ═══════════════════════════════════════════════════════════════════════
# GEO ACCESSION DOWNLOAD
# ═══════════════════════════════════════════════════════════════════════

def load_geo_accession(accession):
    """
    Download and load scRNA-seq data from GEO by accession number.
    Supports GSM (sample) and GSE (series) accessions.
    Auto-detects 10x Matrix Market or tabular count matrix formats.

    Returns (X_sparse, gene_names, barcodes, cell_labels_or_None).
    """
    accession = accession.strip()
    acc_upper = accession.upper()
    if not (acc_upper.startswith("GSM") or acc_upper.startswith("GSE")):
        raise ValueError(f"Accession must start with GSM or GSE, got: {accession}")

    data_dir = os.path.join(REPO_ROOT, "data", accession)
    os.makedirs(data_dir, exist_ok=True)

    _download_geo_supplementary(accession, data_dir)
    return _detect_and_load_geo(data_dir, accession)


def _download_geo_supplementary(accession, data_dir):
    """Download supplementary files from GEO and extract if archived."""
    existing = [f for f in os.listdir(data_dir)
                if not f.startswith('.') and f != '.DS_Store']
    if existing:
        print(f"  Using cached data ({len(existing)} files in {accession}/)")
        return

    url = f"https://www.ncbi.nlm.nih.gov/geo/download/?acc={accession}&format=file"
    print(f"  Downloading {accession} from GEO ...")

    req = urllib.request.Request(url, headers={"User-Agent": "Python/bulk-deconv"})
    with urllib.request.urlopen(req) as response:
        content_disp = response.headers.get("Content-Disposition", "")
        if "filename=" in content_disp:
            fname = content_disp.split("filename=")[-1].strip("\"' ")
        else:
            fname = f"{accession}_RAW.tar"

        out_path = os.path.join(data_dir, fname)
        with open(out_path, "wb") as f:
            shutil.copyfileobj(response, f)

    size_mb = os.path.getsize(out_path) / 1e6
    print(f"  Downloaded: {fname} ({size_mb:.1f} MB)")

    try:
        if tarfile.is_tarfile(out_path):
            print(f"  Extracting archive ...")
            with tarfile.open(out_path) as tf:
                try:
                    tf.extractall(data_dir, filter="data")
                except TypeError:
                    tf.extractall(data_dir)
            n_new = len([f for f in os.listdir(data_dir) if f != fname])
            print(f"  Extracted {n_new} file(s)")
    except Exception as exc:
        print(f"  Note: not a tar archive ({exc})")


def _detect_and_load_geo(data_dir, accession):
    """Walk *data_dir*, detect the file format, and load expression data."""
    all_files = []
    for root, _dirs, files in os.walk(data_dir):
        for f in files:
            if f.startswith(".") or f.endswith("_RAW.tar"):
                continue
            all_files.append(os.path.relpath(os.path.join(root, f), data_dir))

    if not all_files:
        raise FileNotFoundError(f"No data files found in {data_dir}")

    print(f"\n  Files in {accession}/:")
    for f in sorted(all_files):
        size = os.path.getsize(os.path.join(data_dir, f))
        unit, val = ("MB", size / 1e6) if size > 1e6 else ("KB", size / 1e3)
        print(f"    {f}  ({val:.1f} {unit})")

    # --- Strategy 1: 10x Matrix Market ---
    mtx = [f for f in all_files if f.endswith((".mtx.gz", ".mtx"))]
    feat = [f for f in all_files
            if ("features" in f.lower() or "genes" in f.lower())
            and f.endswith((".tsv.gz", ".tsv"))]
    bcs = [f for f in all_files
           if "barcodes" in f.lower() and f.endswith((".tsv.gz", ".tsv"))]

    if mtx and feat and bcs:
        return _load_10x_from_dir(data_dir, mtx[0], feat[0], bcs[0], accession)

    # --- Strategy 2: tabular count matrix ---
    tabular = [f for f in all_files
               if f.endswith((".csv", ".csv.gz", ".tsv", ".tsv.gz",
                              ".txt", ".txt.gz"))]
    if tabular:
        if len(tabular) > 1:
            print("\n  Multiple tabular files found — select one:")
            for i, t in enumerate(tabular, 1):
                print(f"    [{i}] {t}")
            while True:
                pick = input(f"  Enter choice [1-{len(tabular)}]: ").strip()
                if pick.isdigit() and 1 <= int(pick) <= len(tabular):
                    chosen = tabular[int(pick) - 1]
                    break
                print("  Invalid choice.")
        else:
            chosen = tabular[0]
        return _load_tabular_geo(os.path.join(data_dir, chosen), accession)

    raise FileNotFoundError(
        f"No supported format detected in {data_dir}.\n"
        f"Files: {all_files}\n"
        "Supported: 10x Matrix Market (.mtx.gz + features + barcodes), "
        "tabular count matrices (.csv/.tsv/.txt, optionally gzipped)"
    )


def _load_10x_from_dir(data_dir, mtx_file, features_file, barcodes_file, accession):
    """Load 10x Genomics Matrix Market files from *data_dir*."""
    print(f"\n  Detected 10x Matrix Market format")

    mtx_path = os.path.join(data_dir, mtx_file)
    feat_path = os.path.join(data_dir, features_file)
    bc_path = os.path.join(data_dir, barcodes_file)

    if mtx_path.endswith(".gz"):
        with gzip.open(mtx_path, "rb") as f:
            X = sio.mmread(f).tocsr()
    else:
        X = sio.mmread(mtx_path).tocsr()

    if feat_path.endswith(".gz"):
        feat = read_tsv_gz(feat_path)
    else:
        feat = pd.read_csv(feat_path, sep="\t", header=None)
    gene_names = feat.iloc[:, 1 if feat.shape[1] >= 2 else 0].astype(str).values

    if bc_path.endswith(".gz"):
        bc = read_tsv_gz(bc_path)
    else:
        bc = pd.read_csv(bc_path, sep="\t", header=None)
    barcodes = bc.iloc[:, 0].astype(str).values

    print(f"  Loaded {accession}: {X.shape[0]} genes × {X.shape[1]} cells")
    return X, gene_names, barcodes, None


def _load_tabular_geo(filepath, accession):
    """Load a CSV / TSV / TXT count matrix downloaded from GEO."""
    basename = os.path.basename(filepath)
    print(f"\n  Loading tabular file: {basename}")

    if filepath.endswith(".gz"):
        with gzip.open(filepath, "rt") as fh:
            first_line = fh.readline()
    else:
        with open(filepath, "r") as fh:
            first_line = fh.readline()

    sep = "\t" if "\t" in first_line else ","
    compression = "gzip" if filepath.endswith(".gz") else None
    df = pd.read_csv(filepath, sep=sep, index_col=0, compression=compression)

    label_candidates = [
        "assigned_cluster", "cell_type", "celltype", "cluster", "label",
    ]
    cell_labels = None
    for candidate in label_candidates:
        matches = [c for c in df.columns if c.lower() == candidate]
        if matches:
            cell_labels = df[matches[0]].values
            break

    meta_names = {
        "assigned_cluster", "cell_type", "celltype", "cluster", "label",
        "cell_id", "barcode", "well", "plate",
    }
    meta_cols = [c for c in df.columns if c.lower() in meta_names]
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c not in meta_cols]

    if df.shape[0] < df.shape[1] and numeric_cols:
        print(f"  Layout: {df.shape[0]} cells × {len(numeric_cols)} genes (transposing)")
        expr = df[numeric_cols].values.T.astype(np.float32)
        gene_names = np.array(numeric_cols)
        barcodes = np.array(df.index.astype(str))
    else:
        cols = numeric_cols if numeric_cols else list(df.columns)
        expr = df[cols].values.astype(np.float32)
        gene_names = np.array(df.index.astype(str))
        barcodes = np.array(cols)
        print(f"  Layout: {expr.shape[0]} genes × {expr.shape[1]} cells/samples")

    X = sp.csr_matrix(expr)
    print(f"  Loaded {accession}: {X.shape[0]} genes × {X.shape[1]} cells")
    return X, gene_names, barcodes, cell_labels


# ═══════════════════════════════════════════════════════════════════════
# PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════

def preprocess(X, gene_names, min_genes_per_cell=200, min_cells_per_gene=3,
               target_sum=1e4, max_cells=20000, seed=42):
    """Filter cells/genes, normalize to CP10K, and create log1p copy."""
    genes_per_cell = np.asarray((X > 0).sum(axis=0)).ravel()
    cells_per_gene = np.asarray((X > 0).sum(axis=1)).ravel()

    cell_mask = genes_per_cell >= min_genes_per_cell
    gene_mask = cells_per_gene >= min_cells_per_gene

    Xf = X[gene_mask][:, cell_mask].tocsr()
    genes_f = gene_names[gene_mask]

    print(f"  After filtering: {Xf.shape[0]} genes × {Xf.shape[1]} cells")

    if Xf.shape[1] > max_cells:
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(Xf.shape[1], size=max_cells, replace=False))
        Xf = Xf[:, keep]
        print(f"  Subsampled to: {Xf.shape[0]} genes × {Xf.shape[1]} cells")

    total = np.asarray(Xf.sum(axis=0)).ravel()
    total[total == 0] = 1.0
    scale = (target_sum / total).astype(np.float32)
    X_norm = Xf.multiply(scale).tocsr().astype(np.float32)

    X_log = X_norm.copy()
    X_log.data = np.log1p(X_log.data)

    return X_norm, X_log, genes_f, cell_mask, gene_mask


def select_hvgs(X_log, n_top=2000):
    """Select top highly-variable genes by variance/mean (dispersion)."""
    mean = np.asarray(X_log.mean(axis=1)).ravel()
    mean_sq = np.asarray(X_log.power(2).mean(axis=1)).ravel()
    var = np.maximum(mean_sq - mean ** 2, 0)
    top_idx = np.argsort(var)[-n_top:]
    top_idx.sort()
    return top_idx, var[top_idx]


# ═══════════════════════════════════════════════════════════════════════
# CLUSTERING & SIGNATURE MATRIX
# ═══════════════════════════════════════════════════════════════════════

def cluster_cells(X_log, K=10, n_hvg=2000, svd_dim=30, ref_frac=0.7,
                  cell_labels=None, seed=42):
    """
    Split cells into reference/bulk-pool, then cluster.
    If cell_labels is provided (e.g. from GSE84133), uses those instead of KMeans.
    Returns ref_cells, bulk_pool_cells, labels_ref, labels_bulkpool, K_actual.
    """
    hvg_idx, _ = select_hvgs(X_log, n_top=n_hvg)
    n_cells = X_log.shape[1]
    all_cells = np.arange(n_cells)

    ref_cells, bulk_pool_cells = train_test_split(
        all_cells, train_size=ref_frac, random_state=seed, shuffle=True
    )
    ref_cells = np.sort(ref_cells)
    bulk_pool_cells = np.sort(bulk_pool_cells)

    print(f"  Reference cells: {len(ref_cells)}, Bulk-pool cells: {len(bulk_pool_cells)}")

    if cell_labels is not None:
        unique_labels = np.unique(cell_labels)
        label_map = {l: i for i, l in enumerate(unique_labels)}
        int_labels = np.array([label_map[l] for l in cell_labels])
        labels_ref = int_labels[ref_cells]
        labels_bulkpool = int_labels[bulk_pool_cells]
        K_actual = len(unique_labels)
        print(f"  Using provided cell-type labels ({K_actual} types): {', '.join(unique_labels)}")
    else:
        Xh_ref = X_log[hvg_idx, :][:, ref_cells].T
        Xh_bulk = X_log[hvg_idx, :][:, bulk_pool_cells].T

        svd = TruncatedSVD(n_components=svd_dim, random_state=seed)
        Z_ref = svd.fit_transform(Xh_ref)
        Z_bulk = svd.transform(Xh_bulk)

        kmeans = KMeans(n_clusters=K, random_state=seed, n_init=20)
        labels_ref = kmeans.fit_predict(Z_ref)
        labels_bulkpool = kmeans.predict(Z_bulk)
        K_actual = K
        print(f"  KMeans clustering into {K} pseudo cell types")

    for k in range(K_actual):
        n_ref = (labels_ref == k).sum()
        n_bulk = (labels_bulkpool == k).sum()
        print(f"    cluster {k}: ref={n_ref}, bulk_pool={n_bulk}")

    return ref_cells, bulk_pool_cells, labels_ref, labels_bulkpool, K_actual


def build_signature_matrix(X_norm, ref_cells, labels_ref, K):
    """S[g, k] = mean expression of gene g in reference cells of cluster k."""
    X_ref = X_norm[:, ref_cells].tocsc()
    S = np.zeros((X_ref.shape[0], K), dtype=np.float32)
    for k in range(K):
        cols = np.where(labels_ref == k)[0]
        if len(cols) == 0:
            raise ValueError(f"No reference cells in cluster {k}.")
        S[:, k] = np.asarray(X_ref[:, cols].mean(axis=1)).ravel()
    return S


# ═══════════════════════════════════════════════════════════════════════
# BULK SIMULATION
# ═══════════════════════════════════════════════════════════════════════

def simulate_bulks(X_norm, bulk_pool_cells, labels_bulkpool, K,
                   n_samples=800, cells_per_bulk=300, dirichlet_alpha=2.0, seed=42):
    """Create synthetic bulk mixtures from single cells with known proportions."""
    rng = np.random.default_rng(seed)
    X_pool = X_norm[:, bulk_pool_cells].tocsc()
    pool_by_k = [np.where(labels_bulkpool == k)[0] for k in range(K)]

    for k in range(K):
        if len(pool_by_k[k]) == 0:
            raise ValueError(f"No bulk-pool cells in cluster {k}.")

    B = np.zeros((X_pool.shape[0], n_samples), dtype=np.float32)
    P = np.zeros((n_samples, K), dtype=np.float32)

    for i in range(n_samples):
        props = rng.dirichlet(np.ones(K) * dirichlet_alpha)
        counts = rng.multinomial(cells_per_bulk, props)
        P[i, :] = counts / counts.sum()

        selected = []
        for k in range(K):
            if counts[k] > 0:
                pick = rng.choice(pool_by_k[k], size=counts[k], replace=True)
                selected.extend(pick.tolist())

        B[:, i] = np.asarray(X_pool[:, selected].sum(axis=1)).ravel() / cells_per_bulk

    print(f"  Simulated {n_samples} bulks ({cells_per_bulk} cells each, α={dirichlet_alpha})")
    return B, P


def select_deconv_genes(B, genes_f, n_genes=2000):
    """Select top-variable genes across bulk samples."""
    gene_var = B.var(axis=1)
    nonzero = gene_var > 0
    idx = np.where(nonzero)[0]
    top = idx[np.argsort(gene_var[nonzero])[-n_genes:]]
    top.sort()
    return top


# ═══════════════════════════════════════════════════════════════════════
# DECONVOLUTION HELPERS
# ═══════════════════════════════════════════════════════════════════════

def _print_metrics(method_name, metrics):
    """Print global and per-cell-type metrics for a deconvolution method."""
    print(f"  RMSE={metrics['RMSE']:.4f}  MAE={metrics['MAE']:.4f}  "
          f"r={metrics['Pearson']:.4f}  rho={metrics['Spearman']:.4f}  "
          f"CCC={metrics['CCC']:.4f}")
    print(f"\n  Per-cell-type breakdown:")
    for ct in metrics["per_celltype"]:
        print(f"    {ct['cell_type']:>20s}  RMSE={ct['RMSE']:.4f}  "
              f"MAE={ct['MAE']:.4f}  r={ct['Pearson']:.4f}  "
              f"CCC={ct['CCC']:.4f}  bias={ct['bias']:+.4f}")


# ═══════════════════════════════════════════════════════════════════════
# DECONVOLUTION: NNLS (Reference-based)
# ═══════════════════════════════════════════════════════════════════════

def run_nnls(S, B_test, P_test, K, cell_type_names=None):
    """Solve min ||Sp - b|| s.t. p >= 0 for each bulk sample, then normalize.

    NNLS is a direct solver (no training phase), so it only needs the test set.
    """
    print("\n── NNLS (Reference-based) ─────────────────────")
    n_samples = B_test.shape[1]
    P_pred = np.zeros((n_samples, K), dtype=np.float32)

    for i in range(n_samples):
        p, _ = nnls(S, B_test[:, i])
        if p.sum() > 0:
            p = p / p.sum()
        P_pred[i, :] = p.astype(np.float32)

    metrics = compute_all_metrics(P_pred, P_test, K, cell_type_names)
    _print_metrics("NNLS", metrics)
    return P_pred, metrics


# ═══════════════════════════════════════════════════════════════════════
# DECONVOLUTION: NMF (Reference-free)
# ═══════════════════════════════════════════════════════════════════════

def _match_components(P_pred, P_true):
    """Match NMF components to true clusters via Hungarian algorithm."""
    K = P_true.shape[1]
    corr = np.zeros((K, K), dtype=np.float32)
    for i in range(K):
        for j in range(K):
            a, b = P_true[:, i], P_pred[:, j]
            if np.std(a) == 0 or np.std(b) == 0:
                corr[i, j] = 0.0
            else:
                corr[i, j] = np.corrcoef(a, b)[0, 1]
    row_ind, col_ind = linear_sum_assignment(1 - corr)
    return P_pred[:, col_ind], corr, col_ind


def run_nmf(B_train, B_test, P_train, P_test, K, cell_type_names=None, seed=42):
    """Non-negative Matrix Factorization on bulk data (reference-free).

    Fits on B_train, learns component mapping from P_train (Hungarian),
    then transforms B_test and evaluates against P_test.
    """
    print("\n── NMF (Reference-free) ───────────────────────")
    model = NMF(n_components=K, init="nndsvda", random_state=seed, max_iter=2000)

    W_train = model.fit_transform(B_train.T)
    W_train = np.maximum(W_train, 0)
    row_sums = W_train.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    P_train_raw = W_train / row_sums

    _, _, mapping = _match_components(P_train_raw, P_train)
    print(f"  Component mapping (learned on train): {mapping}")

    W_test = model.transform(B_test.T)
    W_test = np.maximum(W_test, 0)
    row_sums = W_test.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    P_test_raw = W_test / row_sums
    P_pred = P_test_raw[:, mapping]

    metrics = compute_all_metrics(P_pred, P_test, K, cell_type_names)
    _print_metrics("NMF", metrics)
    return P_pred, metrics


# ═══════════════════════════════════════════════════════════════════════
# DECONVOLUTION: NEURAL W-CLS v3 (Deep Learning)
# ═══════════════════════════════════════════════════════════════════════

def run_neural_wcls(S, B_train, P_train, B_test, P_test, K,
                    cell_type_names=None, epochs=500, patience=40):
    """Train a neural weighted constrained least-squares model.

    Trains on (B_train, P_train) with an internal 85/15 train/val sub-split,
    then evaluates on the shared (B_test, P_test) held-out set.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

    print("\n── Neural W-CLS v3 (Deep Learning) ────────────")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    class WeightNet(nn.Module):
        def __init__(self, n_genes, hidden=64, dropout=0.2):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_genes, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, n_genes),
            )

        def forward(self, x_log):
            logw = self.net(x_log)
            logw = logw - logw.mean(dim=1, keepdim=True)
            logw = torch.clamp(logw, min=-2.0, max=2.0)
            return torch.exp(logw), logw

    class WRidgeSolver(nn.Module):
        def __init__(self, S_tensor, lam_init=1e-2):
            super().__init__()
            self.register_buffer("S", S_tensor)
            self.log_lam = nn.Parameter(torch.tensor(np.log(lam_init), dtype=torch.float32))
            self.log_temp = nn.Parameter(torch.tensor(0.0))

        def forward(self, b, w):
            S_mat = self.S
            n_k = S_mat.shape[1]
            lam = F.softplus(self.log_lam) + 1e-6
            temp = F.softplus(self.log_temp) + 0.1

            Sw = S_mat.unsqueeze(0) * w.unsqueeze(-1)
            A = torch.matmul(Sw.transpose(1, 2), Sw)
            I = torch.eye(n_k, device=b.device, dtype=b.dtype).unsqueeze(0)
            A = A + lam * I
            rhs = torch.matmul(Sw.transpose(1, 2), (w * b).unsqueeze(-1))
            p_raw = torch.linalg.solve(A, rhs).squeeze(-1)
            return F.softmax(p_raw / temp, dim=1), p_raw

    class NeuralWCLS(nn.Module):
        def __init__(self, S_tensor, hidden=64, dropout=0.2, lam_init=1e-2):
            super().__init__()
            n_genes = S_tensor.shape[0]
            self.weightnet = WeightNet(n_genes, hidden, dropout)
            self.solver = WRidgeSolver(S_tensor, lam_init)

        def forward(self, b):
            x_log = torch.log1p(torch.clamp(b, min=0))
            w, logw = self.weightnet(x_log)
            p, p_raw = self.solver(b, w)
            return p, w, logw, p_raw

    def kl_div(y_true, y_pred, eps=1e-8):
        return torch.sum(
            y_true * (torch.log(y_true + eps) - torch.log(y_pred + eps)), dim=1
        ).mean()

    # --- Data prep: sub-split training data into train/val ---
    X_train_all = B_train.T.astype(np.float32)
    Y_train_all = P_train.astype(np.float32)

    idx_all = np.arange(X_train_all.shape[0])
    idx_tr, idx_val = train_test_split(idx_all, test_size=0.15, random_state=42)

    print(f"  Train: {len(idx_tr)}, Val: {len(idx_val)}, "
          f"Test (shared): {P_test.shape[0]}")

    X_tr_t = torch.from_numpy(X_train_all[idx_tr]).to(device)
    Y_tr_t = torch.from_numpy(Y_train_all[idx_tr]).to(device)
    X_val_t = torch.from_numpy(X_train_all[idx_val]).to(device)
    Y_val_t = torch.from_numpy(Y_train_all[idx_val]).to(device)
    X_te_t = torch.from_numpy(B_test.T.astype(np.float32)).to(device)

    loader = DataLoader(TensorDataset(X_tr_t, Y_tr_t), batch_size=64,
                        shuffle=True, drop_last=False)
    S_t = torch.from_numpy(S.astype(np.float32)).to(device)

    # --- Train ---
    model = NeuralWCLS(S_t, hidden=64, dropout=0.2, lam_init=1e-2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    mse_fn = nn.MSELoss()

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            p_pred, w, logw, p_raw = model(xb)

            loss_mse = mse_fn(p_pred, yb)
            loss_kl = kl_div(yb, p_pred)
            loss_entropy = torch.sum(p_pred * torch.log(p_pred + 1e-8), dim=1).mean()
            reg_w = (logw ** 2).mean()

            loss = loss_mse + 0.2 * loss_kl + 0.05 * loss_entropy + 1e-4 * reg_w
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())

        scheduler.step()

        model.eval()
        with torch.no_grad():
            p_val, _, _, _ = model(X_val_t)
            val_loss = mse_fn(p_val, Y_val_t).item()

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if epoch % 50 == 0 or epoch == 1:
            lr_now = scheduler.get_last_lr()[0]
            print(f"  Epoch {epoch:3d} | train={np.mean(losses):.5f} | "
                  f"val={val_loss:.5f} | lr={lr_now:.2e} | "
                  f"patience={patience_counter}/{patience}")

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        print(f"  Restored best model (val MSE = {best_val_loss:.5f})")

    # --- Evaluate on shared test set ---
    model.eval()
    with torch.no_grad():
        P_te, _, _, _ = model(X_te_t)
    P_pred = P_te.cpu().numpy()

    metrics = compute_all_metrics(P_pred, P_test, K, cell_type_names)
    _print_metrics("Neural W-CLS v3", metrics)
    return P_pred, metrics


# ═══════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════

def plot_bars(P_pred, P_true, K, method_name, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3))
    x = np.arange(K)
    true_mean = P_true.mean(axis=0)
    pred_mean = P_pred.mean(axis=0)
    true_std = P_true.std(axis=0)
    pred_std = P_pred.std(axis=0)
    ax.bar(x - 0.2, true_mean, width=0.4, yerr=true_std, capsize=3, label="True")
    ax.bar(x + 0.2, pred_mean, width=0.4, yerr=pred_std, capsize=3, label=method_name)
    ax.set_xticks(x)
    ax.set_xticklabels([f"C{k}" for k in range(K)])
    ax.set_ylim(0, None)
    ax.set_ylabel("Proportion")
    ax.set_title(f"Mean across all samples: {method_name}")
    ax.legend()
    return ax


def plot_comparison_table(all_metrics):
    """Print a comparison table of all methods."""
    w = 72
    print(f"\n╔{'═' * w}╗")
    print(f"║{'METHOD COMPARISON':^{w}}║")
    print(f"╠{'═' * w}╣")
    header = (f"║  {'Method':<18s} {'RMSE':>7s}  {'MAE':>7s}  "
              f"{'r':>7s}  {'rho':>7s}  {'CCC':>7s}")
    print(f"{header:<{w + 1}}║")
    print(f"╠{'─' * w}╣")
    for name, m in all_metrics.items():
        line = (f"║  {name:<18s} {m['RMSE']:7.4f}  {m['MAE']:7.4f}  "
                f"{m['Pearson']:7.4f}  {m['Spearman']:7.4f}  {m['CCC']:7.4f}")
        print(f"{line:<{w + 1}}║")
    print(f"╚{'═' * w}╝")


def plot_all_results(results, P_test, K, cell_type_names=None):
    """Create a grid of scatter + bar plots for all methods."""
    n_methods = len(results)
    fig, axes = plt.subplots(2, n_methods, figsize=(5 * n_methods, 9))
    if n_methods == 1:
        axes = axes.reshape(-1, 1)

    for col, (name, (P_pred, P_true_used)) in enumerate(results.items()):
        ax = axes[0, col]
        ax.scatter(P_true_used.ravel(), P_pred.ravel(), s=10, alpha=0.6)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
        ax.set_xlabel("True proportion")
        ax.set_ylabel("Predicted proportion")
        r = pearson_corr_flat(P_pred, P_true_used)
        ccc = concordance_corr_coef(P_pred, P_true_used)
        ax.set_title(f"{name}\n(r={r:.3f}, CCC={ccc:.3f}, "
                     f"RMSE={rmse(P_pred, P_true_used):.4f})")
        plot_bars(P_pred, P_true_used, K, name, ax=axes[1, col])

    plt.tight_layout()
    plt.show()


# ═══════════════════════════════════════════════════════════════════════
# MENU SYSTEM
# ═══════════════════════════════════════════════════════════════════════

SEPARATOR = "─" * 56


def print_header():
    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║   Bulk RNA-seq Deconvolution Using Deep Learning    ║")
    print("║                                                      ║")
    print("║   Reference-based (NNLS) · Reference-free (NMF)     ║")
    print("║   Neural Weighted CLS v3 (Deep Learning)            ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()


def menu_data():
    gse84133_exists = os.path.isdir(os.path.join(REPO_ROOT, "data", "GSE84133_RAW"))

    print(SEPARATOR)
    print("  SELECT DATA SOURCE")
    print(SEPARATOR)
    print()
    print("  [1]  GSM8696075 — 10x scRNA-seq from GEO")
    print("       (downloads ~30 MB if not cached)")
    print()
    if gse84133_exists:
        print("  [2]  GSE84133 — Pancreatic islet scRNA-seq (local)")
        print("       Donors: human1, human2, human3, human4, mouse1, mouse2")
    else:
        print("  [2]  GSE84133 — (not found in data/GSE84133_RAW/)")
    print()
    print("  [3]  Synthetic — Generated reference (fast, no download)")
    print("       2000 genes, 5 cell types, 1000 cells")
    print()
    print("  [4]  GEO Accession — Download any dataset by ID")
    print("       (e.g., GSM4041647, GSE136148)")
    print()

    while True:
        choice = input("  Enter choice [1/2/3/4]: ").strip()
        if choice in ("1", "2", "3", "4"):
            break
        print("  Invalid choice, try again.")

    donor = None
    geo_accession = None

    if choice == "2":
        if not gse84133_exists:
            print("  GSE84133 data not found. Falling back to synthetic.")
            choice = "3"
        else:
            donor = input("  Donor [human1/human2/human3/human4/mouse1/mouse2] (default: human1): ").strip()
            if not donor:
                donor = "human1"

    elif choice == "4":
        geo_accession = input("  Enter GEO accession (e.g., GSM4041647 or GSE136148): ").strip()
        if not geo_accession:
            print("  No accession entered. Falling back to synthetic.")
            choice = "3"

    return choice, donor, geo_accession


def menu_model():
    print()
    print(SEPARATOR)
    print("  SELECT DECONVOLUTION METHOD(S)")
    print(SEPARATOR)
    print()
    print("  [1]  NNLS — Non-negative least squares (reference-based)")
    print("  [2]  NMF  — Non-negative matrix factorization (reference-free)")
    print("  [3]  Neural W-CLS v3 — Deep learning (requires PyTorch)")
    print("  [4]  All three methods (compare)")
    print()

    while True:
        choice = input("  Enter choice [1/2/3/4]: ").strip()
        if choice in ("1", "2", "3", "4"):
            break
        print("  Invalid choice, try again.")

    return choice


def menu_params():
    """Optionally let the user tweak simulation parameters."""
    print()
    print(SEPARATOR)
    print("  SIMULATION PARAMETERS (press Enter for defaults)")
    print(SEPARATOR)

    def ask_int(prompt, default):
        val = input(f"  {prompt} [{default}]: ").strip()
        return int(val) if val else default

    def ask_float(prompt, default):
        val = input(f"  {prompt} [{default}]: ").strip()
        return float(val) if val else default

    K = ask_int("Number of cell types (K)", 10)
    n_bulks = ask_int("Number of synthetic bulk samples", 800)
    cells_per_bulk = ask_int("Cells per bulk sample", 300)
    dirichlet_alpha = ask_float("Dirichlet alpha (higher=more uniform)", 2.0)
    n_deconv_genes = ask_int("Genes for deconvolution", 2000)

    return {
        "K": K,
        "n_bulks": n_bulks,
        "cells_per_bulk": cells_per_bulk,
        "dirichlet_alpha": dirichlet_alpha,
        "n_deconv_genes": n_deconv_genes,
    }


# ═══════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════

def main():
    print_header()

    # ── Menu ──
    data_choice, donor, geo_accession = menu_data()
    model_choice = menu_model()
    params = menu_params()

    K = params["K"]
    run_models = {
        "1": ["nnls"],
        "2": ["nmf"],
        "3": ["neural"],
        "4": ["nnls", "nmf", "neural"],
    }[model_choice]

    # ── Load data ──
    print(f"\n{'═' * 56}")
    print("  LOADING DATA")
    print(f"{'═' * 56}")

    cell_labels = None

    if data_choice == "1":
        X_counts, gene_names, barcodes = load_gsm8696075()

    elif data_choice == "2":
        X_counts, gene_names, barcodes, cell_labels = load_gse84133(donor=donor)
        if cell_labels is not None:
            K = len(np.unique(cell_labels))
            print(f"  Overriding K={params['K']} → {K} (from cell-type labels)")

    elif data_choice == "3":
        X_counts, gene_names, barcodes, cell_labels = build_synthetic_reference(
            n_genes=2000, K=K, n_cells_per_type=200, seed=42
        )

    elif data_choice == "4":
        X_counts, gene_names, barcodes, cell_labels = load_geo_accession(geo_accession)
        if cell_labels is not None:
            K = len(np.unique(cell_labels))
            print(f"  Detected K={K} cell types from labels")

    # ── Preprocess ──
    print(f"\n{'═' * 56}")
    print("  PREPROCESSING")
    print(f"{'═' * 56}")

    X_norm, X_log, genes_f, _, _ = preprocess(X_counts, gene_names)

    # If cell_labels came from the loader, filter to match surviving cells
    if cell_labels is not None and len(cell_labels) > X_norm.shape[1]:
        genes_per_cell = np.asarray((X_counts > 0).sum(axis=0)).ravel()
        cell_mask = genes_per_cell >= 200
        cell_labels = cell_labels[cell_mask]
        if len(cell_labels) > X_norm.shape[1]:
            cell_labels = cell_labels[:X_norm.shape[1]]

    # ── Cluster & build signature ──
    print(f"\n{'═' * 56}")
    print("  CLUSTERING & SIGNATURE MATRIX")
    print(f"{'═' * 56}")

    ref_cells, bulk_pool_cells, labels_ref, labels_bulkpool, K = cluster_cells(
        X_log, K=K, cell_labels=cell_labels
    )

    S_full = build_signature_matrix(X_norm, ref_cells, labels_ref, K)
    print(f"  Signature matrix: {S_full.shape[0]} genes × {K} types")

    # ── Simulate bulks ──
    print(f"\n{'═' * 56}")
    print("  SIMULATING BULK MIXTURES")
    print(f"{'═' * 56}")

    B_full, P_true = simulate_bulks(
        X_norm, bulk_pool_cells, labels_bulkpool, K,
        n_samples=params["n_bulks"],
        cells_per_bulk=params["cells_per_bulk"],
        dirichlet_alpha=params["dirichlet_alpha"],
    )

    # ── Gene selection for deconvolution ──
    top_genes = select_deconv_genes(B_full, genes_f, n_genes=params["n_deconv_genes"])
    B = B_full[top_genes, :]
    S = S_full[top_genes, :]
    print(f"  Selected {len(top_genes)} genes for deconvolution")

    # ── Common train/test split (fair comparison across methods) ──
    print(f"\n{'═' * 56}")
    print("  TRAIN / TEST SPLIT")
    print(f"{'═' * 56}")

    idx_all = np.arange(P_true.shape[0])
    idx_train, idx_test = train_test_split(
        idx_all, test_size=0.20, random_state=42
    )
    B_train, B_test = B[:, idx_train], B[:, idx_test]
    P_train, P_test = P_true[idx_train], P_true[idx_test]
    print(f"  Train samples: {len(idx_train)}, Test samples: {len(idx_test)}")
    print(f"  All methods evaluated on the SAME {len(idx_test)} test samples")

    # Resolve cell-type names for reporting
    ct_names = None
    if cell_labels is not None:
        ct_names = list(np.unique(cell_labels))

    # ── Run selected models ──
    print(f"\n{'═' * 56}")
    print("  RUNNING DECONVOLUTION")
    print(f"{'═' * 56}")

    all_metrics = {}
    results = {}

    if "nnls" in run_models:
        P_nnls, m_nnls = run_nnls(S, B_test, P_test, K, ct_names)
        all_metrics["NNLS"] = m_nnls
        results["NNLS"] = (P_nnls, P_test)

    if "nmf" in run_models:
        P_nmf, m_nmf = run_nmf(
            B_train, B_test, P_train, P_test, K, ct_names
        )
        all_metrics["NMF"] = m_nmf
        results["NMF"] = (P_nmf, P_test)

    if "neural" in run_models:
        P_neural, m_neural = run_neural_wcls(
            S, B_train, P_train, B_test, P_test, K, ct_names
        )
        all_metrics["Neural W-CLS v3"] = m_neural
        results["Neural W-CLS v3"] = (P_neural, P_test)

    # ── Results ──
    print(f"\n{'═' * 56}")
    print("  RESULTS")
    print(f"{'═' * 56}")

    plot_comparison_table(all_metrics)
    plot_all_results(results, P_test, K, ct_names)

    # ── Save artifacts ──
    eval_dir = os.path.join(SRC_ROOT, "results", "deconvolution_evaluation")
    os.makedirs(eval_dir, exist_ok=True)

    save_data = {"P_test": P_test, "idx_test": idx_test}
    if ct_names is not None:
        save_data["cell_type_names"] = np.array(ct_names)
    for name, (P_pred, _) in results.items():
        key = name.replace(" ", "_").replace("-", "_")
        save_data[f"P_pred_{key}"] = P_pred
    np.savez(os.path.join(eval_dir, "predictions.npz"), **save_data)

    rows = []
    for name, m in all_metrics.items():
        rows.append({
            "method": name, "RMSE": m["RMSE"], "MAE": m["MAE"],
            "Pearson": m["Pearson"], "Spearman": m["Spearman"], "CCC": m["CCC"],
        })
    pd.DataFrame(rows).to_csv(
        os.path.join(eval_dir, "metrics_summary.csv"), index=False
    )

    ct_rows = []
    for name, m in all_metrics.items():
        for ct in m["per_celltype"]:
            ct_rows.append({"method": name, **ct})
    pd.DataFrame(ct_rows).to_csv(
        os.path.join(eval_dir, "metrics_per_celltype.csv"), index=False
    )

    print(f"\n  Saved artifacts to {eval_dir}/")
    print(f"    predictions.npz, metrics_summary.csv, metrics_per_celltype.csv")
    print("\nDone.")


if __name__ == "__main__":
    main()
