#!/usr/bin/env python3
"""
Batch Integration of Bulk and Single-Cell RNA-seq
==================================================
Integrates bulk RNA-seq (GTEx whole blood) with single-cell-derived pseudobulk
(HCA, Aging PBMC, Tabula Sapiens) using three methods:

  1. Harmony  - PCA-based batch correction (fast, linear)
  2. scANVI   - VAE with semi-supervised cell type labels (deep learning)
  3. Scanorama - Panoramic stitching via mutual nearest neighbours

Each method produces a corrected low-dimensional embedding.  We cluster them
with Leiden, compute integration quality metrics, and compare in a single
multi-panel figure.

Usage
-----
  python batch_integration.py                  # full run
  python batch_integration.py --max-cells 2000 # lighter scANVI run
  python batch_integration.py --skip-scanvi    # skip the slow method
"""

import argparse
import gc
import os
import sys
import time
import warnings

import anndata as ad
import harmonypy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scanorama
import scipy.sparse as sp
from matplotlib.gridspec import GridSpec
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── paths ─────────────────────────────────────────────────────────────
BASEDIR = "/Users/rls/Desktop/programming-projects/single-cell/bulk-project"
GTEX_PATH = "/Users/rls/Downloads/gene_reads_v11_whole_blood.gct.gz"
PSEUDO_PATH = os.path.join(BASEDIR, "pseudobulk/hca_blood_pseudobulk.npz")
AGING_DIR = os.path.join(BASEDIR, "data/downloaded_sc/aging_pbmc")
TABULA_PATH = os.path.join(BASEDIR, "data/downloaded_sc/tabula_sapiens/blood.h5ad")
OUT_DIR = os.path.join(BASEDIR, "src/results/integration")

AGING_FILES = ["b_plasma.h5ad", "cd4_naive.h5ad", "monocyte_dc.h5ad", "nk_ilc.h5ad"]

# ── dark theme (project standard) ────────────────────────────────────
BG = "#0e1117"; CARD = "#1a1d23"; TEXT = "#e6edf3"
MUTED = "#7d8590"; GRID = "#21262d"
C_GTEX = "#f78166"; C_HCA = "#3fb950"; C_AGING = "#58a6ff"; C_TABULA = "#d2a8ff"
SOURCE_COLORS = {"GTEx": C_GTEX, "HCA": C_HCA, "AgingPBMC": C_AGING, "TabulaSapiens": C_TABULA}

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": CARD, "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT, "text.color": TEXT, "xtick.color": MUTED,
    "ytick.color": MUTED, "grid.color": GRID, "grid.alpha": 0.5,
    "font.family": "sans-serif", "font.size": 11,
})


# ═══════════════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ═══════════════════════════════════════════════════════════════════════

def _cpm_log2(expr_raw):
    """CPM normalise then log2(CPM + 1).  Input shape: (genes, samples)."""
    lib = expr_raw.sum(axis=0, keepdims=True)
    lib[lib == 0] = 1
    cpm = expr_raw / lib * 1e6
    return np.log2(cpm + 1)


def load_gtex(path=GTEX_PATH):
    """Load GTEx bulk blood → (samples × genes) float32, gene_names."""
    print("Loading GTEx bulk blood …")
    df = pd.read_csv(path, sep="\t", skiprows=2, compression="gzip")
    expr_raw = df.iloc[:, 2:].values.astype(np.float64)  # genes × samples
    gene_names = df["Description"].values.astype(str)
    expr_log = _cpm_log2(expr_raw)
    print(f"  {expr_log.shape[1]} samples × {expr_log.shape[0]:,} genes")
    return expr_log.T.astype(np.float32), gene_names  # samples × genes


def load_hca(path=PSEUDO_PATH):
    """Load HCA pseudobulk → (donors × genes) float32, gene_names."""
    print("Loading HCA pseudobulk …")
    d = np.load(path, allow_pickle=True)
    expr_raw = d["expr"].astype(np.float64)  # genes × donors
    gene_names = d["gene_names"].astype(str)
    expr_log = _cpm_log2(expr_raw)
    print(f"  {expr_log.shape[1]} donors × {expr_log.shape[0]:,} genes")
    return expr_log.T.astype(np.float32), gene_names


def load_h5ad_pseudobulk(h5ad_path, source_name="h5ad"):
    """Load h5ad, aggregate to pseudobulk by donor_id.

    Uses backed mode to read obs/var metadata, then loads X in chunks
    to avoid loading the full matrix into memory at once.

    Returns (donors × genes) float32, gene_names, donor_id list.
    """
    import h5py

    print(f"Loading {os.path.basename(h5ad_path)} for pseudobulk …")

    # Step 1: read metadata with backed mode (low memory)
    adata_backed = ad.read_h5ad(h5ad_path, backed="r")
    n_cells, n_genes = adata_backed.shape
    print(f"  {n_cells:,} cells × {n_genes:,} genes")

    # gene names
    if "feature_name" in adata_backed.var.columns:
        gene_names = np.array(adata_backed.var["feature_name"].astype(str))
    elif "gene_symbols" in adata_backed.var.columns:
        gene_names = np.array(adata_backed.var["gene_symbols"].astype(str))
    else:
        gene_names = np.array([str(g) for g in adata_backed.var_names])

    donors = np.array(adata_backed.obs["donor_id"].values)
    unique_donors = np.unique(donors)
    print(f"  {len(unique_donors)} unique donors")
    adata_backed.file.close()

    # Step 2: aggregate by donor using chunked reading
    pseudobulk = np.zeros((n_genes, len(unique_donors)), dtype=np.float64)
    donor_to_idx = {d: j for j, d in enumerate(unique_donors)}

    chunk_size = 10000
    with h5py.File(h5ad_path, "r") as f:
        X_h5 = f["X"]
        is_sparse = "encoding-type" in X_h5.attrs

        if is_sparse:
            # CSR format in h5ad
            data = X_h5["data"]
            indices = X_h5["indices"]
            indptr = np.array(X_h5["indptr"])

            for start in range(0, n_cells, chunk_size):
                end = min(start + chunk_size, n_cells)
                ptr_start = indptr[start]
                ptr_end = indptr[end]
                chunk_data = data[ptr_start:ptr_end]
                chunk_indices = indices[ptr_start:ptr_end]
                chunk_indptr = indptr[start:end+1] - indptr[start]
                chunk_sparse = sp.csr_matrix(
                    (chunk_data, chunk_indices, chunk_indptr),
                    shape=(end - start, n_genes),
                )
                chunk_donors = donors[start:end]
                for d in np.unique(chunk_donors):
                    mask = chunk_donors == d
                    j = donor_to_idx[d]
                    pseudobulk[:, j] += np.asarray(
                        chunk_sparse[mask].sum(axis=0)
                    ).ravel()

                if (start // chunk_size) % 10 == 0:
                    print(f"    chunk {start:,}/{n_cells:,}")
        else:
            # Dense matrix
            for start in range(0, n_cells, chunk_size):
                end = min(start + chunk_size, n_cells)
                chunk = X_h5[start:end, :]
                chunk_donors = donors[start:end]
                for d in np.unique(chunk_donors):
                    mask = chunk_donors == d
                    j = donor_to_idx[d]
                    pseudobulk[:, j] += chunk[mask].sum(axis=0)

    gc.collect()

    expr_log = _cpm_log2(pseudobulk)
    print(f"  Pseudobulk: {expr_log.shape[1]} donors × {expr_log.shape[0]:,} genes")
    return expr_log.T.astype(np.float32), gene_names, list(unique_donors)


def load_h5ad_cells(h5ad_path, max_cells=5000, seed=42):
    """Load h5ad, subsample cells.  Returns an AnnData with raw counts."""
    print(f"Loading {os.path.basename(h5ad_path)} for scANVI ({max_cells} cells) …")
    adata = ad.read_h5ad(h5ad_path)

    if max_cells and adata.shape[0] > max_cells:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(adata.shape[0], max_cells, replace=False))
        adata = adata[idx].copy()
        print(f"  Subsampled → {adata.shape[0]:,} cells")

    # ensure gene symbols as var_names
    if "feature_name" in adata.var.columns:
        adata.var_names = adata.var["feature_name"].astype(str)
    elif "gene_symbols" in adata.var.columns:
        adata.var_names = adata.var["gene_symbols"].astype(str)
    adata.var_names_make_unique()

    # keep cell_type
    if "cell_type" not in adata.obs.columns:
        adata.obs["cell_type"] = "Unknown"
    return adata


# ═══════════════════════════════════════════════════════════════════════
# 2. GENE INTERSECTION
# ═══════════════════════════════════════════════════════════════════════

def _build_gene_map(gene_names):
    """Uppercase first-occurrence map: name -> index."""
    m = {}
    for i, n in enumerate(gene_names):
        key = n.upper()
        if key not in m:
            m[key] = i
    return m


def intersect_genes(*gene_name_arrays):
    """Return sorted gene names present in ALL arrays (uppercased)."""
    maps = [_build_gene_map(g) for g in gene_name_arrays]
    shared = set(maps[0].keys())
    for m in maps[1:]:
        shared &= m.keys()
    return sorted(shared)


def align_to_genes(expr, gene_names, shared_genes):
    """Reindex (samples × genes) matrix to shared_genes order."""
    gmap = _build_gene_map(gene_names)
    idx = np.array([gmap[g] for g in shared_genes])
    return expr[:, idx]


# ═══════════════════════════════════════════════════════════════════════
# 3. INTEGRATION METHODS
# ═══════════════════════════════════════════════════════════════════════

def run_harmony(expr, source_labels, n_pcs=50):
    """Harmony on PCA embedding.  Returns corrected (n × n_pcs)."""
    print("\n── Harmony ─────────────────────────────────────")
    t0 = time.time()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(expr)
    pca = PCA(n_components=n_pcs, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    print(f"  PCA variance explained: {pca.explained_variance_ratio_.sum():.1%}")

    meta = pd.DataFrame({"source": source_labels})
    ho = harmonypy.run_harmony(X_pca, meta, "source", max_iter_harmony=30)
    corrected = ho.Z_corr  # (n × n_pcs)
    print(f"  Harmony done in {time.time()-t0:.1f}s")
    return corrected


def run_scanorama_integration(expr_list, gene_names, source_names):
    """Scanorama panoramic stitching.  Returns corrected embedding."""
    print("\n── Scanorama ───────────────────────────────────")
    t0 = time.time()
    # scanorama expects list of numpy arrays and list of gene lists
    genes_list = [list(gene_names)] * len(expr_list)
    # Filter out empty datasets
    valid = [(e, g) for e, g in zip(expr_list, genes_list) if e.shape[0] > 0]
    if not valid:
        raise ValueError("No valid datasets for Scanorama")
    datasets = [e.astype(np.float64) for e, _ in valid]
    gene_lists = [g for _, g in valid]
    integrated, _ = scanorama.integrate(datasets, gene_lists)
    result = np.vstack(integrated)
    print(f"  Scanorama done in {time.time()-t0:.1f}s  →  {result.shape}")
    return result


def run_scanvi(
    cell_adatas,
    bulk_expr,
    bulk_gene_names,
    bulk_source_labels,
    shared_genes,
    n_latent=30,
    max_epochs_scvi=100,
    max_epochs_scanvi=50,
):
    """scANVI: semi-supervised VAE on cell-level + bulk/pseudobulk data."""
    import scvi as scvi_module

    print("\n── scANVI ──────────────────────────────────────")
    t0 = time.time()

    # Combine cell-level adatas
    for a in cell_adatas:
        a.obs["source"] = a.obs.get("_source", "SingleCell")
    adata_cells = ad.concat(cell_adatas, join="inner")
    adata_cells.obs_names_make_unique()
    print(f"  Combined cells: {adata_cells.shape}")

    # Restrict to shared genes
    shared_set = set(shared_genes)
    cell_genes_upper = [g.upper() for g in adata_cells.var_names]
    cell_keep = [i for i, g in enumerate(cell_genes_upper) if g in shared_set]
    adata_cells = adata_cells[:, cell_keep].copy()

    # Remap var_names to uppercase for consistency
    adata_cells.var_names = [g.upper() for g in adata_cells.var_names]
    adata_cells.var_names_make_unique()

    # Build bulk AnnData (treat each sample as a "cell")
    bulk_gmap = _build_gene_map(bulk_gene_names)
    # Align bulk to the cell var_names
    cell_var = list(adata_cells.var_names)
    bulk_idx = []
    keep_vars = []
    for i, g in enumerate(cell_var):
        if g in bulk_gmap:
            bulk_idx.append(bulk_gmap[g])
            keep_vars.append(i)
    bulk_aligned = bulk_expr[:, bulk_idx].copy()

    # filter adata_cells to same genes
    adata_cells = adata_cells[:, keep_vars].copy()

    # Create bulk adata
    adata_bulk = ad.AnnData(
        X=sp.csr_matrix(np.expm1(bulk_aligned * np.log(2)).astype(np.float32)),
        obs=pd.DataFrame({
            "cell_type": "Unknown",
            "source": bulk_source_labels,
            "donor_id": [f"bulk_{i}" for i in range(bulk_aligned.shape[0])],
        }),
    )
    adata_bulk.var_names = adata_cells.var_names.copy()
    adata_bulk.obs_names = [f"bulk_{i}" for i in range(adata_bulk.shape[0])]

    # Ensure cells have required columns
    if "source" not in adata_cells.obs.columns:
        adata_cells.obs["source"] = "SingleCell"
    if "cell_type" not in adata_cells.obs.columns:
        adata_cells.obs["cell_type"] = "Unknown"

    # Concatenate
    combined = ad.concat([adata_cells, adata_bulk], join="inner")
    combined.obs_names_make_unique()

    # Ensure X is raw counts (integer-ish) for scVI
    if sp.issparse(combined.X):
        combined.X = combined.X.toarray()
    combined.X = np.abs(np.round(combined.X)).astype(np.float32)

    print(f"  Combined for scANVI: {combined.shape}")
    print(f"  Labels: {combined.obs['cell_type'].value_counts().head(10).to_dict()}")

    # Setup and train
    scvi_module.model.SCVI.setup_anndata(combined, batch_key="source")
    vae = scvi_module.model.SCVI(combined, n_latent=n_latent)
    vae.train(max_epochs=max_epochs_scvi, early_stopping=True)

    scanvi_model = scvi_module.model.SCANVI.from_scvi_model(
        vae, unlabeled_category="Unknown", labels_key="cell_type",
    )
    scanvi_model.train(max_epochs=max_epochs_scanvi)

    latent = scanvi_model.get_latent_representation()
    print(f"  scANVI done in {time.time()-t0:.1f}s")

    # Split: cell-level vs bulk
    n_cells = adata_cells.shape[0]
    latent_cells = latent[:n_cells]
    latent_bulk = latent[n_cells:]

    return latent_bulk, latent_cells, combined.obs.iloc[n_cells:]


# ═══════════════════════════════════════════════════════════════════════
# 4. CLUSTERING & METRICS
# ═══════════════════════════════════════════════════════════════════════

def cluster_embedding(embedding, resolution=0.5):
    """Leiden clustering on a kNN graph of the embedding.
    Automatically increases resolution if only 1 cluster is found."""
    adata = ad.AnnData(X=embedding)
    sc.pp.neighbors(adata, use_rep="X", n_neighbors=min(15, embedding.shape[0] - 1))
    for res in [resolution, resolution * 2, resolution * 4, resolution * 8]:
        sc.tl.leiden(adata, resolution=res, flavor="igraph", n_iterations=2)
        labels = adata.obs["leiden"].values.astype(str)
        n_clusters = len(set(labels))
        if n_clusters >= 2:
            print(f"  Leiden: {n_clusters} clusters (resolution={res})")
            return labels
    print(f"  Leiden: {n_clusters} cluster(s) (resolution={res})")
    return labels


def batch_mixing_entropy(embedding, batch_labels, n_neighbors=50):
    """Average KNN-based entropy of batch labels (higher = better mixing)."""
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=n_neighbors).fit(embedding)
    _, indices = nn.kneighbors(embedding)
    labels = np.array(batch_labels)
    unique = np.unique(labels)
    n = len(unique)
    entropies = []
    for i in range(len(embedding)):
        neighbour_labels = labels[indices[i]]
        probs = np.array([(neighbour_labels == u).sum() for u in unique]) / n_neighbors
        probs = probs[probs > 0]
        entropies.append(-np.sum(probs * np.log(probs)) / np.log(n))
    return np.mean(entropies)


# ═══════════════════════════════════════════════════════════════════════
# 5. VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════

def compute_umap(embedding, n_neighbors=15, min_dist=0.3):
    """UMAP 2D projection."""
    import umap
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    return reducer.fit_transform(embedding)


def _scatter(ax, coords, labels, palette, title, point_size=8, alpha=0.7):
    """Categorical scatter on dark background."""
    unique_labels = sorted(set(labels))
    for lab in unique_labels:
        mask = np.array(labels) == lab
        c = palette.get(lab, "#888888")
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   s=point_size, c=c, alpha=alpha, label=lab, edgecolors="none")
    ax.set_title(title, fontsize=12, fontweight="bold", color=TEXT)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(fontsize=7, loc="upper right", framealpha=0.3,
              labelcolor=TEXT, facecolor=CARD, edgecolor=GRID)


def plot_comparison(results, out_dir):
    """Main 2×4 comparison figure."""
    methods = [r for r in results if r is not None]
    n_methods = len(methods)

    fig = plt.figure(figsize=(7 * n_methods + 5, 13))
    gs = GridSpec(2, n_methods + 1, figure=fig, wspace=0.25, hspace=0.25)

    # Build a cluster palette (tab20)
    all_clusters = set()
    for r in methods:
        all_clusters.update(r["clusters"])
    cluster_palette = {}
    cmap = plt.cm.get_cmap("tab20", len(all_clusters))
    for i, c in enumerate(sorted(all_clusters)):
        cluster_palette[c] = plt.cm.colors.to_hex(cmap(i))

    for col, r in enumerate(methods):
        # Row 0: by source
        ax0 = fig.add_subplot(gs[0, col])
        _scatter(ax0, r["umap"], r["sources"], SOURCE_COLORS,
                 f'{r["name"]} – by source', point_size=6)

        # Row 1: by cluster
        ax1 = fig.add_subplot(gs[1, col])
        _scatter(ax1, r["umap"], r["clusters"], cluster_palette,
                 f'{r["name"]} – Leiden clusters', point_size=6)

    # Metrics bar chart
    ax_m = fig.add_subplot(gs[:, n_methods])
    method_names = [r["name"] for r in methods]
    sil = [r["silhouette"] for r in methods]
    bme = [r["batch_mixing"] for r in methods]
    x = np.arange(len(methods))
    w = 0.35
    bars1 = ax_m.bar(x - w/2, sil, w, label="Silhouette", color=C_GTEX, alpha=0.85)
    bars2 = ax_m.bar(x + w/2, bme, w, label="Batch mixing", color=C_HCA, alpha=0.85)
    ax_m.set_xticks(x)
    ax_m.set_xticklabels(method_names, fontsize=10)
    ax_m.set_ylabel("Score", fontsize=11)
    ax_m.set_title("Integration quality", fontsize=12, fontweight="bold", color=TEXT)
    ax_m.legend(fontsize=9, labelcolor=TEXT, facecolor=CARD, edgecolor=GRID)
    ax_m.set_ylim(0, 1)
    for bars in [bars1, bars2]:
        for b in bars:
            ax_m.text(b.get_x() + b.get_width()/2, b.get_height() + 0.02,
                      f"{b.get_height():.2f}", ha="center", va="bottom",
                      fontsize=9, color=TEXT)

    fig.suptitle("Bulk + Single-Cell Integration Comparison",
                 fontsize=16, fontweight="bold", color=TEXT, y=0.98)
    path = os.path.join(out_dir, "integration_comparison.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved comparison figure → {path}")


def plot_single_method(umap_coords, sources, clusters, name, out_dir):
    """Standalone detail plot for one method."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    _scatter(ax1, umap_coords, sources, SOURCE_COLORS,
             f"{name} – by source", point_size=10)
    # cluster palette
    unique_c = sorted(set(clusters))
    cmap = plt.cm.get_cmap("tab20", len(unique_c))
    cpal = {c: plt.cm.colors.to_hex(cmap(i)) for i, c in enumerate(unique_c)}
    _scatter(ax2, umap_coords, clusters, cpal,
             f"{name} – Leiden clusters", point_size=10)
    fig.suptitle(name, fontsize=14, fontweight="bold", color=TEXT)
    path = os.path.join(out_dir, f"{name.lower().replace(' ', '_')}_umap.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


# ═══════════════════════════════════════════════════════════════════════
# 6. MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="Bulk + SC batch integration")
    ap.add_argument("--max-cells", type=int, default=5000,
                    help="Max cells per h5ad for scANVI (default 5000)")
    ap.add_argument("--n-pcs", type=int, default=50)
    ap.add_argument("--resolution", type=float, default=1.0)
    ap.add_argument("--scvi-epochs", type=int, default=30,
                    help="Max epochs for SCVI pre-training (default 30)")
    ap.add_argument("--scanvi-epochs", type=int, default=20,
                    help="Max epochs for SCANVI fine-tuning (default 20)")
    ap.add_argument("--skip-scanvi", action="store_true",
                    help="Skip scANVI (saves time/memory)")
    ap.add_argument("--skip-scanorama", action="store_true")
    ap.add_argument("--skip-harmony", action="store_true")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    # ── 1. Load all data sources as pseudobulk ────────────────────────
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)

    gtex_expr, gtex_genes = load_gtex()
    hca_expr, hca_genes = load_hca()

    # Aging PBMC: aggregate each h5ad to pseudobulk
    aging_exprs, aging_genes_list, aging_donors = [], [], []
    cell_adatas = []  # for scANVI
    for fname in AGING_FILES:
        fpath = os.path.join(AGING_DIR, fname)
        if not os.path.exists(fpath):
            print(f"  SKIP {fname} (not found)")
            continue
        try:
            e, g, d = load_h5ad_pseudobulk(fpath, source_name="AgingPBMC")
        except (OSError, Exception) as exc:
            print(f"  SKIP {fname} (error: {exc})")
            continue
        aging_exprs.append(e)
        aging_genes_list.append(g)
        aging_donors.extend(d)

        if not args.skip_scanvi:
            try:
                ca = load_h5ad_cells(fpath, max_cells=args.max_cells)
                ca.obs["_source"] = "AgingPBMC"
                cell_adatas.append(ca)
            except (OSError, Exception):
                pass

        gc.collect()

    # Tabula Sapiens
    tabula_expr, tabula_genes, tabula_donors = load_h5ad_pseudobulk(TABULA_PATH, "TabulaSapiens")
    if not args.skip_scanvi:
        ca = load_h5ad_cells(TABULA_PATH, max_cells=args.max_cells)
        ca.obs["_source"] = "TabulaSapiens"
        cell_adatas.append(ca)
    gc.collect()

    # ── 2. Gene intersection ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("GENE INTERSECTION")
    print("=" * 60)

    all_gene_arrays = [gtex_genes, hca_genes, tabula_genes] + aging_genes_list
    shared = intersect_genes(*all_gene_arrays)
    print(f"Shared genes across all sources: {len(shared):,}")

    # save gene list
    with open(os.path.join(OUT_DIR, "shared_genes.txt"), "w") as f:
        f.write("\n".join(shared))

    # Align all matrices
    gtex_aligned = align_to_genes(gtex_expr, gtex_genes, shared)
    hca_aligned = align_to_genes(hca_expr, hca_genes, shared)
    tabula_aligned = align_to_genes(tabula_expr, tabula_genes, shared)

    aging_aligned_list = []
    for e, g in zip(aging_exprs, aging_genes_list):
        aging_aligned_list.append(align_to_genes(e, g, shared))

    aging_aligned = np.vstack(aging_aligned_list) if aging_aligned_list else np.empty((0, len(shared)))

    # Combined matrix + labels
    combined = np.vstack([gtex_aligned, hca_aligned, aging_aligned, tabula_aligned])
    source_labels = (
        ["GTEx"] * gtex_aligned.shape[0]
        + ["HCA"] * hca_aligned.shape[0]
        + ["AgingPBMC"] * aging_aligned.shape[0]
        + ["TabulaSapiens"] * tabula_aligned.shape[0]
    )
    donor_ids = (
        [f"GTEx_{i}" for i in range(gtex_aligned.shape[0])]
        + [f"HCA_{i}" for i in range(hca_aligned.shape[0])]
        + [f"Aging_{d}" for d in aging_donors]
        + [f"Tabula_{d}" for d in tabula_donors]
    )

    print(f"\nCombined matrix: {combined.shape[0]} samples × {combined.shape[1]} genes")
    for src in SOURCE_COLORS:
        n = sum(1 for s in source_labels if s == src)
        if n > 0:
            print(f"  {src}: {n} samples")

    # ── 3. Run integration methods ────────────────────────────────────
    results = []

    # --- Harmony ---
    if not args.skip_harmony:
        harm_emb = run_harmony(combined, source_labels, n_pcs=args.n_pcs)
        harm_clusters = cluster_embedding(harm_emb, resolution=args.resolution)
        harm_umap = compute_umap(harm_emb)
        n_clust = len(set(harm_clusters))
        harm_sil = silhouette_score(harm_emb, harm_clusters) if n_clust >= 2 else 0.0
        harm_bme = batch_mixing_entropy(harm_emb, source_labels)
        print(f"  Silhouette: {harm_sil:.3f}  |  Batch mixing: {harm_bme:.3f}")

        results.append({
            "name": "Harmony",
            "embedding": harm_emb,
            "umap": harm_umap,
            "clusters": harm_clusters,
            "sources": source_labels,
            "silhouette": harm_sil,
            "batch_mixing": harm_bme,
        })
        np.save(os.path.join(OUT_DIR, "harmony_embedding.npy"), harm_emb)
        plot_single_method(harm_umap, source_labels, harm_clusters, "Harmony", OUT_DIR)

    # --- Scanorama ---
    if not args.skip_scanorama:
        scan_emb = run_scanorama_integration(
            [gtex_aligned, hca_aligned, aging_aligned, tabula_aligned],
            shared, list(SOURCE_COLORS.keys()),
        )
        scan_clusters = cluster_embedding(scan_emb, resolution=args.resolution)
        scan_umap = compute_umap(scan_emb)
        n_clust = len(set(scan_clusters))
        scan_sil = silhouette_score(scan_emb, scan_clusters) if n_clust >= 2 else 0.0
        scan_bme = batch_mixing_entropy(scan_emb, source_labels)
        print(f"  Silhouette: {scan_sil:.3f}  |  Batch mixing: {scan_bme:.3f}")

        results.append({
            "name": "Scanorama",
            "embedding": scan_emb,
            "umap": scan_umap,
            "clusters": scan_clusters,
            "sources": source_labels,
            "silhouette": scan_sil,
            "batch_mixing": scan_bme,
        })
        np.save(os.path.join(OUT_DIR, "scanorama_embedding.npy"), scan_emb)
        plot_single_method(scan_umap, source_labels, scan_clusters, "Scanorama", OUT_DIR)

    # --- scANVI ---
    if not args.skip_scanvi and cell_adatas:
        scanvi_bulk_emb, scanvi_cell_emb, scanvi_bulk_meta = run_scanvi(
            cell_adatas,
            combined,
            np.array(shared),  # gene names for bulk
            source_labels,
            shared,
            n_latent=30,
            max_epochs_scvi=args.scvi_epochs,
            max_epochs_scanvi=args.scanvi_epochs,
        )
        scanvi_clusters = cluster_embedding(scanvi_bulk_emb, resolution=args.resolution)
        scanvi_umap = compute_umap(scanvi_bulk_emb)
        n_clust = len(set(scanvi_clusters))
        scanvi_sil = silhouette_score(scanvi_bulk_emb, scanvi_clusters) if n_clust >= 2 else 0.0
        scanvi_bme = batch_mixing_entropy(scanvi_bulk_emb, source_labels)
        print(f"  Silhouette: {scanvi_sil:.3f}  |  Batch mixing: {scanvi_bme:.3f}")

        results.append({
            "name": "scANVI",
            "embedding": scanvi_bulk_emb,
            "umap": scanvi_umap,
            "clusters": scanvi_clusters,
            "sources": source_labels,
            "silhouette": scanvi_sil,
            "batch_mixing": scanvi_bme,
        })
        np.save(os.path.join(OUT_DIR, "scanvi_embedding.npy"), scanvi_bulk_emb)
        plot_single_method(scanvi_umap, source_labels, scanvi_clusters, "scANVI", OUT_DIR)

    # ── 4. Comparison ─────────────────────────────────────────────────
    if results:
        plot_comparison(results, OUT_DIR)

        # Save metadata
        meta_df = pd.DataFrame({
            "donor_id": donor_ids,
            "source": source_labels,
        })
        for r in results:
            meta_df[f'cluster_{r["name"]}'] = r["clusters"]
        meta_df.to_csv(os.path.join(OUT_DIR, "sample_metadata.csv"), index=False)

        # Save metrics
        metrics = []
        for r in results:
            metrics.append({
                "method": r["name"],
                "silhouette_score": r["silhouette"],
                "batch_mixing_entropy": r["batch_mixing"],
                "n_clusters": len(set(r["clusters"])),
            })
        pd.DataFrame(metrics).to_csv(
            os.path.join(OUT_DIR, "metrics_summary.csv"), index=False
        )
        print("\nMetrics:")
        print(pd.DataFrame(metrics).to_string(index=False))

    print(f"\nAll outputs saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
