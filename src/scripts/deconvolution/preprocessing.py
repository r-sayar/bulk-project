#!/usr/bin/env python3
"""
Preprocessing for bulk RNA-seq deconvolution pipeline.

Provides filtering, normalization, and HVG selection with optional disk caching.
"""

import os
import numpy as np
import scipy.sparse as sp
from joblib import Memory

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
REPO_ROOT = os.path.abspath(os.path.join(SRC_ROOT, ".."))

# Default cache directory (add to .gitignore)
DEFAULT_CACHE_DIR = os.path.join(REPO_ROOT, ".cache", "preprocessing")
_memory = None


def _get_memory(cache_dir=None):
    """Lazily create Memory instance for caching."""
    global _memory
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR
    if _memory is None or _memory.location != cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        _memory = Memory(cache_dir, verbose=0)
    return _memory


def preprocess(X, gene_names, min_genes_per_cell=200, min_cells_per_gene=3,
               target_sum=1e4, max_cells=20000, seed=42, use_cache=True,
               cache_dir=None):
    """Filter cells/genes, normalize to CP10K, and create log1p copy.

    When use_cache=True (default), results are cached to disk. Cache is
    invalidated when X, gene_names, or any parameter changes.

    Returns:
        X_norm: CP10K-normalized sparse matrix (genes × cells)
        X_log: log1p-transformed copy
        genes_f: filtered gene names
        cell_mask: boolean mask of cells kept
        gene_mask: boolean mask of genes kept
    """
    if use_cache:
        mem = _get_memory(cache_dir)
        return mem.cache(_preprocess_impl)(
            X, gene_names,
            min_genes_per_cell=min_genes_per_cell,
            min_cells_per_gene=min_cells_per_gene,
            target_sum=target_sum,
            max_cells=max_cells,
            seed=seed,
        )
    return _preprocess_impl(
        X, gene_names,
        min_genes_per_cell=min_genes_per_cell,
        min_cells_per_gene=min_cells_per_gene,
        target_sum=target_sum,
        max_cells=max_cells,
        seed=seed,
    )


def _preprocess_impl(X, gene_names, min_genes_per_cell=200, min_cells_per_gene=3,
                     target_sum=1e4, max_cells=20000, seed=42):
    """Internal implementation (no caching)."""
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
