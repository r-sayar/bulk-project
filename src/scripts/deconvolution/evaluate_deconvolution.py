#!/usr/bin/env python3
"""
Deconvolution Evaluation & Benchmarking
========================================

Post-hoc evaluation of deconvolution models (NNLS, NMF, Neural W-CLS v3).

Two modes of operation:
  1. Load saved artifacts from a previous pipeline run (predictions.npz)
  2. Re-run the full pipeline from scratch (imports from the main script)

Produces:
  - Comprehensive global + per-cell-type metrics
  - Scatter plots (true vs predicted, colored by cell type)
  - Per-cell-type bar charts (RMSE/MAE per method)
  - CCC heatmap (methods x cell types)
  - Bland-Altman plots
  - Summary CSV tables

Usage:
    python evaluate_deconvolution.py                        # load saved results
    python evaluate_deconvolution.py --rerun                # force re-run
    python evaluate_deconvolution.py --donor human2         # different donor
    python evaluate_deconvolution.py --methods nnls neural  # subset

Original pipeline:
    deconvolution_of_bulk_rna_seq_using_deep_learning.py
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
REPO_ROOT = os.path.abspath(os.path.join(SRC_ROOT, ".."))
EVAL_DIR = os.path.join(SRC_ROOT, "results", "deconvolution_evaluation")

sys.path.insert(0, SCRIPT_DIR)

from deconvolution_of_bulk_rna_seq_using_deep_learning import (
    rmse, mae, pearson_corr_flat, concordance_corr_coef,
    compute_all_metrics,
    load_gse84133, build_synthetic_reference,
    cluster_cells, build_signature_matrix,
    simulate_bulks, select_deconv_genes,
    run_nnls, run_nmf, run_neural_wcls,
)
from preprocessing import preprocess
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr


# ═══════════════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════════════

COLORS_METHOD = {
    "NNLS": "#d95f02",
    "NMF": "#7570b3",
    "Neural W-CLS v3": "#e7298a",
}

CT_CMAP = matplotlib.colormaps.get_cmap("tab20")


def plot_scatter_grid(results, P_test, K, ct_names, out_dir):
    """True vs predicted scatter per method, points colored by cell type."""
    methods = list(results.keys())
    n = len(methods)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5), squeeze=False)

    for col, name in enumerate(methods):
        ax = axes[0, col]
        P_pred = results[name]
        for k in range(K):
            label = ct_names[k] if ct_names is not None else f"C{k}"
            ax.scatter(P_test[:, k], P_pred[:, k], s=12, alpha=0.5,
                       color=CT_CMAP(k / max(K - 1, 1)), label=label)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4, lw=1)
        r = pearson_corr_flat(P_pred, P_test)
        ccc = concordance_corr_coef(P_pred, P_test)
        ax.set_title(f"{name}\nr={r:.3f}  CCC={ccc:.3f}  "
                     f"RMSE={rmse(P_pred, P_test):.4f}")
        ax.set_xlabel("True proportion")
        ax.set_ylabel("Predicted proportion")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(K, 7),
               fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    path = os.path.join(out_dir, "scatter_grid.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_per_celltype_bars(all_metrics, K, ct_names, out_dir):
    """Grouped bar chart: RMSE and MAE per cell type, grouped by method."""
    methods = list(all_metrics.keys())
    n_methods = len(methods)
    x = np.arange(K)
    width = 0.8 / n_methods

    fig, axes = plt.subplots(1, 2, figsize=(7 + K * 0.4, 5))

    for metric_name, ax in zip(["RMSE", "MAE"], axes):
        for i, mname in enumerate(methods):
            vals = [ct[metric_name] for ct in all_metrics[mname]["per_celltype"]]
            color = COLORS_METHOD.get(mname, f"C{i}")
            ax.bar(x + i * width - 0.4 + width / 2, vals,
                   width=width, label=mname, color=color, alpha=0.85)
        labels = ([ct_names[k] for k in range(K)]
                  if ct_names is not None else [f"C{k}" for k in range(K)])
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel(metric_name)
        ax.set_title(f"Per-cell-type {metric_name}")
        ax.legend(fontsize=8)

    fig.tight_layout()
    path = os.path.join(out_dir, "per_celltype_bars.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_ccc_heatmap(all_metrics, K, ct_names, out_dir):
    """Heatmap: CCC values, methods (rows) x cell types (columns)."""
    methods = list(all_metrics.keys())
    matrix = np.zeros((len(methods), K))
    for i, mname in enumerate(methods):
        for k in range(K):
            matrix[i, k] = all_metrics[mname]["per_celltype"][k]["CCC"]

    labels = ([ct_names[k] for k in range(K)]
              if ct_names is not None else [f"C{k}" for k in range(K)])

    fig, ax = plt.subplots(figsize=(max(6, K * 0.6), 2 + len(methods) * 0.6))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=-0.2, vmax=1.0, aspect="auto")
    ax.set_xticks(range(K))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods, fontsize=9)
    for i in range(len(methods)):
        for j in range(K):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color="black" if matrix[i, j] > 0.3 else "white")
    fig.colorbar(im, ax=ax, label="CCC", shrink=0.8)
    ax.set_title("Concordance Correlation Coefficient per Cell Type")
    fig.tight_layout()
    path = os.path.join(out_dir, "ccc_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_bland_altman(results, P_test, out_dir):
    """Bland-Altman plots: (pred - true) vs (pred + true)/2 per method."""
    methods = list(results.keys())
    n = len(methods)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.5), squeeze=False)

    for col, name in enumerate(methods):
        ax = axes[0, col]
        P_pred = results[name]
        diff = (P_pred - P_test).ravel()
        mean_val = ((P_pred + P_test) / 2).ravel()
        ax.scatter(mean_val, diff, s=8, alpha=0.3,
                   color=COLORS_METHOD.get(name, "C0"))
        ax.axhline(0, color="k", lw=0.8, ls="--")
        md = np.mean(diff)
        sd = np.std(diff)
        ax.axhline(md, color="red", lw=1, ls="-", label=f"mean={md:.4f}")
        ax.axhline(md + 1.96 * sd, color="red", lw=0.8, ls=":")
        ax.axhline(md - 1.96 * sd, color="red", lw=0.8, ls=":")
        ax.set_xlabel("Mean of true & predicted")
        ax.set_ylabel("Predicted - True")
        ax.set_title(f"Bland-Altman: {name}")
        ax.legend(fontsize=8)

    fig.tight_layout()
    path = os.path.join(out_dir, "bland_altman.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════
# LOADING / RE-RUNNING
# ═══════════════════════════════════════════════════════════════════════

def load_saved_artifacts(eval_dir):
    """Load predictions.npz and return (results, P_test, ct_names)."""
    npz_path = os.path.join(eval_dir, "predictions.npz")
    if not os.path.exists(npz_path):
        return None, None, None

    data = np.load(npz_path, allow_pickle=True)
    P_test = data["P_test"]
    ct_names = (list(data["cell_type_names"])
                if "cell_type_names" in data else None)

    results = {}
    for key in data.files:
        if key.startswith("P_pred_"):
            method = key[len("P_pred_"):].replace("_", " ")
            method = method.title()
            if "Nmf" in method:
                method = "NMF"
            elif "Nnls" in method:
                method = "NNLS"
            elif "Neural" in method:
                method = "Neural W-CLS v3"
            results[method] = data[key]

    return results, P_test, ct_names


def rerun_pipeline(donor="human1", n_bulks=800, cells_per_bulk=300,
                   dirichlet_alpha=2.0, n_deconv_genes=2000,
                   methods=None):
    """Re-run the deconvolution pipeline and return results for evaluation."""
    if methods is None:
        methods = ["nnls", "nmf", "neural"]

    print(f"\n{'═' * 56}")
    print("  RE-RUNNING PIPELINE")
    print(f"{'═' * 56}")

    if donor == "synthetic":
        X_counts, gene_names, barcodes, cell_labels = build_synthetic_reference(
            n_genes=2000, K=10, n_cells_per_type=200, seed=42,
        )
        K = 10
    else:
        X_counts, gene_names, barcodes, cell_labels = load_gse84133(donor=donor)
        K = len(np.unique(cell_labels)) if cell_labels is not None else 10

    X_norm, X_log, genes_f, _, _ = preprocess(X_counts, gene_names)

    if cell_labels is not None and len(cell_labels) > X_norm.shape[1]:
        genes_per_cell = np.asarray((X_counts > 0).sum(axis=0)).ravel()
        cell_mask = genes_per_cell >= 200
        cell_labels = cell_labels[cell_mask]
        if len(cell_labels) > X_norm.shape[1]:
            cell_labels = cell_labels[:X_norm.shape[1]]

    ref_cells, bulk_pool_cells, labels_ref, labels_bulkpool, K = cluster_cells(
        X_log, K=K, cell_labels=cell_labels,
    )
    S_full = build_signature_matrix(X_norm, ref_cells, labels_ref, K)

    B_full, P_true = simulate_bulks(
        X_norm, bulk_pool_cells, labels_bulkpool, K,
        n_samples=n_bulks, cells_per_bulk=cells_per_bulk,
        dirichlet_alpha=dirichlet_alpha,
    )
    top_genes = select_deconv_genes(B_full, genes_f, n_genes=n_deconv_genes)
    B = B_full[top_genes, :]
    S = S_full[top_genes, :]

    idx_all = np.arange(P_true.shape[0])
    idx_train, idx_test = train_test_split(idx_all, test_size=0.20, random_state=42)
    B_train, B_test = B[:, idx_train], B[:, idx_test]
    P_train, P_test = P_true[idx_train], P_true[idx_test]

    ct_names = list(np.unique(cell_labels)) if cell_labels is not None else None

    results = {}
    all_metrics = {}

    if "nnls" in methods:
        P_nnls, m = run_nnls(S, B_test, P_test, K, ct_names)
        results["NNLS"] = P_nnls
        all_metrics["NNLS"] = m

    if "nmf" in methods:
        P_nmf, m = run_nmf(B_train, B_test, P_train, P_test, K, ct_names)
        results["NMF"] = P_nmf
        all_metrics["NMF"] = m

    if "neural" in methods:
        P_neural, m = run_neural_wcls(
            S, B_train, P_train, B_test, P_test, K, ct_names,
        )
        results["Neural W-CLS v3"] = P_neural
        all_metrics["Neural W-CLS v3"] = m

    # Save artifacts
    os.makedirs(EVAL_DIR, exist_ok=True)
    save_data = {"P_test": P_test, "idx_test": idx_test}
    if ct_names is not None:
        save_data["cell_type_names"] = np.array(ct_names)
    for name, P_pred in results.items():
        key = name.replace(" ", "_").replace("-", "_")
        save_data[f"P_pred_{key}"] = P_pred
    np.savez(os.path.join(EVAL_DIR, "predictions.npz"), **save_data)

    return results, P_test, ct_names, all_metrics


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate deconvolution model predictions",
    )
    parser.add_argument("--rerun", action="store_true",
                        help="Force re-run of the pipeline (ignore saved results)")
    parser.add_argument("--donor", default="human1",
                        help="GSE84133 donor or 'synthetic' (default: human1)")
    parser.add_argument("--n-bulks", type=int, default=800,
                        help="Number of simulated bulk samples (default: 800)")
    parser.add_argument("--cells-per-bulk", type=int, default=300)
    parser.add_argument("--dirichlet-alpha", type=float, default=2.0)
    parser.add_argument("--n-deconv-genes", type=int, default=2000)
    parser.add_argument("--methods", nargs="+", default=["nnls", "nmf", "neural"],
                        choices=["nnls", "nmf", "neural"],
                        help="Which methods to evaluate")
    parser.add_argument("--out-dir", default=None,
                        help=f"Output directory (default: {EVAL_DIR})")
    args = parser.parse_args()

    out_dir = args.out_dir or EVAL_DIR
    os.makedirs(out_dir, exist_ok=True)

    print("╔══════════════════════════════════════════════════════╗")
    print("║         Deconvolution Evaluation Pipeline            ║")
    print("╚══════════════════════════════════════════════════════╝")

    # --- Load or re-run ---
    all_metrics = None
    if not args.rerun:
        results, P_test, ct_names = load_saved_artifacts(out_dir)
        if results is not None:
            print(f"\n  Loaded saved predictions from {out_dir}/predictions.npz")
            print(f"  Methods found: {', '.join(results.keys())}")
            print(f"  Test samples: {P_test.shape[0]}, Cell types: {P_test.shape[1]}")
        else:
            print("\n  No saved predictions found, re-running pipeline...")
            args.rerun = True

    if args.rerun:
        results, P_test, ct_names, all_metrics = rerun_pipeline(
            donor=args.donor, n_bulks=args.n_bulks,
            cells_per_bulk=args.cells_per_bulk,
            dirichlet_alpha=args.dirichlet_alpha,
            n_deconv_genes=args.n_deconv_genes,
            methods=args.methods,
        )

    K = P_test.shape[1]

    # --- Compute metrics (if not already computed during re-run) ---
    if all_metrics is None:
        all_metrics = {}
        for name, P_pred in results.items():
            all_metrics[name] = compute_all_metrics(P_pred, P_test, K, ct_names)

    # --- Print summary ---
    print(f"\n{'═' * 72}")
    print("  EVALUATION SUMMARY")
    print(f"{'═' * 72}")

    w = 72
    print(f"\n╔{'═' * w}╗")
    print(f"║{'GLOBAL METRICS':^{w}}║")
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

    print(f"\n  Per-cell-type detail:")
    for name, m in all_metrics.items():
        print(f"\n  --- {name} ---")
        for ct in m["per_celltype"]:
            print(f"    {ct['cell_type']:>20s}  RMSE={ct['RMSE']:.4f}  "
                  f"MAE={ct['MAE']:.4f}  r={ct['Pearson']:.4f}  "
                  f"CCC={ct['CCC']:.4f}  bias={ct['bias']:+.4f}")

    # --- Save metric tables ---
    rows = []
    for name, m in all_metrics.items():
        rows.append({
            "method": name, "RMSE": m["RMSE"], "MAE": m["MAE"],
            "Pearson": m["Pearson"], "Spearman": m["Spearman"], "CCC": m["CCC"],
        })
    summary_path = os.path.join(out_dir, "metrics_summary.csv")
    pd.DataFrame(rows).to_csv(summary_path, index=False)

    ct_rows = []
    for name, m in all_metrics.items():
        for ct in m["per_celltype"]:
            ct_rows.append({"method": name, **ct})
    ct_path = os.path.join(out_dir, "metrics_per_celltype.csv")
    pd.DataFrame(ct_rows).to_csv(ct_path, index=False)

    print(f"\n  Saved: {summary_path}")
    print(f"  Saved: {ct_path}")

    # --- Generate plots ---
    print(f"\n{'═' * 72}")
    print("  GENERATING PLOTS")
    print(f"{'═' * 72}\n")

    plot_scatter_grid(results, P_test, K, ct_names, out_dir)
    plot_per_celltype_bars(all_metrics, K, ct_names, out_dir)
    plot_ccc_heatmap(all_metrics, K, ct_names, out_dir)
    plot_bland_altman(results, P_test, out_dir)

    print(f"\n  All outputs saved to: {out_dir}/")
    print("\nDone.")


if __name__ == "__main__":
    main()
