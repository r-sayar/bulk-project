#!/usr/bin/env python3
"""
Plot cell-proportion distributions for human GSE84133 donors and synthetic bulks.

Outputs:
  - human_cell_proportion_per_sample.csv
  - human_cell_proportion_per_sample.png
  - synthetic_cell_proportion_distribution.csv
  - synthetic_cell_proportion_distribution_with_centroids.png
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
EVAL_DIR = os.path.join(SRC_ROOT, "results", "deconvolution_evaluation")

sys.path.insert(0, SCRIPT_DIR)

from deconvolution_of_bulk_rna_seq_using_deep_learning import (
    build_synthetic_reference,
    load_gse84133,
    simulate_bulks,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot human and synthetic cell-proportion distributions."
    )
    parser.add_argument(
        "--donors",
        nargs="+",
        default=["human1", "human2", "human3", "human4"],
        help="Human GSE84133 donors to include.",
    )
    parser.add_argument(
        "--n-bulks",
        type=int,
        default=800,
        help="Number of synthetic bulks to simulate.",
    )
    parser.add_argument(
        "--cells-per-bulk",
        type=int,
        default=300,
        help="Cells per synthetic bulk sample.",
    )
    parser.add_argument(
        "--dirichlet-alpha",
        type=float,
        default=2.0,
        help="Dirichlet alpha used for synthetic compositions.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Number of synthetic cell types.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--out-dir",
        default=EVAL_DIR,
        help=f"Output directory (default: {EVAL_DIR})",
    )
    return parser.parse_args()


def human_celltype_proportions(donors):
    rows = []

    for donor in donors:
        _, _, _, cell_labels = load_gse84133(donor=donor)
        if cell_labels is None:
            raise ValueError(f"No `assigned_cluster` labels found for donor {donor}.")

        counts = pd.Series(cell_labels).value_counts().sort_index()
        props = counts / counts.sum()
        row = {"sample": donor, "n_cells": int(counts.sum())}
        row.update(props.to_dict())
        rows.append(row)

    df = pd.DataFrame(rows).fillna(0.0)
    celltype_cols = [c for c in df.columns if c not in ("sample", "n_cells")]
    df = df[["sample", "n_cells", *sorted(celltype_cols)]]
    return df


def plot_human_proportions(df, out_dir):
    celltype_cols = [c for c in df.columns if c not in ("sample", "n_cells")]
    colors = list(plt.cm.tab20.colors) + list(plt.cm.tab20b.colors) + list(plt.cm.tab20c.colors)

    fig, ax = plt.subplots(figsize=(12, 6))
    bottom = np.zeros(len(df), dtype=float)

    for i, cell_type in enumerate(celltype_cols):
        vals = df[cell_type].to_numpy(dtype=float)
        ax.bar(
            df["sample"],
            vals,
            bottom=bottom,
            label=cell_type,
            color=colors[i % len(colors)],
            edgecolor="white",
            linewidth=0.6,
        )
        bottom += vals

    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Cell proportion")
    ax.set_xlabel("Human sample")
    ax.set_title("Human GSE84133 Cell-Type Proportions per Sample")
    ax.legend(
        title="Cell type",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=8,
        title_fontsize=9,
        frameon=False,
    )
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    csv_path = os.path.join(out_dir, "human_cell_proportion_per_sample.csv")
    png_path = os.path.join(out_dir, "human_cell_proportion_per_sample.png")
    df.to_csv(csv_path, index=False)
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return csv_path, png_path


def synthetic_proportion_distribution(n_bulks, cells_per_bulk, dirichlet_alpha, k, seed):
    x_norm, _, _, labels = build_synthetic_reference(
        n_genes=2000,
        K=k,
        n_cells_per_type=200,
        seed=seed,
    )
    bulk_pool_cells = np.arange(x_norm.shape[1])
    _, p_true = simulate_bulks(
        x_norm,
        bulk_pool_cells,
        labels,
        k,
        n_samples=n_bulks,
        cells_per_bulk=cells_per_bulk,
        dirichlet_alpha=dirichlet_alpha,
        seed=seed,
    )
    return p_true


def plot_synthetic_distribution(p_true, out_dir):
    k = p_true.shape[1]
    labels = [f"C{i}" for i in range(k)]
    centroids = p_true.mean(axis=0)

    records = []
    for i, label in enumerate(labels):
        for value in p_true[:, i]:
            records.append(
                {"cell_type": label, "proportion": float(value), "centroid": float(centroids[i])}
            )
    df = pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=(12, 6))
    parts = ax.violinplot(
        [p_true[:, i] for i in range(k)],
        positions=np.arange(1, k + 1),
        showmeans=False,
        showmedians=True,
        showextrema=False,
    )

    colors = plt.cm.Set3(np.linspace(0, 1, k))
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor("black")
        body.set_alpha(0.8)

    ax.scatter(
        np.arange(1, k + 1),
        centroids,
        color="black",
        marker="D",
        s=48,
        label="Centroid (mean proportion)",
        zorder=3,
    )
    ax.set_xticks(np.arange(1, k + 1))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Cell proportion")
    ax.set_xlabel("Synthetic cell type")
    ax.set_title("Synthetic Bulk Cell-Proportion Distribution with Centroids")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()

    csv_path = os.path.join(out_dir, "synthetic_cell_proportion_distribution.csv")
    png_path = os.path.join(out_dir, "synthetic_cell_proportion_distribution_with_centroids.png")
    df.to_csv(csv_path, index=False)
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return csv_path, png_path


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    human_df = human_celltype_proportions(args.donors)
    human_csv, human_png = plot_human_proportions(human_df, args.out_dir)

    p_true = synthetic_proportion_distribution(
        n_bulks=args.n_bulks,
        cells_per_bulk=args.cells_per_bulk,
        dirichlet_alpha=args.dirichlet_alpha,
        k=args.k,
        seed=args.seed,
    )
    synthetic_csv, synthetic_png = plot_synthetic_distribution(p_true, args.out_dir)

    print(f"Saved: {human_csv}")
    print(f"Saved: {human_png}")
    print(f"Saved: {synthetic_csv}")
    print(f"Saved: {synthetic_png}")


if __name__ == "__main__":
    main()
