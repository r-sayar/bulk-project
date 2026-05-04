#!/usr/bin/env python3
"""
Predict cluster proportions from HVGs with a simple MLP.

This script reuses the existing preprocessing, clustering, and synthetic bulk
simulation utilities, then trains a small multi-layer perceptron on HVG-only
bulk features to predict how much each cluster appears in a sample.

Examples:
    python hvg_mlp_cluster_proportions.py
    python hvg_mlp_cluster_proportions.py --data-source gse84133 --donor human2
    python hvg_mlp_cluster_proportions.py --data-source geo --geo-accession GSM4041647
"""

import argparse
import json
import os
import random
import sys

import numpy as np
from sklearn.model_selection import train_test_split

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
REPO_ROOT = os.path.abspath(os.path.join(SRC_ROOT, ".."))
RESULTS_DIR = os.path.join(SRC_ROOT, "results", "deconvolution_hvg_mlp")

sys.path.insert(0, SCRIPT_DIR)

from preprocessing import preprocess, select_hvgs
from deconvolution_of_bulk_rna_seq_using_deep_learning import (
    build_synthetic_reference,
    cluster_cells,
    compute_all_metrics,
    load_geo_accession,
    load_gse84133,
    simulate_bulks,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a simple HVG-based MLP to predict cluster proportions."
    )
    parser.add_argument(
        "--data-source",
        choices=["synthetic", "gse84133", "geo"],
        default="synthetic",
        help="Reference data source.",
    )
    parser.add_argument(
        "--donor",
        default="human1",
        help="Donor to use with --data-source gse84133.",
    )
    parser.add_argument(
        "--geo-accession",
        default="",
        help="GEO accession to use with --data-source geo.",
    )
    parser.add_argument("--k", type=int, default=10, help="Number of clusters.")
    parser.add_argument(
        "--n-hvgs",
        type=int,
        default=2000,
        help="Number of highly variable genes used as MLP inputs.",
    )
    parser.add_argument(
        "--n-bulks",
        type=int,
        default=800,
        help="Number of synthetic bulk mixtures to generate.",
    )
    parser.add_argument(
        "--cells-per-bulk",
        type=int,
        default=300,
        help="Number of cells per synthetic bulk.",
    )
    parser.add_argument(
        "--dirichlet-alpha",
        type=float,
        default=2.0,
        help="Dirichlet alpha for synthetic proportion generation.",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
        help="Hidden layer width for the MLP.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout rate for the MLP.",
    )
    parser.add_argument("--epochs", type=int, default=300, help="Maximum epochs.")
    parser.add_argument(
        "--patience",
        type=int,
        default=30,
        help="Early stopping patience on validation loss.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Training batch size.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Optimizer learning rate.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="AdamW weight decay.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--output-dir",
        default=RESULTS_DIR,
        help="Directory for saved predictions and metrics.",
    )
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def align_cell_labels(cell_labels, cell_mask, n_cells_after_preprocess):
    if cell_labels is None:
        return None

    filtered = np.asarray(cell_labels)[cell_mask]
    if len(filtered) != n_cells_after_preprocess:
        filtered = filtered[:n_cells_after_preprocess]
    return filtered


def load_reference_data(args):
    cell_labels = None

    if args.data_source == "gse84133":
        X_counts, gene_names, _, cell_labels = load_gse84133(donor=args.donor)
    elif args.data_source == "geo":
        if not args.geo_accession:
            raise ValueError("--geo-accession is required when --data-source geo")
        X_counts, gene_names, _, cell_labels = load_geo_accession(args.geo_accession)
    else:
        X_counts, gene_names, _, cell_labels = build_synthetic_reference(
            n_genes=max(args.n_hvgs, 2000),
            K=args.k,
            n_cells_per_type=200,
            seed=args.seed,
        )

    return X_counts, gene_names, cell_labels


def run_hvg_mlp(
    B_train,
    P_train,
    B_test,
    P_test,
    K,
    cell_type_names=None,
    hidden_dim=128,
    dropout=0.2,
    epochs=300,
    patience=30,
    batch_size=64,
    lr=1e-3,
    weight_decay=1e-4,
    seed=42,
):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    class HVGMLP(nn.Module):
        def __init__(self, n_inputs, n_outputs, hidden_dim=128, dropout=0.2):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_inputs, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, n_outputs),
            )

        def forward(self, x):
            logits = self.net(x)
            return torch.softmax(logits, dim=1)

    X_train_all = np.log1p(B_train.T.astype(np.float32))
    X_test = np.log1p(B_test.T.astype(np.float32))
    Y_train_all = P_train.astype(np.float32)

    idx_all = np.arange(X_train_all.shape[0])
    idx_tr, idx_val = train_test_split(
        idx_all, test_size=0.15, random_state=seed, shuffle=True
    )

    mean = X_train_all[idx_tr].mean(axis=0, keepdims=True)
    std = X_train_all[idx_tr].std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0

    X_train_all = (X_train_all - mean) / std
    X_test = (X_test - mean) / std

    X_tr = torch.from_numpy(X_train_all[idx_tr]).to(device)
    Y_tr = torch.from_numpy(Y_train_all[idx_tr]).to(device)
    X_val = torch.from_numpy(X_train_all[idx_val]).to(device)
    Y_val = torch.from_numpy(Y_train_all[idx_val]).to(device)
    X_te = torch.from_numpy(X_test).to(device)

    loader = DataLoader(
        TensorDataset(X_tr, Y_tr),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    model = HVGMLP(
        n_inputs=X_train_all.shape[1],
        n_outputs=K,
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    history = {"epoch": [], "train_loss": [], "val_loss": []}

    print(
        f"  Train: {len(idx_tr)}, Val: {len(idx_val)}, Test: {P_test.shape[0]}, "
        f"Features: {X_train_all.shape[1]}"
    )

    for epoch in range(1, epochs + 1):
        model.train()
        batch_losses = []

        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        train_loss = float(np.mean(batch_losses)) if batch_losses else 0.0

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = float(loss_fn(val_pred, Y_val).item())

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch == 1 or epoch % 25 == 0:
            print(
                f"  Epoch {epoch:3d} | train={train_loss:.5f} | "
                f"val={val_loss:.5f} | patience={patience_counter}/{patience}"
            )

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        print(f"  Restored best model (val MSE = {best_val_loss:.5f})")

    model.eval()
    with torch.no_grad():
        P_pred = model(X_te).cpu().numpy()

    metrics = compute_all_metrics(P_pred, P_test, K, cell_type_names)
    return P_pred, metrics, history


def main():
    args = parse_args()
    set_seed(args.seed)

    print("\n=== HVG MLP Cluster Proportion Predictor ===\n")

    print("Loading data...")
    X_counts, gene_names, cell_labels = load_reference_data(args)

    print("\nPreprocessing...")
    X_norm, X_log, genes_f, cell_mask, _ = preprocess(
        X_counts,
        gene_names,
        seed=args.seed,
    )
    cell_labels = align_cell_labels(cell_labels, cell_mask, X_norm.shape[1])

    if cell_labels is not None:
        args.k = len(np.unique(cell_labels))
        print(f"  Using {args.k} labeled cell types from metadata")

    print("\nClustering cells...")
    ref_cells, bulk_pool_cells, labels_ref, labels_bulkpool, K = cluster_cells(
        X_log,
        K=args.k,
        n_hvg=min(args.n_hvgs, X_log.shape[0]),
        cell_labels=cell_labels,
        seed=args.seed,
    )

    print("\nSimulating bulk mixtures...")
    B_full, P_true = simulate_bulks(
        X_norm,
        bulk_pool_cells,
        labels_bulkpool,
        K,
        n_samples=args.n_bulks,
        cells_per_bulk=args.cells_per_bulk,
        dirichlet_alpha=args.dirichlet_alpha,
        seed=args.seed,
    )

    print("\nSelecting HVG features...")
    hvg_idx, _ = select_hvgs(X_log, n_top=min(args.n_hvgs, X_log.shape[0]))
    hvg_genes = genes_f[hvg_idx]
    B_hvg = B_full[hvg_idx, :]
    print(f"  Selected {len(hvg_idx)} HVGs")

    idx_all = np.arange(P_true.shape[0])
    idx_train, idx_test = train_test_split(
        idx_all, test_size=0.20, random_state=args.seed, shuffle=True
    )
    B_train, B_test = B_hvg[:, idx_train], B_hvg[:, idx_test]
    P_train, P_test = P_true[idx_train], P_true[idx_test]

    ct_names = list(np.unique(cell_labels)) if cell_labels is not None else None

    print("\nTraining simple MLP on HVGs...")
    P_pred, metrics, history = run_hvg_mlp(
        B_train,
        P_train,
        B_test,
        P_test,
        K,
        cell_type_names=ct_names,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )

    print("\nResults")
    print(
        f"  RMSE={metrics['RMSE']:.4f}  MAE={metrics['MAE']:.4f}  "
        f"Pearson={metrics['Pearson']:.4f}  Spearman={metrics['Spearman']:.4f}  "
        f"CCC={metrics['CCC']:.4f}"
    )
    for row in metrics["per_celltype"]:
        print(
            f"  {row['cell_type']:>20s}  RMSE={row['RMSE']:.4f}  "
            f"MAE={row['MAE']:.4f}  Pearson={row['Pearson']:.4f}"
        )

    os.makedirs(args.output_dir, exist_ok=True)
    np.savez(
        os.path.join(args.output_dir, "hvg_mlp_predictions.npz"),
        P_test=P_test,
        P_pred=P_pred,
        idx_test=idx_test,
        hvg_idx=hvg_idx,
        hvg_genes=hvg_genes,
        labels_ref=labels_ref,
        labels_bulkpool=labels_bulkpool,
    )

    with open(os.path.join(args.output_dir, "hvg_mlp_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(os.path.join(args.output_dir, "hvg_mlp_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"\nSaved outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
