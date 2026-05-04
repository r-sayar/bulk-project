#!/usr/bin/env python3
"""
DRVI on bulk RNA-seq
====================
Adapt theislab/DRVI (a disentangled VAE for single-cell omics) to GTEx bulk
RNA-seq + HCA pseudobulk. Trained at batch_size=1 since each bulk sample is
one observation.

WHY DRVI HERE?
--------------
DRVI's split-decoder forces each latent dimension to act additively in
expression space. For bulk RNA-seq this could yield latent axes that
correspond to implicit cell-type proportions / tissue states — useful as
features for downstream deconvolution.

INPUT
-----
- GTEx blood (raw counts, GCT):  ~803 donors × 74k genes
- HCA pseudobulk (raw counts, NPZ): variable donor count

OUTPUT
------
- Trained DRVI model in OUT_DIR/drvi_model/
- Loss curves: drvi_loss_curves.png
- Latent t-SNE: drvi_latent_tsne.png
- model.history pickle: drvi_history.pkl

USAGE
-----
  # Sanity test (small + fast)
  python drvi_bulk.py --epochs 50 --batch-size 1 --n-latent 8 --max-samples 100

  # Full run
  python drvi_bulk.py --epochs 400 --batch-size 1 --n-latent 8
"""

from __future__ import annotations
import argparse
import gzip
import os
import pickle
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────────
BASEDIR = "/Users/rls/Desktop/programming-projects/single-cell/bulk-project"
GTEX_PATH = "/Users/rls/Downloads/gene_reads_v11_whole_blood.gct.gz"
PSEUDO = os.path.join(BASEDIR, "pseudobulk/hca_blood_pseudobulk.npz")
OUT_DIR = os.path.join(BASEDIR, "drvi_outputs")

# ── plot style (same look as cross_modality_vae.py) ────────────────────
BG, CARD, TEXT, MUTED, GRID = "#0e1117", "#1a1d23", "#e6edf3", "#7d8590", "#21262d"
C_G, C_H = "#f78166", "#3fb950"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": CARD, "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT, "text.color": TEXT, "xtick.color": MUTED,
    "ytick.color": MUTED, "grid.color": GRID, "grid.alpha": 0.5,
    "font.family": "sans-serif", "font.size": 11,
})


# ══════════════════════════════════════════════════════════════════════
# 1. RAW-COUNT DATA LOADERS  (DRVI wants raw counts for pnb_softmax)
# ══════════════════════════════════════════════════════════════════════

def load_gtex_raw(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load GTEx GCT file.
    Returns (counts: float32 [n_samples, n_genes], gene_symbols, sample_ids).
    """
    print(f"Loading GTEx raw counts from {path} ...")
    df = pd.read_csv(path, sep="\t", skiprows=2, compression="gzip")
    sample_ids = np.array(df.columns[2:].tolist(), dtype=str)
    gene_symbols = df["Description"].astype(str).values
    counts = df.iloc[:, 2:].values.astype(np.float32).T  # → (samples, genes)
    print(f"  {counts.shape[0]} samples × {counts.shape[1]:,} genes (raw counts)")
    return counts, gene_symbols, sample_ids


def load_hca_raw(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load HCA pseudobulk NPZ. Expected keys: 'expr' (genes, donors) raw counts,
    'gene_names', and ideally 'donor_ids'.
    """
    print(f"Loading HCA pseudobulk from {path} ...")
    d = np.load(path, allow_pickle=True)
    expr = d["expr"].astype(np.float32).T  # → (donors, genes)
    gene_names = np.array(d["gene_names"]).astype(str)
    donor_ids = (
        np.array(d["donor_ids"]).astype(str)
        if "donor_ids" in d.files
        else np.array([f"hca_{i}" for i in range(expr.shape[0])])
    )
    print(f"  {expr.shape[0]} donors × {expr.shape[1]:,} genes (pseudobulk counts)")
    return expr, gene_names, donor_ids


def shared_gene_intersect(
    bulk_x: np.ndarray, bulk_g: np.ndarray,
    sc_x: np.ndarray, sc_g: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align by upper-cased gene symbol (first occurrence wins)."""
    def first_idx_map(names):
        m = {}
        for i, n in enumerate(names):
            k = n.upper()
            if k not in m:
                m[k] = i
        return m

    a, b = first_idx_map(bulk_g), first_idx_map(sc_g)
    shared = sorted(set(a) & set(b))
    ai = np.array([a[g] for g in shared], dtype=int)
    bi = np.array([b[g] for g in shared], dtype=int)
    print(f"  Shared genes: {len(shared):,}")
    return bulk_x[:, ai], sc_x[:, bi], np.array(shared)


def filter_low_expressed(counts: np.ndarray, gene_names: np.ndarray,
                         min_count: int = 1, min_frac: float = 0.1
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Keep genes with ≥ count `min_count` in ≥ `min_frac` of samples."""
    keep = (counts >= min_count).mean(axis=0) >= min_frac
    print(f"  Filter low-expressed: kept {keep.sum():,} / {len(keep):,} genes "
          f"(≥{min_count} counts in ≥{min_frac*100:.0f}% samples)")
    return counts[:, keep], gene_names[keep]


# ══════════════════════════════════════════════════════════════════════
# 2. BUILD AnnData FOR DRVI
# ══════════════════════════════════════════════════════════════════════

def build_adata(use_pseudobulk: bool = True, max_samples: int | None = None):
    """
    Build a single AnnData with bulk + (optional) pseudobulk.
    Each row is one sample (donor); X is RAW counts.
    obs['modality'] ∈ {'bulk', 'pseudobulk'} — passed as DRVI categorical covariate.
    """
    import anndata as ad

    bulk_x, bulk_g, bulk_ids = load_gtex_raw(GTEX_PATH)

    if use_pseudobulk and os.path.exists(PSEUDO):
        sc_x, sc_g, sc_ids = load_hca_raw(PSEUDO)
        bulk_x, sc_x, shared = shared_gene_intersect(bulk_x, bulk_g, sc_x, sc_g)
        X = np.vstack([bulk_x, sc_x])
        obs_modality = np.array(["bulk"] * bulk_x.shape[0] + ["pseudobulk"] * sc_x.shape[0])
        obs_ids = np.concatenate([bulk_ids, sc_ids])
        gene_names = shared
    else:
        X = bulk_x
        obs_modality = np.array(["bulk"] * bulk_x.shape[0])
        obs_ids = bulk_ids
        gene_names = bulk_g

    X, gene_names = filter_low_expressed(X, gene_names, min_count=1, min_frac=0.1)

    # Subsample for sanity tests
    if max_samples is not None and X.shape[0] > max_samples:
        rng = np.random.default_rng(42)
        idx = np.sort(rng.choice(X.shape[0], max_samples, replace=False))
        X = X[idx]
        obs_modality = obs_modality[idx]
        obs_ids = obs_ids[idx]
        print(f"  Subsampled to {max_samples} samples")

    obs = pd.DataFrame(
        {"modality": pd.Categorical(obs_modality), "sample_id": obs_ids},
        index=obs_ids,
    )
    var = pd.DataFrame(index=gene_names)
    # DRVI wants integer counts under pnb_softmax
    adata = ad.AnnData(X=X.astype(np.float32), obs=obs, var=var)
    adata.layers["counts"] = adata.X.copy()
    print(f"  AnnData: {adata.shape[0]} obs × {adata.shape[1]:,} vars")
    print(f"  Modality breakdown: {obs['modality'].value_counts().to_dict()}")
    return adata


# ══════════════════════════════════════════════════════════════════════
# 3. TRAIN DRVI
# ══════════════════════════════════════════════════════════════════════

def train_drvi(adata, n_latent: int, max_epochs: int, batch_size: int,
               lr: float, encoder_dims: tuple[int, ...],
               decoder_dims: tuple[int, ...], dropout: float,
               accelerator: str):
    """Build & train a DRVI model on the given AnnData."""
    from drvi.model import DRVI

    # If we have only one modality, no covariates needed
    cat_keys = ["modality"] if adata.obs["modality"].nunique() > 1 else None

    DRVI.setup_anndata(
        adata,
        layer="counts",
        is_count_data=True,
        categorical_covariate_keys=cat_keys,
    )

    print(f"\nDRVI config: n_latent={n_latent}, encoder_dims={encoder_dims}, "
          f"decoder_dims={decoder_dims}, dropout={dropout}")
    model = DRVI(
        adata,
        n_latent=n_latent,
        encoder_dims=list(encoder_dims),
        decoder_dims=list(decoder_dims),
        gene_likelihood="pnb",       # log-Negative-Binomial (DRVI default)
        prior="normal",
        encoder_dropout_rate=dropout,
        decoder_dropout_rate=0.0,
    )
    print(model)

    print(f"\nTraining: max_epochs={max_epochs}, batch_size={batch_size}, "
          f"lr={lr}, accelerator={accelerator}")
    model.train(
        max_epochs=max_epochs,
        batch_size=batch_size,
        plan_kwargs=dict(lr=lr, n_epochs_kl_warmup=max_epochs),
        early_stopping=False,
        accelerator=accelerator,
        check_val_every_n_epoch=1,
    )
    return model


# ══════════════════════════════════════════════════════════════════════
# 4. EVALUATION & PLOTS
# ══════════════════════════════════════════════════════════════════════

def plot_loss_curves(history: dict, out_path: str):
    """Plot training/validation loss components from model.history."""
    keys = list(history.keys())
    print(f"\nmodel.history keys ({len(keys)}): {keys[:20]}{'...' if len(keys) > 20 else ''}")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    # Total loss
    for k, color, label in [
        ("elbo_train", C_G, "ELBO train"),
        ("elbo_validation", C_H, "ELBO val"),
    ]:
        if k in history:
            df = history[k]
            ax[0].plot(df.index, df.values.flatten(), color=color, label=label, lw=1.5)
    ax[0].set_xlabel("Epoch"); ax[0].set_ylabel("ELBO (lower is better)")
    ax[0].set_title("Total ELBO"); ax[0].legend(loc="best", framealpha=0.3)
    ax[0].grid(True, alpha=0.3)

    # Reconstruction + KL
    for k, color, label in [
        ("reconstruction_loss_train", C_G, "Recon train"),
        ("reconstruction_loss_validation", C_H, "Recon val"),
    ]:
        if k in history:
            df = history[k]
            ax[1].plot(df.index, df.values.flatten(), color=color, label=label, lw=1.5)
    if "kl_local_train" in history:
        df = history["kl_local_train"]
        ax2 = ax[1].twinx()
        ax2.plot(df.index, df.values.flatten(), color="#a371f7", label="KL train", lw=1.0, linestyle="--")
        ax2.set_ylabel("KL", color="#a371f7")
        ax2.tick_params(axis="y", colors="#a371f7")
    ax[1].set_xlabel("Epoch"); ax[1].set_ylabel("Reconstruction loss")
    ax[1].set_title("Reconstruction & KL"); ax[1].legend(loc="best", framealpha=0.3)
    ax[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, facecolor=BG)
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_latent_tsne(model, adata, out_path: str):
    """t-SNE of DRVI latent space, colored by modality."""
    from sklearn.manifold import TSNE

    z = model.get_latent_representation(adata, batch_size=4096)
    print(f"\nLatent shape: {z.shape}  (n_obs × n_latent)")
    print(f"Latent dim usage (std per dim):")
    for i, s in enumerate(z.std(axis=0)):
        marker = "✓" if s > 0.1 else "·"
        print(f"  dim {i:2d}: std={s:7.4f}  {marker}")

    if z.shape[0] < 4:
        print("  Too few samples for t-SNE")
        return

    perp = min(30, max(2, z.shape[0] // 4))
    coords = TSNE(n_components=2, perplexity=perp, random_state=42,
                  init="pca", learning_rate="auto").fit_transform(z)

    fig, ax = plt.subplots(figsize=(7, 6))
    palette = {"bulk": C_G, "pseudobulk": C_H}
    for mod in adata.obs["modality"].unique():
        mask = adata.obs["modality"].values == mod
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   color=palette.get(mod, "#888"), s=22, alpha=0.7,
                   edgecolor="none", label=mod)
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    ax.set_title(f"DRVI latent space (dim={z.shape[1]}, n={z.shape[0]})")
    ax.legend(framealpha=0.3); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, facecolor=BG)
    plt.close(fig)
    print(f"  Saved {out_path}")


# ══════════════════════════════════════════════════════════════════════
# 5. MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epochs", type=int, default=400, help="max_epochs")
    p.add_argument("--batch-size", type=int, default=1, help="mini-batch size (1 for true bulk)")
    p.add_argument("--n-latent", type=int, default=8, help="latent dim")
    p.add_argument("--encoder-dims", type=int, nargs="+", default=[64, 64])
    p.add_argument("--decoder-dims", type=int, nargs="+", default=[64, 64])
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max-samples", type=int, default=None,
                   help="Subsample for quick sanity tests")
    p.add_argument("--no-pseudobulk", action="store_true",
                   help="Skip HCA pseudobulk; bulk only")
    p.add_argument("--accelerator", type=str, default="cpu",
                   help="cpu / mps / cuda")
    p.add_argument("--out-dir", type=str, default=OUT_DIR)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print("="*72)
    print(f"DRVI on bulk RNA-seq")
    print(f"  out_dir = {args.out_dir}")
    print(f"  cli args: {vars(args)}")
    print("="*72)

    # 1. Data
    adata = build_adata(
        use_pseudobulk=not args.no_pseudobulk,
        max_samples=args.max_samples,
    )

    # 2. Train
    model = train_drvi(
        adata,
        n_latent=args.n_latent,
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        encoder_dims=tuple(args.encoder_dims),
        decoder_dims=tuple(args.decoder_dims),
        dropout=args.dropout,
        accelerator=args.accelerator,
    )

    # 3. Save model + history
    model_dir = os.path.join(args.out_dir, "drvi_model")
    model.save(model_dir, overwrite=True)
    print(f"\nSaved model to {model_dir}")

    with open(os.path.join(args.out_dir, "drvi_history.pkl"), "wb") as f:
        pickle.dump(model.history, f)

    # 4. Plots
    plot_loss_curves(model.history, os.path.join(args.out_dir, "drvi_loss_curves.png"))
    plot_latent_tsne(model, adata, os.path.join(args.out_dir, "drvi_latent_tsne.png"))

    # 5. Final loss summary
    print("\n" + "=" * 72)
    print("FINAL METRICS")
    print("=" * 72)
    for k in ["elbo_train", "elbo_validation",
              "reconstruction_loss_train", "reconstruction_loss_validation",
              "kl_local_train", "kl_local_validation"]:
        if k in model.history:
            v = model.history[k].values.flatten()
            print(f"  {k:35s}  first={v[0]:10.3f}   last={v[-1]:10.3f}   "
                  f"Δ={v[-1] - v[0]:+10.3f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
