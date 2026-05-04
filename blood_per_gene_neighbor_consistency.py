"""
Per-(holdout, gene) nearest-train-donor sets, and their overlap across
genes within the same holdout.

Question:
    For holdout sample h and gene g, the K nearest train samples (by 1D
    distance on log2(scaled count + 1)) form a set NN(h, g) ⊆ {1..753}.

    Across the 16,355 genes of holdout h, are these sets consistently
    drawn from a small pool of donors (twin behaviour), or scattered
    across all 753 train donors (mosaic behaviour)?

Two complementary measures:
  (1) Donor-frequency distribution per holdout: for each train donor t,
      how many genes have t in their K-NN set? Skewed = twins exist.
  (2) Pairwise gene overlap: pick gene g, look at NN(h, g). For each
      other gene g' ∈ NN(h, g)... actually we want symmetric: the mean
      Jaccard between NN(h, g) and NN(h, g') across many gene pairs.

We sample 1,000 random gene pairs per holdout to estimate (2).

Outputs
    blood_technical_noise/
        per_holdout_neighbor_consistency.csv
        top_twin_donors_per_holdout.csv
        blood_per_gene_neighbor_consistency.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings("ignore")

PROJECT = Path("/Users/rls/Desktop/programming-projects/single-cell/bulk-project")
GCT     = Path("/Users/rls/Downloads/gene_reads_v11_whole_blood.gct.gz")
OUT_DIR = PROJECT / "blood_technical_noise"
OUT_DIR.mkdir(exist_ok=True)

CPM_THRESHOLD   = 1.0
MIN_SAMPLE_FRAC = 0.10
HOLDOUT         = 50
K               = 10
N_PAIRS_OVERLAP = 1000
SEED            = 0


print("Loading GTEx whole blood ...")
df = pd.read_csv(GCT, sep="\t", skiprows=2, compression="gzip")
gene_names = df["Description"].astype(str).values
counts = df.iloc[:, 2:].values.astype(np.float64)
n_g_raw, n_s = counts.shape
lib_all = counts.sum(axis=0)
cpm_all = counts / lib_all * 1e6
expressed = (cpm_all > CPM_THRESHOLD).sum(axis=1) >= int(MIN_SAMPLE_FRAC * n_s)
counts = counts[expressed]
gene_names = gene_names[expressed]
n_genes = counts.shape[0]
print(f"  expressed: {n_genes:,} genes")

rng = np.random.default_rng(SEED)
perm = rng.permutation(n_s)
counts = counts[:, perm]
counts_train = counts[:, :n_s - HOLDOUT]
counts_test  = counts[:, n_s - HOLDOUT:]
lib_train = counts_train.sum(axis=0)
lib_test  = counts_test.sum(axis=0)
n_train   = counts_train.shape[1]
ref_lib   = float(np.median(lib_train))

# log2(CPM+1) for distance metric (size-factor normalize first)
counts_train_scaled = counts_train * (ref_lib / lib_train)[None, :]
counts_test_scaled  = counts_test  * (ref_lib / lib_test)[None, :]
log_train = np.log2(counts_train_scaled + 1)
log_test  = np.log2(counts_test_scaled  + 1)


# ── 1. Per-(holdout, gene) K-NN train donors ─────────────────────────
# Vectorize per gene to limit memory.
print(f"\nComputing K={K} nearest train donors per (holdout, gene) ...")
nn_idx = np.zeros((n_genes, HOLDOUT, K), dtype=np.int32)
for g in range(n_genes):
    # train values 753, test values 50; distance |a - b|
    d = np.abs(log_train[g][None, :] - log_test[g][:, None])     # 50 x 753
    # argpartition for top-K smallest distances
    nn_idx[g] = np.argpartition(d, K, axis=1)[:, :K]
print("  done")


# ── 2. Per-holdout summaries ────────────────────────────────────────
print("\nComputing per-holdout twin / mosaic structure ...")
rows = []
twin_rows = []
overlap_rows = []
n_overlap_pairs = N_PAIRS_OVERLAP

for h in range(HOLDOUT):
    nn_h = nn_idx[:, h, :]                                       # n_genes × K
    # Donor-frequency distribution
    flat = nn_h.flatten()
    bc = np.bincount(flat, minlength=n_train)                    # 753
    distinct_donors = int((bc > 0).sum())
    top1_count   = int(bc.max())
    top1_id      = int(np.argmax(bc))
    top10_total  = int(np.sort(bc)[::-1][:10].sum())
    top1_frac    = top1_count / n_genes
    top10_frac   = top10_total / (n_genes * K)
    # Genes per donor: skewness
    nonzero_bc = bc[bc > 0]
    median_freq = float(np.median(nonzero_bc))
    p90_freq    = float(np.quantile(nonzero_bc, 0.90))

    # Pairwise gene overlap on random pairs
    rng_h = np.random.default_rng(SEED + h)
    pair_ids = rng_h.integers(0, n_genes, size=(n_overlap_pairs, 2))
    overlaps = []
    for a, b in pair_ids:
        if a == b: continue
        sa = set(nn_h[a]); sb = set(nn_h[b])
        overlaps.append(len(sa & sb) / K)
    overlaps = np.array(overlaps)
    mean_overlap   = float(overlaps.mean())
    median_overlap = float(np.median(overlaps))

    rows.append({
        "holdout":            h,
        "distinct_donors":    distinct_donors,
        "frac_donors_used":   distinct_donors / n_train,
        "top1_donor_id":      top1_id,
        "top1_gene_count":    top1_count,
        "top1_frac_of_genes": top1_frac,
        "top10_frac_of_NN_slots": top10_frac,
        "median_freq_of_used_donors": median_freq,
        "p90_freq_of_used_donors":    p90_freq,
        "mean_pairwise_overlap":  mean_overlap,
        "median_pairwise_overlap": median_overlap,
    })

    # Save top-20 most-frequent donors per holdout
    top_idx = np.argsort(bc)[::-1][:20]
    for rank, t in enumerate(top_idx):
        twin_rows.append({
            "holdout":  h,
            "rank":     rank + 1,
            "donor":    int(t),
            "gene_count": int(bc[t]),
            "frac_of_genes": bc[t] / n_genes,
        })

per_holdout_df = pd.DataFrame(rows)
per_holdout_df.to_csv(OUT_DIR / "per_holdout_neighbor_consistency.csv", index=False)
twin_df = pd.DataFrame(twin_rows)
twin_df.to_csv(OUT_DIR / "top_twin_donors_per_holdout.csv", index=False)


# ── 3. Reporting ──────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("PER-HOLDOUT TWIN STRUCTURE  (K=10)")
print(f"{'='*70}")
print(f"  distinct_donors / 753 used as a top-K match (mean): "
      f"{per_holdout_df['distinct_donors'].mean():.1f}  "
      f"(min {per_holdout_df['distinct_donors'].min()}, "
      f"max {per_holdout_df['distinct_donors'].max()})")
print(f"  fraction of train pool ever used as top-K (mean):   "
      f"{per_holdout_df['frac_donors_used'].mean()*100:.2f}%")
print(f"  top-1 donor's coverage of all genes (mean):          "
      f"{per_holdout_df['top1_frac_of_genes'].mean()*100:.2f}%  "
      f"(if random: {1/n_train * K * 100:.3f}% expected = K/n_train)")
print(f"  top-10 donors fill what frac of all NN slots (mean): "
      f"{per_holdout_df['top10_frac_of_NN_slots'].mean()*100:.2f}%")
print(f"  median pairwise gene-overlap of NN sets (mean):      "
      f"{per_holdout_df['median_pairwise_overlap'].mean():.4f}  "
      f"(if random: {K / n_train:.4f})")
print(f"  mean pairwise overlap (mean):                        "
      f"{per_holdout_df['mean_pairwise_overlap'].mean():.4f}")

# Sanity: random expectation of pairwise K-NN overlap when each pick is
# uniform from {1..753}: P(t in both) = K/n * K/n; expected overlap fraction
# = K^2 / (n_train * K) = K / n_train per slot → ~0.013
print(f"\n  random baseline pairwise overlap = K/n_train = {K/n_train:.4f}")


# ── 4. Visualization ─────────────────────────────────────────────────
BG, CARD, TEXT, MUTED, GRID = '#0e1117', '#1a1d23', '#e6edf3', '#7d8590', '#21262d'
plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': CARD, 'axes.edgecolor': GRID,
    'axes.labelcolor': TEXT, 'text.color': TEXT, 'xtick.color': MUTED,
    'ytick.color': MUTED, 'grid.color': GRID, 'grid.alpha': 0.5,
    'font.family': 'sans-serif', 'font.size': 11,
})

fig = plt.figure(figsize=(15, 11))
gs = GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)
fig.suptitle(f"K-NN train-donor consistency across genes within each holdout  "
             f"(K={K})", color=TEXT, fontsize=13)

ax = fig.add_subplot(gs[0, 0])
ax.hist(per_holdout_df["distinct_donors"], bins=20,
        color="#3fb950", edgecolor="#1f6f33", alpha=0.85)
ax.axvline(n_train, color="#f78166", ls="--", lw=1.5,
           label=f"max possible = {n_train}")
ax.set_xlabel("# distinct train donors used as top-K match")
ax.set_ylabel("# holdouts")
ax.set_title("How many train donors does each holdout 'use'?")
ax.legend(facecolor=CARD, edgecolor=GRID, labelcolor=TEXT, fontsize=8)

ax = fig.add_subplot(gs[0, 1])
ax.hist(per_holdout_df["top1_frac_of_genes"], bins=30,
        color="#58a6ff", edgecolor="#1f4e8f", alpha=0.85)
ax.axvline(K / n_train, color="#f78166", ls="--", lw=1.5,
           label=f"random: {K/n_train:.4f}")
ax.set_xlabel("fraction of genes for which the #1-twin is in top-K")
ax.set_ylabel("# holdouts")
ax.set_title("Strength of the single best twin")
ax.legend(facecolor=CARD, edgecolor=GRID, labelcolor=TEXT, fontsize=8)

ax = fig.add_subplot(gs[0, 2])
ax.hist(per_holdout_df["top10_frac_of_NN_slots"], bins=30,
        color="#d2a8ff", edgecolor="#7c5fbf", alpha=0.85)
ax.axvline(10 * K / (n_train * K), color="#f78166", ls="--", lw=1.5,
           label=f"random: {10/n_train:.4f}")
ax.set_xlabel("fraction of NN slots filled by top-10 twins")
ax.set_ylabel("# holdouts")
ax.set_title("Top-10 twin coverage")
ax.legend(facecolor=CARD, edgecolor=GRID, labelcolor=TEXT, fontsize=8)

ax = fig.add_subplot(gs[1, 0])
ax.hist(per_holdout_df["mean_pairwise_overlap"], bins=30,
        color="#f78166", edgecolor="#a13c1f", alpha=0.85,
        label="mean overlap")
ax.axvline(K / n_train, color="#3fb950", ls="--", lw=1.5,
           label=f"random: {K/n_train:.4f}")
ax.set_xlabel("mean pairwise NN-set overlap across genes")
ax.set_ylabel("# holdouts")
ax.set_title("Pairwise gene overlap (twin consistency)")
ax.legend(facecolor=CARD, edgecolor=GRID, labelcolor=TEXT, fontsize=8)

# Concentration plot for one example holdout
ax = fig.add_subplot(gs[1, 1])
h_pick = int(per_holdout_df["mean_pairwise_overlap"].idxmax())
nn_h = nn_idx[:, h_pick, :]
bc = np.bincount(nn_h.flatten(), minlength=n_train)
sorted_bc = np.sort(bc)[::-1]
cum = np.cumsum(sorted_bc) / (n_genes * K)
ax.plot(np.arange(1, n_train + 1), cum, color="#3fb950", lw=2)
ax.axhline(0.5, color="#f78166", ls=":", lw=1, alpha=0.7,
           label="50% NN-slots covered")
ax.axhline(0.9, color="#f0883e", ls=":", lw=1, alpha=0.7,
           label="90% NN-slots covered")
half = int(np.argmax(cum >= 0.5)) + 1
ninety = int(np.argmax(cum >= 0.9)) + 1
ax.set_xscale("log")
ax.set_xlabel("# top-N donors")
ax.set_ylabel("cumulative fraction of NN slots")
ax.set_title(f"Holdout {h_pick}: 50%@{half}, 90%@{ninety}")
ax.legend(facecolor=CARD, edgecolor=GRID, labelcolor=TEXT, fontsize=8)

# Concentration curve averaged across holdouts
ax = fig.add_subplot(gs[1, 2])
all_cum = np.zeros((HOLDOUT, n_train))
for h in range(HOLDOUT):
    nn_h = nn_idx[:, h, :]
    bc = np.bincount(nn_h.flatten(), minlength=n_train)
    sorted_bc = np.sort(bc)[::-1]
    all_cum[h] = np.cumsum(sorted_bc) / (n_genes * K)
mean_curve   = all_cum.mean(axis=0)
median_curve = np.median(all_cum, axis=0)
ax.plot(np.arange(1, n_train + 1), mean_curve, color="#3fb950", lw=2,
        label="mean")
ax.plot(np.arange(1, n_train + 1), median_curve, color="#58a6ff", lw=1.5,
        ls="--", label="median")
# Random uniform baseline = (K * N_top) / (n_train * K) = N_top / n_train
ax.plot(np.arange(1, n_train + 1),
        np.arange(1, n_train + 1) / n_train,
        color="#f78166", lw=1.5, ls=":", label="random (uniform)")
ax.set_xscale("log")
ax.set_xlabel("# top-N donors")
ax.set_ylabel("cumulative fraction of NN slots")
ax.set_title("Concentration curve (averaged across holdouts)")
ax.legend(facecolor=CARD, edgecolor=GRID, labelcolor=TEXT, fontsize=8)

# Show top-20 twins for the most-twin-like holdout
ax = fig.add_subplot(gs[2, 0])
sub = twin_df[twin_df["holdout"] == h_pick].sort_values("gene_count", ascending=False).head(20)
ax.barh(np.arange(len(sub))[::-1], sub["frac_of_genes"][::-1],
        color="#f78166")
ax.set_yticks(np.arange(len(sub))[::-1])
ax.set_yticklabels([f"donor {int(d)}" for d in sub["donor"][::-1]], fontsize=8)
ax.set_xlabel("fraction of genes")
ax.set_title(f"Top-20 twins for holdout {h_pick}")

# Heatmap: holdout vs train donor, frac of genes covered
ax = fig.add_subplot(gs[2, 1])
mat = np.zeros((HOLDOUT, n_train))
for h in range(HOLDOUT):
    nn_h = nn_idx[:, h, :]
    bc = np.bincount(nn_h.flatten(), minlength=n_train)
    mat[h] = bc / n_genes
# Sort columns by max coverage to focus on heavy hitters
col_order = np.argsort(mat.max(axis=0))[::-1]
im = ax.imshow(mat[:, col_order[:50]], aspect="auto", cmap="magma",
               interpolation="nearest")
plt.colorbar(im, ax=ax, label="frac genes covered")
ax.set_xlabel("train donor (top 50 by max coverage)")
ax.set_ylabel("holdout idx")
ax.set_title("Donor-coverage heatmap")

# Distribution of pairwise overlaps for one example
ax = fig.add_subplot(gs[2, 2])
nn_h = nn_idx[:, h_pick, :]
rng_h = np.random.default_rng(SEED + h_pick)
pair_ids = rng_h.integers(0, n_genes, size=(5000, 2))
pos = []
for a, b in pair_ids:
    if a == b: continue
    sa = set(nn_h[a]); sb = set(nn_h[b])
    pos.append(len(sa & sb) / K)
ax.hist(pos, bins=21, color="#d2a8ff", edgecolor="#7c5fbf")
ax.axvline(K / n_train, color="#f78166", ls="--", lw=1.5,
           label=f"random {K/n_train:.4f}")
ax.set_xlabel("pairwise NN overlap fraction")
ax.set_ylabel("# pairs")
ax.set_title(f"Pairwise overlap distribution (holdout {h_pick})")
ax.legend(facecolor=CARD, edgecolor=GRID, labelcolor=TEXT, fontsize=8)

fig.tight_layout()
out_png = OUT_DIR / "blood_per_gene_neighbor_consistency.png"
fig.savefig(out_png, dpi=150, facecolor=BG, bbox_inches="tight")
plt.close(fig)
print(f"\nWrote {out_png}")
print(f"Wrote {OUT_DIR/'per_holdout_neighbor_consistency.csv'}")
print(f"Wrote {OUT_DIR/'top_twin_donors_per_holdout.csv'}")
