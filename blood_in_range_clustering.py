"""
"In-range neighbours" instead of K nearest neighbours.

For each (holdout h, gene g), define the IN-RANGE train donors as
    inrange(h, g) = { s :  |x_{s,g} − y_{h,g}|  ≤  K_SD · σ_tech(x_{s,g}) }
where σ_tech(x) = sqrt(x + (α x)²) with α = 0.14.

VARIABLE size per (h, g) — abundant genes near a typical donor's value
have many in-range donors, bimodal/extreme values have few.

Cluster gene-pairs by SHARED in-range donor identity:
    For genes a, b: shared(a, b) = |inrange(h, a) ∩ inrange(h, b)|.

Per-holdout summaries:
  • mean # in-range donors per gene
  • mean # OTHER genes sharing ≥ M in-range donors with each gene,
    for M ∈ {5, 10, 20, 50}.

Sweep n_train ∈ {100, 200, 300, 500, 753}.

Implementation: A is a (n_genes, n_train) bool matrix; A·A.T computed
in row-chunks gives a (n_genes, n_genes) int matrix of shared counts;
threshold and row-sum to get cluster sizes.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings("ignore")

PROJECT = Path("/Users/rls/Desktop/programming-projects/single-cell/bulk-project")
GCT     = Path("/Users/rls/Downloads/gene_reads_v11_whole_blood.gct.gz")
OUT_DIR = PROJECT / "blood_technical_noise"

CPM_THRESHOLD   = 1.0
MIN_SAMPLE_FRAC = 0.10
HOLDOUT         = 10        # subsample holdouts (still 10 of 50 — robust enough for trend)
ALPHA           = 0.14
K_SD            = 2.0
N_TRAINS        = [100, 200, 300, 500, 753]
M_VALUES        = [5, 10, 20, 50]
SEED            = 0
ROW_CHUNK       = 2000


def sigma_tech(x):
    return np.sqrt(x + (ALPHA * x) ** 2)


print("Loading ...")
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
counts_train_full = counts[:, :n_s - HOLDOUT]
counts_test       = counts[:, n_s - HOLDOUT:]
lib_train_full = counts_train_full.sum(axis=0)
lib_test = counts_test.sum(axis=0)
ref_lib = float(np.median(lib_train_full))

sd_raw_full = sigma_tech(counts_train_full)
scale_full  = ref_lib / lib_train_full
lower_full  = np.maximum((counts_train_full - K_SD * sd_raw_full) * scale_full[None, :], 0.0)
upper_full  = (counts_train_full + K_SD * sd_raw_full) * scale_full[None, :]
counts_test_scaled = counts_test * (ref_lib / lib_test)[None, :]


def cluster_sizes_for_holdout(lower, upper, y, M_values, row_chunk=ROW_CHUNK):
    """Compute # of OTHER genes sharing >=M in-range donors with each gene."""
    in_range = (y[:, None] >= lower) & (y[:, None] <= upper)
    A = in_range.astype(np.float32)        # float32 → BLAS-accelerated matmul
    n_g = A.shape[0]
    n_in_range = A.sum(axis=1)

    out = {m: np.zeros(n_g, dtype=np.int32) for m in M_values}
    for i in range(0, n_g, row_chunk):
        # block: chunk x n_genes integer (shared donor counts)
        block = A[i : i + row_chunk] @ A.T
        for m in M_values:
            row_counts = (block >= m).sum(axis=1) - 1  # exclude self
            out[m][i : i + row_chunk] = np.maximum(row_counts, 0)
    return out, n_in_range


print(f"\n{'='*100}")
print(f"In-range donors at K_SD={K_SD} (variable per gene)  vs n_train")
print(f"{'='*100}")
header = f"  {'n_train':>8}  {'mean_n_in':>10}  {'med_n_in':>9}"
for m in M_VALUES:
    header += f"  {'≥' + str(m):>10}"
print(header)
print("  " + "-" * 100)

results = []
example_clusters = []
for n_t in N_TRAINS:
    print(f"  computing n_train={n_t} ...", end="", flush=True)
    t0 = time.time()
    lower = lower_full[:, :n_t]
    upper = upper_full[:, :n_t]

    means_inrange = []
    medians_inrange = []
    means_shared = {m: [] for m in M_VALUES}
    medians_shared = {m: [] for m in M_VALUES}
    biggest_per_holdout = {m: [] for m in M_VALUES}

    for h in range(HOLDOUT):
        out, n_in_range = cluster_sizes_for_holdout(
            lower, upper, counts_test_scaled[:, h], M_VALUES,
            row_chunk=ROW_CHUNK,
        )
        means_inrange.append(n_in_range.mean())
        medians_inrange.append(int(np.median(n_in_range)))
        for m in M_VALUES:
            means_shared[m].append(out[m].mean())
            medians_shared[m].append(int(np.median(out[m])))
            biggest_per_holdout[m].append(int(out[m].max()))

    elapsed = time.time() - t0
    print(f"  done in {elapsed:.0f}s")
    print(f"    mean #in-range/gene: {np.mean(means_inrange):.1f}  "
          f"(median {np.median(medians_inrange):.0f})")
    for m in M_VALUES:
        print(f"    cluster size at M={m:>2}: mean={np.mean(means_shared[m]):.1f}  "
              f"median={np.median(medians_shared[m]):.0f}  "
              f"max-per-holdout median={np.median(biggest_per_holdout[m]):.0f}")

    row = {
        "n_train": n_t,
        "mean_n_in_range": float(np.mean(means_inrange)),
        "median_n_in_range": float(np.median(medians_inrange)),
    }
    for m in M_VALUES:
        row[f"mean_shared_M{m}"]   = float(np.mean(means_shared[m]))
        row[f"median_shared_M{m}"] = float(np.median(medians_shared[m]))
        row[f"max_shared_M{m}"]    = int(np.median(biggest_per_holdout[m]))
    results.append(row)


res_df = pd.DataFrame(results)
res_df.to_csv(OUT_DIR / "in_range_cluster_vs_n.csv", index=False)


# ── Plot ─────────────────────────────────────────────────────────────
BG, CARD, TEXT, MUTED, GRID = '#0e1117', '#1a1d23', '#e6edf3', '#7d8590', '#21262d'
plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': CARD, 'axes.edgecolor': GRID,
    'axes.labelcolor': TEXT, 'text.color': TEXT, 'xtick.color': MUTED,
    'ytick.color': MUTED, 'grid.color': GRID, 'grid.alpha': 0.5,
    'font.family': 'sans-serif', 'font.size': 11,
})

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(f"In-range gene clustering (band ±{K_SD:.0f}σ_tech) vs train pool size",
             color=TEXT, fontsize=13)

ax = axes[0]
ax.plot(res_df["n_train"], res_df["mean_n_in_range"], "o-",
        color="#3fb950", lw=2, label="mean")
ax.plot(res_df["n_train"], res_df["median_n_in_range"], "s--",
        color="#58a6ff", lw=2, label="median")
ax.plot(res_df["n_train"], 0.30 * res_df["n_train"], color="#f78166",
        ls=":", lw=1, label="0.30 × n_train")
ax.set_xlabel("n_train")
ax.set_ylabel("# in-range donors per gene")
ax.set_title("In-range support grows linearly with n")
ax.legend(facecolor=CARD, edgecolor=GRID, labelcolor=TEXT, fontsize=9)

ax = axes[1]
colors = ["#f78166", "#f0883e", "#3fb950", "#58a6ff"]
for m, c in zip(M_VALUES, colors):
    ax.plot(res_df["n_train"], res_df[f"mean_shared_M{m}"], "o-",
            color=c, lw=2, label=f"≥ {m} shared")
ax.set_xlabel("n_train")
ax.set_ylabel("mean cluster size (# OTHER genes sharing ≥M)")
ax.set_yscale("symlog")
ax.set_title("Cluster size grows with n")
ax.legend(facecolor=CARD, edgecolor=GRID, labelcolor=TEXT, fontsize=9)

ax = axes[2]
for m, c in zip(M_VALUES, colors):
    ax.plot(res_df["n_train"], res_df[f"max_shared_M{m}"], "o-",
            color=c, lw=2, label=f"≥ {m} shared")
ax.set_xlabel("n_train")
ax.set_ylabel("median (over holdouts) of max cluster size per holdout")
ax.set_yscale("symlog")
ax.set_title("Largest cluster grows with n")
ax.legend(facecolor=CARD, edgecolor=GRID, labelcolor=TEXT, fontsize=9)

fig.tight_layout()
out_png = OUT_DIR / "blood_in_range_clustering_vs_n.png"
fig.savefig(out_png, dpi=150, facecolor=BG, bbox_inches="tight")
plt.close(fig)
print(f"\nWrote {out_png}")
print(f"Wrote {OUT_DIR/'in_range_cluster_vs_n.csv'}")
