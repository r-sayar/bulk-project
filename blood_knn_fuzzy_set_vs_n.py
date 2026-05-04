"""
Same K=5 NN-tuple clustering, but with a SOFT threshold:
two genes are in the same "neighbour cluster" if they share at least M
out of K nearest train donors.

Sweep M ∈ {1, 2, 3, 4, 5} and n_train ∈ {100..753}.

For each (holdout, M, n_train):
    For each gene g, count genes g' with |NN(g) ∩ NN(g')| ≥ M.
    Mean of that count = average "soft set size" for g.

The exact match (M=K=5) shrinks toward 1.0 as n grows because the bucket
count C(n, K) blows up. Soft thresholds are robust because they tolerate
the "neighbour drift" caused by adding samples — when new donors are
closer to gene A than to gene B, A and B may share 4/5 donors instead
of 5/5, which the soft metric still counts.

Algorithm (efficient):
    Build inverse index: for each donor d, list of (gene, K-NN-tuple).
    For each gene g, iterate donors in NN(g), accumulate Counter of
    candidate genes. Each candidate's count = # shared donors with g.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings("ignore")

PROJECT = Path("/Users/rls/Desktop/programming-projects/single-cell/bulk-project")
GCT     = Path("/Users/rls/Downloads/gene_reads_v11_whole_blood.gct.gz")
OUT_DIR = PROJECT / "blood_technical_noise"

CPM_THRESHOLD   = 1.0
MIN_SAMPLE_FRAC = 0.10
HOLDOUT         = 50
K               = 5
N_TRAINS        = [100, 200, 300, 500, 753]
M_VALUES        = [1, 2, 3, 4, 5]
SEED            = 0


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

rng = np.random.default_rng(SEED)
perm = rng.permutation(n_s)
counts = counts[:, perm]
counts_train = counts[:, :n_s - HOLDOUT]
counts_test  = counts[:, n_s - HOLDOUT:]
lib_train = counts_train.sum(axis=0)
lib_test  = counts_test.sum(axis=0)
ref_lib   = float(np.median(lib_train))

counts_train_scaled = counts_train * (ref_lib / lib_train)[None, :]
counts_test_scaled  = counts_test  * (ref_lib / lib_test)[None, :]
log_train = np.log2(counts_train_scaled + 1)
log_test  = np.log2(counts_test_scaled  + 1)


def soft_cluster_sizes(n_t, h, K=K):
    """For one (n_train, holdout), returns array (n_genes, K+1)
    where col j = # other genes that share exactly j donors with g."""
    nn = np.zeros((n_genes, K), dtype=np.int32)
    for g in range(n_genes):
        d = np.abs(log_train[g, :n_t] - log_test[g, h])
        nn[g] = np.argpartition(d, K)[:K]

    # inverse index: donor -> list of gene indices
    inv = defaultdict(list)
    for g in range(n_genes):
        for d in nn[g]:
            inv[int(d)].append(g)

    # for each gene, count genes that share donors via Counter
    counts_at_m = np.zeros((n_genes, K + 1), dtype=np.int32)  # share 0..K
    for g in range(n_genes):
        c = Counter()
        for d in nn[g]:
            c.update(inv[d])
        # c[g] = K (gene g itself contributes K times). subtract.
        c[g] -= K
        if c[g] == 0:
            del c[g]
        # bucket by shared count
        for other, sh in c.items():
            counts_at_m[g, sh] += 1
    return counts_at_m, nn


print(f"\nFuzzy K-NN clustering: 'cluster size' = # of OTHER genes sharing "
      f">=M of K={K} nearest train donors")
print(f"\n{'='*90}")
print(f"  {'n_train':>8}  ", end="")
for m in M_VALUES:
    print(f"{'M=' + str(m):>10}  ", end="")
print(f"  {'M=K':>5}")
print("  " + "-" * 86)

results = []
for n_t in N_TRAINS:
    means_per_holdout = {m: [] for m in M_VALUES}
    medians_per_holdout = {m: [] for m in M_VALUES}
    for h in range(HOLDOUT):
        counts_at_m, _ = soft_cluster_sizes(n_t, h)
        # cumulative: # of genes with shared >= m
        for m in M_VALUES:
            sums = counts_at_m[:, m:].sum(axis=1)
            means_per_holdout[m].append(sums.mean())
            medians_per_holdout[m].append(np.median(sums))

    print(f"  {n_t:>8}  ", end="")
    row = {"n_train": n_t}
    for m in M_VALUES:
        mm = float(np.mean(means_per_holdout[m]))
        med = float(np.median(medians_per_holdout[m]))
        print(f"{mm:>10.2f}  ", end="")
        row[f"mean_M{m}"] = mm
        row[f"median_M{m}"] = med
    print()
    results.append(row)


res_df = pd.DataFrame(results)
res_df.to_csv(OUT_DIR / "knn_fuzzy_set_vs_n_train.csv", index=False)


# ── Plot ─────────────────────────────────────────────────────────────
BG, CARD, TEXT, MUTED, GRID = '#0e1117', '#1a1d23', '#e6edf3', '#7d8590', '#21262d'
plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': CARD, 'axes.edgecolor': GRID,
    'axes.labelcolor': TEXT, 'text.color': TEXT, 'xtick.color': MUTED,
    'ytick.color': MUTED, 'grid.color': GRID, 'grid.alpha': 0.5,
    'font.family': 'sans-serif', 'font.size': 11,
})

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Fuzzy K-NN gene-clustering vs train pool size  (K=5)",
             color=TEXT, fontsize=13)

ax = axes[0]
colors = ["#f78166", "#f0883e", "#3fb950", "#58a6ff", "#d2a8ff"]
for m, c in zip(M_VALUES, colors):
    ax.plot(res_df["n_train"], res_df[f"mean_M{m}"], "o-",
            color=c, lw=2,
            label=f"share ≥ {m}/{K} ({'exact' if m==K else 'fuzzy'})")
ax.set_xlabel("n_train")
ax.set_ylabel("mean cluster size (# OTHER genes sharing ≥M donors)")
ax.set_yscale("symlog")
ax.set_title("Mean cluster size by threshold")
ax.legend(facecolor=CARD, edgecolor=GRID, labelcolor=TEXT, fontsize=9)

ax = axes[1]
ax.plot(res_df["n_train"], res_df["mean_M5"], "o-",
        color="#f78166", lw=2, label="exact (M=K=5) — shrinks")
ax.plot(res_df["n_train"], res_df["mean_M3"] / res_df["mean_M3"].iloc[0], "s-",
        color="#3fb950", lw=2, label="M=3/5 normalized to n=100")
ax.plot(res_df["n_train"], res_df["mean_M2"] / res_df["mean_M2"].iloc[0], "d-",
        color="#58a6ff", lw=2, label="M=2/5 normalized to n=100")
ax.set_xlabel("n_train")
ax.set_ylabel("relative mean cluster size")
ax.set_yscale("log")
ax.set_title("Exact match shrinks; fuzzy match stable / grows")
ax.legend(facecolor=CARD, edgecolor=GRID, labelcolor=TEXT, fontsize=9)

fig.tight_layout()
out_png = OUT_DIR / "blood_knn_fuzzy_set_vs_n_train.png"
fig.savefig(out_png, dpi=150, facecolor=BG, bbox_inches="tight")
plt.close(fig)
print(f"\nWrote {out_png}")
print(f"Wrote {OUT_DIR/'knn_fuzzy_set_vs_n_train.csv'}")
