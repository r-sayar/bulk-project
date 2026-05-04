"""
Per-holdout: cluster genes by their EXACT K-NN tuple.

For each (holdout h, gene g) we have the sorted tuple of K nearest train
donors NN(h, g). Two genes belong to the same "set" iff they have the
identical K-NN tuple. With K nearest neighbors, exact-match clustering
gives n_unique_tuples sets — and the set sizes show whether multiple
genes share the same train-donor bucket.

A gene-weighted mean is the natural metric: for each gene, its "set size"
is the number of OTHER genes (including itself) with the same K-NN
tuple. The grand mean across all genes and holdouts answers
    "on average, how many of the 16,355 genes share a holdout's K nearest
     train donors with you?"

Sweep K ∈ {1, 2, 3, 5, 7, 10} as requested. Larger mean set size at low
K is expected (fewer slots to match); the question is whether the
*absolute* numbers are above the random-uniform null and whether they
plateau or continue to fall as K grows.

Random-uniform null:
    Probability that two random K-tuples from n_train donors match
    exactly = 1 / C(n_train, K). With n_train=753 and 16,355 genes, the
    expected gene-weighted mean set size under the null is
    1 + (n_genes - 1) / C(n_train, K).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from scipy.special import comb
import warnings
warnings.filterwarnings("ignore")

PROJECT = Path("/Users/rls/Desktop/programming-projects/single-cell/bulk-project")
GCT     = Path("/Users/rls/Downloads/gene_reads_v11_whole_blood.gct.gz")
OUT_DIR = PROJECT / "blood_technical_noise"

CPM_THRESHOLD   = 1.0
MIN_SAMPLE_FRAC = 0.10
HOLDOUT         = 50
KS              = [1, 2, 3, 5, 7, 10]
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
n_train   = counts_train.shape[1]
ref_lib   = float(np.median(lib_train))

counts_train_scaled = counts_train * (ref_lib / lib_train)[None, :]
counts_test_scaled  = counts_test  * (ref_lib / lib_test)[None, :]
log_train = np.log2(counts_train_scaled + 1)
log_test  = np.log2(counts_test_scaled  + 1)


def topk_indices(g, h, k):
    """K nearest train donors for (gene g, holdout h)."""
    d = np.abs(log_train[g] - log_test[g, h])
    return np.argpartition(d, k)[:k]


print("\nSweeping K ...")
print(f"  K  : gene-weighted mean set size  (random null)   "
      f"max set size  median  # multi-gene sets  largest-set example")
print("  " + "-" * 100)

results = []
big_sets_examples = {}     # K -> list of largest sets across holdouts

for k in KS:
    # null expectation
    n_buckets = comb(n_train, k, exact=True)
    null_mean = 1 + (n_genes - 1) / n_buckets

    per_holdout_means   = []
    per_holdout_max     = []
    per_holdout_median  = []
    per_holdout_n_multi = []
    largest_sets_for_k  = []

    for h in range(HOLDOUT):
        # Compute K-NN tuple for each gene (sort to canonicalize)
        tuples = []
        for g in range(n_genes):
            d = np.abs(log_train[g] - log_test[g, h])
            top = np.argpartition(d, k)[:k]
            tuples.append(tuple(np.sort(top)))

        c = Counter(tuples)
        gene_sizes = np.array([c[t] for t in tuples], dtype=np.int64)
        per_holdout_means.append(gene_sizes.mean())
        per_holdout_max.append(gene_sizes.max())
        per_holdout_median.append(np.median(gene_sizes))
        per_holdout_n_multi.append(int((gene_sizes > 1).sum()))

        # capture the largest sets (size > 2) — examples for the report
        size_to_genes = {}
        for g, t in enumerate(tuples):
            size_to_genes.setdefault(t, []).append(g)
        # take top by size
        big = sorted(size_to_genes.items(), key=lambda kv: -len(kv[1]))[:5]
        for t, gs in big:
            if len(gs) >= 3:
                largest_sets_for_k.append({
                    "holdout": h, "K": k,
                    "set_size": len(gs),
                    "tuple_first5": str(t[:5]),
                    "genes_first10": ";".join(gene_names[gs[:10]]),
                })

    big_sets_examples[k] = largest_sets_for_k
    mean_mean = float(np.mean(per_holdout_means))
    median_mean = float(np.median(per_holdout_means))
    max_max = int(np.max(per_holdout_max))
    median_max = float(np.median(per_holdout_max))
    n_multi_med = float(np.median(per_holdout_n_multi))

    # An example largest-set across the 50 holdouts
    example = ""
    if largest_sets_for_k:
        biggest = max(largest_sets_for_k, key=lambda d: d["set_size"])
        example = f"size {biggest['set_size']}: {biggest['genes_first10'][:60]}"

    print(f"  {k:>2}  : {mean_mean:>22.3f}    ({null_mean:.4f})    "
          f"{median_max:>10.1f}    {np.median(per_holdout_median):>4.1f}    "
          f"{n_multi_med:>15.1f}    {example}")
    results.append({
        "K": k,
        "n_buckets": int(n_buckets) if n_buckets < 1e15 else float(n_buckets),
        "null_mean_set_size": null_mean,
        "obs_mean_set_size":  mean_mean,
        "obs_median_set_size": median_mean,
        "obs_max_set_size_median_over_holdouts": median_max,
        "obs_max_set_size_global":  max_max,
        "obs_n_multi_sets_median": n_multi_med,
        "fold_over_null":     mean_mean / null_mean,
    })

results_df = pd.DataFrame(results)
results_df.to_csv(OUT_DIR / "knn_exact_set_clustering_summary.csv", index=False)


# Save the largest-set example details for K=5
all_examples = []
for k, examples in big_sets_examples.items():
    for ex in examples:
        all_examples.append(ex)
pd.DataFrame(all_examples).to_csv(
    OUT_DIR / "knn_exact_set_largest_examples.csv", index=False
)


# ── Plot ─────────────────────────────────────────────────────────────
BG, CARD, TEXT, MUTED, GRID = '#0e1117', '#1a1d23', '#e6edf3', '#7d8590', '#21262d'
plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': CARD, 'axes.edgecolor': GRID,
    'axes.labelcolor': TEXT, 'text.color': TEXT, 'xtick.color': MUTED,
    'ytick.color': MUTED, 'grid.color': GRID, 'grid.alpha': 0.5,
    'font.family': 'sans-serif', 'font.size': 11,
})

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle("Exact-K-NN-tuple gene clustering: do genes share train-donor sets?",
             color=TEXT, fontsize=13)

ax = axes[0, 0]
ax.plot(results_df["K"], results_df["obs_mean_set_size"], "o-",
        color="#3fb950", lw=2, label="observed (gene-weighted)")
ax.plot(results_df["K"], results_df["null_mean_set_size"], "s--",
        color="#f78166", lw=2, label="random-uniform null")
ax.set_yscale("log")
ax.set_xlabel("K (# nearest train donors)")
ax.set_ylabel("mean set size")
ax.set_title("Mean set size vs K")
ax.legend(facecolor=CARD, edgecolor=GRID, labelcolor=TEXT, fontsize=9)

ax = axes[0, 1]
ax.plot(results_df["K"], results_df["fold_over_null"], "o-",
        color="#58a6ff", lw=2)
ax.set_yscale("log")
ax.set_xlabel("K")
ax.set_ylabel("fold over random null")
ax.set_title("Signal strength vs K")

ax = axes[1, 0]
ax.plot(results_df["K"], results_df["obs_max_set_size_median_over_holdouts"], "o-",
        color="#d2a8ff", lw=2, label="median across holdouts")
ax.plot(results_df["K"], results_df["obs_max_set_size_global"], "s--",
        color="#f0883e", lw=2, label="global max")
ax.set_xlabel("K")
ax.set_ylabel("largest set size")
ax.set_title("Largest set size per holdout")
ax.legend(facecolor=CARD, edgecolor=GRID, labelcolor=TEXT, fontsize=9)

ax = axes[1, 1]
ax.plot(results_df["K"], results_df["obs_n_multi_sets_median"], "o-",
        color="#f78166", lw=2)
ax.set_xlabel("K")
ax.set_ylabel("# multi-gene sets (median across holdouts)")
ax.set_title("Number of genes in any non-singleton set")

fig.tight_layout()
out_png = OUT_DIR / "blood_knn_set_clustering.png"
fig.savefig(out_png, dpi=150, facecolor=BG, bbox_inches="tight")
plt.close(fig)
print(f"\nWrote {out_png}")
print(f"Wrote {OUT_DIR/'knn_exact_set_clustering_summary.csv'}")
print(f"Wrote {OUT_DIR/'knn_exact_set_largest_examples.csv'}")


# Print top largest sets at K=5 (the user's preferred K)
print("\nTop 30 largest sets across all 50 holdouts at K=5:")
k5 = pd.DataFrame(big_sets_examples[5]).sort_values("set_size", ascending=False).head(30)
print(k5[["holdout", "set_size", "genes_first10"]].to_string(index=False))
