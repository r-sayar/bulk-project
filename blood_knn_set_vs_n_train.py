"""
How does the K=5 exact-K-NN-tuple gene-clustering evolve as we add
more train samples?

For each n_train ∈ {100, 200, 300, 400, 500, 600, 700, 753}:
    Build per-(gene, holdout) K-NN tuples on a uniformly-shuffled prefix
    of the training pool of size n_train. Group genes by exact tuple.
    Report mean / median / max set size, # multi-gene sets, and the
    random-uniform null mean set size.

Two competing forces:
  - Random-null shrinks 1/C(n, K), so collisions become rarer with more
    samples → mean set size drifts toward 1.0.
  - Biological co-expression survives: pathway-coherent genes still pick
    the same train donors as the pool grows, so the *largest* sets
    should plateau or grow rather than vanish.
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
K               = 5
N_TRAINS        = [100, 200, 300, 400, 500, 600, 700, 753]
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
print(f"  expressed: {n_genes:,} genes")

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


print(f"\n{'='*100}")
print(f"K={K} exact-tuple clustering as a function of n_train")
print(f"{'='*100}")
print(f"  {'n_train':>8}  {'mean':>7}  {'null':>9}  {'fold':>6}  "
      f"{'median':>7}  {'max':>5}  {'#multi':>7}  example")
print(f"  {'-' * 95}")

results = []
top_sets = []
for n_t in N_TRAINS:
    null_mean = 1 + (n_genes - 1) / comb(n_t, K, exact=True)

    means = []; maxes = []; meds = []; n_multi = []
    largest_examples = []
    for h in range(HOLDOUT):
        tuples = []
        for g in range(n_genes):
            d = np.abs(log_train[g, :n_t] - log_test[g, h])
            top = np.argpartition(d, K)[:K]
            tuples.append(tuple(np.sort(top)))
        c = Counter(tuples)
        gene_sizes = np.array([c[t] for t in tuples], dtype=np.int64)
        means.append(gene_sizes.mean())
        maxes.append(int(gene_sizes.max()))
        meds.append(int(np.median(gene_sizes)))
        n_multi.append(int((gene_sizes > 1).sum()))
        # Top sets
        size_to_genes = {}
        for g, t in enumerate(tuples):
            size_to_genes.setdefault(t, []).append(g)
        for t, gs in sorted(size_to_genes.items(), key=lambda kv: -len(kv[1]))[:3]:
            if len(gs) >= 3:
                largest_examples.append({
                    "n_train": n_t, "holdout": h,
                    "set_size": len(gs),
                    "genes_first10": ";".join(gene_names[gs[:10]]),
                })

    mean_mean = float(np.mean(means))
    median_max = float(np.median(maxes))
    multi_med = float(np.median(n_multi))
    if largest_examples:
        ex = max(largest_examples, key=lambda d: d["set_size"])
        ex_str = f"size {ex['set_size']}: {ex['genes_first10'][:55]}"
    else:
        ex_str = "—"
    print(f"  {n_t:>8}  {mean_mean:>7.4f}  {null_mean:>9.4f}  "
          f"{mean_mean / null_mean:>6.3f}  {np.median(meds):>7.1f}  "
          f"{median_max:>5.1f}  {multi_med:>7.1f}  {ex_str}")
    results.append({
        "n_train": n_t,
        "mean_set_size":     mean_mean,
        "null_mean":         null_mean,
        "fold_over_null":    mean_mean / null_mean,
        "median_max_set_size": median_max,
        "median_n_multi_sets": multi_med,
    })
    top_sets.extend(largest_examples)

res_df = pd.DataFrame(results)
res_df.to_csv(OUT_DIR / "knn_set_vs_n_train.csv", index=False)
pd.DataFrame(top_sets).to_csv(OUT_DIR / "knn_set_vs_n_train_examples.csv", index=False)


# ── Plot ─────────────────────────────────────────────────────────────
BG, CARD, TEXT, MUTED, GRID = '#0e1117', '#1a1d23', '#e6edf3', '#7d8590', '#21262d'
plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': CARD, 'axes.edgecolor': GRID,
    'axes.labelcolor': TEXT, 'text.color': TEXT, 'xtick.color': MUTED,
    'ytick.color': MUTED, 'grid.color': GRID, 'grid.alpha': 0.5,
    'font.family': 'sans-serif', 'font.size': 11,
})

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(f"Exact-K-NN-tuple clustering vs train pool size  (K={K})",
             color=TEXT, fontsize=13)

ax = axes[0]
ax.plot(res_df["n_train"], res_df["mean_set_size"], "o-",
        color="#3fb950", lw=2, label="observed")
ax.plot(res_df["n_train"], res_df["null_mean"], "s--",
        color="#f78166", lw=2, label="random null")
ax.set_xlabel("n_train")
ax.set_ylabel("mean set size (gene-weighted)")
ax.set_title("Mean set size vs n_train")
ax.legend(facecolor=CARD, edgecolor=GRID, labelcolor=TEXT, fontsize=9)

ax = axes[1]
ax.plot(res_df["n_train"], res_df["fold_over_null"], "o-",
        color="#58a6ff", lw=2)
ax.axhline(1.0, color="#f78166", ls=":", lw=1, alpha=0.7)
ax.set_xlabel("n_train")
ax.set_ylabel("fold over random null")
ax.set_title("Signal strength vs n_train")

ax = axes[2]
ax.plot(res_df["n_train"], res_df["median_max_set_size"], "o-",
        color="#d2a8ff", lw=2, label="median max set size")
ax.plot(res_df["n_train"], res_df["median_n_multi_sets"], "s--",
        color="#f0883e", lw=2, label="median # multi-gene sets")
ax.set_xlabel("n_train")
ax.set_ylabel("count")
ax.set_yscale("symlog")
ax.set_title("Largest sets and multi-gene-set count")
ax.legend(facecolor=CARD, edgecolor=GRID, labelcolor=TEXT, fontsize=9)

fig.tight_layout()
out_png = OUT_DIR / "blood_knn_set_vs_n_train.png"
fig.savefig(out_png, dpi=150, facecolor=BG, bbox_inches="tight")
plt.close(fig)
print(f"\nWrote {out_png}")
print(f"Wrote {OUT_DIR/'knn_set_vs_n_train.csv'}")
print(f"Wrote {OUT_DIR/'knn_set_vs_n_train_examples.csv'}")
