"""
Per-gene neighbor-popularity inside a single holdout.

For holdout h:
  1. For every gene g, find the set of "in-range neighbours":
        N_g(h) = { train donor d :
                   |x_{d,g} - y_{h,g}|  ≤  2 · σ_tech(x_{d,g}) }
  2. For every train donor d, count its overall popularity for this
     holdout — i.e. across how many genes d appears as an in-range
     neighbour:
        pop(d, h) = | { g : d ∈ N_g(h) } |
                  = column-sum of the in-range bool matrix A.
  3. For every gene g, summarise the popularity of its neighbours:
        mean_pop(g)   = mean_{d ∈ N_g}( pop(d) − 1[d covers g itself] )
        median_pop(g) = same with median
     and a SCALE-INVARIANT specificity:
        spec(g) = 1 - mean_pop(g) / n_genes
                  high → neighbours are exclusive to g (informative)
                  low  → neighbours cover everything (uninformative)

Step-by-step in the code below — read the comments.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

PROJECT = Path("/Users/rls/Desktop/programming-projects/single-cell/bulk-project")
GCT     = Path("/Users/rls/Downloads/gene_reads_v11_whole_blood.gct.gz")
OUT_DIR = PROJECT / "blood_technical_noise"

CPM_THRESHOLD   = 1.0
MIN_SAMPLE_FRAC = 0.10
HOLDOUT         = 50
ALPHA           = 0.14
K_SD            = 2.0
SEED            = 0


def sigma_tech(x):
    return np.sqrt(x + (ALPHA * x) ** 2)


# ── 1.  Load + filter (same as the other scripts) ─────────────────────
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
print(f"  expressed genes: {n_genes:,}")

# ── 2.  Same train / holdout split as the saturation analyses ─────────
rng = np.random.default_rng(SEED)
perm = rng.permutation(n_s)
counts = counts[:, perm]
counts_train = counts[:, :n_s - HOLDOUT]
counts_test  = counts[:, n_s - HOLDOUT:]
lib_train = counts_train.sum(axis=0)
lib_test  = counts_test.sum(axis=0)
n_train   = counts_train.shape[1]
ref_lib   = float(np.median(lib_train))
print(f"  train: {n_train}, holdout: {HOLDOUT}, ref lib: {ref_lib:,.0f}")

# Per-donor noise band on RAW counts, scaled to ref library.
# (lower, upper) is shape (n_genes, n_train).
sd_raw  = sigma_tech(counts_train)
scale   = ref_lib / lib_train
lower   = np.maximum((counts_train - K_SD * sd_raw) * scale[None, :], 0.0)
upper   = (counts_train + K_SD * sd_raw) * scale[None, :]
y_test  = counts_test * (ref_lib / lib_test)[None, :]


# ── 3.  Pick one holdout to walk through, then aggregate over all 50 ──
def per_gene_neighbor_pop(h):
    """Returns:
        A: bool (n_genes, n_train)  in-range mask
        pop_d: int (n_train,)       # genes each donor is in-range for
        n_neighbors: int (n_genes,) # donors in-range for each gene
        mean_pop: float (n_genes,)  # mean pop of g's neighbours, excl. g
        median_pop: float (n_genes,)
    """
    y = y_test[:, h]
    # 3a. In-range bool matrix:
    #     A[g, d] = True  iff donor d's noise band contains the holdout
    #              value for gene g.
    A = (y[:, None] >= lower) & (y[:, None] <= upper)        # (n_genes, n_train)

    # 3b. Donor popularity for THIS holdout: column-sum of A.
    #     pop_d[d] = number of genes where donor d is in-range.
    pop_d = A.sum(axis=0).astype(np.int32)                   # (n_train,)

    # 3c. # neighbours each gene has for this holdout: row-sum of A.
    n_neighbors = A.sum(axis=1).astype(np.int32)             # (n_genes,)

    # 3d. For each gene g, the popularity vector of its neighbours is
    #     pop_d masked by A[g]. We want mean and median of that
    #     vector, *excluding the contribution from gene g itself*
    #     (every donor that's in-range for g counts gene g once in
    #     pop_d, so the "popularity excluding g" is pop_d − 1 for any
    #     d that's actually in N_g).
    #     We compute via:
    #       sum_pop[g]  = sum over d of A[g, d] * (pop_d[d] - 1)
    #       mean_pop[g] = sum_pop[g] / n_neighbors[g]
    sum_pop = A.astype(np.int64) @ (pop_d - 1)               # (n_genes,)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_pop = np.where(n_neighbors > 0,
                             sum_pop / n_neighbors, 0.0)

    # Median is harder to vectorise, so do it per-gene with a small
    # for-loop. (16k iterations of np.median on ≤753 ints — fast.)
    median_pop = np.zeros(n_genes, dtype=np.float64)
    pop_minus_1 = (pop_d - 1).astype(np.int32)
    for g in range(n_genes):
        idx = np.flatnonzero(A[g])
        if len(idx) == 0:
            continue
        median_pop[g] = float(np.median(pop_minus_1[idx]))

    return A, pop_d, n_neighbors, mean_pop, median_pop


# Walk through one holdout
print("\nWalking through holdout 0 first ...")
A0, pop0, nn0, mean_pop0, median_pop0 = per_gene_neighbor_pop(0)
print(f"  A.shape:           {A0.shape}  (n_genes × n_train)")
print(f"  donors:            {n_train}")
print(f"  pop_d (donor → # genes covered):")
print(f"     min  = {pop0.min():>6d}     # gene-coverage of the LEAST-popular donor")
print(f"     mean = {pop0.mean():>9.1f}")
print(f"     med  = {int(np.median(pop0)):>6d}")
print(f"     max  = {pop0.max():>6d}     # gene-coverage of the MOST-popular twin donor")
print(f"  n_neighbors (gene → # donors in-range):")
print(f"     min  = {nn0.min():>6d}")
print(f"     mean = {nn0.mean():>9.1f}     ≈ 32 % of {n_train}")
print(f"     max  = {nn0.max():>6d}")
print(f"  mean_pop (gene → avg popularity of its neighbours):")
print(f"     min  = {mean_pop0.min():>9.1f}     # gene with most-exclusive neighbour set")
print(f"     mean = {mean_pop0.mean():>9.1f}")
print(f"     max  = {mean_pop0.max():>9.1f}     # gene with most-generic neighbour set")


# Aggregate across all 50 holdouts
print("\nAggregating across all 50 holdouts ...")
mean_pop_all   = np.zeros((HOLDOUT, n_genes))
median_pop_all = np.zeros((HOLDOUT, n_genes))
nn_all         = np.zeros((HOLDOUT, n_genes), dtype=np.int32)
pop_d_all      = np.zeros((HOLDOUT, n_train), dtype=np.int32)

for h in range(HOLDOUT):
    A, pop_d, nn, mp, medp = per_gene_neighbor_pop(h)
    mean_pop_all[h]   = mp
    median_pop_all[h] = medp
    nn_all[h]         = nn
    pop_d_all[h]      = pop_d

# Per-gene mean popularity across the 50 holdouts (one number per gene)
gene_mean_pop      = mean_pop_all.mean(axis=0)
gene_median_pop    = median_pop_all.mean(axis=0)
gene_n_neighbors   = nn_all.mean(axis=0)
# Specificity: 1 - mean_pop / n_genes  (higher = more exclusive)
specificity = 1 - gene_mean_pop / n_genes

# Save table
out = pd.DataFrame({
    "gene":               gene_names,
    "mean_n_neighbors":   gene_n_neighbors,
    "mean_neighbor_popularity":   gene_mean_pop,
    "median_neighbor_popularity": gene_median_pop,
    "specificity":        specificity,
})
out.sort_values("specificity", ascending=False) \
   .to_csv(OUT_DIR / "neighbor_popularity_per_gene.csv", index=False)

print(f"\n  per-gene mean(neighbor popularity):")
print(f"    mean   = {gene_mean_pop.mean():>10.1f}")
print(f"    median = {np.median(gene_mean_pop):>10.1f}")
print(f"    min    = {gene_mean_pop.min():>10.1f}")
print(f"    max    = {gene_mean_pop.max():>10.1f}")

print(f"\n  per-gene specificity (1 - mean_pop / n_genes; HIGH = exclusive):")
print(f"    mean   = {specificity.mean():>9.4f}")
print(f"    median = {np.median(specificity):>9.4f}")
print(f"    p90    = {np.quantile(specificity, 0.90):>9.4f}")
print(f"    p99    = {np.quantile(specificity, 0.99):>9.4f}")

# Top "specific" genes — those whose neighbours are NOT shared with most others.
print("\nTOP 25 most specific genes (small, exclusive neighbour set):")
top_spec = out.sort_values("specificity", ascending=False).head(25)
print(top_spec.to_string(index=False))

print("\nBOTTOM 15 least specific (everyone-is-a-neighbour bookkeeping genes):")
bot_spec = out.sort_values("specificity", ascending=True).head(15)
print(bot_spec.to_string(index=False))


# ── 4.  Visualisation ────────────────────────────────────────────────
BG, CARD, TEXT, MUTED, GRID = '#0e1117', '#1a1d23', '#e6edf3', '#7d8590', '#21262d'
plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': CARD, 'axes.edgecolor': GRID,
    'axes.labelcolor': TEXT, 'text.color': TEXT, 'xtick.color': MUTED,
    'ytick.color': MUTED, 'grid.color': GRID, 'grid.alpha': 0.5,
    'font.family': 'sans-serif', 'font.size': 11,
})

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Per-gene neighbour popularity (in-range donors as 'neighbours')",
             color=TEXT, fontsize=13)

ax = axes[0, 0]
ax.hist(gene_n_neighbors, bins=60, color="#3fb950", edgecolor="#1f6f33")
ax.set_xlabel("mean # in-range donors per gene (across 50 holdouts)")
ax.set_ylabel("# genes")
ax.set_title("How many neighbours does each gene have?")

ax = axes[0, 1]
ax.hist(gene_mean_pop, bins=60, color="#58a6ff", edgecolor="#1f4e8f")
ax.axvline(n_genes, color="#f78166", ls="--", lw=1.5,
           label=f"n_genes = {n_genes}")
ax.set_xlabel("mean popularity of g's neighbours (avg # genes those donors cover)")
ax.set_ylabel("# genes")
ax.set_title("How popular are the neighbours?")
ax.legend(facecolor=CARD, edgecolor=GRID, labelcolor=TEXT, fontsize=9)

ax = axes[1, 0]
ax.hist(specificity, bins=60, color="#d2a8ff", edgecolor="#7c5fbf")
ax.set_xlabel("specificity = 1 - mean_pop / n_genes")
ax.set_ylabel("# genes")
ax.set_yscale("log")
ax.set_title("Specificity distribution (high = exclusive neighbour set)")

ax = axes[1, 1]
# Joint: # neighbours vs specificity
hb = ax.hexbin(gene_n_neighbors, specificity,
               gridsize=50, mincnt=1, cmap="magma", bins="log")
ax.set_xlabel("mean # neighbours per gene")
ax.set_ylabel("specificity")
ax.set_title("Joint: how the two things relate")
plt.colorbar(hb, ax=ax, label="log10 # genes")

fig.tight_layout()
out_png = OUT_DIR / "blood_neighbor_popularity.png"
fig.savefig(out_png, dpi=150, facecolor=BG, bbox_inches="tight")
plt.close(fig)
print(f"\nWrote {out_png}")
print(f"Wrote {OUT_DIR/'neighbor_popularity_per_gene.csv'}")
