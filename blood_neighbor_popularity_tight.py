"""
Neighbour-popularity / specificity sweep for tighter noise bands.

Same setup as blood_neighbor_popularity.py but sweep the band width
multiplier k_SD ∈ {2.0, 1.0, 0.5, 0.25, 0.1}. Tighter bands → fewer
donors in-range per gene → more discriminating popularity scores.
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
K_SD_VALUES     = [2.0, 1.0, 0.5, 0.25, 0.1]
SEED            = 0


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

rng = np.random.default_rng(SEED)
perm = rng.permutation(n_s)
counts = counts[:, perm]
counts_train = counts[:, :n_s - HOLDOUT]
counts_test  = counts[:, n_s - HOLDOUT:]
lib_train = counts_train.sum(axis=0)
lib_test  = counts_test.sum(axis=0)
n_train   = counts_train.shape[1]
ref_lib   = float(np.median(lib_train))

# Pre-compute σ_raw and scale (k-independent)
sd_raw  = sigma_tech(counts_train)
scale   = ref_lib / lib_train
y_test  = counts_test * (ref_lib / lib_test)[None, :]


print(f"\n{'='*100}")
print(f"Specificity sweep: tighter bands → more discriminating neighbour set")
print(f"{'='*100}")
print(f"  {'k_SD':>5}  {'mean_n_nbr':>11}  {'med_n_nbr':>10}  "
      f"{'mean_pop':>10}  {'mean_spec':>10}  {'p99 spec':>10}  "
      f"{'spec range':>11}")
print("  " + "-" * 85)

results = []
top_specific_per_k = {}

for k_sd in K_SD_VALUES:
    # Build per-donor band at this k_sd
    lower = np.maximum((counts_train - k_sd * sd_raw) * scale[None, :], 0.0)
    upper = (counts_train + k_sd * sd_raw) * scale[None, :]

    n_nbr_acc       = np.zeros((HOLDOUT, n_genes), dtype=np.int32)
    mean_pop_acc    = np.zeros((HOLDOUT, n_genes))

    for h in range(HOLDOUT):
        y = y_test[:, h]
        A = (y[:, None] >= lower) & (y[:, None] <= upper)
        pop_d   = A.sum(axis=0).astype(np.int32)               # (n_train,)
        n_nbr   = A.sum(axis=1).astype(np.int32)               # (n_genes,)
        sum_pop = A.astype(np.int64) @ (pop_d - 1)             # (n_genes,)
        with np.errstate(invalid="ignore", divide="ignore"):
            mp = np.where(n_nbr > 0, sum_pop / n_nbr, 0.0)
        n_nbr_acc[h] = n_nbr
        mean_pop_acc[h] = mp

    gene_n_nbr = n_nbr_acc.mean(axis=0)
    gene_mean_pop = mean_pop_acc.mean(axis=0)
    spec = 1 - gene_mean_pop / n_genes

    # Many genes will have 0 neighbours at very tight k → mean_pop=0,
    # spec=1. Those aren't informative — they're "no donor close enough".
    # Report fraction with no neighbours and exclude from spec stats.
    zero_nbr = gene_n_nbr < 1
    spec_have_nbr = spec[~zero_nbr]
    print(f"  {k_sd:>5.2f}  {gene_n_nbr.mean():>11.1f}  "
          f"{int(np.median(gene_n_nbr)):>10d}  {gene_mean_pop[~zero_nbr].mean():>10.0f}  "
          f"{spec_have_nbr.mean():>10.4f}  "
          f"{np.quantile(spec_have_nbr, 0.99):>10.4f}  "
          f"{spec_have_nbr.min():.3f}–{spec_have_nbr.max():.3f}")

    # Also print fraction-with-no-neighbours
    print(f"          (fraction of genes with NO neighbours: "
          f"{zero_nbr.mean()*100:.1f}%)")

    out = pd.DataFrame({
        "gene":             gene_names,
        "mean_n_neighbors": gene_n_nbr,
        "mean_neighbor_popularity": gene_mean_pop,
        "specificity":      spec,
        "any_neighbor":     ~zero_nbr,
    })
    out.to_csv(OUT_DIR / f"neighbor_popularity_k{k_sd}.csv", index=False)

    top_specific_per_k[k_sd] = out[out.any_neighbor].sort_values(
        "specificity", ascending=False).head(15)
    results.append({
        "k_sd": k_sd,
        "mean_n_nbr": float(gene_n_nbr.mean()),
        "median_n_nbr": float(np.median(gene_n_nbr)),
        "frac_zero_nbr": float(zero_nbr.mean()),
        "mean_specificity": float(spec_have_nbr.mean()),
        "p99_specificity": float(np.quantile(spec_have_nbr, 0.99)),
        "min_specificity": float(spec_have_nbr.min()),
        "max_specificity": float(spec_have_nbr.max()),
    })


pd.DataFrame(results).to_csv(OUT_DIR / "neighbor_popularity_sweep.csv", index=False)

# Print top-15 specific genes at each k for inspection
print("\n" + "=" * 80)
print("TOP-15 most-specific genes at each k_SD")
print("=" * 80)
for k_sd in K_SD_VALUES:
    print(f"\n  k_SD = {k_sd}:")
    for _, row in top_specific_per_k[k_sd].iterrows():
        print(f"    {row['gene']:>14}  spec={row['specificity']:.3f}  "
              f"n_nbr={row['mean_n_neighbors']:.1f}  "
              f"mean_pop={row['mean_neighbor_popularity']:.0f}")


# ── Plots ─────────────────────────────────────────────────────────────
BG, CARD, TEXT, MUTED, GRID = '#0e1117', '#1a1d23', '#e6edf3', '#7d8590', '#21262d'
plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': CARD, 'axes.edgecolor': GRID,
    'axes.labelcolor': TEXT, 'text.color': TEXT, 'xtick.color': MUTED,
    'ytick.color': MUTED, 'grid.color': GRID, 'grid.alpha': 0.5,
    'font.family': 'sans-serif', 'font.size': 11,
})
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Neighbour-popularity / specificity: tighter bands sweep",
             color=TEXT, fontsize=13)

ax = axes[0]
res_df = pd.DataFrame(results)
ax.plot(res_df["k_sd"], res_df["mean_n_nbr"], "o-",
        color="#3fb950", lw=2, label="mean")
ax.plot(res_df["k_sd"], res_df["median_n_nbr"], "s--",
        color="#58a6ff", lw=2, label="median")
ax.set_yscale("log")
ax.invert_xaxis()
ax.set_xlabel("k_SD (band width multiplier)")
ax.set_ylabel("# in-range donors per gene")
ax.set_title("# neighbours shrinks with tighter bands")
ax.legend(facecolor=CARD, edgecolor=GRID, labelcolor=TEXT, fontsize=9)

ax = axes[1]
ax.plot(res_df["k_sd"], res_df["mean_specificity"], "o-",
        color="#3fb950", lw=2, label="mean")
ax.plot(res_df["k_sd"], res_df["p99_specificity"], "s--",
        color="#f78166", lw=2, label="p99")
ax.plot(res_df["k_sd"], res_df["max_specificity"], "d-",
        color="#d2a8ff", lw=2, label="max")
ax.invert_xaxis()
ax.set_xlabel("k_SD")
ax.set_ylabel("specificity")
ax.set_title("Specificity grows with tighter bands")
ax.legend(facecolor=CARD, edgecolor=GRID, labelcolor=TEXT, fontsize=9)

ax = axes[2]
ax.plot(res_df["k_sd"], res_df["frac_zero_nbr"] * 100, "o-",
        color="#f0883e", lw=2)
ax.invert_xaxis()
ax.set_xlabel("k_SD")
ax.set_ylabel("% genes with zero neighbours")
ax.set_title("Some genes drop out at very tight k")

fig.tight_layout()
out_png = OUT_DIR / "blood_neighbor_popularity_sweep.png"
fig.savefig(out_png, dpi=150, facecolor=BG, bbox_inches="tight")
plt.close(fig)
print(f"\nWrote {out_png}")
