"""
How close are we to the maximum search-space restriction?

Three tests:

  (1) Sweep k_SD from very loose (10) to very tight (0.01) — find a
      peak in the # of multi-gene sets. Both ends are uninformative:
        loose: every donor in-range for every gene → no clusters.
        tight: no donor in-range → empty bucket dominates.

  (2) PERMUTATION NULL. For each k_SD, shuffle each gene's train values
      independently across donors (preserves marginal expression
      distribution per gene; breaks all gene-gene correlations). Recompute
      gene sets. Real biology gives observed >> null; noise collapses
      to null.

  (3) FOLD-OVER-NULL. observed / null at each k_SD. Peak of this curve
      = the optimal search-space restriction. Above peak (looser) we
      lose specificity; below peak (tighter) we lose signal to noise.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

PROJECT = Path("/Users/rls/Desktop/programming-projects/single-cell/bulk-project")
GCT     = Path("/Users/rls/Downloads/gene_reads_v11_whole_blood.gct.gz")
OUT_DIR = PROJECT / "blood_technical_noise"

CPM_THRESHOLD   = 1.0
MIN_SAMPLE_FRAC = 0.10
HOLDOUT         = 10                 # subsample for speed
ALPHA           = 0.14
K_SD_VALUES     = [10.0, 5.0, 3.0, 2.0, 1.0, 0.5, 0.25, 0.1, 0.05, 0.025, 0.01]
SEED            = 0


def sigma_tech(x):
    return np.sqrt(x + (ALPHA * x) ** 2)


def gene_set_stats(A):
    """Given in-range bool matrix (n_g x n_train) return:
        n_sets_total, n_size_ge1 (non-empty tuple), n_size_ge2, n_size_ge5,
        max_set_size, mean_set_size_excluding_empty
    """
    n_g = A.shape[0]
    tuples = [tuple(np.flatnonzero(A[g])) for g in range(n_g)]
    c = Counter(tuples)
    n_total = len(c)
    n_ge1 = sum(1 for k in c if k != ())
    n_ge2 = sum(1 for k, v in c.items() if k != () and v >= 2)
    n_ge5 = sum(1 for k, v in c.items() if k != () and v >= 5)
    n_ge10 = sum(1 for k, v in c.items() if k != () and v >= 10)
    sizes_ne = [v for k, v in c.items() if k != ()]
    return {
        "n_sets_total": n_total,
        "n_ge1":  n_ge1,
        "n_ge2":  n_ge2,
        "n_ge5":  n_ge5,
        "n_ge10": n_ge10,
        "max_size": max(sizes_ne) if sizes_ne else 0,
        "mean_size_ne": np.mean(sizes_ne) if sizes_ne else 0,
    }


# ── Load + split ────────────────────────────────────────────────────
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
counts_tr = counts[:, :n_s - HOLDOUT]
counts_te = counts[:, n_s - HOLDOUT:]
lib_tr = counts_tr.sum(axis=0)
lib_te = counts_te.sum(axis=0)
n_train = counts_tr.shape[1]
ref = float(np.median(lib_tr))

sd_tr = sigma_tech(counts_tr)
sc_tr = ref / lib_tr
y_te  = counts_te * (ref / lib_te)[None, :]


# ── Permutation: shuffle each gene's train values independently ─────
print("Building permuted train (each gene shuffled across donors) ...")
counts_tr_perm = counts_tr.copy()
rng_perm = np.random.default_rng(SEED + 1)
for g in range(n_genes):
    rng_perm.shuffle(counts_tr_perm[g])
sd_tr_perm = sigma_tech(counts_tr_perm)


# ── Sweep ───────────────────────────────────────────────────────────
print(f"\nSweeping k_SD ∈ {K_SD_VALUES} ...")
print(f"\n{'k_SD':>6}  {'mean_n_nbr':>11}  "
      f"{'OBS n_ge2':>10}  {'NULL n_ge2':>11}  {'fold':>6}  "
      f"{'OBS n_ge5':>10}  {'NULL n_ge5':>11}  {'fold':>6}  "
      f"{'OBS max':>9}  {'NULL max':>10}")
print("-" * 110)

results = []
for ksd in K_SD_VALUES:
    lower_obs = np.maximum((counts_tr - ksd * sd_tr) * sc_tr[None, :], 0.0)
    upper_obs = (counts_tr + ksd * sd_tr) * sc_tr[None, :]
    lower_perm = np.maximum((counts_tr_perm - ksd * sd_tr_perm) * sc_tr[None, :], 0.0)
    upper_perm = (counts_tr_perm + ksd * sd_tr_perm) * sc_tr[None, :]

    obs_stats = []
    null_stats = []
    n_nbr_obs = []
    for h in range(HOLDOUT):
        y = y_te[:, h]
        A_obs  = (y[:, None] >= lower_obs)  & (y[:, None] <= upper_obs)
        A_perm = (y[:, None] >= lower_perm) & (y[:, None] <= upper_perm)
        n_nbr_obs.append(A_obs.sum(axis=1).mean())
        obs_stats.append(gene_set_stats(A_obs))
        null_stats.append(gene_set_stats(A_perm))

    def avg(stats, key):
        return float(np.mean([s[key] for s in stats]))

    obs_n2  = avg(obs_stats,  "n_ge2");  null_n2  = avg(null_stats, "n_ge2")
    obs_n5  = avg(obs_stats,  "n_ge5");  null_n5  = avg(null_stats, "n_ge5")
    obs_max = avg(obs_stats,  "max_size"); null_max = avg(null_stats, "max_size")
    fold_n2 = obs_n2 / max(null_n2, 1e-9)
    fold_n5 = obs_n5 / max(null_n5, 1e-9)

    print(f"{ksd:>6.3f}  {np.mean(n_nbr_obs):>11.1f}  "
          f"{obs_n2:>10.1f}  {null_n2:>11.1f}  {fold_n2:>6.2f}  "
          f"{obs_n5:>10.1f}  {null_n5:>11.1f}  {fold_n5:>6.2f}  "
          f"{obs_max:>9.1f}  {null_max:>10.1f}")
    results.append({
        "k_sd": ksd,
        "mean_n_neighbors": float(np.mean(n_nbr_obs)),
        "obs_n_ge2": obs_n2, "null_n_ge2": null_n2, "fold_n_ge2": fold_n2,
        "obs_n_ge5": obs_n5, "null_n_ge5": null_n5, "fold_n_ge5": fold_n5,
        "obs_max":   obs_max, "null_max":   null_max,
    })

res_df = pd.DataFrame(results)
res_df.to_csv(OUT_DIR / "search_space_limit_sweep.csv", index=False)


# ── Plot: peak in fold-over-null = the validation answer ─────────────
BG, CARD, TEXT, MUTED, GRID = '#0e1117', '#1a1d23', '#e6edf3', '#7d8590', '#21262d'
plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': CARD, 'axes.edgecolor': GRID,
    'axes.labelcolor': TEXT, 'text.color': TEXT, 'xtick.color': MUTED,
    'ytick.color': MUTED, 'grid.color': GRID, 'grid.alpha': 0.5,
    'font.family': 'sans-serif', 'font.size': 11,
})
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Search-space restriction limit: observed vs permutation null",
             color=TEXT, fontsize=13)

ax = axes[0]
ax.plot(res_df["k_sd"], res_df["obs_n_ge2"], "o-",
        color="#3fb950", lw=2, label="observed ≥2")
ax.plot(res_df["k_sd"], res_df["null_n_ge2"], "s--",
        color="#f78166", lw=2, label="permutation null ≥2")
ax.plot(res_df["k_sd"], res_df["obs_n_ge5"], "d-",
        color="#58a6ff", lw=2, label="observed ≥5")
ax.plot(res_df["k_sd"], res_df["null_n_ge5"], "^--",
        color="#d2a8ff", lw=2, label="null ≥5")
ax.set_xscale("log"); ax.invert_xaxis()
ax.set_xlabel("k_SD (band width)")
ax.set_ylabel("# multi-gene sets")
ax.set_yscale("symlog")
ax.set_title("Multi-gene sets — observed vs null")
ax.legend(facecolor=CARD, edgecolor=GRID, labelcolor=TEXT, fontsize=8)

ax = axes[1]
ax.plot(res_df["k_sd"], res_df["fold_n_ge2"], "o-",
        color="#3fb950", lw=2, label="≥2 genes/set")
ax.plot(res_df["k_sd"], res_df["fold_n_ge5"], "s-",
        color="#58a6ff", lw=2, label="≥5 genes/set")
ax.set_xscale("log"); ax.invert_xaxis()
ax.set_xlabel("k_SD")
ax.set_ylabel("fold over null")
ax.set_title("Fold-over-null — peak = optimal restriction")
ax.set_yscale("log")
ax.legend(facecolor=CARD, edgecolor=GRID, labelcolor=TEXT, fontsize=8)

ax = axes[2]
ax.plot(res_df["k_sd"], res_df["obs_max"], "o-",
        color="#3fb950", lw=2, label="observed")
ax.plot(res_df["k_sd"], res_df["null_max"], "s--",
        color="#f78166", lw=2, label="null")
ax.set_xscale("log"); ax.invert_xaxis()
ax.set_xlabel("k_SD")
ax.set_ylabel("max single-set size")
ax.set_yscale("symlog")
ax.set_title("Largest cluster — observed vs null")
ax.legend(facecolor=CARD, edgecolor=GRID, labelcolor=TEXT, fontsize=9)

fig.tight_layout()
out_png = OUT_DIR / "blood_search_space_limit.png"
fig.savefig(out_png, dpi=150, facecolor=BG, bbox_inches="tight")
plt.close(fig)
print(f"\nWrote {out_png}")
print(f"Wrote {OUT_DIR/'search_space_limit_sweep.csv'}")
