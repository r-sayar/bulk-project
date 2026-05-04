"""
Sweep the band-width multiplier k in band = x ± k·σ_tech(x).

For each k ∈ {2.0, 1.5, 1.0, 0.75, 0.5, 0.25, 0.1}:
  - Build per-sample bands, merge per-gene, clip to [0, max_train]
  - Compute uncovered_fraction per gene (high = tight)
  - Compute holdout_inside_rate per gene (high = reproducible)
  - Joint count: genes where uncov > T AND all 50 holdouts inside

This shows the trade-off:
  k large  → bands tile everything, uncov ≈ 0, holdouts trivially in
  k small  → bands are pinpoints, uncov ≈ 1, holdouts fall in only if
             EVERY new donor lands within ±k·σ_tech of some train donor
             — that's the genuine saturation question.
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
ALPHA           = 0.14
SEED            = 0
K_SWEEP         = [2.0, 1.5, 1.0, 0.75, 0.5, 0.25, 0.1]


def sigma_tech(x):
    return np.sqrt(x + (ALPHA * x) ** 2)


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

# Pre-compute these once
sd_raw_train        = sigma_tech(counts_train)
scale_train         = ref_lib / lib_train
counts_train_centers = counts_train * scale_train[None, :]
counts_test_scaled   = counts_test  * (ref_lib / lib_test)[None, :]
sd_scaled            = sd_raw_train * scale_train[None, :]
mean_train           = counts_train_centers.mean(axis=1)
max_train            = counts_train_centers.max(axis=1)


def merge_intervals(starts, ends):
    order = np.argsort(starts)
    s = starts[order]; e = ends[order]
    out_s = [s[0]]; out_e = [e[0]]
    for i in range(1, len(s)):
        if s[i] <= out_e[-1]:
            out_e[-1] = max(out_e[-1], e[i])
        else:
            out_s.append(s[i]); out_e.append(e[i])
    return np.array(out_s), np.array(out_e)


# Run sweep
results = []          # one row per k, with summary stats
per_gene_tables = {}  # k -> DataFrame

print("\n" + "=" * 90)
print(f"{'k':>5}  {'mean_unc':>9}  {'med_unc':>8}  {'p75_unc':>8}  "
      f"{'p95_unc':>8}  {'mean_hin':>9}  {'med_hin':>8}  "
      f"{'#unc>0.5∧50/50':>15}  {'#unc>0.7∧50/50':>15}  {'#unc>0.9∧50/50':>15}")
print("=" * 90)

for k in K_SWEEP:
    uncov   = np.zeros(n_genes)
    n_int   = np.zeros(n_genes, dtype=int)
    hin_rate = np.zeros(n_genes)
    for g in range(n_genes):
        if max_train[g] <= 0:
            continue
        lo = np.maximum(counts_train_centers[g] - k * sd_scaled[g], 0.0)
        up = counts_train_centers[g] + k * sd_scaled[g]
        ms, me = merge_intervals(lo, up)
        # clip to [0, max_train_g]
        clo = np.maximum(ms, 0.0)
        chi = np.minimum(me, max_train[g])
        keep = chi > clo
        clo, chi = clo[keep], chi[keep]
        cw = float((chi - clo).sum())
        uncov[g] = 1 - cw / max_train[g]
        n_int[g] = len(clo)
        # holdout
        y = counts_test_scaled[g]
        # only "inside" if y <= max_train AND in some clipped interval
        above = y > max_train[g]
        in_any = ((y[:, None] >= clo[None, :]) &
                  (y[:, None] <= chi[None, :])).any(axis=1) & ~above
        hin_rate[g] = in_any.mean()

    all_in = hin_rate == 1.0
    n_tight5 = int(((uncov > 0.5) & all_in).sum())
    n_tight7 = int(((uncov > 0.7) & all_in).sum())
    n_tight9 = int(((uncov > 0.9) & all_in).sum())
    n_tight_any5 = int((uncov > 0.5).sum())
    n_tight_any7 = int((uncov > 0.7).sum())
    n_tight_any9 = int((uncov > 0.9).sum())
    print(f"{k:>5.2f}  {uncov.mean():>9.4f}  {np.median(uncov):>8.4f}  "
          f"{np.quantile(uncov,0.75):>8.4f}  {np.quantile(uncov,0.95):>8.4f}  "
          f"{hin_rate.mean():>9.4f}  {np.median(hin_rate):>8.4f}  "
          f"{n_tight5:>8d} / {n_tight_any5:>4d}  "
          f"{n_tight7:>8d} / {n_tight_any7:>4d}  "
          f"{n_tight9:>8d} / {n_tight_any9:>4d}")

    results.append({
        "k": k,
        "mean_uncov": float(uncov.mean()),
        "median_uncov": float(np.median(uncov)),
        "p75_uncov": float(np.quantile(uncov, 0.75)),
        "p95_uncov": float(np.quantile(uncov, 0.95)),
        "mean_holdout_inside": float(hin_rate.mean()),
        "median_holdout_inside": float(np.median(hin_rate)),
        "n_tight_unc05_and_all_in": n_tight5,
        "n_tight_unc05_total":      n_tight_any5,
        "n_tight_unc07_and_all_in": n_tight7,
        "n_tight_unc07_total":      n_tight_any7,
        "n_tight_unc09_and_all_in": n_tight9,
        "n_tight_unc09_total":      n_tight_any9,
        "n_genes":  n_genes,
    })
    per_gene_tables[k] = pd.DataFrame({
        "gene": gene_names,
        "mean_train": mean_train,
        "max_train":  max_train,
        f"uncov_k{k}":      uncov,
        f"hin_rate_k{k}":   hin_rate,
        f"n_intervals_k{k}": n_int,
    })


pd.DataFrame(results).to_csv(OUT_DIR / "band_sd_sweep_summary.csv", index=False)


# Persist per-gene table at the smallest k (most informative)
k_smallest = K_SWEEP[-1]
per_gene_tables[k_smallest].sort_values(f"uncov_k{k_smallest}", ascending=False) \
    .to_csv(OUT_DIR / f"per_gene_at_k_{k_smallest}.csv", index=False)


# At each k, list the top "tight + all in" genes (the saturation winners)
print("\nTop 'tight & all 50 inside' counts at each k:")
for r in results:
    print(f"  k={r['k']:>4.2f}: uncov>0.5 ∧ 50/50 = {r['n_tight_unc05_and_all_in']:>5}, "
          f"uncov>0.7 ∧ 50/50 = {r['n_tight_unc07_and_all_in']:>5}, "
          f"uncov>0.9 ∧ 50/50 = {r['n_tight_unc09_and_all_in']:>5}")


# ── Visualization ─────────────────────────────────────────────────────
BG, CARD, TEXT, MUTED, GRID = '#0e1117', '#1a1d23', '#e6edf3', '#7d8590', '#21262d'
plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': CARD, 'axes.edgecolor': GRID,
    'axes.labelcolor': TEXT, 'text.color': TEXT, 'xtick.color': MUTED,
    'ytick.color': MUTED, 'grid.color': GRID, 'grid.alpha': 0.5,
    'font.family': 'sans-serif', 'font.size': 11,
})

fig = plt.figure(figsize=(15, 11))
gs = GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.32)
fig.suptitle("Band-width sweep: tighter bands → narrower acceptable region → "
             "is saturation real?", color=TEXT, fontsize=13)

ks = np.array([r["k"] for r in results])
mean_uncov_arr = np.array([r["mean_uncov"] for r in results])
median_uncov_arr = np.array([r["median_uncov"] for r in results])
mean_hin_arr   = np.array([r["mean_holdout_inside"] for r in results])
median_hin_arr = np.array([r["median_holdout_inside"] for r in results])

ax = fig.add_subplot(gs[0, 0])
ax.plot(ks, mean_uncov_arr, "o-", color="#3fb950", label="mean uncov")
ax.plot(ks, median_uncov_arr, "s--", color="#58a6ff", label="median uncov")
ax.set_xlabel("k (band = ±k·σ_tech)")
ax.set_ylabel("uncovered fraction in [0, max_train]")
ax.set_title("Band tightness vs k")
ax.invert_xaxis()
ax.legend(facecolor=CARD, edgecolor=GRID, labelcolor=TEXT, fontsize=9)

ax = fig.add_subplot(gs[0, 1])
ax.plot(ks, mean_hin_arr,  "o-", color="#f78166", label="mean")
ax.plot(ks, median_hin_arr, "s--", color="#d2a8ff", label="median")
ax.set_xlabel("k")
ax.set_ylabel("holdout-inside rate")
ax.set_title("Reproducibility vs k")
ax.invert_xaxis()
ax.legend(facecolor=CARD, edgecolor=GRID, labelcolor=TEXT, fontsize=9)

ax = fig.add_subplot(gs[0, 2])
n5 = np.array([r["n_tight_unc05_and_all_in"] for r in results])
n7 = np.array([r["n_tight_unc07_and_all_in"] for r in results])
n9 = np.array([r["n_tight_unc09_and_all_in"] for r in results])
ax.plot(ks, n5, "o-", color="#3fb950", label="uncov>0.5 ∧ 50/50")
ax.plot(ks, n7, "s-", color="#58a6ff", label="uncov>0.7 ∧ 50/50")
ax.plot(ks, n9, "d-", color="#f78166", label="uncov>0.9 ∧ 50/50")
ax.set_xlabel("k")
ax.set_ylabel("# genes")
ax.set_title("'Tight & reproducible' gene count")
ax.set_yscale("symlog")
ax.invert_xaxis()
ax.legend(facecolor=CARD, edgecolor=GRID, labelcolor=TEXT, fontsize=8)

# Histograms of uncov and hin at each k
ax = fig.add_subplot(gs[1, 0])
for r, color in zip(results, plt.cm.viridis(np.linspace(0, 1, len(results)))):
    k = r["k"]
    sub = per_gene_tables[k][f"uncov_k{k}"].values
    ax.hist(sub, bins=50, alpha=0.4, color=color, label=f"k={k}")
ax.set_xlabel("uncovered fraction")
ax.set_ylabel("# genes")
ax.set_yscale("log")
ax.set_title("Distribution of uncovered fraction across k")
ax.legend(facecolor=CARD, edgecolor=GRID, labelcolor=TEXT, fontsize=8)

ax = fig.add_subplot(gs[1, 1])
for r, color in zip(results, plt.cm.viridis(np.linspace(0, 1, len(results)))):
    k = r["k"]
    sub = per_gene_tables[k][f"hin_rate_k{k}"].values
    ax.hist(sub, bins=50, alpha=0.4, color=color, label=f"k={k}")
ax.set_xlabel("holdout-inside rate")
ax.set_ylabel("# genes")
ax.set_yscale("log")
ax.set_title("Distribution of holdout-inside across k")
ax.legend(facecolor=CARD, edgecolor=GRID, labelcolor=TEXT, fontsize=8)

# Joint hexbin at k=0.1
ax = fig.add_subplot(gs[1, 2])
k = K_SWEEP[-1]
sub = per_gene_tables[k]
hb = ax.hexbin(sub[f"uncov_k{k}"], sub[f"hin_rate_k{k}"],
               gridsize=40, mincnt=1, cmap="magma", bins="log")
ax.set_xlabel(f"uncov at k={k}")
ax.set_ylabel(f"holdout-inside at k={k}")
ax.set_title(f"Joint distribution at k={k}")
plt.colorbar(hb, ax=ax, label="log10 # genes")

# Per-gene track for HBB and CD1A across k values
def gene_track(gname):
    g = int(np.where(gene_names == gname)[0][0])
    uncov_vals = [per_gene_tables[k].iloc[g][f"uncov_k{k}"] for k in K_SWEEP]
    hin_vals   = [per_gene_tables[k].iloc[g][f"hin_rate_k{k}"] for k in K_SWEEP]
    return uncov_vals, hin_vals

ax = fig.add_subplot(gs[2, 0])
for gname, color in zip(["HBB", "CD1A", "DDIT4", "JUN", "ACTB"],
                        ["#f78166", "#3fb950", "#58a6ff", "#d2a8ff", "#f0883e"]):
    u, _ = gene_track(gname)
    ax.plot(K_SWEEP, u, "o-", color=color, label=gname)
ax.invert_xaxis()
ax.set_xlabel("k")
ax.set_ylabel("uncovered fraction")
ax.set_title("Per-gene uncovered vs k")
ax.legend(facecolor=CARD, edgecolor=GRID, labelcolor=TEXT, fontsize=9)

ax = fig.add_subplot(gs[2, 1])
for gname, color in zip(["HBB", "CD1A", "DDIT4", "JUN", "ACTB"],
                        ["#f78166", "#3fb950", "#58a6ff", "#d2a8ff", "#f0883e"]):
    _, h = gene_track(gname)
    ax.plot(K_SWEEP, h, "o-", color=color, label=gname)
ax.invert_xaxis()
ax.set_xlabel("k")
ax.set_ylabel("holdout-inside rate")
ax.set_title("Per-gene holdout-inside vs k")
ax.axhline(1.0, color="white", ls=":", lw=1, alpha=0.5)
ax.legend(facecolor=CARD, edgecolor=GRID, labelcolor=TEXT, fontsize=9)

# Decile breakdown at smallest k (the strict version)
ax = fig.add_subplot(gs[2, 2])
ranks = np.argsort(np.argsort(mean_train))
decile = np.minimum(ranks * 10 // n_genes, 9)
strict_uncov = per_gene_tables[K_SWEEP[-1]][f"uncov_k{K_SWEEP[-1]}"].values
strict_hin   = per_gene_tables[K_SWEEP[-1]][f"hin_rate_k{K_SWEEP[-1]}"].values
xs = np.arange(10)
ax.bar(xs - 0.18, [strict_uncov[decile==d].mean() for d in range(10)],
       width=0.36, color="#d2a8ff", label="uncov")
ax.bar(xs + 0.18, [strict_hin[decile==d].mean() for d in range(10)],
       width=0.36, color="#3fb950", label="holdout-in")
ax.set_xlabel("expression decile")
ax.set_ylabel("mean")
ax.set_title(f"Decile breakdown at k={K_SWEEP[-1]}")
ax.legend(facecolor=CARD, edgecolor=GRID, labelcolor=TEXT, fontsize=8)

fig.tight_layout()
out_png = OUT_DIR / "blood_band_sd_sweep.png"
fig.savefig(out_png, dpi=150, facecolor=BG, bbox_inches="tight")
plt.close(fig)
print(f"\nWrote {out_png}")
print(f"Wrote {OUT_DIR/'band_sd_sweep_summary.csv'}")
print(f"Wrote {OUT_DIR/f'per_gene_at_k_{k_smallest}.csv'}")
