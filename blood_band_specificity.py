"""
Per-gene band SPECIFICITY:
    For each gene g, define the total possible range as [0, max_train_g].
    Build the merged union of 753 per-sample ±2σ_tech bands, clipped to
    that range. Compute:

        uncovered_fraction_g = 1 - clipped_union_width_g / max_train_g

    HIGH uncovered fraction = tight, informative bands (most of the
    [0, max] range is forbidden by the model).
    LOW uncovered fraction  = bands span everything; trivially safe.

    Then for each holdout, check whether its value falls inside the
    clipped union. The combined "good saturation" signal is:

        uncovered fraction is high  AND  holdouts fall inside.

    If both are true, we have *specific* coverage (not just wide bands).
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
SD_BAND         = 2.0
SEED            = 0


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
n_train = counts_train.shape[1]
ref_lib = float(np.median(lib_train))

# Per-sample bands on raw counts, scale edges to ref library
sd_raw      = sigma_tech(counts_train)
scale_train = ref_lib / lib_train
lower_s = np.maximum((counts_train - SD_BAND * sd_raw) * scale_train[None, :], 0.0)
upper_s = (counts_train + SD_BAND * sd_raw) * scale_train[None, :]
counts_train_scaled = counts_train * scale_train[None, :]
counts_test_scaled  = counts_test  * (ref_lib / lib_test)[None, :]

mean_train = counts_train_scaled.mean(axis=1)
max_train  = counts_train_scaled.max(axis=1)
max_test   = counts_test_scaled.max(axis=1)


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


print("\nComputing band specificity, clipped to [0, max_train] per gene ...")
clipped_width   = np.zeros(n_genes)
n_clipped_int   = np.zeros(n_genes, dtype=int)
holdout_inside  = np.zeros((n_genes, HOLDOUT), dtype=bool)
holdout_above_max = np.zeros((n_genes, HOLDOUT), dtype=bool)

for g in range(n_genes):
    total_max = max_train[g]
    if total_max <= 0:
        continue
    ms, me = merge_intervals(lower_s[g], upper_s[g])
    # Clip to [0, total_max]
    lo = np.maximum(ms, 0.0)
    hi = np.minimum(me, total_max)
    keep = hi > lo
    lo, hi = lo[keep], hi[keep]
    clipped_width[g] = float((hi - lo).sum())
    n_clipped_int[g] = len(lo)
    # Holdout check (NB: holdout values >max_train are forced outside)
    y = counts_test_scaled[g]
    above = y > total_max
    inside_clipped = ((y[:, None] >= lo[None, :]) &
                     (y[:, None] <= hi[None, :])).any(axis=1) & ~above
    holdout_inside[g] = inside_clipped
    holdout_above_max[g] = above

uncovered_fraction = 1 - clipped_width / np.where(max_train > 0, max_train, 1)
holdout_inside_rate = holdout_inside.mean(axis=1)
holdout_above_rate  = holdout_above_max.mean(axis=1)


# ── Stats over all genes ─────────────────────────────────────────────
print("\n" + "=" * 80)
print("UNCOVERED FRACTION distribution: 1 - (band width within [0, max_train]) / max_train")
print("HIGH = tight bands (good); LOW = bands cover most of [0, max] (less informative)")
print("=" * 80)
v = uncovered_fraction
for stat, val in [("mean", v.mean()), ("median", np.median(v))]:
    print(f"  {stat:>6}  {val:.4f}")
for q in (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99):
    print(f"  q={q:>4.2f}  {np.quantile(v, q):.4f}")

print(f"\n  fraction of genes with uncovered_fraction > 0.5: "
      f"{(v > 0.5).mean()*100:.1f}%")
print(f"  fraction with uncovered_fraction > 0.7: {(v > 0.7).mean()*100:.1f}%")
print(f"  fraction with uncovered_fraction > 0.9: {(v > 0.9).mean()*100:.1f}%")


print("\n" + "=" * 80)
print("HOLDOUT INSIDE-CLIPPED-UNION rate per gene")
print(f"(should be ~1.0 to argue saturation; combined with high uncovered_fraction "
      "this is the strong signal)")
print("=" * 80)
v2 = holdout_inside_rate
for stat, val in [("mean", v2.mean()), ("median", np.median(v2))]:
    print(f"  {stat:>6}  {val:.4f}")
for q in (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99):
    print(f"  q={q:>4.2f}  {np.quantile(v2, q):.4f}")


# Joint distribution: how many genes are tight AND have all holdouts inside
tight = uncovered_fraction > 0.5
all_in = holdout_inside_rate == 1.0
print(f"\n  genes with uncovered>0.5 AND 50/50 holdouts inside: "
      f"{(tight & all_in).sum():,} / {n_genes:,} ({(tight & all_in).mean()*100:.1f}%)")
print(f"  genes with uncovered>0.7 AND 50/50 holdouts inside: "
      f"{((uncovered_fraction>0.7) & all_in).sum():,} "
      f"({((uncovered_fraction>0.7) & all_in).mean()*100:.1f}%)")
print(f"  genes with uncovered>0.9 AND 50/50 holdouts inside: "
      f"{((uncovered_fraction>0.9) & all_in).sum():,} "
      f"({((uncovered_fraction>0.9) & all_in).mean()*100:.1f}%)")


# Stratify by expression decile
print("\n" + "=" * 80)
print("UNCOVERED FRACTION + HOLDOUT INSIDE RATE by expression decile")
print("=" * 80)
ranks = np.argsort(np.argsort(mean_train))
decile = np.minimum(ranks * 10 // n_genes, 9)
print(f"  {'decile':>6}  {'mean_count':>11}  {'uncovered':>10}  "
      f"{'holdout_in':>11}  {'tight∧all_in':>12}")
for d in range(10):
    m = decile == d
    tight_and_in = ((uncovered_fraction > 0.5) & all_in) & m
    print(f"  {d:>6}  {mean_train[m].mean():>11.2f}  "
          f"{uncovered_fraction[m].mean():>10.4f}  "
          f"{holdout_inside_rate[m].mean():>11.4f}  "
          f"{tight_and_in.sum() / m.sum() * 100:>11.1f}%")


# Top "tightest" genes (highest uncovered, but with holdouts fully inside)
print("\n" + "=" * 80)
print("TOP 20 'tight + reproducible' genes (uncovered_fraction × holdout_inside_rate)")
print("=" * 80)
score = uncovered_fraction * holdout_inside_rate
order = np.argsort(score)[::-1][:20]
df_top = pd.DataFrame({
    "gene":               gene_names[order],
    "mean_train":         mean_train[order],
    "max_train":          max_train[order],
    "uncovered_frac":     uncovered_fraction[order],
    "holdout_inside_rate": holdout_inside_rate[order],
    "n_clipped_intervals": n_clipped_int[order],
}).round(4)
print(df_top.to_string(index=False))

print("\n" + "=" * 80)
print("TOP 20 'loose' genes (lowest uncovered_fraction = bands cover everything)")
print("=" * 80)
order_loose = np.argsort(uncovered_fraction)[:20]
df_loose = pd.DataFrame({
    "gene":               gene_names[order_loose],
    "mean_train":         mean_train[order_loose],
    "max_train":          max_train[order_loose],
    "uncovered_frac":     uncovered_fraction[order_loose],
    "holdout_inside_rate": holdout_inside_rate[order_loose],
    "n_clipped_intervals": n_clipped_int[order_loose],
}).round(4)
print(df_loose.to_string(index=False))


# Save full table
pd.DataFrame({
    "gene":                gene_names,
    "mean_train":          mean_train,
    "max_train":           max_train,
    "uncovered_fraction":  uncovered_fraction,
    "covered_within_total": clipped_width,
    "n_clipped_intervals": n_clipped_int,
    "holdout_inside_rate": holdout_inside_rate,
    "holdout_above_rate":  holdout_above_rate,
    "tight_and_holdout_all_in": tight & all_in,
}).to_csv(OUT_DIR / "per_gene_band_specificity.csv", index=False)


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
fig.suptitle("Band SPECIFICITY: how much of [0, max_train] is NOT covered "
             "by the 753 per-sample bands?",
             color=TEXT, fontsize=13)

ax = fig.add_subplot(gs[0, 0])
ax.hist(uncovered_fraction, bins=50, color="#3fb950", edgecolor="#1f6f33")
for q, c in [(0.5, "#f78166"), (0.7, "#f0883e"), (0.9, "#d2a8ff")]:
    ax.axvline(q, color=c, ls="--", lw=1, alpha=0.7,
               label=f"{q:.1f}: {(uncovered_fraction>q).mean()*100:.0f}% above")
ax.set_xlabel("uncovered fraction = 1 - covered_width/max_train")
ax.set_ylabel("# genes")
ax.set_title("Distribution of band tightness")
ax.legend(facecolor=CARD, edgecolor=GRID, labelcolor=TEXT, fontsize=8)

ax = fig.add_subplot(gs[0, 1])
ax.hist(holdout_inside_rate, bins=50, color="#58a6ff", edgecolor="#1f4e8f")
ax.axvline(holdout_inside_rate.mean(), color="white", ls=":", lw=1.5,
           label=f"mean = {holdout_inside_rate.mean():.4f}")
ax.set_xlabel("fraction of 50 holdouts inside the clipped union")
ax.set_ylabel("# genes")
ax.set_yscale("log")
ax.set_title("Per-gene holdout pass rate")
ax.legend(facecolor=CARD, edgecolor=GRID, labelcolor=TEXT, fontsize=8)

ax = fig.add_subplot(gs[0, 2])
hb = ax.hexbin(uncovered_fraction, holdout_inside_rate,
               gridsize=40, mincnt=1, cmap="magma", bins="log")
ax.set_xlabel("uncovered fraction (band tightness)")
ax.set_ylabel("holdout inside-union rate")
ax.set_title("Joint: tight ∧ reproducible")
plt.colorbar(hb, ax=ax, label="log10 # genes")

ax = fig.add_subplot(gs[1, 0])
log_mu = np.log10(mean_train + 1e-3)
ax.scatter(log_mu, uncovered_fraction, s=2, alpha=0.4, color="#d2a8ff",
           rasterized=True)
bins_ed = np.linspace(log_mu.min(), log_mu.max(), 25)
binc = 0.5 * (bins_ed[:-1] + bins_ed[1:])
mean_in_bin = np.array([
    uncovered_fraction[(log_mu >= b0) & (log_mu < b1)].mean()
    if ((log_mu >= b0) & (log_mu < b1)).any() else np.nan
    for b0, b1 in zip(bins_ed[:-1], bins_ed[1:])
])
ax.plot(binc, mean_in_bin, color="#f78166", lw=2, label="binned mean")
ax.set_xlabel("log10 mean train count")
ax.set_ylabel("uncovered fraction")
ax.set_title("Tightness vs expression level")
ax.legend(facecolor=CARD, edgecolor=GRID, labelcolor=TEXT, fontsize=8)

ax = fig.add_subplot(gs[1, 1])
ax.scatter(log_mu, holdout_inside_rate, s=2, alpha=0.4, color="#3fb950",
           rasterized=True)
mean_in_bin2 = np.array([
    holdout_inside_rate[(log_mu >= b0) & (log_mu < b1)].mean()
    if ((log_mu >= b0) & (log_mu < b1)).any() else np.nan
    for b0, b1 in zip(bins_ed[:-1], bins_ed[1:])
])
ax.plot(binc, mean_in_bin2, color="#f78166", lw=2, label="binned mean")
ax.axhline(1.0, color="#f0883e", ls=":", lw=1)
ax.set_xlabel("log10 mean train count")
ax.set_ylabel("holdout inside-union rate")
ax.set_title("Holdout pass-rate vs expression level")
ax.legend(facecolor=CARD, edgecolor=GRID, labelcolor=TEXT, fontsize=8)

# Decile bar chart for uncovered fraction
ax = fig.add_subplot(gs[1, 2])
deciles_uncov = [uncovered_fraction[decile == d].mean() for d in range(10)]
deciles_in    = [holdout_inside_rate[decile == d].mean() for d in range(10)]
xs = np.arange(10)
ax.bar(xs - 0.18, deciles_uncov, width=0.36, color="#d2a8ff",
       label="uncovered fraction (mean)")
ax.bar(xs + 0.18, deciles_in,    width=0.36, color="#3fb950",
       label="holdout inside rate")
ax.axhline(1.0, color="#f78166", ls=":", lw=1)
ax.set_xlabel("expression decile")
ax.set_ylabel("mean")
ax.set_title("Stratified by expression")
ax.legend(facecolor=CARD, edgecolor=GRID, labelcolor=TEXT, fontsize=8)

# Top-20 tightest reproducible genes (bar)
ax = fig.add_subplot(gs[2, 0])
top20 = df_top.head(20)
ax.barh(np.arange(len(top20))[::-1], top20["uncovered_frac"][::-1],
        color="#3fb950", label="uncovered")
ax.barh(np.arange(len(top20))[::-1], top20["holdout_inside_rate"][::-1] * 0.4,
        color="#58a6ff", alpha=0.5, label="holdout-in × 0.4")
ax.set_yticks(np.arange(len(top20))[::-1])
ax.set_yticklabels(top20["gene"][::-1], fontsize=8)
ax.set_xlabel("score")
ax.set_title("Top 20 tight ∧ reproducible")
ax.legend(facecolor=CARD, edgecolor=GRID, labelcolor=TEXT, fontsize=8)

# Example: pick HBB and a tight reproducible one and a loose one
def example_panel(ax_, g, title):
    total_max = max_train[g]
    ms, me = merge_intervals(lower_s[g], upper_s[g])
    lo = np.maximum(ms, 0.0); hi = np.minimum(me, total_max)
    for s, e in zip(lo, hi):
        if e > s:
            ax_.axvspan(s, e, color="#3fb950", alpha=0.18)
    ax_.axvspan(0, total_max, color="white", alpha=0.0)
    ax_.scatter(counts_train_scaled[g], np.full(n_train, 0.4) + np.random.uniform(-0.05, 0.05, n_train),
                color="#58a6ff", s=2, alpha=0.4, rasterized=True, label="train")
    yt = counts_test_scaled[g]
    ax_.scatter(yt, np.full(HOLDOUT, 0.7), color="#f78166",
                edgecolor="white", s=40, label="holdout")
    ax_.set_xlim(-0.05 * total_max, total_max * 1.05)
    ax_.set_xlabel("scaled count")
    ax_.set_yticks([])
    ax_.set_title(f"{title}: uncov={uncovered_fraction[g]:.2f}  "
                  f"hold-in={holdout_inside_rate[g]:.2f}  n_int={n_clipped_int[g]}",
                  fontsize=9)
    ax_.legend(facecolor=CARD, edgecolor=GRID, labelcolor=TEXT, fontsize=7,
               loc="upper right")


# Pick HBB
hbb_idx = int(np.where(gene_names == "HBB")[0][0])
ax = fig.add_subplot(gs[2, 1])
example_panel(ax, hbb_idx, f"HBB (highest expressed)")

# Pick a high-uncovered-fraction reproducible gene (bimodal candidate)
high_u = df_top.iloc[0]["gene"]
g_high = int(np.where(gene_names == high_u)[0][0])
ax = fig.add_subplot(gs[2, 2])
example_panel(ax, g_high, f"{high_u} (tight + all holdouts in)")

fig.tight_layout()
out_png = OUT_DIR / "blood_band_specificity.png"
fig.savefig(out_png, dpi=150, facecolor=BG, bbox_inches="tight")
plt.close(fig)
print(f"\nWrote {out_png}")
print(f"Wrote {OUT_DIR/'per_gene_band_specificity.csv'}")
