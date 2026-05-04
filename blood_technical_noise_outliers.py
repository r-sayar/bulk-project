"""
Per-sample technical-noise bands → per-gene acceptable region → check
how many genes in each held-out sample fall outside.

Model (from screenshot):
    σ_tech(x) = sqrt(x + (α·x)²)        with α = 0.14

For each train sample s and each gene g:
    band_{s,g} = [ x_{s,g} - 2 σ_tech(x_{s,g}),
                   x_{s,g} + 2 σ_tech(x_{s,g}) ]

The acceptable region for gene g is the UNION of the 753 train bands:
    accept_g = ⋃_s band_{s,g}

For each held-out sample h, count genes whose value falls outside accept_g.

Library-size correction:
    The noise model is on raw counts in a single library. To make bands
    from differently-sequenced samples comparable, we apply the model on
    raw counts within each sample's native library size, then linearly
    scale the band edges to the train-median reference library. The
    uncertainty propagates linearly with the scale factor.

Outputs:
    blood_technical_noise/
        per_gene_intervals.tsv          # union intervals per gene
        outlier_summary_per_sample.csv
        outlier_summary_per_gene.csv
        blood_technical_noise_outliers.png
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

BG, CARD, TEXT, MUTED, GRID = '#0e1117', '#1a1d23', '#e6edf3', '#7d8590', '#21262d'
plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': CARD, 'axes.edgecolor': GRID,
    'axes.labelcolor': TEXT, 'text.color': TEXT, 'xtick.color': MUTED,
    'ytick.color': MUTED, 'grid.color': GRID, 'grid.alpha': 0.5,
    'font.family': 'sans-serif', 'font.size': 11,
})


def sigma_tech(x):
    return np.sqrt(x + (ALPHA * x) ** 2)


# ── 1. Load + filter ──────────────────────────────────────────────────
print("Loading GTEx whole blood ...")
df = pd.read_csv(GCT, sep="\t", skiprows=2, compression="gzip")
gene_names = df["Description"].astype(str).values
counts = df.iloc[:, 2:].values.astype(np.float64)
n_genes_raw, n_samples = counts.shape
print(f"  raw: {n_genes_raw:,} genes × {n_samples} samples")

lib_all = counts.sum(axis=0)
cpm_all = counts / lib_all * 1e6
expressed = (cpm_all > CPM_THRESHOLD).sum(axis=1) >= int(MIN_SAMPLE_FRAC * n_samples)
counts = counts[expressed]
gene_names = gene_names[expressed]
n_genes = counts.shape[0]
print(f"  expressed: {n_genes:,} genes")


# ── 2. Train / holdout split ──────────────────────────────────────────
rng = np.random.default_rng(SEED)
perm = rng.permutation(n_samples)
counts = counts[:, perm]
test_idx  = np.arange(n_samples - HOLDOUT, n_samples)
train_idx = np.arange(0, n_samples - HOLDOUT)
counts_train = counts[:, train_idx]
counts_test  = counts[:, test_idx]
lib_train = counts_train.sum(axis=0)
lib_test  = counts_test.sum(axis=0)
n_train = counts_train.shape[1]
print(f"  train: {n_train} samples, test: {HOLDOUT} samples")

ref_lib = float(np.median(lib_train))
print(f"  ref library size (train median): {ref_lib:,.0f}")


# ── 3. Per-sample bands on raw counts, then scale to reference ────────
# σ on raw counts; band is x ± 2σ ; scale to reference by L_ref / L_s.
print("\nComputing per-sample bands on raw counts then scaling to reference ...")
sd_raw_train   = sigma_tech(counts_train)                      # genes × n_train
scale_train    = ref_lib / lib_train                            # n_train
lower_train_s  = (counts_train - SD_BAND * sd_raw_train) * scale_train[None, :]
upper_train_s  = (counts_train + SD_BAND * sd_raw_train) * scale_train[None, :]
lower_train_s  = np.maximum(lower_train_s, 0.0)

# Holdout values on the reference scale (a single point per (gene, sample))
counts_test_scaled = counts_test * (ref_lib / lib_test)[None, :]


# ── 4. Sanity-check the noise model against the screenshot table ───────
print("\nNoise-model sanity check (raw counts, α=0.14):")
for m in (1, 10, 100, 1_000, 10_000):
    sd = sigma_tech(m)
    print(f"  μ={m:>6} → σ_tech={sd:6.2f}  ±2σ = [{max(0, m-2*sd):.0f}, {m+2*sd:.0f}]")


# ── 5. For each gene, build merged union of intervals AND
#       answer "is each holdout value inside any interval?" ─────────────
def merge_intervals(starts, ends):
    """Sort by start, merge overlapping/contiguous intervals, return as 2 arrays."""
    order = np.argsort(starts)
    s = starts[order]; e = ends[order]
    out_s = [s[0]]; out_e = [e[0]]
    for i in range(1, len(s)):
        if s[i] <= out_e[-1]:
            out_e[-1] = max(out_e[-1], e[i])
        else:
            out_s.append(s[i]); out_e.append(e[i])
    return np.array(out_s), np.array(out_e)


print("\nMerging per-sample bands into per-gene acceptable regions ...")
n_intervals_per_gene = np.zeros(n_genes, dtype=np.int32)
covered_holdout      = np.zeros((n_genes, HOLDOUT), dtype=bool)
total_band_width     = np.zeros(n_genes)
acceptable_intervals = []  # list of (gene, start_str, end_str) for first 100 genes

for g in range(n_genes):
    starts = lower_train_s[g]
    ends   = upper_train_s[g]
    ms, me = merge_intervals(starts, ends)
    n_intervals_per_gene[g] = len(ms)
    total_band_width[g] = float((me - ms).sum())
    # check holdout values
    y = counts_test_scaled[g]                        # (50,)
    # broadcast: holdout (50,1) vs intervals (1,K)
    in_any = ((y[:, None] >= ms[None, :]) & (y[:, None] <= me[None, :])).any(axis=1)
    covered_holdout[g] = in_any
    if g < 100:
        # Record first 100 genes' merged intervals for human inspection
        intervals_str = "; ".join(f"{s:.1f}-{e:.1f}" for s, e in zip(ms, me))
        acceptable_intervals.append({
            "gene":          gene_names[g],
            "n_intervals":   len(ms),
            "min_lower":     float(ms.min()),
            "max_upper":     float(me.max()),
            "total_width":   float((me - ms).sum()),
            "intervals":     intervals_str[:1000],
        })

print(f"  done; mean n-intervals/gene = {n_intervals_per_gene.mean():.1f}, "
      f"median = {np.median(n_intervals_per_gene)}")

pd.DataFrame(acceptable_intervals).to_csv(
    OUT_DIR / "per_gene_intervals_first100.tsv", sep="\t", index=False
)
# Save the per-gene summary for ALL genes (without the long interval string)
pd.DataFrame({
    "gene":               gene_names,
    "n_merged_intervals": n_intervals_per_gene,
    "total_band_width":   total_band_width,
    "frac_holdouts_outside": 1 - covered_holdout.mean(axis=1),
}).to_csv(OUT_DIR / "per_gene_summary.tsv", sep="\t", index=False)


# ── 6. Per-sample summary ─────────────────────────────────────────────
outside_holdout = ~covered_holdout                   # genes × 50
frac_out_per_sample = outside_holdout.mean(axis=0)   # 50
n_out_per_sample    = outside_holdout.sum(axis=0)    # 50
frac_out_per_gene   = outside_holdout.mean(axis=1)   # n_genes

# Whether holdout value is ABOVE or BELOW the union envelope
# (above = larger than the maximum upper of all intervals; below = smaller than min lower)
gene_min_lower = np.array([float(lower_train_s[g].min()) for g in range(n_genes)])
gene_max_upper = np.array([float(upper_train_s[g].max()) for g in range(n_genes)])
above_envelope = counts_test_scaled > gene_max_upper[:, None]
below_envelope = counts_test_scaled < gene_min_lower[:, None]
in_gap         = outside_holdout & ~above_envelope & ~below_envelope  # in a "hole"

print(f"\nHoldout outlier rates:")
print(f"  per-sample fraction outside union of bands:")
print(f"    mean   = {frac_out_per_sample.mean():.4f}")
print(f"    median = {np.median(frac_out_per_sample):.4f}")
print(f"    range  = {frac_out_per_sample.min():.4f}..{frac_out_per_sample.max():.4f}")
print(f"    expected under pure tech noise (≥1 of 753 covers): "
      f"~{1 - 0.95**753:.2e} (effectively 0)")

print(f"\n  outside-union genes per holdout, decomposed:")
print(f"    above envelope: mean {above_envelope.mean(axis=0).mean():.4f}")
print(f"    below envelope: mean {below_envelope.mean(axis=0).mean():.4f}")
print(f"    in 'hole' between bands: mean {in_gap.mean(axis=0).mean():.4f}")

# Train baseline (leave-one-out approximation): how often is a train sample's
# value outside the union of the OTHER train samples' bands?
# Cheap check: just see whether each train sample's value is in the union built
# from ALL train samples (it should always be — the sample's own band is
# guaranteed to cover its own value).  So compare against a 5/5 simulation:
# what would 5% per-gene technical noise predict if there was no biological
# variance? Approximated via the per-gene observed train SD divided by σ_tech.
obs_sd_train = counts_train_scaled = counts_train * (ref_lib / lib_train)[None, :]  # placeholder
obs_sd_train = counts_train_scaled.std(axis=1)
mean_train  = counts_train_scaled.mean(axis=1)
sd_pred     = sigma_tech(mean_train) * 1.0    # ≈ on the reference scale
ratio       = obs_sd_train / np.where(sd_pred > 0, sd_pred, np.nan)


# Save tables
pd.DataFrame({
    "holdout_idx":          np.arange(HOLDOUT),
    "library_size":         lib_test,
    "n_genes_outside":      n_out_per_sample,
    "frac_genes_outside":   frac_out_per_sample,
    "frac_above_envelope":  above_envelope.mean(axis=0),
    "frac_below_envelope":  below_envelope.mean(axis=0),
    "frac_in_gap":          in_gap.mean(axis=0),
}).to_csv(OUT_DIR / "outlier_summary_per_sample.csv", index=False)


# Top "always-outside" genes
gene_summary = pd.DataFrame({
    "gene":                 gene_names,
    "train_mean_count":     mean_train,
    "train_obs_sd":         obs_sd_train,
    "predicted_sd_tech":    sd_pred,
    "obs_sd_over_tech_sd":  ratio,
    "n_merged_intervals":   n_intervals_per_gene,
    "total_band_width":     total_band_width,
    "frac_holdouts_outside": frac_out_per_gene,
})
gene_summary.sort_values("frac_holdouts_outside", ascending=False) \
            .to_csv(OUT_DIR / "outlier_summary_per_gene.csv", index=False)

print("\nTop 30 genes outside-union for the most holdouts:")
print(gene_summary.sort_values("frac_holdouts_outside", ascending=False)
      .head(30).to_string(index=False))


# ── 7. Visualization ─────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 11))
gs = GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.32)
fig.suptitle(f"Per-sample noise bands → per-gene union → holdout coverage  "
             f"(α={ALPHA}, ±{SD_BAND:.0f}σ)", color=TEXT, fontsize=13)

# (1) Per-sample fraction outside union
ax = fig.add_subplot(gs[0, 0])
ax.hist(frac_out_per_sample, bins=20, color="#3fb950",
        edgecolor="#1f6f33", alpha=0.85)
ax.axvline(frac_out_per_sample.mean(), color="white", ls=":", lw=1.5,
           label=f"mean = {frac_out_per_sample.mean():.3f}")
ax.set_xlabel("fraction of genes outside union of 753 train bands")
ax.set_ylabel("# holdouts")
ax.set_title("Per-holdout outlier rate")
ax.legend(facecolor=CARD, edgecolor=GRID, labelcolor=TEXT, fontsize=8)

# (2) Decomposition: above vs below vs in gap
ax = fig.add_subplot(gs[0, 1])
order = np.argsort(frac_out_per_sample)
ab = above_envelope.mean(axis=0)[order]
be = below_envelope.mean(axis=0)[order]
ga = in_gap.mean(axis=0)[order]
xs = np.arange(HOLDOUT)
ax.bar(xs, ab, color="#f78166", label="above envelope")
ax.bar(xs, be, bottom=ab, color="#58a6ff", label="below envelope")
ax.bar(xs, ga, bottom=ab + be, color="#d2a8ff", label="in gap between bands")
ax.set_xlabel("holdout (sorted by outlier rate)")
ax.set_ylabel("fraction of genes")
ax.set_title("Decomposition")
ax.legend(facecolor=CARD, edgecolor=GRID, labelcolor=TEXT, fontsize=8)

# (3) Per-sample outlier rate vs library size
ax = fig.add_subplot(gs[0, 2])
ax.scatter(lib_test / 1e6, frac_out_per_sample, s=40, color="#3fb950",
           edgecolor="white", linewidth=0.5)
ax.set_xlabel("holdout library size (M reads)")
ax.set_ylabel("fraction outside union")
ax.set_title("Outlier rate vs sequencing depth")

# (4) Per-gene: # of merged intervals (a proxy for how 'fragmented' the
# acceptable region is — many intervals = bimodal/multimodal expression)
ax = fig.add_subplot(gs[1, 0])
ax.hist(n_intervals_per_gene, bins=np.arange(1, n_intervals_per_gene.max() + 2),
        color="#d2a8ff", edgecolor="#7c5fbf")
ax.set_yscale("log")
ax.set_xlabel("# merged intervals per gene")
ax.set_ylabel("# genes")
ax.set_title("Acceptable-region fragmentation")

# (5) Per-gene fraction of holdouts outside
ax = fig.add_subplot(gs[1, 1])
ax.hist(frac_out_per_gene, bins=50, color="#f0883e", edgecolor="#c45a14")
ax.set_xlabel("frac of 50 holdouts where this gene is outside union")
ax.set_ylabel("# genes")
ax.set_yscale("log")
ax.set_title("Per-gene holdout outlier rate")

# (6) obs_SD vs predicted_tech_SD (on reference scale).
# Genes far above the diagonal = biological variance >> technical.
ax = fig.add_subplot(gs[1, 2])
m = (mean_train > 0) & (sd_pred > 0)
sc = ax.scatter(np.log10(sd_pred[m] + 1e-3),
                np.log10(obs_sd_train[m] + 1e-3),
                c=frac_out_per_gene[m], cmap="magma",
                s=2, rasterized=True, alpha=0.6)
lo, hi = ax.get_xlim()
ax.plot([-3, 6], [-3, 6], color="#f78166", ls="--", lw=1.5, label="y=x")
plt.colorbar(sc, ax=ax, label="frac holdouts outside")
ax.set_xlabel("log10 predicted σ_tech (on ref scale)")
ax.set_ylabel("log10 observed train SD")
ax.set_title("Observed vs predicted technical SD")
ax.legend(facecolor=CARD, edgecolor=GRID, labelcolor=TEXT, fontsize=8)

# (7) Top 20 always-outside genes
ax = fig.add_subplot(gs[2, 0])
top20 = gene_summary.sort_values("frac_holdouts_outside", ascending=False).head(20)
ax.barh(np.arange(len(top20))[::-1], top20["frac_holdouts_outside"].values[::-1],
        color="#f78166")
ax.set_yticks(np.arange(len(top20))[::-1])
ax.set_yticklabels(top20["gene"].values[::-1], fontsize=8)
ax.set_xlabel("frac holdouts outside")
ax.set_title("Top 20 'always outside' genes")

# (8) For one gene, show train counts (rug), per-sample bands as horizontal
# lines, and the 50 holdout values overlaid.
ax = fig.add_subplot(gs[2, 1])
top_gene_g = int(np.argsort(frac_out_per_gene)[-1])
xt = counts_train_scaled[top_gene_g]
yt = counts_test_scaled[top_gene_g]
ms_g, me_g = merge_intervals(lower_train_s[top_gene_g], upper_train_s[top_gene_g])
for s, e in zip(ms_g, me_g):
    ax.axvspan(s, e, color="#3fb950", alpha=0.18)
ax.scatter(xt, np.zeros_like(xt) + 0.4, marker="|",
           color="#58a6ff", alpha=0.5, s=40, label="train")
y_in = yt[covered_holdout[top_gene_g]]
y_out = yt[~covered_holdout[top_gene_g]]
ax.scatter(y_in,  np.zeros_like(y_in)  + 0.7, color="#3fb950",
           edgecolor="white", s=50, label="holdout in")
ax.scatter(y_out, np.zeros_like(y_out) + 0.7, color="#f78166",
           edgecolor="white", s=60, label="holdout outside")
ax.set_yticks([])
ax.set_xlabel("scaled count")
ax.set_xscale("symlog", linthresh=1)
ax.set_title(f"{gene_names[top_gene_g]}: {len(ms_g)} intervals, "
             f"{int(outside_holdout[top_gene_g].sum())}/50 outside")
ax.legend(facecolor=CARD, edgecolor=GRID, labelcolor=TEXT, fontsize=8, loc="upper right")

# (9) Per-gene outlier rate vs train mean
ax = fig.add_subplot(gs[2, 2])
log_mu = np.log10(mean_train + 1e-3)
ax.scatter(log_mu, frac_out_per_gene, s=2, color="#d2a8ff",
           alpha=0.4, rasterized=True)
bins = np.linspace(log_mu.min(), log_mu.max(), 25)
binc = 0.5 * (bins[:-1] + bins[1:])
mean_in_bin = np.array([
    frac_out_per_gene[(log_mu >= b0) & (log_mu < b1)].mean()
    if ((log_mu >= b0) & (log_mu < b1)).any() else np.nan
    for b0, b1 in zip(bins[:-1], bins[1:])
])
ax.plot(binc, mean_in_bin, color="#f78166", lw=2, label="binned mean")
ax.set_xlabel("log10 train mean count (ref scale)")
ax.set_ylabel("frac holdouts outside")
ax.set_title("Outlier rate vs expression")
ax.legend(facecolor=CARD, edgecolor=GRID, labelcolor=TEXT, fontsize=8)

fig.tight_layout()
out_png = OUT_DIR / "blood_technical_noise_outliers.png"
fig.savefig(out_png, dpi=150, facecolor=BG, bbox_inches="tight")
plt.close(fig)
print(f"\nWrote {out_png}")
print(f"Wrote {OUT_DIR/'outlier_summary_per_sample.csv'}")
print(f"Wrote {OUT_DIR/'outlier_summary_per_gene.csv'}")
print(f"Wrote {OUT_DIR/'per_gene_intervals_first100.tsv'}")
print(f"Wrote {OUT_DIR/'per_gene_summary.tsv'}")
