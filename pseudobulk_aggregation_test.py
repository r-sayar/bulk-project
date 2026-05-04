"""
Pseudobulk Aggregation Test
============================
Key question: is HCA's low CV (0.059) because it has only 8 donors,
or because summing many samples together mathematically reduces variance?

Test: randomly pool GTEx's 803 samples into 8 groups of ~100,
sum their raw counts (mimicking pseudobulk aggregation), then compute CV.

If CV drops to ~HCA level  → aggregation itself (averaging) causes low CV.
If CV stays near 0.25      → HCA's low CV is truly about having few donors.

We run N_ITER random groupings to check robustness.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import gaussian_kde
import warnings
warnings.filterwarnings('ignore')

# ── paths ──────────────────────────────────────────────────────────────
BASEDIR   = '/Users/rls/Desktop/programming-projects/single-cell/bulk-project'
GTEX_PATH = '/Users/rls/Downloads/gene_reads_v11_whole_blood.gct.gz'
PSEUDO    = f'{BASEDIR}/pseudobulk/hca_blood_pseudobulk.npz'

# ── style ──────────────────────────────────────────────────────────────
BG = '#0e1117'; CARD = '#1a1d23'; TEXT = '#e6edf3'; MUTED = '#7d8590'; GRID = '#21262d'
C_G = '#f78166'; C_H = '#3fb950'; ACCENT1 = '#58a6ff'; ACCENT4 = '#d2a8ff'
C_AGG = '#f0883e'

plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': CARD, 'axes.edgecolor': GRID,
    'axes.labelcolor': TEXT, 'text.color': TEXT, 'xtick.color': MUTED,
    'ytick.color': MUTED, 'grid.color': GRID, 'grid.alpha': 0.5,
    'font.family': 'sans-serif', 'font.size': 11,
})

CPM_THRESHOLD   = 1
MIN_SAMPLE_FRAC = 0.1
N_GROUPS        = 8     # match HCA donor count
N_ITER          = 100   # random grouping iterations
# Also test different aggregation sizes
AGG_SIZES = [1, 2, 4, 8, 16, 32, 64, 100, 200]
N_ITER_CURVE = 50


def cpm_log(expr_raw):
    lib = expr_raw.sum(axis=0)
    cpm = expr_raw / lib * 1e6
    return np.log2(cpm + 1), cpm


def filter_genes(expr_raw, gene_names, n_samples=None):
    if n_samples is None:
        n_samples = expr_raw.shape[1]
    lib  = expr_raw.sum(axis=0)
    cpm  = expr_raw / lib * 1e6
    min_s = max(1, int(MIN_SAMPLE_FRAC * n_samples))
    keep  = (cpm > CPM_THRESHOLD).sum(axis=1) >= min_s
    return expr_raw[keep], gene_names[keep]


def cv_of(expr_log, min_mean=0.5):
    means = expr_log.mean(axis=1)
    stds  = expr_log.std(axis=1)
    expressed = means > min_mean
    cvs = np.full(len(means), np.nan)
    cvs[expressed] = stds[expressed] / means[expressed]
    return cvs, means, stds, expressed


def aggregate_to_n(expr_raw, n_groups, rng):
    """
    Randomly split 803 samples into n_groups groups, sum raw counts within
    each group. Returns aggregated matrix of shape (genes, n_groups).
    """
    n_genes, n_samples = expr_raw.shape
    # shuffle sample indices and split into n_groups roughly equal chunks
    idx = rng.permutation(n_samples)
    chunks = np.array_split(idx, n_groups)
    agg = np.zeros((n_genes, n_groups), dtype=np.float64)
    for g, chunk in enumerate(chunks):
        agg[:, g] = expr_raw[:, chunk].sum(axis=1)
    return agg


def run_aggregation_cv(expr_raw, gene_names, n_groups, n_iter, rng):
    """
    Aggregate to n_groups n_iter times, compute CV each time.
    Returns arrays of median_cv and mean_cv.
    """
    med_cvs = np.full(n_iter, np.nan)
    for it in range(n_iter):
        agg = aggregate_to_n(expr_raw, n_groups, rng)
        # filter genes on the aggregated matrix
        agg_filt, _ = filter_genes(agg, gene_names, n_samples=n_groups)
        e_log, _ = cpm_log(agg_filt)
        cvs, _, _, _ = cv_of(e_log)
        valid = cvs[~np.isnan(cvs)]
        med_cvs[it] = np.median(valid)
    return med_cvs


# ══════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════

print("Loading GTEx whole blood ...")
df = pd.read_csv(GTEX_PATH, sep='\t', skiprows=2, compression='gzip')
expr_raw   = df.iloc[:, 2:].values.astype(np.float64)
gene_names = df['Description'].values.astype(str)
n_genes, n_samples = expr_raw.shape
print(f"  {n_genes:,} genes × {n_samples} samples")

# GTEx full-dataset baseline
expr_filt_full, names_full = filter_genes(expr_raw, gene_names)
e_log_full, _ = cpm_log(expr_filt_full)
cvs_full, _, _, _ = cv_of(e_log_full)
gtex_full_med = np.nanmedian(cvs_full)
print(f"  GTEx full (n=803) median CV = {gtex_full_med:.4f}")

# HCA pseudobulk baseline
print("Loading HCA pseudobulk ...")
d = np.load(PSEUDO, allow_pickle=True)
expr_raw_h   = d['expr'].astype(np.float64)
gene_names_h = d['gene_names'].astype(str)
expr_filt_h, _ = filter_genes(expr_raw_h, gene_names_h)
e_log_h, _ = cpm_log(expr_filt_h)
cvs_h, _, _, _ = cv_of(e_log_h)
hca_med = np.nanmedian(cvs_h[~np.isnan(cvs_h)])
print(f"  HCA pseudobulk (n=8 donors) median CV = {hca_med:.4f}")

# ══════════════════════════════════════════════════════════════════════
# MAIN EXPERIMENT: aggregate GTEx 803 → 8
# ══════════════════════════════════════════════════════════════════════

rng = np.random.default_rng(42)

print(f"\n{'='*65}")
print(f"AGGREGATING GTEx 803 SAMPLES → {N_GROUPS} PSEUDO-GROUPS ({N_ITER} random splits)")
print(f"{'='*65}")

med_cvs_agg8 = run_aggregation_cv(expr_raw, gene_names, N_GROUPS, N_ITER, rng)

print(f"\nGTEx aggregated to n={N_GROUPS}:")
print(f"  Median CV — mean={med_cvs_agg8.mean():.4f}  "
      f"std={med_cvs_agg8.std():.4f}  "
      f"range=[{med_cvs_agg8.min():.4f}, {med_cvs_agg8.max():.4f}]")
print(f"\nComparison:")
print(f"  {'Condition':<40} {'Median CV':>10}")
print(f"  {'-'*52}")
print(f"  {'GTEx full (n=803 individual samples)':<40} {gtex_full_med:>10.4f}")
print(f"  {'GTEx aggregated 803→8 (mean over runs)':<40} {med_cvs_agg8.mean():>10.4f}")
print(f"  {'HCA pseudobulk (n=8 real donors)':<40} {hca_med:>10.4f}")

gap_full_to_hca = gtex_full_med - hca_med
gap_agg_to_hca  = med_cvs_agg8.mean() - hca_med
pct_closed = (1 - gap_agg_to_hca / gap_full_to_hca) * 100
print(f"\n  Gap closed by aggregation: {pct_closed:.1f}%  "
      f"(full gap = {gap_full_to_hca:.4f}, remaining after agg = {gap_agg_to_hca:.4f})")

# ══════════════════════════════════════════════════════════════════════
# CV-vs-aggregation-size CURVE
# ══════════════════════════════════════════════════════════════════════

print(f"\nCV vs aggregation size curve ...")
agg_curve = []
for agg_n in AGG_SIZES:
    n_it = N_ITER if agg_n <= 8 else N_ITER_CURVE
    meds = run_aggregation_cv(expr_raw, gene_names, agg_n, n_it, rng)
    agg_curve.append({
        'n': agg_n,
        'med_med': np.median(meds),
        'med_q10': np.percentile(meds, 10),
        'med_q90': np.percentile(meds, 90),
        'all_meds': meds,
    })
    print(f"  agg to n={agg_n:>3}: median CV = {np.median(meds):.4f}  "
          f"[{np.percentile(meds,10):.4f} – {np.percentile(meds,90):.4f}]")

# ══════════════════════════════════════════════════════════════════════
# ONE EXAMPLE: full CV distribution of aggregated vs original
# ══════════════════════════════════════════════════════════════════════

# get one specific aggregated run's full CV distribution for plotting
rng2 = np.random.default_rng(0)
agg_example = aggregate_to_n(expr_raw, N_GROUPS, rng2)
agg_filt, _ = filter_genes(agg_example, gene_names, n_samples=N_GROUPS)
e_log_agg, _ = cpm_log(agg_filt)
cvs_agg, means_agg, _, _ = cv_of(e_log_agg)
cvs_agg_valid = cvs_agg[~np.isnan(cvs_agg)]

# ══════════════════════════════════════════════════════════════════════
# FIGURE
# ══════════════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(24, 16))
gs  = GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.3,
               left=0.06, right=0.97, top=0.91, bottom=0.06)
fig.suptitle(
    'Does Aggregating GTEx 803→8 Reproduce HCA\'s Low CV?',
    fontsize=17, fontweight='bold', color=TEXT, y=0.97)

# ── Panel 1: bootstrap distribution of aggregated median CV ──────────
ax = fig.add_subplot(gs[0, 0])
ax.hist(med_cvs_agg8, bins=30, color=C_AGG, alpha=0.8, edgecolor='none',
        density=True, label=f'GTEx 803→8 ({N_ITER} random splits)')
ax.axvline(med_cvs_agg8.mean(), color=C_AGG, ls='--', lw=2.5,
           label=f'GTEx agg mean = {med_cvs_agg8.mean():.4f}')
ax.axvline(gtex_full_med, color=C_G, ls=':', lw=2,
           label=f'GTEx full (n=803) = {gtex_full_med:.4f}')
ax.axvline(hca_med, color=C_H, ls='-', lw=2.5,
           label=f'HCA pseudobulk = {hca_med:.4f}')
ax.set_xlabel('Median CV')
ax.set_ylabel('Density')
ax.set_title(f'Distribution of Median CV\nGTEx Aggregated 803→{N_GROUPS}', fontsize=13, fontweight='bold', pad=10)
ax.legend(fontsize=9.5, facecolor=CARD, edgecolor=GRID, labelcolor=TEXT)
ax.grid(axis='y', alpha=0.3)

# ── Panel 2: CV-vs-aggregation-size curve ───────────────────────────
ax = fig.add_subplot(gs[0, 1])
ns   = [d['n']       for d in agg_curve]
meds = [d['med_med'] for d in agg_curve]
lo   = [d['med_q10'] for d in agg_curve]
hi   = [d['med_q90'] for d in agg_curve]
ax.plot(ns, meds, color=C_AGG, lw=2.5, marker='o', ms=7, label='GTEx aggregated')
ax.fill_between(ns, lo, hi, alpha=0.25, color=C_AGG, label='10th–90th pct')
ax.axhline(hca_med, color=C_H, ls='-', lw=2, label=f'HCA pseudobulk ({hca_med:.4f})')
ax.axhline(gtex_full_med, color=C_G, ls=':', lw=2, label=f'GTEx full ({gtex_full_med:.4f})')
ax.set_xscale('log')
ax.set_xlabel('Number of groups after aggregation (log scale)')
ax.set_ylabel('Median CV')
ax.set_title('Median CV vs Aggregation Level\n(how much does pooling reduce CV?)', fontsize=13, fontweight='bold', pad=10)
ax.legend(fontsize=9, facecolor=CARD, edgecolor=GRID, labelcolor=TEXT)
ax.grid(alpha=0.3)

# ── Panel 3: Three-way CV distribution overlay ──────────────────────
ax = fig.add_subplot(gs[0, 2])
cvs_gtex_valid = cvs_full[~np.isnan(cvs_full)]
cvs_hca_valid  = cvs_h[~np.isnan(cvs_h)]

ax.hist(cvs_gtex_valid, bins=200, range=(0, 1.5), density=True,
        alpha=0.45, color=C_G, edgecolor='none',
        label=f'GTEx full  n=803  med={np.median(cvs_gtex_valid):.3f}')
ax.hist(cvs_agg_valid, bins=100, range=(0, 1.5), density=True,
        alpha=0.55, color=C_AGG, edgecolor='none',
        label=f'GTEx agg 803→8  med={np.median(cvs_agg_valid):.3f}')
ax.hist(cvs_hca_valid, bins=80, range=(0, 1.5), density=True,
        alpha=0.55, color=C_H, edgecolor='none',
        label=f'HCA pseudobulk  n=8  med={np.median(cvs_hca_valid):.3f}')
for med, col in [(np.median(cvs_gtex_valid), C_G),
                 (np.median(cvs_agg_valid),  C_AGG),
                 (np.median(cvs_hca_valid),  C_H)]:
    ax.axvline(med, color=col, ls='--', lw=1.8)
ax.set_xlabel('CV')
ax.set_ylabel('Density')
ax.set_title('Full CV Distribution — Three-Way Comparison', fontsize=13, fontweight='bold', pad=10)
ax.legend(fontsize=9.5, facecolor=CARD, edgecolor=GRID, labelcolor=TEXT)
ax.grid(axis='y', alpha=0.3)

# Build shared gene lookup once using the full filtered gene set
# (agg_example was filtered from the same gene_names, just subset of rows)
agg_lib   = agg_example.sum(axis=0)
agg_cpm_m = agg_example / agg_lib * 1e6
agg_keep  = (agg_cpm_m > CPM_THRESHOLD).sum(axis=1) >= 1
agg_names_filt = gene_names[agg_keep]
agg_expr_filt  = agg_example[agg_keep]
agg_elog, _    = cpm_log(agg_expr_filt)
agg_means_all  = agg_elog.mean(axis=1)
agg_cvs_all, _, _, _ = cv_of(agg_elog)

g_name_upper = np.array([n.upper() for n in names_full])
a_name_upper = np.array([n.upper() for n in agg_names_filt])

g_map_mean = dict(zip(g_name_upper, e_log_full.mean(axis=1)))
a_map_mean = dict(zip(a_name_upper, agg_means_all))
g_map_cv   = {n: cvs_full[i] for i, n in enumerate(g_name_upper) if not np.isnan(cvs_full[i])}
a_map_cv   = {n: agg_cvs_all[i] for i, n in enumerate(a_name_upper) if not np.isnan(agg_cvs_all[i])}

shared_mean = sorted(set(g_map_mean.keys()) & set(a_map_mean.keys()))
gm = np.array([g_map_mean[n] for n in shared_mean])
am = np.array([a_map_mean[n] for n in shared_mean])

shared_cv_names = sorted(set(g_map_cv.keys()) & set(a_map_cv.keys()))
gcv = np.array([g_map_cv[n] for n in shared_cv_names])
acv = np.array([a_map_cv[n] for n in shared_cv_names])

from scipy.stats import pearsonr

# ── Panel 4: mean expression scatter ─────────────────────────────────
ax = fig.add_subplot(gs[1, 0])
ax.scatter(gm, am, s=1.5, alpha=0.2, c=ACCENT4, rasterized=True)
lim = max(gm.max(), am.max()) * 1.02
ax.plot([0, lim], [0, lim], '--', color=MUTED, lw=1.5, label='y = x')
r, _ = pearsonr(gm, am)
ax.text(0.05, 0.95, f'Pearson r = {r:.4f}\nn = {len(shared_mean):,}',
        transform=ax.transAxes, fontsize=11, color=TEXT, va='top',
        bbox=dict(facecolor=CARD, edgecolor=GRID, alpha=0.85, pad=4))
ax.set_xlabel('GTEx full — Mean log₂(CPM+1)')
ax.set_ylabel('GTEx aggregated 803→8 — Mean log₂(CPM+1)')
ax.set_title('Mean Expression: Full vs Aggregated GTEx\n(sanity check — should be ~identical)',
             fontsize=13, fontweight='bold', pad=10)
ax.legend(fontsize=9, facecolor=CARD, edgecolor=GRID, labelcolor=TEXT)
ax.grid(alpha=0.3)

# ── Panel 5: CV scatter ───────────────────────────────────────────────
ax = fig.add_subplot(gs[1, 1])
ax.scatter(gcv, acv, s=1.5, alpha=0.2, c=C_AGG, rasterized=True)
lim_cv = min(max(gcv.max(), acv.max()) * 1.02, 3.0)
ax.plot([0, lim_cv], [0, lim_cv], '--', color=MUTED, lw=1.5, label='y = x')
r_cv, _ = pearsonr(np.log1p(gcv), np.log1p(acv))
ax.text(0.05, 0.95, f'r (log CV) = {r_cv:.4f}\nn = {len(shared_cv_names):,}',
        transform=ax.transAxes, fontsize=11, color=TEXT, va='top',
        bbox=dict(facecolor=CARD, edgecolor=GRID, alpha=0.85, pad=4))
ax.set_xlabel('GTEx full CV')
ax.set_ylabel('GTEx aggregated 803→8 CV')
ax.set_title('CV Correlation: Full vs Aggregated GTEx',
             fontsize=13, fontweight='bold', pad=10)
ax.legend(fontsize=9, facecolor=CARD, edgecolor=GRID, labelcolor=TEXT)
ax.grid(alpha=0.3)

# ── Panel 6: Summary bar chart ────────────────────────────────────────
ax = fig.add_subplot(gs[1, 2])
labels   = ['GTEx\nfull (n=803)', f'GTEx\nagg 803→8', 'HCA\npseudob. (n=8)']
medians  = [gtex_full_med, med_cvs_agg8.mean(), hca_med]
errors   = [0, med_cvs_agg8.std(), 0]
colors_b = [C_G, C_AGG, C_H]

bars = ax.bar(range(3), medians, color=colors_b, alpha=0.8, edgecolor='none', width=0.55)
ax.errorbar(range(3), medians, yerr=errors, fmt='none', color=TEXT,
            capsize=8, lw=2, capthick=2)
for i, (v, e) in enumerate(zip(medians, errors)):
    label = f'{v:.4f}' if e == 0 else f'{v:.4f}\n±{e:.4f}'
    ax.text(i, v + max(errors) * 0.15 + 0.003, label,
            ha='center', fontsize=11, color=TEXT, fontweight='bold')

# annotate gap arrows
y_arrow = max(medians) * 1.18
ax.annotate('', xy=(2, hca_med), xytext=(1, med_cvs_agg8.mean()),
            arrowprops=dict(arrowstyle='<->', color=MUTED, lw=2))
ax.text(1.5, (hca_med + med_cvs_agg8.mean()) / 2,
        f'residual\ngap = {med_cvs_agg8.mean() - hca_med:.4f}',
        ha='center', fontsize=9, color=MUTED)

ax.set_xticks(range(3))
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylabel('Median CV (across all expressed genes)')
ax.set_title('Summary: Median CV by Condition',
             fontsize=13, fontweight='bold', pad=10)
ax.set_ylim(0, max(medians) * 1.45)
ax.grid(axis='y', alpha=0.3)

plt.savefig(f'{BASEDIR}/pseudobulk_aggregation_test.png',
            dpi=180, bbox_inches='tight', facecolor=BG)
plt.close()
print(f"\nSaved: {BASEDIR}/pseudobulk_aggregation_test.png")
print("\nAll done.")
