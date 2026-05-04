"""
Sample-Size Effect and Sequencing-Depth Confounding
====================================================
Two analyses:

A) BOOTSTRAP SUBSAMPLING (n=8 from GTEx 803)
   Repeatedly draw 8 random GTEx donors (matching HCA n), compute CV each time.
   This separates: how much of the GTEx vs HCA CV gap is just because of
   n=803 vs n=8, vs how much is real biological signal or technical difference.

B) LIBRARY-SIZE BINNING
   GTEx sequencing depth spans 10 M – 170 M reads per sample.
   Even after CPM normalization the *variance* of CPM estimates can still
   depend on depth (Poisson noise: CV_noise ≈ 1/sqrt(depth)).
   Test: bin samples by library size, compute per-gene CV in each bin,
   check whether high-depth bins have systematically lower CV.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import gaussian_kde, pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

# ── paths ──────────────────────────────────────────────────────────────
BASEDIR   = '/Users/rls/Desktop/programming-projects/single-cell/bulk-project'
GTEX_PATH = '/Users/rls/Downloads/gene_reads_v11_whole_blood.gct.gz'
PSEUDO    = f'{BASEDIR}/pseudobulk/hca_blood_pseudobulk.npz'

# ── style ──────────────────────────────────────────────────────────────
BG = '#0e1117'; CARD = '#1a1d23'; TEXT = '#e6edf3'; MUTED = '#7d8590'; GRID = '#21262d'
C_G = '#f78166'; C_H = '#3fb950'; ACCENT1 = '#58a6ff'; ACCENT4 = '#d2a8ff'
BIN_COLORS = ['#58a6ff', '#3fb950', '#f0883e', '#d2a8ff', '#f78166']

plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': CARD, 'axes.edgecolor': GRID,
    'axes.labelcolor': TEXT, 'text.color': TEXT, 'xtick.color': MUTED,
    'ytick.color': MUTED, 'grid.color': GRID, 'grid.alpha': 0.5,
    'font.family': 'sans-serif', 'font.size': 11,
})

# ══════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════

CPM_THRESHOLD   = 1
MIN_SAMPLE_FRAC = 0.1


def load_gtex_raw(path):
    """Load GTEx blood, return raw counts + gene names (no filtering yet)."""
    print("Loading GTEx whole blood ...")
    df = pd.read_csv(path, sep='\t', skiprows=2, compression='gzip')
    expr_raw   = df.iloc[:, 2:].values.astype(np.float64)   # (genes, samples)
    gene_names = df['Description'].values.astype(str)
    print(f"  Shape: {expr_raw.shape[0]:,} genes × {expr_raw.shape[1]} samples")
    return expr_raw, gene_names


def filter_genes(expr_raw, gene_names):
    """CPM filter: keep genes with CPM > 1 in ≥10% of samples."""
    lib   = expr_raw.sum(axis=0)
    cpm   = expr_raw / lib * 1e6
    min_s = max(1, int(MIN_SAMPLE_FRAC * expr_raw.shape[1]))
    keep  = (cpm > CPM_THRESHOLD).sum(axis=1) >= min_s
    return expr_raw[keep], gene_names[keep]


def cpm_log(expr_raw):
    """CPM-normalize then log2(CPM+1)."""
    lib = expr_raw.sum(axis=0)
    cpm = expr_raw / lib * 1e6
    return np.log2(cpm + 1), cpm


def cv_of(expr_log, min_mean=0.5):
    """Per-gene CV on log2 scale. Only for genes with mean > min_mean."""
    means = expr_log.mean(axis=1)
    stds  = expr_log.std(axis=1)
    expressed = means > min_mean
    cvs = np.full(len(means), np.nan)
    cvs[expressed] = stds[expressed] / means[expressed]
    return cvs, means, stds, expressed


def load_hca(path):
    print("Loading HCA pseudobulk ...")
    d = np.load(path, allow_pickle=True)
    expr_raw   = d['expr'].astype(np.float64)
    gene_names = d['gene_names'].astype(str)
    expr_filt, names_filt = filter_genes(expr_raw, gene_names)
    e_log, e_cpm = cpm_log(expr_filt)
    cvs, means, stds, expr = cv_of(e_log)
    hca_cvs = cvs[~np.isnan(cvs)]
    print(f"  {len(names_filt):,} genes | median CV = {np.nanmedian(cvs):.4f}")
    return hca_cvs, np.nanmedian(cvs), np.nanmean(cvs)


# ══════════════════════════════════════════════════════════════════════
# A. BOOTSTRAP SUBSAMPLING
# For each bootstrap iteration:
#   1. Randomly draw N_BOOT_DONORS samples from the 803 GTEx donors
#   2. Re-filter genes (CPM > 1 in ≥10% of SELECTED samples)
#   3. CPM + log2 transform
#   4. Compute median CV and mean CV across all expressed genes
# Do this N_ITER times to get the sampling distribution.
# ══════════════════════════════════════════════════════════════════════

N_BOOT_DONORS = 8      # match HCA
N_ITER        = 300    # bootstrap iterations
RNG           = np.random.default_rng(42)

# Also test multiple subsampling sizes to draw the CV-vs-n curve
N_SIZES = [4, 8, 16, 32, 64, 128, 256, 512, 803]
N_ITER_CURVE = 100   # fewer iterations per size for the curve (speed)


def bootstrap_cv(expr_raw, gene_names, n_donors, n_iter, rng, min_mean=0.5):
    """
    Run n_iter bootstrap rounds drawing n_donors columns from expr_raw.
    Returns arrays of median_cv and mean_cv, one per iteration.
    """
    n_genes, n_samples = expr_raw.shape
    med_cvs  = np.full(n_iter, np.nan)
    mean_cvs = np.full(n_iter, np.nan)
    q25_cvs  = np.full(n_iter, np.nan)
    q75_cvs  = np.full(n_iter, np.nan)

    min_s = max(1, int(MIN_SAMPLE_FRAC * n_donors))

    for it in range(n_iter):
        cols = rng.choice(n_samples, n_donors, replace=False)
        e    = expr_raw[:, cols]

        # re-filter genes for this subsample
        lib  = e.sum(axis=0)
        cpm  = e / lib * 1e6
        keep = (cpm > CPM_THRESHOLD).sum(axis=1) >= min_s
        e    = e[keep]

        e_log, _ = cpm_log(e)
        cvs, _, _, _ = cv_of(e_log, min_mean)
        valid = cvs[~np.isnan(cvs)]

        med_cvs[it]  = np.median(valid)
        mean_cvs[it] = valid.mean()
        q25_cvs[it]  = np.percentile(valid, 25)
        q75_cvs[it]  = np.percentile(valid, 75)

    return med_cvs, mean_cvs, q25_cvs, q75_cvs


# ══════════════════════════════════════════════════════════════════════
# B. LIBRARY-SIZE BINNING
# Bin GTEx samples into depth quintiles (equal number of samples per bin).
# For each bin, compute per-gene CV. Then:
#   1. Plot CV distribution per bin → do they shift?
#   2. Plot median CV per bin vs median library size → is there a trend?
#   3. Scatter: library size vs per-gene CV for a fixed set of genes
#   4. Spearman r between library size and CV for each gene → distribution
# ══════════════════════════════════════════════════════════════════════

N_BINS = 5   # quintiles


def depth_bin_analysis(expr_raw, gene_names, n_bins=N_BINS):
    """
    For each depth quintile of samples, compute per-gene CV.
    Returns dict with bin info and CV arrays.
    """
    lib_sizes = expr_raw.sum(axis=0)
    bin_edges = np.percentile(lib_sizes, np.linspace(0, 100, n_bins + 1))
    bin_labels = [
        f'Q{i+1}: {bin_edges[i]/1e6:.0f}–{bin_edges[i+1]/1e6:.0f}M'
        for i in range(n_bins)
    ]
    print(f"\nLibrary size range: {lib_sizes.min()/1e6:.1f}M – {lib_sizes.max()/1e6:.1f}M")
    print(f"Bin edges (M reads): {[f'{e/1e6:.1f}' for e in bin_edges]}")

    # Filter genes on the FULL dataset first (consistent gene set)
    expr_filt, names_filt = filter_genes(expr_raw, gene_names)

    bins = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (lib_sizes >= lo) & (lib_sizes <= hi)
        else:
            mask = (lib_sizes >= lo) & (lib_sizes < hi)
        e_bin = expr_filt[:, mask]
        e_log, e_cpm = cpm_log(e_bin)
        cvs, means, stds, expressed = cv_of(e_log)
        valid_cvs = cvs[~np.isnan(cvs)]
        med_lib = np.median(lib_sizes[mask])
        print(f"  Bin {i+1} ({bin_labels[i]}): {mask.sum()} samples, "
              f"median_lib={med_lib/1e6:.1f}M, "
              f"median_CV={np.median(valid_cvs):.4f}, "
              f"mean_CV={np.mean(valid_cvs):.4f}")
        bins.append({
            'label':    bin_labels[i],
            'n':        mask.sum(),
            'lib_med':  med_lib,
            'lib_vals': lib_sizes[mask],
            'cvs':      cvs,
            'valid_cvs': valid_cvs,
            'means':    means,
            'stds':     stds,
        })

    # Per-gene: Spearman correlation between library size and CPM-normalized expression
    # (test whether depth still affects per-gene variability after CPM)
    e_log_full, _ = cpm_log(expr_filt)
    cvs_full, _, _, expressed_full = cv_of(e_log_full)

    # For each expressed gene, compute Spearman r between sample lib sizes
    # and that gene's log-CPM values across all 803 samples
    print("\nComputing per-gene Spearman r (library size vs log-expression) ...")
    lib_ranks = lib_sizes.argsort().argsort()  # rank once
    expressed_idx = np.where(expressed_full)[0]
    n_expr = len(expressed_idx)
    spear_r = np.full(n_expr, np.nan)

    for j, gi in enumerate(expressed_idx):
        r, _ = spearmanr(lib_sizes, e_log_full[gi, :])
        spear_r[j] = r

    print(f"  Median |r|: {np.abs(spear_r).mean():.4f}  (0 = no depth confounding)")
    print(f"  Fraction |r| > 0.3: {(np.abs(spear_r) > 0.3).mean()*100:.1f}%")
    print(f"  Fraction |r| > 0.5: {(np.abs(spear_r) > 0.5).mean()*100:.1f}%")

    # CVs for expressed genes only (aligned with spear_r)
    cvs_expr = cvs_full[expressed_idx]

    return bins, bin_labels, lib_sizes, expr_filt, names_filt, cvs_full, spear_r, cvs_expr


# ══════════════════════════════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════════════════════════════

def fig_bootstrap(med_cvs_8, mean_cvs_8, q25_8, q75_8,
                  hca_med, hca_mean, gtex_full_med, gtex_full_mean,
                  cv_curve, out):
    """
    3-panel figure:
      1. Distribution of median CV from 300 bootstraps at n=8  vs HCA median
      2. CV-vs-n curve (how median CV drops as subsample size grows)
      3. Box plot showing spread of bootstrap CVs at different n
    """
    fig = plt.figure(figsize=(22, 9))
    gs  = GridSpec(1, 3, figure=fig, hspace=0.35, wspace=0.32,
                   left=0.06, right=0.97, top=0.88, bottom=0.09)
    fig.suptitle(
        'Sample-Size Effect on CV: Bootstrap Subsampling from GTEx (n=803)',
        fontsize=16, fontweight='bold', color=TEXT, y=0.97)

    # ── panel 1: bootstrap distribution at n=8 ──────────────────────
    ax = fig.add_subplot(gs[0])
    ax.hist(med_cvs_8, bins=40, color=C_G, alpha=0.75, edgecolor='none',
            density=True, label=f'GTEx subsampled (n={N_BOOT_DONORS}, {N_ITER} draws)')
    ax.axvline(np.median(med_cvs_8), color=C_G, ls='--', lw=2,
               label=f'GTEx sub median = {np.median(med_cvs_8):.4f}')
    ax.axvline(hca_med, color=C_H, ls='-', lw=2.5,
               label=f'HCA pseudobulk = {hca_med:.4f}')
    ax.axvline(gtex_full_med, color=ACCENT1, ls=':', lw=2,
               label=f'GTEx full (n=803) = {gtex_full_med:.4f}')

    # annotate gap
    gap = np.median(med_cvs_8) - hca_med
    ax.annotate('', xy=(hca_med, 0.55), xytext=(np.median(med_cvs_8), 0.55),
                xycoords=('data', 'axes fraction'), textcoords=('data', 'axes fraction'),
                arrowprops=dict(arrowstyle='<->', color=TEXT, lw=1.5))
    ax.text((hca_med + np.median(med_cvs_8)) / 2, 0.58,
            f'residual gap\n= {gap:.4f}',
            ha='center', va='bottom', transform=ax.get_xaxis_transform(),
            fontsize=9, color=TEXT)

    ax.set_xlabel('Median CV across all expressed genes')
    ax.set_ylabel('Density')
    ax.set_title(f'Bootstrap Distribution of Median CV\n(n={N_BOOT_DONORS} donors, {N_ITER} draws)',
                 fontsize=13, fontweight='bold', pad=10)
    ax.legend(fontsize=9, facecolor=CARD, edgecolor=GRID, labelcolor=TEXT)
    ax.grid(axis='y', alpha=0.3)

    # ── panel 2: median CV vs n curve ───────────────────────────────
    ax = fig.add_subplot(gs[1])
    ns   = [d['n']       for d in cv_curve]
    meds = [d['med_med'] for d in cv_curve]
    lo   = [d['med_q10'] for d in cv_curve]
    hi   = [d['med_q90'] for d in cv_curve]

    ax.plot(ns, meds, color=C_G, lw=2.5, marker='o', ms=6, label='GTEx subsampled')
    ax.fill_between(ns, lo, hi, alpha=0.25, color=C_G, label='10th–90th pct')
    ax.axhline(hca_med, color=C_H, ls='-', lw=2, label=f'HCA ({hca_med:.4f})')
    ax.axhline(gtex_full_med, color=ACCENT1, ls=':', lw=2, label=f'GTEx full ({gtex_full_med:.4f})')
    ax.set_xscale('log')
    ax.set_xlabel('Number of donors (log scale)')
    ax.set_ylabel('Median CV')
    ax.set_title('Median CV vs Sample Size\n(how fast does CV stabilize?)',
                 fontsize=13, fontweight='bold', pad=10)
    ax.legend(fontsize=9, facecolor=CARD, edgecolor=GRID, labelcolor=TEXT)
    ax.grid(alpha=0.3)

    # annotate at n=8
    idx8 = next(i for i, d in enumerate(cv_curve) if d['n'] == N_BOOT_DONORS)
    ax.annotate(f'n=8: {meds[idx8]:.4f}',
                xy=(8, meds[idx8]), xytext=(14, meds[idx8] + 0.005),
                fontsize=9, color=C_G,
                arrowprops=dict(arrowstyle='->', color=C_G, lw=1.2))

    # ── panel 3: box plot at each n ──────────────────────────────────
    ax = fig.add_subplot(gs[2])
    box_data   = [d['all_meds'] for d in cv_curve]
    positions  = list(range(len(ns)))
    bp = ax.boxplot(box_data, positions=positions, widths=0.55,
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color=TEXT, lw=2),
                    whiskerprops=dict(color=MUTED),
                    capprops=dict(color=MUTED),
                    flierprops=dict(marker='o', color=MUTED, ms=2))
    for patch in bp['boxes']:
        patch.set_facecolor(C_G); patch.set_alpha(0.6)

    ax.axhline(hca_med, color=C_H, ls='-', lw=2, label=f'HCA ({hca_med:.4f})')
    ax.axhline(gtex_full_med, color=ACCENT1, ls=':', lw=2,
               label=f'GTEx full ({gtex_full_med:.4f})')
    ax.set_xticks(positions)
    ax.set_xticklabels([str(n) for n in ns], fontsize=9)
    ax.set_xlabel('Number of donors')
    ax.set_ylabel('Median CV')
    ax.set_title('Spread of Median CV at Each Sample Size\n(box = IQR, whiskers = 5–95th pct)',
                 fontsize=13, fontweight='bold', pad=10)
    ax.legend(fontsize=9, facecolor=CARD, edgecolor=GRID, labelcolor=TEXT)
    ax.grid(axis='y', alpha=0.3)

    plt.savefig(out, dpi=180, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"Saved: {out}")


def fig_depth(bins, bin_labels, lib_sizes, spear_r, cvs_expr, out):
    """
    5-panel figure:
      1. Library size distribution with bin boundaries
      2. CV distribution per depth bin (overlay)
      3. Median CV vs median library size (scatter + trend)
      4. Distribution of per-gene Spearman r (lib size vs expression)
      5. Std of per-gene expression vs library size (sample-level scatter)
    """
    fig = plt.figure(figsize=(26, 14))
    gs  = GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.3,
                   left=0.06, right=0.97, top=0.91, bottom=0.07)
    fig.suptitle(
        'Sequencing Depth Effect on CV — GTEx Whole Blood (803 samples)',
        fontsize=16, fontweight='bold', color=TEXT, y=0.97)

    # ── panel 1: library size histogram with bin boundaries ──────────
    ax = fig.add_subplot(gs[0, 0])
    ax.hist(lib_sizes / 1e6, bins=60, color=ACCENT1, alpha=0.8, edgecolor='none')
    # mark bin edges as vertical lines
    seen_edges = set()
    for b in bins:
        lo = b['lib_vals'].min() / 1e6
        hi = b['lib_vals'].max() / 1e6
        for edge in [lo, hi]:
            if edge not in seen_edges:
                ax.axvline(edge, color=MUTED, ls=':', lw=1.2, alpha=0.7)
                seen_edges.add(edge)
    for i, b in enumerate(bins):
        mid = b['lib_med'] / 1e6
        ax.text(mid, ax.get_ylim()[1] * 0.85, f'Q{i+1}\nn={b["n"]}',
                ha='center', fontsize=8, color=BIN_COLORS[i])
    ax.set_xlabel('Total reads per sample (millions)')
    ax.set_ylabel('Number of samples')
    ax.set_title('Library Size Distribution\n(quintile bins)', fontsize=13, fontweight='bold', pad=10)
    ax.grid(axis='y', alpha=0.3)

    # ── panel 2: CV distribution per bin ─────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    for i, b in enumerate(bins):
        vc = b['valid_cvs']
        ax.hist(vc, bins=100, range=(0, 1.5), density=True,
                alpha=0.45, color=BIN_COLORS[i], edgecolor='none',
                label=f"{b['label']} (med={np.median(vc):.4f})")
        ax.axvline(np.median(vc), color=BIN_COLORS[i], ls='--', lw=1.5)
    ax.set_xlabel('CV'); ax.set_ylabel('Density')
    ax.set_title('CV Distribution per Depth Bin', fontsize=13, fontweight='bold', pad=10)
    ax.legend(fontsize=8.5, facecolor=CARD, edgecolor=GRID, labelcolor=TEXT)
    ax.grid(axis='y', alpha=0.3)

    # ── panel 3: median CV vs median library size ────────────────────
    ax = fig.add_subplot(gs[0, 2])
    lib_meds = np.array([b['lib_med'] / 1e6 for b in bins])
    cv_meds  = np.array([np.median(b['valid_cvs']) for b in bins])
    cv_q25   = np.array([np.percentile(b['valid_cvs'], 25) for b in bins])
    cv_q75   = np.array([np.percentile(b['valid_cvs'], 75) for b in bins])

    ax.errorbar(lib_meds, cv_meds, yerr=[cv_meds - cv_q25, cv_q75 - cv_meds],
                fmt='o', color=ACCENT1, ms=10, lw=2, capsize=6,
                ecolor=MUTED, label='Median CV (IQR)')
    for i, (x, y, b) in enumerate(zip(lib_meds, cv_meds, bins)):
        ax.annotate(b['label'], (x, y), xytext=(5, 5),
                    textcoords='offset points', fontsize=8, color=BIN_COLORS[i])

    # Pearson r between lib size and median CV
    r, p = pearsonr(lib_meds, cv_meds)
    ax.text(0.05, 0.95, f'Pearson r = {r:.4f}\np = {p:.3f}',
            transform=ax.transAxes, fontsize=11, color=TEXT, va='top',
            bbox=dict(facecolor=CARD, edgecolor=GRID, alpha=0.85, pad=4))

    ax.set_xlabel('Median library size per bin (millions reads)')
    ax.set_ylabel('Median CV across all expressed genes')
    ax.set_title('Depth vs Median CV\n(does CV drop with more reads?)',
                 fontsize=13, fontweight='bold', pad=10)
    ax.legend(fontsize=9, facecolor=CARD, edgecolor=GRID, labelcolor=TEXT)
    ax.grid(alpha=0.3)

    # ── panel 4: per-gene Spearman r distribution ────────────────────
    ax = fig.add_subplot(gs[1, 0])
    ax.hist(spear_r, bins=100, range=(-0.7, 0.7), color=ACCENT4,
            alpha=0.8, edgecolor='none', density=True)
    ax.axvline(0,              color=MUTED, ls=':', lw=2)
    ax.axvline(np.mean(spear_r), color=C_G, ls='--', lw=2,
               label=f'Mean r = {np.mean(spear_r):.4f}')
    ax.axvline(np.median(spear_r), color=ACCENT1, ls='-', lw=2,
               label=f'Median r = {np.median(spear_r):.4f}')
    frac30 = (np.abs(spear_r) > 0.3).mean() * 100
    frac50 = (np.abs(spear_r) > 0.5).mean() * 100
    ax.text(0.98, 0.95,
            f'|r| > 0.3: {frac30:.1f}%\n|r| > 0.5: {frac50:.1f}%',
            transform=ax.transAxes, ha='right', va='top', fontsize=10,
            color=TEXT, bbox=dict(facecolor=CARD, edgecolor=GRID, alpha=0.85, pad=4))
    ax.set_xlabel('Spearman r  (library size vs log₂(CPM+1))')
    ax.set_ylabel('Density')
    ax.set_title('Per-Gene Depth Correlation\n(after CPM normalization)',
                 fontsize=13, fontweight='bold', pad=10)
    ax.legend(fontsize=10, facecolor=CARD, edgecolor=GRID, labelcolor=TEXT)
    ax.grid(axis='y', alpha=0.3)

    # ── panel 5: CV vs Spearman r scatter ────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    # We need to pair cvs_full with spear_r (both indexed on expressed genes)
    # spear_r was computed on expressed genes only
    ax.scatter(spear_r, np.full_like(spear_r, 0.5),  # placeholder; overwritten below
               s=0, alpha=0)  # invisible, just to set up axis
    # spear_r is over expressed genes; bins[0].cvs is full gene set
    # pass spear_r and matching CVs
    ax.set_visible(False)  # will be repurposed below

    # ── panel 5 (proper): std of per-sample depth vs median CV per sample ──
    ax5 = fig.add_subplot(gs[1, 1])
    # For each sample compute its "deviation from median depth"
    med_lib = np.median(lib_sizes)
    rel_depth = lib_sizes / med_lib  # 1.0 = median depth
    # For each sample compute its per-gene CV (expensive → use std of log-CPM)
    # Approximate: log-CPM std across genes as measure of sample-level variability
    expr_filt_tmp, _ = filter_genes(
        np.vstack([bins[0]['lib_vals']]),  # dummy; use the bins' valid_cvs aggregated
        np.array(['dummy']))

    # simpler: show per-bin boxplot of CV with depth on x-axis
    ax5.set_facecolor(CARD)
    for spine in ax5.spines.values(): spine.set_edgecolor(GRID)
    positions = np.arange(len(bins))
    bp2 = ax5.boxplot([b['valid_cvs'] for b in bins],
                      positions=positions, widths=0.55, showfliers=False,
                      patch_artist=True,
                      medianprops=dict(color=TEXT, lw=2),
                      whiskerprops=dict(color=MUTED),
                      capprops=dict(color=MUTED))
    for i, patch in enumerate(bp2['boxes']):
        patch.set_facecolor(BIN_COLORS[i]); patch.set_alpha(0.7)
    ax5.set_xticks(positions)
    ax5.set_xticklabels([b['label'].replace(' ', '\n') for b in bins], fontsize=8)
    ax5.set_xlabel('Depth bin')
    ax5.set_ylabel('CV distribution')
    ax5.set_title('CV Boxplot per Depth Bin\n(whiskers = 5–95th pct)',
                  fontsize=13, fontweight='bold', pad=10)
    ax5.tick_params(colors=MUTED)
    ax5.grid(axis='y', alpha=0.3)

    # ── panel 6: Spearman r vs CV for expressed genes ─────────────────
    ax = fig.add_subplot(gs[1, 2])
    ax.scatter(spear_r, cvs_expr, s=1.5, alpha=0.18, c=ACCENT4, rasterized=True)
    r2, _ = pearsonr(spear_r, cvs_expr)
    ax.text(0.05, 0.95, f'Pearson r = {r2:.4f}',
            transform=ax.transAxes, fontsize=11, color=TEXT, va='top',
            bbox=dict(facecolor=CARD, edgecolor=GRID, alpha=0.85, pad=4))
    ax.axvline(0, color=MUTED, ls=':', lw=1.5)
    ax.set_xlabel('Spearman r  (depth vs expression per gene)')
    ax.set_ylabel('Gene CV (full dataset)')
    ax.set_title('High-CV Genes: Depth-Correlated?\n(r ≠ 0 → residual depth effect)',
                 fontsize=13, fontweight='bold', pad=10)
    ax.grid(alpha=0.3)

    plt.savefig(out, dpi=180, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"Saved: {out}")


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':

    # ── Load data ──────────────────────────────────────────────────────
    expr_raw, gene_names = load_gtex_raw(GTEX_PATH)
    hca_cvs, hca_med, hca_mean = load_hca(PSEUDO)

    # GTEx full-dataset stats
    expr_filt_full, names_full = filter_genes(expr_raw, gene_names)
    e_log_full, _ = cpm_log(expr_filt_full)
    cvs_full, _, _, _ = cv_of(e_log_full)
    gtex_full_med  = np.nanmedian(cvs_full)
    gtex_full_mean = np.nanmean(cvs_full)
    print(f"\nGTEx full (n=803): median CV = {gtex_full_med:.4f}")
    print(f"HCA pseudobulk (n=8): median CV = {hca_med:.4f}")

    # ── A. Bootstrap subsampling ───────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"BOOTSTRAP SUBSAMPLING (n={N_BOOT_DONORS}, {N_ITER} iterations)")
    print(f"{'='*65}")

    rng = np.random.default_rng(42)
    med_cvs_8, mean_cvs_8, q25_8, q75_8 = bootstrap_cv(
        expr_raw, gene_names, N_BOOT_DONORS, N_ITER, rng)

    print(f"\nWith n=8 GTEx donors:")
    print(f"  Median CV — mean={np.mean(med_cvs_8):.4f}  "
          f"std={np.std(med_cvs_8):.4f}  "
          f"range=[{med_cvs_8.min():.4f}, {med_cvs_8.max():.4f}]")
    print(f"  HCA pseudobulk median CV: {hca_med:.4f}")
    print(f"  Residual gap (GTEx n=8 median – HCA): "
          f"{np.median(med_cvs_8) - hca_med:+.4f}")
    print(f"  GTEx full gap (GTEx n=803 median – HCA): "
          f"{gtex_full_med - hca_med:+.4f}")
    pct_explained = (1 - (np.median(med_cvs_8) - hca_med) /
                     (gtex_full_med - hca_med)) * 100
    print(f"  % of CV gap explained by sample-size effect: {pct_explained:.1f}%")

    # CV-vs-n curve
    print(f"\nComputing CV-vs-n curve ...")
    cv_curve = []
    for n in N_SIZES:
        n_it = N_ITER if n <= 8 else N_ITER_CURVE
        meds, _, _, _ = bootstrap_cv(expr_raw, gene_names, n, n_it, rng)
        cv_curve.append({
            'n':        n,
            'med_med':  np.median(meds),
            'med_q10':  np.percentile(meds, 10),
            'med_q90':  np.percentile(meds, 90),
            'all_meds': meds,
        })
        print(f"  n={n:>4}: median CV = {np.median(meds):.4f}  "
              f"[{np.percentile(meds,10):.4f} – {np.percentile(meds,90):.4f}]")

    # ── B. Library-size binning ────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"LIBRARY-SIZE DEPTH BINNING ({N_BINS} quintile bins)")
    print(f"{'='*65}")
    bins, bin_labels, lib_sizes, expr_filt, names_filt, cvs_full2, spear_r, cvs_expr_aligned = \
        depth_bin_analysis(expr_raw, gene_names)

    # ── Figures ────────────────────────────────────────────────────────
    print("\nGenerating figures ...")
    fig_bootstrap(
        med_cvs_8, mean_cvs_8, q25_8, q75_8,
        hca_med, hca_mean, gtex_full_med, gtex_full_mean,
        cv_curve,
        f'{BASEDIR}/sample_size_bootstrap.png')

    fig_depth(bins, bin_labels, lib_sizes, spear_r, cvs_expr_aligned,
              f'{BASEDIR}/depth_binning_cv.png')

    # ── C. Housekeeping vs Global: mean comparison ────────────────────
    print(f"\n{'='*65}")
    print(f"HOUSEKEEPING vs GLOBAL GENES — MEAN EXPRESSION COMPARISON")
    print(f"{'='*65}")

    HOUSEKEEPING_GENES = {
        'ACTB', 'ACTG1', 'GAPDH', 'B2M', 'HPRT1', 'TBP', 'SDHA', 'YWHAZ',
        'PPIA', 'IPO8', 'HMBS', 'PGK1', 'LDHA', 'ENO1', 'TPI1', 'PKM',
        'MDH2', 'CS', 'ATP5F1B', 'PGAM1', 'EEF1A1', 'EEF1B2', 'EEF2',
        'ARF1', 'GNB1',
        'RPL3','RPL4','RPL5','RPL6','RPL7','RPL7A','RPL8','RPL9','RPL10',
        'RPL10A','RPL11','RPL12','RPL13','RPL13A','RPL14','RPL15','RPL17',
        'RPL18','RPL18A','RPL19','RPL21','RPL22','RPL23','RPL23A','RPL24',
        'RPL26','RPL27','RPL27A','RPL28','RPL29','RPL30','RPL31','RPL32',
        'RPL34','RPL35','RPL35A','RPL36','RPL37','RPL37A','RPL38','RPL39',
        'RPL41',
        'RPS2','RPS3','RPS3A','RPS4X','RPS5','RPS6','RPS7','RPS8','RPS9',
        'RPS10','RPS11','RPS12','RPS13','RPS14','RPS15','RPS15A','RPS16',
        'RPS17','RPS18','RPS19','RPS20','RPS21','RPS23','RPS24','RPS25',
        'RPS26','RPS27','RPS27A','RPS28','RPS29',
    }

    # GTEx: means for housekeeping vs all expressed genes
    e_log_g, e_cpm_g = cpm_log(expr_filt_full)
    means_g = e_log_g.mean(axis=1)
    stds_g  = e_log_g.std(axis=1)
    cvs_g, _, _, expressed_g = cv_of(e_log_g)

    hk_upper = {x.upper() for x in HOUSEKEEPING_GENES}
    names_upper = np.array([n.upper() for n in names_full])
    is_hk_g = np.array([n in hk_upper for n in names_upper])
    is_expr_g = expressed_g

    hk_expr_g    = is_hk_g & is_expr_g
    other_expr_g = (~is_hk_g) & is_expr_g

    print(f"\nGTEx (n=803 donors):")
    print(f"  {'Category':<28} {'n':>6} {'Mean log2':>10} {'Median log2':>12} "
          f"{'Std log2':>10} {'Median CV':>10}")
    print(f"  {'-'*73}")
    for label, mask in [('All expressed genes',    is_expr_g),
                        ('Housekeeping genes',       hk_expr_g),
                        ('Non-housekeeping expressed', other_expr_g)]:
        n  = mask.sum()
        m  = means_g[mask].mean();  med_m = np.median(means_g[mask])
        s  = stds_g[mask].mean()
        cv = np.nanmedian(cvs_g[mask])
        print(f"  {label:<28} {n:>6,} {m:>10.3f} {med_m:>12.3f} {s:>10.3f} {cv:>10.4f}")

    # HCA: same breakdown
    d = np.load(PSEUDO, allow_pickle=True)
    expr_raw_h   = d['expr'].astype(np.float64)
    gene_names_h = d['gene_names'].astype(str)
    expr_filt_h, names_filt_h = filter_genes(expr_raw_h, gene_names_h)
    e_log_h, _ = cpm_log(expr_filt_h)
    means_h = e_log_h.mean(axis=1)
    stds_h  = e_log_h.std(axis=1)
    cvs_h, _, _, expressed_h = cv_of(e_log_h)

    names_h_upper = np.array([n.upper() for n in names_filt_h])
    is_hk_h = np.array([n in hk_upper for n in names_h_upper])
    hk_expr_h    = is_hk_h & expressed_h
    other_expr_h = (~is_hk_h) & expressed_h

    print(f"\nHCA Pseudobulk (n=8 donors):")
    print(f"  {'Category':<28} {'n':>6} {'Mean log2':>10} {'Median log2':>12} "
          f"{'Std log2':>10} {'Median CV':>10}")
    print(f"  {'-'*73}")
    for label, mask in [('All expressed genes',    expressed_h),
                        ('Housekeeping genes',       hk_expr_h),
                        ('Non-housekeeping expressed', other_expr_h)]:
        n  = mask.sum()
        m  = means_h[mask].mean();  med_m = np.median(means_h[mask])
        s  = stds_h[mask].mean()
        cv = np.nanmedian(cvs_h[mask])
        print(f"  {label:<28} {n:>6,} {m:>10.3f} {med_m:>12.3f} {s:>10.3f} {cv:>10.4f}")

    # Cross-dataset comparison: housekeeping vs global ratio
    print(f"\n  Enrichment of housekeeping genes relative to global (fold-difference in mean):")
    gtex_hk_med  = np.median(means_g[hk_expr_g])
    gtex_all_med = np.median(means_g[is_expr_g])
    hca_hk_med   = np.median(means_h[hk_expr_h])
    hca_all_med  = np.median(means_h[expressed_h])
    print(f"  GTEx:  HK median log2 = {gtex_hk_med:.3f}  All median = {gtex_all_med:.3f}  "
          f"Δ = {gtex_hk_med - gtex_all_med:+.3f} (HK expressed {gtex_hk_med - gtex_all_med:.1f} log2 units higher)")
    print(f"  HCA:   HK median log2 = {hca_hk_med:.3f}   All median = {hca_all_med:.3f}  "
          f"Δ = {hca_hk_med - hca_all_med:+.3f} (HK expressed {hca_hk_med - hca_all_med:.1f} log2 units higher)")

    print("\nAll done.")
