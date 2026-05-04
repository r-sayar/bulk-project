"""
GTEx Tissue Variance Analysis Pipeline
=======================================
Structured analysis of gene expression variance across GTEx tissues.
Runs identical pipeline on each tissue for direct comparison.

The pipeline:
  1. Loads raw read counts from GCT files (GTEx bulk RNA-seq format)
  2. Filters out lowly-expressed genes (noise)
  3. Normalizes to CPM (counts per million) to correct for sequencing depth
  4. Log-transforms to stabilize variance and compress dynamic range
  5. Computes per-gene variance statistics (mean, std, CV)
  6. Clusters samples using PCA + KMeans to find subgroups
  7. Measures whether within-cluster CV is lower than whole-dataset CV
  8. Generates per-tissue and cross-tissue comparison figures
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════
# CONFIG
# All tunable parameters in one place. Add new tissues by adding a
# single entry to TISSUES — the rest of the pipeline runs automatically.
# ══════════════════════════════════════════════════════════════════════

TISSUES = {
    'Whole Blood': '/Users/rls/Downloads/gene_reads_v11_whole_blood.gct.gz',
    'Liver': '/Users/rls/Downloads/gene_reads_v11_liver.gct.gz',
}

N_CLUSTERS = 10        # Number of KMeans clusters for sample grouping
N_HVG = 5000           # Number of highly variable genes to use for clustering
N_PCA = 50             # PCA components to retain before clustering
CPM_THRESHOLD = 1      # Minimum CPM to consider a gene "detected" in a sample
MIN_SAMPLE_FRAC = 0.1  # Gene must be detected in at least this fraction of samples to pass filter

# ══════════════════════════════════════════════════════════════════════
# STYLE
# Dark theme colors for all matplotlib figures. Consistent across all
# panels so the output looks cohesive.
# ══════════════════════════════════════════════════════════════════════

BG = '#0e1117'; CARD = '#1a1d23'; TEXT = '#e6edf3'; MUTED = '#7d8590'; GRID = '#21262d'
PALETTE = ['#58a6ff','#f78166','#3fb950','#d2a8ff','#f0883e',
           '#79c0ff','#ffa657','#56d364','#bc8cff','#e3b341']
TISSUE_COLORS = {'Whole Blood': '#f78166', 'Liver': '#58a6ff'}

plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': CARD, 'axes.edgecolor': GRID,
    'axes.labelcolor': TEXT, 'text.color': TEXT, 'xtick.color': MUTED,
    'ytick.color': MUTED, 'grid.color': GRID, 'grid.alpha': 0.5,
    'font.family': 'sans-serif', 'font.size': 11,
})

# ══════════════════════════════════════════════════════════════════════
# 1. LOAD & OVERVIEW
# GCT is the standard GTEx format: tab-separated, with 2 header lines
# (version + dimensions), then a matrix where columns 0-1 are gene
# identifiers and remaining columns are samples.
# ══════════════════════════════════════════════════════════════════════

def load_gct(path):
    """Parse a GCT v1.2 file into expression matrix + metadata arrays."""
    df = pd.read_csv(path, sep='\t', skiprows=2, compression='gzip')
    expr = df.iloc[:, 2:].values.astype(np.float64)  # Raw integer read counts
    gene_names = df['Description'].values              # Gene symbols (e.g. "HBB")
    gene_ids = df['Name'].values                       # Ensembl IDs (e.g. "ENSG00000244734.4")
    sample_ids = df.columns[2:].values                 # GTEx sample IDs
    return expr, gene_names, gene_ids, sample_ids


def print_overview(name, expr, gene_names):
    """Print basic sanity-check stats on the raw count matrix."""
    n_genes, n_samples = expr.shape
    print(f"\n{'='*70}")
    print(f"  {name.upper()} — FILE OVERVIEW")
    print(f"{'='*70}")
    print(f"  Genes:        {n_genes:,}")
    print(f"  Samples:      {n_samples}")
    print(f"  Data points:  {n_genes * n_samples:,}")
    print(f"  % zeros:      {(expr == 0).sum() / expr.size * 100:.1f}%")
    print(f"  Min:          {expr.min():,.0f}")
    print(f"  Max:          {expr.max():,.0f}")
    print(f"  Global mean:  {expr.mean():,.2f}")
    print(f"  Global std:   {expr.std():,.2f}")
    print(f"  Median:       {np.median(expr):,.1f}")

# ══════════════════════════════════════════════════════════════════════
# 2. PREPROCESSING
# Standard bulk RNA-seq normalization in 3 steps:
#   a) Filter — remove genes that are barely detected (noise).
#   b) CPM — normalize for sequencing depth differences between samples.
#   c) Log2 — compress the huge dynamic range (0 to millions) and
#      stabilize variance so that high-expression genes don't dominate.
# ══════════════════════════════════════════════════════════════════════

def preprocess(expr, gene_names, gene_ids):
    """Filter low-expression genes, CPM-normalize, and log-transform."""
    n_genes, n_samples = expr.shape

    # Step a: Filter — compute CPM per gene per sample, keep genes where
    # CPM > 1 in at least 10% of samples. This removes ~58k silent genes
    # that would add noise without biological signal.
    lib_sizes = expr.sum(axis=0)              # Total reads per sample
    cpm = expr / lib_sizes * 1e6              # Counts per million
    min_samples = int(MIN_SAMPLE_FRAC * n_samples)
    keep = (cpm > CPM_THRESHOLD).sum(axis=1) >= min_samples
    expr_filt = expr[keep]
    names_filt = gene_names[keep]
    ids_filt = gene_ids[keep]

    # Step b: CPM normalize — divides each gene's count by the sample's
    # total counts, then multiplies by 1M. Corrects for the fact that
    # some samples were sequenced to 10M reads and others to 170M.
    expr_cpm = expr_filt / expr_filt.sum(axis=0) * 1e6

    # Step c: Log2(CPM + 1) — log-transform to stabilize variance.
    # Without this, a gene at 1M CPM would dominate all statistics.
    # The +1 prevents log(0) = -infinity for zero-count entries.
    expr_log = np.log2(expr_cpm + 1)

    print(f"  Filtered: {keep.sum():,} / {n_genes:,} genes kept")
    return expr_filt, expr_cpm, expr_log, names_filt, ids_filt

# ══════════════════════════════════════════════════════════════════════
# 3. GLOBAL VARIANCE STATISTICS
# Computes CV (coefficient of variation = std/mean) for each gene.
# CV measures *relative* variability — how much a gene varies as a
# proportion of its average expression. This allows fair comparison
# between a gene at 1M CPM and one at 10 CPM.
#
# Key outputs:
#   - CV threshold breakdown (how many genes fall in each CV bin)
#   - Top most variable genes (highest CV — sporadic/tissue-contamination)
#   - Top most stable genes (lowest CV — housekeeping genes)
#   - CPM percentiles (context for what "high" and "low" expression means)
# ══════════════════════════════════════════════════════════════════════

def compute_variance_stats(expr_log, expr_cpm, names_filt):
    """Compute per-gene variance stats and print summary tables."""
    n_genes, n_samples = expr_log.shape
    means = expr_log.mean(axis=1)       # Mean log2(CPM+1) per gene across all samples
    stds = expr_log.std(axis=1)         # Std dev per gene across all samples

    # Only compute CV for genes with meaningful expression.
    # A gene with mean ~0 would have CV = inf (dividing by ~0).
    # Threshold of 0.5 in log2(CPM+1) space corresponds to ~0.4 CPM.
    expressed = means > 0.5
    cvs = np.full(n_genes, np.nan)
    cvs[expressed] = stds[expressed] / means[expressed]

    cpm_means = expr_cpm.mean(axis=1)   # Mean CPM per gene (for human-readable context)

    stats = {
        'means': means, 'stds': stds, 'cvs': cvs,
        'expressed': expressed, 'cpm_means': cpm_means,
    }

    # Per-gene variance summary
    valid_cvs = cvs[~np.isnan(cvs)]
    print(f"\n  Per-gene CV (expressed genes, n={expressed.sum():,}):")
    print(f"    Mean CV:   {valid_cvs.mean():.4f}")
    print(f"    Median CV: {np.median(valid_cvs):.4f}")

    # Expression level distribution
    print(f"\n  Expression level distribution:")
    bins = [
        ('Zero (mean=0)',       means == 0),
        ('Very low (0–1)',      (means > 0) & (means <= 1)),
        ('Low (1–100 CPM)',     (means > 1) & (cpm_means <= 100)),
        ('Medium (100–10k CPM)',(cpm_means > 100) & (cpm_means <= 10000)),
        ('High (>10k CPM)',     cpm_means > 10000),
    ]
    for label, mask in bins:
        n = mask.sum()
        print(f"    {label:<25} {n:>6,} ({n/n_genes*100:.1f}%)")

    # CV threshold table
    thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.50]
    print(f"\n  CV threshold breakdown:")
    print(f"    {'CV range':<14} {'# genes':>8} {'Mean log2':>10} {'Std log2':>9} {'Mean CPM':>10}")
    print(f"    {'-'*55}")
    prev = 0
    for t in thresholds:
        mask = expressed & (cvs >= prev) & (cvs < t)
        n = mask.sum()
        if n > 0:
            m = means[mask].mean()
            s = stds[mask].mean()
            raw = cpm_means[mask].mean()
            print(f"    {prev:.2f}–{t:.2f}     {n:>8,} {m:>10.3f} {s:>9.3f} {raw:>10.1f}")
        prev = t
    mask = expressed & (cvs >= 0.50)
    n = mask.sum()
    if n > 0:
        print(f"    ≥0.50         {n:>8,} {means[mask].mean():>10.3f} {stds[mask].mean():>9.3f} {cpm_means[mask].mean():>10.1f}")

    # Top 20 most variable
    print(f"\n  Top 20 most variable genes (by CV, expressed):")
    print(f"    {'Gene':<16} {'CV':>7} {'Mean log2':>10} {'Std log2':>9} {'Mean CPM':>10} {'%zeros':>7}")
    print(f"    {'-'*62}")
    top_idx = np.argsort(np.nan_to_num(cvs, nan=-1))[::-1][:20]
    for idx in top_idx:
        pz = (expr_log[idx] == 0).sum() / n_samples * 100
        print(f"    {names_filt[idx]:<16} {cvs[idx]:>7.3f} {means[idx]:>10.3f} {stds[idx]:>9.3f} {cpm_means[idx]:>10.2f} {pz:>6.1f}%")

    # Top 20 most stable (mean CPM > 100)
    print(f"\n  Top 20 most stable genes (by CV, mean CPM > 100):")
    print(f"    {'Gene':<16} {'CV':>7} {'Mean log2':>10} {'Mean CPM':>10}")
    print(f"    {'-'*48}")
    stable_cvs = cvs.copy()
    stable_cvs[cpm_means <= 100] = np.inf
    stable_cvs[np.isnan(stable_cvs)] = np.inf
    stable_idx = np.argsort(stable_cvs)[:20]
    for idx in stable_idx:
        print(f"    {names_filt[idx]:<16} {cvs[idx]:>7.4f} {means[idx]:>10.3f} {cpm_means[idx]:>10.1f}")

    # CPM percentiles
    nonzero_cpm = expr_cpm[expr_cpm > 0]
    print(f"\n  CPM percentiles (non-zero values):")
    for p in [25, 50, 75, 90, 95, 99, 99.9]:
        print(f"    {p:>5}th: {np.percentile(nonzero_cpm, p):>10.2f} CPM")

    # Library size stats
    lib_sums = expr_cpm.sum(axis=0)  # should be ~1M each after CPM
    raw_sums = expr_log.sum(axis=0)  # not meaningful for lib size, use raw
    print(f"\n  Samples: {n_samples}")

    return stats


# ══════════════════════════════════════════════════════════════════════
# 4. CLUSTERING & PER-CLUSTER CV
# The idea: if samples within a cluster are biologically similar, then
# gene expression should be more consistent within each cluster than
# across the whole dataset. We measure this by comparing within-cluster
# CV to whole-dataset CV. A large "CV reduction" means the clustering
# captured real biological structure (not just noise).
#
# Steps: HVG selection → Z-score → PCA → KMeans → per-cluster CV
# ══════════════════════════════════════════════════════════════════════

def cluster_and_cv(expr_log, names_filt, n_samples):
    """Cluster samples and measure whether within-cluster CV is lower."""
    n_genes = expr_log.shape[0]

    # Select top 5000 highly variable genes (HVGs) for clustering.
    # Clustering on all ~16k genes would be dominated by noise from
    # low-expression genes. HVGs carry the most biological signal.
    gene_var = np.var(expr_log, axis=1)
    n_hvg = min(N_HVG, n_genes)
    hvg_idx = np.argsort(gene_var)[::-1][:n_hvg]
    expr_hvg = expr_log[hvg_idx]

    # Z-score each gene (mean=0, std=1 across samples).
    # Without this, highly expressed genes dominate PCA/KMeans purely
    # because their absolute values are larger, not because they carry
    # more information about sample differences.
    scaler = StandardScaler()
    expr_scaled = scaler.fit_transform(expr_hvg.T).T  # Scale across samples per gene
    X = expr_scaled.T  # Transpose to (samples x genes) for PCA

    # PCA: reduce from 5000 dimensions to 50.
    # Removes noise dimensions and makes KMeans tractable.
    # Typically captures ~85-90% of total variance.
    n_pca = min(N_PCA, min(X.shape) - 1)
    pca = PCA(n_components=n_pca, random_state=42)
    X_pca = pca.fit_transform(X)
    print(f"  PCA: {n_pca} components, {pca.explained_variance_ratio_.sum():.1%} variance")

    # KMeans: partition samples into k groups.
    # n_init=20 runs the algorithm 20 times with different random seeds
    # and picks the best result (most stable clustering).
    # Ensure at least 5 samples per cluster for meaningful CV computation.
    n_clust = min(N_CLUSTERS, n_samples // 5)
    km = KMeans(n_clusters=n_clust, n_init=20, random_state=42)
    labels = km.fit_predict(X_pca)

    # Compute whole-dataset CV as the baseline to compare against
    means_all = expr_log.mean(axis=1)
    stds_all = expr_log.std(axis=1)
    expr_mask = means_all > 0.5
    cvs_all = stds_all[expr_mask] / means_all[expr_mask]

    print(f"\n  {'Cluster':>8} {'N':>5} {'Mean CV':>9} {'Median CV':>10}")
    print(f"  {'-'*38}")

    # For each cluster, compute CV using only that cluster's samples.
    # If the clustering is meaningful, these CVs should be lower than
    # the whole-dataset CV because within-group variance < total variance.
    cluster_data = []
    for c in range(n_clust):
        mask = labels == c
        cexpr = expr_log[:, mask]         # Subset to this cluster's samples
        cm = cexpr.mean(axis=1)           # Per-gene mean within cluster
        cs = cexpr.std(axis=1)            # Per-gene std within cluster
        exp = cm > 0.5
        ccvs = cs[exp] / cm[exp]          # Per-gene CV within cluster
        cluster_data.append({
            'n': mask.sum(), 'mean_cv': ccvs.mean(),
            'median_cv': np.median(ccvs), 'cvs': ccvs
        })
        print(f"  {c:>8} {mask.sum():>5} {ccvs.mean():>9.4f} {np.median(ccvs):>10.4f}")

    print(f"  {'WHOLE':>8} {n_samples:>5} {cvs_all.mean():>9.4f} {np.median(cvs_all):>10.4f}")

    # CV reduction = how much lower within-cluster CV is vs whole-dataset.
    # 28% reduction means clustering explained ~28% of the total variance.
    avg_within = np.mean([d['mean_cv'] for d in cluster_data])
    reduction = (1 - avg_within / cvs_all.mean()) * 100
    print(f"\n  CV reduction from clustering: {reduction:.1f}%")

    return labels, X_pca, pca, cluster_data, cvs_all, n_clust


# ══════════════════════════════════════════════════════════════════════
# 5. LIBRARY SIZE STATS
# "Library size" = total reads per sample. Varies because some samples
# are sequenced deeper than others. Large variation here means CPM
# normalization was essential. The CV of library sizes tells you how
# uneven the sequencing was.
# ══════════════════════════════════════════════════════════════════════

def library_stats(expr_raw):
    """Report sequencing depth variation across samples."""
    sums = expr_raw.sum(axis=0)  # Total raw reads per sample
    print(f"\n  Library size (raw reads per sample):")
    print(f"    Mean:   {sums.mean():>14,.0f}")
    print(f"    Std:    {sums.std():>14,.0f}")
    print(f"    Min:    {sums.min():>14,.0f}")
    print(f"    Max:    {sums.max():>14,.0f}")
    print(f"    CV:     {sums.std()/sums.mean():>14.4f}")
    return sums


# ══════════════════════════════════════════════════════════════════════
# 6. RUN PIPELINE
# Loops over each tissue and runs the identical pipeline. This ensures
# blood and liver are processed with exactly the same parameters for
# fair comparison. Results are stored in a dict keyed by tissue name.
# ══════════════════════════════════════════════════════════════════════

results = {}

for tissue, path in TISSUES.items():
    print(f"\n{'#'*70}")
    print(f"  PROCESSING: {tissue.upper()}")
    print(f"{'#'*70}")

    expr_raw, gene_names, gene_ids, sample_ids = load_gct(path)
    print_overview(tissue, expr_raw, gene_names)
    lib_sums = library_stats(expr_raw)

    expr_filt, expr_cpm, expr_log, names_filt, ids_filt = preprocess(expr_raw, gene_names, gene_ids)
    stats = compute_variance_stats(expr_log, expr_cpm, names_filt)
    labels, X_pca, pca, cluster_data, cvs_all, n_clust = cluster_and_cv(
        expr_log, names_filt, len(sample_ids))

    results[tissue] = {
        'expr_raw': expr_raw, 'expr_log': expr_log, 'expr_cpm': expr_cpm,
        'names_filt': names_filt, 'stats': stats,
        'labels': labels, 'X_pca': X_pca, 'pca': pca,
        'cluster_data': cluster_data, 'cvs_all': cvs_all,
        'n_clust': n_clust, 'lib_sums': lib_sums,
        'sample_ids': sample_ids, 'n_samples': len(sample_ids),
    }


# ══════════════════════════════════════════════════════════════════════
# 7. COMPARISON SUMMARY
# Side-by-side table of key metrics across tissues.
# ══════════════════════════════════════════════════════════════════════

print(f"\n{'#'*70}")
print(f"  TISSUE COMPARISON")
print(f"{'#'*70}")

print(f"\n  {'Metric':<35} {'Whole Blood':>14} {'Liver':>14}")
print(f"  {'-'*65}")

for metric, fn in [
    ('Samples', lambda r: f"{r['n_samples']}"),
    ('Genes after filtering', lambda r: f"{r['expr_log'].shape[0]:,}"),
    ('% zeros (raw)', lambda r: f"{(r['expr_raw']==0).sum()/r['expr_raw'].size*100:.1f}%"),
    ('Mean library size', lambda r: f"{r['lib_sums'].mean()/1e6:.1f}M"),
    ('Whole-dataset mean CV', lambda r: f"{r['cvs_all'].mean():.4f}"),
    ('Whole-dataset median CV', lambda r: f"{np.median(r['cvs_all']):.4f}"),
    ('Avg within-cluster CV', lambda r: f"{np.mean([d['mean_cv'] for d in r['cluster_data']]):.4f}"),
    ('CV reduction from clustering', lambda r: f"{(1-np.mean([d['mean_cv'] for d in r['cluster_data']])/r['cvs_all'].mean())*100:.1f}%"),
]:
    vals = [fn(results[t]) for t in TISSUES]
    print(f"  {metric:<35} {vals[0]:>14} {vals[1]:>14}")


# ══════════════════════════════════════════════════════════════════════
# 8. VISUALIZATIONS
# Two types of figures:
#   a) Per-tissue (8 panels): CPM distribution, mean-variance, CV
#      distribution, library sizes, PCA clusters, per-cluster CV,
#      sparsity, and top expressed genes.
#   b) Cross-tissue comparison (6 panels): overlaid distributions,
#      cluster effectiveness, and gene-level CV correlation scatter.
# ══════════════════════════════════════════════════════════════════════

def make_tissue_figure(tissue, r, outpath):
    """Generate 8-panel variance analysis figure for a single tissue."""
    expr_log = r['expr_log']
    expr_cpm = r['expr_cpm']
    stats = r['stats']
    labels = r['labels']
    X_pca = r['X_pca']
    pca = r['pca']
    n_clust = r['n_clust']
    names = r['names_filt']
    n_samples = r['n_samples']
    lib_sums = r['lib_sums']
    means = stats['means']
    stds = stats['stds']
    cvs = stats['cvs']
    expressed = stats['expressed']
    cpm_means = stats['cpm_means']
    color = TISSUE_COLORS[tissue]

    fig = plt.figure(figsize=(22, 28))
    gs = GridSpec(4, 2, figure=fig, hspace=0.35, wspace=0.25,
                  left=0.07, right=0.95, top=0.94, bottom=0.03)
    fig.suptitle(f'GTEx v11 — {tissue} Variance Analysis',
                 fontsize=20, fontweight='bold', color=TEXT, y=0.975)
    fig.text(0.5, 0.96, f'{expr_log.shape[0]:,} genes  ·  {n_samples} samples  ·  preprocessed log2(CPM+1)',
             ha='center', fontsize=13, color=MUTED)

    # Panel 1: CPM distribution — shows the overall shape of expression levels.
    # Plotted on log10 scale because CPM spans ~0.01 to ~100,000.
    # Subsample to 5M points for histogram speed.
    ax = fig.add_subplot(gs[0, 0])
    nonzero_cpm = expr_cpm[expr_cpm > 0]
    rng = np.random.default_rng(42)
    if len(nonzero_cpm) > 5_000_000:
        nonzero_cpm = nonzero_cpm[rng.choice(len(nonzero_cpm), 5_000_000, replace=False)]
    ax.hist(np.log10(nonzero_cpm), bins=150, color=color, alpha=0.85, edgecolor='none')
    ax.axvline(np.log10(np.median(nonzero_cpm)), color='#3fb950', ls='--', lw=2,
               label=f'Median = {np.median(nonzero_cpm):.1f} CPM')
    ax.set_xlabel('log₁₀(CPM)'); ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Non-Zero CPM Values', fontsize=14, fontweight='bold', pad=10)
    ax.legend(fontsize=10, facecolor=CARD, edgecolor=GRID, labelcolor=TEXT)
    ax.grid(axis='y', alpha=0.3)

    # Panel 2: Mean-Variance relationship — in RNA-seq, variance always
    # exceeds the mean (overdispersion). If data were Poisson, points
    # would fall on the var=mean line. They sit above it, confirming
    # biological + technical overdispersion (negative binomial behavior).
    ax = fig.add_subplot(gs[0, 1])
    gm = means[expressed]; gv = stds[expressed]**2
    ax.scatter(gm, np.log10(gv), s=1.5, alpha=0.25, c=color, rasterized=True)
    x_r = np.linspace(gm.min(), gm.max(), 100)
    ax.set_xlabel('Mean expression (log2 CPM+1)'); ax.set_ylabel('log₁₀(variance)')
    ax.set_title('Mean–Variance Relationship', fontsize=14, fontweight='bold', pad=10)
    ax.grid(alpha=0.3)
    # Label top 5
    top5 = np.argsort(np.nan_to_num(cvs, nan=-1))[::-1][:5]
    for i in top5:
        ax.annotate(names[i], (means[i], np.log10(stds[i]**2)),
                     fontsize=8, color='#f78166', fontweight='bold',
                     xytext=(5, 5), textcoords='offset points')

    # Panel 3: CV distribution — how variable is each gene relative to
    # its mean? CV=1 reference line marks where std equals the mean.
    # Most genes cluster below 0.5; the long tail contains sporadic genes.
    ax = fig.add_subplot(gs[1, 0])
    valid_cvs = cvs[~np.isnan(cvs)]
    ax.hist(valid_cvs, bins=200, color='#d2a8ff', alpha=0.85, edgecolor='none', range=(0, 2.5))
    ax.axvline(np.median(valid_cvs), color='#f78166', ls='--', lw=2,
               label=f'Median CV = {np.median(valid_cvs):.3f}')
    ax.axvline(1.0, color='#3fb950', ls=':', lw=2, label='CV = 1')
    ax.set_xlabel('CV'); ax.set_ylabel('# genes')
    ax.set_title('Distribution of Per-Gene CV', fontsize=14, fontweight='bold', pad=10)
    ax.legend(fontsize=10, facecolor=CARD, edgecolor=GRID, labelcolor=TEXT)
    ax.grid(axis='y', alpha=0.3)

    # Panel 4: Library size distribution — were all samples sequenced
    # to similar depth? Outliers here could drive spurious variance.
    ax = fig.add_subplot(gs[1, 1])
    lib_m = lib_sums / 1e6
    ax.hist(lib_m, bins=40, color='#3fb950', alpha=0.85, edgecolor='none')
    ax.axvline(np.median(lib_m), color='#f78166', ls='--', lw=2,
               label=f'Median = {np.median(lib_m):.1f}M')
    ax.set_xlabel('Total reads (millions)'); ax.set_ylabel('# samples')
    ax.set_title('Library Size Distribution', fontsize=14, fontweight='bold', pad=10)
    ax.legend(fontsize=10, facecolor=CARD, edgecolor=GRID, labelcolor=TEXT)
    ax.grid(axis='y', alpha=0.3)

    # Panel 5: PCA scatter colored by KMeans cluster — shows whether
    # clusters correspond to visible structure in the top 2 PCs.
    ax = fig.add_subplot(gs[2, 0])
    for c in range(n_clust):
        mask = labels == c
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], s=15, alpha=0.7,
                   c=PALETTE[c % len(PALETTE)], label=f'C{c} ({mask.sum()})', edgecolors='none')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax.set_title(f'PCA — {n_clust} Clusters', fontsize=14, fontweight='bold', pad=10)
    ax.legend(fontsize=7, facecolor=CARD, edgecolor=GRID, labelcolor=TEXT, ncol=2, markerscale=2)
    ax.grid(alpha=0.3)

    # Panel 6: Per-cluster CV bars — compares mean and median CV within
    # each cluster vs the whole dataset ("ALL" bar). If clustering is
    # meaningful, all cluster bars should be shorter than "ALL".
    ax = fig.add_subplot(gs[2, 1])
    cluster_means = [d['mean_cv'] for d in r['cluster_data']]
    cluster_medians = [d['median_cv'] for d in r['cluster_data']]
    x = np.arange(n_clust + 1)
    bcolors = [PALETTE[i % len(PALETTE)] for i in range(n_clust)] + [TEXT]
    all_means = cluster_means + [r['cvs_all'].mean()]
    all_medians = cluster_medians + [np.median(r['cvs_all'])]
    xlabels = [f'C{i}' for i in range(n_clust)] + ['ALL']
    ax.bar(x - 0.17, all_means, 0.32, color=bcolors, alpha=0.85, edgecolor='none', label='Mean CV')
    ax.bar(x + 0.17, all_medians, 0.32, color=bcolors, alpha=0.45, edgecolor='none', label='Median CV')
    ax.set_xticks(x); ax.set_xticklabels(xlabels, fontsize=9)
    ax.set_ylabel('CV')
    ax.set_title('Per-Cluster CV', fontsize=14, fontweight='bold', pad=10)
    ax.legend(fontsize=10, facecolor=CARD, edgecolor=GRID, labelcolor=TEXT)
    ax.grid(axis='y', alpha=0.3)

    # Panel 7: Sparsity — what fraction of samples has zero counts for
    # each gene? Bimodal distribution = genes are either broadly expressed
    # (detected in most samples) or rarely expressed (zero in most).
    ax = fig.add_subplot(gs[3, 0])
    pct_zeros = (expr_cpm == 0).sum(axis=1) / n_samples * 100
    ax.hist(pct_zeros, bins=100, color='#f0883e', alpha=0.85, edgecolor='none')
    ax.axvline(np.median(pct_zeros), color='#58a6ff', ls='--', lw=2,
               label=f'Median = {np.median(pct_zeros):.0f}%')
    ax.set_xlabel('% zero samples per gene'); ax.set_ylabel('# genes')
    ax.set_title('Gene Sparsity', fontsize=14, fontweight='bold', pad=10)
    ax.legend(fontsize=10, facecolor=CARD, edgecolor=GRID, labelcolor=TEXT)
    ax.grid(axis='y', alpha=0.3)

    # Panel 8: Top 25 expressed genes — horizontal bars show mean CPM,
    # orange dots on the secondary x-axis show CV. Lets you see at a
    # glance whether the most abundant genes are also the most stable.
    ax = fig.add_subplot(gs[3, 1])
    top25 = np.argsort(cpm_means)[::-1][:25][::-1]
    ax.barh(range(25), cpm_means[top25] / 1e3, color=color, alpha=0.7, edgecolor='none', height=0.7)
    ax.set_yticks(range(25))
    ax.set_yticklabels([names[i] for i in top25], fontsize=8)
    ax.set_xlabel('Mean CPM (thousands)')
    ax.set_title('Top 25 Expressed Genes', fontsize=14, fontweight='bold', pad=10)
    ax8b = ax.twiny()
    ax8b.scatter([cvs[i] for i in top25], range(25), color='#f78166', s=40, zorder=5, edgecolors='none')
    ax8b.set_xlabel('CV', color='#f78166')
    ax8b.tick_params(colors='#f78166')
    ax.grid(axis='x', alpha=0.3)

    plt.savefig(outpath, dpi=180, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"  Saved: {outpath}")


def make_comparison_figure(results, outpath):
    """Generate 6-panel cross-tissue comparison figure."""
    fig = plt.figure(figsize=(22, 18))
    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3,
                  left=0.07, right=0.95, top=0.93, bottom=0.05)
    fig.suptitle('GTEx v11 — Whole Blood vs Liver Comparison',
                 fontsize=20, fontweight='bold', color=TEXT, y=0.975)

    tissues = list(results.keys())

    # Panel 1: Overlaid CV distributions — direct comparison of how
    # variable each tissue is. A leftward shift = more stable tissue.
    ax = fig.add_subplot(gs[0, 0])
    for t in tissues:
        cvs = results[t]['cvs_all']
        ax.hist(cvs, bins=200, range=(0, 2.0), alpha=0.55, color=TISSUE_COLORS[t],
                label=f'{t} (median={np.median(cvs):.3f})', edgecolor='none', density=True)
    ax.set_xlabel('CV'); ax.set_ylabel('Density')
    ax.set_title('CV Distribution Comparison', fontsize=14, fontweight='bold', pad=10)
    ax.legend(fontsize=10, facecolor=CARD, edgecolor=GRID, labelcolor=TEXT)
    ax.grid(axis='y', alpha=0.3)

    # Panel 2: Library size comparison — were both tissues sequenced
    # to similar depth? Differences here affect statistical power.
    ax = fig.add_subplot(gs[0, 1])
    for t in tissues:
        lib_m = results[t]['lib_sums'] / 1e6
        ax.hist(lib_m, bins=40, alpha=0.55, color=TISSUE_COLORS[t],
                label=f'{t} (median={np.median(lib_m):.0f}M)', edgecolor='none', density=True)
    ax.set_xlabel('Total reads (millions)'); ax.set_ylabel('Density')
    ax.set_title('Library Size Comparison', fontsize=14, fontweight='bold', pad=10)
    ax.legend(fontsize=10, facecolor=CARD, edgecolor=GRID, labelcolor=TEXT)
    ax.grid(axis='y', alpha=0.3)

    # Panel 3: CPM distribution comparison — do the tissues have similar
    # expression level profiles, or does one tissue concentrate reads
    # more heavily in fewer genes?
    ax = fig.add_subplot(gs[1, 0])
    for t in tissues:
        cpm_nz = results[t]['expr_cpm'][results[t]['expr_cpm'] > 0]
        rng = np.random.default_rng(42)
        if len(cpm_nz) > 2_000_000:
            cpm_nz = cpm_nz[rng.choice(len(cpm_nz), 2_000_000, replace=False)]
        ax.hist(np.log10(cpm_nz), bins=150, alpha=0.55, color=TISSUE_COLORS[t],
                label=t, edgecolor='none', density=True)
    ax.set_xlabel('log₁₀(CPM)'); ax.set_ylabel('Density')
    ax.set_title('CPM Distribution Comparison', fontsize=14, fontweight='bold', pad=10)
    ax.legend(fontsize=10, facecolor=CARD, edgecolor=GRID, labelcolor=TEXT)
    ax.grid(axis='y', alpha=0.3)

    # Panel 4: Whole vs within-cluster CV — which tissue benefits more
    # from clustering? A bigger gap = more biological substructure.
    ax = fig.add_subplot(gs[1, 1])
    x = np.arange(2)
    for i, t in enumerate(tissues):
        whole_cv = results[t]['cvs_all'].mean()
        within_cv = np.mean([d['mean_cv'] for d in results[t]['cluster_data']])
        ax.bar(i - 0.15, whole_cv, 0.28, color=TISSUE_COLORS[t], alpha=0.85, label=f'{t} whole')
        ax.bar(i + 0.15, within_cv, 0.28, color=TISSUE_COLORS[t], alpha=0.45, label=f'{t} within-cluster')
        ax.text(i - 0.15, whole_cv + 0.005, f'{whole_cv:.3f}', ha='center', fontsize=10, color=TEXT)
        ax.text(i + 0.15, within_cv + 0.005, f'{within_cv:.3f}', ha='center', fontsize=10, color=TEXT)
    ax.set_xticks(x); ax.set_xticklabels(tissues)
    ax.set_ylabel('Mean CV')
    ax.set_title('Whole vs Within-Cluster CV', fontsize=14, fontweight='bold', pad=10)
    ax.legend(fontsize=9, facecolor=CARD, edgecolor=GRID, labelcolor=TEXT)
    ax.grid(axis='y', alpha=0.3)

    # Panel 5: Sparsity comparison — which tissue has more genes with
    # zero counts? Higher sparsity = more genes not detected.
    ax = fig.add_subplot(gs[2, 0])
    for t in tissues:
        pz = (results[t]['expr_cpm'] == 0).sum(axis=1) / results[t]['n_samples'] * 100
        ax.hist(pz, bins=100, alpha=0.55, color=TISSUE_COLORS[t],
                label=f'{t} (median={np.median(pz):.0f}%)', edgecolor='none', density=True)
    ax.set_xlabel('% zero samples per gene'); ax.set_ylabel('Density')
    ax.set_title('Sparsity Comparison', fontsize=14, fontweight='bold', pad=10)
    ax.legend(fontsize=10, facecolor=CARD, edgecolor=GRID, labelcolor=TEXT)
    ax.grid(axis='y', alpha=0.3)

    # Panel 6: Gene-level CV scatter — for each gene expressed in both
    # tissues, plot its blood CV vs liver CV. Points on the y=x line
    # have equal variability in both tissues. Points above = more
    # variable in liver; below = more variable in blood. Pearson r
    # tells you how correlated variability is across tissues.
    ax = fig.add_subplot(gs[2, 1])
    # Match genes by name across both tissues
    blood = results['Whole Blood']
    liver = results['Liver']
    b_dict = {n: i for i, n in enumerate(blood['names_filt'])}
    shared = []
    for i, n in enumerate(liver['names_filt']):
        if n in b_dict:
            bi = b_dict[n]
            bcv = blood['stats']['cvs'][bi]
            lcv = liver['stats']['cvs'][i]
            if not (np.isnan(bcv) or np.isnan(lcv)):
                shared.append((bcv, lcv, n))

    if shared:
        bcvs, lcvs, snames = zip(*shared)
        bcvs = np.array(bcvs); lcvs = np.array(lcvs)
        ax.scatter(bcvs, lcvs, s=3, alpha=0.3, c='#d2a8ff', rasterized=True)
        lim = max(bcvs.max(), lcvs.max()) * 1.05
        ax.plot([0, lim], [0, lim], '--', color=MUTED, lw=1.5, label='y = x')
        ax.set_xlabel('Whole Blood CV'); ax.set_ylabel('Liver CV')
        ax.set_title(f'Gene CV: Blood vs Liver ({len(shared):,} shared genes)', fontsize=14, fontweight='bold', pad=10)
        # Label outliers
        diff = np.array(lcvs) - np.array(bcvs)
        top_liver = np.argsort(diff)[::-1][:5]
        top_blood = np.argsort(diff)[:5]
        for idx in list(top_liver) + list(top_blood):
            ax.annotate(snames[idx], (bcvs[idx], lcvs[idx]),
                         fontsize=7, color='#f78166', xytext=(3, 3), textcoords='offset points')
        corr = np.corrcoef(bcvs, lcvs)[0, 1]
        ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes,
                fontsize=12, color=TEXT, va='top')
        ax.legend(fontsize=10, facecolor=CARD, edgecolor=GRID, labelcolor=TEXT)
    ax.grid(alpha=0.3)

    plt.savefig(outpath, dpi=180, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"  Saved: {outpath}")


# Generate figures
print(f"\n{'#'*70}")
print(f"  GENERATING FIGURES")
print(f"{'#'*70}")

basedir = '/Users/rls/Desktop/programming-projects/single-cell/bulk-project'
for tissue in TISSUES:
    slug = tissue.lower().replace(' ', '_')
    make_tissue_figure(tissue, results[tissue], f'{basedir}/gtex_{slug}_analysis.png')

make_comparison_figure(results, f'{basedir}/gtex_blood_vs_liver.png')
print("\nDone.")
