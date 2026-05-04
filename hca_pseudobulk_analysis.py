"""
HCA Blood Pseudobulk Analysis
==============================
Downloads single-cell 10x CellRanger h5 count matrices from the Human Cell
Atlas (HematopoieticImmuneCellAtlas project), aggregates cells per donor into
pseudobulk, and runs the same variance analysis pipeline as the GTEx bulk
RNA-seq comparison.

Data source:
  - HCA Data Portal: HematopoieticImmuneCellAtlas project
  - Peripheral blood (BL) samples, 8 donors, 10x 3' v2
  - Each donor has 7 sequencing runs (lanes), each producing an h5 file

Pipeline:
  1. Download raw_feature_bc_matrix.h5 files from HCA
  2. Filter barcodes to real cells (>= 200 UMIs)
  3. Sum all cells per donor across all runs -> pseudobulk matrix
  4. Preprocess: filter genes, CPM normalize, log2 transform
  5. Compute per-gene variance stats (mean, std, CV)
  6. Compare against GTEx bulk blood
  7. Visualize
"""

import pandas as pd
import numpy as np
import h5py
import scipy.sparse as sp
import os
import glob
from collections import defaultdict
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════

BASEDIR = '/Users/rls/Desktop/programming-projects/single-cell/bulk-project'
H5_DIR = f'{BASEDIR}/pseudobulk/blood_h5'
PSEUDOBULK_PATH = f'{BASEDIR}/pseudobulk/hca_blood_pseudobulk.npz'
GTEX_BLOOD_PATH = '/Users/rls/Downloads/gene_reads_v11_whole_blood.gct.gz'

# Cell filtering: only keep barcodes with >= this many UMIs.
# CellRanger raw h5 includes all barcodes (including empty droplets).
# Real cells typically have >= 200 UMIs; empties have < 50.
MIN_UMI = 200

# Gene filtering (same as GTEx pipeline)
CPM_THRESHOLD = 1
MIN_SAMPLE_FRAC = 0.1

# ══════════════════════════════════════════════════════════════════════
# STYLE
# ══════════════════════════════════════════════════════════════════════

BG = '#0e1117'; CARD = '#1a1d23'; TEXT = '#e6edf3'; MUTED = '#7d8590'; GRID = '#21262d'
C_GTEX = '#f78166'; C_HCA = '#3fb950'

plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': CARD, 'axes.edgecolor': GRID,
    'axes.labelcolor': TEXT, 'text.color': TEXT, 'xtick.color': MUTED,
    'ytick.color': MUTED, 'grid.color': GRID, 'grid.alpha': 0.5,
    'font.family': 'sans-serif', 'font.size': 11,
})

# ══════════════════════════════════════════════════════════════════════
# 1. CREATE PSEUDOBULK FROM SINGLE-CELL H5 FILES
# Each h5 file is a CellRanger "raw_feature_bc_matrix" containing a
# sparse matrix of (genes x barcodes). We:
#   a) Filter to real cells (>= MIN_UMI total counts)
#   b) Sum all cells for that donor+run
#   c) Accumulate across all 7 runs per donor
# Result: one pseudobulk column per donor (8 total).
# ══════════════════════════════════════════════════════════════════════

def get_donor_from_filename(fname):
    """Extract donor ID from filename like '1_BL3_cells__MantonBL3_...'"""
    return os.path.basename(fname).split('_')[1]  # "BL3"


def create_pseudobulk(h5_dir, min_umi=MIN_UMI):
    """
    Aggregate single-cell h5 files into a pseudobulk matrix.

    Returns:
        pseudobulk: (n_genes, n_donors) array of summed counts
        gene_names: array of gene symbols
        gene_ids: array of Ensembl IDs
        donors: sorted list of donor IDs
        total_cells: dict mapping donor -> number of real cells
    """
    files = sorted(glob.glob(f'{h5_dir}/*_BL*_raw_feature_bc_matrix.h5'))
    if not files:
        raise FileNotFoundError(f"No h5 files found in {h5_dir}")

    print(f"Found {len(files)} h5 files")

    # Group files by donor
    donor_files = defaultdict(list)
    for f in files:
        donor_files[get_donor_from_filename(f)].append(f)

    donors = sorted(donor_files.keys())
    print(f"Donors: {donors}")

    # Read gene info from first file (same across all files)
    with h5py.File(files[0], 'r') as f:
        gene_ids = f['matrix/features/id'][:].astype(str)
        gene_names = f['matrix/features/name'][:].astype(str)
    n_genes = len(gene_ids)

    # Aggregate cells per donor
    pseudobulk = np.zeros((n_genes, len(donors)), dtype=np.float64)
    total_cells = {}

    for i, donor in enumerate(donors):
        n_cells_donor = 0
        for fpath in donor_files[donor]:
            with h5py.File(fpath, 'r') as f:
                # Read sparse matrix in CSC format (genes x barcodes)
                data = f['matrix/data'][:]
                indices = f['matrix/indices'][:]
                indptr = f['matrix/indptr'][:]
                shape = f['matrix/shape'][:]
                mat = sp.csc_matrix((data, indices, indptr), shape=shape)

                # Filter to real cells: total UMI per barcode >= threshold.
                # Empty droplets have very few UMIs and would add noise.
                cell_umi = np.array(mat.sum(axis=0)).ravel()
                real_cells = cell_umi >= min_umi
                mat_filtered = mat[:, real_cells]

                # Sum across all real cells -> one pseudobulk vector
                pseudobulk[:, i] += np.array(mat_filtered.sum(axis=1)).ravel()
                n_cells_donor += real_cells.sum()

        total_cells[donor] = n_cells_donor
        print(f"  {donor}: {n_cells_donor:,} cells, counts: {pseudobulk[:, i].sum():,.0f}")

    return pseudobulk, gene_names, gene_ids, donors, total_cells


# ══════════════════════════════════════════════════════════════════════
# 2. PREPROCESSING
# Same steps as the GTEx pipeline: filter low genes, CPM, log2.
# ══════════════════════════════════════════════════════════════════════

def preprocess_pseudobulk(expr_raw, gene_names):
    """Filter, CPM-normalize, and log-transform the pseudobulk matrix."""
    n_genes, n_samples = expr_raw.shape

    # Filter: CPM > 1 in at least 10% of samples
    lib_sizes = expr_raw.sum(axis=0)
    cpm = expr_raw / lib_sizes * 1e6
    min_samples = max(1, int(MIN_SAMPLE_FRAC * n_samples))
    keep = (cpm > CPM_THRESHOLD).sum(axis=1) >= min_samples

    expr_filt = expr_raw[keep]
    names_filt = gene_names[keep]

    # CPM normalize
    expr_cpm = expr_filt / expr_filt.sum(axis=0) * 1e6

    # Log2(CPM + 1)
    expr_log = np.log2(expr_cpm + 1)

    print(f"Filtered: {keep.sum():,} / {n_genes:,} genes kept")
    return expr_filt, expr_cpm, expr_log, names_filt


# ══════════════════════════════════════════════════════════════════════
# 3. VARIANCE STATISTICS
# ══════════════════════════════════════════════════════════════════════

def compute_stats(expr_log, expr_cpm, names_filt):
    """Compute per-gene mean, std, CV and print summary tables."""
    n_genes, n_samples = expr_log.shape
    means = expr_log.mean(axis=1)
    stds = expr_log.std(axis=1)
    cpm_means = expr_cpm.mean(axis=1)

    # CV only for expressed genes (mean > 0.5 in log2 space)
    expressed = means > 0.5
    cvs = np.full(n_genes, np.nan)
    cvs[expressed] = stds[expressed] / means[expressed]
    valid_cvs = cvs[~np.isnan(cvs)]
    n_expr = expressed.sum()

    print(f"\nExpressed genes: {n_expr:,}")
    print(f"Mean CV:   {valid_cvs.mean():.4f}")
    print(f"Median CV: {np.median(valid_cvs):.4f}")

    # CV threshold table
    print(f"\nCV threshold breakdown:")
    print(f"  {'CV range':<14} {'# genes':>8} {'Mean log2':>10} {'Med log2':>9} {'Std log2':>9} {'Mean CPM':>10} {'Med CPM':>9}")
    print(f"  {'-'*73}")
    thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.50]
    prev = 0
    for t in thresholds:
        mask = expressed & (cvs >= prev) & (cvs < t)
        n = mask.sum()
        if n > 0:
            print(f"  {prev:.2f}–{t:.2f}     {n:>8,} {means[mask].mean():>10.3f} {np.median(means[mask]):>9.3f} {stds[mask].mean():>9.3f} {cpm_means[mask].mean():>10.1f} {np.median(cpm_means[mask]):>9.1f}")
        prev = t
    mask = expressed & (cvs >= 0.50)
    n = mask.sum()
    if n > 0:
        print(f"  ≥0.50         {n:>8,} {means[mask].mean():>10.3f} {np.median(means[mask]):>9.3f} {stds[mask].mean():>9.3f} {cpm_means[mask].mean():>10.1f} {np.median(cpm_means[mask]):>9.1f}")

    # Top 20 most variable
    print(f"\nTop 20 most variable genes:")
    print(f"  {'Gene':<16} {'CV':>7} {'Mean log2':>10} {'Mean CPM':>10}")
    print(f"  {'-'*48}")
    top_idx = np.argsort(np.nan_to_num(cvs, nan=-1))[::-1][:20]
    for idx in top_idx:
        print(f"  {names_filt[idx]:<16} {cvs[idx]:>7.3f} {means[idx]:>10.3f} {cpm_means[idx]:>10.2f}")

    # Top 20 most stable (CPM > 100)
    print(f"\nTop 20 most stable genes (CPM > 100):")
    print(f"  {'Gene':<16} {'CV':>7} {'Mean log2':>10} {'Mean CPM':>10}")
    print(f"  {'-'*48}")
    stable_cvs = cvs.copy()
    stable_cvs[cpm_means <= 100] = np.inf
    stable_cvs[np.isnan(stable_cvs)] = np.inf
    stable_idx = np.argsort(stable_cvs)[:20]
    for idx in stable_idx:
        print(f"  {names_filt[idx]:<16} {cvs[idx]:>7.4f} {means[idx]:>10.3f} {cpm_means[idx]:>10.1f}")

    return {
        'means': means, 'stds': stds, 'cvs': cvs,
        'cpm_means': cpm_means, 'expressed': expressed,
    }


# ══════════════════════════════════════════════════════════════════════
# 4. LOAD GTEX FOR COMPARISON
# ══════════════════════════════════════════════════════════════════════

def load_and_preprocess_gtex(path):
    """Load GTEx bulk blood data and run same preprocessing."""
    df = pd.read_csv(path, sep='\t', skiprows=2, compression='gzip')
    expr_raw = df.iloc[:, 2:].values.astype(np.float64)
    gene_names = df['Description'].values
    n_samples = expr_raw.shape[1]

    lib_sizes = expr_raw.sum(axis=0)
    cpm = expr_raw / lib_sizes * 1e6
    keep = (cpm > CPM_THRESHOLD).sum(axis=1) >= int(MIN_SAMPLE_FRAC * n_samples)

    expr_filt = expr_raw[keep]
    names_filt = gene_names[keep]
    expr_cpm = expr_filt / expr_filt.sum(axis=0) * 1e6
    expr_log = np.log2(expr_cpm + 1)

    means = expr_log.mean(axis=1)
    stds = expr_log.std(axis=1)
    expressed = means > 0.5
    cvs_all = stds[expressed] / means[expressed]

    return {
        'expr_log': expr_log, 'expr_cpm': expr_cpm,
        'names_filt': names_filt, 'lib_sizes': lib_sizes,
        'means': means, 'stds': stds, 'expressed': expressed,
        'cvs': cvs_all, 'n_samples': n_samples,
    }


# ══════════════════════════════════════════════════════════════════════
# 5. VISUALIZATION
# 6-panel comparison figure:
#   1. CV distribution overlay (GTEx vs HCA)
#   2. CPM distribution overlay
#   3. Gene-level CV scatter (shared genes between both datasets)
#   4. HCA library sizes per donor
#   5. Cells per donor
#   6. Top 15 most variable HCA genes
# ══════════════════════════════════════════════════════════════════════

def make_comparison_figure(hca, gtex, donors, total_cells, lib_sizes, outpath):
    """Generate 6-panel GTEx vs HCA comparison figure."""
    fig = plt.figure(figsize=(22, 14))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3,
                  left=0.06, right=0.96, top=0.92, bottom=0.06)
    fig.suptitle('HCA Blood Pseudobulk (8 donors, ~44k cells each) vs GTEx Bulk Blood (803 donors)',
                 fontsize=17, fontweight='bold', color=TEXT, y=0.97)

    hca_cvs = hca['cvs'][~np.isnan(hca['cvs'])]
    gtex_cvs = gtex['cvs']

    # Panel 1: CV distribution overlay — direct comparison of variability.
    # HCA pseudobulk should be much tighter (lower CV) because:
    #   a) only 8 donors (less biological diversity sampled)
    #   b) pseudobulk averaging smooths stochastic noise
    ax = fig.add_subplot(gs[0, 0])
    ax.hist(gtex_cvs, bins=200, range=(0, 2), alpha=0.55, color=C_GTEX, edgecolor='none',
            density=True, label=f'GTEx (med={np.median(gtex_cvs):.3f})')
    ax.hist(hca_cvs, bins=100, range=(0, 2), alpha=0.55, color=C_HCA, edgecolor='none',
            density=True, label=f'HCA pseudo (med={np.median(hca_cvs):.3f})')
    ax.set_xlabel('CV'); ax.set_ylabel('Density')
    ax.set_title('CV Distribution', fontsize=14, fontweight='bold', pad=10)
    ax.legend(fontsize=10, facecolor=CARD, edgecolor=GRID, labelcolor=TEXT)
    ax.grid(axis='y', alpha=0.3)

    # Panel 2: CPM distribution overlay — do both datasets have similar
    # expression level profiles?
    ax = fig.add_subplot(gs[0, 1])
    gtex_nz = gtex['expr_cpm'][gtex['expr_cpm'] > 0]
    rng = np.random.default_rng(42)
    if len(gtex_nz) > 2_000_000:
        gtex_nz = gtex_nz[rng.choice(len(gtex_nz), 2_000_000, replace=False)]
    hca_nz = hca['expr_cpm'][hca['expr_cpm'] > 0]
    ax.hist(np.log10(gtex_nz), bins=150, alpha=0.55, color=C_GTEX, edgecolor='none',
            density=True, label='GTEx')
    ax.hist(np.log10(hca_nz), bins=150, alpha=0.55, color=C_HCA, edgecolor='none',
            density=True, label='HCA pseudo')
    ax.set_xlabel('log₁₀(CPM)'); ax.set_ylabel('Density')
    ax.set_title('CPM Distribution', fontsize=14, fontweight='bold', pad=10)
    ax.legend(fontsize=10, facecolor=CARD, edgecolor=GRID, labelcolor=TEXT)
    ax.grid(axis='y', alpha=0.3)

    # Panel 3: Gene-level CV scatter — for each gene expressed in both
    # datasets, compare its CV. Pearson r measures how well variability
    # patterns are conserved between bulk and pseudobulk.
    ax = fig.add_subplot(gs[0, 2])
    gtex_cv_full = np.full(len(gtex['means']), np.nan)
    gtex_cv_full[gtex['expressed']] = gtex['cvs']
    hca_dict = {n: i for i, n in enumerate(hca['names_filt'])}
    shared = []
    for i, n in enumerate(gtex['names_filt']):
        if n in hca_dict:
            hi = hca_dict[n]
            gcv = gtex_cv_full[i]
            hcv = hca['cvs'][hi]
            if not (np.isnan(gcv) or np.isnan(hcv)):
                shared.append((gcv, hcv, n))
    if shared:
        gcvs, hcvs, snames = zip(*shared)
        gcvs = np.array(gcvs); hcvs = np.array(hcvs)
        ax.scatter(gcvs, hcvs, s=3, alpha=0.3, c='#d2a8ff', rasterized=True)
        lim = max(gcvs.max(), hcvs.max()) * 1.05
        ax.plot([0, lim], [0, lim], '--', color=MUTED, lw=1.5, label='y = x')
        corr = np.corrcoef(gcvs, hcvs)[0, 1]
        ax.text(0.05, 0.95, f'r = {corr:.3f}\n{len(shared):,} genes',
                transform=ax.transAxes, fontsize=12, color=TEXT, va='top')
        ax.set_xlabel('GTEx Bulk CV'); ax.set_ylabel('HCA Pseudobulk CV')
        ax.set_title('Gene CV Correlation', fontsize=14, fontweight='bold', pad=10)
        ax.legend(fontsize=10, facecolor=CARD, edgecolor=GRID, labelcolor=TEXT)
    ax.grid(alpha=0.3)

    # Panel 4: HCA library sizes — how consistent is sequencing depth
    # across the 8 donors after pseudobulk aggregation?
    ax = fig.add_subplot(gs[1, 0])
    ax.bar(range(len(donors)), lib_sizes / 1e6, color=C_HCA, alpha=0.85, edgecolor='none')
    ax.set_xticks(range(len(donors)))
    ax.set_xticklabels(donors)
    ax.set_ylabel('Total counts (millions)')
    ax.set_title('HCA Pseudobulk Library Sizes', fontsize=14, fontweight='bold', pad=10)
    for i, v in enumerate(lib_sizes):
        ax.text(i, v / 1e6 + 0.5, f'{v / 1e6:.0f}M', ha='center', fontsize=9, color=MUTED)
    ax.grid(axis='y', alpha=0.3)

    # Panel 5: Cells per donor — how many real cells were aggregated?
    # More cells = smoother pseudobulk = lower noise.
    ax = fig.add_subplot(gs[1, 1])
    cells_arr = np.array([total_cells[d] for d in donors])
    ax.bar(range(len(donors)), cells_arr / 1e3, color='#58a6ff', alpha=0.85, edgecolor='none')
    ax.set_xticks(range(len(donors)))
    ax.set_xticklabels(donors)
    ax.set_ylabel('Cells (thousands)')
    ax.set_title('Cells per Donor', fontsize=14, fontweight='bold', pad=10)
    for i, v in enumerate(cells_arr):
        ax.text(i, v / 1e3 + 0.3, f'{v / 1e3:.1f}k', ha='center', fontsize=9, color=MUTED)
    ax.grid(axis='y', alpha=0.3)

    # Panel 6: Top 15 most variable genes in the HCA pseudobulk.
    ax = fig.add_subplot(gs[1, 2])
    hca_cvs_full = hca['cvs']
    top15 = np.argsort(np.nan_to_num(hca_cvs_full, nan=-1))[::-1][:15][::-1]
    ax.barh(range(15), [hca_cvs_full[i] for i in top15], color=C_HCA, alpha=0.85,
            edgecolor='none', height=0.7)
    ax.set_yticks(range(15))
    ax.set_yticklabels([hca['names_filt'][i] for i in top15], fontsize=9)
    ax.set_xlabel('CV')
    ax.set_title('Top 15 Most Variable Genes (HCA)', fontsize=14, fontweight='bold', pad=10)
    ax.grid(axis='x', alpha=0.3)

    plt.savefig(outpath, dpi=180, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"Saved: {outpath}")


# ══════════════════════════════════════════════════════════════════════
# 6. RUN PIPELINE
# ══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    # Step 1: Create pseudobulk (or load if already saved)
    if os.path.exists(PSEUDOBULK_PATH):
        print(f"Loading existing pseudobulk from {PSEUDOBULK_PATH}")
        d = np.load(PSEUDOBULK_PATH)
        expr_raw = d['expr']
        gene_names = d['gene_names']
        donors = list(d['sample_ids'])
        total_cells = {donors[i]: d['total_cells'][i] for i in range(len(donors))}
    else:
        print("Creating pseudobulk from h5 files...")
        expr_raw, gene_names, gene_ids, donors, total_cells = create_pseudobulk(H5_DIR)
        np.savez(PSEUDOBULK_PATH,
                 expr=expr_raw, gene_names=gene_names, gene_ids=gene_ids,
                 sample_ids=np.array(donors),
                 total_cells=np.array([total_cells[d] for d in donors]))

    n_genes, n_samples = expr_raw.shape
    print(f"\nHCA Pseudobulk: {n_genes:,} genes x {n_samples} donors")
    print(f"% zeros: {(expr_raw == 0).sum() / expr_raw.size * 100:.1f}%")

    # Step 2: Preprocess
    expr_filt, expr_cpm, expr_log, names_filt = preprocess_pseudobulk(expr_raw, gene_names)
    lib_sizes = expr_raw.sum(axis=0)

    print(f"\nLibrary size (raw counts per donor):")
    print(f"  Mean: {lib_sizes.mean():>14,.0f}")
    print(f"  Std:  {lib_sizes.std():>14,.0f}")
    print(f"  CV:   {lib_sizes.std() / lib_sizes.mean():.4f}")

    # Step 3: Variance stats
    print(f"\n{'='*60}")
    print(f"HCA BLOOD PSEUDOBULK VARIANCE ANALYSIS")
    print(f"{'='*60}")
    hca_stats = compute_stats(expr_log, expr_cpm, names_filt)

    # Step 4: Load GTEx for comparison
    print(f"\n{'='*60}")
    print(f"LOADING GTEx BULK BLOOD FOR COMPARISON")
    print(f"{'='*60}")
    gtex = load_and_preprocess_gtex(GTEX_BLOOD_PATH)

    hca_cvs = hca_stats['cvs'][~np.isnan(hca_stats['cvs'])]
    print(f"\n{'='*60}")
    print(f"COMPARISON: GTEx Bulk Blood vs HCA Pseudobulk Blood")
    print(f"{'='*60}")
    print(f"  {'Metric':<30} {'GTEx Bulk':>14} {'HCA Pseudo':>14}")
    print(f"  {'-'*60}")
    print(f"  {'Samples/Donors':<30} {gtex['n_samples']:>14} {n_samples:>14}")
    print(f"  {'Genes after filter':<30} {len(gtex['names_filt']):>14,} {len(names_filt):>14,}")
    print(f"  {'Mean library size':<30} {gtex['lib_sizes'].mean()/1e6:>13.1f}M {lib_sizes.mean()/1e6:>13.1f}M")
    print(f"  {'Mean CV':<30} {gtex['cvs'].mean():>14.4f} {hca_cvs.mean():>14.4f}")
    print(f"  {'Median CV':<30} {np.median(gtex['cvs']):>14.4f} {np.median(hca_cvs):>14.4f}")

    # Step 5: Visualize
    hca_for_plot = {
        'cvs': hca_stats['cvs'], 'expr_cpm': expr_cpm,
        'names_filt': names_filt, 'means': hca_stats['means'],
        'expressed': hca_stats['expressed'],
    }
    make_comparison_figure(
        hca_for_plot, gtex, donors, total_cells, lib_sizes,
        f'{BASEDIR}/gtex_vs_hca_pseudobulk.png'
    )

    print("\nDone.")
