"""
100 genes stratified by expression level, categorised by type
(housekeeping / blood-specific / mitochondrial / other).
KDE with mode detection — flag bimodal genes.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks

# ── Load data ─────────────────────────────────────────────────────────
df = pd.read_csv(
    '/Users/rls/Downloads/gene_reads_v11_whole_blood.gct.gz',
    sep='\t', skiprows=2, compression='gzip',
)
expr = df.iloc[:, 2:].values.astype(np.float64)
gene_names = df['Description'].values
n_samples = expr.shape[1]

# CPM + log2 transform
lib_sizes = expr.sum(axis=0, keepdims=True)
cpm = expr / lib_sizes * 1e6
log_expr = np.log2(cpm + 1)

gene_means = log_expr.mean(axis=1)
gene_stds = log_expr.std(axis=1)

# ── Gene categories ───────────────────────────────────────────────────
HOUSEKEEPING = {
    'ACTB', 'GAPDH', 'B2M', 'EEF1A1', 'FTL', 'FKBP8', 'TMSB4X', 'PSAP',
    'RPL13A', 'RPL7', 'RPL3', 'RPS18', 'RPS27A', 'RPL11', 'RPL4', 'RPL8',
    'RPS3', 'RPS4X', 'RPL13', 'RPL6', 'RPL10', 'RPL5', 'RPS2', 'RPS14',
    'EEF2', 'UBC', 'UBB', 'PPIA', 'HSP90AB1', 'YWHAZ', 'ALDOA', 'ENO1',
    'LDHA', 'PKM', 'TPI1', 'PGK1', 'HNRNPA1', 'NPM1', 'CALM1', 'ATP5F1B',
    'NDUFA4', 'COX7C', 'ATP5MC3', 'VIM', 'FLNA', 'TPT1', 'EIF4A1',
}

BLOOD_IMMUNE = {
    'HBB', 'HBA1', 'HBA2', 'HBD', 'HBG1', 'HBG2',
    'S100A9', 'S100A8', 'S100A12', 'S100A6', 'S100A4',
    'LCP1', 'CSF3R', 'IFITM2', 'IFITM3', 'IFITM1',
    'HLA-A', 'HLA-B', 'HLA-C', 'HLA-E', 'HLA-DRA', 'HLA-DRB1',
    'SERPINA1', 'LYZ', 'MNDA', 'FGL2', 'AIF1', 'TYROBP', 'FCER1G',
    'CD74', 'FCN1', 'LST1', 'CTSS', 'CYBB', 'NCF2', 'SPI1',
    'IL1B', 'CXCL8', 'CCL3', 'PTPRC', 'CD14', 'ITGB2',
}

MITOCHONDRIAL = {n for n in gene_names if str(n).startswith('MT-')}

def categorize(name):
    if name in HOUSEKEEPING:
        return 'Housekeeping'
    elif name in BLOOD_IMMUNE:
        return 'Blood/Immune'
    elif name in MITOCHONDRIAL:
        return 'Mitochondrial'
    else:
        return 'Other'

CAT_COLORS = {
    'Housekeeping':   '#3fb950',
    'Blood/Immune':   '#f78166',
    'Mitochondrial':  '#d2a8ff',
    'Other':          '#58a6ff',
}

# ── Select 100 genes stratified across expression range ───────────────
# Filter to genes with reasonable expression (mean > 1 log2CPM)
expressed = np.where(gene_means > 1)[0]
sorted_by_mean = expressed[np.argsort(gene_means[expressed])[::-1]]

# Take evenly spaced genes across the expression range
step = max(1, len(sorted_by_mean) // 100)
selected_idx = sorted_by_mean[::step][:100]

# ── Bimodality detection ─────────────────────────────────────────────
def detect_modes(vals):
    """Return (xs, density, peaks, is_bimodal)."""
    xs = np.linspace(vals.min() - 0.5, vals.max() + 0.5, 1000)
    kde = gaussian_kde(vals, bw_method='scott')
    density = kde(xs)
    peaks, props = find_peaks(density, prominence=density.max() * 0.08,
                              distance=40)
    is_bimodal = len(peaks) >= 2
    return xs, density, peaks, is_bimodal

# Pre-scan all selected genes for bimodality
bimodal_genes = []
for gi in selected_idx:
    vals = log_expr[gi, :]
    if vals.std() < 0.1:
        continue
    _, _, peaks, is_bi = detect_modes(vals)
    if is_bi:
        bimodal_genes.append(gi)

# Also scan ALL expressed genes for bimodal ones and add top ones
print(f"Scanning {len(expressed)} expressed genes for bimodality...")
all_bimodal = []
for gi in expressed:
    vals = log_expr[gi, :]
    if vals.std() < 0.3:
        continue
    _, _, peaks, is_bi = detect_modes(vals)
    if is_bi:
        all_bimodal.append(gi)

print(f"Found {len(all_bimodal)} bimodal genes out of {len(expressed)} expressed genes")

# Ensure we include some bimodal genes in our 100
# Replace last slots with bimodal genes not already in selection
selected_set = set(selected_idx)
extra_bimodal = [g for g in all_bimodal if g not in selected_set]
np.random.seed(42)
if extra_bimodal:
    np.random.shuffle(extra_bimodal)
    n_add = min(20, len(extra_bimodal))
    selected_idx = np.concatenate([selected_idx[:100 - n_add],
                                    np.array(extra_bimodal[:n_add])])

# Sort final selection by mean expression descending
selected_idx = selected_idx[np.argsort(gene_means[selected_idx])[::-1]]

# ── Style ─────────────────────────────────────────────────────────────
BG    = '#0e1117'
CARD  = '#1a1d23'
TEXT  = '#e6edf3'
MUTED = '#7d8590'
GRID  = '#21262d'

plt.rcParams.update({
    'figure.facecolor': BG,
    'axes.facecolor': CARD,
    'axes.edgecolor': GRID,
    'axes.labelcolor': TEXT,
    'text.color': TEXT,
    'xtick.color': MUTED,
    'ytick.color': MUTED,
    'grid.color': GRID,
    'font.family': 'sans-serif',
    'font.size': 8,
})

# ── Plot: 10x10 grid ─────────────────────────────────────────────────
NCOLS = 10
NROWS = 10
fig, axes = plt.subplots(NROWS, NCOLS, figsize=(32, 28))
fig.suptitle(
    'GTEx Whole Blood — 100 Genes Stratified by Expression Level\n'
    'KDE distributions with mode detection  ·  '
    r'$\bf{Green}$=Housekeeping  $\bf{Orange}$=Blood/Immune  '
    r'$\bf{Purple}$=Mitochondrial  $\bf{Blue}$=Other  ·  '
    r'$\bigstar$ = Bimodal',
    fontsize=18, fontweight='bold', color=TEXT, y=0.998,
)

for idx in range(len(selected_idx)):
    row, col = divmod(idx, NCOLS)
    ax = axes[row, col]
    gene_i = selected_idx[idx]
    vals = log_expr[gene_i, :]
    name = gene_names[gene_i]
    cat = categorize(name)
    color = CAT_COLORS[cat]

    xs, density, peaks, is_bimodal = detect_modes(vals)

    ax.fill_between(xs, density, alpha=0.3, color=color)
    ax.plot(xs, density, color=color, lw=1.5)

    # Mark modes
    for pi in peaks:
        mode_val = xs[pi]
        n_near = np.sum(np.abs(vals - mode_val) < 0.5)
        pct = n_near / n_samples * 100
        ax.axvline(mode_val, color='#f0883e', ls='--', lw=1, alpha=0.7)
        ax.plot(mode_val, density[pi], 'o', color='#f0883e', ms=5, zorder=5)

    # Title with category and bimodal flag
    bimodal_flag = ' *' if is_bimodal else ''
    title_color = '#ff7b72' if is_bimodal else TEXT
    ax.set_title(f'{name}{bimodal_flag}\n[{cat}] μ={gene_means[gene_i]:.1f}',
                 fontsize=7, fontweight='bold', color=title_color, pad=3)

    # Mode annotation
    if peaks is not None and len(peaks) > 0:
        lines = []
        for pi in peaks:
            mode_val = xs[pi]
            n_near = np.sum(np.abs(vals - mode_val) < 0.5)
            lines.append(f'{n_near}@{mode_val:.1f}')
        ax.text(0.97, 0.95, '\n'.join(lines), transform=ax.transAxes,
                fontsize=6, color=MUTED, ha='right', va='top')

    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)

    # Border color for bimodal
    if is_bimodal:
        for spine in ax.spines.values():
            spine.set_edgecolor('#ff7b72')
            spine.set_linewidth(2)

# Hide unused
for idx in range(len(selected_idx), NROWS * NCOLS):
    row, col = divmod(idx, NCOLS)
    axes[row, col].set_visible(False)

plt.tight_layout(rect=[0, 0, 1, 0.97])
out = '/Users/rls/Desktop/programming-projects/single-cell/bulk-project/gtex_ranked_distributions.png'
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print(f'Saved: {out}')

# ── Print bimodal gene summary ────────────────────────────────────────
print(f"\n{'='*60}")
print(f"BIMODAL GENES FOUND: {len(all_bimodal)}")
print(f"{'='*60}")
for gi in all_bimodal[:30]:
    name = gene_names[gi]
    cat = categorize(name)
    vals = log_expr[gi, :]
    xs, density, peaks, _ = detect_modes(vals)
    modes_str = ', '.join([f'{xs[p]:.1f} ({np.sum(np.abs(vals - xs[p]) < 0.5)} samples)'
                           for p in peaks])
    print(f"  {name:15s} [{cat:14s}]  modes: {modes_str}")
