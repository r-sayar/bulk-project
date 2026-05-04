import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
import gzip

# Load data
df = pd.read_csv('/Users/rls/Downloads/gene_reads_v11_whole_blood.gct.gz', sep='\t', skiprows=2, compression='gzip')
expr = df.iloc[:, 2:].values.astype(np.float64)
gene_names = df['Description'].values
n_genes, n_samples = expr.shape

# Compute per-gene stats
gene_means = np.mean(expr, axis=1)
gene_vars = np.var(expr, axis=1)
gene_stds = np.std(expr, axis=1)
gene_cvs = np.where(gene_means > 0, gene_stds / gene_means, np.nan)
sample_sums = np.sum(expr, axis=0)

# Color palette
BG = '#0e1117'
CARD = '#1a1d23'
TEXT = '#e6edf3'
MUTED = '#7d8590'
ACCENT1 = '#58a6ff'
ACCENT2 = '#f78166'
ACCENT3 = '#3fb950'
ACCENT4 = '#d2a8ff'
ACCENT5 = '#f0883e'
GRID = '#21262d'

plt.rcParams.update({
    'figure.facecolor': BG,
    'axes.facecolor': CARD,
    'axes.edgecolor': GRID,
    'axes.labelcolor': TEXT,
    'text.color': TEXT,
    'xtick.color': MUTED,
    'ytick.color': MUTED,
    'grid.color': GRID,
    'grid.alpha': 0.5,
    'font.family': 'sans-serif',
    'font.size': 11,
})

fig = plt.figure(figsize=(22, 28))
gs = GridSpec(4, 2, figure=fig, hspace=0.35, wspace=0.25,
             left=0.07, right=0.95, top=0.95, bottom=0.03)

fig.suptitle('GTEx v11 — Whole Blood Gene Expression Variance Analysis',
             fontsize=20, fontweight='bold', color=TEXT, y=0.98)
fig.text(0.5, 0.965, f'{n_genes:,} genes  ·  {n_samples} samples  ·  raw read counts',
         ha='center', fontsize=13, color=MUTED)

# ── 1. Distribution of read counts (log10) ──
ax1 = fig.add_subplot(gs[0, 0])
all_nonzero = expr[expr > 0].ravel()
# subsample for histogram if too many points
rng = np.random.default_rng(42)
if len(all_nonzero) > 5_000_000:
    sample_idx = rng.choice(len(all_nonzero), 5_000_000, replace=False)
    all_nonzero_sub = all_nonzero[sample_idx]
else:
    all_nonzero_sub = all_nonzero
log_counts = np.log10(all_nonzero_sub)
ax1.hist(log_counts, bins=150, color=ACCENT1, alpha=0.85, edgecolor='none')
ax1.axvline(np.log10(np.median(all_nonzero)), color=ACCENT2, ls='--', lw=2,
            label=f'Median = {np.median(all_nonzero):,.0f}')
ax1.axvline(np.log10(np.mean(all_nonzero)), color=ACCENT3, ls='--', lw=2,
            label=f'Mean = {np.mean(all_nonzero):,.0f}')
ax1.set_xlabel('log₁₀(read count)')
ax1.set_ylabel('Frequency')
ax1.set_title('Distribution of Non-Zero Read Counts', fontsize=14, fontweight='bold', pad=10)
ax1.legend(fontsize=10, facecolor=CARD, edgecolor=GRID, labelcolor=TEXT)
ax1.text(0.98, 0.95, f'{(expr==0).sum()/expr.size*100:.0f}% zeros excluded',
         transform=ax1.transAxes, ha='right', va='top', fontsize=9, color=MUTED)
ax1.grid(axis='y', alpha=0.3)

# ── 2. Mean vs Variance (log-log) ──
ax2 = fig.add_subplot(gs[0, 1])
mask_expr = gene_means > 0
gm = gene_means[mask_expr]
gv = gene_vars[mask_expr]
gn = gene_names[mask_expr]
ax2.scatter(np.log10(gm), np.log10(gv), s=1.5, alpha=0.25, c=ACCENT1, rasterized=True)
# Poisson line (var = mean)
x_range = np.linspace(np.log10(gm.min()), np.log10(gm.max()), 100)
ax2.plot(x_range, x_range, color=ACCENT3, ls='--', lw=2, label='Poisson (var = mean)')
# Quadratic line (var = mean^2)
ax2.plot(x_range, 2 * x_range, color=ACCENT2, ls=':', lw=2, label='var = mean²')
# Label top genes
top_var_idx = np.argsort(gv)[::-1][:8]
for i in top_var_idx:
    ax2.annotate(gn[i], (np.log10(gm[i]), np.log10(gv[i])),
                 fontsize=8, color=ACCENT2, fontweight='bold',
                 xytext=(5, 5), textcoords='offset points')
ax2.set_xlabel('log₁₀(mean expression)')
ax2.set_ylabel('log₁₀(variance)')
ax2.set_title('Mean–Variance Relationship', fontsize=14, fontweight='bold', pad=10)
ax2.legend(fontsize=10, facecolor=CARD, edgecolor=GRID, labelcolor=TEXT, loc='upper left')
ax2.grid(alpha=0.3)

# ── 3. Distribution of CV ──
ax3 = fig.add_subplot(gs[1, 0])
valid_cvs = gene_cvs[~np.isnan(gene_cvs) & (gene_means > 1)]
ax3.hist(valid_cvs, bins=200, color=ACCENT4, alpha=0.85, edgecolor='none', range=(0, 15))
ax3.axvline(np.median(valid_cvs), color=ACCENT2, ls='--', lw=2,
            label=f'Median CV = {np.median(valid_cvs):.2f}')
ax3.axvline(1.0, color=ACCENT3, ls=':', lw=2, label='CV = 1 (std = mean)')
ax3.set_xlabel('Coefficient of Variation (CV = std / mean)')
ax3.set_ylabel('Number of genes')
ax3.set_title('Distribution of Per-Gene CV (genes with mean > 1)', fontsize=14, fontweight='bold', pad=10)
ax3.legend(fontsize=10, facecolor=CARD, edgecolor=GRID, labelcolor=TEXT)
ax3.grid(axis='y', alpha=0.3)

# ── 4. Library size distribution ──
ax4 = fig.add_subplot(gs[1, 1])
lib_millions = sample_sums / 1e6
ax4.hist(lib_millions, bins=60, color=ACCENT3, alpha=0.85, edgecolor='none')
ax4.axvline(np.median(lib_millions), color=ACCENT2, ls='--', lw=2,
            label=f'Median = {np.median(lib_millions):.1f}M')
ax4.set_xlabel('Total reads per sample (millions)')
ax4.set_ylabel('Number of samples')
ax4.set_title('Library Size Distribution', fontsize=14, fontweight='bold', pad=10)
ax4.legend(fontsize=10, facecolor=CARD, edgecolor=GRID, labelcolor=TEXT)
ax4.grid(axis='y', alpha=0.3)

# ── 5. Gene expression level categories (pie) ──
ax5 = fig.add_subplot(gs[2, 0])
categories = ['Zero\n(mean=0)', 'Very low\n(0–1)', 'Low\n(1–100)', 'Medium\n(100–10k)', 'High\n(>10k)']
counts = [
    np.sum(gene_means == 0),
    np.sum((gene_means > 0) & (gene_means <= 1)),
    np.sum((gene_means > 1) & (gene_means <= 100)),
    np.sum((gene_means > 100) & (gene_means <= 10000)),
    np.sum(gene_means > 10000),
]
colors = ['#484f58', ACCENT1, ACCENT4, ACCENT3, ACCENT2]
wedges, texts, autotexts = ax5.pie(
    counts, labels=categories, autopct='%1.1f%%', colors=colors,
    textprops={'color': TEXT, 'fontsize': 10},
    pctdistance=0.75, labeldistance=1.12,
    wedgeprops={'linewidth': 1.5, 'edgecolor': BG}
)
for t in autotexts:
    t.set_fontsize(9)
    t.set_color(TEXT)
ax5.set_title('Gene Expression Level Categories', fontsize=14, fontweight='bold', pad=10)

# ── 6. Top 20 most variable genes (bar) ──
ax6 = fig.add_subplot(gs[2, 1])
top20_idx = np.argsort(gene_vars)[::-1][:20][::-1]
top20_names = gene_names[top20_idx]
top20_stds = gene_stds[top20_idx]
top20_means = gene_means[top20_idx]
bars = ax6.barh(range(20), top20_stds / 1e3, color=ACCENT2, alpha=0.85, edgecolor='none', height=0.7)
ax6.set_yticks(range(20))
ax6.set_yticklabels(top20_names, fontsize=9)
ax6.set_xlabel('Standard Deviation (thousands of reads)')
ax6.set_title('Top 20 Most Variable Genes (by std dev)', fontsize=14, fontweight='bold', pad=10)
for i, (bar, m) in enumerate(zip(bars, top20_means)):
    ax6.text(bar.get_width() + 20, bar.get_y() + bar.get_height()/2,
             f'μ={m/1e3:.0f}k', va='center', fontsize=8, color=MUTED)
ax6.grid(axis='x', alpha=0.3)

# ── 7. Sparsity per gene (% zeros) ──
ax7 = fig.add_subplot(gs[3, 0])
pct_zeros = (expr == 0).sum(axis=1) / n_samples * 100
ax7.hist(pct_zeros, bins=100, color=ACCENT5, alpha=0.85, edgecolor='none')
ax7.axvline(np.median(pct_zeros), color=ACCENT1, ls='--', lw=2,
            label=f'Median = {np.median(pct_zeros):.0f}%')
ax7.set_xlabel('% zero samples per gene')
ax7.set_ylabel('Number of genes')
ax7.set_title('Gene Sparsity (Dropout Rate)', fontsize=14, fontweight='bold', pad=10)
ax7.legend(fontsize=10, facecolor=CARD, edgecolor=GRID, labelcolor=TEXT)
ax7.grid(axis='y', alpha=0.3)

# ── 8. Top genes: mean expression with CV overlay ──
ax8 = fig.add_subplot(gs[3, 1])
top_expr_idx = np.argsort(gene_means)[::-1][:25][::-1]
te_names = gene_names[top_expr_idx]
te_means = gene_means[top_expr_idx] / 1e6
te_cvs = gene_cvs[top_expr_idx]

bars2 = ax8.barh(range(25), te_means, color=ACCENT1, alpha=0.7, edgecolor='none', height=0.7)
ax8.set_yticks(range(25))
ax8.set_yticklabels(te_names, fontsize=8)
ax8.set_xlabel('Mean expression (millions of reads)', color=ACCENT1)

ax8b = ax8.twiny()
ax8b.scatter(te_cvs, range(25), color=ACCENT2, s=40, zorder=5, edgecolors='none')
ax8b.set_xlabel('CV (coefficient of variation)', color=ACCENT2)
ax8b.tick_params(colors=ACCENT2)
ax8b.spines['top'].set_color(ACCENT2)

ax8.set_title('Top 25 Expressed Genes: Mean & CV', fontsize=14, fontweight='bold', pad=25)
ax8.grid(axis='x', alpha=0.3)

plt.savefig('/Users/rls/Desktop/programming-projects/single-cell/bulk-project/gtex_variance_analysis.png',
            dpi=180, bbox_inches='tight', facecolor=BG)
plt.close()
print('Saved.')
