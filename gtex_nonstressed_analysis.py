#!/usr/bin/env python3
"""
GTEx Whole Blood: Remove stressed-state samples, then re-analyze bimodal gene states
in the remaining non-stressed population.
"""

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import scipy.cluster.hierarchy as sch
import warnings
warnings.filterwarnings('ignore')

# ── Theme ──
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
    'legend.facecolor': CARD,
    'legend.edgecolor': GRID,
    'font.size': 10,
})

# ── 1. Load GTEx whole blood ──
print("=" * 70)
print("LOADING GTEx whole blood data...")
df = pd.read_csv(
    '/Users/rls/Downloads/gene_reads_v11_whole_blood.gct.gz',
    sep='\t', skiprows=2, compression='gzip'
)
gene_names = df['Description'].values
expr = df.iloc[:, 2:].values.astype(np.float64)
sample_ids = df.columns[2:].values
print(f"  Loaded: {expr.shape[0]} genes x {expr.shape[1]} samples")

# ── 2. log2(CPM+1) ──
print("Computing log2(CPM+1)...")
lib_sizes = expr.sum(axis=0, keepdims=True)
cpm = expr / lib_sizes * 1e6
log_cpm = np.log2(cpm + 1)

# ── 3. Find bimodal genes (full dataset) ──
print("Finding bimodal genes in FULL dataset...")
gene_means = log_cpm.mean(axis=1)
gene_stds  = log_cpm.std(axis=1)
candidate_mask = (gene_means > 1) & (gene_stds > 0.3)
candidate_idx = np.where(candidate_mask)[0]
print(f"  Candidates (mean>1, std>0.3): {len(candidate_idx)}")

def find_bimodal_genes(data, indices, gene_names_arr):
    """Return indices of bimodal genes using KDE + peak finding."""
    bimodal = []
    for i in indices:
        vals = data[i]
        try:
            kde = gaussian_kde(vals, bw_method='scott')
            x_grid = np.linspace(vals.min(), vals.max(), 500)
            density = kde(x_grid)
            prominence_thresh = density.max() * 0.08
            peaks, props = find_peaks(density, prominence=prominence_thresh, distance=40)
            if len(peaks) >= 2:
                bimodal.append(i)
        except Exception:
            continue
    return np.array(bimodal)

bimodal_idx_full = find_bimodal_genes(log_cpm, candidate_idx, gene_names)
print(f"  Bimodal genes (full dataset): {len(bimodal_idx_full)}")

# ── 4. Binary matrix, PCA, split by PC1 ──
print("Building binary matrix (full dataset)...")
binary_full = np.zeros((len(bimodal_idx_full), log_cpm.shape[1]), dtype=np.int8)
thresholds_full = []
for j, gi in enumerate(bimodal_idx_full):
    vals = log_cpm[gi]
    kde = gaussian_kde(vals, bw_method='scott')
    x_grid = np.linspace(vals.min(), vals.max(), 500)
    density = kde(x_grid)
    peaks, _ = find_peaks(density, prominence=density.max() * 0.08, distance=40)
    if len(peaks) >= 2:
        sorted_peaks = np.sort(x_grid[peaks])
        thr = (sorted_peaks[0] + sorted_peaks[1]) / 2
    else:
        thr = np.median(vals)
    thresholds_full.append(thr)
    binary_full[j] = (vals > thr).astype(np.int8)

pca_full = PCA(n_components=2)
scores_full = pca_full.fit_transform(binary_full.T)
state_labels = (scores_full[:, 0] > 0).astype(int)  # 0 or 1

n_state0 = (state_labels == 0).sum()
n_state1 = (state_labels == 1).sum()
print(f"  State 0: {n_state0} samples, State 1: {n_state1} samples")

# ── 5. Determine which state is "stressed" ──
stress_markers = ['DDIT4', 'JUN', 'VEGFA']
stress_scores = {}
for s in [0, 1]:
    mask_s = state_labels == s
    total = 0
    for g in stress_markers:
        idx = np.where(gene_names == g)[0]
        if len(idx) > 0:
            total += log_cpm[idx[0], mask_s].mean()
    stress_scores[s] = total

stressed_state = max(stress_scores, key=stress_scores.get)
nonstressed_state = 1 - stressed_state
print(f"\n  Stress marker scores: State 0 = {stress_scores[0]:.2f}, State 1 = {stress_scores[1]:.2f}")
print(f"  --> Stressed state: {stressed_state}")
print(f"  --> Keeping state {nonstressed_state} (non-stressed)")

# ── 6. Remove stressed samples ──
keep_mask = state_labels == nonstressed_state
log_cpm_ns = log_cpm[:, keep_mask]
sample_ids_ns = sample_ids[keep_mask]
n_ns = log_cpm_ns.shape[1]
print(f"\n  Non-stressed samples retained: {n_ns}")

# ── 7a. Recompute bimodal genes on non-stressed samples ──
print("\n" + "=" * 70)
print("RE-ANALYZING bimodal genes in NON-STRESSED samples only...")
gene_means_ns = log_cpm_ns.mean(axis=1)
gene_stds_ns  = log_cpm_ns.std(axis=1)
candidate_mask_ns = (gene_means_ns > 1) & (gene_stds_ns > 0.3)
candidate_idx_ns = np.where(candidate_mask_ns)[0]
print(f"  Candidates (mean>1, std>0.3): {len(candidate_idx_ns)}")

bimodal_idx_ns = find_bimodal_genes(log_cpm_ns, candidate_idx_ns, gene_names)
bimodal_names_ns = gene_names[bimodal_idx_ns]
print(f"  Bimodal genes in non-stressed: {len(bimodal_idx_ns)}")

# ── 7b. Binary matrix for non-stressed ──
print("Building binary matrix for non-stressed samples...")
binary_ns = np.zeros((len(bimodal_idx_ns), n_ns), dtype=np.int8)
thresholds_ns = []
for j, gi in enumerate(bimodal_idx_ns):
    vals = log_cpm_ns[gi]
    kde = gaussian_kde(vals, bw_method='scott')
    x_grid = np.linspace(vals.min(), vals.max(), 500)
    density = kde(x_grid)
    peaks, _ = find_peaks(density, prominence=density.max() * 0.08, distance=40)
    if len(peaks) >= 2:
        sorted_peaks = np.sort(x_grid[peaks])
        thr = (sorted_peaks[0] + sorted_peaks[1]) / 2
    else:
        thr = np.median(vals)
    thresholds_ns.append(thr)
    binary_ns[j] = (vals > thr).astype(np.int8)

# ── 7c. PCA on new binary matrix ──
print("PCA on non-stressed binary matrix...")
n_comp = min(10, binary_ns.shape[0], binary_ns.shape[1])
pca_ns = PCA(n_components=n_comp)
scores_ns = pca_ns.fit_transform(binary_ns.T)
var_explained = pca_ns.explained_variance_ratio_

print(f"\n  PCA variance explained (non-stressed):")
for i in range(min(5, n_comp)):
    print(f"    PC{i+1}: {var_explained[i]*100:.1f}%")

# ── 7d. Cluster the non-stressed samples ──
# Try K=2 first, check if there are clear states
from sklearn.metrics import silhouette_score

best_k = 2
best_sil = -1
for k in range(2, min(6, n_ns)):
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels_k = km.fit_predict(scores_ns[:, :3])
    sil = silhouette_score(scores_ns[:, :3], labels_k)
    if sil > best_sil:
        best_sil = sil
        best_k = k

print(f"\n  Best K (silhouette): K={best_k} (silhouette={best_sil:.3f})")
km_final = KMeans(n_clusters=best_k, n_init=10, random_state=42)
cluster_labels_ns = km_final.fit_predict(scores_ns[:, :3])

for c in range(best_k):
    print(f"    Cluster {c}: {(cluster_labels_ns == c).sum()} samples")

# ── 7e. Anchor genes for new states ──
print("\n  ANCHOR GENES (most differentially expressed between new clusters):")
if best_k == 2:
    cluster_pairs = [(0, 1)]
else:
    cluster_pairs = [(i, j) for i in range(best_k) for j in range(i+1, best_k)]

diff_results = []
for c1, c2 in cluster_pairs:
    mask1 = cluster_labels_ns == c1
    mask2 = cluster_labels_ns == c2
    print(f"\n  --- Cluster {c1} vs Cluster {c2} ---")
    # Use all genes, not just bimodal
    mean1 = log_cpm_ns[:, mask1].mean(axis=1)
    mean2 = log_cpm_ns[:, mask2].mean(axis=1)
    diff = mean1 - mean2
    abs_diff = np.abs(diff)
    top_idx = np.argsort(abs_diff)[::-1][:20]

    print(f"  {'Gene':<15} {'MeanC'+str(c1):>10} {'MeanC'+str(c2):>10} {'Diff':>10}")
    print(f"  {'-'*45}")
    for idx in top_idx:
        g = gene_names[idx]
        print(f"  {g:<15} {mean1[idx]:>10.2f} {mean2[idx]:>10.2f} {diff[idx]:>10.2f}")
        diff_results.append((g, c1, c2, diff[idx], mean1[idx], mean2[idx]))

# ── 7f. Characterize new states ──
print("\n\n  STATE CHARACTERIZATION:")
for c in range(best_k):
    mask_c = cluster_labels_ns == c
    mean_c = log_cpm_ns[:, mask_c].mean(axis=1)
    mean_other = log_cpm_ns[:, ~mask_c].mean(axis=1)
    diff_c = mean_c - mean_other
    top_up = np.argsort(diff_c)[::-1][:10]
    top_down = np.argsort(diff_c)[:10]

    print(f"\n  Cluster {c} ({mask_c.sum()} samples):")
    print(f"    Top UPREGULATED genes (vs all other clusters):")
    for idx in top_up:
        print(f"      {gene_names[idx]:<15} diff={diff_c[idx]:+.3f}  mean={mean_c[idx]:.2f}")
    print(f"    Top DOWNREGULATED genes (vs all other clusters):")
    for idx in top_down:
        print(f"      {gene_names[idx]:<15} diff={diff_c[idx]:+.3f}  mean={mean_c[idx]:.2f}")

# Check overlap with original bimodal genes
overlap = set(gene_names[bimodal_idx_full]) & set(bimodal_names_ns)
only_new = set(bimodal_names_ns) - set(gene_names[bimodal_idx_full])
lost = set(gene_names[bimodal_idx_full]) - set(bimodal_names_ns)
print(f"\n\n  BIMODAL GENE OVERLAP:")
print(f"    Full dataset bimodal genes:      {len(bimodal_idx_full)}")
print(f"    Non-stressed bimodal genes:      {len(bimodal_idx_ns)}")
print(f"    Overlap (in both):               {len(overlap)}")
print(f"    Lost after removing stressed:    {len(lost)}")
print(f"    New (only in non-stressed):      {len(only_new)}")
if len(lost) > 0:
    print(f"    Lost genes: {sorted(lost)[:30]}...")
if len(only_new) > 0:
    print(f"    New genes:  {sorted(only_new)[:30]}...")

# ══════════════════════════════════════════════════════════════════════
# 8. 4-PANEL PLOT
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Creating 4-panel figure...")

fig, axes = plt.subplots(2, 2, figsize=(16, 14))
fig.suptitle('GTEx Whole Blood: Non-Stressed Sample Re-Analysis',
             fontsize=16, fontweight='bold', color=TEXT, y=0.98)

# Panel 1: Scree plot
ax = axes[0, 0]
pcs = np.arange(1, len(var_explained) + 1)
ax.bar(pcs, var_explained * 100, color='#58a6ff', edgecolor='#58a6ff', alpha=0.8)
ax.plot(pcs, np.cumsum(var_explained) * 100, 'o-', color='#f78166', markersize=5)
ax.set_xlabel('Principal Component')
ax.set_ylabel('Variance Explained (%)')
ax.set_title('PCA Scree Plot (Non-Stressed Binary Matrix)', color=TEXT, fontsize=12)
ax.set_xticks(pcs)
ax.grid(True, alpha=0.3)
# annotate cumulative
for i in range(min(3, len(var_explained))):
    ax.annotate(f'{np.cumsum(var_explained)[i]*100:.1f}%',
                (pcs[i], np.cumsum(var_explained)[i]*100 + 1.5),
                color='#f78166', fontsize=8, ha='center')

# Panel 2: PCA scatter
ax = axes[0, 1]
colors_clusters = ['#58a6ff', '#f78166', '#3fb950', '#d2a8ff', '#f0883e']
for c in range(best_k):
    mask_c = cluster_labels_ns == c
    ax.scatter(scores_ns[mask_c, 0], scores_ns[mask_c, 1],
               c=colors_clusters[c % len(colors_clusters)],
               s=15, alpha=0.6, label=f'Cluster {c} (n={mask_c.sum()})')
ax.set_xlabel(f'PC1 ({var_explained[0]*100:.1f}%)')
ax.set_ylabel(f'PC2 ({var_explained[1]*100:.1f}%)')
ax.set_title('PCA of Non-Stressed Samples', color=TEXT, fontsize=12)
ax.legend(fontsize=9, loc='best')
ax.grid(True, alpha=0.3)

# Panel 3: Binary state heatmap (clustered)
ax = axes[1, 0]
# Subsample if too many genes for clarity
max_genes_plot = 80
if binary_ns.shape[0] > max_genes_plot:
    # Pick genes with highest variance in binary matrix
    bin_var = binary_ns.astype(float).var(axis=1)
    top_var_idx = np.argsort(bin_var)[::-1][:max_genes_plot]
    binary_plot = binary_ns[top_var_idx]
    gene_names_plot = bimodal_names_ns[top_var_idx]
else:
    binary_plot = binary_ns
    gene_names_plot = bimodal_names_ns

# Cluster both axes
try:
    row_link = sch.linkage(binary_plot, method='ward', metric='euclidean')
    row_order = sch.dendrogram(row_link, no_plot=True)['leaves']
except:
    row_order = np.arange(binary_plot.shape[0])

# Order samples by cluster then by PC1
sample_order = np.argsort(cluster_labels_ns * 1000 + scores_ns[:, 0])
binary_ordered = binary_plot[row_order][:, sample_order]

cmap_bin = LinearSegmentedColormap.from_list('binary_dark', [CARD, '#58a6ff'])
im = ax.imshow(binary_ordered, aspect='auto', cmap=cmap_bin, interpolation='nearest')
ax.set_xlabel(f'Samples (n={n_ns})')
ax.set_ylabel(f'Bimodal Genes (top {binary_plot.shape[0]})')
ax.set_title('Binary State Heatmap (Clustered)', color=TEXT, fontsize=12)
ax.set_xticks([])
ax.set_yticks([])

# Add cluster color bar at top
cluster_colors_ordered = [colors_clusters[cluster_labels_ns[i] % len(colors_clusters)]
                          for i in sample_order]
for i, col in enumerate(cluster_colors_ordered):
    ax.plot(i, -0.5, 's', color=col, markersize=1.5, clip_on=False)

# Panel 4: Gene-gene correlation heatmap
ax = axes[1, 1]
# Use top variable bimodal genes
n_corr_genes = min(50, len(bimodal_idx_ns))
bin_var_all = binary_ns.astype(float).var(axis=1)
top_corr_idx = np.argsort(bin_var_all)[::-1][:n_corr_genes]
corr_data = np.corrcoef(log_cpm_ns[bimodal_idx_ns[top_corr_idx]])

# Cluster the correlation matrix
try:
    link_corr = sch.linkage(corr_data, method='ward')
    corr_order = sch.dendrogram(link_corr, no_plot=True)['leaves']
except:
    corr_order = np.arange(n_corr_genes)

corr_ordered = corr_data[corr_order][:, corr_order]
cmap_corr = LinearSegmentedColormap.from_list('corr_dark',
    ['#f78166', CARD, '#58a6ff'])
im2 = ax.imshow(corr_ordered, aspect='auto', cmap=cmap_corr, vmin=-1, vmax=1)
ax.set_title(f'Gene-Gene Correlation (top {n_corr_genes} bimodal)', color=TEXT, fontsize=12)
ax.set_xticks([])
ax.set_yticks([])
plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04, label='Pearson r')

plt.tight_layout(rect=[0, 0, 1, 0.96])
outpath = '/Users/rls/Desktop/programming-projects/single-cell/bulk-project/gtex_nonstressed_states.png'
plt.savefig(outpath, dpi=180, bbox_inches='tight', facecolor=BG)
plt.close()
print(f"  Saved: {outpath}")

# ══════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("DETAILED SUMMARY")
print("=" * 70)
print(f"\n1. FULL DATASET:")
print(f"   - {expr.shape[0]} genes x {expr.shape[1]} samples")
print(f"   - {len(bimodal_idx_full)} bimodal genes identified")
print(f"   - PCA split: State 0 ({n_state0}), State 1 ({n_state1})")
print(f"   - Stressed state = {stressed_state} (higher DDIT4/JUN/VEGFA)")

print(f"\n2. NON-STRESSED SUBSET:")
print(f"   - {n_ns} samples retained (removed {expr.shape[1] - n_ns} stressed)")
print(f"   - {len(bimodal_idx_ns)} bimodal genes detected")
print(f"   - {len(lost)} genes lost bimodality after stress removal")
print(f"   - {len(only_new)} new bimodal genes emerged")

print(f"\n3. PCA OF NON-STRESSED:")
print(f"   - PC1 explains {var_explained[0]*100:.1f}% variance")
print(f"   - PC2 explains {var_explained[1]*100:.1f}% variance")
print(f"   - Best clustering: K={best_k} (silhouette={best_sil:.3f})")
clear_states = best_sil > 0.2 and var_explained[0] > 0.05
print(f"   - Clear states? {'YES' if clear_states else 'Weak/Unclear'}")

print(f"\n4. NEW STATE CLUSTERS:")
for c in range(best_k):
    n_c = (cluster_labels_ns == c).sum()
    print(f"   Cluster {c}: {n_c} samples ({n_c/n_ns*100:.1f}%)")

print("\nDone.")
