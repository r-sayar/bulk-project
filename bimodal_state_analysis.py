#!/usr/bin/env python3
"""Analyze bimodal gene expression states in GTEx whole blood."""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks
from sklearn.decomposition import PCA
import sys
import os

OUTPUT_FILE = '/Users/rls/Desktop/programming-projects/single-cell/bulk-project/bimodal_state_biology.txt'
GCT_FILE = '/Users/rls/Downloads/gene_reads_v11_whole_blood.gct.gz'

out_lines = []
def log(msg=""):
    print(msg)
    out_lines.append(msg)

# 1. Load data
log("=" * 80)
log("BIMODAL GENE STATE ANALYSIS - GTEx Whole Blood")
log("=" * 80)

log("\n[1] Loading GTEx whole blood data...")
df = pd.read_csv(GCT_FILE, sep='\t', skiprows=2, compression='gzip')
log(f"  Raw shape: {df.shape}")
log(f"  Columns preview: {list(df.columns[:5])}")

# Check for metadata columns like ischemic time
all_cols = list(df.columns)
metadata_cols = [c for c in all_cols if not c.startswith('GTEX-') and c not in ['Name', 'Description']]
log(f"\n[10] Metadata column check:")
log(f"  Non-sample columns: {metadata_cols[:20]}")
ischemic_cols = [c for c in all_cols if 'SMTS' in c.upper() or 'ISCH' in c.upper()]
if ischemic_cols:
    log(f"  Ischemic time columns found: {ischemic_cols}")
else:
    log("  No ischemic time metadata (SMTSISCH) found in GCT file.")
    log("  Note: Ischemic time is in the sample annotations file (GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt), not the expression GCT.")

# Extract expression matrix
gene_names = df['Description'].values if 'Description' in df.columns else df['Name'].values
gene_ids = df['Name'].values if 'Name' in df.columns else None
sample_cols = [c for c in df.columns if c.startswith('GTEX-')]
expr_raw = df[sample_cols].values.astype(np.float64)
log(f"  Expression matrix: {expr_raw.shape[0]} genes x {expr_raw.shape[1]} samples")

# 2. Compute log2(CPM+1)
log("\n[2] Computing log2(CPM+1)...")
lib_sizes = expr_raw.sum(axis=0)
cpm = (expr_raw / lib_sizes[np.newaxis, :]) * 1e6
expr = np.log2(cpm + 1)
log(f"  Library sizes range: {lib_sizes.min():.0f} - {lib_sizes.max():.0f}")
log(f"  Expression range after transform: {expr.min():.2f} - {expr.max():.2f}")

# 3. Find bimodal genes using KDE + peak detection
log("\n[3] Finding bimodal genes (KDE + peak detection)...")
gene_means = expr.mean(axis=1)
gene_stds = expr.std(axis=1)

# Filter: mean > 1, std > 0.3
candidates = np.where((gene_means > 1) & (gene_stds > 0.3))[0]
log(f"  Candidate genes (mean>1, std>0.3): {len(candidates)}")

bimodal_genes = []
x_eval = np.linspace(0, 20, 1000)

for idx in candidates:
    vals = expr[idx, :]
    try:
        kde = stats.gaussian_kde(vals, bw_method='scott')
        density = kde(x_eval)
        prominence_thresh = density.max() * 0.08
        peaks, properties = find_peaks(density, prominence=prominence_thresh, distance=40)
        if len(peaks) >= 2:
            bimodal_genes.append(idx)
    except Exception:
        continue

log(f"  Bimodal genes found: {len(bimodal_genes)}")

# 4. Build binary matrix, PCA, split by PC1
log("\n[4] Building binary matrix and PCA...")
bimodal_expr = expr[bimodal_genes, :]

# Binarize: for each gene, above median = 1, below = 0
medians = np.median(bimodal_expr, axis=1, keepdims=True)
binary_matrix = (bimodal_expr > medians).astype(int)

# PCA on samples (transpose so samples are rows)
pca = PCA(n_components=2)
pca_result = pca.fit_transform(binary_matrix.T)

pc1 = pca_result[:, 0]
state_a_mask = pc1 > 0
state_b_mask = pc1 <= 0
n_a = state_a_mask.sum()
n_b = state_b_mask.sum()
log(f"  PCA variance explained: PC1={pca.explained_variance_ratio_[0]:.3f}, PC2={pca.explained_variance_ratio_[1]:.3f}")
log(f"  State A (PC1 > 0): {n_a} samples")
log(f"  State B (PC1 <= 0): {n_b} samples")

# 5-6. For ALL genes, compute mean expression, fold change, and t-test
log("\n[5-6] Computing differential expression for ALL genes...")
mean_a = expr[:, state_a_mask].mean(axis=1)
mean_b = expr[:, state_b_mask].mean(axis=1)

log2fc = mean_a - mean_b  # already in log2 space

pvalues = np.ones(expr.shape[0])
for i in range(expr.shape[0]):
    vals_a = expr[i, state_a_mask]
    vals_b = expr[i, state_b_mask]
    if vals_a.std() > 0 or vals_b.std() > 0:
        t, p = stats.ttest_ind(vals_a, vals_b)
        pvalues[i] = p

# Build results dataframe
results = pd.DataFrame({
    'gene': gene_names,
    'mean_A': mean_a,
    'mean_B': mean_b,
    'log2FC': log2fc,
    'pvalue': pvalues,
})
results['significant'] = results['pvalue'] < 0.001

n_sig = results['significant'].sum()
log(f"  Total genes tested: {len(results)}")
log(f"  Significant (p < 0.001): {n_sig}")

# 7. Top 50 upregulated in State A
log("\n" + "=" * 80)
log("[7] TOP 50 GENES UPREGULATED IN STATE A (higher in State A)")
log("=" * 80)
up_a = results[results['significant']].sort_values('log2FC', ascending=False).head(50)
log(f"{'Gene':<15} {'Mean_A':>8} {'Mean_B':>8} {'log2FC':>8} {'p-value':>12}")
log("-" * 55)
for _, row in up_a.iterrows():
    log(f"{row['gene']:<15} {row['mean_A']:>8.3f} {row['mean_B']:>8.3f} {row['log2FC']:>8.3f} {row['pvalue']:>12.2e}")

# 8. Top 50 upregulated in State B
log("\n" + "=" * 80)
log("[8] TOP 50 GENES UPREGULATED IN STATE B (higher in State B)")
log("=" * 80)
up_b = results[results['significant']].sort_values('log2FC', ascending=True).head(50)
log(f"{'Gene':<15} {'Mean_A':>8} {'Mean_B':>8} {'log2FC':>8} {'p-value':>12}")
log("-" * 55)
for _, row in up_b.iterrows():
    log(f"{row['gene']:<15} {row['mean_A']:>8.3f} {row['mean_B']:>8.3f} {row['log2FC']:>8.3f} {row['pvalue']:>12.2e}")

# 9. Check specific gene categories
log("\n" + "=" * 80)
log("[9] SPECIFIC GENE CATEGORY ANALYSIS")
log("=" * 80)

categories = {
    'Stress Response': ['DDIT4', 'DDIT3', 'VEGFA', 'JUN', 'ATF3', 'HSPA1B', 'HSPA1A', 'FOS', 'FOSB', 'EGR1', 'DUSP1', 'PPP1R15A', 'GADD45B', 'GADD45G'],
    'Hypoxia': ['HIF1A', 'BNIP3', 'SLC2A1', 'LDHA', 'PGK1', 'NDRG1'],
    'Inflammation': ['IL1B', 'CXCL8', 'CCL3', 'CCL4', 'TNF', 'IL6', 'CXCR4'],
    'Apoptosis': ['BCL2', 'BAX', 'CASP3', 'CASP8'],
    'Housekeeping': ['ACTB', 'GAPDH', 'B2M', 'RPL13A'],
}

for cat_name, genes in categories.items():
    log(f"\n--- {cat_name} ---")
    log(f"{'Gene':<15} {'Mean_A':>8} {'Mean_B':>8} {'log2FC':>8} {'p-value':>12} {'Direction':>12}")
    log("-" * 70)
    for g in genes:
        mask = results['gene'] == g
        if mask.any():
            row = results[mask].iloc[0]
            direction = "State A up" if row['log2FC'] > 0 else "State B up"
            sig = "*" if row['pvalue'] < 0.001 else ""
            log(f"{g:<15} {row['mean_A']:>8.3f} {row['mean_B']:>8.3f} {row['log2FC']:>8.3f} {row['pvalue']:>12.2e} {direction:>12} {sig}")
        else:
            log(f"{g:<15} {'NOT FOUND':>50}")

# Summary
log("\n" + "=" * 80)
log("SUMMARY")
log("=" * 80)
stress_genes = ['DDIT4', 'DDIT3', 'VEGFA', 'JUN', 'ATF3', 'HSPA1B', 'HSPA1A', 'FOS', 'FOSB', 'EGR1', 'DUSP1', 'PPP1R15A', 'GADD45B', 'GADD45G']
stress_fcs = []
for g in stress_genes:
    mask = results['gene'] == g
    if mask.any():
        stress_fcs.append(results[mask].iloc[0]['log2FC'])
if stress_fcs:
    avg_stress_fc = np.mean(stress_fcs)
    log(f"  Average stress gene log2FC (A vs B): {avg_stress_fc:.3f}")
    if avg_stress_fc > 0:
        log("  -> Stress genes tend to be HIGHER in State A")
    else:
        log("  -> Stress genes tend to be HIGHER in State B")

hk_genes = ['ACTB', 'GAPDH', 'B2M', 'RPL13A']
hk_fcs = []
for g in hk_genes:
    mask = results['gene'] == g
    if mask.any():
        hk_fcs.append(results[mask].iloc[0]['log2FC'])
if hk_fcs:
    avg_hk_fc = np.mean(hk_fcs)
    log(f"  Average housekeeping gene log2FC (A vs B): {avg_hk_fc:.3f}")

log(f"\n  Total bimodal genes: {len(bimodal_genes)}")
log(f"  State A samples: {n_a}, State B samples: {n_b}")

# Save results
with open(OUTPUT_FILE, 'w') as f:
    f.write('\n'.join(out_lines))
log(f"\nResults saved to {OUTPUT_FILE}")
