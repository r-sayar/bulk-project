#!/usr/bin/env python3
"""
Basic Gene Expression Analysis for a single cell
from GSM2230757_human1_umifm_counts.csv.gz
"""

import pandas as pd
import numpy as np
import gzip
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=" * 60)
print("SINGLE CELL GENE EXPRESSION ANALYSIS")
print("=" * 60)

# Load the data
print("\nLoading data...")
df = pd.read_csv(os.path.join(PROJECT_ROOT, 'data/GSE84133_RAW/GSM2230757_human1_umifm_counts.csv.gz'), compression='gzip')

print(f"\n1. DATASET OVERVIEW")
print("-" * 40)
print(f"Total cells: {len(df):,}")
print(f"Total genes: {len(df.columns) - 3:,}")  # Exclude first 3 columns (index, barcode, cluster)
print(f"Columns: {df.columns[:5].tolist()} ... (and {len(df.columns)-5} more)")

# Get gene columns (all except first 3)
gene_cols = df.columns[3:].tolist()

# Analyze first cell
cell_idx = 0
cell_data = df.iloc[cell_idx]

print(f"\n2. SINGLE CELL INFO (Cell #{cell_idx + 1})")
print("-" * 40)
print(f"Cell ID:    {cell_data.iloc[0]}")
print(f"Barcode:    {cell_data['barcode']}")
print(f"Cluster:    {cell_data['assigned_cluster']}")

# Get expression values for this cell
expr_values = cell_data[gene_cols].astype(float)

print(f"\n3. EXPRESSION SUMMARY STATISTICS")
print("-" * 40)
total_umi = expr_values.sum()
genes_detected = (expr_values > 0).sum()
print(f"Total UMI counts:     {total_umi:,.0f}")
print(f"Genes detected:       {genes_detected:,} / {len(gene_cols):,} ({100*genes_detected/len(gene_cols):.1f}%)")
print(f"Genes not expressed:  {len(gene_cols) - genes_detected:,}")

print(f"\nExpression values (all genes):")
print(f"  Min:    {expr_values.min():.0f}")
print(f"  Max:    {expr_values.max():.0f}")
print(f"  Mean:   {expr_values.mean():.2f}")
print(f"  Median: {expr_values.median():.0f}")
print(f"  Std:    {expr_values.std():.2f}")

print(f"\nExpression values (detected genes only):")
detected_expr = expr_values[expr_values > 0]
print(f"  Min:    {detected_expr.min():.0f}")
print(f"  Max:    {detected_expr.max():.0f}")
print(f"  Mean:   {detected_expr.mean():.2f}")
print(f"  Median: {detected_expr.median():.0f}")
print(f"  Std:    {detected_expr.std():.2f}")

print(f"\n4. EXPRESSION DISTRIBUTION")
print("-" * 40)
# Count genes at different expression levels
expr_0 = (expr_values == 0).sum()
expr_1 = (expr_values == 1).sum()
expr_2_5 = ((expr_values >= 2) & (expr_values <= 5)).sum()
expr_6_10 = ((expr_values >= 6) & (expr_values <= 10)).sum()
expr_11_50 = ((expr_values >= 11) & (expr_values <= 50)).sum()
expr_51_100 = ((expr_values >= 51) & (expr_values <= 100)).sum()
expr_gt100 = (expr_values > 100).sum()

print(f"UMI count distribution:")
print(f"  0 counts:      {expr_0:5,} genes ({100*expr_0/len(gene_cols):.1f}%)")
print(f"  1 count:       {expr_1:5,} genes ({100*expr_1/len(gene_cols):.1f}%)")
print(f"  2-5 counts:    {expr_2_5:5,} genes ({100*expr_2_5/len(gene_cols):.1f}%)")
print(f"  6-10 counts:   {expr_6_10:5,} genes ({100*expr_6_10/len(gene_cols):.1f}%)")
print(f"  11-50 counts:  {expr_11_50:5,} genes ({100*expr_11_50/len(gene_cols):.1f}%)")
print(f"  51-100 counts: {expr_51_100:5,} genes ({100*expr_51_100/len(gene_cols):.1f}%)")
print(f"  >100 counts:   {expr_gt100:5,} genes ({100*expr_gt100/len(gene_cols):.1f}%)")

print(f"\n5. TOP EXPRESSED GENES")
print("-" * 40)
top_genes = expr_values.nlargest(20)
print(f"Top 20 most highly expressed genes:")
for gene, count in top_genes.items():
    pct_of_total = 100 * count / total_umi
    print(f"  {gene:15s} {count:6.0f} UMIs ({pct_of_total:5.2f}% of total)")

print(f"\n6. GENE CATEGORIES")
print("-" * 40)
# Look for common gene categories
mito_genes = [g for g in gene_cols if g.startswith('MT-') or g.startswith('MT.')]
ribo_genes = [g for g in gene_cols if g.startswith('RPS') or g.startswith('RPL')]

mito_expr = expr_values[mito_genes].sum() if mito_genes else 0
ribo_expr = expr_values[ribo_genes].sum() if ribo_genes else 0

print(f"Mitochondrial genes:")
print(f"  Genes found:   {len(mito_genes)}")
print(f"  Total UMIs:    {mito_expr:.0f} ({100*mito_expr/total_umi:.1f}% of total)")

print(f"\nRibosomal protein genes:")
print(f"  Genes found:   {len(ribo_genes)}")
print(f"  Total UMIs:    {ribo_expr:.0f} ({100*ribo_expr/total_umi:.1f}% of total)")

# Housekeeping genes check
housekeeping = ['ACTB', 'GAPDH', 'B2M', 'HPRT1', 'PPIA', 'RPLP0', 'UBC', 'YWHAZ']
print(f"\nHousekeeping gene expression:")
for hk in housekeeping:
    if hk in gene_cols:
        val = expr_values[hk]
        print(f"  {hk:8s}: {val:.0f} UMIs")

print(f"\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
