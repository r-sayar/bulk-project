#!/usr/bin/env python3
"""
Basic Gene Expression Analysis for GSE50244 dataset
Comparing two samples: s1 and s2
"""

import pandas as pd
import numpy as np
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load the data
print("=" * 60)
print("GENE EXPRESSION ANALYSIS - GSE50244")
print("=" * 60)

df = pd.read_csv(
    os.path.join(PROJECT_ROOT, "data/GSE50244_Genes_counts_TMM_NormLength_atLeastMAF5_expressed_coord_noChrPrefix.txt"),
    sep="\t"
)

print(f"\n1. DATA OVERVIEW")
print("-" * 40)
print(f"Total genes: {len(df):,}")
print(f"Columns: {list(df.columns)}")
print(f"\nFirst few rows:")
print(df.head(10).to_string())

# Basic statistics
print(f"\n2. SUMMARY STATISTICS")
print("-" * 40)
print(f"\nSample s1:")
print(f"  Min:    {df['s1'].min():,.2f}")
print(f"  Max:    {df['s1'].max():,.2f}")
print(f"  Mean:   {df['s1'].mean():,.2f}")
print(f"  Median: {df['s1'].median():,.2f}")
print(f"  Std:    {df['s1'].std():,.2f}")

print(f"\nSample s2:")
print(f"  Min:    {df['s2'].min():,.2f}")
print(f"  Max:    {df['s2'].max():,.2f}")
print(f"  Mean:   {df['s2'].mean():,.2f}")
print(f"  Median: {df['s2'].median():,.2f}")
print(f"  Std:    {df['s2'].std():,.2f}")

# Correlation analysis (manual calculation to avoid scipy dependency)
print(f"\n3. CORRELATION ANALYSIS")
print("-" * 40)
# Pearson correlation
s1_mean = df['s1'].mean()
s2_mean = df['s2'].mean()
numerator = ((df['s1'] - s1_mean) * (df['s2'] - s2_mean)).sum()
denominator = np.sqrt(((df['s1'] - s1_mean)**2).sum() * ((df['s2'] - s2_mean)**2).sum())
pearson_r = numerator / denominator
print(f"Pearson correlation:  r = {pearson_r:.4f}")

# Spearman correlation (rank-based)
s1_ranks = df['s1'].rank()
s2_ranks = df['s2'].rank()
s1_rank_mean = s1_ranks.mean()
s2_rank_mean = s2_ranks.mean()
numerator_sp = ((s1_ranks - s1_rank_mean) * (s2_ranks - s2_rank_mean)).sum()
denominator_sp = np.sqrt(((s1_ranks - s1_rank_mean)**2).sum() * ((s2_ranks - s2_rank_mean)**2).sum())
spearman_r = numerator_sp / denominator_sp
print(f"Spearman correlation: r = {spearman_r:.4f}")

# Log2 fold change analysis
print(f"\n4. DIFFERENTIAL EXPRESSION (Log2 Fold Change)")
print("-" * 40)
# Add small pseudocount to avoid log(0)
pseudocount = 1
df['log2_s1'] = np.log2(df['s1'] + pseudocount)
df['log2_s2'] = np.log2(df['s2'] + pseudocount)
df['log2FC'] = df['log2_s2'] - df['log2_s1']  # positive = higher in s2

print(f"Log2 Fold Change (s2 vs s1):")
print(f"  Mean log2FC:   {df['log2FC'].mean():.4f}")
print(f"  Median log2FC: {df['log2FC'].median():.4f}")
print(f"  Std log2FC:    {df['log2FC'].std():.4f}")

# Identify differentially expressed genes (arbitrary threshold: |log2FC| > 1)
fc_threshold = 1
upregulated = df[df['log2FC'] > fc_threshold]
downregulated = df[df['log2FC'] < -fc_threshold]
unchanged = df[(df['log2FC'] >= -fc_threshold) & (df['log2FC'] <= fc_threshold)]

print(f"\nUsing |log2FC| > {fc_threshold} threshold:")
print(f"  Upregulated in s2:   {len(upregulated):,} genes ({100*len(upregulated)/len(df):.1f}%)")
print(f"  Downregulated in s2: {len(downregulated):,} genes ({100*len(downregulated)/len(df):.1f}%)")
print(f"  Unchanged:           {len(unchanged):,} genes ({100*len(unchanged)/len(df):.1f}%)")

# Top differentially expressed genes
print(f"\n5. TOP DIFFERENTIALLY EXPRESSED GENES")
print("-" * 40)
df_sorted = df.sort_values('log2FC', key=abs, ascending=False)

print(f"\nTop 10 upregulated in s2:")
top_up = df[df['log2FC'] > 0].nlargest(10, 'log2FC')
for _, row in top_up.iterrows():
    print(f"  {row['geneid']:15s} log2FC = {row['log2FC']:+.3f}")

print(f"\nTop 10 downregulated in s2:")
top_down = df[df['log2FC'] < 0].nsmallest(10, 'log2FC')
for _, row in top_down.iterrows():
    print(f"  {row['geneid']:15s} log2FC = {row['log2FC']:+.3f}")

# Chromosome distribution
print(f"\n6. CHROMOSOME DISTRIBUTION")
print("-" * 40)
chr_counts = df['chrm_probe'].value_counts().sort_index()
print("Genes per chromosome:")
for chrom, count in chr_counts.items():
    print(f"  Chr {str(chrom):2s}: {count:4d} genes")

# Expression level categories
print(f"\n7. EXPRESSION LEVEL DISTRIBUTION")
print("-" * 40)
# Define expression categories based on mean expression
df['mean_expr'] = (df['s1'] + df['s2']) / 2

# Quartile-based categorization
q25, q50, q75 = df['mean_expr'].quantile([0.25, 0.5, 0.75])
low_expr = len(df[df['mean_expr'] <= q25])
med_low_expr = len(df[(df['mean_expr'] > q25) & (df['mean_expr'] <= q50)])
med_high_expr = len(df[(df['mean_expr'] > q50) & (df['mean_expr'] <= q75)])
high_expr = len(df[df['mean_expr'] > q75])

print(f"Quartile thresholds: Q1={q25:.0f}, Q2={q50:.0f}, Q3={q75:.0f}")
print(f"  Low expression (≤Q1):     {low_expr:,} genes")
print(f"  Med-low (Q1-Q2):          {med_low_expr:,} genes")
print(f"  Med-high (Q2-Q3):         {med_high_expr:,} genes")
print(f"  High expression (>Q3):    {high_expr:,} genes")

# Save results to file
print(f"\n8. SAVING RESULTS")
print("-" * 40)
df_results = df[['geneid', 'chrm_probe', 's1', 's2', 'log2FC', 'mean_expr']].copy()
df_results = df_results.sort_values('log2FC', key=abs, ascending=False)
df_results.to_csv(os.path.join(PROJECT_ROOT, 'results/gene_expression_results.csv'), index=False)
print(f"Saved differential expression results to: results/gene_expression_results.csv")

print(f"\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
