#!/usr/bin/env python3
"""
Single-cell RNA-seq preprocessing, QC, and visualization pipeline
Dataset: GSE84133 (Pancreatic islet single-cell data)
"""

import os
import glob
import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# Set scanpy settings
sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=100, facecolor='white', frameon=False)

# Create output directories
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.makedirs(os.path.join(PROJECT_ROOT, 'figures'), exist_ok=True)
os.makedirs(os.path.join(PROJECT_ROOT, 'processed_data'), exist_ok=True)


def load_data(data_dir=None, species='human'):
    if data_dir is None:
        data_dir = os.path.join(PROJECT_ROOT, 'data', 'GSE84133_RAW')
    """
    Load and combine count matrices for specified species.
    
    Parameters:
    -----------
    data_dir : str
        Directory containing the raw data files
    species : str
        Either 'human' or 'mouse'
    
    Returns:
    --------
    AnnData object with combined data
    """
    print(f"\n{'='*60}")
    print(f"Loading {species} data...")
    print('='*60)
    
    # Find all files for the specified species
    pattern = os.path.join(data_dir, f'*{species}*_umifm_counts.csv.gz')
    files = sorted(glob.glob(pattern))
    
    if not files:
        raise FileNotFoundError(f"No files found for {species} in {data_dir}")
    
    print(f"Found {len(files)} files:")
    for f in files:
        print(f"  - {os.path.basename(f)}")
    
    # Load and combine data
    adatas = []
    for i, filepath in enumerate(files, 1):
        print(f"\nLoading file {i}/{len(files)}: {os.path.basename(filepath)}")
        
        # Read the CSV file
        df = pd.read_csv(filepath, index_col=0)
        
        # The data format: rows are cells, columns are genes
        # First few columns might be metadata (barcode, assigned_cluster, etc.)
        
        # Identify metadata columns (non-gene columns)
        # Gene columns typically start with uppercase letter
        meta_cols = []
        gene_cols = []
        
        for col in df.columns:
            # Check if column looks like a gene name
            if col[0].isupper() and not col.startswith('assigned'):
                gene_cols.append(col)
            else:
                meta_cols.append(col)
        
        print(f"  Cells: {df.shape[0]}, Genes: {len(gene_cols)}, Metadata cols: {len(meta_cols)}")
        
        # Extract count matrix and metadata
        if gene_cols:
            counts = df[gene_cols]
            metadata = df[meta_cols] if meta_cols else pd.DataFrame(index=df.index)
        else:
            # If no gene columns identified, assume all numeric columns are genes
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            counts = df[numeric_cols]
            metadata = df.drop(columns=numeric_cols)
        
        # Create AnnData object
        adata = sc.AnnData(X=counts.values.astype(np.float32))
        adata.obs_names = [f"{species}{i}_cell{j}" for j in range(counts.shape[0])]
        adata.var_names = counts.columns.tolist()
        
        # Add metadata
        for col in metadata.columns:
            adata.obs[col] = metadata[col].values
        
        # Add sample information
        sample_name = os.path.basename(filepath).replace('_umifm_counts.csv.gz', '')
        adata.obs['sample'] = sample_name
        adata.obs['donor'] = f"{species}{i}"
        
        adatas.append(adata)
    
    # Combine all samples
    print(f"\nCombining {len(adatas)} samples...")
    adata = sc.concat(adatas, join='outer')
    adata.obs_names_make_unique()
    
    # Fill NaN values with 0 (genes not detected in some samples)
    if np.isnan(adata.X).any():
        adata.X = np.nan_to_num(adata.X, nan=0.0)
    
    print(f"\nCombined data shape: {adata.shape[0]} cells x {adata.shape[1]} genes")
    
    return adata


def calculate_qc_metrics(adata):
    """Calculate QC metrics for the dataset."""
    print(f"\n{'='*60}")
    print("Calculating QC metrics...")
    print('='*60)
    
    # Identify mitochondrial genes
    adata.var['mt'] = adata.var_names.str.startswith(('MT-', 'mt-'))
    
    # Identify ribosomal genes
    adata.var['ribo'] = adata.var_names.str.startswith(('RPS', 'RPL', 'Rps', 'Rpl'))
    
    # Calculate QC metrics
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt', 'ribo'], 
                                percent_top=None, log1p=False, inplace=True)
    
    # Print summary statistics
    print("\nQC Metrics Summary:")
    print("-" * 40)
    print(f"Total counts per cell:")
    print(f"  Mean: {adata.obs['total_counts'].mean():.2f}")
    print(f"  Median: {adata.obs['total_counts'].median():.2f}")
    print(f"  Min: {adata.obs['total_counts'].min():.2f}")
    print(f"  Max: {adata.obs['total_counts'].max():.2f}")
    
    print(f"\nGenes detected per cell:")
    print(f"  Mean: {adata.obs['n_genes_by_counts'].mean():.2f}")
    print(f"  Median: {adata.obs['n_genes_by_counts'].median():.2f}")
    print(f"  Min: {adata.obs['n_genes_by_counts'].min():.2f}")
    print(f"  Max: {adata.obs['n_genes_by_counts'].max():.2f}")
    
    print(f"\nMitochondrial percentage:")
    print(f"  Mean: {adata.obs['pct_counts_mt'].mean():.2f}%")
    print(f"  Median: {adata.obs['pct_counts_mt'].median():.2f}%")
    
    return adata


def plot_qc_metrics(adata, species, save=True):
    """Generate QC visualization plots."""
    print(f"\n{'='*60}")
    print("Generating QC plots...")
    print('='*60)
    
    # Create figure with QC violin plots
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Total counts
    sc.pl.violin(adata, 'total_counts', ax=axes[0], show=False)
    axes[0].set_title('Total Counts per Cell')
    axes[0].set_ylabel('Counts')
    
    # Genes per cell
    sc.pl.violin(adata, 'n_genes_by_counts', ax=axes[1], show=False)
    axes[1].set_title('Genes per Cell')
    axes[1].set_ylabel('Number of Genes')
    
    # MT percentage
    sc.pl.violin(adata, 'pct_counts_mt', ax=axes[2], show=False)
    axes[2].set_title('Mitochondrial %')
    axes[2].set_ylabel('Percentage')
    
    # Ribosomal percentage
    sc.pl.violin(adata, 'pct_counts_ribo', ax=axes[3], show=False)
    axes[3].set_title('Ribosomal %')
    axes[3].set_ylabel('Percentage')
    
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(PROJECT_ROOT, f'figures/{species}_qc_violin.png'), dpi=150, bbox_inches='tight')
        print(f"  Saved: figures/{species}_qc_violin.png")
    plt.close()
    
    # Scatter plots for QC
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    sc.pl.scatter(adata, x='total_counts', y='n_genes_by_counts', 
                  color='pct_counts_mt', ax=axes[0], show=False)
    axes[0].set_title('Counts vs Genes (colored by MT%)')
    
    sc.pl.scatter(adata, x='total_counts', y='pct_counts_mt', ax=axes[1], show=False)
    axes[1].set_title('Counts vs MT%')
    
    sc.pl.scatter(adata, x='n_genes_by_counts', y='pct_counts_mt', ax=axes[2], show=False)
    axes[2].set_title('Genes vs MT%')
    
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(PROJECT_ROOT, f'figures/{species}_qc_scatter.png'), dpi=150, bbox_inches='tight')
        print(f"  Saved: figures/{species}_qc_scatter.png")
    plt.close()
    
    # Per-sample QC
    if 'sample' in adata.obs.columns:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        sc.pl.violin(adata, 'total_counts', groupby='sample', ax=axes[0], 
                     rotation=45, show=False)
        axes[0].set_title('Total Counts by Sample')
        
        sc.pl.violin(adata, 'n_genes_by_counts', groupby='sample', ax=axes[1], 
                     rotation=45, show=False)
        axes[1].set_title('Genes per Cell by Sample')
        
        sc.pl.violin(adata, 'pct_counts_mt', groupby='sample', ax=axes[2], 
                     rotation=45, show=False)
        axes[2].set_title('MT% by Sample')
        
        plt.tight_layout()
        if save:
            plt.savefig(os.path.join(PROJECT_ROOT, f'figures/{species}_qc_by_sample.png'), dpi=150, bbox_inches='tight')
            print(f"  Saved: figures/{species}_qc_by_sample.png")
        plt.close()


def filter_cells_genes(adata, 
                       min_genes=200, 
                       max_genes=5000,
                       min_cells=3,
                       max_mt_pct=20):
    """
    Filter cells and genes based on QC metrics.
    
    Parameters:
    -----------
    adata : AnnData
        Input data
    min_genes : int
        Minimum genes per cell
    max_genes : int
        Maximum genes per cell (filter doublets)
    min_cells : int
        Minimum cells per gene
    max_mt_pct : float
        Maximum mitochondrial percentage
    
    Returns:
    --------
    Filtered AnnData object
    """
    print(f"\n{'='*60}")
    print("Filtering cells and genes...")
    print('='*60)
    
    n_cells_before = adata.n_obs
    n_genes_before = adata.n_vars
    
    print(f"\nBefore filtering: {n_cells_before} cells, {n_genes_before} genes")
    print(f"\nFiltering criteria:")
    print(f"  - Min genes per cell: {min_genes}")
    print(f"  - Max genes per cell: {max_genes}")
    print(f"  - Min cells per gene: {min_cells}")
    print(f"  - Max MT%: {max_mt_pct}%")
    
    # Filter cells
    sc.pp.filter_cells(adata, min_genes=min_genes)
    adata = adata[adata.obs['n_genes_by_counts'] < max_genes, :].copy()
    adata = adata[adata.obs['pct_counts_mt'] < max_mt_pct, :].copy()
    
    # Filter genes
    sc.pp.filter_genes(adata, min_cells=min_cells)
    
    n_cells_after = adata.n_obs
    n_genes_after = adata.n_vars
    
    print(f"\nAfter filtering: {n_cells_after} cells, {n_genes_after} genes")
    print(f"  Removed {n_cells_before - n_cells_after} cells ({100*(n_cells_before - n_cells_after)/n_cells_before:.1f}%)")
    print(f"  Removed {n_genes_before - n_genes_after} genes ({100*(n_genes_before - n_genes_after)/n_genes_before:.1f}%)")
    
    return adata


def preprocess(adata):
    """
    Standard preprocessing pipeline:
    - Normalization
    - Log transformation
    - Highly variable gene selection
    - Scaling
    """
    print(f"\n{'='*60}")
    print("Preprocessing...")
    print('='*60)
    
    # Store raw counts
    adata.raw = adata.copy()
    
    # Normalize to 10,000 counts per cell
    print("\n1. Normalizing to 10,000 counts per cell...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    
    # Log transform
    print("2. Log-transforming...")
    sc.pp.log1p(adata)
    
    # Find highly variable genes
    print("3. Finding highly variable genes...")
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    n_hvg = adata.var['highly_variable'].sum()
    print(f"   Found {n_hvg} highly variable genes")
    
    # Keep all genes but mark HVGs
    print("4. Scaling data...")
    sc.pp.scale(adata, max_value=10)
    
    return adata


def run_dimensionality_reduction(adata):
    """Run PCA and UMAP."""
    print(f"\n{'='*60}")
    print("Dimensionality reduction...")
    print('='*60)
    
    # PCA on highly variable genes
    print("\n1. Running PCA...")
    sc.tl.pca(adata, n_comps=50, use_highly_variable=True)
    
    # Compute neighborhood graph
    print("2. Computing neighborhood graph...")
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30)
    
    # UMAP
    print("3. Running UMAP...")
    sc.tl.umap(adata)
    
    return adata


def run_clustering(adata, resolution=0.5):
    """Run Leiden clustering."""
    print(f"\n{'='*60}")
    print("Clustering...")
    print('='*60)
    
    print(f"\nRunning Leiden clustering (resolution={resolution})...")
    sc.tl.leiden(adata, resolution=resolution)
    
    n_clusters = adata.obs['leiden'].nunique()
    print(f"Found {n_clusters} clusters")
    
    # Print cluster sizes
    print("\nCluster sizes:")
    cluster_counts = adata.obs['leiden'].value_counts().sort_index()
    for cluster, count in cluster_counts.items():
        print(f"  Cluster {cluster}: {count} cells ({100*count/adata.n_obs:.1f}%)")
    
    return adata


def plot_results(adata, species, save=True):
    """Generate visualization plots."""
    print(f"\n{'='*60}")
    print("Generating visualization plots...")
    print('='*60)
    
    # PCA variance plot
    sc.pl.pca_variance_ratio(adata, n_pcs=30, show=False)
    if save:
        plt.savefig(os.path.join(PROJECT_ROOT, f'figures/{species}_pca_variance.png'), dpi=150, bbox_inches='tight')
        print(f"  Saved: figures/{species}_pca_variance.png")
    plt.close()
    
    # UMAP by cluster
    fig, ax = plt.subplots(figsize=(10, 8))
    sc.pl.umap(adata, color='leiden', ax=ax, show=False, 
               legend_loc='on data', legend_fontsize=10,
               title='UMAP - Leiden Clusters')
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(PROJECT_ROOT, f'figures/{species}_umap_clusters.png'), dpi=150, bbox_inches='tight')
        print(f"  Saved: figures/{species}_umap_clusters.png")
    plt.close()
    
    # UMAP by sample
    if 'sample' in adata.obs.columns:
        fig, ax = plt.subplots(figsize=(10, 8))
        sc.pl.umap(adata, color='sample', ax=ax, show=False,
                   title='UMAP - Samples')
        plt.tight_layout()
        if save:
            plt.savefig(os.path.join(PROJECT_ROOT, f'figures/{species}_umap_samples.png'), dpi=150, bbox_inches='tight')
            print(f"  Saved: figures/{species}_umap_samples.png")
        plt.close()
    
    # UMAP by original cluster annotation (if available)
    if 'assigned_cluster' in adata.obs.columns:
        fig, ax = plt.subplots(figsize=(10, 8))
        sc.pl.umap(adata, color='assigned_cluster', ax=ax, show=False,
                   title='UMAP - Original Cell Type Annotations')
        plt.tight_layout()
        if save:
            plt.savefig(os.path.join(PROJECT_ROOT, f'figures/{species}_umap_celltypes.png'), dpi=150, bbox_inches='tight')
            print(f"  Saved: figures/{species}_umap_celltypes.png")
        plt.close()
    
    # Highly variable genes plot
    sc.pl.highly_variable_genes(adata, show=False)
    if save:
        plt.savefig(os.path.join(PROJECT_ROOT, f'figures/{species}_hvg.png'), dpi=150, bbox_inches='tight')
        print(f"  Saved: figures/{species}_hvg.png")
    plt.close()
    
    # Combined QC and clustering summary
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1: UMAP plots
    sc.pl.umap(adata, color='leiden', ax=axes[0, 0], show=False, 
               legend_loc='right margin', title='Clusters')
    sc.pl.umap(adata, color='total_counts', ax=axes[0, 1], show=False,
               title='Total Counts')
    sc.pl.umap(adata, color='n_genes_by_counts', ax=axes[0, 2], show=False,
               title='Genes Detected')
    
    # Row 2: More UMAP plots
    sc.pl.umap(adata, color='pct_counts_mt', ax=axes[1, 0], show=False,
               title='MT%')
    
    if 'sample' in adata.obs.columns:
        sc.pl.umap(adata, color='sample', ax=axes[1, 1], show=False,
                   title='Sample')
    else:
        axes[1, 1].axis('off')
    
    if 'assigned_cluster' in adata.obs.columns:
        sc.pl.umap(adata, color='assigned_cluster', ax=axes[1, 2], show=False,
                   title='Cell Types')
    else:
        axes[1, 2].axis('off')
    
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(PROJECT_ROOT, f'figures/{species}_summary.png'), dpi=150, bbox_inches='tight')
        print(f"  Saved: figures/{species}_summary.png")
    plt.close()


def find_marker_genes(adata, species, save=True):
    """Find marker genes for each cluster."""
    print(f"\n{'='*60}")
    print("Finding marker genes...")
    print('='*60)
    
    # Find marker genes
    sc.tl.rank_genes_groups(adata, 'leiden', method='wilcoxon')
    
    # Plot top markers
    fig, ax = plt.subplots(figsize=(12, 8))
    sc.pl.rank_genes_groups(adata, n_genes=10, ax=ax, show=False)
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(PROJECT_ROOT, f'figures/{species}_markers.png'), dpi=150, bbox_inches='tight')
        print(f"  Saved: figures/{species}_markers.png")
    plt.close()
    
    # Dotplot of top markers
    top_markers = []
    for cluster in adata.obs['leiden'].unique():
        genes = sc.get.rank_genes_groups_df(adata, group=cluster).head(3)['names'].tolist()
        top_markers.extend(genes)
    top_markers = list(dict.fromkeys(top_markers))  # Remove duplicates preserving order
    
    if len(top_markers) > 0:
        fig, ax = plt.subplots(figsize=(14, 8))
        sc.pl.dotplot(adata, var_names=top_markers[:30], groupby='leiden', 
                      ax=ax, show=False)
        plt.tight_layout()
        if save:
            plt.savefig(os.path.join(PROJECT_ROOT, f'figures/{species}_marker_dotplot.png'), dpi=150, bbox_inches='tight')
            print(f"  Saved: figures/{species}_marker_dotplot.png")
        plt.close()
    
    # Print top markers per cluster
    print("\nTop 5 marker genes per cluster:")
    print("-" * 50)
    for cluster in sorted(adata.obs['leiden'].unique().astype(int)):
        df = sc.get.rank_genes_groups_df(adata, group=str(cluster))
        top_genes = df.head(5)['names'].tolist()
        print(f"  Cluster {cluster}: {', '.join(top_genes)}")
    
    return adata


def main():
    """Main analysis pipeline."""
    print("\n" + "="*60)
    print("Single-cell RNA-seq Analysis Pipeline")
    print("Dataset: GSE84133 (Pancreatic Islet Data)")
    print("="*60)
    
    # Process human data
    print("\n\n" + "#"*60)
    print("# PROCESSING HUMAN DATA")
    print("#"*60)
    
    # Load data
    adata_human = load_data(species='human')
    
    # Calculate QC metrics
    adata_human = calculate_qc_metrics(adata_human)
    
    # Plot QC metrics
    plot_qc_metrics(adata_human, species='human')
    
    # Filter cells and genes
    adata_human = filter_cells_genes(adata_human, 
                                      min_genes=200, 
                                      max_genes=4000,
                                      min_cells=3,
                                      max_mt_pct=15)
    
    # Preprocess
    adata_human = preprocess(adata_human)
    
    # Dimensionality reduction
    adata_human = run_dimensionality_reduction(adata_human)
    
    # Clustering
    adata_human = run_clustering(adata_human, resolution=0.5)
    
    # Find marker genes
    adata_human = find_marker_genes(adata_human, species='human')
    
    # Generate visualizations
    plot_results(adata_human, species='human')
    
    # Save processed data
    print(f"\nSaving processed human data...")
    adata_human.write(os.path.join(PROJECT_ROOT, 'processed_data/human_processed.h5ad'))
    print(f"  Saved: processed_data/human_processed.h5ad")
    
    # Process mouse data
    print("\n\n" + "#"*60)
    print("# PROCESSING MOUSE DATA")
    print("#"*60)
    
    # Load data
    adata_mouse = load_data(species='mouse')
    
    # Calculate QC metrics
    adata_mouse = calculate_qc_metrics(adata_mouse)
    
    # Plot QC metrics
    plot_qc_metrics(adata_mouse, species='mouse')
    
    # Filter cells and genes
    adata_mouse = filter_cells_genes(adata_mouse,
                                      min_genes=200,
                                      max_genes=4000,
                                      min_cells=3,
                                      max_mt_pct=15)
    
    # Preprocess
    adata_mouse = preprocess(adata_mouse)
    
    # Dimensionality reduction
    adata_mouse = run_dimensionality_reduction(adata_mouse)
    
    # Clustering
    adata_mouse = run_clustering(adata_mouse, resolution=0.5)
    
    # Find marker genes
    adata_mouse = find_marker_genes(adata_mouse, species='mouse')
    
    # Generate visualizations
    plot_results(adata_mouse, species='mouse')
    
    # Save processed data
    print(f"\nSaving processed mouse data...")
    adata_mouse.write(os.path.join(PROJECT_ROOT, 'processed_data/mouse_processed.h5ad'))
    print(f"  Saved: processed_data/mouse_processed.h5ad")
    
    print("\n\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)
    print("\nOutput files:")
    print("  - figures/human_*.png - Human QC and visualization plots")
    print("  - figures/mouse_*.png - Mouse QC and visualization plots")
    print("  - processed_data/human_processed.h5ad - Processed human data")
    print("  - processed_data/mouse_processed.h5ad - Processed mouse data")
    
    return adata_human, adata_mouse


if __name__ == '__main__':
    adata_human, adata_mouse = main()
