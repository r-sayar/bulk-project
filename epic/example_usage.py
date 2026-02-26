#!/usr/bin/env python3
"""
Example usage of EPIC deconvolution with pancreatic islet data.

This script demonstrates how to:
1. Load pseudo-bulk data and create synthetic bulk samples
2. Build reference profiles from known cell type expression
3. Run EPIC deconvolution
4. Validate results against known true proportions
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from epic import (
    EPIC,
    build_reference_from_pseudobulk,
    ReferenceProfiles,
)
from epic.signature_genes import PANCREAS_MARKERS


def load_pseudobulk_data(filepath: str) -> pd.DataFrame:
    """Load pseudo-bulk expression data."""
    df = pd.read_csv(filepath, index_col=0)
    print(f"Loaded pseudo-bulk data: {df.shape[0]} genes x {df.shape[1]} cell types")
    print(f"Cell types: {list(df.columns)}")
    return df


def create_synthetic_bulk_samples(
    pseudobulk: pd.DataFrame,
    n_samples: int = 10,
    seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create synthetic bulk samples with known proportions.
    
    This simulates bulk RNA-seq by mixing cell type-specific profiles
    with random proportions, allowing us to validate EPIC.
    
    Parameters
    ----------
    pseudobulk : DataFrame
        Pseudo-bulk expression (genes x cell_types)
    n_samples : int
        Number of synthetic bulk samples to create
    seed : int
        Random seed for reproducibility
    
    Returns
    -------
    Tuple[DataFrame, DataFrame]
        bulk_samples (genes x samples), true_proportions (samples x cell_types)
    """
    np.random.seed(seed)
    
    cell_types = pseudobulk.columns.tolist()
    n_types = len(cell_types)
    genes = pseudobulk.index.tolist()
    
    # Generate random proportions that sum to 1
    proportions = np.random.dirichlet(np.ones(n_types), size=n_samples)
    
    # Create synthetic bulk samples
    bulk_samples = {}
    for i in range(n_samples):
        sample_name = f"sample_{i+1}"
        # Weighted sum of cell type profiles
        bulk_expr = (pseudobulk.values * proportions[i]).sum(axis=1)
        # Add some noise (Poisson-like)
        noise_scale = 0.05  # 5% noise
        noise = np.random.normal(0, bulk_expr * noise_scale)
        bulk_expr = np.maximum(bulk_expr + noise, 0)  # Keep non-negative
        bulk_samples[sample_name] = bulk_expr
    
    bulk_df = pd.DataFrame(bulk_samples, index=genes)
    
    # True proportions DataFrame
    true_props = pd.DataFrame(
        proportions,
        index=[f"sample_{i+1}" for i in range(n_samples)],
        columns=cell_types
    )
    
    print(f"\nCreated {n_samples} synthetic bulk samples")
    print(f"True proportions summary:")
    print(true_props.describe().round(3))
    
    return bulk_df, true_props


def run_epic_example():
    """Run EPIC deconvolution example."""
    print("=" * 70)
    print("EPIC DECONVOLUTION EXAMPLE - Pancreatic Islet Data")
    print("=" * 70)
    
    # Path to pseudo-bulk data
    data_path = Path(__file__).parent.parent / "results" / "pseudo_bulk_by_celltype.csv"
    
    if not data_path.exists():
        print(f"\nError: Could not find {data_path}")
        print("Please run the R script first to generate pseudo_bulk_by_celltype.csv")
        return None
    
    # Load pseudo-bulk data
    pseudobulk = load_pseudobulk_data(str(data_path))
    
    # Create synthetic bulk samples
    bulk_samples, true_proportions = create_synthetic_bulk_samples(
        pseudobulk, n_samples=10
    )
    
    # Build reference profiles from pseudo-bulk
    # We'll use a subset of cell types as "known" and treat the rest as "unknown"
    known_cell_types = ["alpha", "beta", "delta", "gamma", "ductal", "acinar"]
    unknown_cell_types = [ct for ct in pseudobulk.columns if ct not in known_cell_types]
    
    print(f"\n--- Building reference profiles ---")
    print(f"Known cell types: {known_cell_types}")
    print(f"Unknown cell types (will be 'other_cells'): {unknown_cell_types}")
    
    # Build reference with known cell types only
    ref_pseudobulk = pseudobulk[known_cell_types]
    reference = build_reference_from_pseudobulk(
        ref_pseudobulk,
        normalize=True,
        signature_genes=PANCREAS_MARKERS,
    )
    
    print(f"\nReference profiles: {reference.n_genes} genes x {len(reference.cell_types)} cell types")
    
    # For pseudo-bulk data with complete expression profiles, we can use
    # highly variable genes instead of predefined markers for better results
    # Let's identify genes with high variance across cell types
    gene_variance = ref_pseudobulk.var(axis=1)
    gene_mean = ref_pseudobulk.mean(axis=1)
    
    # Filter: expressed genes with high coefficient of variation
    expressed_genes = gene_mean > gene_mean.quantile(0.25)
    cv = gene_variance / (gene_mean + 1)
    high_cv = cv > cv.quantile(0.75)
    
    # Use top variable genes as signature
    sig_genes = gene_variance[expressed_genes & high_cv].nlargest(500).index.tolist()
    print(f"Using {len(sig_genes)} highly variable genes as markers")
    
    # Run EPIC deconvolution
    print(f"\n--- Running EPIC deconvolution ---")
    result = EPIC(
        bulk=bulk_samples,
        reference=reference,
        signature_genes=sig_genes,
        constrained_sum=True,
        with_other_cells=True,
    )
    
    print(f"\nDeconvolution complete!")
    print(f"Genes used: {len(result.genes_used)}")
    
    # Display results
    print(f"\n--- Estimated Cell Fractions ---")
    print(result.cell_fractions.round(3))
    
    # Compare with true proportions
    print(f"\n--- Validation: Comparing with True Proportions ---")
    
    # Aggregate unknown cell types in true proportions
    true_known = true_proportions[known_cell_types].copy()
    true_known["other_cells"] = true_proportions[unknown_cell_types].sum(axis=1)
    
    # Calculate per-cell-type accuracy
    print(f"\nComparison (True vs Estimated):")
    for ct in known_cell_types + ["other_cells"]:
        if ct in result.cell_fractions.columns and ct in true_known.columns:
            true_vals = true_known[ct]
            est_vals = result.cell_fractions[ct]
            
            # Correlation
            corr = np.corrcoef(true_vals, est_vals)[0, 1]
            
            # RMSE
            rmse = np.sqrt(np.mean((true_vals - est_vals)**2))
            
            # Mean absolute error
            mae = np.mean(np.abs(true_vals - est_vals))
            
            print(f"  {ct:20s}: r={corr:.3f}, RMSE={rmse:.3f}, MAE={mae:.3f}")
    
    # Overall correlation
    all_true = true_known.values.flatten()
    all_est = result.cell_fractions[true_known.columns].values.flatten()
    overall_corr = np.corrcoef(all_true, all_est)[0, 1]
    overall_rmse = np.sqrt(np.mean((all_true - all_est)**2))
    print(f"\n  Overall: r={overall_corr:.3f}, RMSE={overall_rmse:.3f}")
    
    # Check convergence
    print(f"\n--- Optimization Convergence ---")
    converged = sum(1 for gof in result.fit_gof.values() if gof.converged)
    print(f"Converged: {converged}/{len(result.fit_gof)} samples")
    
    return result, true_known


def run_simple_example():
    """Run a simple example without external data files."""
    print("=" * 70)
    print("EPIC DECONVOLUTION - Simple Example")
    print("=" * 70)
    
    # Create simple synthetic data
    np.random.seed(42)
    
    # Define cell types and genes
    cell_types = ["CellType_A", "CellType_B", "CellType_C"]
    n_genes = 100
    gene_names = [f"Gene_{i}" for i in range(n_genes)]
    
    # Create reference profiles with distinct patterns
    reference_data = {}
    for i, ct in enumerate(cell_types):
        # Each cell type has some marker genes with high expression
        expr = np.random.exponential(10, n_genes)
        # Add marker genes (genes 0-9 for A, 10-19 for B, 20-29 for C)
        marker_start = i * 10
        marker_end = (i + 1) * 10
        expr[marker_start:marker_end] *= 10  # Higher expression for markers
        reference_data[ct] = expr
    
    reference = pd.DataFrame(reference_data, index=gene_names)
    print(f"\nReference profiles: {reference.shape[0]} genes x {reference.shape[1]} cell types")
    
    # Create bulk samples with known proportions
    true_proportions = {
        "sample_1": {"CellType_A": 0.5, "CellType_B": 0.3, "CellType_C": 0.2},
        "sample_2": {"CellType_A": 0.2, "CellType_B": 0.5, "CellType_C": 0.3},
        "sample_3": {"CellType_A": 0.1, "CellType_B": 0.1, "CellType_C": 0.8},
    }
    
    bulk_samples = {}
    for sample, props in true_proportions.items():
        bulk_expr = np.zeros(n_genes)
        for ct, prop in props.items():
            bulk_expr += reference[ct].values * prop
        # Add noise
        bulk_expr += np.random.normal(0, bulk_expr * 0.05)
        bulk_expr = np.maximum(bulk_expr, 0)
        bulk_samples[sample] = bulk_expr
    
    bulk = pd.DataFrame(bulk_samples, index=gene_names)
    true_props_df = pd.DataFrame(true_proportions).T
    
    print(f"Bulk samples: {bulk.shape[0]} genes x {bulk.shape[1]} samples")
    print(f"\nTrue proportions:")
    print(true_props_df)
    
    # Run EPIC
    print(f"\n--- Running EPIC ---")
    result = EPIC(
        bulk=bulk,
        reference=reference,
        constrained_sum=True,
        with_other_cells=False,  # No unknown cell types in this example
    )
    
    print(f"\nEstimated cell fractions:")
    print(result.cell_fractions.round(3))
    
    # Compare
    print(f"\n--- Comparison ---")
    print(f"True proportions:")
    print(true_props_df.round(3))
    
    # Calculate error
    error = (result.cell_fractions - true_props_df).abs()
    print(f"\nAbsolute errors:")
    print(error.round(3))
    print(f"\nMean absolute error: {error.values.mean():.3f}")
    
    return result


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Running Simple Example (no external files needed)")
    print("=" * 70 + "\n")
    
    simple_result = run_simple_example()
    
    print("\n\n" + "=" * 70)
    print("Running Full Example with Pancreatic Data")
    print("=" * 70 + "\n")
    
    try:
        full_result, true_props = run_epic_example()
    except Exception as e:
        print(f"\nNote: Full example requires pseudo_bulk_by_celltype.csv")
        print(f"Error: {e}")
        full_result = None
    
    print("\n" + "=" * 70)
    print("Examples complete!")
    print("=" * 70)
