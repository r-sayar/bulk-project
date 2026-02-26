"""
Utility functions for EPIC deconvolution.

Includes functions for:
- TPM normalization
- mRNA content scaling
- Weight calculation for least squares
- Goodness of fit metrics
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Union, Tuple
from dataclasses import dataclass


@dataclass
class GoodnessOfFit:
    """Container for goodness of fit metrics."""
    rmse: float
    correlation: float
    r_squared: float
    residuals: np.ndarray
    converged: bool
    converge_message: str


def normalize_to_tpm(
    counts: Union[pd.DataFrame, np.ndarray],
    gene_lengths: Optional[Union[pd.Series, np.ndarray]] = None,
    target_sum: float = 1e6
) -> Union[pd.DataFrame, np.ndarray]:
    """
    Normalize count data to TPM (Transcripts Per Million).
    
    If gene_lengths is provided, performs full TPM normalization.
    If gene_lengths is None, performs CPM-like normalization
    (counts per million, assuming equal gene lengths).
    
    Parameters
    ----------
    counts : DataFrame or ndarray
        Raw count matrix (genes x samples) or (samples x genes)
    gene_lengths : Series or ndarray, optional
        Gene lengths in base pairs for TPM calculation
    target_sum : float, default 1e6
        Target sum for normalization (1e6 for TPM/CPM)
    
    Returns
    -------
    DataFrame or ndarray
        TPM-normalized expression matrix
    
    Notes
    -----
    TPM formula: TPM_i = (count_i / length_i) / sum(count_j / length_j) * 1e6
    CPM formula: CPM_i = count_i / sum(count_j) * 1e6
    """
    is_dataframe = isinstance(counts, pd.DataFrame)
    
    if is_dataframe:
        data = counts.values.astype(float)
        index = counts.index
        columns = counts.columns
    else:
        data = counts.astype(float)
    
    if gene_lengths is not None:
        # Full TPM normalization
        if isinstance(gene_lengths, pd.Series):
            lengths = gene_lengths.values
        else:
            lengths = gene_lengths
        
        # Normalize by gene length first (RPKM-like)
        # Length in kb
        rpk = data / (lengths[:, np.newaxis] / 1000)
        
        # Then normalize to per million
        scaling_factors = rpk.sum(axis=0) / target_sum
        tpm = rpk / scaling_factors
    else:
        # CPM normalization (no gene length correction)
        scaling_factors = data.sum(axis=0) / target_sum
        # Avoid division by zero
        scaling_factors = np.where(scaling_factors == 0, 1, scaling_factors)
        tpm = data / scaling_factors
    
    if is_dataframe:
        return pd.DataFrame(tpm, index=index, columns=columns)
    return tpm


def scale_by_mrna_content(
    mrna_proportions: Union[pd.DataFrame, pd.Series, Dict[str, float]],
    mrna_per_cell: Dict[str, float],
    cell_types: Optional[list] = None
) -> Union[pd.DataFrame, pd.Series, Dict[str, float]]:
    """
    Convert mRNA proportions to cell fractions using mRNA content per cell.
    
    EPIC first estimates mRNA proportions, then uses this function to
    convert to actual cell fractions based on the fact that different
    cell types contain different amounts of mRNA.
    
    Parameters
    ----------
    mrna_proportions : DataFrame, Series, or Dict
        Estimated mRNA proportions per cell type
    mrna_per_cell : Dict[str, float]
        Amount of mRNA per cell for each cell type (in picograms)
    cell_types : list, optional
        Specific cell types to scale. If None, uses all available.
    
    Returns
    -------
    Same type as input
        Cell fractions (normalized to sum to 1)
    
    Notes
    -----
    Formula: cell_fraction_j = (mrna_proportion_j / r_j) / sum(mrna_proportion_k / r_k)
    where r_j is the mRNA content per cell for cell type j
    """
    if isinstance(mrna_proportions, dict):
        # Dictionary input
        if cell_types is None:
            cell_types = list(mrna_proportions.keys())
        
        scaled = {}
        for ct in cell_types:
            if ct in mrna_proportions and ct in mrna_per_cell:
                scaled[ct] = mrna_proportions[ct] / mrna_per_cell[ct]
            elif ct in mrna_proportions:
                # Use default value if mRNA content not specified
                scaled[ct] = mrna_proportions[ct] / 0.4  # default
        
        # Normalize to sum to 1
        total = sum(scaled.values())
        if total > 0:
            scaled = {k: v / total for k, v in scaled.items()}
        
        return scaled
    
    elif isinstance(mrna_proportions, pd.Series):
        # Series input (single sample)
        if cell_types is None:
            cell_types = mrna_proportions.index.tolist()
        
        scaled = mrna_proportions.copy()
        for ct in cell_types:
            if ct in scaled.index:
                r = mrna_per_cell.get(ct, 0.4)  # default
                scaled[ct] = scaled[ct] / r
        
        # Normalize
        total = scaled.sum()
        if total > 0:
            scaled = scaled / total
        
        return scaled
    
    elif isinstance(mrna_proportions, pd.DataFrame):
        # DataFrame input (multiple samples)
        if cell_types is None:
            cell_types = mrna_proportions.columns.tolist()
        
        scaled = mrna_proportions.copy()
        for ct in cell_types:
            if ct in scaled.columns:
                r = mrna_per_cell.get(ct, 0.4)
                scaled[ct] = scaled[ct] / r
        
        # Normalize each row (sample) to sum to 1
        row_sums = scaled.sum(axis=1)
        row_sums = row_sums.replace(0, 1)  # Avoid division by zero
        scaled = scaled.div(row_sums, axis=0)
        
        return scaled
    
    else:
        raise TypeError(f"Unsupported type: {type(mrna_proportions)}")


def calculate_weights(
    reference_profiles: pd.DataFrame,
    variability: Optional[pd.DataFrame] = None,
    method: str = "iqr"
) -> pd.Series:
    """
    Calculate gene-specific weights for weighted least squares.
    
    Weights give more importance to genes with low variability in
    the reference expression profiles.
    
    Parameters
    ----------
    reference_profiles : DataFrame
        Reference expression profiles (genes x cell_types)
    variability : DataFrame, optional
        Pre-computed variability matrix. If None, computed from reference.
    method : str, default 'iqr'
        Method for computing variability: 'iqr' (interquartile range),
        'std' (standard deviation), or 'cv' (coefficient of variation)
    
    Returns
    -------
    Series
        Gene weights indexed by gene name
    
    Notes
    -----
    Weight formula from EPIC:
        u_i = sum_j(C_ij / (V_ij + epsilon))
        w_i = min(u_i, 100 * median(u))
    """
    genes = reference_profiles.index
    
    if variability is None:
        # Compute variability from reference profiles
        if method == "iqr":
            # Use IQR across cell types for each gene
            q75 = reference_profiles.quantile(0.75, axis=1)
            q25 = reference_profiles.quantile(0.25, axis=1)
            var_per_gene = q75 - q25
        elif method == "std":
            var_per_gene = reference_profiles.std(axis=1)
        elif method == "cv":
            # Coefficient of variation
            means = reference_profiles.mean(axis=1)
            stds = reference_profiles.std(axis=1)
            var_per_gene = stds / (means + 1e-10)
        else:
            raise ValueError(f"Unknown method: {method}")
    else:
        # Use provided variability (sum across cell types)
        var_per_gene = variability.sum(axis=1)
    
    # Small constant to avoid division by zero
    epsilon = 1e-10
    
    # Calculate raw weights: higher expression with lower variability = higher weight
    expr_sum = reference_profiles.sum(axis=1)
    raw_weights = expr_sum / (var_per_gene + epsilon)
    
    # Cap weights to avoid extreme values
    # (100 * median is the EPIC default)
    median_weight = raw_weights.median()
    max_weight = 100 * median_weight
    weights = raw_weights.clip(upper=max_weight)
    
    # Normalize weights to have mean of 1
    weights = weights / weights.mean()
    
    return weights


def goodness_of_fit(
    observed: np.ndarray,
    predicted: np.ndarray,
    weights: Optional[np.ndarray] = None,
    converged: bool = True,
    converge_message: str = "Optimization converged"
) -> GoodnessOfFit:
    """
    Calculate goodness of fit metrics for the deconvolution.
    
    Parameters
    ----------
    observed : ndarray
        Observed bulk expression values
    predicted : ndarray
        Predicted expression from estimated proportions
    weights : ndarray, optional
        Weights used in optimization
    converged : bool, default True
        Whether the optimization converged
    converge_message : str
        Message from optimizer
    
    Returns
    -------
    GoodnessOfFit
        Dataclass containing fit metrics
    """
    residuals = observed - predicted
    
    # Root mean squared error
    if weights is not None:
        # Weighted RMSE
        rmse = np.sqrt(np.average(residuals**2, weights=weights))
    else:
        rmse = np.sqrt(np.mean(residuals**2))
    
    # Pearson correlation
    if len(observed) > 1 and np.std(observed) > 0 and np.std(predicted) > 0:
        correlation = np.corrcoef(observed, predicted)[0, 1]
    else:
        correlation = np.nan
    
    # R-squared (coefficient of determination)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((observed - np.mean(observed))**2)
    if ss_tot > 0:
        r_squared = 1 - (ss_res / ss_tot)
    else:
        r_squared = np.nan
    
    return GoodnessOfFit(
        rmse=rmse,
        correlation=correlation,
        r_squared=r_squared,
        residuals=residuals,
        converged=converged,
        converge_message=converge_message
    )


def subset_to_common_genes(
    bulk: pd.DataFrame,
    reference: pd.DataFrame,
    signature_genes: Optional[list] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, list]:
    """
    Subset bulk and reference data to common genes.
    
    Parameters
    ----------
    bulk : DataFrame
        Bulk expression data (genes x samples)
    reference : DataFrame
        Reference profiles (genes x cell_types)
    signature_genes : list, optional
        List of signature genes to use. If None, uses all common genes.
    
    Returns
    -------
    Tuple[DataFrame, DataFrame, list]
        Subsetted bulk, subsetted reference, and list of common genes used
    """
    # Find common genes
    bulk_genes = set(bulk.index)
    ref_genes = set(reference.index)
    common_genes = bulk_genes.intersection(ref_genes)
    
    if signature_genes is not None:
        # Further subset to signature genes
        sig_set = set(signature_genes)
        common_genes = common_genes.intersection(sig_set)
    
    common_genes = sorted(list(common_genes))
    
    if len(common_genes) == 0:
        raise ValueError(
            "No common genes found between bulk data and reference profiles. "
            "Check that gene names match (case-sensitive)."
        )
    
    bulk_subset = bulk.loc[common_genes]
    ref_subset = reference.loc[common_genes]
    
    return bulk_subset, ref_subset, common_genes


def validate_inputs(
    bulk: Union[pd.DataFrame, pd.Series],
    reference: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Validate and standardize input formats.
    
    Parameters
    ----------
    bulk : DataFrame or Series
        Bulk expression data
    reference : DataFrame
        Reference expression profiles
    
    Returns
    -------
    Tuple[DataFrame, DataFrame]
        Validated bulk (genes x samples) and reference (genes x cell_types)
    """
    # Convert Series to DataFrame if needed
    if isinstance(bulk, pd.Series):
        bulk = bulk.to_frame(name="sample")
    
    # Ensure numeric types
    bulk = bulk.astype(float)
    reference = reference.astype(float)
    
    # Check for negative values
    if (bulk < 0).any().any():
        raise ValueError("Bulk expression contains negative values")
    if (reference < 0).any().any():
        raise ValueError("Reference profiles contain negative values")
    
    return bulk, reference
