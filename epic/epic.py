"""
EPIC: Estimating the Proportion of Immune and Cancer cells

Main deconvolution algorithm using constrained least squares.

Reference:
    Racle et al. (2017) "Simultaneous enumeration of cancer and immune cell 
    types from bulk tumor gene expression data" eLife 6:e26476
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass

from .utils import (
    calculate_weights,
    goodness_of_fit,
    subset_to_common_genes,
    validate_inputs,
    scale_by_mrna_content,
    GoodnessOfFit,
)
from .reference_profiles import ReferenceProfiles


@dataclass
class EPICResult:
    """
    Container for EPIC deconvolution results.
    
    Attributes
    ----------
    mRNA_proportions : DataFrame
        Estimated mRNA proportions per cell type (samples x cell_types)
        These represent the fraction of mRNA from each cell type.
    cell_fractions : DataFrame
        Estimated cell fractions per cell type (samples x cell_types)
        These are mRNA proportions normalized by mRNA content per cell.
    fit_gof : Dict[str, GoodnessOfFit]
        Goodness of fit metrics for each sample
    genes_used : List[str]
        List of genes used in the deconvolution
    cell_types : List[str]
        Cell types included in the deconvolution
    """
    mRNA_proportions: pd.DataFrame
    cell_fractions: pd.DataFrame
    fit_gof: Dict[str, GoodnessOfFit]
    genes_used: List[str]
    cell_types: List[str]
    
    def summary(self) -> pd.DataFrame:
        """Return summary statistics of the results."""
        summary_data = {
            "cell_type": self.cell_types + ["other_cells"],
            "mean_mRNA_proportion": [],
            "mean_cell_fraction": [],
        }
        
        for ct in self.cell_types + ["other_cells"]:
            summary_data["mean_mRNA_proportion"].append(
                self.mRNA_proportions[ct].mean()
            )
            summary_data["mean_cell_fraction"].append(
                self.cell_fractions[ct].mean()
            )
        
        return pd.DataFrame(summary_data)


def EPIC(
    bulk: Union[pd.DataFrame, pd.Series],
    reference: Union[pd.DataFrame, ReferenceProfiles],
    signature_genes: Optional[List[str]] = None,
    mRNA_cell: Optional[Dict[str, float]] = None,
    constrained_sum: bool = True,
    scale_reference: bool = True,
    with_other_cells: bool = True,
) -> EPICResult:
    """
    EPIC deconvolution: estimate cell type proportions from bulk expression.
    
    This implements the EPIC algorithm from GfellerLab, which uses constrained
    least squares to estimate the proportion of different cell types in bulk
    RNA-seq data.
    
    Parameters
    ----------
    bulk : DataFrame or Series
        Bulk expression data. If DataFrame: genes x samples. If Series: genes.
        Values should be in TPM, RPKM, FPKM, or CPM (not raw counts).
    reference : DataFrame or ReferenceProfiles
        Reference expression profiles. If DataFrame: genes x cell_types.
        If ReferenceProfiles: contains profiles, variability, and signature genes.
    signature_genes : List[str], optional
        List of signature genes to use. If None and reference is ReferenceProfiles,
        uses the signature genes from the reference. If None and reference is
        DataFrame, uses all common genes.
    mRNA_cell : Dict[str, float], optional
        mRNA content per cell for each cell type (in picograms).
        Used to convert mRNA proportions to cell fractions.
        If None, uses default values or equal values for all cell types.
    constrained_sum : bool, default True
        If True, constrains the sum of proportions to be <= 1.
        The remainder is assigned to "other_cells" (e.g., cancer cells).
    scale_reference : bool, default True
        If True, scales reference profiles to match bulk data magnitude.
    with_other_cells : bool, default True
        If True, includes "other_cells" category for uncharacterized cells.
        Set to False if reference profiles cover all expected cell types.
    
    Returns
    -------
    EPICResult
        Deconvolution results including mRNA proportions, cell fractions,
        and goodness of fit metrics.
    
    Example
    -------
    >>> import pandas as pd
    >>> from epic import EPIC
    >>> 
    >>> # Load bulk data (genes x samples)
    >>> bulk = pd.read_csv("bulk_expression.csv", index_col=0)
    >>> 
    >>> # Load reference profiles (genes x cell_types)
    >>> reference = pd.read_csv("reference_profiles.csv", index_col=0)
    >>> 
    >>> # Run EPIC
    >>> result = EPIC(bulk, reference)
    >>> 
    >>> # Get cell fractions
    >>> print(result.cell_fractions)
    
    Notes
    -----
    The algorithm solves the following optimization problem:
    
        minimize: sum_i w_i * (b_i - sum_j(C_ij * p_j))^2
        
        subject to:
            p_j >= 0  for all j (non-negativity)
            sum(p_j) <= 1  (if constrained_sum=True)
    
    where:
        b_i = bulk expression of gene i
        C_ij = reference expression of gene i in cell type j
        p_j = proportion of cell type j
        w_i = weight for gene i (based on variability)
    """
    # Extract data from ReferenceProfiles if provided
    if isinstance(reference, ReferenceProfiles):
        ref_profiles = reference.profiles
        ref_variability = reference.variability
        if signature_genes is None:
            signature_genes = reference.get_all_signature_genes()
        if mRNA_cell is None:
            mRNA_cell = reference.mrna_per_cell
    else:
        ref_profiles = reference
        ref_variability = None
    
    # Validate inputs
    bulk, ref_profiles = validate_inputs(bulk, ref_profiles)
    
    # Get cell types
    cell_types = ref_profiles.columns.tolist()
    
    # Subset to common genes (and signature genes if provided)
    bulk_sub, ref_sub, genes_used = subset_to_common_genes(
        bulk, ref_profiles, signature_genes
    )
    
    if len(genes_used) < 10:
        import warnings
        warnings.warn(
            f"Only {len(genes_used)} genes used for deconvolution. "
            "Consider providing more signature genes or checking gene name format."
        )
    
    # Calculate weights
    if ref_variability is not None:
        var_sub = ref_variability.loc[genes_used]
        weights = calculate_weights(ref_sub, var_sub)
    else:
        weights = calculate_weights(ref_sub)
    
    # Scale reference if requested
    if scale_reference:
        # Scale each reference profile to have similar total expression to bulk
        # This helps when reference and bulk come from different normalization schemes
        bulk_totals = bulk_sub.sum(axis=0)  # Total per sample
        ref_totals = ref_sub.sum(axis=0)    # Total per cell type
        
        # Scale reference so mean total matches mean bulk total
        mean_bulk_total = bulk_totals.mean()
        mean_ref_total = ref_totals.mean()
        
        if mean_ref_total > 0:
            scale_factor = mean_bulk_total / mean_ref_total
            ref_sub = ref_sub * scale_factor
    
    # Run deconvolution for each sample
    sample_names = bulk_sub.columns.tolist()
    mrna_props = {}
    fit_metrics = {}
    
    for sample in sample_names:
        bulk_sample = bulk_sub[sample].values
        
        props, gof = _deconvolve_sample(
            bulk_sample=bulk_sample,
            reference=ref_sub.values,
            weights=weights.values,
            cell_types=cell_types,
            constrained_sum=constrained_sum,
        )
        
        mrna_props[sample] = props
        fit_metrics[sample] = gof
    
    # Create mRNA proportions DataFrame
    mrna_df = pd.DataFrame(mrna_props).T
    mrna_df.columns = cell_types
    mrna_df.index.name = "sample"
    
    # Add "other_cells" column
    if with_other_cells:
        mrna_df["other_cells"] = 1.0 - mrna_df.sum(axis=1)
        mrna_df["other_cells"] = mrna_df["other_cells"].clip(lower=0)
    
    # Convert to cell fractions
    if mRNA_cell is None:
        # Use equal values if not provided
        mRNA_cell = {ct: 0.4 for ct in mrna_df.columns}
    
    # Ensure other_cells has mRNA value
    if "other_cells" not in mRNA_cell:
        mRNA_cell["other_cells"] = 0.4
    
    cell_frac_df = scale_by_mrna_content(
        mrna_df,
        mRNA_cell,
        cell_types=mrna_df.columns.tolist()
    )
    
    return EPICResult(
        mRNA_proportions=mrna_df,
        cell_fractions=cell_frac_df,
        fit_gof=fit_metrics,
        genes_used=genes_used,
        cell_types=cell_types,
    )


def epic_deconvolve(
    bulk: Union[pd.DataFrame, pd.Series],
    reference: Union[pd.DataFrame, ReferenceProfiles],
    **kwargs
) -> EPICResult:
    """
    Alias for EPIC function.
    
    See EPIC() for full documentation.
    """
    return EPIC(bulk, reference, **kwargs)


def _deconvolve_sample(
    bulk_sample: np.ndarray,
    reference: np.ndarray,
    weights: np.ndarray,
    cell_types: List[str],
    constrained_sum: bool = True,
    max_iter: int = 1000,
) -> Tuple[Dict[str, float], GoodnessOfFit]:
    """
    Deconvolve a single bulk sample using constrained least squares.
    
    Parameters
    ----------
    bulk_sample : ndarray
        Bulk expression values (n_genes,)
    reference : ndarray
        Reference profile matrix (n_genes x n_cell_types)
    weights : ndarray
        Gene weights (n_genes,)
    cell_types : List[str]
        Names of cell types
    constrained_sum : bool
        Whether to constrain sum of proportions <= 1
    max_iter : int
        Maximum iterations for optimizer
    
    Returns
    -------
    Tuple[Dict[str, float], GoodnessOfFit]
        Estimated proportions and goodness of fit metrics
    """
    n_genes, n_types = reference.shape
    
    # Normalize weights to have mean 1
    weights = weights / (weights.mean() + 1e-10)
    
    # Apply square root of weights to both sides (weighted least squares)
    sqrt_w = np.sqrt(weights)
    bulk_weighted = bulk_sample * sqrt_w
    ref_weighted = reference * sqrt_w[:, np.newaxis]
    
    # Try scipy's non-negative least squares first (fast, but no sum constraint)
    try:
        from scipy.optimize import nnls
        proportions_nnls, _ = nnls(ref_weighted, bulk_weighted)
        
        # If sum constraint needed and sum > 1, normalize
        if constrained_sum and np.sum(proportions_nnls) > 1.0:
            proportions_nnls = proportions_nnls / np.sum(proportions_nnls)
        
        # Use NNLS result as initial guess
        x0 = np.clip(proportions_nnls, 1e-6, 1.0 - 1e-6)
    except Exception:
        # Fallback initial guess
        if constrained_sum:
            x0 = np.ones(n_types) / (n_types + 1)
        else:
            x0 = np.ones(n_types) / n_types
    
    # Objective function: weighted sum of squared residuals
    def objective(p):
        predicted = reference @ p
        residuals = bulk_sample - predicted
        return np.sum(weights * residuals**2)
    
    # Gradient of objective
    def gradient(p):
        predicted = reference @ p
        residuals = bulk_sample - predicted
        grad = -2 * (reference.T @ (weights * residuals))
        return grad
    
    # Constraints
    constraints = []
    if constrained_sum:
        constraints.append({
            'type': 'ineq',
            'fun': lambda p: 1.0 - np.sum(p),
            'jac': lambda p: -np.ones(n_types)
        })
    
    # Bounds: 0 <= p_j <= 1 for all j
    bounds = [(0, 1) for _ in range(n_types)]
    
    # Run optimization
    result = minimize(
        objective,
        x0,
        method='SLSQP',
        jac=gradient,
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': max_iter, 'ftol': 1e-12, 'disp': False}
    )
    
    # Extract proportions
    proportions = result.x
    
    # Clip small negative values to 0 (numerical precision)
    proportions = np.clip(proportions, 0, 1)
    
    # Ensure sum <= 1 for constrained case
    if constrained_sum and np.sum(proportions) > 1.0:
        proportions = proportions / np.sum(proportions)
    
    # Create proportions dictionary
    props_dict = {ct: proportions[i] for i, ct in enumerate(cell_types)}
    
    # Calculate goodness of fit
    predicted = reference @ proportions
    gof = goodness_of_fit(
        observed=bulk_sample,
        predicted=predicted,
        weights=weights,
        converged=result.success,
        converge_message=result.message if hasattr(result, 'message') else "Unknown"
    )
    
    return props_dict, gof


def _weighted_nnls(
    bulk: np.ndarray,
    reference: np.ndarray,
    weights: np.ndarray
) -> np.ndarray:
    """
    Non-negative least squares with weights.
    
    Alternative simpler approach using scipy.optimize.nnls.
    Does not enforce sum <= 1 constraint.
    
    Parameters
    ----------
    bulk : ndarray
        Bulk expression (n_genes,)
    reference : ndarray
        Reference profiles (n_genes x n_cell_types)
    weights : ndarray
        Gene weights (n_genes,)
    
    Returns
    -------
    ndarray
        Estimated proportions (n_cell_types,)
    """
    from scipy.optimize import nnls
    
    # Apply weights
    sqrt_weights = np.sqrt(weights)
    bulk_weighted = bulk * sqrt_weights
    ref_weighted = reference * sqrt_weights[:, np.newaxis]
    
    # Solve NNLS
    proportions, _ = nnls(ref_weighted, bulk_weighted)
    
    return proportions


def validate_deconvolution(
    result: EPICResult,
    true_proportions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Validate deconvolution results against known true proportions.
    
    Useful for benchmarking with synthetic data or known mixtures.
    
    Parameters
    ----------
    result : EPICResult
        EPIC deconvolution results
    true_proportions : DataFrame
        True cell type proportions (samples x cell_types)
    
    Returns
    -------
    DataFrame
        Validation metrics per cell type
    """
    from scipy.stats import pearsonr, spearmanr
    
    # Get common samples and cell types
    common_samples = result.cell_fractions.index.intersection(true_proportions.index)
    common_types = [ct for ct in result.cell_types if ct in true_proportions.columns]
    
    if len(common_samples) == 0:
        raise ValueError("No common samples between results and true proportions")
    
    if len(common_types) == 0:
        raise ValueError("No common cell types between results and true proportions")
    
    metrics = []
    
    for ct in common_types:
        predicted = result.cell_fractions.loc[common_samples, ct]
        true = true_proportions.loc[common_samples, ct]
        
        # Pearson correlation
        if len(predicted) > 2:
            pearson_r, pearson_p = pearsonr(predicted, true)
            spearman_r, spearman_p = spearmanr(predicted, true)
        else:
            pearson_r = pearson_p = spearman_r = spearman_p = np.nan
        
        # RMSE
        rmse = np.sqrt(np.mean((predicted - true)**2))
        
        # Mean absolute error
        mae = np.mean(np.abs(predicted - true))
        
        metrics.append({
            "cell_type": ct,
            "pearson_r": pearson_r,
            "pearson_p": pearson_p,
            "spearman_r": spearman_r,
            "spearman_p": spearman_p,
            "rmse": rmse,
            "mae": mae,
            "n_samples": len(common_samples),
        })
    
    return pd.DataFrame(metrics)
