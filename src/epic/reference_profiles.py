"""
Reference profile management for EPIC deconvolution.

Provides utilities for:
- Loading pre-built reference profiles
- Building reference profiles from single-cell data
- Computing profile statistics (median, variability)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path
from dataclasses import dataclass


@dataclass
class ReferenceProfiles:
    """
    Container for reference expression profiles and associated data.
    
    Attributes
    ----------
    profiles : DataFrame
        Median expression per gene per cell type (genes x cell_types)
    variability : DataFrame
        Variability (IQR or std) per gene per cell type
    signature_genes : Dict[str, List[str]]
        Signature genes for each cell type
    cell_types : List[str]
        List of cell type names
    n_genes : int
        Number of genes in profiles
    mrna_per_cell : Dict[str, float]
        mRNA content per cell for each cell type
    source : str
        Description of where profiles came from
    """
    profiles: pd.DataFrame
    variability: pd.DataFrame
    signature_genes: Dict[str, List[str]]
    cell_types: List[str]
    n_genes: int
    mrna_per_cell: Dict[str, float]
    source: str
    
    def __post_init__(self):
        """Validate after initialization."""
        self.cell_types = list(self.profiles.columns)
        self.n_genes = len(self.profiles)
    
    def subset_genes(self, genes: List[str]) -> 'ReferenceProfiles':
        """Return a new ReferenceProfiles with subset of genes."""
        common = [g for g in genes if g in self.profiles.index]
        return ReferenceProfiles(
            profiles=self.profiles.loc[common],
            variability=self.variability.loc[common],
            signature_genes=self.signature_genes,
            cell_types=self.cell_types,
            n_genes=len(common),
            mrna_per_cell=self.mrna_per_cell,
            source=self.source
        )
    
    def subset_cell_types(self, cell_types: List[str]) -> 'ReferenceProfiles':
        """Return a new ReferenceProfiles with subset of cell types."""
        valid_types = [ct for ct in cell_types if ct in self.cell_types]
        sig_genes = {ct: self.signature_genes.get(ct, []) for ct in valid_types}
        mrna = {ct: self.mrna_per_cell.get(ct, 0.4) for ct in valid_types}
        
        return ReferenceProfiles(
            profiles=self.profiles[valid_types],
            variability=self.variability[valid_types],
            signature_genes=sig_genes,
            cell_types=valid_types,
            n_genes=self.n_genes,
            mrna_per_cell=mrna,
            source=self.source
        )
    
    def get_all_signature_genes(self) -> List[str]:
        """Get combined list of all signature genes."""
        all_genes = set()
        for genes in self.signature_genes.values():
            all_genes.update(genes)
        return sorted(list(all_genes))


def build_reference_from_single_cell(
    expression: pd.DataFrame,
    cell_types: pd.Series,
    method: str = "median",
    min_cells: int = 10,
    signature_genes: Optional[Dict[str, List[str]]] = None,
    mrna_per_cell: Optional[Dict[str, float]] = None
) -> ReferenceProfiles:
    """
    Build reference profiles from single-cell expression data.
    
    Parameters
    ----------
    expression : DataFrame
        Single-cell expression matrix (genes x cells)
    cell_types : Series
        Cell type annotations for each cell, indexed by cell IDs
    method : str, default 'median'
        Aggregation method: 'median' or 'mean'
    min_cells : int, default 10
        Minimum number of cells required per cell type
    signature_genes : Dict[str, List[str]], optional
        Pre-defined signature genes per cell type
    mrna_per_cell : Dict[str, float], optional
        mRNA content per cell. If None, uses default values.
    
    Returns
    -------
    ReferenceProfiles
        Reference profiles built from single-cell data
    
    Example
    -------
    >>> # From AnnData object
    >>> expression = pd.DataFrame(
    ...     adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X,
    ...     index=adata.var_names,
    ...     columns=adata.obs_names
    ... ).T  # genes x cells
    >>> cell_types = adata.obs['cell_type']
    >>> ref = build_reference_from_single_cell(expression, cell_types)
    """
    # Ensure expression has genes as rows
    if expression.shape[0] < expression.shape[1]:
        # Likely cells x genes, need to transpose
        expression = expression.T
    
    # Get unique cell types with enough cells
    type_counts = cell_types.value_counts()
    valid_types = type_counts[type_counts >= min_cells].index.tolist()
    
    if len(valid_types) == 0:
        raise ValueError(
            f"No cell types have >= {min_cells} cells. "
            f"Cell type counts: {type_counts.to_dict()}"
        )
    
    # Compute profiles per cell type
    profiles = {}
    variability = {}
    
    for cell_type in valid_types:
        # Get cells of this type
        cells_mask = cell_types == cell_type
        cells = cell_types[cells_mask].index
        
        # Subset expression to these cells
        expr_subset = expression[cells]
        
        if method == "median":
            profiles[cell_type] = expr_subset.median(axis=1)
            # IQR for variability
            q75 = expr_subset.quantile(0.75, axis=1)
            q25 = expr_subset.quantile(0.25, axis=1)
            variability[cell_type] = q75 - q25
        elif method == "mean":
            profiles[cell_type] = expr_subset.mean(axis=1)
            variability[cell_type] = expr_subset.std(axis=1)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    profiles_df = pd.DataFrame(profiles)
    variability_df = pd.DataFrame(variability)
    
    # Handle signature genes
    if signature_genes is None:
        # Auto-detect based on differential expression
        signature_genes = _detect_signature_genes(profiles_df)
    
    # Handle mRNA per cell
    if mrna_per_cell is None:
        mrna_per_cell = {ct: 0.4 for ct in valid_types}  # default
    
    return ReferenceProfiles(
        profiles=profiles_df,
        variability=variability_df,
        signature_genes=signature_genes,
        cell_types=valid_types,
        n_genes=len(profiles_df),
        mrna_per_cell=mrna_per_cell,
        source="single-cell"
    )


def build_reference_from_pseudobulk(
    pseudobulk: pd.DataFrame,
    normalize: bool = True,
    signature_genes: Optional[Dict[str, List[str]]] = None,
    mrna_per_cell: Optional[Dict[str, float]] = None
) -> ReferenceProfiles:
    """
    Build reference profiles from pseudo-bulk expression data.
    
    This is useful when you have aggregated expression per cell type
    (e.g., from Seurat's AggregateExpression).
    
    Parameters
    ----------
    pseudobulk : DataFrame
        Pseudo-bulk expression (genes x cell_types)
    normalize : bool, default True
        Whether to normalize to TPM/CPM
    signature_genes : Dict[str, List[str]], optional
        Pre-defined signature genes per cell type
    mrna_per_cell : Dict[str, float], optional
        mRNA content per cell
    
    Returns
    -------
    ReferenceProfiles
        Reference profiles built from pseudo-bulk data
    """
    from .utils import normalize_to_tpm
    
    profiles = pseudobulk.copy()
    
    if normalize:
        profiles = normalize_to_tpm(profiles)
    
    cell_types = profiles.columns.tolist()
    
    # For pseudo-bulk, we don't have cell-level variability
    # Use coefficient of variation across cell types as proxy
    row_means = profiles.mean(axis=1)
    row_stds = profiles.std(axis=1)
    cv = row_stds / (row_means + 1e-10)
    
    # Create variability matrix (same CV for all cell types)
    variability = pd.DataFrame(
        np.outer(cv.values, np.ones(len(cell_types))),
        index=profiles.index,
        columns=cell_types
    )
    
    # Handle signature genes
    if signature_genes is None:
        signature_genes = _detect_signature_genes(profiles)
    
    # Handle mRNA per cell
    if mrna_per_cell is None:
        mrna_per_cell = {ct: 0.4 for ct in cell_types}
    
    return ReferenceProfiles(
        profiles=profiles,
        variability=variability,
        signature_genes=signature_genes,
        cell_types=cell_types,
        n_genes=len(profiles),
        mrna_per_cell=mrna_per_cell,
        source="pseudo-bulk"
    )


def load_reference_profiles(
    path: Union[str, Path],
    variability_path: Optional[Union[str, Path]] = None,
    signature_genes: Optional[Dict[str, List[str]]] = None,
    mrna_per_cell: Optional[Dict[str, float]] = None
) -> ReferenceProfiles:
    """
    Load reference profiles from file(s).
    
    Parameters
    ----------
    path : str or Path
        Path to profiles CSV file (genes as rows, cell types as columns)
    variability_path : str or Path, optional
        Path to variability CSV file. If None, computed from profiles.
    signature_genes : Dict, optional
        Signature genes. If None, auto-detected.
    mrna_per_cell : Dict, optional
        mRNA content per cell.
    
    Returns
    -------
    ReferenceProfiles
        Loaded reference profiles
    """
    profiles = pd.read_csv(path, index_col=0)
    
    if variability_path is not None:
        variability = pd.read_csv(variability_path, index_col=0)
    else:
        # Estimate variability from profiles
        row_stds = profiles.std(axis=1)
        variability = pd.DataFrame(
            np.outer(row_stds.values, np.ones(profiles.shape[1])),
            index=profiles.index,
            columns=profiles.columns
        )
    
    cell_types = profiles.columns.tolist()
    
    if signature_genes is None:
        signature_genes = _detect_signature_genes(profiles)
    
    if mrna_per_cell is None:
        mrna_per_cell = {ct: 0.4 for ct in cell_types}
    
    return ReferenceProfiles(
        profiles=profiles,
        variability=variability,
        signature_genes=signature_genes,
        cell_types=cell_types,
        n_genes=len(profiles),
        mrna_per_cell=mrna_per_cell,
        source=str(path)
    )


def _detect_signature_genes(
    profiles: pd.DataFrame,
    top_n: int = 20,
    min_fold_change: float = 2.0
) -> Dict[str, List[str]]:
    """
    Auto-detect signature genes based on differential expression.
    
    For each cell type, finds genes that are highly expressed in that
    cell type relative to others.
    
    Parameters
    ----------
    profiles : DataFrame
        Reference profiles (genes x cell_types)
    top_n : int, default 20
        Maximum number of signature genes per cell type
    min_fold_change : float, default 2.0
        Minimum fold change over mean of other cell types
    
    Returns
    -------
    Dict[str, List[str]]
        Signature genes for each cell type
    """
    signature_genes = {}
    
    for cell_type in profiles.columns:
        # Expression in this cell type
        expr_ct = profiles[cell_type]
        
        # Mean expression in other cell types
        other_cols = [c for c in profiles.columns if c != cell_type]
        expr_others = profiles[other_cols].mean(axis=1)
        
        # Fold change (with pseudocount)
        pseudocount = 1
        fc = (expr_ct + pseudocount) / (expr_others + pseudocount)
        
        # Filter by minimum fold change
        markers = fc[fc >= min_fold_change]
        
        # Sort by fold change and take top N
        markers = markers.sort_values(ascending=False).head(top_n)
        
        signature_genes[cell_type] = markers.index.tolist()
    
    return signature_genes


def combine_reference_profiles(
    *references: ReferenceProfiles,
    resolve_duplicates: str = "first"
) -> ReferenceProfiles:
    """
    Combine multiple reference profile objects.
    
    Parameters
    ----------
    *references : ReferenceProfiles
        Reference profile objects to combine
    resolve_duplicates : str, default 'first'
        How to handle duplicate cell types: 'first', 'last', or 'error'
    
    Returns
    -------
    ReferenceProfiles
        Combined reference profiles
    """
    if len(references) == 0:
        raise ValueError("At least one reference must be provided")
    
    if len(references) == 1:
        return references[0]
    
    # Combine profiles
    all_profiles = []
    all_variability = []
    all_sig_genes = {}
    all_mrna = {}
    
    seen_types = set()
    
    for ref in references:
        for ct in ref.cell_types:
            if ct in seen_types:
                if resolve_duplicates == "error":
                    raise ValueError(f"Duplicate cell type: {ct}")
                elif resolve_duplicates == "first":
                    continue
                # 'last' will overwrite
            
            seen_types.add(ct)
            all_profiles.append(ref.profiles[[ct]])
            all_variability.append(ref.variability[[ct]])
            
            if ct in ref.signature_genes:
                all_sig_genes[ct] = ref.signature_genes[ct]
            if ct in ref.mrna_per_cell:
                all_mrna[ct] = ref.mrna_per_cell[ct]
    
    # Concatenate along columns
    combined_profiles = pd.concat(all_profiles, axis=1)
    combined_variability = pd.concat(all_variability, axis=1)
    
    # Get common genes
    combined_profiles = combined_profiles.dropna()
    combined_variability = combined_variability.loc[combined_profiles.index]
    
    return ReferenceProfiles(
        profiles=combined_profiles,
        variability=combined_variability,
        signature_genes=all_sig_genes,
        cell_types=combined_profiles.columns.tolist(),
        n_genes=len(combined_profiles),
        mrna_per_cell=all_mrna,
        source="combined"
    )
