"""
EPIC: Estimating the Proportion of Immune and Cancer cells

A Python implementation of the EPIC deconvolution method from GfellerLab.
EPIC estimates cell type proportions from bulk gene expression data using
constrained least squares optimization.

Reference:
    Racle et al. (2017) "Simultaneous enumeration of cancer and immune cell 
    types from bulk tumor gene expression data" eLife 6:e26476
    https://doi.org/10.7554/eLife.26476

Original R package: https://github.com/GfellerLab/EPIC
"""

from .epic import EPIC, epic_deconvolve
from .reference_profiles import (
    ReferenceProfiles,
    build_reference_from_single_cell,
    build_reference_from_pseudobulk,
    load_reference_profiles,
)
from .signature_genes import (
    SIGNATURE_GENES,
    get_signature_genes,
    get_all_signature_genes,
)
from .utils import (
    normalize_to_tpm,
    scale_by_mrna_content,
    calculate_weights,
    goodness_of_fit,
)

__version__ = "1.0.0"
__author__ = "Based on GfellerLab EPIC"

__all__ = [
    # Main function
    "EPIC",
    "epic_deconvolve",
    # Reference profiles
    "ReferenceProfiles",
    "build_reference_from_single_cell",
    "build_reference_from_pseudobulk",
    "load_reference_profiles",
    # Signature genes
    "SIGNATURE_GENES",
    "get_signature_genes",
    "get_all_signature_genes",
    # Utilities
    "normalize_to_tpm",
    "scale_by_mrna_content",
    "calculate_weights",
    "goodness_of_fit",
]
