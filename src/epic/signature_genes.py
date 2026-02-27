"""
Signature genes for cell type deconvolution.

These gene markers are from the EPIC paper (Racle et al., 2017, eLife)
Appendix 1, Table 1. Signature genes are expressed by reference cell types
but have low/no expression in uncharacterized cells (e.g., cancer cells).
"""

from typing import List, Dict, Optional, Set

# Signature genes from EPIC paper - Appendix 1, Table 1
# These are genes expressed by specific immune/stromal cell types
# but not by cancer cells, enabling deconvolution

SIGNATURE_GENES: Dict[str, List[str]] = {
    # B cells markers
    "B_cells": [
        "BANK1", "CD79A", "CD79B", "FCER2", "FCRL2", "FCRL5",
        "MS4A1", "PAX5", "POU2AF1", "STAP1", "TCL1A"
    ],
    
    # Cancer-Associated Fibroblasts (CAFs) / Stromal cells
    "CAFs": [
        "ADAM33", "CLDN11", "COL1A1", "COL3A1", "COL14A1", "CRISPLD2",
        "CXCL14", "DPT", "F3", "FBLN1", "ISLR", "LUM", "MEG3", "MFAP5",
        "PRELP", "PTGIS", "SFRP2", "SFRP4", "SYNPO2", "TMEM119"
    ],
    
    # CD4+ T cells
    "CD4_T_cells": [
        "ANKRD55", "DGKA", "FOXP3", "GCNT4", "IL2RA",
        "MDS2", "RCAN3", "TBC1D4", "TRAT1"
    ],
    
    # CD8+ T cells
    "CD8_T_cells": [
        "CD8B", "HAUS3", "JAKMIP1", "NAA16", "TSPYL1"
    ],
    
    # Endothelial cells
    "Endothelial": [
        "CDH5", "CLDN5", "CLEC14A", "CXorf36", "ECSCR", "F2RL3",
        "FLT1", "FLT4", "GPR4", "GPR182", "KDR", "MMRN1", "MMRN2",
        "MYCT1", "PTPRB", "RHOJ", "SLCO2A1", "SOX18", "STAB2", "VWF"
    ],
    
    # Macrophages (tumor-infiltrating)
    "Macrophages": [
        "APOC1", "C1QC", "CD14", "CD163", "CD300C", "CD300E",
        "CSF1R", "F13A1", "FPR3", "HAMP", "IL1B", "LILRB4",
        "MS4A6A", "MSR1", "SIGLEC1", "VSIG4"
    ],
    
    # Monocytes (circulating)
    "Monocytes": [
        "CD33", "CD300C", "CD300E", "CECR1", "CLEC6A", "CPVL",
        "EGR2", "EREG", "MS4A6A", "NAGA", "SLC37A2"
    ],
    
    # Neutrophils
    "Neutrophils": [
        "CEACAM3", "CNTNAP3", "CXCR1", "CYP4F3", "FFAR2",
        "HIST1H2BC", "HIST1H3D", "KY", "MMP25", "PGLYRP1",
        "SLC12A1", "TAS2R40"
    ],
    
    # Natural Killer (NK) cells
    "NK_cells": [
        "CD160", "CLIC3", "FGFBP2", "GNLY", "GNPTAB",
        "KLRF1", "NCR1", "NMUR1", "S1PR5", "SH2D1B"
    ],
    
    # General T cell markers (shared between CD4 and CD8)
    "T_cells": [
        "BCL11B", "CD5", "CD28", "IL7R", "ITK", "THEMIS", "UBASH3A"
    ],
}

# Default mRNA content per cell (in picograms)
# From EPIC paper Figure 1 - Figure Supplement 2
# These values are used to convert mRNA proportions to cell fractions
DEFAULT_MRNA_PER_CELL: Dict[str, float] = {
    "B_cells": 0.28,
    "CD4_T_cells": 0.40,
    "CD8_T_cells": 0.40,
    "NK_cells": 0.40,
    "Monocytes": 1.00,
    "Macrophages": 1.00,  # Using monocyte value as proxy
    "Neutrophils": 0.15,
    "CAFs": 0.40,  # Average value (not measured in paper)
    "Endothelial": 0.40,  # Average value (not measured in paper)
    "other_cells": 0.40,  # Default for cancer/uncharacterized cells
}


def get_signature_genes(cell_type: str) -> List[str]:
    """
    Get signature genes for a specific cell type.
    
    Parameters
    ----------
    cell_type : str
        Name of the cell type (e.g., 'B_cells', 'CD4_T_cells')
    
    Returns
    -------
    List[str]
        List of signature gene symbols
    
    Raises
    ------
    ValueError
        If cell type is not found in SIGNATURE_GENES
    """
    if cell_type not in SIGNATURE_GENES:
        available = list(SIGNATURE_GENES.keys())
        raise ValueError(
            f"Unknown cell type: '{cell_type}'. "
            f"Available cell types: {available}"
        )
    return SIGNATURE_GENES[cell_type].copy()


def get_all_signature_genes(
    cell_types: Optional[List[str]] = None,
    include_t_cells: bool = True
) -> List[str]:
    """
    Get combined list of all signature genes for specified cell types.
    
    Parameters
    ----------
    cell_types : List[str], optional
        List of cell types to include. If None, includes all cell types.
    include_t_cells : bool, default True
        Whether to include general T cell markers in addition to
        CD4/CD8 specific markers.
    
    Returns
    -------
    List[str]
        Combined unique list of signature genes
    """
    if cell_types is None:
        cell_types = list(SIGNATURE_GENES.keys())
    
    # Optionally exclude general T_cells if not wanted
    if not include_t_cells and "T_cells" in cell_types:
        cell_types = [ct for ct in cell_types if ct != "T_cells"]
    
    all_genes: Set[str] = set()
    for cell_type in cell_types:
        if cell_type in SIGNATURE_GENES:
            all_genes.update(SIGNATURE_GENES[cell_type])
    
    return sorted(list(all_genes))


def get_mrna_per_cell(cell_types: Optional[List[str]] = None) -> Dict[str, float]:
    """
    Get mRNA content per cell for specified cell types.
    
    Parameters
    ----------
    cell_types : List[str], optional
        List of cell types. If None, returns all available.
    
    Returns
    -------
    Dict[str, float]
        Dictionary mapping cell type to mRNA content (picograms)
    """
    if cell_types is None:
        return DEFAULT_MRNA_PER_CELL.copy()
    
    result = {}
    for ct in cell_types:
        if ct in DEFAULT_MRNA_PER_CELL:
            result[ct] = DEFAULT_MRNA_PER_CELL[ct]
        else:
            # Use default value for unknown cell types
            result[ct] = DEFAULT_MRNA_PER_CELL["other_cells"]
    
    return result


# Pancreas-specific marker genes (useful for your dataset)
# These can be used to build custom reference profiles
PANCREAS_MARKERS: Dict[str, List[str]] = {
    "alpha": ["GCG", "ARX", "IRX2", "TTR"],
    "beta": ["INS", "IAPP", "MAFA", "NKX6-1", "PDX1", "HADH"],
    "delta": ["SST", "RBP4", "HHEX"],
    "gamma": ["PPY", "SERTM1"],
    "epsilon": ["GHRL", "KCNJ6"],
    "acinar": ["PRSS1", "CELA3A", "CELA2A", "CPA1", "CPA2", "PNLIP"],
    "ductal": ["KRT19", "CFTR", "MUC1", "SOX9"],
    "stellate": ["ACTA2", "PDGFRB", "COL1A1", "RGS5"],
    "endothelial": ["VWF", "CDH5", "PECAM1", "KDR"],
    "macrophage": ["CD68", "CD14", "CD163", "ITGAM"],
    "t_cell": ["CD3D", "CD3E", "CD3G", "CD4", "CD8A", "CD8B"],
    "mast": ["TPSAB1", "CPA3", "MS4A2", "KIT"],
    "schwann": ["S100B", "PLP1", "MPZ", "SOX10"],
}
