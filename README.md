# Bulk RNA-seq Deconvolution & Single-Cell Analysis

Deconvolution of bulk RNA-seq data into cell-type proportions using single-cell references. Includes preprocessing of pancreatic islet scRNA-seq data (GSE84133), bulk RNA-seq analysis (GSE50244), and two deconvolution methods: [EPIC](https://doi.org/10.7554/eLife.26476) and [CDState](https://doi.org/10.1101/2025.03.01.641017).

## Project Structure

```
├── scripts/                 # Analysis pipelines
│   ├── preprocessing_qc.py          # scRNA-seq preprocessing & QC
│   ├── single_cell_analysis.py      # Per-cell gene expression analysis
│   ├── gene_expression_analysis.py  # Bulk RNA-seq differential analysis
│   ├── cellular_localization.R      # Gene localization via HPA
│   └── simple_analysis.r            # Exploratory R analysis
├── epic/                    # EPIC deconvolution (Python implementation)
│   ├── epic.py                      # Constrained least-squares algorithm
│   ├── reference_profiles.py        # Immune cell reference signatures
│   ├── signature_genes.py           # Marker gene sets
│   └── utils.py                     # Shared helpers
├── CDState/                 # CDState deconvolution (git submodule)
├── data/                    # Raw input data (GEO)
├── processed_data/          # Intermediate h5ad files (not tracked — regenerate via scripts)
├── results/                 # Analysis output tables
└── figures/                 # Plots and visualizations
```

## Setup

```bash
git clone --recurse-submodules <repo-url>
cd bulk-project
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

CDState has additional dependencies (JAX + CUDA):

```bash
pip install -r CDState/cdstate_requirements.txt
```

## Data

Raw data is included in `data/` and sourced from GEO:

| Accession | Description |
|-----------|-------------|
| [GSE84133](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE84133) | Pancreatic islet scRNA-seq (human + mouse) |
| [GSE50244](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE50244) | Bulk RNA-seq gene counts |

Processed `.h5ad` files are not tracked due to size (~1.3 GB). Regenerate them by running the preprocessing pipeline:

```bash
python scripts/preprocessing_qc.py
```

## Usage

Run the analysis scripts from the project root:

```bash
# 1. Preprocess scRNA-seq data and generate QC figures
python scripts/preprocessing_qc.py

# 2. Explore single-cell gene expression
python scripts/single_cell_analysis.py

# 3. Bulk RNA-seq analysis
python scripts/gene_expression_analysis.py
```

## References

- Racle et al. (2017). *Simultaneous enumeration of cancer and immune cell types from bulk tumor gene expression data.* eLife 6:e26476.
- Kraft et al. (2025). *CDState: an unsupervised approach to predict malignant cell heterogeneity in tumor bulk RNA-sequencing data.* bioRxiv.
