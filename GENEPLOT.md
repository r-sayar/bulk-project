# geneplot — Quick Gene Expression Visualization

Terminal tool for plotting GTEx whole blood gene expression distributions.

## Quick Start

```bash
# Activate venv first
source venv/bin/activate

# Single gene
python geneplot.py SOD2

# Multiple genes overlaid
python geneplot.py SOD2 HBB ACTB

# Open image after saving
python geneplot.py SOD2 --open
```

## Gene Selection

| Flag | Description | Example |
|------|-------------|---------|
| `GENE [GENE ...]` | Specific genes by name | `geneplot.py HBB ACTB GAPDH` |
| `--top N` | Top N genes by expression | `geneplot.py --top 20` |
| `--top N --by variance` | Top N by variance | `geneplot.py --top 20 --by variance` |
| `--top N --by cv` | Top N by CV | `geneplot.py --top 10 --by cv` |
| `--random N` | N random expressed genes | `geneplot.py --random 15` |
| `--bimodal N` | Top N bimodal genes (KDE) | `geneplot.py --bimodal 10` |
| `--uniform N` | Top N most uniform genes | `geneplot.py --uniform 10` |
| `--similar GENE --n N` | N genes most similar to GENE | `geneplot.py --similar GSTM1 --n 5` |
| `--archetypes N` | Cluster all genes into N shapes | `geneplot.py --archetypes 20` |

## Display Options

| Flag | Description | Example |
|------|-------------|---------|
| `--linear` | Use CPM scale (no log transform) | `geneplot.py HBB --linear` |
| `--kde` | Show KDE density inset | `geneplot.py HBB --kde` |
| `--density` | Dot opacity = local KDE density | `geneplot.py HBB --density` |
| `--train-test` | 700/103 train/test split | `geneplot.py HBB --train-test` |
| `--sc` | Overlay scRNA pseudobulk donors | `geneplot.py HBB --sc` |
| `--sort rank` | Sort samples by per-gene rank (default) | `geneplot.py HBB` |
| `--sort pc1` | Sort samples by PC1 score | `geneplot.py HBB --sort pc1` |
| `--sort pc2` | Sort samples by PC2 score | `geneplot.py HBB --sort pc2` |

## Layout Options

| Flag | Description | Example |
|------|-------------|---------|
| `--per-panel N` | Genes overlaid per panel | `geneplot.py --top 30 --per-panel 10` |
| `--panels N` | Number of panels | `geneplot.py --top 20 --per-panel 5 --panels 4` |

## Output Options

| Flag | Description | Default |
|------|-------------|---------|
| `--out PATH` | Output file path | `geneplot_output.png` |
| `--open` | Open image after saving (macOS) | off |
| `--dpi N` | Resolution | 150 |
| `--seed N` | Random seed | 42 |

## Recipes

```bash
# KDE distributions of top 10 bimodal genes
python geneplot.py --bimodal 10 --kde --open

# 100 genes in groups of 10, log scale
python geneplot.py --top 100 --per-panel 10 --open

# 5 genes most similar to GSTM1, with density dots
python geneplot.py --similar GSTM1 --n 5 --density --open

# Compare HBB in bulk vs scRNA
python geneplot.py HBB --sc --kde --open

# Shape archetypes, linear CPM
python geneplot.py --archetypes 20 --linear --open

# Train/test split for 3 genes
python geneplot.py SOD2 ACTB HBB --train-test --open

# PC1-sorted samples (reveals ischemic time axis)
python geneplot.py HILPDA JUN GSTM1 UTY --sort pc1 --open

# Random 50 genes, 10 per panel, linear scale
python geneplot.py --random 50 --per-panel 10 --linear --open
```

## Data Sources

- **Bulk**: GTEx v11 whole blood (803 samples, 74k genes)
- **scRNA**: HCA blood pseudobulk (8 donors, 33k genes)
- Paths hardcoded in `geneplot.py` — edit `GTEX_PATH` and `HCA_PATH` if needed.
