#!/usr/bin/env bash
# =============================================================================
# Blood Single-Cell Dataset Download Script
# =============================================================================
# SOURCE 1: Tabula Sapiens — Blood
#   85,233 cells, 24 donors, 10x + Smart-seq2
#   2.6 GB h5ad
#
# SOURCE 2: Immunobiology of Aging PBMC (Stanford / Allen Institute)
#   3.76M cells (PBMC), 234 healthy adults age 40-89
#   All blood/PBMC — 40.1 GB all-in-one, or cell-type subsets (~5-11 GB each)
#
# SOURCE 3: OneK1K (GSE196830, Garvan Institute)
#   1.27M PBMCs, 982 donors, 75 pools
#   Raw count CSVs ~1.5 GB compressed total
#
# USAGE:
#   ./download_datasets.sh            # all three sources
#   ./download_datasets.sh tabula     # Tabula Sapiens only
#   ./download_datasets.sh aging      # Aging PBMC only
#   ./download_datasets.sh onek1k     # OneK1K only
# =============================================================================

set -euo pipefail

OUTDIR="$(dirname "$0")/data/downloaded_sc"
mkdir -p "$OUTDIR/tabula_sapiens"
mkdir -p "$OUTDIR/aging_pbmc"
mkdir -p "$OUTDIR/onek1k"

MODE="${1:-all}"

# Use wget with resume (-c) if available, else curl with resume (-C -)
if command -v wget &>/dev/null; then
    dl() { echo "  → $2"; wget -c --show-progress -q -O "$2" "$1" || wget --show-progress -q -O "$2" "$1"; }
else
    dl() { echo "  → $2"; curl -L -C - --progress-bar -o "$2" "$1"; }
fi

echo ""
echo "Output: $OUTDIR"
echo "Free:   $(df -h "$OUTDIR" | awk 'NR==2{print $4}')"
echo ""

# =============================================================================
# SOURCE 1: TABULA SAPIENS — Blood
#   - 85,233 cells across 24 donors
#   - Mix of 10x 3' and Smart-seq2
#   - Annotated cell types: T, B, NK, monocytes, erythrocytes, platelets, etc.
#   - 2.6 GB h5ad
# =============================================================================
if [[ "$MODE" == "all" || "$MODE" == "tabula" ]]; then
    echo "=== SOURCE 1: Tabula Sapiens — Blood (2.6 GB) ==="
    dl "https://datasets.cellxgene.cziscience.com/b225ee37-5e06-4e49-9c25-c3d7b5008dab.h5ad" \
       "$OUTDIR/tabula_sapiens/blood.h5ad"
    echo "  Done."
fi

# =============================================================================
# SOURCE 2: IMMUNOBIOLOGY OF AGING PBMC
#   - 234 healthy adults, age 40-89, 10x Flex
#   - All files are PBMC (blood) — downloading cell-type subsets avoids
#     the 40 GB all-in-one file while still covering every cell type
#   - Total cell-type subsets: ~45 GB
# =============================================================================
if [[ "$MODE" == "all" || "$MODE" == "aging" ]]; then
    echo ""
    echo "=== SOURCE 2: Aging PBMC — cell-type subsets ==="

    echo "  Monocytes + DCs (389k cells, 5.8 GB)"
    dl "https://datasets.cellxgene.cziscience.com/c2068d3f-87e7-4a0e-9795-4dae11bcb9ac.h5ad" \
       "$OUTDIR/aging_pbmc/monocyte_dc.h5ad"

    echo "  B + plasma cells (456k cells, 5.0 GB)"
    dl "https://datasets.cellxgene.cziscience.com/b3bb0e60-379c-4fc6-a78d-75fe4634b86c.h5ad" \
       "$OUTDIR/aging_pbmc/b_plasma.h5ad"

    echo "  NK cells + ILCs (561k cells, 6.8 GB)"
    dl "https://datasets.cellxgene.cziscience.com/a63241c8-5e8f-4ae2-bdd3-18d3c1247241.h5ad" \
       "$OUTDIR/aging_pbmc/nk_ilc.h5ad"

    echo "  Naive CD4 T cells (771k cells, 8.7 GB)"
    dl "https://datasets.cellxgene.cziscience.com/eff8a9fb-7d2f-4b42-bb9d-80db6cb3c138.h5ad" \
       "$OUTDIR/aging_pbmc/cd4_naive.h5ad"

    echo "  CD4 memory + Treg (855k cells, 10.5 GB)"
    dl "https://datasets.cellxgene.cziscience.com/b4d0182b-931d-4626-90e2-c9076aab9e48.h5ad" \
       "$OUTDIR/aging_pbmc/cd4_memory_treg.h5ad"

    echo "  CD8 + gdT + MAIT + dnT (718k cells, 8.7 GB)"
    dl "https://datasets.cellxgene.cziscience.com/e83689e6-c67d-4e1b-98ea-43dbbf23b453.h5ad" \
       "$OUTDIR/aging_pbmc/cd8_gdt_mait.h5ad"

    echo "  Rare cell types (9k cells, 0.1 GB)"
    dl "https://datasets.cellxgene.cziscience.com/982c7fb5-a191-4125-8ff5-edea84790468.h5ad" \
       "$OUTDIR/aging_pbmc/other.h5ad"

    echo "  Done."
fi

# =============================================================================
# SOURCE 3: ONEK1K — GEO GSE196830
#   - 982 donors, 1.27M PBMCs, 75 multiplexed pools
#   - Raw count matrix per pool: CSV.gz (~20 MB each, ~1.5 GB total)
#   - Also downloads barcode-to-donor assignments (Individual_Barcodes.csv.gz)
#   - Does NOT download microarray IDAT files (not needed for RNA analysis)
# =============================================================================
if [[ "$MODE" == "all" || "$MODE" == "onek1k" ]]; then
    echo ""
    echo "=== SOURCE 3: OneK1K scRNA-seq counts (GSE196830, ~1.5 GB) ==="
    echo "  Downloading 75 pool count matrices + barcode assignments ..."
    echo "  (skipping microarray IDAT files)"

    BASE="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE196nnn/GSE196830/suppl"
    FILELIST_URL="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE196nnn/GSE196830/suppl/filelist.txt"

    echo "  Fetching file list from GEO ..."
    # Pull just the RawCounts and Individual_Barcodes filenames from the filelist
    FILES=$(curl -s "$FILELIST_URL" | awk '{print $2}' | grep -E "(RawCounts|Individual_Barcodes)")
    TOTAL=$(echo "$FILES" | wc -l | tr -d ' ')
    echo "  Found $TOTAL files to download"

    i=0
    while IFS= read -r fname; do
        [[ -z "$fname" ]] && continue
        dest="$OUTDIR/onek1k/$fname"
        if [[ -f "$dest" ]]; then
            echo "  skip (exists): $fname"
        else
            i=$((i+1))
            echo "  [$i/$TOTAL] $fname"
            dl "$BASE/$fname" "$dest"
        fi
    done <<< "$FILES"

    echo "  Done."
fi

# =============================================================================
# SUMMARY
# =============================================================================
echo ""
echo "=== Downloaded files ==="
for subdir in tabula_sapiens aging_pbmc onek1k; do
    if ls "$OUTDIR/$subdir/"* &>/dev/null 2>&1; then
        total=$(du -sh "$OUTDIR/$subdir" 2>/dev/null | cut -f1)
        count=$(ls "$OUTDIR/$subdir/" | wc -l | tr -d ' ')
        echo "  $subdir/   $count files   $total"
    fi
done
echo ""
echo "Total used: $(du -sh "$OUTDIR" 2>/dev/null | cut -f1)"
echo "Free remaining: $(df -h "$OUTDIR" | awk 'NR==2{print $4}')"
