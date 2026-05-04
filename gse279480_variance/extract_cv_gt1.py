"""Extract genes with CV > 1 from each stimulation in GSE279480.

Re-runs the same preprocessing as gse279480_stimulation_variance_analysis.py
(filter -> CPM -> log2 -> per-gene mean/std/CV) then writes one CSV per
stimulation listing genes whose CV exceeds 1.
"""

from collections import Counter
from pathlib import Path
import gzip

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
COUNTS_CSV = HERE.parent / "data/GSE279480/GSE279480_P441_genecounts.csv.gz"
SERIES_MATRIX = HERE.parent / "data/GSE279480/GSE279480_series_matrix.txt.gz"

STIMULATIONS = ["Null", "LPS", "Poly I:C", "SEB"]
CPM_THRESHOLD = 1
MIN_SAMPLE_FRAC = 0.1
MEAN_LOG2_CUTOFF = 0.5  # CV only meaningful for expressed genes


def parse_meta(path: Path) -> pd.DataFrame:
    rows: dict[str, list[list[str]]] = {}
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line.startswith("!series_matrix_table_begin"):
                break
            if not line.startswith("!Sample_"):
                continue
            parts = line.rstrip("\n").split("\t")
            rows.setdefault(parts[0], []).append([p.strip('"') for p in parts[1:]])
    meta = pd.DataFrame({
        "lib": rows["!Sample_description"][0],
    })
    for row in rows.get("!Sample_characteristics_ch1", []):
        keys = [c.split(":", 1)[0].strip() for c in row if ":" in c]
        if not keys:
            continue
        key = Counter(keys).most_common(1)[0][0]
        meta[key] = [c.split(":", 1)[1].strip() if ":" in c else "" for c in row]
    return meta


print("Loading counts + metadata...")
meta = parse_meta(SERIES_MATRIX)
counts = pd.read_csv(COUNTS_CSV, index_col=0)
print(f"  {counts.shape[0]:,} genes x {counts.shape[1]} libraries")

summary = []
all_high_cv = {}
for stim in STIMULATIONS:
    libs = [l for l in meta.loc[meta["stimulation"] == stim, "lib"] if l in counts.columns]
    expr = counts[libs].values.astype(np.float64)
    gene_ids = np.array(counts.index)

    lib_sizes = expr.sum(axis=0)
    cpm = expr / lib_sizes * 1e6
    keep = (cpm > CPM_THRESHOLD).sum(axis=1) >= int(MIN_SAMPLE_FRAC * expr.shape[1])
    cpm = expr[keep] / expr[keep].sum(axis=0) * 1e6
    log_cpm = np.log2(cpm + 1)
    ids = gene_ids[keep]

    means = log_cpm.mean(axis=1)
    stds = log_cpm.std(axis=1)
    cpm_means = cpm.mean(axis=1)
    expressed = means > MEAN_LOG2_CUTOFF
    cvs = np.full_like(means, np.nan)
    cvs[expressed] = stds[expressed] / means[expressed]

    high = expressed & (cvs > 1.0)
    df = pd.DataFrame({
        "ensembl_id": ids[high],
        "cv": cvs[high],
        "mean_log2cpm": means[high],
        "std_log2cpm": stds[high],
        "mean_cpm": cpm_means[high],
        "pct_zero_samples": (log_cpm[high] == 0).sum(axis=1) / log_cpm.shape[1] * 100,
    }).sort_values("cv", ascending=False)

    out = HERE / f"gse279480_{stim.lower().replace(' ', '_').replace(':', '')}_cv_gt1.csv"
    df.to_csv(out, index=False)

    all_high_cv[stim] = set(df["ensembl_id"])
    summary.append({
        "stimulation": stim,
        "n_libraries": expr.shape[1],
        "n_genes_filtered": int(keep.sum()),
        "n_expressed": int(expressed.sum()),
        "n_cv_gt_1": int(high.sum()),
        "pct_of_expressed": high.sum() / expressed.sum() * 100,
        "max_cv": float(np.nanmax(cvs)),
    })
    print(f"  {stim:>9}: {high.sum():>5} genes with CV > 1  "
          f"(of {expressed.sum():,} expressed = {high.sum()/expressed.sum()*100:.1f}%)  -> {out.name}")

print("\nSummary:")
print(pd.DataFrame(summary).to_string(index=False))

shared = set.intersection(*all_high_cv.values())
print(f"\nGenes with CV > 1 in ALL 4 stimulations: {len(shared)}")
union = set.union(*all_high_cv.values())
print(f"Genes with CV > 1 in ANY stimulation:    {len(union)}")

shared_df = pd.DataFrame({"ensembl_id": sorted(shared)})
shared_df.to_csv(HERE / "gse279480_cv_gt1_shared_all_stims.csv", index=False)
print(f"Saved: gse279480_cv_gt1_shared_all_stims.csv")
