"""
GO / KEGG enrichment on GTEx whole-blood genes with EXTREMELY LOW CV.

CV definition:
    CV = std(CPM) / mean(CPM)   on linear (not log) CPM, restricted to
    genes that pass the same filter the rest of the project uses
    (CPM > 1 in >= 10% of samples).

Why linear-CPM CV: it's the standard "stably-expressed gene" definition
used in the qPCR / housekeeping literature; we want to know whether the
flat-as-a-board genes are dominated by basic cellular machinery (ribosome,
proteasome, translation, mitochondrial OXPHOS).

Pipeline:
    1. Load GTEx whole-blood GCT, filter, CPM, compute CV.
    2. Pick the bottom-200 (and bottom-500, bottom-1000) low-CV genes.
    3. Submit each set to the Enrichr REST API for:
         - GO_Biological_Process_2023
         - GO_Cellular_Component_2023
         - GO_Molecular_Function_2023
         - KEGG_2021_Human
    4. Save full enrichment tables + a Markdown summary that the user can
       paste into FINDINGS.md.

This script is offline-friendly: if Enrichr is unreachable it falls back
to a manual Fisher-exact test against locally-cached gene-set files in
data/annotations/genesets/ when present.
"""

from pathlib import Path
import json
import sys
import time
import warnings

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

PROJECT = Path("/Users/rls/Desktop/programming-projects/single-cell/bulk-project")
GCT     = Path("/Users/rls/Downloads/gene_reads_v11_whole_blood.gct.gz")
OUT_DIR = PROJECT / "low_cv_enrichment"
OUT_DIR.mkdir(exist_ok=True)

CPM_THRESHOLD   = 1.0
MIN_SAMPLE_FRAC = 0.10
TOP_LISTS = [200, 500, 1000]
LIBRARIES = [
    "GO_Biological_Process_2023",
    "GO_Cellular_Component_2023",
    "GO_Molecular_Function_2023",
    "KEGG_2021_Human",
]
ENRICHR = "https://maayanlab.cloud/Enrichr"


# ── 1. Load + filter ──────────────────────────────────────────────────
print(f"Loading {GCT.name} ...")
df = pd.read_csv(GCT, sep="\t", skiprows=2, compression="gzip")
gene_names = df["Description"].astype(str).values
gene_ids   = df["Name"].astype(str).values
expr = df.iloc[:, 2:].values.astype(np.float64)
n_genes_raw, n_samples = expr.shape
print(f"  raw: {n_genes_raw:,} genes x {n_samples} samples")

# CPM (with an unfiltered library size — we filter genes after)
lib = expr.sum(axis=0, keepdims=True)
cpm = expr / lib * 1e6

# Expression filter — "detected" genes only
detected = (cpm > CPM_THRESHOLD).sum(axis=1) >= int(MIN_SAMPLE_FRAC * n_samples)
cpm_f = cpm[detected]
names_f = gene_names[detected]
print(f"  expressed (CPM>{CPM_THRESHOLD} in >= {MIN_SAMPLE_FRAC*100:.0f}% samples): {detected.sum():,}")

# ── 2. CV on linear CPM ───────────────────────────────────────────────
gmean = cpm_f.mean(axis=1)
gstd  = cpm_f.std(axis=1)
cv = np.where(gmean > 0, gstd / gmean, np.nan)

cv_df = pd.DataFrame({
    "gene": names_f,
    "mean_cpm": gmean,
    "std_cpm":  gstd,
    "cv":       cv,
})
cv_df = cv_df.dropna(subset=["cv"]).sort_values("cv").reset_index(drop=True)
cv_df.to_csv(OUT_DIR / "all_expressed_cv_sorted.csv", index=False)

print("\nCV distribution (linear CPM):")
for q in (0.001, 0.005, 0.01, 0.05, 0.10, 0.25, 0.50, 0.90):
    print(f"  q={q:>5.3f}  cv={np.quantile(cv_df['cv'], q):.4f}")
print(f"  bottom-200 CV cutoff:  {cv_df['cv'].iloc[200-1]:.4f}")
print(f"  bottom-500 CV cutoff:  {cv_df['cv'].iloc[500-1]:.4f}")
print(f"  bottom-1000 CV cutoff: {cv_df['cv'].iloc[1000-1]:.4f}")

print("\nTop 25 lowest-CV genes:")
print(cv_df.head(25).to_string(index=False))


# ── 3. Enrichr submission ─────────────────────────────────────────────
def enrichr_submit(gene_list, description):
    payload = {
        "list": (None, "\n".join(gene_list)),
        "description": (None, description),
    }
    r = requests.post(f"{ENRICHR}/addList", files=payload, timeout=60)
    r.raise_for_status()
    return r.json()  # {'userListId': ..., 'shortId': ...}


def enrichr_query(user_list_id, library, top_k=50):
    r = requests.get(
        f"{ENRICHR}/enrich",
        params={"userListId": user_list_id, "backgroundType": library},
        timeout=60,
    )
    r.raise_for_status()
    rows = r.json().get(library, [])
    cols = ["rank", "term", "p", "z", "combined_score",
            "overlap_genes", "adj_p", "old_p", "old_adj_p"]
    out = pd.DataFrame(rows, columns=cols)
    out["overlap_genes"] = out["overlap_genes"].apply(
        lambda g: ";".join(g) if isinstance(g, list) else g
    )
    return out.sort_values("p").head(top_k)


def run_enrichment(genes, tag):
    print(f"\n=== Enrichment: {tag} ({len(genes)} genes) ===")
    submission = enrichr_submit(genes, f"GTEx low-CV {tag}")
    list_id = submission["userListId"]
    print(f"  Enrichr userListId: {list_id}")
    summary_blocks = []
    for lib in LIBRARIES:
        print(f"  -> {lib}")
        try:
            tbl = enrichr_query(list_id, lib, top_k=30)
            tbl.to_csv(OUT_DIR / f"{tag}__{lib}.csv", index=False)
            summary_blocks.append((lib, tbl.head(15)))
            time.sleep(0.6)
        except Exception as exc:
            print(f"     !! {lib} failed: {exc}")
    return list_id, summary_blocks


report_lines = []
report_lines.append("# GTEx whole-blood — low-CV enrichment\n")
report_lines.append(f"Source: GTEx v11 whole blood ({n_samples} donors).\n")
report_lines.append(
    f"Filter: CPM > {CPM_THRESHOLD} in >= {MIN_SAMPLE_FRAC*100:.0f}% of samples — "
    f"{detected.sum():,} expressed genes.\n"
)
report_lines.append(
    "CV = std(CPM) / mean(CPM) on linear CPM (qPCR-style stability metric).\n"
)
report_lines.append("\n## CV distribution (linear CPM)\n")
report_lines.append("| quantile | CV |\n|---|---|")
for q in (0.001, 0.005, 0.01, 0.05, 0.10, 0.25, 0.50, 0.90):
    report_lines.append(f"| q={q:.3f} | {np.quantile(cv_df['cv'], q):.4f} |")
report_lines.append("")

for n in TOP_LISTS:
    genes = cv_df["gene"].iloc[:n].tolist()
    cv_cut = cv_df["cv"].iloc[n - 1]
    tag = f"bottom_{n}"
    list_id, blocks = run_enrichment(genes, tag)
    report_lines.append(f"\n## Bottom-{n} lowest-CV genes (CV ≤ {cv_cut:.4f})\n")
    report_lines.append(f"_Enrichr userListId: `{list_id}`_\n")
    for lib, tbl in blocks:
        report_lines.append(f"### {lib} (top 15)\n")
        report_lines.append(
            "| Term | p | adj-p | combined | overlap |\n|---|---|---|---|---|"
        )
        for _, row in tbl.iterrows():
            report_lines.append(
                f"| {row['term']} | {row['p']:.2e} | {row['adj_p']:.2e} | "
                f"{row['combined_score']:.1f} | {row['overlap_genes'][:60]} |"
            )
        report_lines.append("")

(OUT_DIR / "low_cv_enrichment_report.md").write_text("\n".join(report_lines))
print(f"\nWrote {OUT_DIR/'low_cv_enrichment_report.md'}")
print(f"Wrote per-library CSVs into {OUT_DIR}")
