"""
lncRNA-focused analysis of GTEx whole blood
============================================

Steps:
  1. Resolve every GTEx Ensembl gene ID -> biotype via the Ensembl REST API
     (POST /lookup/id, 1000 IDs per call). Cache to disk.
  2. Sub-set to lncRNAs and run the same set of summaries we ran on
     protein-coding genes:
       - expression: mean log2(CPM+1), fraction of donors expressed
       - dispersion: CV on linear CPM
       - bimodality: KDE + find_peaks (same threshold as bimodal_state_*)
       - top expressed, top variable, top bimodal, top stable
       - overlap with known blood-relevant lncRNAs (NEAT1, MALAT1, XIST,
         MIR223HG, HOTAIRM1, HOTAIR, NORAD, GAS5, PVT1, ...)
  3. Classify each lncRNA by symbol convention:
       LINC*    intergenic
       *-AS*    antisense
       *-IT*    intronic
       MIR*HG   miRNA host
       SNHG*    snoRNA host
       <ENSG..> unnamed (HGNC has not given a symbol)
       other    named lncRNA (NEAT1, MALAT1, GAS5, ...)
  4. Save tables and a markdown report.
"""

from pathlib import Path
import json
import time
import warnings

import numpy as np
import pandas as pd
import requests
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde

warnings.filterwarnings("ignore")

PROJECT  = Path("/Users/rls/Desktop/programming-projects/single-cell/bulk-project")
GCT      = Path("/Users/rls/Downloads/gene_reads_v11_whole_blood.gct.gz")
OUT_DIR  = PROJECT / "lncrna_analysis"
OUT_DIR.mkdir(exist_ok=True)
BIOTYPE_CACHE = OUT_DIR / "ensembl_biotypes.tsv"

CPM_THRESHOLD = 1.0
MIN_SAMPLE_FRAC = 0.10
TOP_N = 30


# ── 1. Load GTEx ──────────────────────────────────────────────────────
print("Loading GTEx whole blood ...")
df = pd.read_csv(GCT, sep="\t", skiprows=2, compression="gzip")
gene_ids_v = df["Name"].astype(str).values  # versioned ENSG
gene_ids   = pd.Series(gene_ids_v).str.split(".").str[0].values
gene_desc  = df["Description"].astype(str).values
expr = df.iloc[:, 2:].values.astype(np.float64)
n_genes_raw, n_samples = expr.shape
print(f"  raw: {n_genes_raw:,} genes × {n_samples} samples")

# CPM + log2(CPM+1) on the unfiltered matrix (we filter later for the
# expression-defined cohort, but biotype mapping spans all 74k rows).
lib = expr.sum(axis=0, keepdims=True)
cpm = expr / lib * 1e6
log_cpm = np.log2(cpm + 1)


# ── 2. Load biotype mapping (built from GENCODE v47 basic GTF) ─────────
print("\nLoading GENCODE v47 biotype mapping ...")
biotypes = pd.read_csv(BIOTYPE_CACHE, sep="\t")
print(f"  rows in mapping: {len(biotypes):,}")
biotype_map = dict(zip(biotypes["ensg"], biotypes["biotype"]))
symbol_map  = dict(zip(biotypes["ensg"], biotypes["symbol"]))
chrom_map   = dict(zip(biotypes["ensg"], biotypes["chrom"]))


# ── 3. Project biotype back onto the GTEx rows ─────────────────────────
gene_biotype = np.array([biotype_map.get(g, "unknown") for g in gene_ids])
gene_symbol  = np.array([symbol_map.get(g, "")  for g in gene_ids])
gene_chrom   = np.array([chrom_map.get(g, "")   for g in gene_ids])

print("\nBiotype distribution across all 74,628 GTEx rows:")
bt_counts = pd.Series(gene_biotype).value_counts()
print(bt_counts.head(20).to_string())

LNC_BIOTYPES = {
    "lncRNA",
    "antisense",
    "antisense_RNA",
    "lincRNA",
    "macro_lncRNA",
    "bidirectional_promoter_lncRNA",
    "non_coding",
    "processed_transcript",
    "sense_intronic",
    "sense_overlapping",
    "3prime_overlapping_ncRNA",
    "TEC",  # GENCODE's "to be experimentally confirmed" — mostly lncRNA-like
}
is_lnc = np.isin(gene_biotype, list(LNC_BIOTYPES))
print(f"\nlncRNA-class rows: {is_lnc.sum():,} / {n_genes_raw:,}")


# ── 4. Compute expression / dispersion stats on lncRNAs ────────────────
lnc_idx = np.where(is_lnc)[0]
lnc_log = log_cpm[lnc_idx]
lnc_cpm = cpm[lnc_idx]
lnc_lin_mean = lnc_cpm.mean(axis=1)
lnc_lin_std  = lnc_cpm.std(axis=1)
lnc_cv       = np.where(lnc_lin_mean > 0, lnc_lin_std / lnc_lin_mean, np.nan)
lnc_log_mean = lnc_log.mean(axis=1)
lnc_log_std  = lnc_log.std(axis=1)
frac_expr    = (lnc_cpm > CPM_THRESHOLD).mean(axis=1)

lnc_df = pd.DataFrame({
    "ensg":         np.array(gene_ids)[lnc_idx],
    "symbol":       gene_desc[lnc_idx],            # GTEx-shipped symbol
    "ensembl_symbol": gene_symbol[lnc_idx],        # what Ensembl currently calls it
    "chrom":        gene_chrom[lnc_idx],
    "biotype":      gene_biotype[lnc_idx],
    "mean_cpm":     lnc_lin_mean,
    "std_cpm":      lnc_lin_std,
    "cv":           lnc_cv,
    "mean_log2cpm": lnc_log_mean,
    "std_log2cpm":  lnc_log_std,
    "frac_expressed": frac_expr,
})

# Naming-convention tag (separate from biotype)
def tag_symbol(sym):
    s = str(sym)
    if s.startswith("ENSG"): return "unnamed"
    if s.startswith("LINC"): return "LINC"
    if s.endswith("-AS1") or s.endswith("-AS2") or "-AS" in s: return "antisense"
    if s.startswith("MIR") and ("HG" in s or s.endswith("HG")): return "MIR_host"
    if s.startswith("SNHG"): return "SNHG"
    return "named"

lnc_df["name_class"] = lnc_df["symbol"].apply(tag_symbol)

lnc_df.to_csv(OUT_DIR / "all_lncrnas.csv", index=False)


# Apply the project's standard expression filter for the "expressed lncRNA"
# subset.
expressed = (lnc_cpm > CPM_THRESHOLD).sum(axis=1) >= int(MIN_SAMPLE_FRAC * n_samples)
print(f"  expressed lncRNAs (CPM>{CPM_THRESHOLD} in >= {MIN_SAMPLE_FRAC*100:.0f}% samples): "
      f"{expressed.sum():,}")
lnc_expr = lnc_df.loc[expressed].copy().reset_index(drop=True)
lnc_expr.to_csv(OUT_DIR / "expressed_lncrnas.csv", index=False)
lnc_expr_idx = lnc_idx[expressed]


# ── 5. Top tables ─────────────────────────────────────────────────────
def fmt_top(df_, by, n=TOP_N, ascending=False):
    sub = df_.sort_values(by, ascending=ascending).head(n)
    keep = ["symbol", "biotype", "name_class", "chrom",
            "mean_cpm", "mean_log2cpm", "cv", "frac_expressed"]
    return sub[keep].reset_index(drop=True)


top_expr = fmt_top(lnc_expr, "mean_log2cpm", n=30)
top_var  = fmt_top(lnc_expr, "std_log2cpm", n=30)
top_cv_lo = fmt_top(lnc_expr, "cv", n=30, ascending=True)
top_cv_hi = fmt_top(lnc_expr, "cv", n=30, ascending=False)


# ── 6. Bimodality on expressed lncRNAs ────────────────────────────────
print("\nScreening expressed lncRNAs for bimodality (KDE + find_peaks)...")
log_e = log_cpm[lnc_expr_idx]
bimodal_rows = []
x_eval = np.linspace(0, 20, 1000)

# Same thresholds as bimodal_state_analysis.py for consistency
mean_thr, std_thr, prom_frac, dist = 1.0, 0.3, 0.08, 40

for k, gi in enumerate(lnc_expr_idx):
    vals = log_cpm[gi]
    if vals.mean() <= mean_thr or vals.std() <= std_thr:
        continue
    try:
        kde = gaussian_kde(vals, bw_method="scott")
        density = kde(x_eval)
        peaks, _ = find_peaks(density,
                              prominence=density.max() * prom_frac,
                              distance=dist)
        if len(peaks) >= 2:
            peak_vals = sorted(x_eval[peaks])
            bimodal_rows.append({
                "ensg":    gene_ids[gi],
                "symbol":  gene_desc[gi],
                "biotype": gene_biotype[gi],
                "chrom":   gene_chrom[gi],
                "mean_log2cpm": float(vals.mean()),
                "std_log2cpm":  float(vals.std()),
                "n_peaks":      int(len(peaks)),
                "peak_low":     float(peak_vals[0]),
                "peak_high":    float(peak_vals[-1]),
                "peak_gap":     float(peak_vals[-1] - peak_vals[0]),
            })
    except Exception:
        pass

bimodal_df = pd.DataFrame(bimodal_rows)
print(f"  bimodal expressed lncRNAs: {len(bimodal_df)}")
bimodal_df.sort_values("peak_gap", ascending=False).to_csv(
    OUT_DIR / "bimodal_lncrnas.csv", index=False
)


# ── 7. Known blood-relevant lncRNAs ──────────────────────────────────
KNOWN_BLOOD_LNCS = [
    "NEAT1", "MALAT1", "XIST", "TSIX", "MIR223HG", "HOTAIRM1",
    "HOTAIR", "NORAD", "GAS5", "PVT1", "MEG3", "DANCR",
    "FAS-AS1", "PTPRJ-AS1", "MIAT", "FIRRE", "LINC00152", "LINC00963",
    "HULC", "TUG1", "H19", "BANCR", "MALAT-1",
]
known = lnc_expr[lnc_expr["symbol"].isin(KNOWN_BLOOD_LNCS) |
                 lnc_expr["ensembl_symbol"].isin(KNOWN_BLOOD_LNCS)]
known = known.sort_values("mean_log2cpm", ascending=False)
known.to_csv(OUT_DIR / "known_blood_lncrnas.csv", index=False)


# ── 8. Class summary (LINC / antisense / unnamed / etc.) ──────────────
class_summary = (
    lnc_expr.groupby("name_class")
    .agg(n=("symbol", "size"),
         mean_cpm=("mean_cpm", "median"),
         mean_log2cpm=("mean_log2cpm", "median"),
         cv_median=("cv", "median"),
         cv_p10=("cv", lambda x: float(np.quantile(x, 0.10))),
         cv_p90=("cv", lambda x: float(np.quantile(x, 0.90))),
         frac_expressed_median=("frac_expressed", "median"))
    .sort_values("n", ascending=False)
)
class_summary.to_csv(OUT_DIR / "lncrna_class_summary.csv")


# ── 9. Markdown report ─────────────────────────────────────────────────
def df_to_md(d):
    return d.to_markdown(index=False, floatfmt=".3f")


lines = []
lines.append("# GTEx whole-blood lncRNA analysis\n")
lines.append(f"Source: GTEx v11 whole blood, {n_samples} donors, "
             f"{n_genes_raw:,} GENCODE rows total.\n")
lines.append("Biotype source: Ensembl REST `lookup/id` (cached at "
             f"`{BIOTYPE_CACHE.relative_to(PROJECT)}`).\n")

lines.append("\n## 1. Biotype landscape\n")
lines.append(bt_counts.head(15).to_frame("n_rows").to_markdown())

lines.append("\n\n## 2. lncRNA cohort\n")
lines.append(
    f"- Rows classified as lncRNA-like: **{is_lnc.sum():,}**\n"
    f"- After CPM>{CPM_THRESHOLD} in ≥{MIN_SAMPLE_FRAC*100:.0f}% donors: "
    f"**{expressed.sum():,}**\n"
    f"- That's the lncRNAs the rest of the analysis runs on.\n"
)

lines.append("\n## 3. lncRNA naming-class summary (expressed only)\n")
lines.append(class_summary.reset_index().to_markdown(index=False, floatfmt=".3f"))

lines.append("\n\n## 4. Top 30 most-expressed expressed lncRNAs\n")
lines.append(df_to_md(top_expr))

lines.append("\n\n## 5. Top 30 highest-variance expressed lncRNAs (log2CPM std)\n")
lines.append(df_to_md(top_var))

lines.append("\n\n## 6. Top 30 lowest-CV expressed lncRNAs (most stable)\n")
lines.append(df_to_md(top_cv_lo))

lines.append("\n\n## 7. Top 30 highest-CV expressed lncRNAs (most variable across donors)\n")
lines.append(df_to_md(top_cv_hi))

lines.append("\n\n## 8. Bimodal lncRNAs (top 30 by peak gap)\n")
lines.append(
    bimodal_df.sort_values("peak_gap", ascending=False)
    .head(30)
    .to_markdown(index=False, floatfmt=".3f")
)

lines.append("\n\n## 9. Known blood / immune lncRNAs found in this dataset\n")
lines.append(known.to_markdown(index=False, floatfmt=".3f"))

(OUT_DIR / "lncrna_report.md").write_text("\n".join(lines))
print(f"\nWrote:\n  {OUT_DIR/'lncrna_report.md'}\n  "
      f"{OUT_DIR/'all_lncrnas.csv'}\n  {OUT_DIR/'expressed_lncrnas.csv'}\n  "
      f"{OUT_DIR/'bimodal_lncrnas.csv'}\n  {OUT_DIR/'lncrna_class_summary.csv'}\n  "
      f"{OUT_DIR/'known_blood_lncrnas.csv'}")
