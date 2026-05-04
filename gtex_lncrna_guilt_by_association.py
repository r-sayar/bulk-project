"""
"Guilt-by-association" GO/KEGG enrichment for lncRNAs.

Direct GO/KEGG enrichment on lncRNA symbols returns nothing because
neither database has substantial lncRNA annotations. Standard practice
in the lncRNA-functional-genomics literature is to use co-expression:
for a lncRNA L, find the protein-coding genes whose donor-by-donor
expression is most correlated with L, and read functional annotation
off those.

Algorithm:
  1. Build the 803-donor expression matrix on shared expressed cohort
     (2,709 lncRNAs + 13,127 protein-coding genes).
  2. For each lncRNA cohort (stable / variable / bimodal), compute
     Pearson correlation of every cohort lncRNA against every PC gene.
  3. For each cohort, take the union of "top-K PC correlates per
     lncRNA" (K=50, |Pearson| only — we keep both positive and negative
     because for bimodal lncRNAs the anti-correlated PC genes are
     equally informative).
  4. Submit each union set to Enrichr against GO BP / CC / MF + KEGG.
  5. Save raw correlation tables AND enrichment tables.
"""

from pathlib import Path
import time
import numpy as np
import pandas as pd
import requests

PROJECT  = Path("/Users/rls/Desktop/programming-projects/single-cell/bulk-project")
LNC_DIR  = PROJECT / "lncrna_analysis"
OUT_DIR  = LNC_DIR / "guilt_by_association"
OUT_DIR.mkdir(exist_ok=True)
GCT      = Path("/Users/rls/Downloads/gene_reads_v11_whole_blood.gct.gz")
BIOTYPE  = LNC_DIR / "ensembl_biotypes.tsv"
ENRICHR  = "https://maayanlab.cloud/Enrichr"

CPM_THRESHOLD   = 1.0
MIN_SAMPLE_FRAC = 0.10
TOP_K_PC        = 50      # PC correlates per lncRNA
LIBS = [
    "GO_Biological_Process_2023",
    "GO_Cellular_Component_2023",
    "GO_Molecular_Function_2023",
    "KEGG_2021_Human",
]

LNC_BIOTYPES = {
    "lncRNA", "antisense", "antisense_RNA", "lincRNA",
    "macro_lncRNA", "bidirectional_promoter_lncRNA",
    "non_coding", "processed_transcript", "sense_intronic",
    "sense_overlapping", "3prime_overlapping_ncRNA", "TEC",
}


# ── 1. Load + filter ──────────────────────────────────────────────────
print("Loading GTEx whole blood ...")
df = pd.read_csv(GCT, sep="\t", skiprows=2, compression="gzip")
ensg = df["Name"].astype(str).str.split(".").str[0].values
sym  = df["Description"].astype(str).values
expr = df.iloc[:, 2:].values.astype(np.float64)
n_g, n_s = expr.shape
lib = expr.sum(axis=0, keepdims=True)
cpm = expr / lib * 1e6
log_cpm = np.log2(cpm + 1)

bt_map = pd.read_csv(BIOTYPE, sep="\t").set_index("ensg")["biotype"].to_dict()
biotype = np.array([bt_map.get(e, "unknown") for e in ensg])

detected = (cpm > CPM_THRESHOLD).sum(axis=1) >= int(MIN_SAMPLE_FRAC * n_s)
is_lnc = np.isin(biotype, list(LNC_BIOTYPES))
is_pc  = biotype == "protein_coding"

mask_lnc = detected & is_lnc
mask_pc  = detected & is_pc
print(f"  expressed lncRNA: {mask_lnc.sum():,}; expressed PC: {mask_pc.sum():,}")

# Submatrices: rows = genes, cols = donors (so log_cpm is already that shape)
L = log_cpm[mask_lnc]                    # n_lnc × 803
P = log_cpm[mask_pc]                     # n_pc  × 803
lnc_sym  = sym[mask_lnc]
lnc_ensg = ensg[mask_lnc]
pc_sym   = sym[mask_pc]


# ── 2. Build cohorts ──────────────────────────────────────────────────
all_e   = pd.read_csv(LNC_DIR / "expressed_lncrnas.csv")
bimodal = pd.read_csv(LNC_DIR / "bimodal_lncrnas.csv")

# index lookup by ENSG so we can pick rows out of L
ensg_to_row = {e: i for i, e in enumerate(lnc_ensg)}

def rows_for(symbols_or_ensg, all_table):
    """Return integer row indices (in L) for a cohort, identified by ENSG."""
    if "ensg" in all_table.columns:
        return [ensg_to_row[e] for e in all_table["ensg"]
                if e in ensg_to_row]
    return []

cohorts = {
    "stable_top200":   rows_for(None, all_e.sort_values("cv").head(200)),
    "variable_top200": rows_for(None, all_e.sort_values("cv", ascending=False).head(200)),
    "bimodal_all":     rows_for(None, bimodal),
}
for tag, rows in cohorts.items():
    print(f"  cohort {tag}: {len(rows)} lncRNAs in expression matrix")


# ── 3. Compute correlations efficiently ───────────────────────────────
# Z-score rows for stable correlation calculation
def zscore_rows(M):
    M = M - M.mean(axis=1, keepdims=True)
    s = M.std(axis=1, keepdims=True)
    s[s == 0] = 1
    return M / s

print("z-scoring expression matrices ...")
Lz = zscore_rows(L)
Pz = zscore_rows(P)


def cohort_correlates(rows, top_k=TOP_K_PC):
    """For a cohort of lncRNA rows, return:
        - long_corr: DataFrame with top-K positive AND top-K negative PC correlates per lncRNA
        - union_pos / union_neg: union of PC symbols across the cohort
    """
    sub = Lz[rows]                       # k × 803
    corr = sub @ Pz.T / Lz.shape[1]      # k × n_pc
    pieces = []
    union_pos, union_neg = set(), set()
    for i, lnc_row_idx in enumerate(rows):
        c = corr[i]
        # top positive
        top_pos = np.argsort(-c)[:top_k]
        for j in top_pos:
            pieces.append({"lncRNA": lnc_sym[lnc_row_idx],
                           "pc": pc_sym[j], "r": float(c[j]),
                           "direction": "pos"})
            union_pos.add(pc_sym[j])
        # top negative
        top_neg = np.argsort(c)[:top_k]
        for j in top_neg:
            pieces.append({"lncRNA": lnc_sym[lnc_row_idx],
                           "pc": pc_sym[j], "r": float(c[j]),
                           "direction": "neg"})
            union_neg.add(pc_sym[j])
    return pd.DataFrame(pieces), union_pos, union_neg


# ── 4. Enrichr helpers ────────────────────────────────────────────────
def enrichr_submit(genes, description):
    payload = {
        "list": (None, "\n".join(genes)),
        "description": (None, description),
    }
    r = requests.post(f"{ENRICHR}/addList", files=payload, timeout=60)
    r.raise_for_status()
    return r.json()


def enrichr_query(uid, lib, top_k=30):
    r = requests.get(
        f"{ENRICHR}/enrich",
        params={"userListId": uid, "backgroundType": lib},
        timeout=60,
    )
    r.raise_for_status()
    rows = r.json().get(lib, [])
    cols = ["rank", "term", "p", "z", "combined_score",
            "overlap_genes", "adj_p", "old_p", "old_adj_p"]
    out = pd.DataFrame(rows, columns=cols)
    out["overlap_genes"] = out["overlap_genes"].apply(
        lambda g: ";".join(g) if isinstance(g, list) else g
    )
    return out.sort_values("p").head(top_k)


def run_enrichment(genes, tag):
    genes = sorted(g for g in genes if g and not g.startswith("ENSG"))
    print(f"\n=== {tag}: {len(genes)} PC symbols ===")
    if len(genes) < 5:
        print("  too few — skipping")
        return None, []
    sub = enrichr_submit(genes, tag)
    uid = sub["userListId"]
    blocks = []
    for lib in LIBS:
        try:
            tbl = enrichr_query(uid, lib, top_k=30)
            tbl.to_csv(OUT_DIR / f"{tag}__{lib}.csv", index=False)
            if len(tbl):
                top = tbl.iloc[0]
                print(f"  {lib}: top1 = {top['term'][:60]}  "
                      f"(adj-p {top['adj_p']:.2e})")
            else:
                print(f"  {lib}: empty")
            blocks.append((lib, tbl.head(15)))
            time.sleep(0.6)
        except Exception as exc:
            print(f"  !! {lib}: {exc}")
    return uid, blocks


# ── 5. Run cohorts ────────────────────────────────────────────────────
md = ["# lncRNA guilt-by-association GO / KEGG enrichment\n"]
md.append(
    f"Method: for each lncRNA in the cohort, take its top {TOP_K_PC} most\n"
    f"positively-correlated AND top {TOP_K_PC} most negatively-correlated\n"
    "protein-coding genes (Pearson, on log2(CPM+1) across 803 donors).\n"
    "Submit the union to Enrichr GO BP/CC/MF + KEGG_2021_Human.\n"
)

for tag, rows in cohorts.items():
    if not rows:
        print(f"skipping empty cohort {tag}")
        continue
    print(f"\n>>> {tag}: {len(rows)} lncRNAs")
    long_corr, union_pos, union_neg = cohort_correlates(rows)
    long_corr.to_csv(OUT_DIR / f"{tag}__correlates.csv", index=False)
    print(f"  union_pos: {len(union_pos)} PC genes; "
          f"union_neg: {len(union_neg)} PC genes")

    md.append(f"\n## {tag} ({len(rows)} lncRNAs)\n")
    md.append(f"- positive-correlate union: **{len(union_pos)}** PC genes\n"
              f"- negative-correlate union: **{len(union_neg)}** PC genes\n")

    for sub_tag, gset in (("pos", union_pos), ("neg", union_neg)):
        full_tag = f"{tag}__{sub_tag}"
        uid, blocks = run_enrichment(gset, full_tag)
        if uid is None:
            continue
        md.append(f"\n### {sub_tag.upper()} correlates "
                  f"({len(gset)} genes, Enrichr id `{uid}`)\n")
        for lib, tbl in blocks:
            md.append(f"#### {lib} (top 15)\n")
            if not len(tbl):
                md.append("_no significant terms_\n")
                continue
            md.append("| Term | adj-p | combined | overlap |\n|---|---|---|---|")
            for _, row in tbl.iterrows():
                md.append(
                    f"| {row['term']} | {row['adj_p']:.2e} | "
                    f"{row['combined_score']:.1f} | "
                    f"{row['overlap_genes'][:60]} |"
                )
            md.append("")

(OUT_DIR / "lncrna_guilt_by_association_report.md").write_text("\n".join(md))
print(f"\nWrote {OUT_DIR/'lncrna_guilt_by_association_report.md'}")
