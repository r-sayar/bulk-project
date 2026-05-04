"""
Enrichr GO/KEGG enrichment on lncRNA cohorts from the GTEx whole-blood
analysis.

Cohorts submitted:
  - all_expressed_named   : 2,709 expressed lncRNAs (named only)
  - stable_top200         : lowest-CV 200 expressed lncRNAs
  - variable_top200       : highest-CV 200 expressed lncRNAs
  - bimodal_all           : 167 bimodal lncRNAs

Caveat: GO/KEGG term-to-gene mappings are heavily protein-coding-biased
        — fewer than ~30% of lncRNA symbols are annotated in any GO term,
        and even fewer in KEGG. To compensate, we also submit each cohort
        to two lncRNA-specific Enrichr libraries:
          - LncHUB_Lncrna_Co-Expression
          - Lncipedia_lncRNAs_in_KEGG_Pathways  (if available)
        plus the conventional GO BP/CC/MF + KEGG_2021_Human as a sanity check.
"""

from pathlib import Path
import time
import pandas as pd
import requests

PROJECT  = Path("/Users/rls/Desktop/programming-projects/single-cell/bulk-project")
LNC_DIR  = PROJECT / "lncrna_analysis"
OUT_DIR  = LNC_DIR / "enrichment"
OUT_DIR.mkdir(exist_ok=True)
ENRICHR  = "https://maayanlab.cloud/Enrichr"

LIBS = [
    "GO_Biological_Process_2023",
    "GO_Cellular_Component_2023",
    "GO_Molecular_Function_2023",
    "KEGG_2021_Human",
    "LncHUB_Lncrna_Co-Expression",  # lncRNA-specific
]


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


def run(genes, tag):
    genes = sorted(set(g for g in genes if g and not g.startswith("ENSG")))
    print(f"\n=== {tag}: {len(genes)} named symbols ===")
    if len(genes) < 5:
        print("  too few named symbols to enrich; skipping")
        return None, []
    sub = enrichr_submit(genes, tag)
    uid = sub["userListId"]
    print(f"  userListId: {uid}")
    blocks = []
    for lib in LIBS:
        try:
            tbl = enrichr_query(uid, lib, top_k=30)
            tbl.to_csv(OUT_DIR / f"{tag}__{lib}.csv", index=False)
            if len(tbl):
                top = tbl.iloc[0]
                print(f"  {lib}: top1 = {top['term'][:65]}  "
                      f"(adj-p {top['adj_p']:.2e}, overlap "
                      f"{len(top['overlap_genes'].split(';'))})")
            else:
                print(f"  {lib}: empty")
            blocks.append((lib, tbl.head(15)))
            time.sleep(0.6)
        except Exception as exc:
            print(f"  !! {lib}: {exc}")
    return uid, blocks


# ── Build cohorts ─────────────────────────────────────────────────────
all_e = pd.read_csv(LNC_DIR / "expressed_lncrnas.csv")
print(f"loaded {len(all_e):,} expressed lncRNAs from {LNC_DIR/'expressed_lncrnas.csv'}")

cohort_genes = {
    "all_expressed":   all_e["symbol"].tolist(),
    "stable_top200":   all_e.sort_values("cv").head(200)["symbol"].tolist(),
    "variable_top200": all_e.sort_values("cv", ascending=False).head(200)["symbol"].tolist(),
}

bimodal = pd.read_csv(LNC_DIR / "bimodal_lncrnas.csv")
cohort_genes["bimodal_all"] = bimodal["symbol"].tolist()
print(f"  bimodal cohort: {len(bimodal)}")


# ── Run ───────────────────────────────────────────────────────────────
reports = []
for tag, genes in cohort_genes.items():
    uid, blocks = run(genes, tag)
    if uid is not None:
        reports.append((tag, uid, blocks, len(set(g for g in genes if not g.startswith("ENSG")))))


# ── Stitch markdown ──────────────────────────────────────────────────
md = ["# lncRNA cohorts — Enrichr GO / KEGG enrichment\n"]
md.append("**Caveat:** GO and KEGG annotations are protein-coding-biased. "
          "lncRNA symbols that have no GO/KEGG mapping are silently dropped "
          "from the background, so the absolute hit-counts here are systematically "
          "low. Treat results qualitatively. The "
          "`LncHUB_Lncrna_Co-Expression` library compensates by mapping "
          "lncRNAs to the protein-coding genes they co-express with — those "
          "results are the more interpretable lncRNA-side enrichment.\n")

for tag, uid, blocks, n_named in reports:
    md.append(f"\n## {tag} ({n_named} named symbols, Enrichr id `{uid}`)\n")
    for lib, tbl in blocks:
        md.append(f"### {lib} (top 15)\n")
        if not len(tbl):
            md.append("_no significant terms_\n")
            continue
        md.append("| Term | adj-p | combined | overlap |\n|---|---|---|---|")
        for _, row in tbl.iterrows():
            md.append(
                f"| {row['term']} | {row['adj_p']:.2e} | "
                f"{row['combined_score']:.1f} | {row['overlap_genes'][:60]} |"
            )
        md.append("")

(OUT_DIR / "lncrna_enrichment_report.md").write_text("\n".join(md))
print(f"\nWrote {OUT_DIR/'lncrna_enrichment_report.md'}")
