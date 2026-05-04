"""
Enrichr enrichment for the low-CV ∩ high-expression cohorts produced
by gtex_cv_blood_vs_liver.py:

  - blood low-CV∩high-expr (1,396 genes)
  - liver low-CV∩high-expr (1,034 genes)
  - cross-tissue intersection (425 genes)  -- the real housekeeping panel
"""

from pathlib import Path
import time
import pandas as pd
import requests

PROJECT = Path("/Users/rls/Desktop/programming-projects/single-cell/bulk-project")
OUT_DIR = PROJECT / "low_cv_enrichment"
ENRICHR = "https://maayanlab.cloud/Enrichr"
LIBS = [
    "GO_Biological_Process_2023",
    "GO_Cellular_Component_2023",
    "GO_Molecular_Function_2023",
    "KEGG_2021_Human",
]


def enrichr_submit(gene_list, description):
    payload = {
        "list": (None, "\n".join(gene_list)),
        "description": (None, description),
    }
    r = requests.post(f"{ENRICHR}/addList", files=payload, timeout=60)
    r.raise_for_status()
    return r.json()


def enrichr_query(uid, lib, top_k=30):
    r = requests.get(f"{ENRICHR}/enrich",
                     params={"userListId": uid, "backgroundType": lib},
                     timeout=60)
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
    print(f"\n=== Enrichr {tag} ({len(genes)} genes) ===")
    sub = enrichr_submit(genes, tag)
    uid = sub["userListId"]
    print(f"  userListId: {uid}")
    blocks = []
    for lib in LIBS:
        try:
            tbl = enrichr_query(uid, lib, top_k=30)
            tbl.to_csv(OUT_DIR / f"{tag}__{lib}.csv", index=False)
            blocks.append((lib, tbl.head(15)))
            print(f"  -> {lib}: top1 = {tbl.iloc[0]['term'][:70]}  "
                  f"(adj-p {tbl.iloc[0]['adj_p']:.2e})")
            time.sleep(0.6)
        except Exception as exc:
            print(f"  !! {lib}: {exc}")
    return uid, blocks


sources = {
    "blood_low_cv_high_expr": OUT_DIR / "low_cv_high_expr__blood.csv",
    "liver_low_cv_high_expr": OUT_DIR / "low_cv_high_expr__liver.csv",
}
shared_path = OUT_DIR / "shared_low_cv_high_expr_blood_liver.txt"

reports = []
for tag, path in sources.items():
    if not path.exists():
        print(f"missing {path}; run gtex_cv_blood_vs_liver.py first")
        continue
    genes = pd.read_csv(path)["gene"].astype(str).tolist()
    uid, blocks = run(genes, tag)
    reports.append((tag, uid, blocks))

shared = [g.strip() for g in shared_path.read_text().splitlines() if g.strip()]
print(f"\nshared cohort: {len(shared)} genes")
uid, blocks = run(shared, "shared_low_cv_high_expr_blood_liver")
reports.append(("shared_low_cv_high_expr_blood_liver", uid, blocks))


# Markdown report
md = ["# Enrichr — low-CV ∩ high-expression cohorts\n"]
for tag, uid, blocks in reports:
    md.append(f"\n## {tag} (Enrichr id `{uid}`)\n")
    for lib, tbl in blocks:
        md.append(f"### {lib} (top 15)\n")
        md.append(
            "| Term | adj-p | combined | overlap |\n|---|---|---|---|"
        )
        for _, row in tbl.iterrows():
            md.append(
                f"| {row['term']} | {row['adj_p']:.2e} | "
                f"{row['combined_score']:.1f} | {row['overlap_genes'][:60]} |"
            )
        md.append("")

(OUT_DIR / "low_cv_high_expr_enrichment.md").write_text("\n".join(md))
print(f"\nWrote {OUT_DIR/'low_cv_high_expr_enrichment.md'}")
