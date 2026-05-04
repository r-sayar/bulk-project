"""
CV histogram + blood-vs-liver low-CV overlap.

Three things in one run:
  1. Per-gene CV histogram for GTEx whole blood and GTEx liver.
  2. Overlap of the bottom-N low-CV genes between the two tissues
     (Jaccard, top-200 / top-500 / top-1000) → "are the most stable
     genes the same?"
  3. The intersection of "low CV" with "high expression": pull out the
     genes that are simultaneously bottom-quintile CV AND top-quintile
     mean expression. This is the real housekeeping panel candidate.
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

PROJECT = Path("/Users/rls/Desktop/programming-projects/single-cell/bulk-project")
OUT_DIR = PROJECT / "low_cv_enrichment"
OUT_DIR.mkdir(exist_ok=True)

TISSUES = {
    "Blood": Path("/Users/rls/Downloads/gene_reads_v11_whole_blood.gct.gz"),
    "Liver": Path("/Users/rls/Downloads/gene_reads_v11_liver.gct.gz"),
}
CPM_THRESHOLD   = 1.0
MIN_SAMPLE_FRAC = 0.10

BG, CARD, TEXT, MUTED, GRID = '#0e1117', '#1a1d23', '#e6edf3', '#7d8590', '#21262d'
plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': CARD, 'axes.edgecolor': GRID,
    'axes.labelcolor': TEXT, 'text.color': TEXT, 'xtick.color': MUTED,
    'ytick.color': MUTED, 'grid.color': GRID, 'grid.alpha': 0.5,
    'font.family': 'sans-serif', 'font.size': 11,
})


def load_cv(gct_path):
    print(f"  loading {gct_path.name} ...")
    df = pd.read_csv(gct_path, sep="\t", skiprows=2, compression="gzip")
    sym = df["Description"].astype(str).values
    expr = df.iloc[:, 2:].values.astype(np.float64)
    n_g, n_s = expr.shape
    lib = expr.sum(axis=0, keepdims=True)
    cpm = expr / lib * 1e6
    detected = (cpm > CPM_THRESHOLD).sum(axis=1) >= int(MIN_SAMPLE_FRAC * n_s)
    cpm_f = cpm[detected]
    sym_f = sym[detected]
    gmean = cpm_f.mean(axis=1)
    gstd  = cpm_f.std(axis=1)
    cv = np.where(gmean > 0, gstd / gmean, np.nan)
    out = pd.DataFrame({
        "gene": sym_f,
        "mean_cpm": gmean,
        "mean_log2cpm": np.log2(gmean + 1),
        "cv": cv,
    }).dropna(subset=["cv"]).reset_index(drop=True)
    return out, n_s


# ── 1. Build per-tissue CV tables ─────────────────────────────────────
print("Per-tissue CV tables")
tables = {}
sample_counts = {}
for tag, p in TISSUES.items():
    tables[tag], sample_counts[tag] = load_cv(p)
    print(f"  {tag}: {len(tables[tag]):,} expressed genes, {sample_counts[tag]} donors")

blood = tables["Blood"]
liver = tables["Liver"]


# ── 2. Plot: CV histogram + low-CV / high-expr scatter ────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Per-gene CV — GTEx Blood vs Liver", color=TEXT, fontsize=14)

ax = axes[0, 0]
bins = np.linspace(0, 3, 80)
ax.hist(blood["cv"], bins=bins, alpha=0.55, color="#f78166",
        label=f"Blood (n={len(blood):,})", edgecolor="#1a1d23", linewidth=0.4)
ax.hist(liver["cv"], bins=bins, alpha=0.55, color="#3fb950",
        label=f"Liver (n={len(liver):,})", edgecolor="#1a1d23", linewidth=0.4)
for q in (0.001, 0.05, 0.50):
    ax.axvline(np.quantile(blood["cv"], q), color="#f78166",
               ls=":", lw=1, alpha=0.7)
ax.set_xlabel("CV (linear CPM)"); ax.set_ylabel("# expressed genes")
ax.set_title("CV histogram (overlay)")
ax.legend(facecolor=CARD, edgecolor=GRID, labelcolor=TEXT)

ax = axes[0, 1]
ax.hist(np.log10(blood["cv"]), bins=80, alpha=0.55, color="#f78166",
        label="Blood", edgecolor="#1a1d23", linewidth=0.4)
ax.hist(np.log10(liver["cv"]), bins=80, alpha=0.55, color="#3fb950",
        label="Liver", edgecolor="#1a1d23", linewidth=0.4)
ax.set_xlabel("log10(CV)"); ax.set_ylabel("# expressed genes")
ax.set_title("CV histogram (log scale x)")
ax.legend(facecolor=CARD, edgecolor=GRID, labelcolor=TEXT)

# (3) Mean expression vs CV, blood
ax = axes[1, 0]
ax.scatter(blood["mean_log2cpm"], blood["cv"],
           s=2, alpha=0.25, color="#f78166", rasterized=True)
# Highlight the low-CV / high-expr quadrant
hi = blood["mean_log2cpm"] >= np.quantile(blood["mean_log2cpm"], 0.80)
lo = blood["cv"]            <= np.quantile(blood["cv"], 0.20)
quad = blood[hi & lo]
ax.scatter(quad["mean_log2cpm"], quad["cv"],
           s=4, alpha=0.9, color="#58a6ff", label=f"low-CV ∩ high-expr ({len(quad):,})")
ax.axvline(np.quantile(blood["mean_log2cpm"], 0.80), color="#58a6ff",
           ls="--", lw=1, alpha=0.5)
ax.axhline(np.quantile(blood["cv"], 0.20), color="#58a6ff",
           ls="--", lw=1, alpha=0.5)
ax.set_xlabel("mean log2(CPM+1)"); ax.set_ylabel("CV (linear CPM)")
ax.set_title("Blood: mean vs CV")
ax.set_yscale("log")
ax.legend(facecolor=CARD, edgecolor=GRID, labelcolor=TEXT)

ax = axes[1, 1]
ax.scatter(liver["mean_log2cpm"], liver["cv"],
           s=2, alpha=0.25, color="#3fb950", rasterized=True)
hi_l = liver["mean_log2cpm"] >= np.quantile(liver["mean_log2cpm"], 0.80)
lo_l = liver["cv"]            <= np.quantile(liver["cv"], 0.20)
quad_l = liver[hi_l & lo_l]
ax.scatter(quad_l["mean_log2cpm"], quad_l["cv"],
           s=4, alpha=0.9, color="#58a6ff", label=f"low-CV ∩ high-expr ({len(quad_l):,})")
ax.axvline(np.quantile(liver["mean_log2cpm"], 0.80), color="#58a6ff",
           ls="--", lw=1, alpha=0.5)
ax.axhline(np.quantile(liver["cv"], 0.20), color="#58a6ff",
           ls="--", lw=1, alpha=0.5)
ax.set_xlabel("mean log2(CPM+1)"); ax.set_ylabel("CV (linear CPM)")
ax.set_title("Liver: mean vs CV")
ax.set_yscale("log")
ax.legend(facecolor=CARD, edgecolor=GRID, labelcolor=TEXT)

fig.tight_layout()
fig.savefig(OUT_DIR / "cv_blood_vs_liver.png", dpi=150, facecolor=BG, bbox_inches="tight")
plt.close(fig)
print(f"  wrote {OUT_DIR/'cv_blood_vs_liver.png'}")


# ── 3. Bottom-N low-CV overlap between blood and liver ────────────────
print("\nBottom-N low-CV overlap (blood vs liver)")
overlap_rows = []
for n in (200, 500, 1000):
    b_set = set(blood.sort_values("cv")["gene"].head(n))
    l_set = set(liver.sort_values("cv")["gene"].head(n))
    inter = b_set & l_set
    union = b_set | l_set
    jacc = len(inter) / max(1, len(union))
    overlap_rows.append({
        "n": n,
        "blood_only": len(b_set - l_set),
        "liver_only": len(l_set - b_set),
        "shared":     len(inter),
        "jaccard":    jacc,
    })
    print(f"  bottom-{n}: shared={len(inter)} blood-only={len(b_set - l_set)} "
          f"liver-only={len(l_set - b_set)} Jaccard={jacc:.3f}")

ovl_df = pd.DataFrame(overlap_rows)
ovl_df.to_csv(OUT_DIR / "low_cv_blood_vs_liver_overlap.csv", index=False)


# Save the actual shared bottom-200 list — that's the cross-tissue HK panel
b200 = blood.sort_values("cv").head(200)
l200 = liver.sort_values("cv").head(200)
shared200 = pd.merge(b200, l200, on="gene", suffixes=("_blood", "_liver"))
shared200.to_csv(OUT_DIR / "shared_low_cv_top200_blood_liver.csv", index=False)
print(f"  shared bottom-200 saved: {len(shared200)} genes")


# ── 4. low-CV ∩ high-expression cohort (the real HK panel) ────────────
def low_cv_high_expr(tbl, tag):
    cv_thr = np.quantile(tbl["cv"], 0.20)
    ex_thr = np.quantile(tbl["mean_log2cpm"], 0.80)
    sub = tbl[(tbl["cv"] <= cv_thr) & (tbl["mean_log2cpm"] >= ex_thr)]
    sub = sub.sort_values("cv").reset_index(drop=True)
    sub.to_csv(OUT_DIR / f"low_cv_high_expr__{tag}.csv", index=False)
    print(f"  {tag}: low-CV (q<=0.20: cv<={cv_thr:.3f}) & "
          f"high-expr (q>=0.80: log2CPM>={ex_thr:.2f}) = {len(sub):,} genes")
    return sub, cv_thr, ex_thr

print("\nLow-CV ∩ high-expression cohorts")
b_lche, b_cv_thr, b_ex_thr = low_cv_high_expr(blood, "blood")
l_lche, l_cv_thr, l_ex_thr = low_cv_high_expr(liver, "liver")

shared_lche = set(b_lche["gene"]) & set(l_lche["gene"])
print(f"  cross-tissue low-CV∩high-expr (blood ∩ liver): {len(shared_lche):,} genes")
pd.Series(sorted(shared_lche)).to_csv(
    OUT_DIR / "shared_low_cv_high_expr_blood_liver.txt",
    index=False, header=False
)

print("\n  TOP 30 of the shared low-CV∩high-expr panel:")
shared_df = b_lche[b_lche["gene"].isin(shared_lche)].head(30)
print(shared_df.to_string(index=False))


# ── 5. Markdown summary ───────────────────────────────────────────────
md = []
md.append("# CV histogram + blood-vs-liver low-CV comparison\n")
md.append(f"Blood: {sample_counts['Blood']} donors, "
          f"{len(blood):,} expressed genes\n"
          f"Liver: {sample_counts['Liver']} donors, "
          f"{len(liver):,} expressed genes\n")

md.append("\n## CV distribution percentiles\n")
qs = (0.001, 0.005, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90)
md.append("| q | Blood CV | Liver CV |\n|---|---|---|")
for q in qs:
    md.append(f"| {q:.3f} | {np.quantile(blood['cv'], q):.4f} "
              f"| {np.quantile(liver['cv'], q):.4f} |")
md.append("")

md.append("\n## Bottom-N overlap (lowest-CV genes)\n")
md.append(ovl_df.to_markdown(index=False, floatfmt=".3f"))

md.append("\n\n## Low-CV ∩ high-expression cohort\n")
md.append(f"- Blood: cv ≤ {b_cv_thr:.3f} AND log2CPM ≥ {b_ex_thr:.2f} → "
          f"**{len(b_lche):,}** genes\n")
md.append(f"- Liver: cv ≤ {l_cv_thr:.3f} AND log2CPM ≥ {l_ex_thr:.2f} → "
          f"**{len(l_lche):,}** genes\n")
md.append(f"- Shared between tissues: **{len(shared_lche):,}** genes\n")

md.append("\n### Top 30 shared low-CV∩high-expression genes (by blood CV)\n")
md.append(shared_df.to_markdown(index=False, floatfmt=".3f"))

(OUT_DIR / "cv_blood_vs_liver_report.md").write_text("\n".join(md))
print(f"\nWrote {OUT_DIR/'cv_blood_vs_liver_report.md'}")
