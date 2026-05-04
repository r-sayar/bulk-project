"""
Compare lncRNAs to protein-coding genes in GTEx whole blood:
  - CV distribution overlay
  - mean expression vs CV scatter
  - bimodal-gene fraction by biotype
"""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

PROJECT = Path("/Users/rls/Desktop/programming-projects/single-cell/bulk-project")
OUT     = PROJECT / "lncrna_analysis"
GCT     = Path("/Users/rls/Downloads/gene_reads_v11_whole_blood.gct.gz")
BIOTYPE_TSV = OUT / "ensembl_biotypes.tsv"

CPM_THRESHOLD = 1.0
MIN_SAMPLE_FRAC = 0.10

BG, CARD, TEXT, MUTED, GRID = '#0e1117', '#1a1d23', '#e6edf3', '#7d8590', '#21262d'
plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': CARD, 'axes.edgecolor': GRID,
    'axes.labelcolor': TEXT, 'text.color': TEXT, 'xtick.color': MUTED,
    'ytick.color': MUTED, 'grid.color': GRID, 'grid.alpha': 0.5,
    'font.family': 'sans-serif', 'font.size': 11,
})

print("Loading GTEx whole blood ...")
df = pd.read_csv(GCT, sep="\t", skiprows=2, compression="gzip")
ensg = df["Name"].str.split(".").str[0].values
sym = df["Description"].astype(str).values
expr = df.iloc[:, 2:].values.astype(np.float64)
n_g, n_s = expr.shape
lib = expr.sum(axis=0, keepdims=True)
cpm = expr / lib * 1e6
log_cpm = np.log2(cpm + 1)
detected = (cpm > CPM_THRESHOLD).sum(axis=1) >= int(MIN_SAMPLE_FRAC * n_s)
print(f"  expressed: {detected.sum():,}")

bt = pd.read_csv(BIOTYPE_TSV, sep="\t").set_index("ensg")["biotype"].to_dict()
biotype = np.array([bt.get(e, "unknown") for e in ensg])

LNC = {"lncRNA","antisense","lincRNA","macro_lncRNA","bidirectional_promoter_lncRNA",
       "non_coding","processed_transcript","sense_intronic","sense_overlapping",
       "3prime_overlapping_ncRNA","TEC"}
class_ = np.where(np.isin(biotype, list(LNC)), "lncRNA",
        np.where(biotype == "protein_coding", "protein_coding",
        np.where(np.char.endswith(biotype.astype(str), "pseudogene") | (biotype=="processed_pseudogene"),
                 "pseudogene", "other")))

mean_cpm = cpm.mean(axis=1)
std_cpm  = cpm.std(axis=1)
cv = np.where(mean_cpm > 0, std_cpm / mean_cpm, np.nan)
mean_log = log_cpm.mean(axis=1)

mask_e = detected & ~np.isnan(cv)

stats_rows = []
for cls in ["protein_coding", "lncRNA", "pseudogene", "other"]:
    m = mask_e & (class_ == cls)
    if m.sum() == 0:
        continue
    cvs = cv[m]
    stats_rows.append({
        "class": cls,
        "n_expressed": int(m.sum()),
        "cv_p10": float(np.quantile(cvs, 0.10)),
        "cv_p25": float(np.quantile(cvs, 0.25)),
        "cv_median": float(np.median(cvs)),
        "cv_p75": float(np.quantile(cvs, 0.75)),
        "cv_p90": float(np.quantile(cvs, 0.90)),
        "mean_log2cpm_median": float(np.median(mean_log[m])),
    })

stats_df = pd.DataFrame(stats_rows)
stats_df.to_csv(OUT / "biotype_class_cv_stats.csv", index=False)
print(stats_df.to_string(index=False))

# ── Plot ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("lncRNA vs protein-coding — GTEx whole blood", color=TEXT, fontsize=14)

ax = axes[0, 0]
bins = np.linspace(0, 3, 80)
ax.hist(cv[mask_e & (class_ == "protein_coding")], bins=bins,
        alpha=0.55, color="#58a6ff", label="protein_coding",
        edgecolor="#1a1d23", linewidth=0.4)
ax.hist(cv[mask_e & (class_ == "lncRNA")], bins=bins,
        alpha=0.55, color="#d2a8ff", label="lncRNA",
        edgecolor="#1a1d23", linewidth=0.4)
ax.hist(cv[mask_e & (class_ == "pseudogene")], bins=bins,
        alpha=0.55, color="#f0883e", label="pseudogene",
        edgecolor="#1a1d23", linewidth=0.4)
ax.set_xlabel("CV (linear CPM)")
ax.set_ylabel("# expressed genes")
ax.set_title("CV histogram by biotype class")
ax.legend(facecolor=CARD, edgecolor=GRID, labelcolor=TEXT)

ax = axes[0, 1]
for cls, c in [("protein_coding", "#58a6ff"), ("lncRNA", "#d2a8ff"),
               ("pseudogene", "#f0883e")]:
    m = mask_e & (class_ == cls)
    sub_cv = cv[m]
    sub_cv = sub_cv[~np.isnan(sub_cv) & (sub_cv > 0)]
    if len(sub_cv) == 0: continue
    sub_cv_sort = np.sort(sub_cv)
    p = np.linspace(0, 1, len(sub_cv_sort))
    ax.plot(sub_cv_sort, p, color=c, lw=2, label=cls)
ax.set_xscale("log")
ax.set_xlabel("CV (linear CPM, log)")
ax.set_ylabel("ECDF")
ax.set_title("CV ECDF by biotype class")
ax.legend(facecolor=CARD, edgecolor=GRID, labelcolor=TEXT)

ax = axes[1, 0]
m_pc = mask_e & (class_ == "protein_coding")
m_ln = mask_e & (class_ == "lncRNA")
ax.scatter(mean_log[m_pc], cv[m_pc], s=2, alpha=0.2, color="#58a6ff",
           rasterized=True, label=f"protein_coding ({m_pc.sum():,})")
ax.scatter(mean_log[m_ln], cv[m_ln], s=2, alpha=0.4, color="#d2a8ff",
           rasterized=True, label=f"lncRNA ({m_ln.sum():,})")
ax.set_yscale("log")
ax.set_xlabel("mean log2(CPM+1)")
ax.set_ylabel("CV (linear CPM, log)")
ax.set_title("Mean expression vs CV — overlay")
ax.legend(facecolor=CARD, edgecolor=GRID, labelcolor=TEXT, markerscale=3)

ax = axes[1, 1]
labels = stats_df["class"].tolist()
xs = np.arange(len(labels))
ax.bar(xs - 0.2, stats_df["cv_p10"], width=0.18, color="#3fb950", label="p10")
ax.bar(xs,        stats_df["cv_median"], width=0.18, color="#58a6ff", label="median")
ax.bar(xs + 0.2, stats_df["cv_p90"], width=0.18, color="#f78166", label="p90")
ax.set_xticks(xs); ax.set_xticklabels(labels, rotation=15)
ax.set_yscale("log")
ax.set_ylabel("CV")
ax.set_title("CV percentiles by biotype class")
ax.legend(facecolor=CARD, edgecolor=GRID, labelcolor=TEXT)

fig.tight_layout()
out_png = OUT / "lncrna_vs_protein_coding_cv.png"
fig.savefig(out_png, dpi=150, facecolor=BG, bbox_inches="tight")
plt.close(fig)
print(f"\nWrote {out_png}")
