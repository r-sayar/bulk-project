"""
GSE279480 (Smithmyer 2025) Stimulation Variance Analysis Pipeline
==================================================================
Mirrors gtex_tissue_variance_analysis.py, but instead of comparing across
GTEx tissues (Blood vs Liver), it compares the 4 ex-vivo stimulation
conditions of the same whole-blood cohort: Null, LPS, Poly I:C, SEB.

The pipeline (identical to the GTEx version):
  1. Loads raw HTSeq counts and splits by stimulation condition
  2. Filters out lowly-expressed genes (CPM > 1 in >=10% samples)
  3. Normalizes to CPM, then log2(CPM + 1)
  4. Computes per-gene mean / std / CV across samples
  5. Clusters samples (HVG -> z-score -> PCA -> KMeans)
  6. Compares within-cluster CV vs whole-condition CV
  7. Generates per-condition (8 panels) and cross-condition (6 panels)
     figures.

Outputs land alongside this script in gse279480_variance/.
"""

from collections import Counter
from itertools import combinations
from pathlib import Path
import gzip
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════

HERE = Path(__file__).parent
COUNTS_CSV = HERE.parent / "data/GSE279480/GSE279480_P441_genecounts.csv.gz"
SERIES_MATRIX = HERE.parent / "data/GSE279480/GSE279480_series_matrix.txt.gz"

STIMULATIONS = ["Null", "LPS", "Poly I:C", "SEB"]
PAIRWISE_CONTRAST = ("Null", "SEB")  # which pair to scatter in the comparison fig

N_CLUSTERS = 10
N_HVG = 5000
N_PCA = 50
CPM_THRESHOLD = 1
MIN_SAMPLE_FRAC = 0.1

# ══════════════════════════════════════════════════════════════════════
# STYLE
# ══════════════════════════════════════════════════════════════════════

BG = "#0e1117"; CARD = "#1a1d23"; TEXT = "#e6edf3"; MUTED = "#7d8590"; GRID = "#21262d"
PALETTE = ["#58a6ff", "#f78166", "#3fb950", "#d2a8ff", "#f0883e",
           "#79c0ff", "#ffa657", "#56d364", "#bc8cff", "#e3b341"]
STIM_COLORS = {
    "Null":     "#7d8590",
    "LPS":      "#f78166",
    "Poly I:C": "#58a6ff",
    "SEB":      "#3fb950",
}

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": CARD, "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT, "text.color": TEXT, "xtick.color": MUTED,
    "ytick.color": MUTED, "grid.color": GRID, "grid.alpha": 0.5,
    "font.family": "sans-serif", "font.size": 11,
})

# ══════════════════════════════════════════════════════════════════════
# 1. LOAD METADATA + COUNTS
# ══════════════════════════════════════════════════════════════════════

def parse_series_matrix(path: Path) -> pd.DataFrame:
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
        "gsm": rows["!Sample_geo_accession"][0],
        "lib": rows["!Sample_description"][0],
    })
    for row in rows.get("!Sample_characteristics_ch1", []):
        keys = [c.split(":", 1)[0].strip() for c in row if ":" in c]
        if not keys:
            continue
        key = Counter(keys).most_common(1)[0][0]
        meta[key] = [c.split(":", 1)[1].strip() if ":" in c else "" for c in row]
    return meta


def load_counts(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0)


def split_by_stimulation(counts: pd.DataFrame, meta: pd.DataFrame
                         ) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return {stim: (expr, gene_ids, lib_ids)} where expr is (genes, samples)."""
    out = {}
    for stim in STIMULATIONS:
        libs = meta.loc[meta["stimulation"] == stim, "lib"].tolist()
        libs = [l for l in libs if l in counts.columns]
        sub = counts[libs]
        out[stim] = (
            sub.values.astype(np.float64),
            np.array(sub.index),
            np.array(libs),
        )
    return out


def print_overview(name, expr):
    n_genes, n_samples = expr.shape
    print(f"\n{'='*70}")
    print(f"  {name.upper()} — FILE OVERVIEW")
    print(f"{'='*70}")
    print(f"  Genes:        {n_genes:,}")
    print(f"  Samples:      {n_samples}")
    print(f"  Data points:  {n_genes * n_samples:,}")
    print(f"  % zeros:      {(expr == 0).sum() / expr.size * 100:.1f}%")
    print(f"  Min:          {expr.min():,.0f}")
    print(f"  Max:          {expr.max():,.0f}")
    print(f"  Global mean:  {expr.mean():,.2f}")
    print(f"  Global std:   {expr.std():,.2f}")
    print(f"  Median:       {np.median(expr):,.1f}")

# ══════════════════════════════════════════════════════════════════════
# 2. PREPROCESSING
# ══════════════════════════════════════════════════════════════════════

def preprocess(expr, gene_ids):
    n_genes, n_samples = expr.shape
    lib_sizes = expr.sum(axis=0)
    cpm = expr / lib_sizes * 1e6
    min_samples = int(MIN_SAMPLE_FRAC * n_samples)
    keep = (cpm > CPM_THRESHOLD).sum(axis=1) >= min_samples
    expr_filt = expr[keep]
    ids_filt = gene_ids[keep]
    expr_cpm = expr_filt / expr_filt.sum(axis=0) * 1e6
    expr_log = np.log2(expr_cpm + 1)
    print(f"  Filtered: {keep.sum():,} / {n_genes:,} genes kept")
    return expr_filt, expr_cpm, expr_log, ids_filt

# ══════════════════════════════════════════════════════════════════════
# 3. GLOBAL VARIANCE STATS
# ══════════════════════════════════════════════════════════════════════

def compute_variance_stats(expr_log, expr_cpm, names_filt):
    n_genes, n_samples = expr_log.shape
    means = expr_log.mean(axis=1)
    stds = expr_log.std(axis=1)
    expressed = means > 0.5
    cvs = np.full(n_genes, np.nan)
    cvs[expressed] = stds[expressed] / means[expressed]
    cpm_means = expr_cpm.mean(axis=1)

    valid_cvs = cvs[~np.isnan(cvs)]
    print(f"\n  Per-gene CV (expressed genes, n={expressed.sum():,}):")
    print(f"    Mean CV:   {valid_cvs.mean():.4f}")
    print(f"    Median CV: {np.median(valid_cvs):.4f}")

    print(f"\n  Expression level distribution:")
    bins = [
        ("Zero (mean=0)",        means == 0),
        ("Very low (0–1)",       (means > 0) & (means <= 1)),
        ("Low (1–100 CPM)",      (means > 1) & (cpm_means <= 100)),
        ("Medium (100–10k CPM)", (cpm_means > 100) & (cpm_means <= 10000)),
        ("High (>10k CPM)",      cpm_means > 10000),
    ]
    for label, mask in bins:
        n = mask.sum()
        print(f"    {label:<25} {n:>6,} ({n/n_genes*100:.1f}%)")

    thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.50]
    print(f"\n  CV threshold breakdown:")
    print(f"    {'CV range':<14} {'# genes':>8} {'Mean log2':>10} {'Std log2':>9} {'Mean CPM':>10}")
    print(f"    {'-'*55}")
    prev = 0
    for t in thresholds:
        mask = expressed & (cvs >= prev) & (cvs < t)
        n = mask.sum()
        if n > 0:
            m = means[mask].mean(); s = stds[mask].mean(); raw = cpm_means[mask].mean()
            print(f"    {prev:.2f}–{t:.2f}     {n:>8,} {m:>10.3f} {s:>9.3f} {raw:>10.1f}")
        prev = t
    mask = expressed & (cvs >= 0.50)
    if mask.sum() > 0:
        print(f"    ≥0.50         {mask.sum():>8,} {means[mask].mean():>10.3f} "
              f"{stds[mask].mean():>9.3f} {cpm_means[mask].mean():>10.1f}")

    print(f"\n  Top 20 most variable genes (by CV, expressed):")
    print(f"    {'Gene':<20} {'CV':>7} {'Mean log2':>10} {'Std log2':>9} {'Mean CPM':>10} {'%zeros':>7}")
    print(f"    {'-'*68}")
    top_idx = np.argsort(np.nan_to_num(cvs, nan=-1))[::-1][:20]
    for idx in top_idx:
        pz = (expr_log[idx] == 0).sum() / n_samples * 100
        print(f"    {names_filt[idx]:<20} {cvs[idx]:>7.3f} {means[idx]:>10.3f} "
              f"{stds[idx]:>9.3f} {cpm_means[idx]:>10.2f} {pz:>6.1f}%")

    print(f"\n  Top 20 most stable genes (by CV, mean CPM > 100):")
    print(f"    {'Gene':<20} {'CV':>7} {'Mean log2':>10} {'Mean CPM':>10}")
    print(f"    {'-'*52}")
    stable_cvs = cvs.copy()
    stable_cvs[cpm_means <= 100] = np.inf
    stable_cvs[np.isnan(stable_cvs)] = np.inf
    stable_idx = np.argsort(stable_cvs)[:20]
    for idx in stable_idx:
        print(f"    {names_filt[idx]:<20} {cvs[idx]:>7.4f} {means[idx]:>10.3f} {cpm_means[idx]:>10.1f}")

    nonzero_cpm = expr_cpm[expr_cpm > 0]
    print(f"\n  CPM percentiles (non-zero values):")
    for p in [25, 50, 75, 90, 95, 99, 99.9]:
        print(f"    {p:>5}th: {np.percentile(nonzero_cpm, p):>10.2f} CPM")

    print(f"\n  Samples: {n_samples}")

    return {
        "means": means, "stds": stds, "cvs": cvs,
        "expressed": expressed, "cpm_means": cpm_means,
    }

# ══════════════════════════════════════════════════════════════════════
# 4. CLUSTERING & PER-CLUSTER CV
# ══════════════════════════════════════════════════════════════════════

def cluster_and_cv(expr_log, names_filt, n_samples):
    n_genes = expr_log.shape[0]
    gene_var = np.var(expr_log, axis=1)
    n_hvg = min(N_HVG, n_genes)
    hvg_idx = np.argsort(gene_var)[::-1][:n_hvg]
    expr_hvg = expr_log[hvg_idx]
    scaler = StandardScaler()
    expr_scaled = scaler.fit_transform(expr_hvg.T).T
    X = expr_scaled.T
    n_pca = min(N_PCA, min(X.shape) - 1)
    pca = PCA(n_components=n_pca, random_state=42)
    X_pca = pca.fit_transform(X)
    print(f"  PCA: {n_pca} components, {pca.explained_variance_ratio_.sum():.1%} variance")

    n_clust = min(N_CLUSTERS, n_samples // 5)
    km = KMeans(n_clusters=n_clust, n_init=20, random_state=42)
    labels = km.fit_predict(X_pca)

    means_all = expr_log.mean(axis=1)
    stds_all = expr_log.std(axis=1)
    expr_mask = means_all > 0.5
    cvs_all = stds_all[expr_mask] / means_all[expr_mask]

    print(f"\n  {'Cluster':>8} {'N':>5} {'Mean CV':>9} {'Median CV':>10}")
    print(f"  {'-'*38}")
    cluster_data = []
    for c in range(n_clust):
        mask = labels == c
        cexpr = expr_log[:, mask]
        cm = cexpr.mean(axis=1); cs = cexpr.std(axis=1)
        exp = cm > 0.5
        ccvs = cs[exp] / cm[exp]
        cluster_data.append({
            "n": int(mask.sum()), "mean_cv": float(ccvs.mean()),
            "median_cv": float(np.median(ccvs)), "cvs": ccvs,
        })
        print(f"  {c:>8} {mask.sum():>5} {ccvs.mean():>9.4f} {np.median(ccvs):>10.4f}")
    print(f"  {'WHOLE':>8} {n_samples:>5} {cvs_all.mean():>9.4f} {np.median(cvs_all):>10.4f}")

    avg_within = np.mean([d["mean_cv"] for d in cluster_data])
    reduction = (1 - avg_within / cvs_all.mean()) * 100
    print(f"\n  CV reduction from clustering: {reduction:.1f}%")
    return labels, X_pca, pca, cluster_data, cvs_all, n_clust

# ══════════════════════════════════════════════════════════════════════
# 5. LIBRARY SIZE STATS
# ══════════════════════════════════════════════════════════════════════

def library_stats(expr_raw):
    sums = expr_raw.sum(axis=0)
    print(f"\n  Library size (raw reads per sample):")
    print(f"    Mean:   {sums.mean():>14,.0f}")
    print(f"    Std:    {sums.std():>14,.0f}")
    print(f"    Min:    {sums.min():>14,.0f}")
    print(f"    Max:    {sums.max():>14,.0f}")
    print(f"    CV:     {sums.std()/sums.mean():>14.4f}")
    return sums

# ══════════════════════════════════════════════════════════════════════
# 6. RUN PIPELINE
# ══════════════════════════════════════════════════════════════════════

print("Loading metadata + counts...")
meta = parse_series_matrix(SERIES_MATRIX)
counts = load_counts(COUNTS_CSV)
print(f"  Counts matrix: {counts.shape[0]:,} genes x {counts.shape[1]} libraries")
print(f"  Stimulation counts: {dict(meta['stimulation'].value_counts())}")

per_stim = split_by_stimulation(counts, meta)
results = {}
for stim, (expr_raw, gene_ids, lib_ids) in per_stim.items():
    print(f"\n{'#'*70}\n  PROCESSING: {stim.upper()}\n{'#'*70}")
    print_overview(stim, expr_raw)
    lib_sums = library_stats(expr_raw)
    expr_filt, expr_cpm, expr_log, ids_filt = preprocess(expr_raw, gene_ids)
    stats = compute_variance_stats(expr_log, expr_cpm, ids_filt)
    labels, X_pca, pca, cluster_data, cvs_all, n_clust = cluster_and_cv(
        expr_log, ids_filt, expr_log.shape[1])
    results[stim] = {
        "expr_raw": expr_raw, "expr_log": expr_log, "expr_cpm": expr_cpm,
        "names_filt": ids_filt, "stats": stats,
        "labels": labels, "X_pca": X_pca, "pca": pca,
        "cluster_data": cluster_data, "cvs_all": cvs_all,
        "n_clust": n_clust, "lib_sums": lib_sums,
        "sample_ids": lib_ids, "n_samples": expr_log.shape[1],
    }

# ══════════════════════════════════════════════════════════════════════
# 7. COMPARISON SUMMARY
# ══════════════════════════════════════════════════════════════════════

print(f"\n{'#'*70}\n  STIMULATION COMPARISON\n{'#'*70}\n")
header = f"  {'Metric':<35}" + "".join(f" {s:>12}" for s in STIMULATIONS)
print(header)
print(f"  {'-' * (35 + 13 * len(STIMULATIONS))}")
for metric, fn in [
    ("Samples", lambda r: f"{r['n_samples']}"),
    ("Genes after filtering", lambda r: f"{r['expr_log'].shape[0]:,}"),
    ("% zeros (raw)", lambda r: f"{(r['expr_raw']==0).sum()/r['expr_raw'].size*100:.1f}%"),
    ("Mean library size", lambda r: f"{r['lib_sums'].mean()/1e6:.1f}M"),
    ("Whole-set mean CV", lambda r: f"{r['cvs_all'].mean():.4f}"),
    ("Whole-set median CV", lambda r: f"{np.median(r['cvs_all']):.4f}"),
    ("Avg within-cluster CV",
        lambda r: f"{np.mean([d['mean_cv'] for d in r['cluster_data']]):.4f}"),
    ("CV reduction from clustering",
        lambda r: f"{(1-np.mean([d['mean_cv'] for d in r['cluster_data']])/r['cvs_all'].mean())*100:.1f}%"),
]:
    vals = [fn(results[s]) for s in STIMULATIONS]
    print(f"  {metric:<35}" + "".join(f" {v:>12}" for v in vals))

# ══════════════════════════════════════════════════════════════════════
# 8. VISUALIZATIONS
# ══════════════════════════════════════════════════════════════════════

def make_stim_figure(stim, r, outpath):
    expr_log = r["expr_log"]; expr_cpm = r["expr_cpm"]; stats = r["stats"]
    labels = r["labels"]; X_pca = r["X_pca"]; pca = r["pca"]
    n_clust = r["n_clust"]; names = r["names_filt"]; n_samples = r["n_samples"]
    lib_sums = r["lib_sums"]
    means = stats["means"]; stds = stats["stds"]; cvs = stats["cvs"]
    expressed = stats["expressed"]; cpm_means = stats["cpm_means"]
    color = STIM_COLORS[stim]

    fig = plt.figure(figsize=(22, 28))
    gs = GridSpec(4, 2, figure=fig, hspace=0.35, wspace=0.25,
                  left=0.07, right=0.95, top=0.94, bottom=0.03)
    fig.suptitle(f"GSE279480 — {stim} Variance Analysis",
                 fontsize=20, fontweight="bold", color=TEXT, y=0.975)
    fig.text(0.5, 0.96,
             f"{expr_log.shape[0]:,} genes  ·  {n_samples} samples  ·  preprocessed log2(CPM+1)",
             ha="center", fontsize=13, color=MUTED)

    ax = fig.add_subplot(gs[0, 0])
    nonzero_cpm = expr_cpm[expr_cpm > 0]
    rng = np.random.default_rng(42)
    if len(nonzero_cpm) > 5_000_000:
        nonzero_cpm = nonzero_cpm[rng.choice(len(nonzero_cpm), 5_000_000, replace=False)]
    ax.hist(np.log10(nonzero_cpm), bins=150, color=color, alpha=0.85, edgecolor="none")
    ax.axvline(np.log10(np.median(nonzero_cpm)), color="#3fb950", ls="--", lw=2,
               label=f"Median = {np.median(nonzero_cpm):.1f} CPM")
    ax.set_xlabel("log₁₀(CPM)"); ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Non-Zero CPM Values", fontsize=14, fontweight="bold", pad=10)
    ax.legend(fontsize=10, facecolor=CARD, edgecolor=GRID, labelcolor=TEXT)
    ax.grid(axis="y", alpha=0.3)

    ax = fig.add_subplot(gs[0, 1])
    gm = means[expressed]; gv = stds[expressed]**2
    ax.scatter(gm, np.log10(gv), s=1.5, alpha=0.25, c=color, rasterized=True)
    ax.set_xlabel("Mean expression (log2 CPM+1)"); ax.set_ylabel("log₁₀(variance)")
    ax.set_title("Mean–Variance Relationship", fontsize=14, fontweight="bold", pad=10)
    ax.grid(alpha=0.3)
    top5 = np.argsort(np.nan_to_num(cvs, nan=-1))[::-1][:5]
    for i in top5:
        ax.annotate(names[i], (means[i], np.log10(stds[i]**2)),
                    fontsize=8, color="#f78166", fontweight="bold",
                    xytext=(5, 5), textcoords="offset points")

    ax = fig.add_subplot(gs[1, 0])
    valid_cvs = cvs[~np.isnan(cvs)]
    ax.hist(valid_cvs, bins=200, color="#d2a8ff", alpha=0.85, edgecolor="none", range=(0, 2.5))
    ax.axvline(np.median(valid_cvs), color="#f78166", ls="--", lw=2,
               label=f"Median CV = {np.median(valid_cvs):.3f}")
    ax.axvline(1.0, color="#3fb950", ls=":", lw=2, label="CV = 1")
    ax.set_xlabel("CV"); ax.set_ylabel("# genes")
    ax.set_title("Distribution of Per-Gene CV", fontsize=14, fontweight="bold", pad=10)
    ax.legend(fontsize=10, facecolor=CARD, edgecolor=GRID, labelcolor=TEXT)
    ax.grid(axis="y", alpha=0.3)

    ax = fig.add_subplot(gs[1, 1])
    lib_m = lib_sums / 1e6
    ax.hist(lib_m, bins=40, color="#3fb950", alpha=0.85, edgecolor="none")
    ax.axvline(np.median(lib_m), color="#f78166", ls="--", lw=2,
               label=f"Median = {np.median(lib_m):.1f}M")
    ax.set_xlabel("Total reads (millions)"); ax.set_ylabel("# samples")
    ax.set_title("Library Size Distribution", fontsize=14, fontweight="bold", pad=10)
    ax.legend(fontsize=10, facecolor=CARD, edgecolor=GRID, labelcolor=TEXT)
    ax.grid(axis="y", alpha=0.3)

    ax = fig.add_subplot(gs[2, 0])
    for c in range(n_clust):
        mask = labels == c
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], s=15, alpha=0.7,
                   c=PALETTE[c % len(PALETTE)], label=f"C{c} ({mask.sum()})",
                   edgecolors="none")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_title(f"PCA — {n_clust} Clusters", fontsize=14, fontweight="bold", pad=10)
    ax.legend(fontsize=7, facecolor=CARD, edgecolor=GRID, labelcolor=TEXT, ncol=2, markerscale=2)
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[2, 1])
    cluster_means = [d["mean_cv"] for d in r["cluster_data"]]
    cluster_medians = [d["median_cv"] for d in r["cluster_data"]]
    x = np.arange(n_clust + 1)
    bcolors = [PALETTE[i % len(PALETTE)] for i in range(n_clust)] + [TEXT]
    all_means = cluster_means + [r["cvs_all"].mean()]
    all_medians = cluster_medians + [np.median(r["cvs_all"])]
    xlabels = [f"C{i}" for i in range(n_clust)] + ["ALL"]
    ax.bar(x - 0.17, all_means, 0.32, color=bcolors, alpha=0.85, edgecolor="none", label="Mean CV")
    ax.bar(x + 0.17, all_medians, 0.32, color=bcolors, alpha=0.45, edgecolor="none", label="Median CV")
    ax.set_xticks(x); ax.set_xticklabels(xlabels, fontsize=9)
    ax.set_ylabel("CV")
    ax.set_title("Per-Cluster CV", fontsize=14, fontweight="bold", pad=10)
    ax.legend(fontsize=10, facecolor=CARD, edgecolor=GRID, labelcolor=TEXT)
    ax.grid(axis="y", alpha=0.3)

    ax = fig.add_subplot(gs[3, 0])
    pct_zeros = (expr_cpm == 0).sum(axis=1) / n_samples * 100
    ax.hist(pct_zeros, bins=100, color="#f0883e", alpha=0.85, edgecolor="none")
    ax.axvline(np.median(pct_zeros), color="#58a6ff", ls="--", lw=2,
               label=f"Median = {np.median(pct_zeros):.0f}%")
    ax.set_xlabel("% zero samples per gene"); ax.set_ylabel("# genes")
    ax.set_title("Gene Sparsity", fontsize=14, fontweight="bold", pad=10)
    ax.legend(fontsize=10, facecolor=CARD, edgecolor=GRID, labelcolor=TEXT)
    ax.grid(axis="y", alpha=0.3)

    ax = fig.add_subplot(gs[3, 1])
    top25 = np.argsort(cpm_means)[::-1][:25][::-1]
    ax.barh(range(25), cpm_means[top25] / 1e3, color=color, alpha=0.7, edgecolor="none", height=0.7)
    ax.set_yticks(range(25))
    ax.set_yticklabels([names[i] for i in top25], fontsize=8)
    ax.set_xlabel("Mean CPM (thousands)")
    ax.set_title("Top 25 Expressed Genes", fontsize=14, fontweight="bold", pad=10)
    ax8b = ax.twiny()
    ax8b.scatter([cvs[i] for i in top25], range(25), color="#f78166", s=40, zorder=5,
                 edgecolors="none")
    ax8b.set_xlabel("CV", color="#f78166")
    ax8b.tick_params(colors="#f78166")
    ax.grid(axis="x", alpha=0.3)

    plt.savefig(outpath, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Saved: {outpath}")


def make_comparison_figure(results, outpath):
    stims = list(results.keys())
    fig = plt.figure(figsize=(22, 18))
    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3,
                  left=0.07, right=0.95, top=0.93, bottom=0.05)
    fig.suptitle("GSE279480 — Stimulation Comparison",
                 fontsize=20, fontweight="bold", color=TEXT, y=0.975)

    ax = fig.add_subplot(gs[0, 0])
    for s in stims:
        cvs = results[s]["cvs_all"]
        ax.hist(cvs, bins=200, range=(0, 2.0), alpha=0.45, color=STIM_COLORS[s],
                label=f"{s} (median={np.median(cvs):.3f})", edgecolor="none", density=True)
    ax.set_xlabel("CV"); ax.set_ylabel("Density")
    ax.set_title("CV Distribution Comparison", fontsize=14, fontweight="bold", pad=10)
    ax.legend(fontsize=10, facecolor=CARD, edgecolor=GRID, labelcolor=TEXT)
    ax.grid(axis="y", alpha=0.3)

    ax = fig.add_subplot(gs[0, 1])
    for s in stims:
        lib_m = results[s]["lib_sums"] / 1e6
        ax.hist(lib_m, bins=40, alpha=0.45, color=STIM_COLORS[s],
                label=f"{s} (median={np.median(lib_m):.0f}M)", edgecolor="none", density=True)
    ax.set_xlabel("Total reads (millions)"); ax.set_ylabel("Density")
    ax.set_title("Library Size Comparison", fontsize=14, fontweight="bold", pad=10)
    ax.legend(fontsize=10, facecolor=CARD, edgecolor=GRID, labelcolor=TEXT)
    ax.grid(axis="y", alpha=0.3)

    ax = fig.add_subplot(gs[1, 0])
    for s in stims:
        cpm_nz = results[s]["expr_cpm"][results[s]["expr_cpm"] > 0]
        rng = np.random.default_rng(42)
        if len(cpm_nz) > 2_000_000:
            cpm_nz = cpm_nz[rng.choice(len(cpm_nz), 2_000_000, replace=False)]
        ax.hist(np.log10(cpm_nz), bins=150, alpha=0.45, color=STIM_COLORS[s],
                label=s, edgecolor="none", density=True)
    ax.set_xlabel("log₁₀(CPM)"); ax.set_ylabel("Density")
    ax.set_title("CPM Distribution Comparison", fontsize=14, fontweight="bold", pad=10)
    ax.legend(fontsize=10, facecolor=CARD, edgecolor=GRID, labelcolor=TEXT)
    ax.grid(axis="y", alpha=0.3)

    ax = fig.add_subplot(gs[1, 1])
    x = np.arange(len(stims))
    for i, s in enumerate(stims):
        whole_cv = results[s]["cvs_all"].mean()
        within_cv = np.mean([d["mean_cv"] for d in results[s]["cluster_data"]])
        ax.bar(i - 0.18, whole_cv, 0.32, color=STIM_COLORS[s], alpha=0.85,
               label=f"{s} whole" if i == 0 else None)
        ax.bar(i + 0.18, within_cv, 0.32, color=STIM_COLORS[s], alpha=0.45,
               label=f"{s} within" if i == 0 else None)
        ax.text(i - 0.18, whole_cv + 0.005, f"{whole_cv:.3f}", ha="center", fontsize=10, color=TEXT)
        ax.text(i + 0.18, within_cv + 0.005, f"{within_cv:.3f}", ha="center", fontsize=10, color=TEXT)
    ax.set_xticks(x); ax.set_xticklabels(stims)
    ax.set_ylabel("Mean CV")
    ax.set_title("Whole vs Within-Cluster CV", fontsize=14, fontweight="bold", pad=10)
    ax.legend(fontsize=9, facecolor=CARD, edgecolor=GRID, labelcolor=TEXT)
    ax.grid(axis="y", alpha=0.3)

    ax = fig.add_subplot(gs[2, 0])
    for s in stims:
        pz = (results[s]["expr_cpm"] == 0).sum(axis=1) / results[s]["n_samples"] * 100
        ax.hist(pz, bins=100, alpha=0.45, color=STIM_COLORS[s],
                label=f"{s} (median={np.median(pz):.0f}%)", edgecolor="none", density=True)
    ax.set_xlabel("% zero samples per gene"); ax.set_ylabel("Density")
    ax.set_title("Sparsity Comparison", fontsize=14, fontweight="bold", pad=10)
    ax.legend(fontsize=10, facecolor=CARD, edgecolor=GRID, labelcolor=TEXT)
    ax.grid(axis="y", alpha=0.3)

    ax = fig.add_subplot(gs[2, 1])
    a, b = PAIRWISE_CONTRAST
    a_idx = {n: i for i, n in enumerate(results[a]["names_filt"])}
    shared = []
    for i, n in enumerate(results[b]["names_filt"]):
        if n in a_idx:
            ai = a_idx[n]
            acv = results[a]["stats"]["cvs"][ai]
            bcv = results[b]["stats"]["cvs"][i]
            if not (np.isnan(acv) or np.isnan(bcv)):
                shared.append((acv, bcv, n))
    if shared:
        acvs, bcvs, snames = zip(*shared)
        acvs = np.array(acvs); bcvs = np.array(bcvs)
        ax.scatter(acvs, bcvs, s=3, alpha=0.3, c="#d2a8ff", rasterized=True)
        lim = max(acvs.max(), bcvs.max()) * 1.05
        ax.plot([0, lim], [0, lim], "--", color=MUTED, lw=1.5, label="y = x")
        ax.set_xlabel(f"{a} CV"); ax.set_ylabel(f"{b} CV")
        ax.set_title(f"Gene CV: {a} vs {b} ({len(shared):,} shared genes)",
                     fontsize=14, fontweight="bold", pad=10)
        diff = bcvs - acvs
        for idx in list(np.argsort(diff)[::-1][:5]) + list(np.argsort(diff)[:5]):
            ax.annotate(snames[idx], (acvs[idx], bcvs[idx]),
                        fontsize=7, color="#f78166", xytext=(3, 3), textcoords="offset points")
        corr = np.corrcoef(acvs, bcvs)[0, 1]
        ax.text(0.05, 0.95, f"r = {corr:.3f}", transform=ax.transAxes,
                fontsize=12, color=TEXT, va="top")
        ax.legend(fontsize=10, facecolor=CARD, edgecolor=GRID, labelcolor=TEXT)
    ax.grid(alpha=0.3)

    plt.savefig(outpath, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Saved: {outpath}")


print(f"\n{'#'*70}\n  GENERATING FIGURES\n{'#'*70}")
for stim in STIMULATIONS:
    slug = stim.lower().replace(" ", "_").replace(":", "")
    make_stim_figure(stim, results[stim], HERE / f"gse279480_{slug}_analysis.png")
make_comparison_figure(results, HERE / "gse279480_stimulation_comparison.png")
print("\nDone.")
