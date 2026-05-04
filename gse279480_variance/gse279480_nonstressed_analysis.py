"""
GSE279480 (Smithmyer 2025, Null/unstimulated only): bimodal-state stress
analysis, mirroring gtex_nonstressed_analysis.py.

  1. Filter, CPM, log2(CPM+1) on Null libraries.
  2. Find bimodal genes (mean>1, std>0.3) via KDE + peak detection.
  3. Build binary state matrix using KDE peak midpoints as thresholds.
  4. PCA on binary matrix -> split samples into 2 states.
  5. Identify "stressed" state via DDIT4/JUN/VEGFA marker scores
     (Ensembl IDs resolved from ensembl_to_symbol.tsv).
  6. Drop stressed samples, re-find bimodal genes, re-cluster non-stressed
     by silhouette-tuned KMeans, identify anchor genes, characterize states.
  7. 4-panel figure: scree, PCA, binary heatmap, gene-gene corr.
"""

from collections import Counter
from pathlib import Path
import gzip
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import scipy.cluster.hierarchy as sch
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")

# ── Config ──
HERE = Path(__file__).parent
COUNTS_CSV = HERE.parent / "data/GSE279480/GSE279480_P441_genecounts.csv.gz"
SERIES_MATRIX = HERE.parent / "data/GSE279480/GSE279480_series_matrix.txt.gz"
SYMBOL_MAP = HERE / "ensembl_to_symbol.tsv"
STIMULATION = "Null"
CPM_THRESHOLD = 1
MIN_SAMPLE_FRAC = 0.1

# ── Theme ──
BG = "#0e1117"; CARD = "#1a1d23"; TEXT = "#e6edf3"; MUTED = "#7d8590"; GRID = "#21262d"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": CARD, "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT, "text.color": TEXT, "xtick.color": MUTED,
    "ytick.color": MUTED, "grid.color": GRID, "legend.facecolor": CARD,
    "legend.edgecolor": GRID, "font.size": 10,
})

# ── 1. Load Null counts + metadata ──
print("=" * 70)
print(f"LOADING GSE279480 ({STIMULATION} libraries only)...")

rows = {}
with gzip.open(SERIES_MATRIX, "rt") as fh:
    for line in fh:
        if line.startswith("!series_matrix_table_begin"):
            break
        if not line.startswith("!Sample_"):
            continue
        parts = line.rstrip("\n").split("\t")
        rows.setdefault(parts[0], []).append([p.strip('"') for p in parts[1:]])
meta = pd.DataFrame({"lib": rows["!Sample_description"][0]})
for r in rows.get("!Sample_characteristics_ch1", []):
    keys = [c.split(":", 1)[0].strip() for c in r if ":" in c]
    if not keys:
        continue
    key = Counter(keys).most_common(1)[0][0]
    meta[key] = [c.split(":", 1)[1].strip() if ":" in c else "" for c in r]

null_libs = meta.loc[meta["stimulation"] == STIMULATION, "lib"].tolist()
counts = pd.read_csv(COUNTS_CSV, index_col=0)
null_libs = [l for l in null_libs if l in counts.columns]
expr = counts[null_libs].values.astype(np.float64)
gene_ids = np.array(counts.index)
print(f"  Loaded: {expr.shape[0]:,} genes x {expr.shape[1]} {STIMULATION} samples")

# ── 2. Filter + log2 CPM ──
print("Filtering low-expression genes + CPM normalize...")
lib_sizes = expr.sum(axis=0)
cpm = expr / lib_sizes * 1e6
keep = (cpm > CPM_THRESHOLD).sum(axis=1) >= int(MIN_SAMPLE_FRAC * expr.shape[1])
expr_f = expr[keep]
ids_f = gene_ids[keep]
log_cpm = np.log2(expr_f / expr_f.sum(axis=0) * 1e6 + 1)
n_genes, n_samples = log_cpm.shape
print(f"  Kept {n_genes:,} / {expr.shape[0]:,} genes")

sym_df = pd.read_csv(SYMBOL_MAP, sep="\t").drop_duplicates("ensembl_id")
ens_to_sym = dict(zip(sym_df["ensembl_id"], sym_df["symbol"]))
sym_to_ens = {v: k for k, v in ens_to_sym.items() if isinstance(v, str)}
gene_names = np.array([ens_to_sym.get(g, g) for g in ids_f])

# ── 3. Find bimodal genes (full Null cohort) ──
print("Finding bimodal genes in FULL Null cohort...")
gm, gs = log_cpm.mean(axis=1), log_cpm.std(axis=1)
candidate_mask = (gm > 1) & (gs > 0.3)
candidate_idx = np.where(candidate_mask)[0]
print(f"  Candidates (mean>1, std>0.3): {len(candidate_idx):,}")


def find_bimodal(data, indices):
    out = []
    for i in indices:
        vals = data[i]
        try:
            kde = gaussian_kde(vals, bw_method="scott")
            x_grid = np.linspace(vals.min(), vals.max(), 500)
            density = kde(x_grid)
            peaks, _ = find_peaks(density, prominence=density.max() * 0.08, distance=40)
            if len(peaks) >= 2:
                out.append(i)
        except Exception:
            continue
    return np.array(out)


bimodal_full = find_bimodal(log_cpm, candidate_idx)
print(f"  Bimodal genes (full): {len(bimodal_full)}")


def binary_matrix(data, bimodal_idx):
    bm = np.zeros((len(bimodal_idx), data.shape[1]), dtype=np.int8)
    thresholds = []
    for j, gi in enumerate(bimodal_idx):
        vals = data[gi]
        kde = gaussian_kde(vals, bw_method="scott")
        x_grid = np.linspace(vals.min(), vals.max(), 500)
        density = kde(x_grid)
        peaks, _ = find_peaks(density, prominence=density.max() * 0.08, distance=40)
        if len(peaks) >= 2:
            sp = np.sort(x_grid[peaks])
            thr = (sp[0] + sp[1]) / 2
        else:
            thr = np.median(vals)
        thresholds.append(thr)
        bm[j] = (vals > thr).astype(np.int8)
    return bm, thresholds


bin_full, thr_full = binary_matrix(log_cpm, bimodal_full)
pca_full = PCA(n_components=2)
scores_full = pca_full.fit_transform(bin_full.T)
state_labels = (scores_full[:, 0] > 0).astype(int)
n0, n1 = int((state_labels == 0).sum()), int((state_labels == 1).sum())
print(f"  State 0: {n0} samples, State 1: {n1} samples")

# ── 5. Stress-marker scoring ──
stress_markers = ["DDIT4", "JUN", "VEGFA"]
stress_scores = {}
for s in (0, 1):
    total = 0.0
    for sym in stress_markers:
        ens = sym_to_ens.get(sym)
        if ens is None:
            continue
        rows_idx = np.where(ids_f == ens)[0]
        if len(rows_idx) == 0:
            continue
        total += log_cpm[rows_idx[0], state_labels == s].mean()
    stress_scores[s] = total
stressed_state = max(stress_scores, key=stress_scores.get)
nonstressed_state = 1 - stressed_state
print(f"\n  Stress-marker scores: State 0 = {stress_scores[0]:.2f}, State 1 = {stress_scores[1]:.2f}")
print(f"  -> Stressed state: {stressed_state}")

# ── 6. Drop stressed, re-analyze ──
keep_mask = state_labels == nonstressed_state
log_cpm_ns = log_cpm[:, keep_mask]
n_ns = log_cpm_ns.shape[1]
print(f"\n  Non-stressed samples retained: {n_ns} / {n_samples}")

print("\n" + "=" * 70)
print("RE-ANALYZING bimodal genes in NON-STRESSED samples...")
gm_ns, gs_ns = log_cpm_ns.mean(axis=1), log_cpm_ns.std(axis=1)
cand_ns = np.where((gm_ns > 1) & (gs_ns > 0.3))[0]
print(f"  Candidates (mean>1, std>0.3): {len(cand_ns):,}")
bimodal_ns = find_bimodal(log_cpm_ns, cand_ns)
bimodal_names_ns = gene_names[bimodal_ns]
print(f"  Bimodal genes in non-stressed: {len(bimodal_ns)}")

bin_ns, thr_ns = binary_matrix(log_cpm_ns, bimodal_ns)
n_comp = min(10, bin_ns.shape[0], bin_ns.shape[1])
pca_ns = PCA(n_components=n_comp)
scores_ns = pca_ns.fit_transform(bin_ns.T)
var_explained = pca_ns.explained_variance_ratio_
print("\n  PCA variance explained (non-stressed):")
for i in range(min(5, n_comp)):
    print(f"    PC{i+1}: {var_explained[i]*100:.1f}%")

# Best K via silhouette
best_k, best_sil = 2, -1
for k in range(2, min(6, n_ns)):
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    lbl = km.fit_predict(scores_ns[:, :3])
    sil = silhouette_score(scores_ns[:, :3], lbl)
    if sil > best_sil:
        best_sil, best_k = sil, k
print(f"\n  Best K (silhouette): K={best_k} (silhouette={best_sil:.3f})")
km_final = KMeans(n_clusters=best_k, n_init=10, random_state=42)
cluster_ns = km_final.fit_predict(scores_ns[:, :3])
for c in range(best_k):
    print(f"    Cluster {c}: {(cluster_ns == c).sum()} samples")

# Anchor genes
print("\n  ANCHOR GENES (top differentially expressed between non-stressed clusters):")
pairs = [(0, 1)] if best_k == 2 else [(i, j) for i in range(best_k) for j in range(i + 1, best_k)]
for c1, c2 in pairs:
    m1 = log_cpm_ns[:, cluster_ns == c1].mean(axis=1)
    m2 = log_cpm_ns[:, cluster_ns == c2].mean(axis=1)
    diff = m1 - m2
    top_idx = np.argsort(np.abs(diff))[::-1][:20]
    print(f"\n  --- Cluster {c1} vs Cluster {c2} ---")
    print(f"  {'Gene':<18} {'MeanC'+str(c1):>10} {'MeanC'+str(c2):>10} {'Diff':>10}")
    print(f"  {'-' * 50}")
    for idx in top_idx:
        print(f"  {gene_names[idx]:<18} {m1[idx]:>10.2f} {m2[idx]:>10.2f} {diff[idx]:>10.2f}")

# State characterization
print("\n\n  STATE CHARACTERIZATION:")
for c in range(best_k):
    mask_c = cluster_ns == c
    mc = log_cpm_ns[:, mask_c].mean(axis=1)
    mo = log_cpm_ns[:, ~mask_c].mean(axis=1)
    diff_c = mc - mo
    top_up = np.argsort(diff_c)[::-1][:10]
    top_dn = np.argsort(diff_c)[:10]
    print(f"\n  Cluster {c} ({mask_c.sum()} samples):")
    print("    Top UPREGULATED:")
    for idx in top_up:
        print(f"      {gene_names[idx]:<18} diff={diff_c[idx]:+.3f}  mean={mc[idx]:.2f}")
    print("    Top DOWNREGULATED:")
    for idx in top_dn:
        print(f"      {gene_names[idx]:<18} diff={diff_c[idx]:+.3f}  mean={mc[idx]:.2f}")

# Bimodal overlap
overlap = set(gene_names[bimodal_full]) & set(bimodal_names_ns)
only_new = set(bimodal_names_ns) - set(gene_names[bimodal_full])
lost = set(gene_names[bimodal_full]) - set(bimodal_names_ns)
print(f"\n\n  BIMODAL GENE OVERLAP:")
print(f"    Full cohort bimodal:           {len(bimodal_full)}")
print(f"    Non-stressed bimodal:          {len(bimodal_ns)}")
print(f"    Overlap:                       {len(overlap)}")
print(f"    Lost after stress removal:     {len(lost)}")
print(f"    New (only in non-stressed):    {len(only_new)}")

# ── 8. Figure ──
print("\n" + "=" * 70)
print("Creating 4-panel figure...")

fig, axes = plt.subplots(2, 2, figsize=(16, 14))
fig.suptitle(f"GSE279480 Null — Non-Stressed Sample Re-Analysis",
             fontsize=16, fontweight="bold", color=TEXT, y=0.98)

# Panel 1: scree
ax = axes[0, 0]
pcs = np.arange(1, len(var_explained) + 1)
ax.bar(pcs, var_explained * 100, color="#58a6ff", alpha=0.8)
ax.plot(pcs, np.cumsum(var_explained) * 100, "o-", color="#f78166", markersize=5)
ax.set_xlabel("Principal Component"); ax.set_ylabel("Variance Explained (%)")
ax.set_title("PCA Scree (Non-Stressed Binary Matrix)", color=TEXT, fontsize=12)
ax.set_xticks(pcs); ax.grid(True, alpha=0.3)
for i in range(min(3, len(var_explained))):
    ax.annotate(f"{np.cumsum(var_explained)[i]*100:.1f}%",
                (pcs[i], np.cumsum(var_explained)[i] * 100 + 1.5),
                color="#f78166", fontsize=8, ha="center")

# Panel 2: PCA scatter
ax = axes[0, 1]
cluster_colors = ["#58a6ff", "#f78166", "#3fb950", "#d2a8ff", "#f0883e"]
for c in range(best_k):
    mask_c = cluster_ns == c
    ax.scatter(scores_ns[mask_c, 0], scores_ns[mask_c, 1],
               c=cluster_colors[c % len(cluster_colors)],
               s=15, alpha=0.6, label=f"Cluster {c} (n={mask_c.sum()})")
ax.set_xlabel(f"PC1 ({var_explained[0]*100:.1f}%)")
ax.set_ylabel(f"PC2 ({var_explained[1]*100:.1f}%)")
ax.set_title("PCA of Non-Stressed Samples", color=TEXT, fontsize=12)
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# Panel 3: binary heatmap
ax = axes[1, 0]
max_genes_plot = 80
if bin_ns.shape[0] > max_genes_plot:
    bv = bin_ns.astype(float).var(axis=1)
    top_v = np.argsort(bv)[::-1][:max_genes_plot]
    binary_plot = bin_ns[top_v]
else:
    binary_plot = bin_ns
try:
    row_link = sch.linkage(binary_plot, method="ward", metric="euclidean")
    row_order = sch.dendrogram(row_link, no_plot=True)["leaves"]
except Exception:
    row_order = np.arange(binary_plot.shape[0])
sample_order = np.argsort(cluster_ns * 1000 + scores_ns[:, 0])
binary_ord = binary_plot[row_order][:, sample_order]
cmap_bin = LinearSegmentedColormap.from_list("binary_dark", [CARD, "#58a6ff"])
ax.imshow(binary_ord, aspect="auto", cmap=cmap_bin, interpolation="nearest")
ax.set_xlabel(f"Samples (n={n_ns})"); ax.set_ylabel(f"Bimodal Genes (top {binary_plot.shape[0]})")
ax.set_title("Binary State Heatmap (Clustered)", color=TEXT, fontsize=12)
ax.set_xticks([]); ax.set_yticks([])
for i, idx in enumerate(sample_order):
    col = cluster_colors[cluster_ns[idx] % len(cluster_colors)]
    ax.plot(i, -0.5, "s", color=col, markersize=1.5, clip_on=False)

# Panel 4: gene-gene correlation
ax = axes[1, 1]
n_corr = min(50, len(bimodal_ns))
bv_all = bin_ns.astype(float).var(axis=1)
top_corr_idx = np.argsort(bv_all)[::-1][:n_corr]
corr = np.corrcoef(log_cpm_ns[bimodal_ns[top_corr_idx]])
try:
    link_c = sch.linkage(corr, method="ward")
    corr_order = sch.dendrogram(link_c, no_plot=True)["leaves"]
except Exception:
    corr_order = np.arange(n_corr)
corr_ord = corr[corr_order][:, corr_order]
cmap_corr = LinearSegmentedColormap.from_list("corr_dark", ["#f78166", CARD, "#58a6ff"])
im = ax.imshow(corr_ord, aspect="auto", cmap=cmap_corr, vmin=-1, vmax=1)
ax.set_title(f"Gene-Gene Correlation (top {n_corr} bimodal)", color=TEXT, fontsize=12)
ax.set_xticks([]); ax.set_yticks([])
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson r")

plt.tight_layout(rect=[0, 0, 1, 0.96])
out = HERE / "gse279480_null_nonstressed_states.png"
plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=BG)
plt.close()
print(f"  Saved: {out}")

# ── Summary ──
print("\n" + "=" * 70)
print("DETAILED SUMMARY")
print("=" * 70)
print(f"\n1. FULL NULL COHORT:")
print(f"   - {expr.shape[0]:,} genes x {n_samples} samples")
print(f"   - {len(bimodal_full)} bimodal genes")
print(f"   - PCA split: State 0 ({n0}), State 1 ({n1})")
print(f"   - Stressed state = {stressed_state} (higher DDIT4/JUN/VEGFA)")

print(f"\n2. NON-STRESSED SUBSET:")
print(f"   - {n_ns} samples (removed {n_samples - n_ns} stressed)")
print(f"   - {len(bimodal_ns)} bimodal genes")
print(f"   - {len(lost)} bimodal genes lost; {len(only_new)} new")

print(f"\n3. PCA OF NON-STRESSED:")
print(f"   - PC1: {var_explained[0]*100:.1f}%   PC2: {var_explained[1]*100:.1f}%")
print(f"   - Best K = {best_k} (silhouette={best_sil:.3f})")
clear = best_sil > 0.2 and var_explained[0] > 0.05
print(f"   - Clear states? {'YES' if clear else 'Weak/Unclear'}")

print(f"\n4. NEW STATE CLUSTERS:")
for c in range(best_k):
    n_c = int((cluster_ns == c).sum())
    print(f"   Cluster {c}: {n_c} samples ({n_c/n_ns*100:.1f}%)")

# Save cluster assignments for downstream work
out_csv = HERE / "gse279480_null_nonstressed_clusters.csv"
pd.DataFrame({
    "lib": [null_libs[i] for i in np.where(keep_mask)[0]],
    "cluster": cluster_ns,
}).to_csv(out_csv, index=False)
print(f"\nSaved cluster assignments: {out_csv}")

# Save bimodal gene lists
pd.DataFrame({"ensembl_id": ids_f[bimodal_full], "symbol": gene_names[bimodal_full]}
).to_csv(HERE / "gse279480_null_bimodal_genes_full.csv", index=False)
pd.DataFrame({"ensembl_id": ids_f[bimodal_ns], "symbol": gene_names[bimodal_ns]}
).to_csv(HERE / "gse279480_null_bimodal_genes_nonstressed.csv", index=False)
print("Done.")
