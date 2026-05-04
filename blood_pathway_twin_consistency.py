"""
Are CO-EXPRESSED gene groups served by the SAME twin donors?

The full-transcriptome version of this question gave a near-random
answer (mean overlap 0.016 vs random 0.013). That's expected — no donor
is a global twin of any holdout.

The right version: are there *gene subsets* — pathways or co-expression
modules — for which the same train donor wins multiple genes?

Tests:
  (A) Random pair baseline.
  (B) Highly correlated train-side gene pairs (top 1% of train Pearson).
  (C) A handful of biologically-defined pathways:
      - Ribosomal proteins (RPL*, RPS*)
      - MHC class I (HLA-A/B/C/E)
      - Hemoglobin (HBA1, HBA2, HBB)
      - Heat-shock (HSPA1A, HSPA1B, HSPB1, HSPH1, DNAJB1, BAG3)
      - Stress / immediate-early TFs (JUN, FOS, FOSB, EGR1, ATF3)
      - Inflammation (CXCL8, CCL3, CCL4, TNF, IL6)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

PROJECT = Path("/Users/rls/Desktop/programming-projects/single-cell/bulk-project")
GCT     = Path("/Users/rls/Downloads/gene_reads_v11_whole_blood.gct.gz")
OUT_DIR = PROJECT / "blood_technical_noise"

CPM_THRESHOLD   = 1.0
MIN_SAMPLE_FRAC = 0.10
HOLDOUT         = 50
K               = 10
SEED            = 0

print("Loading ...")
df = pd.read_csv(GCT, sep="\t", skiprows=2, compression="gzip")
gene_names = df["Description"].astype(str).values
counts = df.iloc[:, 2:].values.astype(np.float64)
n_g_raw, n_s = counts.shape
lib_all = counts.sum(axis=0)
cpm_all = counts / lib_all * 1e6
expressed = (cpm_all > CPM_THRESHOLD).sum(axis=1) >= int(MIN_SAMPLE_FRAC * n_s)
counts = counts[expressed]
gene_names = gene_names[expressed]
n_genes = counts.shape[0]

rng = np.random.default_rng(SEED)
perm = rng.permutation(n_s)
counts = counts[:, perm]
counts_train = counts[:, :n_s - HOLDOUT]
counts_test  = counts[:, n_s - HOLDOUT:]
lib_train = counts_train.sum(axis=0)
lib_test  = counts_test.sum(axis=0)
n_train   = counts_train.shape[1]
ref_lib   = float(np.median(lib_train))

counts_train_scaled = counts_train * (ref_lib / lib_train)[None, :]
counts_test_scaled  = counts_test  * (ref_lib / lib_test)[None, :]
log_train = np.log2(counts_train_scaled + 1)
log_test  = np.log2(counts_test_scaled  + 1)

# Build per-(gene, holdout) K-NN
print("Building K-NN tables ...")
nn_idx = np.zeros((n_genes, HOLDOUT, K), dtype=np.int32)
for g in range(n_genes):
    d = np.abs(log_train[g][None, :] - log_test[g][:, None])
    nn_idx[g] = np.argpartition(d, K, axis=1)[:, :K]


def overlap_pair(g1, g2):
    """Mean overlap between K-NN sets of g1 and g2 across the 50 holdouts."""
    overlaps = []
    for h in range(HOLDOUT):
        s1 = set(nn_idx[g1, h]); s2 = set(nn_idx[g2, h])
        overlaps.append(len(s1 & s2) / K)
    return np.mean(overlaps)


def overlap_set(gene_indices):
    """Mean overlap across all gene-pairs in a set, averaged over holdouts."""
    overlaps = []
    n = len(gene_indices)
    for i in range(n):
        for j in range(i + 1, n):
            overlaps.append(overlap_pair(gene_indices[i], gene_indices[j]))
    return overlaps


def name_to_idx(names):
    out = []
    for n in names:
        idx = np.where(gene_names == n)[0]
        if len(idx):
            out.append(int(idx[0]))
    return out


# ── (A) Random baseline ──────────────────────────────────────────────
print("\nRandom-pair baseline ...")
n_random = 2000
random_pairs = rng.integers(0, n_genes, size=(n_random, 2))
rand_overlaps = []
for a, b in random_pairs:
    if a == b: continue
    rand_overlaps.append(overlap_pair(a, b))
rand_overlaps = np.array(rand_overlaps)
print(f"  random pairs (n={len(rand_overlaps)}): mean={rand_overlaps.mean():.4f}, "
      f"median={np.median(rand_overlaps):.4f}, "
      f"theoretical={K/n_train:.4f}")


# ── (B) High-correlation pairs ───────────────────────────────────────
# Compute correlation-based "near twin" pairs on the train set.
# 16k x 16k is 256M entries — too much. Subsample.
print("\nHigh-correlation pair test (subsample) ...")
sub_idx = rng.choice(n_genes, size=4000, replace=False)
log_train_sub = log_train[sub_idx]
# z-score
log_train_z = (log_train_sub - log_train_sub.mean(axis=1, keepdims=True))
sds = log_train_z.std(axis=1, keepdims=True)
sds[sds == 0] = 1
log_train_z = log_train_z / sds
corr = log_train_z @ log_train_z.T / log_train_z.shape[1]    # 4000 × 4000
# Take top 1% positive correlation (excluding diagonal)
np.fill_diagonal(corr, -1.0)
flat = corr.ravel()
thresh = np.quantile(flat, 0.99)
i, j = np.where(corr >= thresh)
mask = i < j
i, j = i[mask], j[mask]
n_use = min(2000, len(i))
keep = rng.choice(len(i), size=n_use, replace=False)
i, j = i[keep], j[keep]
high_corr_overlaps = []
for a, b in zip(sub_idx[i], sub_idx[j]):
    high_corr_overlaps.append(overlap_pair(int(a), int(b)))
high_corr_overlaps = np.array(high_corr_overlaps)
print(f"  high-corr pairs (n={len(high_corr_overlaps)}): "
      f"mean={high_corr_overlaps.mean():.4f}, "
      f"median={np.median(high_corr_overlaps):.4f}  "
      f"({high_corr_overlaps.mean() / rand_overlaps.mean():.1f}x random)")


# ── (C) Pathway-defined gene sets ─────────────────────────────────────
PATHWAYS = {
    "Ribosomal_RPL_RPS": [g for g in gene_names
                          if g.startswith("RPL") or g.startswith("RPS")],
    "MHC_class_I":       ["HLA-A", "HLA-B", "HLA-C", "HLA-E", "HLA-F", "B2M"],
    "Hemoglobin":        ["HBA1", "HBA2", "HBB", "HBD"],
    "Heat_shock":        ["HSPA1A", "HSPA1B", "HSPA8", "HSPB1", "HSPH1",
                          "DNAJB1", "BAG3", "HSP90AA1", "HSP90AB1"],
    "Stress_IEG":        ["JUN", "FOS", "FOSB", "JUNB", "EGR1", "ATF3",
                          "DDIT4", "PPP1R15A", "MAFF", "BHLHE40"],
    "Inflammation":      ["CXCL8", "CCL3", "CCL4", "TNF", "IL6", "IL1B",
                          "CXCR4", "CCL3L1", "CCL4L2"],
    "Mitochondrial_OXPHOS": [g for g in gene_names if g.startswith("MT-")
                             or g.startswith("NDUF") or g.startswith("COX")
                             or g.startswith("ATP5") or g.startswith("UQCR")],
    "Bimodal_state_anchor": ["DDIT4", "FRAT1", "JUN", "BHLHE40", "G0S2",
                             "VEGFA", "HILPDA", "PLIN2", "HBEGF",
                             "PTAFR", "MED18", "RER1"],
    "Immunoglobulin_V": [g for g in gene_names
                         if g.startswith("IGHV") or g.startswith("IGKV")
                         or g.startswith("IGLV")],
    "TCR_Vsegments":  [g for g in gene_names if g.startswith("TR")
                       and (("V" in g and not g.startswith("TRIM"))
                            or g.startswith("TRA") or g.startswith("TRB")
                            or g.startswith("TRG") or g.startswith("TRD"))],
}


print("\nPathway-defined gene sets:")
print(f"  {'pathway':>26}  {'n_genes':>8}  {'n_pairs':>8}  "
      f"{'mean_ovl':>9}  {'median':>7}  {'×random':>7}")
print("  " + "-" * 80)
results = []
for name, genes in PATHWAYS.items():
    if isinstance(genes[0], str):
        idxs = name_to_idx(genes)
    else:
        idxs = genes
    if len(idxs) < 3:
        print(f"  {name:>26}  {len(idxs):>8}  too few"); continue
    # Subsample large pathways to keep pairs manageable
    if len(idxs) > 60:
        idxs_sample = list(rng.choice(idxs, size=60, replace=False))
    else:
        idxs_sample = idxs
    overlaps = overlap_set(idxs_sample)
    n_pairs = len(overlaps)
    if n_pairs == 0:
        continue
    mean_ovl = float(np.mean(overlaps))
    median_ovl = float(np.median(overlaps))
    print(f"  {name:>26}  {len(idxs):>8}  {n_pairs:>8}  "
          f"{mean_ovl:>9.4f}  {median_ovl:>7.4f}  "
          f"{mean_ovl / rand_overlaps.mean():>6.1f}x")
    results.append({
        "pathway":         name,
        "n_genes":         len(idxs),
        "n_pairs":         n_pairs,
        "mean_overlap":    mean_ovl,
        "median_overlap":  median_ovl,
        "fold_over_random": mean_ovl / rand_overlaps.mean(),
    })
print(f"\n  {'random':>26}  {'-':>8}  {len(rand_overlaps):>8}  "
      f"{rand_overlaps.mean():>9.4f}  {np.median(rand_overlaps):>7.4f}  1.0x")

pd.DataFrame(results).to_csv(OUT_DIR / "pathway_twin_consistency.csv", index=False)


# ── Plot ─────────────────────────────────────────────────────────────
BG, CARD, TEXT, MUTED, GRID = '#0e1117', '#1a1d23', '#e6edf3', '#7d8590', '#21262d'
plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': CARD, 'axes.edgecolor': GRID,
    'axes.labelcolor': TEXT, 'text.color': TEXT, 'xtick.color': MUTED,
    'ytick.color': MUTED, 'grid.color': GRID, 'grid.alpha': 0.5,
    'font.family': 'sans-serif', 'font.size': 11,
})

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Pathway-level twin consistency  (K=10 nearest train donors per gene)",
             color=TEXT, fontsize=13)

ax = axes[0, 0]
bins = np.linspace(0, 1, 41)
ax.hist(rand_overlaps,    bins=bins, alpha=0.55, color="#58a6ff",
        label=f"random (n={len(rand_overlaps)})", edgecolor="#1a1d23")
ax.hist(high_corr_overlaps, bins=bins, alpha=0.55, color="#3fb950",
        label=f"high-corr (top 1%)", edgecolor="#1a1d23")
ax.axvline(K / n_train, color="white", ls=":", lw=1, alpha=0.6,
           label=f"theoretical random {K/n_train:.4f}")
ax.set_xlabel("pairwise NN-set overlap")
ax.set_ylabel("# pairs")
ax.set_title("Random pairs vs high-correlation pairs")
ax.legend(facecolor=CARD, edgecolor=GRID, labelcolor=TEXT, fontsize=9)

ax = axes[0, 1]
res_df = pd.DataFrame(results).sort_values("mean_overlap", ascending=False)
ax.barh(np.arange(len(res_df))[::-1], res_df["mean_overlap"][::-1],
        color="#3fb950")
ax.axvline(rand_overlaps.mean(), color="#f78166", ls="--", lw=1.5,
           label=f"random {rand_overlaps.mean():.4f}")
ax.set_yticks(np.arange(len(res_df))[::-1])
ax.set_yticklabels(res_df["pathway"][::-1], fontsize=9)
ax.set_xlabel("mean pairwise NN-overlap")
ax.set_title("Pathway-level twin consistency")
ax.legend(facecolor=CARD, edgecolor=GRID, labelcolor=TEXT, fontsize=9)

ax = axes[1, 0]
labels = ["random"] + res_df["pathway"].tolist() + ["high-corr (top 1%)"]
values = [rand_overlaps.mean()] + res_df["mean_overlap"].tolist() + [high_corr_overlaps.mean()]
xs = np.arange(len(values))
colors = (["#58a6ff"] + ["#3fb950"] * len(res_df) + ["#f78166"])
ax.bar(xs, values, color=colors)
ax.axhline(K / n_train, color="white", ls=":", lw=1, alpha=0.5)
ax.set_xticks(xs)
ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=9)
ax.set_ylabel("mean pairwise NN-overlap")
ax.set_title("Mean overlap by gene-set")

ax = axes[1, 1]
fold = res_df["fold_over_random"]
ax.barh(np.arange(len(res_df))[::-1], fold[::-1], color="#d2a8ff")
ax.axvline(1.0, color="#f78166", ls="--", lw=1.5)
ax.set_yticks(np.arange(len(res_df))[::-1])
ax.set_yticklabels(res_df["pathway"][::-1], fontsize=9)
ax.set_xlabel("fold over random baseline")
ax.set_title("Pathway/random fold")

fig.tight_layout()
out_png = OUT_DIR / "blood_pathway_twin_consistency.png"
fig.savefig(out_png, dpi=150, facecolor=BG, bbox_inches="tight")
plt.close(fig)
print(f"\nWrote {out_png}")
print(f"Wrote {OUT_DIR/'pathway_twin_consistency.csv'}")
