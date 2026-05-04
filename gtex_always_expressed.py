#!/usr/bin/env python3
"""
Count genes that are "always expressed" in the GTEx whole blood bulk dataset,
restricted to non-stressed samples.

"Always expressed" = gene has log2(CPM+1) above a threshold in EVERY non-stressed sample.
Reports counts across a range of thresholds so we can locate the ~340 figure.
"""

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

GTEX = '/Users/rls/Downloads/gene_reads_v11_whole_blood.gct.gz'

print("Loading GTEx whole blood...")
df = pd.read_csv(GTEX, sep='\t', skiprows=2, compression='gzip')
gene_names = df['Description'].values
expr = df.iloc[:, 2:].values.astype(np.float64)
print(f"  {expr.shape[0]} genes x {expr.shape[1]} samples")

# log2(CPM+1)
lib = expr.sum(axis=0, keepdims=True)
log_cpm = np.log2(expr / lib * 1e6 + 1)

# --- Identify bimodal genes (same criteria as gtex_nonstressed_analysis.py) ---
print("Finding bimodal genes for stress-state PCA...")
gm, gs = log_cpm.mean(axis=1), log_cpm.std(axis=1)
cand = np.where((gm > 1) & (gs > 0.3))[0]

def bimodal_indices(data, idx):
    out = []
    for i in idx:
        v = data[i]
        try:
            kde = gaussian_kde(v, bw_method='scott')
            xg = np.linspace(v.min(), v.max(), 500)
            d = kde(xg)
            peaks, _ = find_peaks(d, prominence=d.max() * 0.08, distance=40)
            if len(peaks) >= 2:
                out.append(i)
        except Exception:
            continue
    return np.array(out)

bi = bimodal_indices(log_cpm, cand)
print(f"  Bimodal genes: {len(bi)}")

# Binary matrix -> PCA -> split samples into two states
binmat = np.zeros((len(bi), log_cpm.shape[1]), dtype=np.int8)
for j, gi in enumerate(bi):
    v = log_cpm[gi]
    kde = gaussian_kde(v, bw_method='scott')
    xg = np.linspace(v.min(), v.max(), 500)
    d = kde(xg)
    peaks, _ = find_peaks(d, prominence=d.max() * 0.08, distance=40)
    if len(peaks) >= 2:
        sp = np.sort(xg[peaks])
        thr = (sp[0] + sp[1]) / 2
    else:
        thr = np.median(v)
    binmat[j] = (v > thr).astype(np.int8)

scores = PCA(n_components=2).fit_transform(binmat.T)
state = (scores[:, 0] > 0).astype(int)

# Figure out which state is stressed (higher DDIT4 / JUN / VEGFA)
stress_markers = ['DDIT4', 'JUN', 'VEGFA']
sm_score = {}
for s in (0, 1):
    total = 0.0
    for g in stress_markers:
        idx = np.where(gene_names == g)[0]
        if len(idx):
            total += log_cpm[idx[0], state == s].mean()
    sm_score[s] = total
stressed = max(sm_score, key=sm_score.get)
nonstressed = 1 - stressed
keep = state == nonstressed
log_cpm_ns = log_cpm[:, keep]
n_ns = log_cpm_ns.shape[1]
print(f"  Stressed state = {stressed} (score {sm_score[stressed]:.2f} vs "
      f"{sm_score[nonstressed]:.2f})")
print(f"  Non-stressed samples kept: {n_ns} / {log_cpm.shape[1]}")

# --- Always-expressed counts at different thresholds ---
print("\nGenes always expressed (min log2(CPM+1) across all non-stressed samples "
      "exceeds threshold):")
print(f"  {'threshold':>10}  {'# genes':>8}")
gene_min = log_cpm_ns.min(axis=1)
for thr in np.arange(0.0, 10.25, 0.25):
    n = int((gene_min > thr).sum())
    print(f"  {thr:>10.2f}  {n:>8d}")

# Also report on raw-count basis: gene has >=1 read in every non-stressed sample
expr_ns = expr[:, keep]
n_nonzero_all = int((expr_ns.min(axis=1) > 0).sum())
n_ge5_all = int((expr_ns.min(axis=1) >= 5).sum())
print(f"\n  Genes with >=1 read in EVERY non-stressed sample:  {n_nonzero_all}")
print(f"  Genes with >=5 reads in EVERY non-stressed sample: {n_ge5_all}")

# --- Union: genes expressed in AT LEAST ONE non-stressed sample ---
print("\nGenes expressed (union) — max log2(CPM+1) across non-stressed samples "
      "exceeds threshold:")
print(f"  {'threshold':>10}  {'# genes':>8}")
gene_max = log_cpm_ns.max(axis=1)
for thr in np.arange(0.0, 10.25, 0.25):
    n = int((gene_max > thr).sum())
    print(f"  {thr:>10.2f}  {n:>8d}")

n_union_reads = int((expr_ns.max(axis=1) > 0).sum())
n_union_reads5 = int((expr_ns.max(axis=1) >= 5).sum())
print(f"\n  Genes with >=1 read in AT LEAST ONE non-stressed sample:  {n_union_reads}")
print(f"  Genes with >=5 reads in AT LEAST ONE non-stressed sample: {n_union_reads5}")
print(f"  Total genes in annotation:                                {expr.shape[0]}")

# --- Low-variance / low-CV among the always-expressed (log2(CPM+1)>0) genes ---
print("\n=== Variance / CV of always-expressed genes (log2(CPM+1) > 0 in every "
      "non-stressed sample) ===")
ae_mask = log_cpm_ns.min(axis=1) > 0
ae_idx = np.where(ae_mask)[0]
print(f"Always-expressed pool: {len(ae_idx)} genes")

ae_log = log_cpm_ns[ae_idx]                    # log2(CPM+1) matrix
lib_ns = expr_ns.sum(axis=0, keepdims=True)
cpm_ns = expr_ns / lib_ns * 1e6
ae_cpm = cpm_ns[ae_idx]                        # linear CPM matrix

log_var = ae_log.var(axis=1)
log_std = ae_log.std(axis=1)
cpm_mean = ae_cpm.mean(axis=1)
cpm_std = ae_cpm.std(axis=1)
cv_linear = cpm_std / cpm_mean                 # CV on linear CPM
cv_log = log_std / ae_log.mean(axis=1)         # CV on log scale (less standard)

def quantiles(a, qs=(0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)):
    return {q: float(np.quantile(a, q)) for q in qs}

print("\nVariance of log2(CPM+1) — quantiles:")
for q, v in quantiles(log_var).items():
    print(f"  q={q:>4.2f}  {v:.4f}")

print("\nCV (std/mean) on linear CPM — quantiles:")
for q, v in quantiles(cv_linear).items():
    print(f"  q={q:>4.2f}  {v:.4f}")

print("\nCounts at low-variance thresholds (log2(CPM+1) variance):")
print(f"  {'var <':>8}  {'# genes':>8}  {'fraction':>8}")
for thr in [0.01, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.50, 1.0]:
    n = int((log_var < thr).sum())
    print(f"  {thr:>8.3f}  {n:>8d}  {n/len(ae_idx):>8.3f}")

print("\nCounts at low-CV thresholds (CV on linear CPM):")
print(f"  {'CV <':>8}  {'# genes':>8}  {'fraction':>8}")
for thr in [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.0]:
    n = int((cv_linear < thr).sum())
    print(f"  {thr:>8.3f}  {n:>8d}  {n/len(ae_idx):>8.3f}")

# Top 25 lowest-variance always-expressed genes (housekeeping candidates)
order = np.argsort(log_var)[:25]
print("\nTop 25 lowest log-variance always-expressed genes:")
print(f"  {'gene':<20} {'mean_log':>9} {'var_log':>9} {'CV_lin':>8}")
for k in order:
    gi = ae_idx[k]
    print(f"  {gene_names[gi]:<20} {ae_log[k].mean():>9.3f} "
          f"{log_var[k]:>9.4f} {cv_linear[k]:>8.3f}")
