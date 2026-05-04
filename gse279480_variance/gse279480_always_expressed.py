"""
GSE279480 Null: count "always expressed" genes after stress removal,
mirroring gtex_always_expressed.py.

Reuses the cluster assignment produced by gse279480_nonstressed_analysis.py
to identify non-stressed samples without re-running bimodal detection.
"""

from collections import Counter
from pathlib import Path
import gzip
import warnings

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent
COUNTS_CSV = HERE.parent / "data/GSE279480/GSE279480_P441_genecounts.csv.gz"
SERIES_MATRIX = HERE.parent / "data/GSE279480/GSE279480_series_matrix.txt.gz"
SYMBOL_MAP = HERE / "ensembl_to_symbol.tsv"

# Load Null counts ----------------------------------------------------
print("Loading GSE279480 Null...")
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

null_libs = [l for l in meta.loc[meta["stimulation"] == "Null", "lib"]
             if l in pd.read_csv(COUNTS_CSV, index_col=0, nrows=0).columns]
counts = pd.read_csv(COUNTS_CSV, index_col=0)
expr = counts[null_libs].values.astype(np.float64)
gene_ids = np.array(counts.index)
print(f"  {expr.shape[0]:,} genes x {expr.shape[1]} Null samples")

# log2(CPM+1) on full count matrix
lib = expr.sum(axis=0, keepdims=True)
log_cpm = np.log2(expr / lib * 1e6 + 1)

# Re-identify stressed/nonstressed via the same bimodal-PCA approach ---
print("Identifying stressed vs non-stressed via bimodal-state PCA...")
gm, gs = log_cpm.mean(axis=1), log_cpm.std(axis=1)
cand = np.where((gm > 1) & (gs > 0.3))[0]


def bimodal_indices(data, idx):
    out = []
    for i in idx:
        v = data[i]
        try:
            kde = gaussian_kde(v, bw_method="scott")
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

binmat = np.zeros((len(bi), log_cpm.shape[1]), dtype=np.int8)
for j, gi in enumerate(bi):
    v = log_cpm[gi]
    kde = gaussian_kde(v, bw_method="scott")
    xg = np.linspace(v.min(), v.max(), 500)
    d = kde(xg)
    peaks, _ = find_peaks(d, prominence=d.max() * 0.08, distance=40)
    if len(peaks) >= 2:
        sp = np.sort(xg[peaks]); thr = (sp[0] + sp[1]) / 2
    else:
        thr = np.median(v)
    binmat[j] = (v > thr).astype(np.int8)

scores = PCA(n_components=2).fit_transform(binmat.T)
state = (scores[:, 0] > 0).astype(int)

sym_df = pd.read_csv(SYMBOL_MAP, sep="\t").drop_duplicates("ensembl_id")
sym_to_ens = dict(zip(sym_df["symbol"], sym_df["ensembl_id"]))

stress_markers = ["DDIT4", "JUN", "VEGFA"]
sm_score = {}
for s in (0, 1):
    total = 0.0
    for g in stress_markers:
        ens = sym_to_ens.get(g)
        if ens is None:
            continue
        idx = np.where(gene_ids == ens)[0]
        if len(idx):
            total += log_cpm[idx[0], state == s].mean()
    sm_score[s] = total
stressed = max(sm_score, key=sm_score.get)
nonstressed = 1 - stressed
keep = state == nonstressed
log_cpm_ns = log_cpm[:, keep]
expr_ns = expr[:, keep]
n_ns = log_cpm_ns.shape[1]
print(f"  Stressed = state {stressed} (score {sm_score[stressed]:.2f} vs {sm_score[nonstressed]:.2f})")
print(f"  Non-stressed samples: {n_ns} / {log_cpm.shape[1]}")

# --- Always-expressed thresholds ---
print("\nGenes always expressed (min log2(CPM+1) across all non-stressed samples > threshold):")
print(f"  {'threshold':>10}  {'# genes':>8}")
gene_min = log_cpm_ns.min(axis=1)
table_intersect = []
for thr in np.arange(0.0, 10.25, 0.25):
    n = int((gene_min > thr).sum())
    table_intersect.append((float(thr), n))
    print(f"  {thr:>10.2f}  {n:>8d}")

n_nonzero_all = int((expr_ns.min(axis=1) > 0).sum())
n_ge5_all = int((expr_ns.min(axis=1) >= 5).sum())
print(f"\n  Genes with >=1 read in EVERY non-stressed sample:  {n_nonzero_all}")
print(f"  Genes with >=5 reads in EVERY non-stressed sample: {n_ge5_all}")

print("\nGenes expressed (union) — max log2(CPM+1) across non-stressed samples > threshold:")
print(f"  {'threshold':>10}  {'# genes':>8}")
table_union = []
gene_max = log_cpm_ns.max(axis=1)
for thr in np.arange(0.0, 10.25, 0.25):
    n = int((gene_max > thr).sum())
    table_union.append((float(thr), n))
    print(f"  {thr:>10.2f}  {n:>8d}")

n_union = int((expr_ns.max(axis=1) > 0).sum())
n_union5 = int((expr_ns.max(axis=1) >= 5).sum())
print(f"\n  Genes with >=1 read in AT LEAST ONE non-stressed sample:  {n_union}")
print(f"  Genes with >=5 reads in AT LEAST ONE non-stressed sample: {n_union5}")
print(f"  Total genes in annotation:                                 {expr.shape[0]}")

out = HERE / "gse279480_null_always_expressed_thresholds.csv"
pd.DataFrame({
    "threshold_log2cpm": [t for t, _ in table_intersect],
    "n_always_expressed": [n for _, n in table_intersect],
    "n_expressed_union": [n for _, n in table_union],
}).to_csv(out, index=False)
print(f"\nSaved: {out}")
