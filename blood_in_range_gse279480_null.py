"""
Same in-range gene-set table as GTEx, but on the GSE279480 NULL
(unstimulated) condition: 100-donor longitudinal cohort, Smithmyer 2025.

Comparable to the GTEx whole-blood Null analysis: same noise model
(α=0.14), same band sweep, same train/holdout split protocol.
"""

from pathlib import Path
import gzip
import numpy as np
import pandas as pd
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

PROJECT = Path("/Users/rls/Desktop/programming-projects/single-cell/bulk-project")
DATA    = PROJECT / "data" / "GSE279480"
OUT_DIR = PROJECT / "blood_technical_noise"
OUT_DIR.mkdir(exist_ok=True)

CPM_THRESHOLD   = 1.0
MIN_SAMPLE_FRAC = 0.10
HOLDOUT_FRAC    = 50 / 803         # match GTEx ratio  ≈ 6.2%
ALPHA           = 0.14
K_SD_VALUES     = [2.0, 1.0, 0.5, 0.25, 0.1]
SEED            = 0


def sigma_tech(x):
    return np.sqrt(x + (ALPHA * x) ** 2)


# ── 1.  Load metadata to identify Null-condition libraries ────────────
print("Parsing series matrix ...")
rows: dict[str, list[list[str]]] = {}
with gzip.open(DATA / "GSE279480_series_matrix.txt.gz", "rt") as fh:
    for line in fh:
        if line.startswith("!series_matrix_table_begin"):
            break
        if not line.startswith("!Sample_"):
            continue
        parts = line.rstrip("\n").split("\t")
        rows.setdefault(parts[0], []).append([p.strip('"') for p in parts[1:]])

n = len(rows["!Sample_geo_accession"][0])
meta = pd.DataFrame({
    "gsm":   rows["!Sample_geo_accession"][0],
    "title": rows["!Sample_title"][0],
    "lib":   rows["!Sample_description"][0],
})
for row in rows["!Sample_characteristics_ch1"]:
    keys = [c.split(":", 1)[0].strip() for c in row if ":" in c]
    if not keys: continue
    key = Counter(keys).most_common(1)[0][0]
    meta[key] = [c.split(":", 1)[1].strip() if ":" in c else "" for c in row]

print(f"  total samples: {len(meta)}")
print(f"  stimulations:  {sorted(meta['stimulation'].unique())}")

null_libs = meta[meta["stimulation"] == "Null"]["lib"].tolist()
print(f"  Null-condition libraries: {len(null_libs)}")


# ── 2.  Load count matrix; subset to Null libs ────────────────────────
print("Loading gene-count matrix ...")
df = pd.read_csv(DATA / "GSE279480_P441_genecounts.csv.gz")
print(f"  raw shape: {df.shape}")
df = df.rename(columns={df.columns[0]: "gene_id"})
present = [c for c in null_libs if c in df.columns]
counts = df[present].values.astype(np.float64)
gene_ids = df["gene_id"].values
n_g_raw, n_s = counts.shape
print(f"  Null shape: {counts.shape} (genes × Null libraries)")


# ── 3.  Standard expression filter ────────────────────────────────────
lib_all = counts.sum(axis=0)
cpm_all = counts / lib_all * 1e6
expressed = (cpm_all > CPM_THRESHOLD).sum(axis=1) >= int(MIN_SAMPLE_FRAC * n_s)
counts = counts[expressed]
gene_ids = gene_ids[expressed]
n_genes = counts.shape[0]
print(f"  expressed: {n_genes:,} genes")


# ── 4.  Train / holdout split (analogous to GTEx, same fraction) ──────
HOLDOUT = max(5, int(round(HOLDOUT_FRAC * n_s)))
print(f"  using HOLDOUT = {HOLDOUT}  (n_s = {n_s})")
rng = np.random.default_rng(SEED)
perm = rng.permutation(n_s)
counts = counts[:, perm]
counts_tr = counts[:, :n_s - HOLDOUT]
counts_te = counts[:, n_s - HOLDOUT:]
lib_tr = counts_tr.sum(axis=0)
lib_te = counts_te.sum(axis=0)
n_train = counts_tr.shape[1]
ref = float(np.median(lib_tr))
print(f"  train: {n_train}, holdout: {HOLDOUT}, ref lib: {ref:,.0f}")

sd_tr = sigma_tech(counts_tr)
sc_tr = ref / lib_tr
y_te  = counts_te * (ref / lib_te)[None, :]


# ── 5.  Run band sweep ────────────────────────────────────────────────
print(f"\n{'k_SD':>5}  {'mean_n_nbr':>11}  {'med_n_nbr':>10}  "
      f"{'mean_d':>9}  {'d/band':>7}  "
      f"{'#sets_ne':>9}  {'empty_size':>11}  "
      f"{'med_largest':>12}  {'biggest':>9}")
print("-" * 105)

results = []
biggest_examples = []
for ksd in K_SD_VALUES:
    lower = np.maximum((counts_tr - ksd * sd_tr) * sc_tr[None, :], 0.0)
    upper = (counts_tr + ksd * sd_tr) * sc_tr[None, :]

    # train values on the ref scale (for distance computation)
    counts_tr_scaled = counts_tr * sc_tr[None, :]

    n_nbr_arr=[]; med_nbr_arr=[]; n_sets_arr=[]; empty_arr=[]
    largest_arr=[]; biggest_overall=0
    mean_dist_count_arr=[]; mean_dist_pct_band_arr=[]
    for h in range(HOLDOUT):
        y = y_te[:, h]
        A = (y[:, None] >= lower) & (y[:, None] <= upper)
        n_nbr = A.sum(axis=1)
        n_nbr_arr.append(n_nbr.mean())
        med_nbr_arr.append(int(np.median(n_nbr)))

        # Average distance from holdout to in-range neighbours
        # |y - x_scaled| in count units on the ref scale
        diff_abs = np.abs(y[:, None] - counts_tr_scaled)            # (n_g, n_train)
        # Sum over neighbours (mask non-neighbours to zero)
        sum_dist = (diff_abs * A).sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_dist_per_gene = np.where(n_nbr > 0, sum_dist / n_nbr, np.nan)
        # Average across genes (mean of mean — gene-weighted)
        mean_dist_count_arr.append(np.nanmean(mean_dist_per_gene))
        # Also as fraction of the band-half-width (k_SD*sigma_tech_scaled)
        # because the band shape determines what 'distance' is meaningful
        sigma_scaled = sd_tr * sc_tr[None, :]
        # mean distance / mean band-half-width over neighbours
        sum_band = (sigma_scaled * ksd * A).sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_band_per_gene = np.where(n_nbr > 0, sum_band / n_nbr, np.nan)
            pct_band = np.where(mean_band_per_gene > 0,
                                mean_dist_per_gene / mean_band_per_gene,
                                np.nan)
        mean_dist_pct_band_arr.append(np.nanmean(pct_band))

        tuples = [tuple(np.flatnonzero(A[g])) for g in range(n_g_raw if False else n_genes)]
        c = Counter(tuples)
        empty = c.get((), 0)
        non_empty_sizes = [v for k, v in c.items() if k != ()]
        empty_arr.append(empty)
        n_sets_arr.append(len(non_empty_sizes))
        if non_empty_sizes:
            top = max(non_empty_sizes)
            largest_arr.append(top)
            if top > biggest_overall:
                biggest_overall = top
                # Also capture the gene-list for inspection
                top_tuple = max((k for k in c if k != ()), key=lambda kk: c[kk])
                cluster_genes = [gene_ids[g] for g, t in enumerate(tuples) if t == top_tuple][:30]
                biggest_examples.append({
                    "k_sd": ksd, "holdout": h, "size": top,
                    "genes": ";".join(map(str, cluster_genes)),
                })

    print(f"{ksd:>5.2f}  {np.mean(n_nbr_arr):>11.1f}  "
          f"{int(np.median(med_nbr_arr)):>10d}  "
          f"{np.nanmean(mean_dist_count_arr):>9.0f}  "
          f"{np.nanmean(mean_dist_pct_band_arr):>7.3f}  "
          f"{int(np.mean(n_sets_arr)):>9d}  "
          f"{int(np.mean(empty_arr)):>11d}  "
          f"{int(np.median(largest_arr)) if largest_arr else 0:>12d}  "
          f"{biggest_overall:>9d}")

    results.append({
        "k_sd": ksd,
        "mean_n_neighbors": float(np.mean(n_nbr_arr)),
        "median_n_neighbors": float(np.median(med_nbr_arr)),
        "mean_neighbor_distance_count": float(np.nanmean(mean_dist_count_arr)),
        "mean_neighbor_distance_pct_of_band": float(np.nanmean(mean_dist_pct_band_arr)),
        "n_gene_sets_non_empty": float(np.mean(n_sets_arr)),
        "empty_bucket_size": float(np.mean(empty_arr)),
        "median_largest_set": float(np.median(largest_arr) if largest_arr else 0),
        "biggest_set_overall": int(biggest_overall),
    })

res_df = pd.DataFrame(results)
res_df.to_csv(OUT_DIR / "in_range_GSE279480_Null.csv", index=False)
pd.DataFrame(biggest_examples).to_csv(
    OUT_DIR / "in_range_GSE279480_Null_biggest_examples.csv", index=False
)


# Side-by-side comparison with the published GTEx numbers
GTEX_TABLE = pd.DataFrame({
    "k_sd":  K_SD_VALUES,
    "GTEx_mean_n_nbr":  [236, 120, 60, 31, 13],
    "GTEx_median_n_nbr":[232, 116, 58, 29, 11],
    "GTEx_#sets_ne":    [16339, 16305, 16248, 16144, 15794],
    "GTEx_empty_size":  [1, 4, 16, 45, 193],
    "GTEx_median_largest": [1, 2, 2, 3, 6],
    "GTEx_biggest":     [1, 4, 16, 64, 59],
})
print("\nSide-by-side comparison:")
sxs = pd.merge(res_df, GTEX_TABLE, on="k_sd")
print(sxs.to_string(index=False))
sxs.to_csv(OUT_DIR / "in_range_GSE279480_vs_GTEx.csv", index=False)
print(f"\nWrote {OUT_DIR/'in_range_GSE279480_Null.csv'}")
print(f"Wrote {OUT_DIR/'in_range_GSE279480_vs_GTEx.csv'}")
