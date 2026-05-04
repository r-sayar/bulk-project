"""
Audit the per-gene "acceptable region" coverage:

  coverage_of_observed   = union_band_width / (max_train - min_train)
                           — does the union FILL the observed across-donor
                             range, or are there gaps?

  coverage_of_zero_to_max = union_band_width / max_train
                           — what fraction of [0, max] does the union cover?
                             (the "trivial coverage" concern).

  band_to_mean_ratio     = (2σ_tech-based per-sample width) / mean_train
                           — how wide is one sample's band relative to its mean?
                             For high-expression genes this is ≈ 0.56 by
                             construction (since 2·0.14 = 0.28 each side).

Also report the highest-expressed gene specifically, plus stratify the
holdout-outlier rate by expression decile so we can see whether high-mean
genes have lower outlier rates because they're trivially covered.
"""

from pathlib import Path
import numpy as np
import pandas as pd

PROJECT = Path("/Users/rls/Desktop/programming-projects/single-cell/bulk-project")
GCT     = Path("/Users/rls/Downloads/gene_reads_v11_whole_blood.gct.gz")
NOISE_DIR = PROJECT / "blood_technical_noise"

CPM_THRESHOLD   = 1.0
MIN_SAMPLE_FRAC = 0.10
HOLDOUT         = 50
ALPHA           = 0.14
SD_BAND         = 2.0
SEED            = 0


def sigma_tech(x):
    return np.sqrt(x + (ALPHA * x) ** 2)


print("Loading GTEx whole blood ...")
df = pd.read_csv(GCT, sep="\t", skiprows=2, compression="gzip")
gene_names = df["Description"].astype(str).values
counts = df.iloc[:, 2:].values.astype(np.float64)
n_genes_raw, n_samples = counts.shape

# Total genes / expressed genes accounting
lib_all = counts.sum(axis=0)
cpm_all = counts / lib_all * 1e6
expressed = (cpm_all > CPM_THRESHOLD).sum(axis=1) >= int(MIN_SAMPLE_FRAC * n_samples)
print(f"\nGENE COUNT ACCOUNTING")
print(f"  total GENCODE rows in GTEx v11:  {n_genes_raw:,}")
print(f"    of which expressed             {expressed.sum():,}  "
      f"({expressed.mean()*100:.2f}%)")
# Cross-reference with GENCODE biotypes if available
bt_path = PROJECT / "lncrna_analysis/ensembl_biotypes.tsv"
if bt_path.exists():
    ensg = df["Name"].astype(str).str.split(".").str[0].values
    bt = pd.read_csv(bt_path, sep="\t").set_index("ensg")["biotype"].to_dict()
    biotype = np.array([bt.get(e, "unknown") for e in ensg])
    pc_mask  = biotype == "protein_coding"
    lnc_mask = np.isin(biotype, ["lncRNA","antisense","lincRNA","macro_lncRNA",
                                  "non_coding","processed_transcript","sense_intronic",
                                  "sense_overlapping","TEC"])
    print(f"  protein-coding rows:             {pc_mask.sum():,}  "
          f"(expressed: {(pc_mask & expressed).sum():,}, "
          f"{(pc_mask & expressed).sum() / pc_mask.sum() * 100:.1f}%)")
    print(f"  lncRNA rows:                     {lnc_mask.sum():,}  "
          f"(expressed: {(lnc_mask & expressed).sum():,}, "
          f"{(lnc_mask & expressed).sum() / lnc_mask.sum() * 100:.1f}%)")

counts = counts[expressed]
gene_names = gene_names[expressed]

# Train / holdout split (same as before)
rng = np.random.default_rng(SEED)
perm = rng.permutation(n_samples)
counts = counts[:, perm]
counts_train = counts[:, :n_samples - HOLDOUT]
counts_test  = counts[:, n_samples - HOLDOUT:]
lib_train = counts_train.sum(axis=0)
lib_test  = counts_test.sum(axis=0)
ref_lib = float(np.median(lib_train))

# Per-sample bands on ref scale
sd_raw_train  = sigma_tech(counts_train)
scale_train   = ref_lib / lib_train
lower_train_s = np.maximum((counts_train - SD_BAND * sd_raw_train) * scale_train[None, :], 0.0)
upper_train_s = (counts_train + SD_BAND * sd_raw_train) * scale_train[None, :]
counts_train_scaled = counts_train * scale_train[None, :]
counts_test_scaled  = counts_test  * (ref_lib / lib_test)[None, :]


def merge_intervals(starts, ends):
    order = np.argsort(starts)
    s = starts[order]; e = ends[order]
    out_s = [s[0]]; out_e = [e[0]]
    for i in range(1, len(s)):
        if s[i] <= out_e[-1]:
            out_e[-1] = max(out_e[-1], e[i])
        else:
            out_s.append(s[i]); out_e.append(e[i])
    return np.array(out_s), np.array(out_e)


# Per-gene coverage statistics
n_genes = counts_train.shape[0]
mean_train = counts_train_scaled.mean(axis=1)
min_train  = counts_train_scaled.min(axis=1)
max_train  = counts_train_scaled.max(axis=1)
obs_range  = max_train - min_train

union_width   = np.zeros(n_genes)
n_intervals   = np.zeros(n_genes, dtype=int)
holdout_out   = np.zeros(n_genes)        # frac of 50 holdouts outside

for g in range(n_genes):
    s_arr = lower_train_s[g]; e_arr = upper_train_s[g]
    ms, me = merge_intervals(s_arr, e_arr)
    union_width[g] = float((me - ms).sum())
    n_intervals[g] = len(ms)
    y = counts_test_scaled[g]
    in_any = ((y[:, None] >= ms[None, :]) & (y[:, None] <= me[None, :])).any(axis=1)
    holdout_out[g] = 1 - in_any.mean()

# Coverage of observed range: union_width / (max - min). Capped at 1
# because the union may extend below min and above max — clip first.
# Tight version: clip the union to [min, max] then take its length.
clipped_union_widths = np.zeros(n_genes)
for g in range(n_genes):
    s_arr = lower_train_s[g]; e_arr = upper_train_s[g]
    ms, me = merge_intervals(s_arr, e_arr)
    lo = np.maximum(ms, min_train[g])
    hi = np.minimum(me, max_train[g])
    clipped_union_widths[g] = float(np.maximum(hi - lo, 0).sum())

cov_observed = np.where(obs_range > 0, clipped_union_widths / obs_range, 1.0)
cov_zero_max = np.where(max_train > 0, union_width / max_train, np.nan)
band_to_mean = np.where(mean_train > 0,
                         2 * SD_BAND * sigma_tech(mean_train) / mean_train, np.nan)


# Highest-expressed gene
top_idx = np.argsort(mean_train)[::-1]
print("\n" + "=" * 80)
print("HIGHEST-EXPRESSED GENES (by train mean count, ref-scale)")
print("=" * 80)
top_df = pd.DataFrame({
    "gene":              gene_names[top_idx[:25]],
    "mean_train_count":  mean_train[top_idx[:25]],
    "min_train":         min_train[top_idx[:25]],
    "max_train":         max_train[top_idx[:25]],
    "obs_range":         obs_range[top_idx[:25]],
    "union_width":       union_width[top_idx[:25]],
    "cov_observed":      cov_observed[top_idx[:25]],
    "cov_zero_to_max":   cov_zero_max[top_idx[:25]],
    "n_intervals":       n_intervals[top_idx[:25]],
    "frac_holdouts_outside": holdout_out[top_idx[:25]],
})
print(top_df.to_string(index=False))


# Coverage distributions
print("\n" + "=" * 80)
print("COVERAGE-OF-OBSERVED-RANGE distribution across all 16,355 expressed genes")
print("=" * 80)
print(f"  mean   = {cov_observed.mean():.4f}")
print(f"  median = {np.median(cov_observed):.4f}")
for q in (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99):
    print(f"  q={q:>4.2f}  {np.quantile(cov_observed, q):.4f}")

print("\n" + "=" * 80)
print("COVERAGE-OF-[0,MAX] distribution")
print("=" * 80)
v = cov_zero_max[~np.isnan(cov_zero_max)]
print(f"  mean   = {v.mean():.4f}")
print(f"  median = {np.median(v):.4f}")
for q in (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99):
    print(f"  q={q:>4.2f}  {np.quantile(v, q):.4f}")

print("\n" + "=" * 80)
print("BAND-TO-MEAN ratio (2·2σ_tech(μ) / μ)")
print("Interpretation: width of one sample's band relative to its mean.")
print("=" * 80)
v = band_to_mean[~np.isnan(band_to_mean)]
print(f"  mean   = {v.mean():.4f}")
print(f"  median = {np.median(v):.4f}")
for q in (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99):
    print(f"  q={q:>4.2f}  {np.quantile(v, q):.4f}")


# Stratified outlier rate by expression decile
print("\n" + "=" * 80)
print("HOLDOUT OUTLIER RATE BY EXPRESSION DECILE")
print("(answers: do high-expression genes look 'safe' just because their bands are wider?)")
print("=" * 80)
ranks = np.argsort(np.argsort(mean_train))         # 0..n-1
decile = np.minimum(ranks * 10 // n_genes, 9)
print(f"  {'decile':>6}  {'mean_count':>12}  {'cov_observed':>13}  "
      f"{'band/mean':>10}  {'frac_out':>10}  {'n_intervals':>12}")
for d in range(10):
    m = decile == d
    print(f"  {d:>6}  {mean_train[m].mean():>12.2f}  "
          f"{cov_observed[m].mean():>13.4f}  "
          f"{np.nanmean(band_to_mean[m]):>10.4f}  "
          f"{holdout_out[m].mean():>10.4f}  "
          f"{n_intervals[m].mean():>12.2f}")

# Save full per-gene table
out_df = pd.DataFrame({
    "gene": gene_names,
    "mean_train": mean_train,
    "min_train": min_train,
    "max_train": max_train,
    "obs_range": obs_range,
    "union_width": union_width,
    "clipped_union_width": clipped_union_widths,
    "cov_observed": cov_observed,
    "cov_zero_to_max": cov_zero_max,
    "band_to_mean": band_to_mean,
    "n_intervals": n_intervals,
    "frac_holdouts_outside": holdout_out,
})
out_df.to_csv(NOISE_DIR / "per_gene_coverage_audit.csv", index=False)
print(f"\nWrote {NOISE_DIR/'per_gene_coverage_audit.csv'}")
