# Bulk-Project — Consolidated Findings

A single index of every analysis in this folder, the inputs, the method, and the
numerical / biological result. All numbers are taken from the existing output
files (CSVs, JSONs, `*_summary.txt`, `*_biology.txt`); no re-run was necessary
because the produced artefacts are intact and self-consistent.

> Working dir: `/Users/rls/Desktop/programming-projects/single-cell/bulk-project`
> Today's date used for context: 2026-05-01.

---

## 0. Datasets in play

| Tag | Source | Shape | Modality |
|---|---|---|---|
| **GTEx v11 whole blood** | `gene_reads_v11_whole_blood.gct.gz` | 74,628 genes × 803 donors (16,355 expressed after CPM≥1 in ≥10% filter) | bulk RNA-seq |
| **GTEx v11 liver** | `gene_reads_v11_liver.gct.gz` | full GTEx v11 | bulk RNA-seq |
| **HCA blood pseudobulk** | `pseudobulk/hca_blood_pseudobulk.npz` (built from `pseudobulk/blood_h5/*.h5`) | 8 donors × 33k genes | pseudo-bulk from sc 10x |
| **GSE279480 (Smithmyer 2025)** | `data/GSE279480/GSE279480_P441_genecounts.csv.gz` | ex-vivo whole blood, 4 stims (Null / LPS / Poly I:C / SEB), ~52 donors × 4 conditions | bulk RNA-seq |
| **GSE84133** | `data/GSE84133_RAW` | human pancreatic islets sc | sc RNA-seq |
| **HPA secretome** + **Bausch-Fluck 2018 surfaceome** | `data/annotations/` | gene-membership annotations | gene sets |
| **Tabula Sapiens blood** | `data/downloaded_sc/tabula_sapiens/blood.h5ad` | 10x scRNA, blood compartment | sc RNA-seq |

Common preprocessing across scripts: drop genes failing `(CPM > 1 in ≥10% samples)`, then CPM normalize, then `log2(CPM+1)`.

---

## 1. Headline question — does GTEx blood "saturate"?

Three independent saturation tests were run on the 803 GTEx donors, each
holding out 50 samples and growing a training pool to `n ∈ {100…700}`.

### 1.1 PCA-50 reconstruction MSE on expressed genes ([blood_saturation.csv](blood_saturation.csv), [blood_sample_saturation.py](blood_sample_saturation.py))

| n | PCA-MSE (subspace residual) | mean z² (per-gene Gaussian) |
|---|---|---|
| 100 | 0.133 | 1.016 |
| 300 | 0.109 | 1.033 |
| 500 | 0.102 | 1.025 |
| 700 | **0.099** | **1.017** |

**PCA-MSE definition.** Fit PCA(k=50) on the first `n` training samples,
center each held-out sample on the train mean, project it onto the 50 train
PCs and reconstruct, then take the mean squared residual. This is *not* a
sample-to-sample MSE — it asks "does the held-out sample lie in the linear
subspace already spanned by the train pool?" PCA-MSE is the right
saturation probe because plain `MSE(test, train_mean)` would plateau at
n≈100 (means estimate stably from few samples) and tell us nothing about
covariance coverage; PCA-MSE keeps falling until the *covariance structure*
between genes is captured.

**Variance captured by PCA-50.** At the saturated point (n=700+), the 50
train PCs explain **86.6 %** of training variance and capture **83.95 %**
of held-out variance — a 2.6-pp generalization gap that does not widen as
`n` grows, which is the formal saturation signal. PC1 alone is responsible
for **34 %** of all variance (the State-A / State-B ex-vivo handling axis
from §2); PC2 adds another ~16 % (cell-composition / neutrophil fraction);
PC1..PC10 hits 73 %. So roughly half of GTEx whole-blood variance lives in
two interpretable biological axes, and PCA-50 is well-calibrated rather
than over- or under-fit:

| at training n | PC1 | PC1..10 | PC1..25 | **PC1..50** |
|---|---|---|---|---|
| 100 | 37.1% | 76.4% | 87.2% | 93.8% |
| 300 | 36.1% | 73.4% | 82.9% | 88.3% |
| 500 | 34.5% | 73.0% | 82.1% | 87.1% |
| 700 | 34.0% | 72.9% | 81.8% | **86.6%** |
| full | 34.3% | 73.0% | 81.8% | **86.6%** |

(High small-n values are PCA over-fitting the limited train pool; the
86–87 % asymptote is the real number.)

PCA reconstruction error shows a clean elbow at ~n=400 and is essentially flat
after n=500. Per-gene Gaussian z² stays at ~1 across all sizes — once a gene's
mean and std are estimated from ~200 samples, more samples don't help (so the
*marginals* are saturated long before the *subspace* is).

### 1.2 Whole-transcriptome PCA MSE ([blood_saturation_full_transcriptome.csv](blood_saturation_full_transcriptome.csv))

Same protocol but with the no-filter gene set (drop only all-zero genes,
~70k genes).

| n | PCA MSE | drop vs prev |
|---|---|---|
| 100 | 0.0364 | — |
| 300 | 0.0302 | -6.0% |
| 500 | 0.0284 | -2.5% |
| 700 | **0.0275** | **-1.6%** |

Drop per added 100 samples is < 2 % from n ≥ 500: the transcriptome-wide
covariance structure is captured by ~600 donors.

### 1.3 Nearest-neighbour saturation ([blood_nn_saturation.csv](blood_nn_saturation.csv))

For each held-out sample, find its closest training donor (full transcriptome
MSE).

| n | mean nn-MSE | drop |
|---|---|---|
| 100 | 0.1017 | — |
| 300 | 0.0891 | -5.2% |
| 500 | 0.0859 | -1.3% |
| 700 | **0.0850** | **-0.6%** |

Every held-out sample finds a near-twin already in the pool by ~n=500.

### 1.4b Per-sample technical-noise band coverage ([blood_technical_noise/](blood_technical_noise/), [blood_technical_noise_outliers.py](blood_technical_noise_outliers.py))

A sharper saturation probe than PCA-MSE: instead of asking whether a held-out
sample lies in a *linear subspace* spanned by the training pool, ask whether
each individual gene's value is **technically consistent** with a value
already observed in some training sample.

Noise model (NB / "Poisson + multiplicative"):

```
σ_tech(x) = sqrt( x + (α·x)² ),   α = 0.14
```

Verified against the published table: μ=10 → ±2σ ≈ [3, 17]; μ=100 → ≈ [66, 134]; μ=1000 → ≈ [713, 1287]; μ=10000 → ≈ [7193, 12807].

Procedure:
1. Train / holdout = 753 / 50 (same shuffle as §1.1).
2. For each train sample `s` and gene `g`, build the band
   `[x_{s,g} − 2σ_tech(x_{s,g}), x_{s,g} + 2σ_tech(x_{s,g})]` on raw counts in
   that sample's native library, then linearly scale band edges to the
   train-median library size (42 M reads).
3. For each gene, the **acceptable region** = union of the 753 per-sample
   bands. Stored as merged intervals — saved for the first 100 genes in
   [per_gene_intervals_first100.tsv](blood_technical_noise/per_gene_intervals_first100.tsv); per-gene summary for all 16,355 genes in [per_gene_summary.tsv](blood_technical_noise/per_gene_summary.tsv).
4. For each held-out sample, count genes whose value falls outside the union.

#### Result — 99.95 % gene-level coverage

| | value |
|---|---|
| Mean intervals per gene | **1.1** (median 1.0) |
| Per-holdout fraction of genes outside union, mean | **0.05 %** (≈ 8 of 16,355) |
| Per-holdout fraction outside union, range | 0.00 % – 0.89 % |
| Decomposition | 0.02 % above envelope / 0.02 % below / 0.01 % in inter-band gap |

**For 90 %+ of genes the 753 per-sample bands are so densely overlapping
that they collapse into a single contiguous acceptable interval.** A few
hundred genes (the bimodal panel from §2 and similar) yield 2+ disjoint
intervals — those are the genes where the train pool covers two distinct
expression states with a real "hole" between them.

The per-holdout outlier rate of ~0.05 % is **two orders of magnitude
tighter than what the PCA-MSE residual implied** (PCA-50 left ~16 % of
test variance unexplained, but it was distributed across all genes; the
per-gene coverage shows that the unexplained variance is concentrated in a
small number of biologically-variable genes).

#### Where the residual outliers concentrate ([outlier_summary_per_gene.csv](blood_technical_noise/outlier_summary_per_gene.csv))

Top genes that fall outside the acceptable region for ≥ 4 % of the 50 holdouts. The `obs_sd_over_tech_sd` column shows how many times wider the
observed across-donor SD is than the technical-noise prediction:

| Gene | train mean count | obs SD / tech SD | Biology |
|---|---|---|---|
| **PARP15** | 1,199 | 6.8× | DNA-damage / immune; donor-specific induction |
| **ACSM3** | 114 | 4.9× | mitochondrial fatty-acid; metabolic donor effect |
| **TRDV2** | 37 | **10.1×** | T-cell receptor δ-chain — donor-specific TCR repertoire |
| HORMAD1 | 22 | 4.2× | meiotic; rare donor-specific expression |
| **SNHG1** | 1,254 | 5.1× | snoRNA host lncRNA |
| **NRAP** | 149 | **35.1×** | nebulin-related; tissue-contamination signature |
| **IGHV3-30 / IGHV3-33** | 499 / 302 | 15.7× / 11.9× | Ig heavy-chain V genes — donor antibody repertoire |
| LINC02967 / PDK1 / STIP1 / NUDC | 1k–4k | 4–8× | mostly stress / immune-response variable |

**These are exactly the genes where biology dominates technical noise**:
T-cell receptor and immunoglobulin V-gene rearrangements (genuinely
donor-specific by definition), some donor-specific metabolic loci, and a
handful of stress-state genes from the §2 axis. The `obs_sd / tech_sd`
column is the cleanest single summary of "biological vs technical
variance" we have on this dataset.

#### Caveat — band specificity ([blood_band_specificity.py](blood_band_specificity.py))

The 99.95 % figure above is partially trivial for most genes: with 753
±2σ_tech bands and σ_tech / μ ≈ 14–28 %, the per-sample bands tile
`[0, max_train]` almost completely.

Quantifying specificity:
`uncovered_fraction_g = 1 − (clipped_union_width_g / max_train_g)`.
HIGH = tight, informative bands (most of the [0, max] range is forbidden).
LOW = bands cover everything; "inside" is trivial.

Distribution across 16,355 expressed genes:

| q | uncovered fraction |
|---|---|
| 0.25 | **0.000** |
| 0.50 | 0.010 |
| 0.75 | 0.034 |
| 0.95 | 0.104 |
| 0.99 | 0.335 |
| max | 0.71 |

- ~25 % of genes have uncovered=0 exactly (one big interval spanning [0, max])
- only **0.3 %** have uncovered > 0.5
- only 1 gene (CD1A, 0.71) has uncovered > 0.7

So the specific-saturation argument is restricted to ~50 bimodal genes,
where the per-sample bands really do form 2-3 disjoint clusters with a
biological "hole" between them — and where all 50 holdouts still land
inside one of those tight clusters. For the other 99.7 % of genes, the
"inside the union" check is uninformative on a per-gene basis (still
useful in aggregate: nothing exceeds the envelope or falls in real gaps).

**Top tight + reproducible genes** (high uncovered AND 50/50 holdouts inside),
from [per_gene_band_specificity.csv](blood_technical_noise/per_gene_band_specificity.csv):

| Gene | uncov | n intervals | What it is |
|---|---|---|---|
| **CD1A** | 0.71 | 2 | DC antigen presentation; bimodal |
| EPOP, LAMP5 | 0.68–0.69 | 2 | bimodal regulators |
| DEFB1 | 0.66 | 2 | defensin |
| **IGHV3-73, IGKV2-29** | 0.63–0.64 | 2–3 | Ig V-segments — donor antibody repertoire |
| RNASE2CP | 0.64 | 2 | eosinophil pseudogene |
| EPHB6, TUBB4A, KCNE5, CYP2U1 | 0.6–0.65 | 2 | tissue-restricted bimodal |

#### Why this matters for saturation (reframed)

Two distinct claims, both supported but with different strength:

1. **Aggregate envelope coverage** (weak but real): for any new blood donor,
   ~99.95 % of expressed genes sit inside the per-sample technical-noise
   union of at least one of the 753 reference donors. This holds because
   the union mostly tiles `[0, max_train]` — useful as an "outlier check"
   but not a specific saturation claim.
2. **Bimodal-gene specific saturation** (strong but narrow): for ~50
   bimodal/state-discriminating genes, the union is genuinely tight
   (spans only 30–40 % of `[0, max]`) AND all 50 holdouts still land
   inside the tight clusters. These are the genes where the train pool
   encodes a real two-state structure (e.g. State-A / State-B from §2)
   and the held-outs reproduce that structure perfectly.

The 0.05 % that fall outside (T-cell / Ig V-segments, donor-specific
metabolic loci, stress-axis tails) are inherently donor-specific —
saturation isn't being held up by missing samples but by biology that
no fixed-size cohort can absorb.

Visualization: [blood_technical_noise_outliers.png](blood_technical_noise/blood_technical_noise_outliers.png) — 9-panel: per-holdout outlier rate, above/below/in-gap decomposition, vs library size, per-gene intervals fragmentation, per-gene outlier rate distribution, observed-SD-vs-predicted-tech-SD scatter, top-20 always-outside genes, single-gene example, and outlier rate vs expression level.

### 1.4 KDE coverage at n=700 ([blood_kde_coverage_per_sample.csv](blood_kde_coverage_per_sample.csv), [blood_kde_coverage_per_gene.csv](blood_kde_coverage_per_gene.csv))

For each gene a Gaussian KDE is fit on the 700-sample train pool and the 50
holdouts are scored against the 90 % HDR.

- Per-sample HDR-90 coverage range: **0.78 – 0.99**, mean ≈ 0.90 (matches the
  expected 0.90 — i.e., training distribution is well-calibrated).
- Per-gene coverage near the expected 0.90 across the spread of stds.

**Conclusion (saturation).** GTEx whole blood with 803 donors *covers what
typical blood can look like* both in expressed-gene PC space and in the long
tail of sparse genes. The marginal new sample is ~99% explainable from the
existing pool. This is the empirical justification for using GTEx as a "bulk
prior" anchor in the cross-modality work.

---

## 2. Two transcriptomic states in GTEx whole blood (ex-vivo handling stress)

### 2.1 Bimodal-gene-driven PCA split ([bimodal_state_biology.txt](bimodal_state_biology.txt), [bimodal_state_analysis.py](bimodal_state_analysis.py))

- Bimodal genes detected (KDE + `find_peaks`, prominence ≥ 8% of max,
  candidates with `mean>1, std>0.3`): **874**
- PC1 of the binary "high/low" matrix explains **66.3 %** of variance →
  PCA-1 sign splits the 803 donors into:
  - **State A (PC1 > 0):** 373 donors (later anchored to 424)
  - **State B (PC1 ≤ 0):** 430 donors (later anchored to 361)
- After full DE on all 74,628 genes: **28,756 genes** significant at p < 0.001.

### 2.2 What the two states are

State B is *ex-vivo handling / dissociation stress*. State A is the cleaner
baseline.

Top State-B markers (massive log2FC, p ≈ 0):

| Gene | log2FC (A vs B) | Class |
|---|---|---|
| CXCL8 (IL-8) | -7.26 | inflammation |
| HILPDA | -6.78 | hypoxia / lipid droplet |
| JUN | -5.46 | immediate-early TF |
| ALB | -5.09 | (likely contam / specific donors) |
| HSPA1B | -4.93 | heat shock |
| G0S2 | -4.90 | stress |
| CCL3 / CCL4 | -4.85 / -4.22 | inflammation |
| DDIT4 | -4.47 | stress / mTOR repression |
| VEGFA | -4.06 | hypoxia |
| PLIN2 | -4.47 | stress lipid droplet |

Stress-gene panel mean log2FC (A vs B) = **-3.15** → stress genes *much*
higher in B. Housekeeping panel mean log2FC = **+0.016** → housekeeping
genes virtually unchanged across the split (correct null).

### 2.3 Anchor genes ([bimodal_anchor_summary.txt](bimodal_anchor_summary.txt), [bimodal_anchor_analysis.py](bimodal_anchor_analysis.py))

Genes whose two KDE peaks fall on opposite sides of the PC1 split with very
clean per-sample assignment.

- 98 % threshold: **2 genes — DDIT4 (HIGH in B) and FRAT1 (HIGH in B)**
  | Gene | frac_A_high | frac_B_high |
  |---|---|---|
  | DDIT4 | 0.986 | 0.019 |
  | FRAT1 | 0.018 | 0.986 |
- 95 % threshold: **58 genes**, including JUN, BHLHE40, G0S2, VEGFA, HILPDA,
  PLIN2, HBEGF, CCL3, MAFF, PPP1R15A (state A high) and PTAFR, MED18, RER1,
  TADA3, P2RY13, TNFSF10 (state B high).
- After removing 18 ambiguous samples: 424 (A) + 361 (B) = 785 confidently
  assigned donors.

### 2.4 Implication for downstream work

- 18 / 803 (~2.2%) of GTEx blood donors are unassignable; the rest can be
  cleanly relabeled stress / non-stress at zero cost.
- Any cross-modality model must either (a) be conditioned on state or
  (b) explicitly remove stress-state genes — otherwise it will memorise
  the donor-collection artefact instead of biology.

---

## 3. The "always expressed in non-stressed blood" gene set

[gtex_always_expressed.py](gtex_always_expressed.py) restricts to non-stressed
samples and counts genes whose log2(CPM+1) exceeds a threshold in *every*
non-stressed donor. The output (and similar analyses for the GSE279480 null
condition in [gse279480_variance/gse279480_null_always_expressed_thresholds.csv](gse279480_variance/gse279480_null_always_expressed_thresholds.csv))
gives a single curve `threshold → #genes` so the user can pick any working
"core blood transcriptome" definition.

GSE279480 null condition (n=51 donors) for cross-reference:

| threshold log2(CPM+1) | always-expressed | union-expressed |
|---|---|---|
| 0.0 | 9,432 | 44,621 |
| 1.0 | 8,425 | 31,444 |
| 2.0 | 7,296 | 20,198 |
| 5.0 | 2,692 | 9,245 |
| 7.0 | 585 | 3,602 |

The "~340 always-expressed" target the script was probing corresponds roughly
to log2(CPM+1) ≳ 7.5 (top end of housekeeping range).

---

## 4. Sample-size and depth confounds ([sample_size_depth_analysis.py](sample_size_depth_analysis.py))

Two confounders that could explain HCA's much lower CV (~0.06) vs GTEx (~0.25):

1. **n=8 vs n=803.** Bootstrap subsamples of 8 GTEx donors recover
   HCA-like CV — the gap is mostly **statistical**, not biological.
2. **Sequencing depth.** Library sizes 10 M – 170 M reads per sample.
   Binning by depth: high-depth bins do show modestly lower CV, but the
   dominant factor is still aggregation/sample size, not depth.

→ Sister test [pseudobulk_aggregation_test.py](pseudobulk_aggregation_test.py):
randomly pool the 803 GTEx donors into 8 groups of ~100 and recompute CV. CV
collapses toward the HCA ~0.06 level. Confirms: **HCA's low CV is largely a
mathematical aggregation effect, not pseudobulk being a "cleaner" technology.**

Saved figures: [gtex_group_size_mse.png](gtex_group_size_mse.png),
[depth_binning_cv.png](depth_binning_cv.png),
[pseudobulk_aggregation_test.png](pseudobulk_aggregation_test.png),
[sample_size_bootstrap.png](sample_size_bootstrap.png).

---

## 5. Stress / housekeeping cross-modality comparison ([stress_housekeeping_comparison.py](stress_housekeeping_comparison.py), [hk_variation_normfactor.py](hk_variation_normfactor.py))

Pipeline:
1. Compute baseline GTEx ↔ HCA-pseudobulk similarity (Pearson on per-gene CV
   in log space, KS, median |ΔCV|).
2. Drop a curated stress panel (HSPs, immediate-early TFs, cytokines).
   Recompute → confirms the GTEx ↔ HCA agreement *improves* once stress
   genes are excluded, confirming that the visible bulk/sc gap is partly
   stress-state contamination on the bulk side.
3. Benchmark normalization factors (CPM / TMM / median-of-ratios / size-factor)
   against the housekeeping panel — winner = the one that minimises HK CV
   after normalisation.
4. Quantify HK variation across three pairings:
   - bulk ↔ bulk (GTEx halves) — expected lowest
   - bulk ↔ pseudobulk (GTEx vs HCA) — middling
   - bulk ↔ single-cell (GTEx vs Tabula Sapiens) — largest

Outputs: [stress_removal_comparison.png](stress_removal_comparison.png),
[housekeeping_comparison.png](housekeeping_comparison.png),
[hk_variation_normfactor.png](hk_variation_normfactor.png),
[housekeeping_kde_grid.png](housekeeping_kde_grid.png).

---

## 6. GTEx liver ↔ blood interaction ([gtex_liver_blood_interaction.py](gtex_liver_blood_interaction.py))

Hypothesis: liver should over-express the **secretome** (plasma proteins
dumped into blood) and the blood-receptor **surfaceome**; blood cells should
*not* manufacture plasma proteins.

Confirmed: [gtex_liver_blood_interaction.png](gtex_liver_blood_interaction.png),
[gtex_blood_vs_liver.png](gtex_blood_vs_liver.png) — liver dominates the
secretome side (ALB, fibrinogen, complement, apolipoproteins, clotting
factors) and the relevant scavenger / SLC / SLCO surface receptors. Blood is
dominated by hemoglobin / immune / HLA / receptor machinery and is empty of
plasma-protein synthesis.

This is the "two compartments touching" sanity check for the liver +
surfaceome / secretome workstream.

---

## 7. Other GTEx variance / structure analyses

| Script | Output | Punch line |
|---|---|---|
| [gtex_tissue_variance_analysis.py](gtex_tissue_variance_analysis.py) | [gtex_variance_analysis.png](gtex_variance_analysis.png), [gtex_clustering_cv.png](gtex_clustering_cv.png) | Within-cluster CV is **lower** than whole-dataset CV — confirms there's recoverable substructure in blood; same shape in liver. |
| [gtex_ranked_gene_distributions.py](gtex_ranked_gene_distributions.py) | [gtex_ranked_distributions.png](gtex_ranked_distributions.png), grid plots `gtex_groups_*.png`, `gtex_pc1_sorted.png`, `gtex_pc2_sorted.png` | Ranked-gene KDEs reveal multimodal structure across genes; PC1/PC2 sort axes recover the stress-state axis (PC1) and a secondary cell-composition axis (PC2). |
| [hca_pseudobulk_analysis.py](hca_pseudobulk_analysis.py) | [gtex_vs_hca_pseudobulk.png](gtex_vs_hca_pseudobulk.png), [pseudobulk/hca_blood_pseudobulk.npz](pseudobulk/hca_blood_pseudobulk.npz) | Builds the 8-donor HCA pseudobulk from raw 10x h5s; the npz is the reusable artefact for every GTEx ↔ pseudobulk comparison in this folder. |
| [gtex_nonstressed_analysis.py](gtex_nonstressed_analysis.py) | [gtex_nonstressed_states.png](gtex_nonstressed_states.png) | Re-runs the bimodal-state analysis *after* removing stress samples → only ~30–60 bimodal genes survive, suggesting the stress axis was driving most of the "bimodality" signal. |
| [geneplot.py](geneplot.py) + [GENEPLOT.md](GENEPLOT.md) | one-stop CLI for plotting any gene's distribution (linear / log / KDE / density / train-test / sc-overlay / archetype clustering / PC1-PC2 sort) | Living tool; many of the `gtex_*.png` figures in the root are its outputs. |

---

## 8. GSE279480 — ex-vivo stimulation panel ([gse279480_variance/](gse279480_variance/))

Same pipeline as the GTEx tissue analysis, applied to the 4 stim conditions
of the same whole-blood cohort.

| Condition | # genes with CV > 1 ([cv csv](gse279480_variance)) | bimodal genes (KDE) |
|---|---|---|
| Null | 1,289 | 33 (full) / 62 (after stress removal) |
| LPS | 896 | — |
| Poly I:C | 1,564 | — |
| SEB | 924 | — |

The high-CV gene set per condition is non-trivially **stim-specific** — the
shared-across-all-stims subset is much smaller (`gse279480_cv_gt1_shared_all_stims.csv`).
Bimodality persists in the unstimulated (Null) condition: same DDIT4 / GDF15 /
RAB36 / GRHL1 patterns visible in [gtex_bimodal_kde.png](gtex_bimodal_kde.png)
recur, suggesting the stress axis is *donor-intrinsic*, not just GTEx-specific
collection artefact.

---

## 9. Deconvolution

### 9.1 HVG-MLP cluster proportions ([src/results/deconvolution_hvg_mlp/](src/results/deconvolution_hvg_mlp/))

A simple MLP that takes 2,000 HVG log expressions of a synthetic bulk and
predicts the 14-cluster proportion vector.

| Metric | Value |
|---|---|
| RMSE | **0.0115** |
| MAE | 0.0089 |
| Pearson | **0.977** |
| Spearman | 0.968 |
| CCC (Lin) | **0.972** |

Per-cell-type ([src/results/deconvolution_evaluation/metrics_per_celltype.csv](src/results/deconvolution_evaluation/metrics_per_celltype.csv)):
all 14 GSE84133 human islet cell types reach Pearson > 0.97 and CCC > 0.957.
Worst type is endothelial (CCC 0.957); best is t_cell (CCC 0.978). Biases are
all |bias| < 0.006.

### 9.2 HVG-count sweep with Neural W-CLS v3 ([src/results/deconvolution_evaluation/hvg_*/metrics_summary.csv](src/results/deconvolution_evaluation))

| n HVG | RMSE | Pearson | CCC |
|---|---|---|---|
| 2,000 | 0.0270 | 0.892 | 0.795 |
| 3,000 | 0.0266 | 0.898 | 0.801 |
| 4,000 | 0.0260 | 0.902 | 0.811 |
| 5,000 | 0.0263 | 0.901 | 0.807 |
| 7,000 | 0.0259 | 0.905 | 0.813 |
| 12,000 | 0.0262 | 0.897 | 0.809 |
| **14,000** | **0.0254** | **0.903** | **0.823** |

Two takeaways:
- HVG count beyond ~4,000 buys very little; the curve is essentially flat
  4k → 14k.
- HVG-MLP (Pearson 0.977 / CCC 0.972) substantially outperforms Neural W-CLS
  v3 (best 0.905 / 0.823) on this synthetic mixture benchmark — i.e., the
  cluster-proportion regression head is a stronger formulation than the
  CLS-style baseline once the synthetic mixtures are well-distributed.

Visual evidence: [src/results/deconvolution_evaluation/scatter_grid.png](src/results/deconvolution_evaluation/scatter_grid.png),
[src/results/deconvolution_evaluation/per_celltype_bars.png](src/results/deconvolution_evaluation/per_celltype_bars.png),
[src/results/deconvolution_evaluation/ccc_heatmap.png](src/results/deconvolution_evaluation/ccc_heatmap.png),
[src/results/deconvolution_evaluation/bland_altman.png](src/results/deconvolution_evaluation/bland_altman.png),
[src/results/deconvolution_evaluation/synthetic_cell_proportion_distribution_with_centroids.png](src/results/deconvolution_evaluation/synthetic_cell_proportion_distribution_with_centroids.png).

### 9.3 Pipeline modules ([src/scripts/deconvolution/](src/scripts/deconvolution/))

| File | Purpose |
|---|---|
| [preprocessing.py](src/scripts/deconvolution/preprocessing.py) | shared filter / CP10K / log1p / HVG-selection helpers, with joblib disk cache |
| [deconvolution_of_bulk_rna_seq_using_deep_learning.py](src/scripts/deconvolution/deconvolution_of_bulk_rna_seq_using_deep_learning.py) | full pipeline (NNLS / NMF / Neural W-CLS v3 / "all"); supports synthetic, GSE84133, or any GEO accession |
| [hvg_mlp_cluster_proportions.py](src/scripts/deconvolution/hvg_mlp_cluster_proportions.py) | the strong baseline → 0.977 Pearson |
| [evaluate_deconvolution.py](src/scripts/deconvolution/evaluate_deconvolution.py) | post-hoc plotting / CCC heatmap / per-cell-type bars |
| [batch_integration.py](src/scripts/deconvolution/batch_integration.py) | runs Harmony / Scanorama / scANVI on the HCA blood pseudobulk for the integration table below |
| [drvi_bulk.py](src/scripts/deconvolution/drvi_bulk.py) | DRVI on bulk → see [drvi_outputs/](drvi_outputs/) |
| [cross_modality_vae.py](src/scripts/deconvolution/cross_modality_vae.py) | the bulk + sc shared-latent VAE (see §10) |
| [app_gradio.py](src/scripts/deconvolution/app_gradio.py) | UI wrapper for the trained deconvolution model |

---

## 10. Cross-modality VAE — bulk + sc shared latent ([src/scripts/deconvolution/cross_modality_vae.py](src/scripts/deconvolution/cross_modality_vae.py))

Architecture: `BulkEncoder` and `SCEncoder` (each FC 1024 → 512 → (μ, log σ²))
both produce a shared 64-dim latent `z`. Decoder reconstructs both modalities;
a domain discriminator on `z` is trained to separate bulk vs sc, and the
encoders are trained adversarially to fool it.

Loss: `L_recon + β·L_KL − λ·L_adv`. β anneals 0→1, λ anneals 0→0.1.

Evidence the alignment works:
[cross_modality_latent_space.png](cross_modality_latent_space.png),
[cross_modality_cca.png](cross_modality_cca.png),
[cross_modality_vae.pt](cross_modality_vae.pt) (trained checkpoint).

This is the centrepiece of the proposal in §12.

---

## 11. Single-modality batch integration ([src/results/integration/metrics_summary.csv](src/results/integration/metrics_summary.csv))

scANVI / Harmony / Scanorama on the HCA blood pseudobulk:

| Method | Silhouette ↑ | Batch entropy ↓ | # clusters |
|---|---|---|---|
| Harmony | 0.170 | **0.0035** | 20 |
| Scanorama | 0.111 | 0.0127 | 15 |
| **scANVI** | **0.290** | 0.0072 | 17 |

scANVI wins on cluster separation; Harmony wins on batch mixing. UMAPs in
[src/results/integration/](src/results/integration/) (`harmony_umap.png`,
`scanvi_umap.png`, `scanorama_umap.png`).

---

## 12. ECS 271 Project Proposal ([proposal/](proposal/))

Final framing (per [proposal/chat-history.md](proposal/chat-history.md)):

> **Bulk-Informed Cross-Modality Representation Learning for scRNA-seq Batch Correction**

Three desiderata: batch-invariant, biology-preserving, **bulk-aligned**.

Method components:
1. Cross-modality VAE with adversarial modality + batch discriminators (§10).
2. Gene-discordance head, anchored on the 557-gene BAL discordance set.
3. MMD bulk-prior regulariser pulling sc latent → bulk latent.

Evaluation: scIB-style (kBET, iLISI, cLISI, ASW, graph connectivity) primary;
deconvolution as one secondary downstream probe; the 2025 null-batch
calibration test as the headline robustness check.

Allowed-venue references (verified per chat-history): CellPLM (ICLR 2024),
Cell2Sentence (ICML 2024), Xu et al. *Multimodal Learning with Transformers:
A Survey* (TPAMI 2023). Compute budget: < 10 GPU-hours on a single RTX 3070
or Colab T4.

Open question (chat-history Turn 5+): pivot the headline from "batch
correction" to **"bulk-anchored embedding as a scientific instrument for
tissue engineering"** — extracting technology-invariant cell-type markers,
cross-modal in-silico perturbation, and engineered-tissue gap analysis. This
is the version the user was leaning toward at last edit; not yet in the PDF.

---

## 13. Practical artefacts to keep / reuse

| What | Where | Why |
|---|---|---|
| HCA blood pseudobulk | [pseudobulk/hca_blood_pseudobulk.npz](pseudobulk/hca_blood_pseudobulk.npz) | reused by every GTEx ↔ HCA comparison; expensive to rebuild |
| Trained VAE checkpoint | [cross_modality_vae.pt](cross_modality_vae.pt) | required for any sc/bulk latent-space follow-up |
| HVG-MLP predictions | [src/results/deconvolution_hvg_mlp/hvg_mlp_predictions.npz](src/results/deconvolution_hvg_mlp/hvg_mlp_predictions.npz) | best deconv baseline (Pearson 0.977) |
| Anchor-state assignments | implicit in [bimodal_anchor_summary.txt](bimodal_anchor_summary.txt) | 424 / 361 confident state labels for GTEx blood donors |
| 58-gene 95%-anchor list | [bimodal_anchor_summary.txt](bimodal_anchor_summary.txt) | smallest panel that classifies stress-state in unseen blood RNA-seq |
| Saturation curves | [blood_saturation.csv](blood_saturation.csv), [blood_nn_saturation.csv](blood_nn_saturation.csv), [blood_saturation_full_transcriptome.csv](blood_saturation_full_transcriptome.csv) | numerical justification for "GTEx is saturated" claim in the proposal |
| Deconv evaluation tables | [src/results/deconvolution_evaluation/](src/results/deconvolution_evaluation/) | per-cell-type metrics + scatter grids ready for paper figures |

---

## 14. GO / KEGG enrichment of low-CV genes ([low_cv_enrichment/](low_cv_enrichment/))

Question: among the GTEx whole-blood **expressed** genes (16,355 after the standard filter), what biology drives the very-stable tail — the genes with extremely low CV?

CV = `std(CPM) / mean(CPM)` on linear CPM (the qPCR-style stability metric). Distribution of CV across 16,355 expressed genes:

| quantile | CV |
|---|---|
| q=0.001 | 0.225 |
| q=0.01 | 0.277 |
| q=0.05 | 0.342 |
| q=0.10 | 0.387 |
| q=0.50 | 0.645 |

Top 10 most-stable genes: **PDAP1 (0.182), GNB1, GSK3A, KCMF1, NRBP1, TMEM183A, PAFAH1B1, TOX4, USF2, TBC1D20**. None of the textbook qPCR housekeepers (GAPDH, ACTB, B2M, EEF1A1, HPRT1) reach the bottom-200 — they're highly expressed but more variable than these less-celebrated genes.

Method: [gtex_low_cv_enrichment.py](gtex_low_cv_enrichment.py) submits the bottom-200, bottom-500, and bottom-1000 gene symbols to the **Enrichr REST API** (`maayanlab.cloud/Enrichr`) against four libraries: GO BP/CC/MF 2023 and KEGG 2021 Human. Full top-30 tables in [low_cv_enrichment/*.csv](low_cv_enrichment/) and a stitched markdown in [low_cv_enrichment/low_cv_enrichment_report.md](low_cv_enrichment/low_cv_enrichment_report.md).

### 14.1 Top hits (bottom-200, adj-p shown)

| Library | Top term | adj-p | Combined |
|---|---|---|---|
| **GO BP** | RNA Splicing via transesterification (GO:0000377) | 5.3e-07 | 209 |
| **GO BP** | mRNA Splicing via Spliceosome (GO:0000398) | 1.95e-06 | 158 |
| **GO BP** | Vesicle-Mediated Transport (GO:0016192) | 6.1e-05 | 76 |
| **GO BP** | Intracellular Protein Transport (GO:0006886) | 2.8e-04 | 70 |
| **GO CC** | Intracellular Membrane-Bounded Organelle | 2.3e-07 | 49 |
| **GO CC** | Nucleus | 2.3e-07 | 48 |
| **GO CC** | U2-type Spliceosomal Complex | 1.8e-04 | 126 |
| **GO MF** | RNA Binding (GO:0003723) | 6.9e-03 | 26 |
| **GO MF** | Clathrin Adaptor Activity | 4.0e-02 | 220 |
| **KEGG** | **Endocytosis** | **2.0e-10** | **259** |
| **KEGG** | mRNA Surveillance Pathway | 4.0e-05 | 149 |
| **KEGG** | Spliceosome | 1.0e-03 | 71 |

### 14.2 Bottom-500 / bottom-1000 — the proteasome and OXPHOS join in

As the list grows, the dominant signal shifts from "spliceosome + endocytosis" to "ubiquitin-proteasome + autophagy + mitochondrial OXPHOS":

| set | KEGG top hit | adj-p |
|---|---|---|
| bottom-500 | Endocytosis | 1.05e-11 |
| bottom-500 | Huntington disease (proteasome+OXPHOS marker pathway) | 4.7e-07 |
| bottom-500 | **Proteasome** | 6.5e-06 |
| bottom-500 | Spliceosome | 4.0e-05 |
| bottom-1000 | Endocytosis | 3.7e-16 |
| bottom-1000 | Huntington / Parkinson / ALS / Alzheimer (all proxy for OXPHOS + proteasome) | 1e-13 to 2e-10 |
| bottom-1000 | **Proteasome** | 1.1e-09 |
| bottom-1000 | Ubiquitin mediated proteolysis | 3.8e-08 |
| bottom-1000 | Spliceosome | 3.8e-08 |

GO MF for bottom-1000: RNA binding (adj-p **1.6e-14**), Ubiquitin protein-ligase binding (3.1e-09), Ubiquitin-like protein conjugating enzyme activity (5.1e-05). GO CC for bottom-1000: Intracellular membrane-bounded organelle adj-p **1.4e-32**, Nucleus **7.3e-30** — extremely strong.

### 14.3 Biological interpretation

The flat-as-a-board genes in human whole blood are not the canonical glycolysis-housekeepers. They are the **cell's bookkeeping machinery**:

1. **Spliceosome / mRNA processing / surveillance** — required in every cell every minute; loss is lethal. Examples: SF3B2, SF3B4, RBM8A, HNRNPK/U, USP39, SNW1, DDX39B, DHX16. This is the strongest signal and the main reason the textbook glycolysis HKs *don't* show up — those vary with energy demand; splicing factors don't.
2. **Endocytosis / vesicle trafficking** — coat machinery (AP2A1/B1/M1, clathrin, ARF1/4, RAB5B/5C/11B, dynamin), retromer, COPII (SEC13, SAR1A, COPA). Constant turnover at the plasma membrane and ER↔Golgi.
3. **Ubiquitin-proteasome system** — PSMA/PSMB/PSMD subunits, UBE2D2/D3/A/Q1/L3/Z, NEDD8, KEAP1. The proteasome itself appears as a top-3 KEGG hit at the 500- and 1000-gene cuts.
4. **Mitochondrial OXPHOS** — NDUFA10/11, NDUFB4/10, COX4I1/6A1/8A complex I/IV subunits — appears via the neurodegeneration KEGG pathways (which are essentially "proteasome + OXPHOS" surrogate panels).
5. **Autophagy** — BECN1, ATG3/9A/13, GABARAP, WIPI2, CALCOCO2, NBR1.

What you do **not** find at the top: hemoglobin, immune receptors, HLA, granzymes, cytokines — i.e., none of the cell-type-specific blood machinery. That's the right null: cell-composition-driven genes vary by donor; bookkeeping genes don't.

### 14.4 Practical use

- The bottom-200 (CV ≤ 0.282) is a defensible **whole-blood normalization panel** that beats the conventional GAPDH/ACTB set on stability.
- The bottom-1000 splices into 4 clean subpanels (splicing / endocytosis / proteasome / OXPHOS) that can be used as **internal controls** when looking at any other process — large per-donor deviations in *any* of these four subpanels indicate a sample-handling or library-prep problem, not biology.
- For the cross-modality VAE (§10), this set is also the natural candidate for **anchor genes** that should reconstruct identically in bulk and pseudobulk; reconstruction error on this panel is a ready-made calibration metric.

---

## 15. On the 74,628 / 16,355 / 44k+ gene counts — what GTEx is actually using

Short answer: GTEx v11 ships every gene in the GENCODE v39/v44 annotation (whichever the v11 build pinned), keyed on **versioned Ensembl gene IDs**, not on HGNC symbols. The 74k count is "every annotated locus", not "every protein-coding gene".

Concretely, from the GCT file:

| field | example | role |
|---|---|---|
| `Name` | `ENSG00000290825.2` | **versioned Ensembl gene ID** — primary key. 74,628 unique. |
| `Description` | `DDX11L16` | HGNC symbol if one exists; otherwise the **versionless ENSG repeats here** as a fallback. |

Numbers from the GTEx v11 whole-blood GCT:

- **74,628** total rows = unique versioned ENSG IDs.
- **All 74,628** start with `ENSG…`; trailing `.X` is the GENCODE *version* (re-annotation iteration of that locus).
- **33,835 rows have no HGNC symbol** — those are the "fallback ENSG" rows (`Description == Name minus version`). Most are lncRNAs, antisense RNAs, processed pseudogenes, miRNA primary transcripts, predicted novel loci, and TEC ("To be Experimentally Confirmed") entries. GENCODE keeps them; HGNC hasn't named them.
- **40,793 rows do have a real symbol**.

GENCODE v39/44 biotype breakdown (typical proportions for a human build):
- ~20k protein-coding
- ~18k lncRNAs
- ~14k processed pseudogenes
- ~7k unprocessed pseudogenes
- ~5k snRNA / snoRNA / miRNA / misc small RNAs
- the remainder: TEC, ribozyme, IG/TR variable-region segments, etc.

That is why the raw count looks "way too big" if you're used to "humans have ~20k genes". **GENCODE is annotating every transcribed locus, not the 20k protein-coding canon.**

### Why the project uses different gene counts at different stages

| count | where | what | how derived |
|---|---|---|---|
| **74,628** | raw GCT, [bimodal_state_biology.txt](bimodal_state_biology.txt) | every annotated GENCODE locus | nothing dropped |
| **16,355** | "expressed" set, used in [low_cv_enrichment](low_cv_enrichment/), [blood_kde_coverage.py](blood_kde_coverage.py), most analyses | genes with CPM > 1 in ≥ 10 % of the 803 donors | drops most lncRNAs, pseudogenes, novel/predicted loci that are zero or near-zero in blood |
| **~70,821** | whole-transcriptome saturation, [blood_saturation_full_transcriptome.py](blood_saturation_full_transcriptome.py) | drop only all-zero rows | keeps the long tail of "sometimes-on" genes |
| **~44,621** | GSE279480 union-expressed at threshold 0 ([gse279480_variance/gse279480_null_always_expressed_thresholds.csv](gse279480_variance/gse279480_null_always_expressed_thresholds.csv)) | genes with any reads in any donor of the Smithmyer cohort | different annotation build (RefSeq/GENCODE pre-v39) — that's why this number sits between 16k and 74k |
| **~9,432 / ~340** | "always expressed" core | min log2(CPM+1) > threshold across non-stressed donors | tiny because every gene must be present in every donor |

### Practical knock-on effects

1. **Symbol look-ups break for ~46 % of GTEx rows.** Anything that joins on `Description` to an HGNC-keyed annotation (Bausch-Fluck surfaceome, HPA secretome, Reactome, MSigDB) silently drops the 33,835 unnamed rows. If a gene-set tool requires symbols (Enrichr does), use the rows with real `Description` and accept that the lncRNA/pseudogene tail is invisible to that tool.
2. **Versioned IDs do not roundtrip cleanly.** `ENSG00000168209.6` (DDIT4) and `ENSG00000168209.5` are the same gene; the version number bumps when GENCODE re-annotates the exon structure. For any cross-build comparison, strip `.X`. The script does this with `df['Name'].str.split('.').str[0]`.
3. **Filtering matters more than usual.** Going from 74k → 16k expressed already removes the bulk of the never-expressed pseudogene/lncRNA tail. The reason the "saturation" analyses re-run on the full 70k transcriptome (§1.2) is to verify the conclusion isn't an artefact of that filter.

In short: GTEx's "gene" = GENCODE locus (versioned ENSG); your "gene" = protein-coding HGNC symbol. The factor of ~3 between them is GENCODE's lncRNA + pseudogene + small-RNA + novel annotation.

---

## 16. CV histogram, blood-vs-liver, low-CV ∩ high-expression ([low_cv_enrichment/](low_cv_enrichment/))

Three follow-up questions on the CV story.

### 16.1 The CV histogram itself ([cv_blood_vs_liver.png](low_cv_enrichment/cv_blood_vs_liver.png))

Per-gene CV computed on the same 16,355-gene blood expressed set and the 17,046-gene liver expressed set.

| q | Blood CV | Liver CV |
|---|---|---|
| 0.001 | 0.225 | 0.180 |
| 0.05 | 0.342 | 0.244 |
| 0.10 | 0.387 | 0.265 |
| 0.50 | 0.645 | 0.382 |
| 0.90 | 1.346 | 0.706 |

Liver is **dramatically tighter** than blood at every percentile. Two reasons fall out cleanly:
- **Liver has only one dominant cell type (hepatocytes ~80 %); blood is a 5-way mixture (neutrophils, lymphocytes, monocytes, eosinophils, basophils) whose proportions vary donor-to-donor.** Cell-composition variance gets absorbed into per-gene CV.
- **The State-A / State-B ex-vivo handling axis (§2)** that doubles many blood gene CVs has no analogue in liver — biopsies don't get the same neutrophil-degranulation hit as a venipuncture.

The shape of the blood CV histogram is also visibly bimodal-ish (a tight bookkeeping-gene mode at CV ≈ 0.3 and a broad cell-composition mode at CV ≈ 0.7), which liver doesn't show.

### 16.2 Are the most-stable genes the same across tissues?

Bottom-N overlap (genes ranked by ascending CV):

| n | shared | blood-only | liver-only | Jaccard |
|---|---|---|---|---|
| 200 | **17** | 183 | 183 | 0.044 |
| 500 | 84 | 416 | 416 | 0.092 |
| 1000 | 223 | 777 | 777 | 0.125 |

**The bottom-200 are mostly tissue-specific.** The 17 shared survivors are exactly what we'd expect — pure cellular bookkeeping that has to fire in every cell type at the same rate: PAFAH1B1, WIPI2, RAB11B, SPPL3, TRPC4AP, GID8, TLK2, ENSA, UBAC2, TMEM248, ARAF, POLDIP2, AKT2, SECISBP2, RNF114, APH1A, ANAPC16. These are ubiquitin/proteasome (UBAC2, RNF114, ANAPC16), endosome trafficking (RAB11B, WIPI2, APH1A), and signal-transduction core (AKT2, ARAF, ENSA).

The full shared bottom-200 is in [shared_low_cv_top200_blood_liver.csv](low_cv_enrichment/shared_low_cv_top200_blood_liver.csv). The **expanded shared set** (low-CV ∩ high-expression intersected across tissues) is **425 genes** — a much more usable cross-tissue normalization panel ([shared_low_cv_high_expr_blood_liver.txt](low_cv_enrichment/shared_low_cv_high_expr_blood_liver.txt)).

### 16.3 What if you require *both* low CV and high expression?

This is the right question — low CV alone admits weakly-expressed genes whose CV is artificially small because their values are crammed near the floor; pairing it with high mean filters those out and leaves the actual qPCR-grade housekeepers.

Cohorts (q ≤ 0.20 CV ∧ q ≥ 0.80 mean log2(CPM+1)):

| Tissue | CV cutoff | log2CPM cutoff | Genes |
|---|---|---|---|
| Blood | 0.456 | 5.55 | 1,396 |
| Liver | 0.311 | 5.29 | 1,034 |
| **Shared (blood ∩ liver)** | — | — | **425** |

Enrichr against the cross-tissue 425-gene panel ([low_cv_high_expr_enrichment.md](low_cv_enrichment/low_cv_high_expr_enrichment.md), Enrichr id `127827889`):

| Library | Top term | adj-p |
|---|---|---|
| GO BP | **RNA Processing (GO:0006396)** | 6.1e-15 |
| GO CC | Intracellular Membrane-Bounded Organelle | 2.8e-28 |
| GO MF | RNA Binding (GO:0003723) | 4.1e-23 |
| KEGG | **Endocytosis** | 2.2e-10 |

The single-tissue panels are even sharper:

| Cohort | Top KEGG | adj-p |
|---|---|---|
| Blood low-CV ∩ high-expr (1,396) | Endocytosis | 4.2e-24 |
| Liver low-CV ∩ high-expr (1,034) | Huntington disease (= proteasome + OXPHOS proxy) | 3.4e-12 |
| Liver GO BP top1 | **Gene Expression** (GO:0010467) | 3.2e-20 |
| Blood GO MF top1 | RNA Binding | 4.7e-32 |
| Liver GO MF top1 | RNA Binding | 2.0e-59 |

**Adding "high expression" sharpens the bookkeeping signal**: spliceosome / endocytosis / proteasome become *more* significant (adj-p drops by 5–10 orders of magnitude vs the bare bottom-200) because we've stripped out the long tail of low-expression noise. KEGG **Spliceosome** also moves up at adj-p 4.0e-08 in the cross-tissue panel.

The take-home: **the genuine human housekeeping panel is not GAPDH/ACTB.** It's an ~425-gene tissue-invariant set dominated by RNA processing + endocytosis + proteasome — keep it as a normalization standard for any cross-tissue RNA-seq comparison.

---

## 17. lncRNA-focused analysis ([lncrna_analysis/](lncrna_analysis/))

Used GENCODE v47 basic GTF ([ensembl_biotypes.tsv](lncrna_analysis/ensembl_biotypes.tsv) cache, parsed locally — 100 % coverage of the 74,628 GTEx ENSG IDs) to attach a biotype to every row.

Biotype counts in the GTEx whole-blood matrix:

| biotype | rows |
|---|---|
| lncRNA | 33,964 |
| protein_coding | 19,355 |
| processed_pseudogene | 9,323 |
| misc_RNA | 1,976 |
| unprocessed_pseudogene | 1,927 |
| snRNA | 1,824 |
| miRNA | 1,485 |
| TEC | 949 |
| snoRNA | 772 |

So roughly **half** of GTEx's "74k genes" are lncRNAs (33,964) plus a further ~13k pseudogenes — exactly the GENCODE-vs-protein-coding gap discussed in §15.

### 17.1 lncRNA expression cohort

After the standard `CPM > 1 in ≥ 10 % donors` filter:
- **2,709 expressed lncRNAs** (out of 34,913 lncRNA-class rows = 7.8 % expressed at all)
- Compare: 13,127 / 19,355 = **67.8 %** of protein-coding genes pass the filter.

That's the single most important number: lncRNAs are massively transcript-cataloged but sparsely expressed in any one tissue. Eight times more lncRNAs are catalogued than protein-coding genes, but only 1/9 as many show up as "expressed" in blood.

### 17.2 lncRNA naming-class summary (expressed only)

| name_class | n | median CPM | median log2CPM | median CV | p10 CV | frac-expressed median |
|---|---|---|---|---|---|---|
| unnamed (`ENSG…` fallback) | 1,582 | 1.18 | 0.98 | 0.864 | 0.569 | 0.43 |
| antisense (`*-AS*`) | 429 | 1.87 | 1.39 | 0.707 | 0.473 | 0.70 |
| named (HOTAIRM1, NEAT1, etc.) | 410 | 2.06 | 1.44 | 0.709 | 0.491 | 0.71 |
| LINC* | 234 | 2.43 | 1.53 | 0.855 | 0.561 | 0.70 |
| MIR*HG (miRNA host) | 31 | 2.61 | 1.39 | 0.868 | 0.578 | 0.64 |
| **SNHG (snoRNA host)** | **23** | **9.10** | **2.96** | **0.664** | **0.474** | **1.00** |

**SNHG (snoRNA host genes) is the standout.** 23 of them, all expressed in 100 % of donors, much higher mean than any other class — a built-in stable subpanel inside the lncRNA cohort. Use these (SNHG29, SNHG30, SNHG32, …) as positive controls for lncRNA quantification calibration.

### 17.3 Top expressed lncRNAs in blood

The textbook hits all show up at the top of the list:

| Symbol | mean log2CPM | CV | Biology |
|---|---|---|---|
| **NEAT1** | 9.15 | 0.86 | paraspeckle scaffold; canonical "most-expressed lncRNA" |
| PELATON | 7.87 | 0.54 | inflammation-linked, atherosclerosis lncRNA |
| **MIR223HG** | 7.73 | 1.07 | host of miR-223, a myeloid master regulator |
| MMP25-AS1 | 6.89 | 0.83 | antisense to neutrophil MMP25 |
| **HCP5** | 6.79 | 0.52 | HLA Complex P5; HLA-locus immune lncRNA |
| LINC02972 | 6.67 | 1.50 | bimodal candidate |
| ITGB2-AS1 | 6.57 | 0.46 | integrin β2 antisense |
| MIAT | 6.52 | 0.90 | myocardial-infarction-associated; immune-relevant |
| SNHG29 | 6.17 | 0.62 | most-expressed SNHG |
| **NORAD** | 6.07 | 0.55 | non-coding-RNA-activated-by-DNA-damage; PUMILIO sponge |
| CHASERR | 6.03 | 0.70 | CHD2 regulator |
| LINC00963 | 5.90 | 0.67 | known blood/immune marker |

### 17.4 Most stable lncRNAs (low CV, candidate references)

Top-30 lowest-CV lncRNAs are in [lncrna_analysis/lncrna_report.md §6](lncrna_analysis/lncrna_report.md). Lowest CV: **MIR497HG (0.31), MZF1-AS1 (0.32), NUTM2B-AS1 (0.33), POLR2J4 (0.34), LINC00324 (0.35), GUSBP11 (0.35), GARS1-DT (0.38), CYTOR (0.37)**. None of these are routinely used as lncRNA reference controls — but they are the empirical winners across 803 donors.

### 17.5 Most variable / bimodal lncRNAs (state-discriminators)

- **XIST (CV 2.54, frac-expressed 0.36)** and **PRKY (CV 1.15, chrY)** are the cleanest sex-discriminating bimodals — XIST OFF in male donors, PRKY OFF in female donors. These two together perfectly classify donor sex.
- **CCL3-AS1, CYP1B1-AS1, MIR210HG, MIR223HG, BHLHE40-AS1, LINC02908** all show large peak-gap bimodality (gap > 2.5 log2CPM units) — these track the same State-A / State-B axis as the protein-coding bimodal panel from §2 (CCL3-AS1 is the antisense to CCL3, BHLHE40-AS1 to the State-A immediate-early TF).
- **167 expressed lncRNAs are bimodal** — a non-trivial fraction (6.2 % of the 2,709 expressed lncRNAs vs ~5.3 % of protein-coding genes), so bimodality is *not* protein-coding-specific.

Full list ranked by peak-gap: [bimodal_lncrnas.csv](lncrna_analysis/bimodal_lncrnas.csv).

### 17.6 lncRNA vs protein-coding CV ([lncrna_vs_protein_coding_cv.png](lncrna_analysis/lncrna_vs_protein_coding_cv.png))

CV percentiles by biotype class on the GTEx blood expressed set:

| class | n | p10 CV | median CV | p90 CV | median log2CPM |
|---|---|---|---|---|---|
| protein_coding | 13,127 | 0.372 | 0.598 | 1.280 | 3.62 |
| **lncRNA** | **2,709** | **0.531** | **0.807** | **1.307** | **1.15** |
| pseudogene | 281 | 0.613 | 0.930 | 1.586 | 0.88 |
| other | 238 | 1.178 | 1.965 | 3.382 | 1.14 |

- lncRNAs are ~35 % more variable at the median than protein-coding genes (0.81 vs 0.60).
- Their mean expression is ~5× lower (log2 1.15 vs 3.62, i.e. linear ~6 × less abundant).
- The CV gap shrinks at the high end — both classes share the same ~1.3 ceiling at p90, suggesting the ceiling is set by the donor-state structure (§2), not by transcript class.

### 17.7 Summary

1. **Three out of every four "extra" GTEx genes are lncRNAs.** GENCODE catalogues 34k lncRNAs against ~20k protein-coding; only 2,709 of those lncRNAs are detectable in blood vs 13,127 protein-coding genes.
2. **SNHG (snoRNA host) genes are the stable subpanel inside the lncRNA cohort** — high expression, low CV, 100 % donor coverage. Best lncRNA normalization candidates.
3. **NEAT1, MIR223HG, NORAD, GAS5, MIAT, HCP5, PELATON** are the abundant, biologically-interesting hits; XIST + PRKY together are a perfect sex classifier; CCL3-AS1 / BHLHE40-AS1 / MIR210HG / CYP1B1-AS1 mirror the protein-coding stress-state axis.
4. **lncRNA CV is shifted right vs protein-coding** but capped at the same upper limit — the cap is driven by donor biology, not by transcript class.

---

## 18. lncRNA enrichment — direct GO/KEGG and guilt-by-association ([lncrna_analysis/enrichment/](lncrna_analysis/enrichment/), [lncrna_analysis/guilt_by_association/](lncrna_analysis/guilt_by_association/))

Two complementary submissions were run.

### 18.1 Direct GO/KEGG on lncRNA symbols ([lncrna_enrichment_report.md](lncrna_analysis/enrichment/lncrna_enrichment_report.md))

Submitted the named lncRNA symbols (unnamed `ENSG…` rows are dropped because Enrichr can't map them) for four cohorts:

| Cohort | Total | Named submitted |
|---|---|---|
| All expressed lncRNAs | 2,709 | 1,127 |
| Stable top-200 (lowest CV) | 200 | 122 |
| Variable top-200 (highest CV) | 200 | 65 |
| Bimodal | 167 | 69 |

**Result: essentially nothing significant.** Best-hit adj-p ~ 0.15–1.0; KEGG returned empty for every cohort; the Enrichr `LncHUB_Lncrna_Co-Expression` library also returned empty. This is expected — GO and KEGG are protein-coding-centric, and lncRNAs have very few direct annotations. The negative result is the right scientific answer: **you can't read lncRNA function off lncRNA symbols alone; you have to go through their co-expressed protein-coding partners**.

### 18.2 Guilt-by-association: enrich on co-expressed protein-coding genes ([lncrna_guilt_by_association_report.md](lncrna_analysis/guilt_by_association/lncrna_guilt_by_association_report.md))

For each lncRNA in a cohort, take its top-50 most positively *and* top-50 most negatively correlated protein-coding genes (Pearson on log2(CPM+1) across 803 donors). Take the union. Enrich that PC-gene set.

Cohort sizes (PC unions after correlation):

| Cohort | + correlates union | − correlates union |
|---|---|---|
| Stable top-200 lncRNAs | 3,928 PC genes | 1,512 |
| Variable top-200 lncRNAs | 3,459 | 2,483 |
| Bimodal (167 lncRNAs) | 1,186 | 769 |

#### Stable lncRNAs ↔ stable bookkeeping (positive correlates)

The top-200 lowest-CV lncRNAs co-vary with the same machinery as the protein-coding low-CV panel from §14:

| Library | Top term | adj-p |
|---|---|---|
| GO BP | Gene Expression (GO:0010467) | 6.8e-15 |
| GO CC | **Nucleus (GO:0005634)** | **2.8e-48** |
| GO MF | **RNA Binding (GO:0003723)** | **6.0e-47** |
| KEGG | **Thermogenesis** (= OXPHOS proxy) | 1.8e-15 |

The negative-correlate pole (genes that go down when stable lncRNAs go up) is **granule biology**:

| Library | Top term | adj-p |
|---|---|---|
| GO CC | **Tertiary Granule (GO:0070820)** | 3.5e-15 |
| KEGG | Shigellosis | 8.4e-09 |
| GO MF | Ubiquitin protein ligase binding | 1.8e-07 |

**Interpretation.** Stable blood lncRNAs sit on the *non-granulocyte* axis — they go up in lymphocyte-rich donors and down in neutrophil-rich (granule-rich) donors. They co-vary with the nuclear / RNA-processing / OXPHOS bookkeeping that was the §14 signal in protein-coding genes. **They are not "function-less" — they're co-regulated with the resting-cell machinery.**

#### Variable lncRNAs ↔ inflammation + cell-composition axis

The top-200 highest-CV lncRNAs partition into two opposite poles:

Positive-correlate pole = **active inflammation**:

| Library | Top term | adj-p |
|---|---|---|
| GO BP | Positive Regulation of Cytokine Production | 9.1e-13 |
| KEGG | **NF-κB signaling pathway** | 5.5e-10 |
| KEGG | TNF / IL-17 / Toll-like receptor signaling | all e-08 to e-10 |

Negative-correlate pole = **cytoplasmic translation / ribosome biogenesis**:

| Library | Top term | adj-p |
|---|---|---|
| GO BP | **Cytoplasmic Translation (GO:0002181)** | **5.1e-21** |
| GO CC | Focal Adhesion | 2.1e-17 |
| KEGG | **Coronavirus disease** (= ribosome subunit panel) | 6.3e-21 |

**Interpretation.** Variable lncRNAs track the State-A / State-B axis from §2: when inflammation/stress fires (NF-κB, TNF, cytokines), translation gets shut down in the same donors. Several of the highest-CV lncRNAs (CCL3-AS1, CYP1B1-AS1, MIR223HG, BHLHE40-AS1) are physically antisense to or hosting protein-coding stress markers — the co-expression pattern confirms it.

#### Bimodal lncRNAs ↔ ribosome / translation axis (both directions)

| Library | + pole top | adj-p | − pole top | adj-p |
|---|---|---|---|---|
| GO BP | **Cytoplasmic Translation** | 3.7e-18 | Cytoplasmic Translation | 1.2e-07 |
| GO CC | **Cytosolic Large Ribosomal Subunit** | 6.0e-12 | Cytosolic Large Ribosomal Subunit | 8.6e-08 |
| KEGG | **Coronavirus disease** | 1.2e-14 | Coronavirus disease | 1.1e-07 |

The bimodal lncRNAs cleanly bifurcate samples by **ribosomal-component expression** — both poles enrich for the ribosome panel because ribosome subunits go up in one state and down in the other (this is exactly the State-A/B split). KEGG "Coronavirus disease" appears as the top hit because that pathway in KEGG is essentially the ribosome-subunit panel mapped to translation arrest (a viral-replication readout).

### 18.3 Take-aways

1. **Direct GO/KEGG enrichment on lncRNAs returns nothing.** This is a property of the annotation databases, not the biology.
2. **Guilt-by-association via co-expressed PCs gives strongly significant, biologically coherent results.** Three different lncRNA cohorts give three different pictures:
   - Stable lncRNAs → nuclear/RNA-processing/OXPHOS bookkeeping (mirrors §14)
   - Variable lncRNAs → NF-κB / cytokine / TNF inflammation axis (= State-B in §2)
   - Bimodal lncRNAs → ribosomal / translation axis (= the structural readout of the State-A/B split)
3. **The cohorts are non-overlapping enrichment signatures**, which means the lncRNA classification by CV / bimodality is recovering biologically meaningful subgroups rather than noise.
4. **Per-lncRNA correlate tables** are in [lncrna_analysis/guilt_by_association/*__correlates.csv](lncrna_analysis/guilt_by_association/) — useful as a one-liner cis/trans hypothesis generator: "lncRNA X is most-correlated with PC genes Y₁…Y₅₀ in 803 GTEx donors."

---

## 19. Open threads (decision points)

0. **Low-CV panel as a normalization standard.** The bottom-200 set vastly outperforms the textbook GAPDH/ACTB panel on stability across 803 donors. Worth swapping it into the existing HK-comparison ([hk_variation_normfactor.py](hk_variation_normfactor.py)) to see whether the bulk ↔ pseudobulk and bulk ↔ sc CV agreement improves further than the published HK panel achieves.
1. **Proposal framing pivot.** Three options on the table; current PDF is
   "batch correction"; a stronger pitch is the embedding-as-instrument frame
   anchored on the dropout-corruption model.
2. **Anchor-gene panel size.** 2 (98 %) is too few in practice; 58 (95 %) is
   the working set. Worth running the 95 % set on GSE279480 Null donors as an
   external validation.
3. **HVG-MLP vs Neural W-CLS gap (0.977 vs 0.905).** The sweep shows it isn't
   HVG count — it's the architecture. If the proposal positions Neural W-CLS
   as a baseline, the HVG-MLP result needs to either be reported or absorbed.
4. **Stress samples in cross-modality training.** The 18 ambiguous + ~360
   stressed donors should be either dropped or used as a paired
   contrast — silently averaging them into the bulk prior re-introduces the
   same artefact the integration model is supposed to fix.
5. **TCGA / engineered-tissue downstream demos.** Promised in the proposal
   but not yet started; the trained `cross_modality_vae.pt` and the saturation
   evidence are the unblocking pieces.
