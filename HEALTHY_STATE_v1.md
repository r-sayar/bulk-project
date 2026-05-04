# What does a healthy whole-blood transcriptome look like?

**v1 — preliminary.** This is the first synthesis pass. Numbers are GTEx v11
whole blood (803 donors), GENCODE v47 biotypes; technical-noise model
σ_tech(x) = √(x + (0.14·x)²); reference library size 42 M reads.

The findings below are preliminary and subject to revision once we add
the held-out validation cohort, run cross-tissue replication beyond
liver, and add an explicit cell-composition correction step. See the
appendix for the underlying analyses and caveats.

---

## TL;DR — the healthy-state fingerprint

A healthy whole-blood RNA-seq sample, defined empirically from 803 GTEx
donors, has these properties:

1. **~16,300 of the ~74,600 catalogued GENCODE loci are detectable**
   (CPM > 1 in ≥ 10 % of donors). That's **67.8 %** of protein-coding
   genes and **7.8 %** of lncRNAs.
2. **Two latent transcriptomic states** account for two-thirds of all
   variance (PC1 = 34 %, PC2 ≈ 16 %): a *baseline* state (~52 % of
   donors) and an *ex-vivo handling-stress* state (~48 %, marked by
   DDIT4 / JUN / HSPA1B / G0S2 / VEGFA / HILPDA / PLIN2 / CCL3). The
   stress state is a venipuncture / collection artefact and should
   probably be filtered out or modelled explicitly before downstream
   analysis.
3. **801–803 donors saturate the linear gene-covariance subspace.**
   PCA-50 captures 86.6 % of train variance and 84.0 % of held-out
   variance with only a 2.6-pp generalization gap (no over-fitting).
4. **A 425-gene tissue-invariant housekeeping panel** dominates by
   *RNA processing + endocytosis + proteasome* — not the textbook
   GAPDH/ACTB. This panel is the right normalization standard for
   cross-sample comparisons.
5. **58 anchor genes** classify any new donor into the two states with
   ≥ 95 % accuracy; **2 genes (DDIT4, FRAT1) classify with ≥ 98 %**.
6. **Donor-level "uniqueness" is concentrated in T-cell / B-cell repertoire
   (TCR / Ig V-segments) and a handful of metabolic / stress-tail genes**.
   These are the genes where biology *cannot* be saturated by adding more
   donors — every individual brings their own.

---

## 1. Composition — what genes are present?

| Category | Count | Expressed | % expressed |
|---|---|---|---|
| Total GENCODE rows | 74,628 | 16,355 | 21.9 % |
| Protein-coding | 19,355 | 13,127 | 67.8 % |
| lncRNA | 34,913 | **2,709** | 7.8 % |
| Pseudogene / small-RNA / TEC | 20,360 | ~520 | ~2.5 % |

Most genes in the catalogue are **not** transcribed in blood — the 74k
total is GENCODE annotating *every transcribed locus across all human
tissues and developmental stages*, including pseudogenes and tissue-
specific lncRNAs that are silent in peripheral blood.

**Highest-expressed gene: HBB** (hemoglobin β) at ~4.2 M counts per
sample on a 42 M-read library — about 10 % of the entire library. The
top-10 list is dominated by erythroid + myeloid / mitochondrial /
ribosomal / immune machinery: HBB, S100A9, ACTB, CSF3R, MT-RNR2,
MT-CO1, HLA-E, MT-ND4, FKBP8, IFITM2.

---

## 2. Variance structure — two latent states

PCA-50 on the expressed-gene log2(CPM+1) matrix (753 train donors):

| PCs | cumulative variance |
|---|---|
| PC1 | **34.0 %** |
| PC1–PC2 | **49.8 %** |
| PC1–PC10 | 73.0 % |
| PC1–PC25 | 81.8 % |
| PC1–PC50 | **86.6 %** |
| PC1–PC100 | 90.2 % |
| PC1–PC500 | 98.1 % (overfit; held-out caps at 89.6 %) |

**PC1 alone — 34 % of all variance — IS the ex-vivo handling axis.** PC2
adds another ~16 %, plausibly a cell-composition / neutrophil-fraction
axis. Two interpretable biological dimensions explain half of the entire
observed transcriptional variance.

### The two states

| Property | State A (baseline) | State B (handling-stress) |
|---|---|---|
| % of GTEx donors | ~52 % (424 / 803 confidently classified) | ~48 % (361) |
| ambiguous donors | 18 (~2 %) | — |
| Top markers | (relative) | **CXCL8** (×7 fold), **HILPDA** (×7), **JUN** (×5), **HSPA1B** (×5), **G0S2** (×5), **CCL3** (×5), **DDIT4** (×4), **VEGFA** (×4), **PLIN2** (×4) |
| Stress-gene panel mean log2FC (A vs B) | 0 | **−3.15** |
| HK panel mean log2FC | 0 | **+0.02** (no shift) |

Stress genes are >8× higher in State B; housekeeping genes don't move.
This is the classical signature of *ex-vivo* heat-shock / hypoxia /
inflammation induction during venipuncture — i.e. it's a measurement
artefact, not biology of interest.

### Anchor genes (state classifiers)

- **2 anchor genes at 98 % accuracy**: DDIT4 (HIGH in B), FRAT1 (HIGH in B)
- **58 additional anchors at 95 % accuracy**, including JUN, BHLHE40,
  G0S2, VEGFA, HILPDA, PLIN2, HBEGF, CCL3, MAFF, PPP1R15A (State-A high)
  and PTAFR, MED18, RER1, TADA3, P2RY13, TNFSF10 (State-B high).

Use these to triage any new whole-blood RNA-seq sample into the state
strata before downstream comparisons.

---

## 3. Saturation — are 803 donors enough?

Three independent saturation tests, all consistent:

| Test | Saturates at |
|---|---|
| Per-gene Gaussian z² | **n ≈ 200** (gene marginals) |
| PCA-50 reconstruction MSE | **n ≈ 500** (gene-gene covariance subspace) |
| Nearest-neighbour MSE | **n ≈ 600–700** (find a near-twin) |
| KDE 90 %-HDR coverage | calibrated at n=700 |

So 803 donors is comfortably above the elbow on every test. Adding
another 200 donors would yield diminishing returns at the gene-marginal
and covariance-subspace levels, and limited returns even at the
sample-twin level.

**Caveat — what saturation does *not* mean.** PCA-50 leaves ~16 % of
held-out variance unexplained, and that floor doesn't move even at
PCA-500. The residual is per-donor technical / Poisson / rare-state
noise that no linear model can capture. Nonlinear (VAE-style) models
plausibly reach lower than 16 % residual.

---

## 4. Per-gene technical-noise band coverage

For each gene, build the union of 753 per-sample ±2σ_tech bands and ask
whether held-out donors land inside.

| Statistic | Value |
|---|---|
| Mean fraction of holdout genes outside the union | **0.05 %** (≈ 8 of 16,355) |
| Range across 50 holdouts | 0 % – 0.89 % |

But this is partially trivial: the bands tile [0, max_train] for most
genes (median uncovered fraction = 1 %; only 0.3 % of genes have an
informatively-tight band structure). The strong saturation claim is
narrower: for ~50 bimodal genes (CD1A, EPOP, LAMP5, DEFB1, IGHV3-73,
…), the union is genuinely sparse (covers 30–40 % of [0, max]) AND
*all* 50 holdouts still fall inside the tight clusters.

### Bandwidth sweep (k in band = ±k·σ_tech)

| k | mean uncov | mean holdout-inside | tight ∧ all 50 in (uncov > 0.5 / > 0.7) |
|---|---|---|---|
| 2.0 (default) | 2.9 % | 99.86 % | 42 / 1 |
| 1.0 | 7.1 % | 99.74 % | 200 / 31 |
| **0.25** | **22.2 %** | **98.29 %** | **540 / 107** |
| 0.1 | 34.4 % | 96.93 % | 542 / 108 |

Even at k=0.1 (essentially pinpoint bands), 97 % of held-out gene
measurements still land inside the union of train bands. That's the
real saturation signal: the train pool densely populates the support of
every gene's distribution.

**The right operating point is k ≈ 0.25**: 540 genes are simultaneously
"tight + reproducible" (uncovered > 50 % AND all 50 holdouts inside).
107 of those are very tight (uncovered > 70 %).

---

## 5. The cross-tissue housekeeping panel

GO/KEGG enrichment on the *low-CV ∩ high-expression* intersection of
GTEx blood and GTEx liver (425 genes shared) is unambiguous:

| Library | Top term | adj-p |
|---|---|---|
| **GO BP** | RNA Processing (GO:0006396) | 6.1 × 10⁻¹⁵ |
| **GO MF** | RNA Binding (GO:0003723) | 4.1 × 10⁻²³ |
| **GO CC** | Intracellular Membrane-Bounded Organelle | 2.8 × 10⁻²⁸ |
| **KEGG** | **Endocytosis** | 2.2 × 10⁻¹⁰ |
| KEGG | Spliceosome | 4.0 × 10⁻⁸ |

The genuine cross-tissue housekeeping panel is **RNA processing +
endocytosis + proteasome**, not glycolysis. Specific gene families:

- **Splicing factors**: SF3B2/4, RBM8A, HNRNPK/U, USP39, SNW1, DDX39B, DHX16
- **Endocytosis / vesicle trafficking**: AP2A1/B1/M1, clathrin, ARF1/4,
  RAB5B/5C/11B, dynamin, COPII (SEC13, SAR1A, COPA)
- **Proteasome**: PSMA/PSMB/PSMD subunits, UBE2D2/D3/A/Q1/L3/Z, KEAP1
- **Mitochondrial OXPHOS**: NDUFA10/11, NDUFB4/10, COX4I1/6A1/8A
- **Autophagy**: BECN1, ATG3/9A/13, GABARAP, WIPI2

None of GAPDH/ACTB/B2M/EEF1A1/HPRT1 reach the bottom-200 panel. Use the
425-gene set instead.

Top stable lncRNAs (low-CV lncRNA panel, candidate reference controls):
**MIR497HG, MZF1-AS1, NUTM2B-AS1, POLR2J4, LINC00324, GUSBP11, GARS1-DT,
CYTOR**.

---

## 6. Per-donor outliers — where biology defeats saturation

Ratio of observed across-donor SD to predicted technical SD per gene
(from the band analysis): typical bookkeeping genes sit at ~1× (pure
technical noise), bimodal stress genes at 3–10×, and a small number of
genes are 10–35× above technical noise:

| Gene | obs SD / tech SD | Why |
|---|---|---|
| **NRAP** | **35×** | nebulin-related; tissue contamination (muscle) |
| **TRDV2** | **10×** | TCR δ-chain V — donor-specific repertoire |
| **IGHV3-30, IGHV3-33** | **15×, 12×** | Ig heavy-chain V — donor antibody repertoire |
| PARP15 | 7× | DNA-damage / immune; donor-specific |
| ACSM3 | 5× | mitochondrial fatty-acid; metabolic donor effect |
| HORMAD1, SNHG1, STIP1, NUDC, … | 4–8× | stress-axis tails / lncRNA |

**T-cell-receptor and immunoglobulin V-genes are constitutionally
donor-specific** (each individual generates them by VDJ recombination)
and constitute the irreducible lower bound on saturation. No matter how
big the cohort, these will keep showing up as "outside the band" for
every new donor.

The K-NN exact-tuple clustering at K=5 picks up the same signal: the
largest gene-sets that share their 5 nearest train donors are T-cell
modules (LCK, CD2, CD8A, KLRK1, TRAC, NFATC2), and the smaller sets
explicitly flag muscle (MYOZ1, PYGM, MYBPC2) and lung-surfactant
(SFTPC, SFTPA1, SFTPA2) contamination in specific holdouts.

---

## 7. Pathway-level twin consistency

For each (holdout h, gene g), the K nearest train donors form a set
NN(h, g). For pairs of genes, mean overlap of NN sets:

| Gene set | mean overlap | × random |
|---|---|---|
| Heat shock (HSPA1A/B, HSPB1, DNAJB1, BAG3, …) | 0.043 | **2.7×** |
| Top 1 % most-correlated train pairs | 0.034 | 2.1× |
| Stress IEGs (JUN, FOS, ATF3, EGR1) | 0.029 | 1.8× |
| Bimodal state anchors | 0.030 | 1.8× |
| Hemoglobin (HBA1/2, HBB) | 0.027 | 1.7× |
| Inflammation (CXCL8, CCL3/4, TNF) | 0.024 | 1.5× |
| Ribosomal RPL/RPS (158 genes) | 0.022 | 1.4× |
| Ig V-segments | 0.022 | 1.4× |
| MHC-I + B2M | 0.018 | 1.1× |
| OXPHOS | 0.018 | 1.1× |
| **random** | **0.016** | **1.0×** |

There are no global twin donors — pairwise overlap on random gene pairs
is at random baseline (0.016, equal to K/n_train). But pathway-coherent
groups are 1.5–3× elevated, confirming that **the train pool encodes
biology in donor strata** (stressed donors cluster, baseline donors
cluster, T-cell-rich donors cluster) rather than as one big similarity
manifold.

---

## 8. What "healthy" excludes — the v1 caveats

This v1 definition has known holes:

1. **The State-B donors are still in the GTEx cohort.** Strictly, "healthy
   baseline" should drop those 361 donors and rebuild every analysis on
   the 424 State-A subset. Most numbers above mix both states.
2. **Cell composition is not corrected.** A donor with 80 % neutrophils
   vs 50 % gives radically different bulk values for any cell-type-
   specific gene, and the variance between donors *is* mostly cell-
   composition variance. The bulk → sc deconvolution work in `src/`
   addresses this but isn't yet folded into this report.
3. **Other tissues**: the cross-tissue panel is blood ∩ liver only.
   Generalising the housekeeping panel needs more tissues.
4. **Validation cohort**: GSE279480 (Smithmyer 2025) is sitting in the
   project and recovers many of the same bimodal genes in its Null
   condition — but full external validation hasn't been written up
   here yet.
5. **Demographic confounders** (age, sex, BMI, ethnicity) are entirely
   ignored. State-A / State-B almost certainly correlates with one or
   more of these; needs sample-attribute joins.
6. **Repeat measurements** (technical replicates of the same donor)
   would let us separate technical noise from biological variability
   within-donor. Not in GTEx; would need a different cohort.
7. **Saturation is for the LINEAR description.** Nonlinear models (VAE)
   plausibly reach below the 16 % PCA residual floor and may reveal
   additional structure that requires more donors to saturate.
8. **The technical-noise model** (α=0.14) is taken from a single
   reference table. Should be re-fit empirically from technical
   replicates if available.

---

## 9. v2 / future work checklist

1. Drop State-B donors and rerun the saturation + housekeeping analyses
   on the State-A subset only. Expect tighter CV, sharper PCA, narrower
   bands.
2. Add explicit cell-composition deconvolution (HVG-MLP or scADen) and
   compute per-cell-type expression normality bands.
3. Extend cross-tissue housekeeping panel beyond blood + liver — at
   least muscle, brain, lung, kidney.
4. Validate on GSE279480 Null condition + on TCGA blood normals.
5. Wire in age / sex / BMI / ethnicity covariates.
6. Re-fit the noise model from any available technical replicates.
7. Train the cross-modality VAE (`src/scripts/deconvolution/cross_modality_vae.py`)
   on the State-A subset only and compare its latent residual to PCA.
8. Build a single "is this sample healthy?" probabilistic gate that
   combines state-classification (anchor genes) + per-gene noise-band
   coverage + cell-composition sanity check.

---

# Appendix — supporting analyses

The technical detail behind every claim above lives in [FINDINGS.md](FINDINGS.md).
The structure there:

| Section | Topic | Key artefact |
|---|---|---|
| §0 | Datasets | — |
| §1 | Saturation tests | [blood_saturation.csv](blood_saturation.csv), [blood_nn_saturation.csv](blood_nn_saturation.csv), [blood_kde_coverage_per_sample.csv](blood_kde_coverage_per_sample.csv), [blood_technical_noise/](blood_technical_noise/) |
| §1.4b | Per-sample noise bands + coverage audit | [per_gene_band_specificity.csv](blood_technical_noise/per_gene_band_specificity.csv), [band_sd_sweep_summary.csv](blood_technical_noise/band_sd_sweep_summary.csv) |
| §2 | Two transcriptomic states + anchor genes | [bimodal_state_biology.txt](bimodal_state_biology.txt), [bimodal_anchor_summary.txt](bimodal_anchor_summary.txt) |
| §3 | "Always expressed" gene set | (in script) |
| §4 | Sample-size / depth confounds | [pseudobulk_aggregation_test.png](pseudobulk_aggregation_test.png), [sample_size_bootstrap.png](sample_size_bootstrap.png) |
| §5 | Stress / housekeeping cross-modality | [stress_removal_comparison.png](stress_removal_comparison.png), [housekeeping_comparison.png](housekeeping_comparison.png) |
| §6 | Liver ↔ blood interaction | [gtex_liver_blood_interaction.png](gtex_liver_blood_interaction.png) |
| §7 | Other variance / structure analyses | [gtex_*.png](.) |
| §8 | GSE279480 stimulation panel | [gse279480_variance/](gse279480_variance/) |
| §9 | Deconvolution (HVG-MLP at Pearson 0.977) | [src/results/deconvolution_evaluation/](src/results/deconvolution_evaluation/) |
| §10 | Cross-modality VAE | [src/scripts/deconvolution/cross_modality_vae.py](src/scripts/deconvolution/cross_modality_vae.py), [cross_modality_vae.pt](cross_modality_vae.pt) |
| §11 | scANVI / Harmony / Scanorama integration | [src/results/integration/metrics_summary.csv](src/results/integration/metrics_summary.csv) |
| §12 | ECS 271 proposal | [proposal/](proposal/) |
| §14 | GO/KEGG enrichment of low-CV genes | [low_cv_enrichment/](low_cv_enrichment/) |
| §15 | Gene-identifier accounting (74k → 16k) | (in §1 of this report) |
| §16 | CV histogram + blood vs liver | [low_cv_enrichment/cv_blood_vs_liver_report.md](low_cv_enrichment/cv_blood_vs_liver_report.md) |
| §17 | lncRNA-focused analysis | [lncrna_analysis/](lncrna_analysis/) |
| §18 | lncRNA guilt-by-association enrichment | [lncrna_analysis/guilt_by_association/](lncrna_analysis/guilt_by_association/) |

### New analyses for this v1 report (not yet in FINDINGS.md)

| Topic | Script | Output |
|---|---|---|
| Per-sample technical-noise bands | [blood_technical_noise_outliers.py](blood_technical_noise_outliers.py) | [blood_technical_noise/](blood_technical_noise/) |
| Band-coverage audit (highest-expressed gene, decile breakdown) | [blood_band_coverage_audit.py](blood_band_coverage_audit.py) | [blood_technical_noise/per_gene_coverage_audit.csv](blood_technical_noise/per_gene_coverage_audit.csv) |
| Band specificity (1 - covered / max_train) | [blood_band_specificity.py](blood_band_specificity.py) | [blood_technical_noise/per_gene_band_specificity.csv](blood_technical_noise/per_gene_band_specificity.csv), [blood_technical_noise/blood_band_specificity.png](blood_technical_noise/blood_band_specificity.png) |
| Bandwidth sweep k ∈ {0.1, 0.25, …, 2.0} | [blood_band_sd_sweep.py](blood_band_sd_sweep.py) | [blood_technical_noise/band_sd_sweep_summary.csv](blood_technical_noise/band_sd_sweep_summary.csv), [blood_technical_noise/blood_band_sd_sweep.png](blood_technical_noise/blood_band_sd_sweep.png) |
| Per-(holdout, gene) K-NN train donors | [blood_per_gene_neighbor_consistency.py](blood_per_gene_neighbor_consistency.py) | [blood_technical_noise/per_holdout_neighbor_consistency.csv](blood_technical_noise/per_holdout_neighbor_consistency.csv) |
| Pathway-level twin overlap | [blood_pathway_twin_consistency.py](blood_pathway_twin_consistency.py) | [blood_technical_noise/pathway_twin_consistency.csv](blood_technical_noise/pathway_twin_consistency.csv) |
| Exact K-NN-tuple gene clustering | [blood_knn_set_clustering.py](blood_knn_set_clustering.py) | [blood_technical_noise/knn_exact_set_clustering_summary.csv](blood_technical_noise/knn_exact_set_clustering_summary.csv), [blood_technical_noise/knn_exact_set_largest_examples.csv](blood_technical_noise/knn_exact_set_largest_examples.csv) |
| Low-CV ∩ high-expression cross-tissue panel | [gtex_cv_blood_vs_liver.py](gtex_cv_blood_vs_liver.py), [gtex_low_cv_high_expr_enrichment.py](gtex_low_cv_high_expr_enrichment.py) | [low_cv_enrichment/](low_cv_enrichment/) |
| lncRNA biotype + cohort analysis | [gtex_lncrna_analysis.py](gtex_lncrna_analysis.py), [gtex_lncrna_vs_pc.py](gtex_lncrna_vs_pc.py) | [lncrna_analysis/](lncrna_analysis/) |
| lncRNA guilt-by-association | [gtex_lncrna_guilt_by_association.py](gtex_lncrna_guilt_by_association.py) | [lncrna_analysis/guilt_by_association/](lncrna_analysis/guilt_by_association/) |

---

*Reproducibility notes: every number in this report is regeneratable from
the scripts above. The GTEx GCT lives outside the repo at
`/Users/rls/Downloads/gene_reads_v11_whole_blood.gct.gz` and the GENCODE
v47 biotype map is cached at `lncrna_analysis/ensembl_biotypes.tsv`.*
