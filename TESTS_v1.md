# Validation tests and open questions — bulk blood healthy-state analysis

A single document to track which validation tests have been run, which are
still open, and where the data points us next. Companion to `HEALTHY_STATE_v1.md`
and `FINDINGS.md`. v1 — preliminary.

---

## Headline assertion: GSE279480 is the better reference dataset

Based on the cross-dataset comparison, **GSE279480 (Smithmyer 2025)
Null condition is a better basis for "healthy whole-blood reference"
than GTEx whole blood**. Reasons, ranked:

1. **Donors are living healthy adults**, not postmortem. GTEx samples
   carry an ex-vivo handling-stress signature on ~48 % of donors
   (State-B, see §2 of FINDINGS). GSE279480 is a controlled
   collection from 100 healthy adults — far less ex-vivo stress
   contamination expected (and visible in the smaller bimodal-gene
   set in the Null condition).
2. **Longitudinal design (~10 visits per donor over 2 years)** lets us
   separate technical-replicate variance from inter-donor variance —
   GTEx is single-shot per donor and conflates the two.
3. **4-condition design (Null / LPS / Poly I:C / SEB)** lets us contrast
   stimulated vs unstimulated to validate which signals are baseline.
4. **Per-donor matched stims** make it possible to anchor cross-modality
   comparisons on donor identity rather than donor-mean assumptions.
5. **Empirical clustering structure is stronger per holdout**: at every
   bandwidth, GSE279480 Null shows larger median largest-set sizes than
   GTEx (4 → 14 vs 1 → 6 across k_SD ∈ {2.0, 1.0, 0.5, 0.25, 0.1}),
   despite having ~3× fewer donors. That is the empirical signature of
   a cleaner, more donor-coherent cohort.
6. **Newer GENCODE annotation pipeline** (18,963 expressed genes vs
   GTEx's 16,355 — different gene-detection threshold but the cohorts
   agree on the ~13 k protein-coding intersection).

### Side-by-side at the standard band

| metric | GTEx (753 train, 50 holdout) | GSE279480 Null (239 train, 16 holdout) |
|---|---|---|
| expressed genes | 16,355 | **18,963** |
| at k_SD=0.25 — # in-range donors / gene | 31 (4.1 %) | 30 (12.6 %) |
| at k_SD=0.25 — median largest gene set / holdout | 3 | **7** |
| at k_SD=0.25 — biggest single set | 64 | 59 |
| at k_SD=0.10 — median largest gene set / holdout | 6 | **14** |

GSE279480 yields ~3× the gene-clustering signal per holdout despite a
much smaller training pool — that's a strong claim that its donor
biology is more *coherent*.

### Caveats — GSE279480 is not strictly better in every dimension

- **Library depth**: ~3.6 M reads/library (GSE279480) vs ~42 M (GTEx).
  Genes near the detection limit have much higher Poisson noise. The
  noise model (α=0.14) handles this, but it also explains why
  GSE279480 has many more genes in the empty-bucket at tight k_SD
  (798 at k=0.1 vs GTEx's 193).
- **Donor diversity**: GTEx 803 donors covers a much wider age and
  ancestry distribution. GSE279480 is 100 donors × 10 visits — better
  longitudinal but narrower demographic.
- **Sample count**: 803 vs 255 Null samples. For PCA-50 saturation,
  more is better (see §1 of FINDINGS).

The right v2 design probably trains a baseline on GSE279480 Null
(cleaner) and uses GTEx (broader) as the held-out validation.

---

## Tests already executed

### T1. Saturation of marginal / covariance / sample structure (FINDINGS §1)

**Done.** Three independent saturation tests on GTEx, all consistent:

| Test | Saturates at |
|---|---|
| Per-gene Gaussian z² (gene marginals) | n ≈ 200 |
| PCA-50 reconstruction MSE (covariance subspace) | n ≈ 500 |
| Nearest-neighbour MSE (find a near-twin) | n ≈ 600–700 |
| KDE 90 %-HDR coverage at n=700 | calibrated |

**Result.** 803 donors saturate every test. Adding more donors gives
diminishing returns at each level.

### T2. Per-gene technical-noise band coverage (FINDINGS §1.4b)

**Done.** Holdout passes the union of train ±2σ_tech bands at
99.95 %. Tightening bands to k_SD=0.1 still gives 97 % pass, with 540
genes having tight + reproducible structure (uncov > 0.5 AND all 50
holdouts inside).

### T3. Permutation null for the in-range clustering (this v1)

**Done.** Permute each gene's train values across donors (preserves
per-gene marginals; breaks gene-gene correlation). Compute # multi-gene
sets observed vs null. **Peak fold-over-null = 27× at k_SD=0.25**;
below k_SD ≈ 0.05, null beats observed (search space too restricted).

| k_SD | obs ≥2-gene sets | null | fold |
|---|---|---|---|
| 10.0 | 132 | 8 | 17× |
| 0.25 | 8 | 0.3 | **27× (peak)** |
| 0.1  | 57 | 23 | 2.5× |
| 0.05 | 227 | 282 | 0.80 (null wins) |
| 0.01 | 778 | 803 | 0.97 |

**Result.** k_SD ≈ 0.25 is the optimal restriction; tighter is not
better at the # of usable donors. We're already close to the
information-theoretic ceiling.

### T4. K-NN-tuple set clustering vs n_train (FINDINGS, this v1)

**Done.** Two complementary metrics:
- Random-null contribution collapses by n=400 (good diagnostic floor).
- Biological cluster structure persists from n=400 onwards (size-21
  RNA-processing cluster at n=753).

### T5. Pathway-level twin consistency (FINDINGS §18)

**Done.** Heat-shock genes 2.7× random; stress IEGs 1.8×; ribosomal
1.4×; OXPHOS / MHC-I at random baseline. Confirms that the train pool
encodes biology in donor strata, not as one similarity manifold.

### T6. Cross-dataset replication: GSE279480 Null vs GTEx (this v1)

**Done.** Same in-range / band sweep applied to both. GSE279480 has
~3× more gene-cluster structure per holdout despite smaller cohort
(see headline table). The biology is reproducible across datasets.

### T7. lncRNA guilt-by-association enrichment (FINDINGS §18)

**Done.** Direct GO/KEGG on lncRNAs returns nothing (annotation gap),
but co-expressed protein-coding partners give clean signatures: stable
lncRNAs co-expressed with RNA-processing/OXPHOS bookkeeping; variable
lncRNAs with NF-κB/cytokine inflammation; bimodal lncRNAs with
ribosomal translation. Three biologically distinct groups confirm the
classification picks real subgroups.

### T8. Cell-composition / aggregation confound (FINDINGS §4)

**Done.** GTEx (n=803, CV ≈ 0.25) vs HCA pseudobulk (n=8, CV ≈ 0.06):
the gap is mostly *aggregation effect* — pooling 803 GTEx donors into
8 groups recovers HCA-like CV. Not a "cleaner technology" effect.

---

## Tests still open — proposed for v2

### O1. Saturation of *state restriction* itself (the prompt for this doc)

**Question.** As we add more train samples, do the gene-clustering
metrics (in-range cluster sizes, search-space-limit fold-over-null,
neighbour-popularity specificity) saturate the way PCA-50 / NN-MSE /
KDE coverage do?

**Status.** Partial. We have the K-NN-tuple sweep across n_train
{100, 200, 300, 400, 500, 600, 700, 753} (FINDINGS) but only at K=5.
The in-range clustering sweep only spans {100, 200, 300, 500, 753}.
**Need.** Run the full bandwidth sweep × n_train sweep, fit a saturation
curve to each, and report the elbow per metric. Does state-restriction
saturate at the same n as PCA?

### O2. State-A-only subset analysis

**Question.** GTEx mixes State-A (baseline) and State-B (handling-stress)
donors. The ~58-anchor classifier (FINDINGS §2.3) splits them at >95 %
accuracy. After dropping the 361 State-B donors, do all the
saturation/clustering numbers tighten? Do new latent axes emerge?

**Need.** Re-run T1–T6 on the 424-donor State-A subset only.

### O3. Longitudinal within-donor variance (GSE279480)

**Question.** GSE279480 has ~10 visits per donor over 2 years.
Within-donor variance is technical + biological-baseline drift;
across-donor variance is the inter-individual axis. Decompose.

**Need.**
- Variance-components ANOVA per gene: σ²(donor) + σ²(time | donor).
- For each gene, ratio σ²(donor) / σ²(time): how stable is the
  per-individual baseline?
- Identify the genes with very HIGH within-donor variability
  (probably immune-state-driven) and the genes with very LOW (the
  real housekeepers).
- Filter the 425-cross-tissue housekeeping panel (FINDINGS §16) on the
  longitudinal stability test → curated stable panel.

### O4. Cell-composition correction

**Question.** Most of the high-CV genes in GTEx blood are cell-type-
specific (TCR, Ig V-segments, granule contents, hemoglobin family).
After explicit deconvolution + per-cell-type normalization, do they
still show outlier behaviour?

**Need.** Use the HVG-MLP deconvolution pipeline (`src/scripts/`),
predict per-donor cell-type fractions, regress per-gene values on
fractions, recompute saturation/clustering on the residuals.

### O5. Outlier-detection sensitivity / specificity

**Question.** The technical-noise band defines an "outlier" as a holdout
whose value falls outside the union. We see ~0.05 % per holdout.
**Is this reliable as a one-sample QC?** Inject simulated outlier values
at known levels and measure detection rate.

**Need.**
- Inject 1×, 2×, 5×, 10× σ_tech offsets to a held-out sample's gene values.
- Measure ROC of "outside-band" vs ground truth.
- Report TPR/FPR at clinically interesting thresholds.

### O6. Bandwidth sweep × n_train sweep

**Question.** The k_SD ≈ 0.25 optimum was found at n_train=743 with the
HOLDOUT=10 subsample. Does the optimal k shift with n_train?

**Need.** 2-D sweep (n_train × k_SD), peak fold-over-null at each n.
Hypothesis: optimal k tightens slightly as n grows (more donors → more
precision available at any k).

### O7. Demographic stratification (GTEx attributes)

**Question.** GTEx has age / sex / BMI / ancestry per donor. Does the
State-A / State-B axis correlate with any of them?

**Need.** Join with `GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt`,
fit logistic on stress-state ~ age + sex + BMI + ancestry. If any
covariate is significant, the saturation analysis must be conditioned
on it (otherwise we're describing a confounded reference).

### O8. Cross-cohort generalisation (model trained on one, scored on the other)

**Question.** Train the per-gene noise band on GSE279480 Null (cleaner
baseline). Apply to GTEx donors and measure outlier rates. Conversely
train on GTEx, apply to GSE279480.

**Need.**
- Build the band model on cohort A.
- Score each cohort-B donor's outlier rate.
- Plot outlier-rate distributions.
- Identify cohort-B donors that are dramatic outliers vs cohort A —
  these would be flagged as "non-baseline" by the v1 reference.

### O9. Within-donor stability of anchor genes (GSE279480 longitudinal)

**Question.** The 58 stress-state anchor genes (FINDINGS §2.3, GTEx-derived)
should NOT vary much within a single donor over time at baseline if they
truly mark stress-state. Test on the GSE279480 Null repeats.

**Need.** Extract the 58 anchors from GSE279480, plot per-donor
trajectories, report within-donor CV.

### O10. Saturation of *coverage* under tighter bands

**Question.** At k_SD=0.1, ~200 GTEx genes have 0 in-range donors per
holdout (the empty bucket). Does this fraction shrink as we add more
donors? (= "do we need more samples to cover the rare states?")

**Need.** Sweep n_train at fixed k_SD=0.1, plot empty-bucket size as
function of n. If it doesn't asymptote at n=753, that's evidence
saturation is **incomplete** at this band tightness even with all of
GTEx.

### O11. Replicate-based noise model recalibration

**Question.** The α=0.14 multiplicative-noise coefficient is from a
single reference table. Refit it from technical replicates if available.

**Need.** GSE279480 has biological replicates (per donor over time)
but **not** technical replicates (no library-prep replicates). Either
acquire technical replicates from a separate cohort or fit a hybrid
model that decomposes biological-baseline + technical noise from the
longitudinal data.

### O12. Sub-cohort robustness (random-subsample bootstrap)

**Question.** Are saturation curves stable under different random
holdouts? Currently we use a fixed seed (SEED=0). What if we draw 50
different seeded holdouts?

**Need.** Bootstrap 50 splits, recompute the headline numbers (PCA-50
test variance, in-range coverage, # multi-gene sets at peak k), report
mean ± std across splits.

### O13. Effect of library-size matching

**Question.** GTEx has 10 M – 170 M reads/library; GSE279480 has
~3.6 M. The technical-noise model handles depth but does the
saturation behaviour really survive a 50× depth difference?

**Need.** Subsample GTEx library to GSE279480 depth, re-run T1–T6.
If saturation is robust, the curves should overlap on the GSE279480
results.

### O14. The cross-modality bridge to single-cell

**Question.** With the bulk-saturated reference defined, can sc / pseudobulk
holdouts (HCA, Tabula Sapiens) be projected onto the same band-coverage
metric? That's the v2 goal of the cross-modality VAE in
`src/scripts/deconvolution/cross_modality_vae.py`.

**Need.** Project HCA pseudobulk onto the GTEx Null band model. Report
holdout pass-rate and identify the genes that systematically fall outside
the band on the sc side (= the genuine bulk↔sc discordance set).

---

## Priority

If the goal is a defensible v2 of the healthy-state reference,
the order I'd run them:

1. **O3** (longitudinal stability in GSE279480) — establishes whether
   GSE279480 should be the v2 reference cohort.
2. **O2** (State-A only on GTEx) — sanity-check that the GTEx mixed-state
   results don't change the qualitative picture.
3. **O8** (cross-cohort generalisation) — validates whichever cohort
   becomes the reference.
4. **O5** (outlier-detection ROC) — needed before any clinical / QC use.
5. **O1, O6, O10** (saturation refinements) — close the saturation
   story.
6. **O14** (sc bridge) — the eventual proposal use case.

Everything else is nice-to-have polish.

---

## Artefacts to consult

| File | What it has |
|---|---|
| [HEALTHY_STATE_v1.md](HEALTHY_STATE_v1.md) | The v1 healthy-state synthesis |
| [FINDINGS.md](FINDINGS.md) | Running technical log |
| [blood_technical_noise/](blood_technical_noise/) | All in-range / band-coverage CSVs and plots |
| [blood_technical_noise/in_range_GSE279480_Null.csv](blood_technical_noise/in_range_GSE279480_Null.csv) | The GSE279480 Null table |
| [blood_technical_noise/in_range_GSE279480_vs_GTEx.csv](blood_technical_noise/in_range_GSE279480_vs_GTEx.csv) | Side-by-side |
| [blood_technical_noise/search_space_limit_sweep.csv](blood_technical_noise/search_space_limit_sweep.csv) | Permutation null + fold table |
| [blood_in_range_gse279480_null.py](blood_in_range_gse279480_null.py) | The GSE279480 analysis script |
| [blood_search_space_limit.py](blood_search_space_limit.py) | The permutation-null sweep |
