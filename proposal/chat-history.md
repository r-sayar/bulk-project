# ECS 271 ML Project Proposal — Chat History

Conversation about drafting the ECS 271 ML project proposal, reframing it, and digging into the underlying research direction.

---

## Turn 1 — User

> @/Users/rls/Downloads/ECS_271 ML - Project Proposal.docx @/Users/rls/Downloads/ECS 271 ML - Project Proposal.zip based on the bulk project please write me document for this:
>
> Please turn in your project proposal by the end of April 24.
>
> **Spec of the Proposed Project:**
> - A machine learning project either on any complex data (cannot be purely numerical data; modality should involve non-numerical data, such as text, images, structured data, etc), or time series.
> - Need to have technical contributions in machine learning or inference, cannot be purely analysis of models.
> - Project is viewed similarly as a scientific paper instead of an engineering effort. Values novel ideas, clever methods and rigorous experimentation; does not value engineering effort like building interfaces, APIs, or any type of application software.
>
> **Spec of the proposal:** 4 pages using the ACL format with unlimited references. Need to clearly describe:
> - Motivation
> - Problem definition
> - Method to be implemented
> - Description of dataset(s) and evaluation protocol (incl. metrics, baseline methods)
> - Make sure your method will conduct a fair comparison with the baseline methods
> - Estimation of computing resources (has to be executable within the course time)
> - Justified novelty (in the Related Work Section) based on three papers published in the following venues in the past three years:
>   - Main conferences (or Findings volumes) of ICML, ICLR, NeurIPS, ACL, EMNLP, NAACL, EACL, CVPR, ICCV or ECCV
>   - Journals including TACL, JMLR, TPAMI
>   - No other venues or workshop papers are allowed
>
> Please follow the exact section structure in the template.

## Turn 1 — Assistant

Built a 4-page ACL-format LaTeX proposal in `proposal/` based on the bulk-project's deconvolution work.

**Title:** *Bulk-Informed Cross-Modality Representation Learning for Calibrated Deconvolution of Bulk RNA-seq*

**Structure** (matching the template exactly):
- **Introduction** — motivates deconvolution + the 557-discordant-gene bias from the batch-effects report
- **Problem Definition and Method Design** — formalises the weighted-CLS objective; proposes a cross-modality VAE + adversarial modality discriminator + WeightNet + differentiable ridge-CLS head (built on existing `cross_modality_vae.py` and `Neural W-CLS v3`)
- **Related Work** — three citations from allowed venues:
  - CellPLM (ICLR 2024)
  - Cell2Sentence (ICML 2024)
  - Xu et al., *Multimodal Learning with Transformers: A Survey* (TPAMI 2023)
- **Evaluation Plan and Resources** — GSE84133/GSE50244 + GTEx + HCA, baselines (NNLS, NMF, Scaden, current W-CLS), ablations, 5-seed eval; <8 GPU-hours on a single RTX 3070/Colab T4

Files written to `proposal/`:
- `acl_latex.tex` — source
- `custom.bib` — three references
- `acl_latex.pdf` — compiled 4-page PDF

Verify the three citations against Google Scholar / official proceedings before submitting — author-order and exact paper titles in .bib files should be double-checked.

---

## Turn 2 — User

> remove the deconvolution part as the main thing

## Turn 2 — Assistant

Reframed the proposal so **batch correction / integration** of scRNA-seq is the main contribution, with deconvolution demoted to one secondary downstream evaluation task.

**New title:** *Bulk-Informed Cross-Modality Representation Learning for scRNA-seq Batch Correction*

**Framing highlights** (motivated by the batch-effects report):
- Opens with the 2025 null-condition calibration crisis (7/8 methods create artefacts) — the real gap in the field
- Positions bulk RNA-seq (GTEx v11, ~17k samples) as the untapped prior that no current method uses
- Three-way contribution: adversarial modality-invariant VAE + gene-discordance head (anchored on the 557 BAL genes) + MMD bulk-prior regulariser
- Primary evaluation is scIB-style integration benchmarking (kBET, iLISI, cLISI, ASW, graph connectivity) plus the 2025 null-batch calibration test
- Baselines: Harmony, Seurat RPCA, scVI, scANVI, Scanorama, ComBat-seq

Files updated at `proposal/`:
- `acl_latex.tex` — reframed source (4-page compiled PDF)
- `custom.bib` — same references + Hu et al. as data source

---

## Turn 3 — User

> I'm trying to figure out the last step. Why a shared embedding space between bulk and sc would be helpful, and how one could use that?
>
> What is the goal of the bulk-project?
>
> The three goals:
> - Transversal between bulk and single
> - Transversing between samples (Understanding the differences) — important because of reproducibility and as a first step for bulk and single transversal
> - Reverse function
>
> what will be the use cases:
> - better scRNA-seq prediction → just for "showing the world" that it helps
> - better explainability
> - better transferability / reproducibility
>
> but how will this be relevant to the progress of tissue engineering?
> ... will reduce noise and help with figuring out the genes which are actually relevant for a cell type
>
> tools: VAE for reverse function, for transversing between samples, lastly for bulk and single we need to (1) "set the values we know are stemming from biological changes because of the sequencing type" and (2) have to pre-adjust the values which stem from technical limitations (dropout)
>
> on (2) we have already done quite some research.
>
> for shared embedding space for bulk and single we need to understand the differences between them (potentially could simulate dropout the in bulk the low expressed genes and we need to compare the dropout)

## Turn 3 — Assistant

### The actual goal of the bulk-project

The three "goals" aren't parallel — they're hierarchical:

- **Reverse function** (latent → expression) is a *precondition*. Without a good decoder, any embedding is just a visualisation. It's a capability the other two need.
- **Sample-to-sample transversal** is *batch invariance*. Needed for reproducibility and needed before any cross-modal claim is trustworthy — if two sc datasets of the same tissue don't align, you can't anchor bulk to either.
- **Bulk ↔ sc transversal** is the *payoff* capability. It's what lets the project deliver scientific value beyond "another integration model."

So the project's goal, in one line: *learn a generative model of tissue biology that's calibrated against both bulk and single-cell measurements, so either modality can query the other at the cellular level.*

### Why a shared embedding actually helps

Not "because alignment is good" — because bulk and sc have **complementary** strengths:

| | Bulk | sc |
|-|-|-|
| Donor coverage | 10³–10⁴ (GTEx, TCGA) | 10¹–10² |
| Cell-type resolution | none | full |
| Technical noise | low, well-characterised | dropout, dissociation, ambient RNA |
| Gene-level variance | reliable | zero-inflated |

A shared embedding lets you **import bulk's statistical strength into sc** and **import sc's cellular resolution into bulk**.

### Concrete uses

1. **Cell-type-resolved TCGA** — project a TCGA bulk tumour into the shared space, read off cell-type composition and cell-type-specific gene expression. Can't do this today without matched sc, which TCGA doesn't have.
2. **Bulk-anchored batch correction** — two sc datasets from different labs should map to the same region in the shared space. Bulk is a third-party anchor independent of both labs' sc protocols. Fixes the 2025 calibration crisis.
3. **sc denoising** — encode sc → latent → decode; the decoder has learned clean bulk statistics, so the output is the bulk-prior-regularised version of the sc input.
4. **Technology-invariant markers** — genes with low reconstruction error in *both* modalities are robust cell-type markers.

### Tissue engineering — the actual stake

A tissue engineering lab needs three things: a target (what should the tissue look like), a cheap measurement (what does it look like), and a comparison (what's off, how to fix it).

Today the choice is binary: pay for sc on every engineered tissue (expensive, adds artefacts), or do bulk and fly blind on cell composition. A shared bulk↔sc space gives a third option:

- Measure your engineered tissue with **bulk** (daily, cheap)
- Project into the shared space, compare to the sc atlas of the target tissue
- Decompose the gap into (i) cell-composition errors, (ii) per-cell-type expression errors, (iii) pure technical noise
- Iterate the protocol

One-liner for the proposal motivation:

> *Bulk is what you can afford to measure every day; sc is what you need to interpret it. The shared space lets you do sc-level interpretation from bulk-level measurements.*

### The "last step" you're circling

**The shared embedding is useless without two decoders (or one conditional decoder).** What works:

- Encode bulk → latent → decode into *sc-like* output → cell-type-resolved inference from bulk
- Encode sc → latent → decode into *bulk-like* output → denoised expression using bulk as a prior

Every concrete use case reduces to one of those two paths.

### On the dropout-simulation idea — build the project around it

Simulating dropout on bulk to produce synthetic sc is the strongest single move you've listed. It gives you *supervised* pairs `(real_bulk, simulated_sc_from_that_bulk)` where you know the ground-truth bulk for every simulated sc. That flips a hard unsupervised alignment problem into a supervised regression:

- Train the sc encoder so that `simulated_sc_from(b)` maps to the same latent as `b` itself. Alignment by construction.
- At inference, real sc maps near this region because the simulator is close to the real corruption.
- **The corruption model itself becomes a deliverable** — a calibrated bulk → sc forward noise model.

Risk: the simulator has to be faithful. At minimum: per-gene dropout probability (fit from real sc), multiplicative count noise, library-size sampling, dissociation-stress upregulation term for the 512 known stress genes. Fit on matched bulk/sc pairs, validate on held-out matched pairs.

### Recommended proposal restructure

1. **Motivation**: tissue engineering needs cheap, interpretable measurement; bulk ↔ sc is the path
2. **Problem**: learn a bulk ↔ sc shared generative model with a calibrated corruption model (bulk → sc) and two decoders
3. **Method**: single-encoder-per-modality VAE, shared latent, two conditional decoders, adversarial alignment, discordance-supervised gene weighting, **plus the dropout-corruption model as a central component**
4. **Evaluation**: four concrete use cases — TCGA cell-type-resolved analysis, batch correction (scIB), sc denoising, synthetic "engineered-tissue gap analysis" benchmark. Batch correction is one metric among four, not the headline.

---

## Turn 4 — User

> have people used VAE for this problem already?

## Turn 4 — Assistant

Short answer: **yes, extensively** — VAE is the dominant architecture in this space, and a few groups have specifically attempted bulk ↔ sc latent spaces. The novelty has to come from the *details*, not the use of a VAE.

### VAEs for sc RNA-seq (sc-only — heavily trodden)

| Method | Year | Architecture | Goal |
|-|-|-|-|
| scVI (Lopez et al.) | 2018 | NB-VAE on sc counts | sc denoising / batch correction |
| scANVI (Xu et al.) | 2021 | scVI + label classifier | semi-supervised integration |
| scGen (Lotfollahi et al.) | 2019 | VAE + latent arithmetic | perturbation prediction |
| trVAE (Lotfollahi et al.) | 2020 | Conditional VAE + MMD | sc batch correction |
| totalVI | 2021 | Multi-output VAE | sc-RNA + protein |
| MultiVI | 2023 | VAE | sc-RNA + ATAC |
| DCA (Eraslan et al.) | 2019 | NB autoencoder | sc denoising |
| DESC, SAUCIE | 2019–20 | Regularised autoencoders | sc batch / clustering |

These are sc-only — bulk is not an input.

### VAEs that touch bulk and sc together (less trodden, but not empty)

- **Bulk2Space** (Liao et al., *Nat. Commun.* 2022) — β-VAE generates synthetic single cells from bulk + sc reference. Bulk → sc generative direction, not a symmetric shared encoder.
- **scDTL** (2024) — denoising autoencoder pretrained on bulk RNA-seq, then MMD-adapted to sc. Bulk pretraining → sc adaptation, unidirectional.
- **TAPE** (Chen et al., *Nat. Commun.* 2022) — autoencoder for tissue-adaptive deconvolution; trained on synthetic bulks from sc, fine-tunes per sample.
- **Scaden** (Menden et al., *Sci. Adv.* 2020) — MLP (not VAE) on synthetic sc-derived bulks.
- **SCRABBLE, scINRB, SIMPLEs-B, Bubble** — autoencoder/NMF hybrids constrained by bulk means.
- **CIBERSORTx, Bisque, DeMixSC, ReDeconv** — non-VAE bulk↔sc calibration.

### What's still open (the actual novelty surface)

1. **Symmetric two-decoder VAE** that goes both directions (bulk → sc-like and sc → bulk-like). Bulk2Space is one-way, scDTL is one-way.
2. **A learned, supervised bulk → sc *corruption* model** as the alignment signal, rather than MMD or adversarial loss alone.
3. **Discordance-supervised per-gene weighting** anchored on the 557 BAL genes inside a VAE.
4. **Tissue-engineering "gap analysis" use case** — none of the above are evaluated on engineered-tissue benchmarks.

### Honest implication

If the headline is "VAE for bulk + sc," a reviewer will respond with *Bulk2Space, scDTL, scVI variants — what's new?* Need the headline to be one of the items in the second list, with the VAE as the implementation, not the novelty. Anchor the novelty on **(2)** the bulk → sc corruption model.

---

## Turn 5 — User

> have those embeddings / encodings then been used to infer biological info from it?

## Turn 5 — Assistant

Surprisingly little — and that's actually the most defensible novelty hook in the whole project.

### What people *have* done with these embeddings

**Bucket 1: Engineering tasks** (the dominant use). 90%+ of the published work uses the embedding as a *preprocessing step* — kNN graph for Leiden clustering, UMAP for visualisation, query-to-reference label transfer. The embedding is a means to better clusters, not a scientific instrument. (scVI, scANVI, DESC, SAUCIE, scDTL.)

**Bucket 2: Latent arithmetic and generation** (some genuine inference).
- **scGen** (Lotfollahi 2019) — adds a "stimulation vector" in latent space to predict perturbation responses on unseen cell types.
- **Bulk2Space** (Liao 2022) — generates synthetic cells from bulk; inference (cell composition, marker discovery) happens on the generated cells via standard sc tools. The embedding itself is a transit station.
- **CellPLM, scGPT, Geneformer** — in-silico perturbation, gene-gene relationship prediction. sc-only.

**Bucket 3: Interpretable-by-construction latents** (the small minority that treats the embedding as the science).
- **MOFA+** (Argelaguet et al., *Genome Biol.* 2020) — multi-omics factor analysis where each factor is forced interpretable.
- **VEGA** (Seninge et al., *Nat. Commun.* 2021) — VAE with a sparse decoder anchored to known pathways.
- **expiMap** (Lotfollahi et al.) — knowledge-primed sc VAE; latent dimensions tied to gene-set memberships.
- **LDVAE** (Svensson et al.) — linearly-decoded VAE where each latent dimension's gene loadings are interpretable.

All sc-only or multi-omic sc.

### What's missing — the gap

For **bulk + sc shared embeddings specifically**, no published work that:

1. Uses the shared latent to **identify reproducible cell-type markers across modalities** (technology-independent markers).
2. Uses **bulk-side density vs. sc-side density** in the latent to flag novel or under-represented cell states.
3. Does **cross-modal in-silico perturbation** — perturb in latent space, decode through the bulk decoder vs. sc decoder.
4. Reports **cell-type-specific gene programmes derived directly from the bulk-anchored latent**.

### Implication for the proposal

The headline shouldn't be "another bulk + sc VAE" or even "batch correction." It should be:

> *A bulk-anchored sc latent space as a scientific instrument: extracting technology-invariant cell-type markers and cell-state structure usable for tissue engineering.*

The architecture (VAE, two decoders, dropout simulator, discordance head) is the means. The contribution is what you read *out* of the latent space:

- **Robust marker genes** = low reconstruction error in *both* modalities + high cell-type-specific loading
- **Discordant genes / artefact genes** = high reconstruction error in sc, low in bulk
- **Engineered-tissue gap** = vector in latent space from current to target; decode through bulk decoder for protocol-readable target, decode through sc decoder for cell-type-resolved diagnostic
- **Cross-modal perturbation prediction** = train scGen-style latent arithmetic on sc, apply to bulk inputs

Each is a concrete biological inference, none well-served by current methods, all are what tissue engineers need.

---

## Turn 6 — User

> i want to see the proposal again

## Turn 6 — Assistant

Showed the current 4-page proposal. Structural summary:

| Section | One-line content |
|-|-|
| **Title** | Bulk-Informed Cross-Modality Representation Learning for scRNA-seq Batch Correction |
| **Abstract** | Frames batch correction as headline; bulk as untapped prior; cross-modality VAE + discordance head + scIB evaluation |
| **§1 Motivation** | 2025 calibration crisis + 557 discordant genes + bulk as missing prior |
| **§2 Problem Definition** | Three desiderata: batch-invariant, biology-preserving, bulk-aligned |
| **§3 Method** | (1) Cross-modality VAE with adversarial modality + batch discriminators; (2) gene-discordance head; (3) MMD bulk-prior regulariser |
| **§4 Datasets and Evaluation** | GTEx + HCA + BAL + GSE84133/GSE50244; scIB metrics primary, deconvolution as secondary probe only |
| **§5 Computing Resources** | <10 GPU-hours, RTX 3070 / Colab T4, 7-week timeline |
| **§6 Related Work** | CellPLM (ICLR 2024), Cell2Sentence (ICML 2024), Xu et al. TPAMI 2023 |

Flagged: this version is still framed as "batch correction is the headline." If you want the "embedding-as-scientific-instrument for tissue engineering" spine (with dropout-corruption as central novelty and four downstream biological-inference results sections), that's the next pass.

---

## Turn 7 — User

> give me the papers

## Turn 7 — Assistant

### Currently in the proposal (custom.bib)

| # | Paper | Venue | Year | Allowed? | Confidence |
|-|-|-|-|-|-|
| 1 | **CellPLM: Pre-training of Cell Language Model Beyond Single Cells** — Wen, Tang, Dai, Ding, Jin, Xie, Tang | ICLR | 2024 | ✅ | High |
| 2 | **Cell2Sentence: Teaching Large Language Models the Language of Biology** — Levine, Rizvi, et al. | ICML | 2024 | ✅ | High |
| 3 | **Multimodal Learning with Transformers: A Survey** — Xu, Zhu, Clifton | TPAMI 45(10):12113–12132 | 2023 | ✅ | High |
| 4 | **BAL matched bulk/sc benchmark, 557 discordant genes** — Hu et al. | bioRxiv | 2026 | ❌ (data only) | Medium — verify |

### Mentioned in discussion (NOT in proposal — none in allowed venues)

| Paper | Venue | Year | What it does |
|-|-|-|-|
| **scVI** — Lopez, Regier, Cole, Jordan, Yosef | Nature Methods | 2018 | NB-VAE for sc, foundational |
| **scANVI** — Xu, Lopez, Mehlman, Regier, Jordan, Yosef | Mol. Syst. Biol. / NMI | 2021 | Semi-supervised scVI |
| **scGen** — Lotfollahi, Wolf, Theis | Nature Methods | 2019 | VAE + latent arithmetic for perturbation |
| **trVAE** — Lotfollahi et al. | Bioinformatics | 2020 | Conditional VAE + MMD batch correction |
| **DCA** — Eraslan, Simon, Mircea, Mueller, Theis | Nature Communications | 2019 | Deep count autoencoder for sc denoising |
| **Bulk2Space** — Liao et al. | Nature Communications | 2022 | β-VAE generates synthetic sc from bulk |
| **scDTL** | likely Nat. Commun. or Bioinformatics | 2024 | Bulk-pretrained autoencoder, MMD-adapted to sc |
| **TAPE** — Chen et al. | Nature Communications | 2022 | Tissue-adaptive autoencoder for deconvolution |
| **Scaden** — Menden et al. | Science Advances | 2020 | MLP deconvolution on synthetic bulks |
| **MOFA+** — Argelaguet et al. | Genome Biology | 2020 | Multi-omics factor analysis, interpretable factors |
| **VEGA** — Seninge et al. | Nature Communications | 2021 | Pathway-anchored sparse-decoder VAE |
| **expiMap** — Lotfollahi et al. | Nature Cell Biology | 2023 | Knowledge-primed sc VAE, latent dims = gene sets |
| **LDVAE** — Svensson, Gayoso, Yosef, Pachter | Bioinformatics | 2020 | Linearly-decoded VAE for interpretability |
| **Geneformer** — Theodoris et al. | Nature | 2023 | sc transformer foundation model |
| **scGPT** — Cui et al. | Nature Methods | 2024 | sc multi-omic foundation model |
| **CIBERSORTx** — Newman et al. | Nature Biotechnology | 2019 | Bulk↔sc deconvolution with signature adjustment |
| **DeMixSC** | Genome Research | 2025 | Discordant-gene-aware deconvolution |
| **Bisque** — Jew et al. | Nature Communications | 2020 | Gene-specific bulk↔sc transformations |
| **DISCERN** — Hrovatin et al. | likely Nat. Commun. | 2023 | Adversarial autoencoder for sc batch correction |
| **DANN (Domain-Adversarial NN)** — Ganin et al. | JMLR | 2016 | ✅ but >3 yr — too old to count |

### Caveats

- **Verify every citation** before submission — author orders, exact titles, year, page numbers.
- **The allowed-venue constraint is brutal for this field.** Single-cell ML lives at Nature Methods / Nature Communications / Genome Biology / Bioinformatics — almost none goes to ICML/NeurIPS/ICLR/TPAMI. The three picked (CellPLM, Cell2Sentence, Xu et al.) are essentially the *only* defensible citations from the allowed list for this exact topic in the past 3 years.
- **If you change the framing** (e.g. to "embedding-as-instrument" or to deconvolution-first), the third paper might want to swap.

---

## Open threads / decisions to make next

1. **Final framing** — three options on the table:
   - **Current**: batch correction as headline (in PDF now)
   - **Original**: deconvolution as headline (replaced)
   - **Proposed in Turn 3 / Turn 5**: "embedding-as-scientific-instrument for tissue engineering" with dropout-corruption model as central novelty and four downstream biological-inference results
2. **Verify the three allowed-venue citations** before submission.
3. **Decide whether to add a 4th component** — the bulk → sc corruption model — as the central novelty, displacing the discordance head's central role.
4. **Tissue-engineering motivation** — currently absent from the proposal text; would change the abstract and §1 entirely if added.
