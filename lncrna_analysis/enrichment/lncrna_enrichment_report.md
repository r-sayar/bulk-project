# lncRNA cohorts — Enrichr GO / KEGG enrichment

**Caveat:** GO and KEGG annotations are protein-coding-biased. lncRNA symbols that have no GO/KEGG mapping are silently dropped from the background, so the absolute hit-counts here are systematically low. Treat results qualitatively. The `LncHUB_Lncrna_Co-Expression` library compensates by mapping lncRNAs to the protein-coding genes they co-express with — those results are the more interpretable lncRNA-side enrichment.


## all_expressed (1127 named symbols, Enrichr id `127829212`)

### GO_Biological_Process_2023 (top 15)

| Term | adj-p | combined | overlap |
|---|---|---|---|
| rRNA Base Methylation (GO:0070475) | 1.00e+00 | 9.4 | NSUN5P1;NSUN5P2 |
| Protein C-linked Glycosylation Via 2'-alpha-mannosyl-L-tryptophan (GO:0018406) | 1.00e+00 | 5.8 | DPY19L2P2 |
| Protein C-linked Glycosylation Via Tryptophan (GO:0018317) | 1.00e+00 | 5.8 | DPY19L2P2 |
| Regulation Of Apoptotic DNA Fragmentation (GO:1902510) | 1.00e+00 | 5.8 | ST20 |
| Superoxide Anion Generation (GO:0042554) | 1.00e+00 | 2.4 | NCF1B;NCF1C |
| Transcription Initiation At RNA Polymerase I Promoter (GO:0006361) | 1.00e+00 | 3.1 | RRN3P2 |
| Response To UV-C (GO:0010225) | 1.00e+00 | 3.1 | ST20 |
| rRNA Methylation (GO:0031167) | 1.00e+00 | 1.7 | NSUN5P1;NSUN5P2 |
| Nucleotide-Sugar Metabolic Process (GO:0009225) | 1.00e+00 | 0.9 | CMAHP |
| Superoxide Metabolic Process (GO:0006801) | 1.00e+00 | 0.6 | NCF1B;NCF1C |
| Transcription By RNA Polymerase I (GO:0006360) | 1.00e+00 | 0.5 | RRN3P2 |
| Ribosomal Small Subunit Assembly (GO:0000028) | 1.00e+00 | 0.5 | RRP7BP |
| Protein Mannosylation (GO:0035268) | 1.00e+00 | 0.3 | DPY19L2P2 |
| Negative Regulation Of TORC1 Signaling (GO:1904262) | 1.00e+00 | 0.2 | CASTOR3P |
| Cellular Response To Amino Acid Stimulus (GO:0071230) | 1.00e+00 | 0.1 | CASTOR3P |

### GO_Cellular_Component_2023 (top 15)

| Term | adj-p | combined | overlap |
|---|---|---|---|
| NADPH Oxidase Complex (GO:0043020) | 1.00e+00 | 3.1 | NCF1B;NCF1C |
| Nuclear Inner Membrane (GO:0005637) | 1.00e+00 | 0.1 | DPY19L2P2 |
| SCF Ubiquitin Ligase Complex (GO:0019005) | 1.00e+00 | 0.0 | FBXL9P |
| Clathrin-Coated Endocytic Vesicle Membrane (GO:0030669) | 1.00e+00 | 0.0 | FCGR1BP |
| Clathrin-Coated Endocytic Vesicle (GO:0045334) | 1.00e+00 | 0.0 | FCGR1BP |
| Clathrin-Coated Vesicle Membrane (GO:0030665) | 1.00e+00 | 0.0 | FCGR1BP |
| Early Endosome Membrane (GO:0031901) | 1.00e+00 | 0.0 | FCGR1BP |
| Endocytic Vesicle Membrane (GO:0030666) | 1.00e+00 | 0.0 | FCGR1BP |
| cullin-RING Ubiquitin Ligase Complex (GO:0031461) | 1.00e+00 | 0.0 | FBXL9P |
| Intracellular Membrane-Bounded Organelle (GO:0043231) | 1.00e+00 | 0.0 | MIR9-1HG;RRN3P2;NFE4;ESRG;LINC-PINT;CMAHP |
| Nucleus (GO:0005634) | 1.00e+00 | 0.0 | MIR9-1HG;RRN3P2;NFE4;ESRG;LINC-PINT;CMAHP |
| Intracellular Non-Membrane-Bounded Organelle (GO:0043232) | 1.00e+00 | 0.0 | NSUN5P1;NSUN5P2;CMAHP |
| Nuclear Membrane (GO:0031965) | 1.00e+00 | 0.0 | DPY19L2P2 |
| Nuclear Lumen (GO:0031981) | 1.00e+00 | 0.0 | NSUN5P1;NSUN5P2 |
| Nucleolus (GO:0005730) | 1.00e+00 | 0.0 | NSUN5P1;NSUN5P2 |

### GO_Molecular_Function_2023 (top 15)

| Term | adj-p | combined | overlap |
|---|---|---|---|
| Superoxide-Generating NADPH Oxidase Activator Activity (GO:0016176) | 1.00e+00 | 11.7 | NCF1B;NCF1C |
| Oxidoreductase Activity, Acting On Paired Donors, With Incorporation Or Reduction Of Molecular Oxygen, Another Compound As One Donor, And Incorporation Of One Atom Of Oxygen (GO:0016716) | 1.00e+00 | 5.8 | CMAHP |
| Carbonate Dehydratase Activity (GO:0004089) | 1.00e+00 | 0.9 | CA5BP1 |
| Cysteine-Type Endopeptidase Activator Activity Involved In Apoptotic Process (GO:0008656) | 1.00e+00 | 0.3 | ST20 |
| Peptidase Activator Activity Involved In Apoptotic Process (GO:0016505) | 1.00e+00 | 0.3 | ST20 |
| Mannosyltransferase Activity (GO:0000030) | 1.00e+00 | 0.2 | DPY19L2P2 |
| RNA Polymerase Core Enzyme Binding (GO:0043175) | 1.00e+00 | 0.1 | RRN3P2 |
| Hydro-Lyase Activity (GO:0016836) | 1.00e+00 | 0.0 | CA5BP1 |
| Hexosyltransferase Activity (GO:0016758) | 1.00e+00 | 0.0 | DPY19L2P2 |
| Cis-Regulatory Region Sequence-Specific DNA Binding (GO:0000987) | 1.00e+00 | 0.0 | NFE4 |
| Protein Homodimerization Activity (GO:0042803) | 1.00e+00 | 0.0 | NFE4 |
| Transcription Cis-Regulatory Region Binding (GO:0000976) | 1.00e+00 | 0.0 | NFE4 |
| DNA-binding Transcription Factor Binding (GO:0140297) | 1.00e+00 | 0.0 | NFE4 |

### KEGG_2021_Human (top 15)

_no significant terms_

### LncHUB_Lncrna_Co-Expression (top 15)

_no significant terms_


## stable_top200 (122 named symbols, Enrichr id `127829221`)

### GO_Biological_Process_2023 (top 15)

| Term | adj-p | combined | overlap |
|---|---|---|---|
| Negative Regulation Of TORC1 Signaling (GO:1904262) | 3.29e-01 | 13.3 | CASTOR3P |
| Cellular Response To Amino Acid Stimulus (GO:0071230) | 3.29e-01 | 7.9 | CASTOR3P |
| Positive Regulation Of ATP-dependent Activity (GO:0032781) | 3.29e-01 | 6.7 | AHSA2P |
| SCF-dependent Proteasomal Ubiquitin-Dependent Protein Catabolic Process (GO:0031146) | 3.29e-01 | 5.7 | FBXL9P |
| Regulation Of TORC1 Signaling (GO:1903432) | 3.29e-01 | 4.5 | CASTOR3P |
| Negative Regulation Of TOR Signaling (GO:0032007) | 3.29e-01 | 3.9 | CASTOR3P |
| Proteasome-Mediated Ubiquitin-Dependent Protein Catabolic Process (GO:0043161) | 8.60e-01 | 0.1 | FBXL9P |

### GO_Cellular_Component_2023 (top 15)

| Term | adj-p | combined | overlap |
|---|---|---|---|
| SCF Ubiquitin Ligase Complex (GO:0019005) | 6.24e-01 | 3.2 | FBXL9P |
| cullin-RING Ubiquitin Ligase Complex (GO:0031461) | 6.57e-01 | 0.4 | FBXL9P |

### GO_Molecular_Function_2023 (top 15)

| Term | adj-p | combined | overlap |
|---|---|---|---|
| Carbonate Dehydratase Activity (GO:0004089) | 1.53e-01 | 35.2 | CA5BP1 |
| Hydro-Lyase Activity (GO:0016836) | 2.46e-01 | 5.1 | CA5BP1 |

### KEGG_2021_Human (top 15)

_no significant terms_

### LncHUB_Lncrna_Co-Expression (top 15)

_no significant terms_


## variable_top200 (65 named symbols, Enrichr id `127829230`)

### GO_Biological_Process_2023 (top 15)

| Term | adj-p | combined | overlap |
|---|---|---|---|
| Skeletal System Development (GO:0001501) | 7.59e-01 | 2.0 | GUSBP3 |
| Nervous System Development (GO:0007399) | 7.59e-01 | 0.2 | GUSBP3 |

### GO_Cellular_Component_2023 (top 15)

| Term | adj-p | combined | overlap |
|---|---|---|---|
| Intracellular Membrane-Bounded Organelle (GO:0043231) | 1.00e+00 | 0.0 | ESRG |
| Nucleus (GO:0005634) | 1.00e+00 | 0.0 | ESRG |

### GO_Molecular_Function_2023 (top 15)

_no significant terms_

### KEGG_2021_Human (top 15)

_no significant terms_

### LncHUB_Lncrna_Co-Expression (top 15)

_no significant terms_


## bimodal_all (69 named symbols, Enrichr id `127829238`)

### GO_Biological_Process_2023 (top 15)

| Term | adj-p | combined | overlap |
|---|---|---|---|
| Regulation Of Immune Response (GO:0050776) | 2.34e-01 | 5.6 | FCGR1BP |

### GO_Cellular_Component_2023 (top 15)

| Term | adj-p | combined | overlap |
|---|---|---|---|
| Clathrin-Coated Endocytic Vesicle Membrane (GO:0030669) | 5.79e-01 | 6.8 | FCGR1BP |
| Clathrin-Coated Endocytic Vesicle (GO:0045334) | 5.79e-01 | 4.7 | FCGR1BP |
| Clathrin-Coated Vesicle Membrane (GO:0030665) | 5.79e-01 | 4.3 | FCGR1BP |
| Early Endosome Membrane (GO:0031901) | 5.79e-01 | 2.8 | FCGR1BP |
| Endocytic Vesicle Membrane (GO:0030666) | 5.94e-01 | 1.6 | FCGR1BP |
| Early Endosome (GO:0005769) | 7.16e-01 | 0.4 | FCGR1BP |
| Endosome Membrane (GO:0010008) | 7.16e-01 | 0.3 | FCGR1BP |

### GO_Molecular_Function_2023 (top 15)

_no significant terms_

### KEGG_2021_Human (top 15)

_no significant terms_

### LncHUB_Lncrna_Co-Expression (top 15)

_no significant terms_
