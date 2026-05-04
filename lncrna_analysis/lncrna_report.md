# GTEx whole-blood lncRNA analysis

Source: GTEx v11 whole blood, 803 donors, 74,628 GENCODE rows total.

Biotype source: Ensembl REST `lookup/id` (cached at `lncrna_analysis/ensembl_biotypes.tsv`).


## 1. Biotype landscape

|                                    |   n_rows |
|:-----------------------------------|---------:|
| lncRNA                             |    33964 |
| protein_coding                     |    19355 |
| processed_pseudogene               |     9323 |
| misc_RNA                           |     1976 |
| unprocessed_pseudogene             |     1927 |
| snRNA                              |     1824 |
| miRNA                              |     1485 |
| transcribed_unprocessed_pseudogene |      981 |
| TEC                                |      949 |
| snoRNA                             |      772 |
| transcribed_processed_pseudogene   |      620 |
| rRNA_pseudogene                    |      481 |
| IG_V_pseudogene                    |      186 |
| IG_V_gene                          |      144 |
| transcribed_unitary_pseudogene     |      109 |


## 2. lncRNA cohort

- Rows classified as lncRNA-like: **34,913**
- After CPM>1.0 in ≥10% donors: **2,709**
- That's the lncRNAs the rest of the analysis runs on.


## 3. lncRNA naming-class summary (expressed only)

| name_class   |    n |   mean_cpm |   mean_log2cpm |   cv_median |   cv_p10 |   cv_p90 |   frac_expressed_median |
|:-------------|-----:|-----------:|---------------:|------------:|---------:|---------:|------------------------:|
| unnamed      | 1582 |      1.180 |          0.981 |       0.864 |    0.569 |    1.349 |                   0.430 |
| antisense    |  429 |      1.869 |          1.394 |       0.707 |    0.473 |    1.112 |                   0.704 |
| named        |  410 |      2.062 |          1.436 |       0.709 |    0.491 |    1.171 |                   0.714 |
| LINC         |  234 |      2.426 |          1.528 |       0.855 |    0.561 |    1.329 |                   0.703 |
| MIR_host     |   31 |      2.608 |          1.394 |       0.868 |    0.578 |    1.431 |                   0.638 |
| SNHG         |   23 |      9.096 |          2.962 |       0.664 |    0.474 |    0.929 |                   1.000 |


## 4. Top 30 most-expressed expressed lncRNAs

| symbol          | biotype   | name_class   | chrom   |   mean_cpm |   mean_log2cpm |    cv |   frac_expressed |
|:----------------|:----------|:-------------|:--------|-----------:|---------------:|------:|-----------------:|
| NEAT1           | lncRNA    | named        | chr11   |    778.775 |          9.153 | 0.855 |            1.000 |
| PELATON         | lncRNA    | named        | chr20   |    275.454 |          7.872 | 0.544 |            1.000 |
| MIR223HG        | lncRNA    | MIR_host     | chrX    |    441.388 |          7.731 | 1.065 |            1.000 |
| MMP25-AS1       | lncRNA    | antisense    | chr16   |    155.770 |          6.888 | 0.831 |            1.000 |
| HCP5            | lncRNA    | named        | chr6    |    126.141 |          6.794 | 0.519 |            1.000 |
| LINC02972       | lncRNA    | LINC         | chr12   |    210.763 |          6.672 | 1.498 |            0.998 |
| ITGB2-AS1       | lncRNA    | antisense    | chr21   |    104.477 |          6.568 | 0.460 |            1.000 |
| MIAT            | lncRNA    | named        | chr22   |    119.884 |          6.524 | 0.904 |            1.000 |
| SNHG29          | lncRNA    | SNHG         | chr17   |     83.512 |          6.166 | 0.622 |            1.000 |
| ENSG00000290937 | lncRNA    | unnamed      | chr3    |     81.713 |          6.089 | 0.626 |            1.000 |
| NORAD           | lncRNA    | named        | chr20   |     77.889 |          6.070 | 0.548 |            1.000 |
| CHASERR         | lncRNA    | named        | chr15   |     82.061 |          6.033 | 0.702 |            1.000 |
| GARS1-DT        | lncRNA    | named        | chr7    |     65.384 |          5.949 | 0.380 |            1.000 |
| LINC00963       | lncRNA    | LINC         | chr9    |     74.254 |          5.903 | 0.665 |            1.000 |
| OIP5-AS1        | lncRNA    | antisense    | chr15   |     57.578 |          5.764 | 0.403 |            1.000 |
| LUCAT1          | lncRNA    | named        | chr5    |     64.952 |          5.753 | 0.640 |            1.000 |
| GUSBP11         | lncRNA    | named        | chr22   |     55.014 |          5.723 | 0.350 |            1.000 |
| ENSG00000288156 | lncRNA    | unnamed      | chr19   |     79.979 |          5.718 | 0.851 |            1.000 |
| LINC-PINT       | lncRNA    | LINC         | chr7    |     60.821 |          5.652 | 0.697 |            1.000 |
| ENSG00000278600 | lncRNA    | unnamed      | chr15   |     60.237 |          5.613 | 0.766 |            1.000 |
| SNHG32          | lncRNA    | SNHG         | chr6    |     54.725 |          5.565 | 0.595 |            1.000 |
| PCED1B-AS1      | lncRNA    | antisense    | chr12   |     58.587 |          5.564 | 0.693 |            1.000 |
| FMNL1-DT        | lncRNA    | named        | chr17   |     46.782 |          5.447 | 0.426 |            1.000 |
| ENSG00000310473 | lncRNA    | unnamed      | chr2    |    102.568 |          5.386 | 1.360 |            0.998 |
| HCG27           | lncRNA    | named        | chr6    |     69.162 |          5.354 | 0.992 |            0.999 |
| FGD5-AS1        | lncRNA    | antisense    | chr3    |     45.155 |          5.352 | 0.498 |            1.000 |
| ENSG00000279933 | TEC       | unnamed      | chr22   |     43.345 |          5.342 | 0.421 |            1.000 |
| ENSG00000308579 | lncRNA    | unnamed      | chr1    |     46.659 |          5.342 | 0.581 |            1.000 |
| PCBP1-AS1       | lncRNA    | antisense    | chr2    |     44.735 |          5.306 | 0.590 |            1.000 |
| SLC39A13-AS1    | lncRNA    | antisense    | chr11   |     41.965 |          5.282 | 0.437 |            0.999 |


## 5. Top 30 highest-variance expressed lncRNAs (log2CPM std)

| symbol          | biotype   | name_class   | chrom   |   mean_cpm |   mean_log2cpm |    cv |   frac_expressed |
|:----------------|:----------|:-------------|:--------|-----------:|---------------:|------:|-----------------:|
| ENSG00000310473 | lncRNA    | unnamed      | chr2    |    102.568 |          5.386 | 1.360 |            0.998 |
| PRKY            | lncRNA    | named        | chrY    |     13.280 |          2.748 | 1.150 |            0.676 |
| ENSG00000288853 | lncRNA    | unnamed      | chr11   |     21.466 |          3.348 | 1.182 |            0.793 |
| XIST            | lncRNA    | named        | chrX    |      9.731 |          1.667 | 2.541 |            0.360 |
| ENSG00000287255 | lncRNA    | unnamed      | chr2    |     29.704 |          3.823 | 1.165 |            0.904 |
| CCL3-AS1        | lncRNA    | antisense    | chr17   |     10.147 |          2.311 | 1.349 |            0.575 |
| MIR223HG        | lncRNA    | MIR_host     | chrX    |    441.388 |          7.731 | 1.065 |            1.000 |
| LINC02207       | lncRNA    | LINC         | chr15   |     43.050 |          4.405 | 1.153 |            0.990 |
| MIR210HG        | lncRNA    | MIR_host     | chr11   |     11.608 |          2.489 | 1.509 |            0.638 |
| LINC02972       | lncRNA    | LINC         | chr12   |    210.763 |          6.672 | 1.498 |            0.998 |
| LINC01127       | lncRNA    | LINC         | chr2    |     63.668 |          5.083 | 1.040 |            0.989 |
| CYP1B1-AS1      | lncRNA    | antisense    | chr2    |     10.314 |          2.334 | 1.479 |            0.646 |
| LINC03078       | lncRNA    | LINC         | chr19   |     56.315 |          4.972 | 1.205 |            0.981 |
| ENSG00000290933 | lncRNA    | unnamed      | chr19   |      7.241 |          2.018 | 1.551 |            0.574 |
| SLED1           | lncRNA    | named        | chr4    |      6.069 |          1.768 | 1.588 |            0.532 |
| ENSG00000293441 | lncRNA    | unnamed      | chr5    |     19.770 |          3.504 | 1.030 |            0.964 |
| ENSG00000303148 | lncRNA    | unnamed      | chr15   |      5.466 |          1.225 | 2.523 |            0.324 |
| LINC02940       | lncRNA    | LINC         | chr21   |      8.799 |          2.266 | 1.644 |            0.672 |
| ENSG00000280138 | TEC       | unnamed      | chr12   |     38.930 |          4.437 | 1.158 |            1.000 |
| HCG27           | lncRNA    | named        | chr6    |     69.162 |          5.354 | 0.992 |            0.999 |
| ENSG00000289039 | lncRNA    | unnamed      | chr10   |     11.687 |          2.762 | 1.430 |            0.858 |
| ENSG00000289172 | lncRNA    | unnamed      | chr20   |     16.988 |          3.312 | 1.589 |            0.919 |
| PRADX           | lncRNA    | named        | chr11   |     20.973 |          3.614 | 1.226 |            0.993 |
| ENSG00000302221 | lncRNA    | unnamed      | chr7    |      4.751 |          1.567 | 1.755 |            0.501 |
| ENSG00000230492 | lncRNA    | unnamed      | chr20   |     14.009 |          3.079 | 1.258 |            0.898 |
| HLA-DRB6        | lncRNA    | named        | chr6    |     18.830 |          3.634 | 0.951 |            0.925 |
| LINC02289       | lncRNA    | LINC         | chr14   |     13.207 |          3.019 | 1.202 |            0.903 |
| ENSG00000290976 | lncRNA    | unnamed      | chr17   |      5.171 |          1.793 | 1.404 |            0.588 |
| ENSG00000290556 | lncRNA    | unnamed      | chr5    |     11.911 |          2.993 | 1.040 |            0.866 |
| GSEC            | lncRNA    | named        | chr11   |     31.941 |          4.369 | 0.897 |            1.000 |


## 6. Top 30 lowest-CV expressed lncRNAs (most stable)

| symbol          | biotype   | name_class   | chrom   |   mean_cpm |   mean_log2cpm |    cv |   frac_expressed |
|:----------------|:----------|:-------------|:--------|-----------:|---------------:|------:|-----------------:|
| MIR497HG        | lncRNA    | MIR_host     | chr17   |      9.401 |          3.318 | 0.314 |            1.000 |
| MZF1-AS1        | lncRNA    | antisense    | chr19   |     12.992 |          3.739 | 0.323 |            1.000 |
| NUTM2B-AS1      | lncRNA    | antisense    | chr10   |     19.094 |          4.252 | 0.333 |            0.999 |
| ENSG00000269958 | lncRNA    | unnamed      | chr14   |     11.756 |          3.601 | 0.335 |            1.000 |
| POLR2J4         | lncRNA    | named        | chr7    |     15.669 |          3.984 | 0.344 |            1.000 |
| LINC00324       | lncRNA    | LINC         | chr17   |      8.310 |          3.144 | 0.347 |            1.000 |
| GUSBP11         | lncRNA    | named        | chr22   |     55.014 |          5.723 | 0.350 |            1.000 |
| RABGEF1P1       | lncRNA    | named        | chr7    |     18.333 |          4.190 | 0.360 |            1.000 |
| ITGA6-AS1       | lncRNA    | antisense    | chr2    |      3.472 |          2.105 | 0.364 |            0.994 |
| ENSG00000268858 | lncRNA    | unnamed      | chr20   |      9.192 |          3.265 | 0.365 |            1.000 |
| PRANCR          | lncRNA    | named        | chr12   |      1.951 |          1.520 | 0.366 |            0.939 |
| RELA-DT         | lncRNA    | named        | chr11   |      4.591 |          2.418 | 0.369 |            1.000 |
| RBM15-AS1       | lncRNA    | antisense    | chr1    |      1.908 |          1.499 | 0.369 |            0.928 |
| CYTOR           | lncRNA    | named        | chr2    |     13.415 |          3.773 | 0.370 |            1.000 |
| LINC03009       | lncRNA    | LINC         | chr7    |      3.249 |          2.030 | 0.370 |            0.990 |
| ENSG00000303893 | lncRNA    | unnamed      | chr17   |      2.305 |          1.677 | 0.373 |            0.960 |
| ENSG00000278730 | lncRNA    | unnamed      | chr17   |      6.319 |          2.795 | 0.376 |            0.999 |
| ENSG00000291068 | lncRNA    | unnamed      | chr1    |      5.731 |          2.679 | 0.377 |            1.000 |
| ENSG00000278133 | lncRNA    | unnamed      | chr16   |      9.507 |          3.306 | 0.378 |            1.000 |
| ENSG00000291078 | lncRNA    | unnamed      | chr3    |      3.168 |          2.017 | 0.378 |            0.996 |
| KIF9-AS1        | lncRNA    | antisense    | chr3    |      1.421 |          1.239 | 0.379 |            0.793 |
| GARS1-DT        | lncRNA    | named        | chr7    |     65.384 |          5.949 | 0.380 |            1.000 |
| ENSG00000260257 | lncRNA    | unnamed      | chr20   |     16.522 |          4.042 | 0.381 |            1.000 |
| ENSG00000260927 | lncRNA    | unnamed      | chr16   |     23.209 |          4.497 | 0.387 |            1.000 |
| SNHG30          | lncRNA    | SNHG         | chr17   |      3.558 |          2.125 | 0.388 |            0.994 |
| CD27-AS1        | lncRNA    | antisense    | chr12   |      6.795 |          2.883 | 0.388 |            1.000 |
| ARRDC1-AS1      | lncRNA    | antisense    | chr9    |     12.701 |          3.683 | 0.391 |            1.000 |
| MIR3667HG       | lncRNA    | MIR_host     | chr22   |      8.813 |          3.203 | 0.393 |            1.000 |
| ENSG00000310523 | lncRNA    | unnamed      | chr2    |      5.626 |          2.652 | 0.395 |            0.998 |
| ENSG00000289161 | lncRNA    | unnamed      | chr8    |      6.758 |          2.880 | 0.396 |            1.000 |


## 7. Top 30 highest-CV expressed lncRNAs (most variable across donors)

| symbol          | biotype   | name_class   | chrom   |   mean_cpm |   mean_log2cpm |    cv |   frac_expressed |
|:----------------|:----------|:-------------|:--------|-----------:|---------------:|------:|-----------------:|
| ENSG00000281383 | lncRNA    | unnamed      | chr21   |      3.886 |          0.821 | 8.273 |            0.215 |
| ENSG00000261026 | lncRNA    | unnamed      | chr8    |      2.122 |          0.744 | 6.407 |            0.278 |
| ENSG00000276980 | lncRNA    | unnamed      | chr19   |      1.094 |          0.407 | 6.018 |            0.116 |
| ENSG00000300109 | lncRNA    | unnamed      | chr5    |      1.774 |          0.529 | 4.105 |            0.144 |
| ENSG00000309652 | lncRNA    | unnamed      | chr12   |      0.567 |          0.381 | 3.844 |            0.105 |
| UMODL1-AS1      | lncRNA    | antisense    | chr21   |      1.262 |          0.618 | 3.691 |            0.228 |
| ENSG00000285888 | lncRNA    | unnamed      | chr6    |      0.476 |          0.396 | 3.409 |            0.130 |
| RNASE2CP        | lncRNA    | named        | chr14   |      7.556 |          2.605 | 3.085 |            0.936 |
| H19             | lncRNA    | named        | chr11   |      4.310 |          1.467 | 3.025 |            0.512 |
| ENSG00000297875 | lncRNA    | unnamed      | chr19   |      0.598 |          0.448 | 2.925 |            0.126 |
| ENSG00000287535 | lncRNA    | unnamed      | chr8    |      3.581 |          0.996 | 2.865 |            0.318 |
| LINC01554       | lncRNA    | LINC         | chr5    |      1.549 |          1.013 | 2.791 |            0.407 |
| ENSG00000282024 | lncRNA    | unnamed      | chr6    |      1.332 |          0.888 | 2.620 |            0.329 |
| ENSG00000289933 | lncRNA    | unnamed      | chr9    |      2.082 |          1.053 | 2.587 |            0.396 |
| ENSG00000307683 | lncRNA    | unnamed      | chr20   |      1.619 |          0.933 | 2.584 |            0.308 |
| ENSG00000287097 | lncRNA    | unnamed      | chr6    |      0.775 |          0.533 | 2.546 |            0.162 |
| XIST            | lncRNA    | named        | chrX    |      9.731 |          1.667 | 2.541 |            0.360 |
| ENSG00000303148 | lncRNA    | unnamed      | chr15   |      5.466 |          1.225 | 2.523 |            0.324 |
| HLA-V           | lncRNA    | named        | chr6    |      2.607 |          0.747 | 2.522 |            0.192 |
| LINC02887       | lncRNA    | LINC         | chr17   |      1.801 |          1.106 | 2.482 |            0.405 |
| ENSG00000259986 | lncRNA    | unnamed      | chr15   |      0.629 |          0.527 | 2.472 |            0.148 |
| ENSG00000287927 | lncRNA    | unnamed      | chr8    |      0.365 |          0.295 | 2.450 |            0.105 |
| CTXN2-AS1       | lncRNA    | antisense    | chr15   |      0.434 |          0.327 | 2.446 |            0.123 |
| ANKRD36BP2      | lncRNA    | named        | chr2    |      0.545 |          0.492 | 2.425 |            0.111 |
| ENSG00000307670 | lncRNA    | unnamed      | chr4    |      0.494 |          0.446 | 2.345 |            0.130 |
| ENSG00000282917 | lncRNA    | unnamed      | chr4    |      0.695 |          0.503 | 2.305 |            0.181 |
| LINC03125       | lncRNA    | LINC         | chr20   |      0.517 |          0.444 | 2.287 |            0.116 |
| ESRG            | lncRNA    | named        | chr3    |      2.648 |          1.151 | 2.281 |            0.417 |
| THBS1-IT1       | lncRNA    | named        | chr15   |      0.677 |          0.537 | 2.271 |            0.193 |
| ENSG00000289561 | lncRNA    | unnamed      | chr15   |      1.087 |          0.651 | 2.224 |            0.265 |


## 8. Bimodal lncRNAs (top 30 by peak gap)

| ensg            | symbol          | biotype   | chrom   |   mean_log2cpm |   std_log2cpm |   n_peaks |   peak_low |   peak_high |   peak_gap |
|:----------------|:----------------|:----------|:--------|---------------:|--------------:|----------:|-----------:|------------:|-----------:|
| ENSG00000287255 | ENSG00000287255 | lncRNA    | chr2    |          3.823 |         1.983 |         2 |      1.301 |       5.826 |      4.525 |
| ENSG00000288853 | ENSG00000288853 | lncRNA    | chr11   |          3.348 |         2.019 |         2 |      0.761 |       5.185 |      4.424 |
| ENSG00000277089 | CCL3-AS1        | lncRNA    | chr17   |          2.311 |         1.945 |         2 |      0.300 |       4.184 |      3.884 |
| ENSG00000258476 | LINC02207       | lncRNA    | chr15   |          4.405 |         1.880 |         2 |      2.583 |       6.466 |      3.884 |
| ENSG00000274536 | MIR223HG        | lncRNA    | chrX    |          7.731 |         1.920 |         2 |      6.026 |       9.690 |      3.664 |
| ENSG00000232973 | CYP1B1-AS1      | lncRNA    | chr2    |          2.334 |         1.808 |         2 |      0.741 |       4.384 |      3.644 |
| ENSG00000229807 | XIST            | lncRNA    | chrX    |          1.667 |         1.992 |         2 |      0.320 |       3.944 |      3.624 |
| ENSG00000247095 | MIR210HG        | lncRNA    | chr11   |          2.489 |         1.879 |         2 |      0.521 |       3.964 |      3.443 |
| ENSG00000293441 | ENSG00000293441 | lncRNA    | chr5    |          3.504 |         1.698 |         2 |      1.862 |       5.285 |      3.423 |
| ENSG00000280832 | GSEC            | lncRNA    | chr11   |          4.369 |         1.504 |         2 |      2.603 |       5.846 |      3.243 |
| ENSG00000310473 | ENSG00000310473 | lncRNA    | chr2    |          5.386 |         2.095 |         2 |      3.984 |       7.067 |      3.083 |
| ENSG00000280138 | ENSG00000280138 | TEC       | chr12   |          4.437 |         1.648 |         2 |      2.903 |       5.946 |      3.043 |
| ENSG00000290933 | ENSG00000290933 | lncRNA    | chr19   |          2.018 |         1.714 |         2 |      0.300 |       3.303 |      3.003 |
| ENSG00000294355 | ENSG00000294355 | lncRNA    | chr22   |          1.803 |         1.421 |         2 |      0.501 |       3.504 |      3.003 |
| ENSG00000206344 | HCG27           | lncRNA    | chr6    |          5.354 |         1.597 |         2 |      4.044 |       6.987 |      2.943 |
| ENSG00000180539 | LINC02908       | lncRNA    | chr9    |          4.587 |         1.491 |         2 |      3.223 |       6.146 |      2.923 |
| ENSG00000259268 | ENSG00000259268 | lncRNA    | chr15   |          1.703 |         1.370 |         2 |      0.460 |       3.363 |      2.903 |
| ENSG00000290976 | ENSG00000290976 | lncRNA    | chr17   |          1.793 |         1.541 |         2 |      0.200 |       3.103 |      2.903 |
| ENSG00000291135 | FCGR1BP         | lncRNA    | chr1    |          2.319 |         1.396 |         2 |      0.941 |       3.784 |      2.843 |
| ENSG00000279884 | ENSG00000279884 | TEC       | chr2    |          1.844 |         1.352 |         2 |      0.641 |       3.443 |      2.803 |
| ENSG00000270640 | ENSG00000270640 | lncRNA    | chr2    |          1.750 |         1.384 |         2 |      0.360 |       3.083 |      2.723 |
| ENSG00000300307 | ENSG00000300307 | lncRNA    | chr5    |          2.921 |         1.415 |         2 |      1.502 |       4.184 |      2.683 |
| ENSG00000288612 | ENSG00000288612 | lncRNA    | chr6    |          3.055 |         1.421 |         2 |      1.682 |       4.364 |      2.683 |
| ENSG00000235027 | PRADX           | lncRNA    | chr11   |          3.614 |         1.578 |         2 |      2.122 |       4.785 |      2.663 |
| ENSG00000288156 | ENSG00000288156 | lncRNA    | chr19   |          5.718 |         1.417 |         2 |      4.605 |       7.247 |      2.643 |
| ENSG00000273179 | ENSG00000273179 | lncRNA    | chr4    |          2.203 |         1.386 |         2 |      0.761 |       3.363 |      2.603 |
| ENSG00000268595 | ENSG00000268595 | lncRNA    | chr19   |          2.386 |         1.299 |         2 |      1.181 |       3.784 |      2.603 |
| ENSG00000185168 | LINC00482       | lncRNA    | chr17   |          2.182 |         1.307 |         2 |      0.921 |       3.423 |      2.503 |
| ENSG00000272501 | ENSG00000272501 | lncRNA    | chr6    |          3.890 |         1.354 |         2 |      2.723 |       5.225 |      2.503 |
| ENSG00000235831 | BHLHE40-AS1     | lncRNA    | chr3    |          2.081 |         1.335 |         2 |      0.781 |       3.283 |      2.503 |


## 9. Known blood / immune lncRNAs found in this dataset

| ensg            | symbol    | ensembl_symbol   | chrom   | biotype   |   mean_cpm |   std_cpm |    cv |   mean_log2cpm |   std_log2cpm |   frac_expressed | name_class   |
|:----------------|:----------|:-----------------|:--------|:----------|-----------:|----------:|------:|---------------:|--------------:|-----------------:|:-------------|
| ENSG00000245532 | NEAT1     | NEAT1            | chr11   | lncRNA    |    778.775 |   665.548 | 0.855 |          9.153 |         1.151 |            1.000 | named        |
| ENSG00000274536 | MIR223HG  | MIR223HG         | chrX    | lncRNA    |    441.388 |   470.220 | 1.065 |          7.731 |         1.920 |            1.000 | MIR_host     |
| ENSG00000225783 | MIAT      | MIAT             | chr22   | lncRNA    |    119.884 |   108.365 | 0.904 |          6.524 |         1.050 |            1.000 | named        |
| ENSG00000260032 | NORAD     | NORAD            | chr20   | lncRNA    |     77.889 |    42.712 | 0.548 |          6.070 |         0.860 |            1.000 | named        |
| ENSG00000204054 | LINC00963 | LINC00963        | chr9    | lncRNA    |     74.254 |    49.390 | 0.665 |          5.903 |         1.025 |            1.000 | LINC         |
| ENSG00000234741 | GAS5      | GAS5             | chr1    | lncRNA    |     42.488 |    34.044 | 0.801 |          5.096 |         0.988 |            1.000 | named        |
| ENSG00000249859 | PVT1      | PVT1             | chr8    | lncRNA    |      8.648 |     5.719 | 0.661 |          3.054 |         0.788 |            0.996 | named        |
| ENSG00000226950 | DANCR     | DANCR            | chr4    | lncRNA    |      8.327 |     4.796 | 0.576 |          3.038 |         0.735 |            0.999 | named        |
| ENSG00000233429 | HOTAIRM1  | HOTAIRM1         | chr7    | lncRNA    |      5.081 |     3.186 | 0.627 |          2.421 |         0.729 |            0.976 | named        |
| ENSG00000214548 | MEG3      | MEG3             | chr14   | lncRNA    |      5.493 |     6.339 | 1.154 |          2.263 |         1.068 |            0.879 | named        |
| ENSG00000229807 | XIST      | XIST             | chrX    | lncRNA    |      9.731 |    24.724 | 2.541 |          1.667 |         1.992 |            0.360 | named        |
| ENSG00000130600 | H19       | H19              | chr11   | lncRNA    |      4.310 |    13.035 | 3.025 |          1.467 |         1.280 |            0.512 | named        |