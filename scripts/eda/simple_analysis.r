#nolint start
# script to perform standard workflow steps to analyze single cell RNA-Seq data
# data: Human pancreatic islet single-cell RNA-seq (GSE84133)
# data source: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM2230757

# Resolve project root (parent of scripts/)
get_project_root <- function() {
  # Try multiple methods to find the script location
  script_dir <- tryCatch(dirname(rstudioapi::getSourceEditorContext()$path), error = function(e) NULL)
  if (is.null(script_dir)) {
    args <- commandArgs(trailingOnly = FALSE)
    file_arg <- grep("^--file=", args, value = TRUE)
    if (length(file_arg) > 0) script_dir <- dirname(sub("^--file=", "", file_arg))
  }
  if (is.null(script_dir)) script_dir <- "scripts/eda"
  normalizePath(file.path(script_dir, "..", ".."), mustWork = FALSE)
}
project_root <- get_project_root()
setwd(project_root)

# load libraries
library(Seurat)
library(tidyverse)

# Load the GSE84133 human pancreatic islet dataset (CSV format)
# This CSV has cells as rows and genes as columns - need to transpose for Seurat

data <- read.csv("data/GSE84133_RAW/GSM2230757_human1_umifm_counts.csv", row.names = 1)

# Extract metadata (barcode and assigned_cluster)
cell_metadata <- data.frame(
  barcode = data$barcode,
  cell_type = data$assigned_cluster,
  row.names = rownames(data)
)

# Remove metadata columns to get just the count matrix
count_data <- data[, !(colnames(data) %in% c("barcode", "assigned_cluster"))]

# Transpose: Seurat expects genes as rows, cells as columns
count_matrix <- t(as.matrix(count_data))

# Create Seurat object
seurat_mtx <- CreateSeuratObject(counts = count_matrix, meta.data = cell_metadata)

# Add cell type information to the object
seurat_mtx$cell_type <- cell_metadata$cell_type

# 20125 features across 1937 samples



# 1. QC -------
View(seurat_mtx@meta.data)

# Note: This dataset does NOT contain mitochondrial genes (MT-*)
# They were filtered out during original processing
# We'll skip MT% filtering and use only nFeature/nCount QC

VlnPlot(seurat_mtx, features = c("nFeature_RNA", "nCount_RNA"), ncol = 2)
scatter_before <- FeatureScatter(seurat_mtx, feature1 = "nCount_RNA", feature2 = "nFeature_RNA") +
  geom_smooth(method = 'lm') +
  scale_y_continuous(n.breaks = 10)

# 2. Filtering -----------------
# Filter cells based on QC metrics (no MT filtering available)
seurat_mtx <- subset(seurat_mtx, subset = nFeature_RNA > 200 & nFeature_RNA < 5000)

scatter_after <- FeatureScatter(seurat_mtx, feature1 = "nCount_RNA", feature2 = "nFeature_RNA") +
  geom_smooth(method = 'lm')

# 3. Normalize data ----------
seurat_mtx <- NormalizeData(seurat_mtx)
str(seurat_mtx)


# 4. Identify highly variable features --------------
seurat_mtx <- FindVariableFeatures(seurat_mtx, selection.method = "vst", nfeatures = 2000)

# Identify the 10 most highly variable genes
top10 <- head(VariableFeatures(seurat_mtx), 10)

# plot variable features with and without labels
plot1 <- VariableFeaturePlot(seurat_mtx)
LabelPoints(plot = plot1, points = top10, repel = TRUE)


# 5. Scaling -------------
all.genes <- rownames(seurat_mtx)
seurat_mtx <- ScaleData(seurat_mtx, features = all.genes)

str(seurat_mtx)

plot2 <- VariableFeaturePlot(seurat_mtx)
LabelPoints(plot = plot2, points = top10, repel = TRUE)


# 6. Perform Linear dimensionality reduction --------------
seurat_mtx <- RunPCA(seurat_mtx, features = VariableFeatures(object = seurat_mtx))

# visualize PCA results
print(seurat_mtx[["pca"]], dims = 1:5, nfeatures = 5)
DimHeatmap(seurat_mtx, dims = 1, cells = 500, balanced = TRUE)


# determine dimensionality of the data
ElbowPlot(seurat_mtx)


# 7. Clustering ------------
seurat_mtx <- FindNeighbors(seurat_mtx, dims = 1:15)

# understanding resolution
seurat_mtx <- FindClusters(seurat_mtx, resolution = c(0.1, 0.3, 0.5, 0.7, 1))
#View(seurat_mtx@meta.data)

DimPlot(seurat_mtx, group.by = "RNA_snn_res.0.3", label = TRUE)

# non-linear dimensionality reduction --------------
# If you haven't installed UMAP, you can do so via reticulate::py_install(packages =
# 'umap-learn')
seurat_mtx <- RunUMAP(seurat_mtx, dims = 1:15)
# note that you can set `label = TRUE` or use the LabelClusters function to help label
# individual clusters
DimPlot(seurat_mtx, reduction = "umap")

# Compare with original cell type annotations from the dataset
DimPlot(seurat_mtx, reduction = "umap", group.by = "cell_type", label = TRUE)

# =============================================================================
# 8. Create pseudo-bulk matrix: genes x clusters (sum of expression per cluster)
# =============================================================================

# Use AggregateExpression to sum counts by cell type
# This creates a matrix with genes as rows and cell types as columns
pseudo_bulk <- AggregateExpression(
  seurat_mtx,
  group.by = "cell_type",
  assays = "RNA",
  slot = "counts",  # Use raw counts for summing
  return.seurat = FALSE
)

# Extract the matrix
pseudo_bulk_matrix <- as.data.frame(pseudo_bulk$RNA)

# View dimensions and preview
cat("Pseudo-bulk matrix dimensions:", dim(pseudo_bulk_matrix), "\n")
cat("Genes:", nrow(pseudo_bulk_matrix), "\n")
cat("Cell types:", ncol(pseudo_bulk_matrix), "\n")
cat("Cell types:", paste(colnames(pseudo_bulk_matrix), collapse = ", "), "\n")

# Preview first few genes and all cell types
head(pseudo_bulk_matrix)

# Save to CSV
write.csv(pseudo_bulk_matrix, "results/pseudo_bulk_by_celltype.csv", row.names = TRUE)
cat("Saved pseudo-bulk matrix to: results/pseudo_bulk_by_celltype.csv\n")

# =============================================================================
# 9. Count genes with zero expression per cell type
# =============================================================================

# Count zeros per column (cell type)
zeros_per_celltype <- colSums(pseudo_bulk_matrix == 0)

# Count non-zeros per column
nonzeros_per_celltype <- colSums(pseudo_bulk_matrix > 0)

# Calculate percentage of zeros
total_genes <- nrow(pseudo_bulk_matrix)
percent_zeros <- round(zeros_per_celltype / total_genes * 100, 2)

# Create summary table
zero_summary <- data.frame(
  cell_type = names(zeros_per_celltype),
  genes_with_zero = zeros_per_celltype,
  genes_expressed = nonzeros_per_celltype,
  total_genes = total_genes,
  percent_zero = percent_zeros
)
rownames(zero_summary) <- NULL

# Print summary
cat("\n=== Genes with zero expression per cell type ===\n")
print(zero_summary)

# Visualize
barplot(zeros_per_celltype, 
        main = "Number of genes with zero expression per cell type",
        ylab = "Number of genes with 0 expression",
        las = 2,  # Rotate labels
        col = "steelblue",
        cex.names = 0.8)

# =============================================================================
# 10. Compare zeros: bulk file vs cumulative cell type sums
# =============================================================================

# 1) Load bulk file and count zeros
bulk_data <- read.csv("data/bulk_GSM2230757_human1.csv")
bulk_zeros <- sum(bulk_data$expression == 0)
bulk_total <- nrow(bulk_data)
cat("\n=== Zeros in bulk_GSM2230757_human1.csv ===\n")
cat("Genes with zero expression:", bulk_zeros, "/", bulk_total, 
    "(", round(bulk_zeros/bulk_total*100, 2), "%)\n")

# 2) Progressive cumulative sum of cell types
# Sum columns progressively: 1, 1+2, 1+2+3, etc.
cell_types <- colnames(pseudo_bulk_matrix)
n_celltypes <- length(cell_types)

cumulative_results <- data.frame(
  n_celltypes_summed = integer(),
  celltypes_included = character(),
  zeros_remaining = integer(),
  percent_zero = numeric()
)

cumulative_sum <- rep(0, nrow(pseudo_bulk_matrix))

for (i in 1:n_celltypes) {
  # Add current cell type to cumulative sum
  cumulative_sum <- cumulative_sum + pseudo_bulk_matrix[, i]
  
  # Count zeros
  zeros <- sum(cumulative_sum == 0)
  
  cumulative_results <- rbind(cumulative_results, data.frame(
    n_celltypes_summed = i,
    celltypes_included = paste(cell_types[1:i], collapse = " + "),
    zeros_remaining = zeros,
    percent_zero = round(zeros / total_genes * 100, 2)
  ))
}

cat("\n=== Zeros remaining when progressively summing cell types ===\n")
print(cumulative_results)

# Plot the decrease in zeros
plot(cumulative_results$n_celltypes_summed, cumulative_results$zeros_remaining,
     type = "b", pch = 19, col = "darkred",
     main = "Zeros remaining as cell types are summed",
     xlab = "Number of cell types summed",
     ylab = "Genes with zero expression",
     xaxt = "n")
axis(1, at = 1:n_celltypes, labels = 1:n_celltypes)

# Add reference line for bulk file zeros
abline(h = bulk_zeros, col = "blue", lty = 2)
legend("topright", legend = c("Cumulative sum", "Bulk file"), 
       col = c("darkred", "blue"), lty = c(1, 2), pch = c(19, NA))

# =============================================================================
# 11. Pathway and Gene Category Analysis
# =============================================================================

# Install msigdbr if needed
if (!requireNamespace("msigdbr", quietly = TRUE)) {
  cat("Installing msigdbr package...\n")
  if (!requireNamespace("BiocManager", quietly = TRUE)) {
    install.packages("BiocManager")
  }
  BiocManager::install("msigdbr")
}
library(msigdbr)

# Get all available MSigDB collections for human
all_gene_sets <- msigdbr(species = "Homo sapiens")
cat("\n=== MSigDB Gene Set Collections ===\n")
print(unique(all_gene_sets[, c("gs_cat", "gs_subcat")]))

# Define categories to analyze
categories <- list(
  # Hallmark gene sets (well-defined biological states/processes)
  "H_HALLMARK" = msigdbr(species = "Homo sapiens", category = "H"),
  
 # Curated pathways
  "C2_KEGG" = msigdbr(species = "Homo sapiens", category = "C2", subcategory = "CP:KEGG"),
  "C2_REACTOME" = msigdbr(species = "Homo sapiens", category = "C2", subcategory = "CP:REACTOME"),
  "C2_WIKIPATHWAYS" = msigdbr(species = "Homo sapiens", category = "C2", subcategory = "CP:WIKIPATHWAYS"),
  "C2_BIOCARTA" = msigdbr(species = "Homo sapiens", category = "C2", subcategory = "CP:BIOCARTA"),
  "C2_PID" = msigdbr(species = "Homo sapiens", category = "C2", subcategory = "CP:PID"),
  
 # Gene Ontology
  "C5_GO_BP" = msigdbr(species = "Homo sapiens", category = "C5", subcategory = "GO:BP"),
  "C5_GO_MF" = msigdbr(species = "Homo sapiens", category = "C5", subcategory = "GO:MF"),
  "C5_GO_CC" = msigdbr(species = "Homo sapiens", category = "C5", subcategory = "GO:CC"),
  
  # Transcription Factor Targets
  "C3_TFT_GTRD" = msigdbr(species = "Homo sapiens", category = "C3", subcategory = "TFT:GTRD"),
  "C3_TFT_TFT_Legacy" = msigdbr(species = "Homo sapiens", category = "C3", subcategory = "TFT:TFT_Legacy"),
  
  # miRNA targets
  "C3_MIR_MIRDB" = msigdbr(species = "Homo sapiens", category = "C3", subcategory = "MIR:MIRDB"),
  "C3_MIR_MIR_Legacy" = msigdbr(species = "Homo sapiens", category = "C3", subcategory = "MIR:MIR_Legacy"),
  
  # Oncogenic signatures
  "C6_ONCOGENIC" = msigdbr(species = "Homo sapiens", category = "C6"),
  
  # Immunologic signatures
  "C7_IMMUNESIGDB" = msigdbr(species = "Homo sapiens", category = "C7", subcategory = "IMMUNESIGDB"),
  
  # Cell type signatures
  "C8_CELL_TYPE" = msigdbr(species = "Homo sapiens", category = "C8")
)

# Get all genes in our pseudo-bulk matrix
our_genes <- rownames(pseudo_bulk_matrix)

# Function to analyze gene set overlap and expression
analyze_category <- function(gene_set_df, category_name, pseudo_bulk, our_genes) {
  
  # Get unique genes in this category
  category_genes <- unique(gene_set_df$gene_symbol)
  
  # Find overlap with our data
  overlap_genes <- intersect(our_genes, category_genes)
  
  # Get number of gene sets
  n_gene_sets <- length(unique(gene_set_df$gs_name))
  
  # Calculate mean expression per cell type for overlapping genes
  if (length(overlap_genes) > 0) {
    subset_expr <- pseudo_bulk[overlap_genes, , drop = FALSE]
    mean_expr_per_celltype <- colMeans(subset_expr)
    total_expr_per_celltype <- colSums(subset_expr)
    
    # Count zeros in this category
    zeros_in_category <- sum(subset_expr == 0)
    total_values <- length(overlap_genes) * ncol(subset_expr)
    
    return(list(
      category = category_name,
      n_gene_sets = n_gene_sets,
      total_genes_in_category = length(category_genes),
      genes_in_our_data = length(overlap_genes),
      overlap_percent = round(length(overlap_genes) / length(category_genes) * 100, 1),
      zeros_count = zeros_in_category,
      zeros_percent = round(zeros_in_category / total_values * 100, 1),
      mean_expr = mean_expr_per_celltype,
      total_expr = total_expr_per_celltype,
      genes = overlap_genes
    ))
  } else {
    return(NULL)
  }
}

# Analyze all categories
cat("\n=== Analyzing gene categories ===\n")
category_results <- list()

for (cat_name in names(categories)) {
  cat("Processing:", cat_name, "...\n")
  result <- analyze_category(categories[[cat_name]], cat_name, pseudo_bulk_matrix, our_genes)
  if (!is.null(result)) {
    category_results[[cat_name]] <- result
  }
}

# Create summary table
summary_table <- data.frame(
  category = sapply(category_results, function(x) x$category),
  n_gene_sets = sapply(category_results, function(x) x$n_gene_sets),
  total_genes = sapply(category_results, function(x) x$total_genes_in_category),
  genes_found = sapply(category_results, function(x) x$genes_in_our_data),
  overlap_pct = sapply(category_results, function(x) x$overlap_percent),
  zeros_pct = sapply(category_results, function(x) x$zeros_percent)
)
rownames(summary_table) <- NULL

cat("\n=== Gene Category Summary ===\n")
print(summary_table)

# Create expression heatmap data for categories
expr_by_category <- do.call(rbind, lapply(category_results, function(x) x$mean_expr))
rownames(expr_by_category) <- names(category_results)

cat("\n=== Mean Expression per Cell Type by Category ===\n")
print(round(expr_by_category, 2))

# Heatmap of category expression across cell types
if (requireNamespace("pheatmap", quietly = TRUE)) {
  library(pheatmap)
  
  # Log transform for better visualization
  expr_log <- log2(expr_by_category + 1)
  
  pheatmap(expr_log,
           main = "Mean Gene Expression by Category and Cell Type (log2)",
           cluster_rows = TRUE,
           cluster_cols = TRUE,
           scale = "row",
           fontsize_row = 8,
           fontsize_col = 10)
} else {
  # Base R heatmap
  heatmap(log2(expr_by_category + 1), 
          main = "Mean Expression by Category (log2)",
          scale = "row",
          cexRow = 0.7)
}

# Save category results
write.csv(summary_table, "results/gene_category_summary.csv", row.names = FALSE)
cat("\nSaved category summary to: results/gene_category_summary.csv\n")

# =============================================================================
# 11b. Specific Gene Lists (Transcription Factors, Kinases, etc.)
# =============================================================================

# Extract specific functional gene lists from GO terms
cat("\n=== Extracting Specific Functional Gene Lists ===\n")

# Get GO Molecular Function terms (fetch fresh if categories doesn't exist)
if (!exists("categories") || is.null(categories[["C5_GO_MF"]])) {
  cat("Fetching GO Molecular Function terms...\n")
  go_mf <- msigdbr(species = "Homo sapiens", category = "C5", subcategory = "GO:MF")
} else {
  go_mf <- categories[["C5_GO_MF"]]
}

# Make sure our_genes exists
if (!exists("our_genes")) {
  our_genes <- rownames(pseudo_bulk_matrix)
}

# Transcription factors (DNA-binding transcription factor activity)
tf_terms <- go_mf[grep("transcription factor", go_mf$gs_name, ignore.case = TRUE), ]
transcription_factors <- unique(tf_terms$gene_symbol)
tf_in_data <- intersect(our_genes, transcription_factors)
cat("Transcription factors found:", length(tf_in_data), "\n")

# Kinases
kinase_terms <- go_mf[grep("kinase", go_mf$gs_name, ignore.case = TRUE), ]
kinases <- unique(kinase_terms$gene_symbol)
kinases_in_data <- intersect(our_genes, kinases)
cat("Kinases found:", length(kinases_in_data), "\n")

# Receptors
receptor_terms <- go_mf[grep("receptor", go_mf$gs_name, ignore.case = TRUE), ]
receptors <- unique(receptor_terms$gene_symbol)
receptors_in_data <- intersect(our_genes, receptors)
cat("Receptors found:", length(receptors_in_data), "\n")

# Ion channels
channel_terms <- go_mf[grep("channel", go_mf$gs_name, ignore.case = TRUE), ]
ion_channels <- unique(channel_terms$gene_symbol)
channels_in_data <- intersect(our_genes, ion_channels)
cat("Ion channels found:", length(channels_in_data), "\n")

# Transporters
transporter_terms <- go_mf[grep("transporter", go_mf$gs_name, ignore.case = TRUE), ]
transporters <- unique(transporter_terms$gene_symbol)
transporters_in_data <- intersect(our_genes, transporters)
cat("Transporters found:", length(transporters_in_data), "\n")

# Proteases
protease_terms <- go_mf[grep("protease|peptidase", go_mf$gs_name, ignore.case = TRUE), ]
proteases <- unique(protease_terms$gene_symbol)
proteases_in_data <- intersect(our_genes, proteases)
cat("Proteases found:", length(proteases_in_data), "\n")

# Create specific gene list summary
specific_lists <- list(
  "Transcription_Factors" = tf_in_data,
  "Kinases" = kinases_in_data,
  "Receptors" = receptors_in_data,
  "Ion_Channels" = channels_in_data,
  "Transporters" = transporters_in_data,
  "Proteases" = proteases_in_data
)

specific_summary <- data.frame(
  gene_class = names(specific_lists),
  n_genes = sapply(specific_lists, length),
  zeros_count = sapply(specific_lists, function(genes) {
    if (length(genes) > 0) sum(pseudo_bulk_matrix[genes, ] == 0) else 0
  }),
  zeros_pct = sapply(specific_lists, function(genes) {
    if (length(genes) > 0) {
      round(sum(pseudo_bulk_matrix[genes, ] == 0) / 
            (length(genes) * ncol(pseudo_bulk_matrix)) * 100, 1)
    } else 0
  })
)

cat("\n=== Specific Gene Class Summary ===\n")
print(specific_summary)

# Expression of specific gene classes per cell type
cat("\n=== Mean Expression by Gene Class and Cell Type ===\n")
specific_expr <- do.call(rbind, lapply(names(specific_lists), function(class_name) {
  genes <- specific_lists[[class_name]]
  if (length(genes) > 0) {
    colMeans(pseudo_bulk_matrix[genes, , drop = FALSE])
  } else {
    rep(0, ncol(pseudo_bulk_matrix))
  }
}))
rownames(specific_expr) <- names(specific_lists)
print(round(specific_expr, 2))

# Save specific gene lists
for (class_name in names(specific_lists)) {
  genes <- specific_lists[[class_name]]
  if (length(genes) > 0) {
    gene_expr <- pseudo_bulk_matrix[genes, , drop = FALSE]
    write.csv(gene_expr, paste0("results/genes_", tolower(class_name), ".csv"))
  }
}
cat("\nSaved individual gene class files (genes_*.csv)\n")

# Barplot of gene class sizes
barplot(specific_summary$n_genes, 
        names.arg = specific_summary$gene_class,
        main = "Number of Genes per Functional Class",
        ylab = "Number of genes",
        las = 2,
        col = rainbow(nrow(specific_summary)),
        cex.names = 0.8)

#nolint end