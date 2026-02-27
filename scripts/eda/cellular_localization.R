# Cellular Localization Analysis
# Categorize genes by GO Cellular Component annotations

# Resolve project root (parent of scripts/)
get_project_root <- function() {
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

# Install required packages if not available
if (!requireNamespace("BiocManager", quietly = TRUE)) {
  install.packages("BiocManager")
}

required_packages <- c("org.Hs.eg.db", "AnnotationDbi", "GO.db")
for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    BiocManager::install(pkg)
  }
}

library(org.Hs.eg.db)
library(AnnotationDbi)
library(GO.db)

# Read gene expression data
data <- read.table('data/bulk_GSM2230757_human1.csv', header=TRUE, row.names=1, sep=',')
gene_names <- rownames(data)
counts <- as.numeric(data[,1])

cat("Total genes:", length(gene_names), "\n")

# Map gene symbols to Entrez IDs
entrez_ids <- mapIds(org.Hs.eg.db, 
                     keys = gene_names, 
                     column = "ENTREZID", 
                     keytype = "SYMBOL",
                     multiVals = "first")

cat("Genes mapped to Entrez IDs:", sum(!is.na(entrez_ids)), "\n")

# Get GO Cellular Component annotations
go_cc <- select(org.Hs.eg.db, 
                keys = na.omit(entrez_ids), 
                columns = c("SYMBOL", "GO", "ONTOLOGY"),
                keytype = "ENTREZID")

# Filter for Cellular Component only
go_cc <- go_cc[go_cc$ONTOLOGY == "CC" & !is.na(go_cc$GO), ]

cat("GO CC annotations found:", nrow(go_cc), "\n")

# Define major cellular compartments and their GO terms
compartments <- list(
  "Nucleus" = c("GO:0005634", "GO:0005654", "GO:0005730", "GO:0000785"),
  "Cytoplasm" = c("GO:0005737", "GO:0005829"),
  "Plasma Membrane" = c("GO:0005886", "GO:0016020", "GO:0031224"),
  "Mitochondrion" = c("GO:0005739", "GO:0005743", "GO:0005758"),
  "Endoplasmic Reticulum" = c("GO:0005783", "GO:0005789"),
  "Golgi Apparatus" = c("GO:0005794", "GO:0000139"),
  "Lysosome/Endosome" = c("GO:0005764", "GO:0005768", "GO:0005769"),
  "Ribosome" = c("GO:0005840", "GO:0022625", "GO:0022627"),
  "Cytoskeleton" = c("GO:0005856", "GO:0015630", "GO:0005874"),
  "Extracellular" = c("GO:0005576", "GO:0005615", "GO:0031012")
)

# Get all descendant terms for each compartment
get_descendants <- function(go_id) {
  tryCatch({
    offspring <- GOCCOFFSPRING[[go_id]]
    if (is.null(offspring)) return(go_id)
    return(c(go_id, offspring))
  }, error = function(e) {
    return(go_id)
  })
}

# Expand compartment terms to include descendants
expanded_compartments <- lapply(compartments, function(terms) {
  unique(unlist(lapply(terms, get_descendants)))
})

# Assign genes to compartments
gene_compartments <- data.frame(
  gene = gene_names,
  expression = counts,
  stringsAsFactors = FALSE
)

for (comp_name in names(expanded_compartments)) {
  gene_compartments[[comp_name]] <- gene_names %in% 
    go_cc$SYMBOL[go_cc$GO %in% expanded_compartments[[comp_name]]]
}

# Create primary localization (highest priority match)
priority_order <- c("Nucleus", "Plasma Membrane", "Mitochondrion", 
                    "Endoplasmic Reticulum", "Golgi Apparatus", 
                    "Lysosome/Endosome", "Ribosome", "Cytoskeleton",
                    "Extracellular", "Cytoplasm")

gene_compartments$primary_localization <- "Unknown"
for (comp in rev(priority_order)) {
  gene_compartments$primary_localization[gene_compartments[[comp]]] <- comp
}

# Summary statistics
cat("\n=== Gene Counts by Cellular Localization ===\n")
loc_summary <- table(gene_compartments$primary_localization)
loc_summary <- sort(loc_summary, decreasing = TRUE)
print(loc_summary)

# Expression by localization
cat("\n=== Mean Expression by Localization ===\n")
expr_by_loc <- aggregate(expression ~ primary_localization, 
                         data = gene_compartments, 
                         FUN = function(x) c(mean = mean(x), median = median(x), n = length(x)))
print(expr_by_loc)

# Create visualizations
pdf("figures/cellular_localization_analysis.pdf", width = 12, height = 10)

# Plot 1: Bar chart of gene counts by localization
par(mar = c(10, 5, 4, 2))
barplot(loc_summary, 
        las = 2, 
        col = rainbow(length(loc_summary)),
        main = "Number of Genes by Cellular Localization",
        ylab = "Number of Genes",
        cex.names = 0.8)

# Plot 2: Expression distribution by localization (boxplot)
par(mar = c(10, 5, 4, 2))
boxplot(log1p(expression) ~ primary_localization, 
        data = gene_compartments,
        las = 2,
        col = rainbow(length(unique(gene_compartments$primary_localization))),
        main = "Gene Expression by Cellular Localization",
        ylab = "Log(Expression + 1)",
        cex.axis = 0.7)

# Plot 3: Pie chart of proportions
par(mar = c(2, 2, 4, 2))
pie(loc_summary, 
    main = "Proportion of Genes by Localization",
    col = rainbow(length(loc_summary)),
    cex = 0.7)

# Plot 4: Top expressed genes per compartment
par(mar = c(5, 12, 4, 2))
top_genes_per_comp <- do.call(rbind, lapply(names(compartments), function(comp) {
  subset_data <- gene_compartments[gene_compartments$primary_localization == comp, ]
  if (nrow(subset_data) > 0) {
    top <- head(subset_data[order(-subset_data$expression), ], 3)
    data.frame(compartment = comp, gene = top$gene, expression = top$expression)
  }
}))

if (nrow(top_genes_per_comp) > 0) {
  barplot(top_genes_per_comp$expression, 
          names.arg = paste(top_genes_per_comp$compartment, "-", top_genes_per_comp$gene),
          las = 2,
          horiz = TRUE,
          col = rainbow(nrow(top_genes_per_comp)),
          main = "Top 3 Expressed Genes per Compartment",
          xlab = "Expression",
          cex.names = 0.6)
}

dev.off()

# Save results to CSV
write.csv(gene_compartments, "results/genes_by_localization.csv", row.names = FALSE)

cat("\n=== Output Files ===\n")
cat("1. cellular_localization_analysis.pdf - Visualizations\n")
cat("2. genes_by_localization.csv - Full gene list with localization\n")
cat("\nDone!\n")
