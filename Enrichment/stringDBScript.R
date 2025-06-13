library(STRINGdb)
library("jsonlite")

args <- commandArgs(trailingOnly = TRUE)

arg_map <- list(
  type = args[1],
  geneListFile     = args[2],
  backgroundFlag   = as.integer(args[3]),
  backgroundFile = args[4],
  scoreTreshold = as.integer(args[5]),
  outputCsvFile    = args[6]
)

# Initialize STRINGdb object
string_db <- STRINGdb$new(
  version = "12.0",             # STRING version (e.g., "11.5", "12.0")
  species = 9606,             # NCBI taxonomy ID (e.g., 9606 for human)
  score_threshold = arg_map$scoreTreshold,     # Confidence score cutoff (0–1000)
  input_directory = "."     # Folder for downloads (must be a string)
)

gene_list <- fromJSON(arg_map$geneListFile)
gene_df <- data.frame(gene = gene_list)
mapped_genes <- string_db$map(gene_df, "gene", removeUnmappedRows = TRUE)

if (arg_map$backgroundFlag == 1) {
  background <- fromJSON(arg_map$backgroundFile)
  set_background(background)

  enrichment_results <- string_db$get_enrichment(mapped_genes$STRING_id, category=arg_map$type)

}else{
  enrichment_results <- string_db$get_enrichment(mapped_genes$STRING_id, category=arg_map$type)
}

enrichment_results <- as.data.frame(enrichment_results)

if (is.null(enrichment_results) || nrow(enrichment_results) == 0) {
  cat("Nessun risultato trovato.\n")
  quit(status = 0)
}

# Salva in CSV solo se ci sono dati
if (!is.null(enrichment_results)) {
  write.csv(enrichment_results, file = arg_map$outputCsvFile, row.names = TRUE)
  cat("Output salvato in", arg_map$outputCsvFile, "\n")
} else {
  cat("Nessun risultato da salvare.\n")
}