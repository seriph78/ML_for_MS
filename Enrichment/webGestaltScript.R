#!/usr/bin/env Rscript

library("WebGestaltR")
library("jsonlite")

testFunction <- function (...) {
  return(tryCatch(WebGestaltR(...), error=function(e) NULL))
}

args <- commandArgs(trailingOnly = TRUE)

arg_map <- list(
  type = args[1],
  geneListFile     = args[2],
  enrichmentMethod = args[3],
  organism         = args[4],
  backgroundFlag   = as.integer(args[5]),
  referenceGene     = args[6],
  interestGeneType = args[7],
  fdrMethod        = args[8],
  fdrThr           = as.numeric(args[9]),
  minNum           = as.integer(args[10]),
  maxNum           = as.integer(args[11]),
  nThreads         = as.integer(args[12]),
  outputCsvFile    = args[13]
)

gene_list <- fromJSON(arg_map$geneListFile)

if (arg_map$backgroundFlag == 1) {
  background <- fromJSON(arg_map$referenceGene)

  wg <- mapply(
    testFunction,
    enrichDatabase = arg_map$type,
    MoreArgs = list(
      enrichMethod = arg_map$enrichmentMethod,
      organism = arg_map$organism,
      referenceGene = background,
      referenceGeneType = arg_map$interestGeneType,
      interestGene = gene_list,
      interestGeneType = arg_map$interestGeneType,
      fdrMethod = arg_map$fdrMethod,
      fdrThr = arg_map$fdrThr,
      minNum = arg_map$minNum,
      maxNum = arg_map$maxNum,
      reportNum = 40,
      isOutput = FALSE,
      nThreads = arg_map$nThreads,
      hostName = "https://www.webgestalt.org"
    )
  )
}else{
  wg <- mapply(
    testFunction,
    enrichDatabase = arg_map$type,
    MoreArgs = list(
      enrichMethod = arg_map$enrichmentMethod,
      organism = arg_map$organism,
      referenceSet = "genome",
      interestGene = gene_list,
      interestGeneType = arg_map$interestGeneType,
      fdrMethod = arg_map$fdrMethod,
      fdrThr = arg_map$fdrThr,
      minNum = arg_map$minNum,
      maxNum = arg_map$maxNum,
      reportNum = 40,
      isOutput = FALSE,
      nThreads = arg_map$nThreads,
      hostName = "https://www.webgestalt.org"
    )
  )
}

# Combina i risultati in un unico data.frame (se ci sono risultati)
df <- as.data.frame(wg)

if (is.null(df) || nrow(df) == 0) {
  cat("Nessun risultato trovato.\n")
  quit(status = 0)
}

df[] <- lapply(df, function(col) {
  if (is.list(col)) {
    sapply(col, function(x) paste(x, collapse = "|"))
  } else {
    col
  }
})

# Salva in CSV solo se ci sono dati
if (!is.null(df)) {
  write.csv(df, file = arg_map$outputCsvFile, row.names = TRUE)
  cat("Output salvato in", arg_map$outputCsvFile, "\n")
} else {
  cat("Nessun risultato da salvare.\n")
}