# STAAR single-gene region tests (STAAR-O omnibus = burden+SKAT+ACAT) on cotton hit genes.
# Estimand: single-gene variant-set aggregation (the dedicated rare-variant tool, common-variant caveat).
# Usage: Rscript run_staar.R <A|D>
suppressMessages(library(STAAR))
args <- commandArgs(trailingOnly = TRUE); sub <- args[1]
root <- "/mnt/7302share/fast_ysp/U7_GWAS"
gw <- file.path(root, "results/phase7/ext_baseline/gwaspoly")

man <- read.delim(file.path(root, "results/phase7/ext_baseline/candidate_manifest.tsv"),
                  stringsAsFactors = FALSE)
man <- man[man$subgenome == sub, ]
ph <- read.csv(file.path(gw, "pheno.csv"), stringsAsFactors = FALSE)
con <- file(file.path(gw, paste0(sub, "_geno.csv")), "r")
hdr <- strsplit(readLines(con, 1), ",")[[1]]; samp <- hdr[-(1:3)]
want <- unique(man$snp_id)
geno <- matrix(NA, nrow = length(want), ncol = length(samp), dimnames = list(want, samp))
got <- 0
while (got < length(want)) {
  line <- readLines(con, 1); if (length(line) == 0) break
  f <- strsplit(line, ",")[[1]]
  if (f[1] %in% want) { geno[f[1], ] <- as.numeric(f[-(1:3)]); got <- got + 1 }
}
close(con); stopifnot(all(samp == ph$Sample))

genes <- unique(man[, c("hit", "trait", "gene")])
out <- list()
for (i in seq_len(nrow(genes))) {
  g <- genes[i, ]
  snps <- man$snp_id[man$gene == g$gene]
  Z <- t(geno[snps, , drop = FALSE])
  y <- ph[[g$trait]]
  keep <- is.finite(y) & complete.cases(Z)
  df <- data.frame(y = y[keep])
  obj <- fit_null_glm(y ~ 1, data = df, family = gaussian(link = "identity"))
  Zk <- as.matrix(Z[keep, , drop = FALSE])
  st <- STAAR(Zk, obj, annotation_phred = NULL, rare_maf_cutoff = 1, rv_num_cutoff = 1)
  # STAAR-O omnibus p across burden/SKAT/ACAT with MAF weights
  p_o <- st$results_STAAR_O
  p_burden <- st$results_STAAR_S_1_25[1]  # first weight-scheme burden col (Burden(1,25))
  out[[i]] <- data.frame(
    tool = "STAAR", estimand = "single-gene STAAR-O (burden+SKAT+ACAT)",
    hit = g$hit, trait = g$trait, gene = g$gene, subgenome = sub,
    n_snp = length(snps), n_used = sum(keep),
    p_STAAR_O = signif(p_o, 4))
}
res <- do.call(rbind, out)
print(res)
outf <- file.path(root, "results/phase7/ext_baseline", paste0("staar_", sub, ".tsv"))
write.table(res, outf, sep = "\t", quote = FALSE, row.names = FALSE)
cat("wrote", outf, "\n")
