# SKAT single-gene region tests (burden + SKAT + SKAT-O) on cotton hit genes.
# Estimand: single-gene variant-set aggregation (rare-variant machinery applied to common SNPs).
# Quantitative null y~1 (no kinship); regenie burden provides the structure-corrected counterpart.
# Usage: Rscript run_skat.R <A|D>
suppressMessages(library(SKAT))
args <- commandArgs(trailingOnly = TRUE); sub <- args[1]
root <- "/mnt/7302share/fast_ysp/U7_GWAS"
gw <- file.path(root, "results/phase7/ext_baseline/gwaspoly")

man <- read.delim(file.path(root, "results/phase7/ext_baseline/candidate_manifest.tsv"),
                  stringsAsFactors = FALSE)
man <- man[man$subgenome == sub, ]
ph <- read.csv(file.path(gw, "pheno.csv"), stringsAsFactors = FALSE)
# genotype: read geno.csv rows for the hit-gene SNPs only (small)
con <- file(file.path(gw, paste0(sub, "_geno.csv")), "r")
hdr <- strsplit(readLines(con, 1), ",")[[1]]
samp <- hdr[-(1:3)]
want <- unique(man$snp_id)
geno <- matrix(NA, nrow = length(want), ncol = length(samp),
               dimnames = list(want, samp))
got <- 0
while (got < length(want)) {
  line <- readLines(con, 1); if (length(line) == 0) break
  f <- strsplit(line, ",")[[1]]
  if (f[1] %in% want) { geno[f[1], ] <- as.numeric(f[-(1:3)]); got <- got + 1 }
}
close(con)
stopifnot(all(samp == ph$Sample))

genes <- unique(man[, c("hit", "trait", "gene")])
out <- list()
for (i in seq_len(nrow(genes))) {
  g <- genes[i, ]
  snps <- man$snp_id[man$gene == g$gene]
  Z <- t(geno[snps, , drop = FALSE])          # samples x SNPs (dosage 0/1/2)
  y <- ph[[g$trait]]
  keep <- is.finite(y) & complete.cases(Z)
  obj <- SKAT_Null_Model(y[keep] ~ 1, out_type = "C")
  Zk <- Z[keep, , drop = FALSE]
  p_burden <- SKAT(Zk, obj, r.corr = 1)$p.value      # burden
  p_skat   <- SKAT(Zk, obj, r.corr = 0)$p.value       # SKAT
  p_skato  <- SKAT(Zk, obj, method = "SKATO")$p.value  # SKAT-O
  out[[i]] <- data.frame(
    tool = "SKAT", estimand = "single-gene burden/SKAT/SKAT-O",
    hit = g$hit, trait = g$trait, gene = g$gene, subgenome = sub,
    n_snp = length(snps), n_used = sum(keep),
    p_burden = signif(p_burden, 4), p_skat = signif(p_skat, 4),
    p_skato = signif(p_skato, 4))
}
res <- do.call(rbind, out)
print(res)
outf <- file.path(root, "results/phase7/ext_baseline", paste0("skat_", sub, ".tsv"))
write.table(res, outf, sep = "\t", quote = FALSE, row.names = FALSE)
cat("wrote", outf, "\n")
