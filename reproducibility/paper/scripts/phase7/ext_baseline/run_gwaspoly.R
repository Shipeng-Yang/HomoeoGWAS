# GWASpoly single-locus polyploid GWAS on one cotton subgenome.
# Estimand: single-marker additive association (polyploid-aware mixed model).
# Reports best -log10P among each hit gene's SNPs vs the genome-wide Bonferroni threshold.
# Usage: Rscript run_gwaspoly.R <A|D>
suppressMessages(library(GWASpoly))
args <- commandArgs(trailingOnly = TRUE)
sub <- args[1]
root <- "/mnt/7302share/fast_ysp/U7_GWAS"
gw <- file.path(root, "results/phase7/ext_baseline/gwaspoly")

# hit genes living on this subgenome -> matched trait + their SNP ids (from frozen manifest)
man <- read.delim(file.path(root, "results/phase7/ext_baseline/candidate_manifest.tsv"),
                  stringsAsFactors = FALSE)
man <- man[man$subgenome == sub, ]
genes <- unique(man[, c("hit", "trait", "gene", "gene_role")])

cat("== read.GWASpoly", sub, "==\n")
data <- read.GWASpoly(ploidy = 2,
                      pheno.file = file.path(gw, "pheno.csv"),
                      geno.file = file.path(gw, paste0(sub, "_geno.csv")),
                      format = "numeric", n.traits = 2, delim = ",")
cat("== set.K ==\n")
data <- set.K(data, LOCO = FALSE, n.core = 8)
traits <- c("fiber_length_BLUE", "length_uniformity_BLUE")
cat("== GWASpoly additive ==\n")
data <- GWASpoly(data, models = "additive", traits = traits, n.core = 8)
data <- set.threshold(data, method = "Bonferroni", level = 0.05)

# scores: list by trait, matrix markers x models of -log10(p); map gives marker names
mk <- data@map$Marker
out <- list()
for (i in seq_len(nrow(genes))) {
  g <- genes[i, ]
  snps <- man$snp_id[man$gene == g$gene]
  sc <- data@scores[[g$trait]][, "additive"]
  idx <- match(snps, mk)
  vals <- sc[idx]
  vals <- vals[is.finite(vals)]
  best_logp <- if (length(vals)) max(vals) else NA
  thr <- data@threshold[g$trait, "additive"]
  out[[length(out) + 1]] <- data.frame(
    tool = "GWASpoly", estimand = "single-marker additive (polyploid GLM)",
    hit = g$hit, trait = g$trait, gene = g$gene, subgenome = sub,
    n_snp = length(snps), best_neglog10p = round(best_logp, 4),
    best_p = signif(10^(-best_logp), 4),
    gw_bonf_threshold_neglog10p = round(thr, 4),
    pass_genomewide = isTRUE(best_logp >= thr),
    n_markers_subgenome = length(mk))
}
res <- do.call(rbind, out)
print(res)
outf <- file.path(root, "results/phase7/ext_baseline", paste0("gwaspoly_", sub, ".tsv"))
write.table(res, outf, sep = "\t", quote = FALSE, row.names = FALSE)
cat("wrote", outf, "\n")
