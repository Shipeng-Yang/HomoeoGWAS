#!/usr/bin/env Rscript
# Publication-grade GWAS figures for HomoeoGWAS via CMplot (genome-wide) and
# locuszoomr (per-locus). Driven by `homoeogwas rplot`; can also be run
# standalone. Reads a HomoeoGWAS sumstats TSV (snp_id/chrom/pos/p ...) and,
# for locus plots, a precomputed window TSV (snp_id/chrom/pos/p/r2_to_lead)
# so R needs no genotype/LD access. R, CMplot, locuszoomr and ensembldb are
# all OPTIONAL — this script reports cleanly when one is missing.

# headless: raster devices must not try to open an X11 display
options(bitmapType = "cairo")

suppressWarnings(suppressMessages({
  ok_dt <- requireNamespace("data.table", quietly = TRUE)
}))

# ---- tiny `--key value` arg parser (no optparse dependency) ----------------
parse_args <- function(a) {
  out <- list()
  i <- 1
  while (i <= length(a)) {
    k <- a[i]
    if (startsWith(k, "--")) {
      key <- sub("^--", "", k)
      if (i + 1 <= length(a) && !startsWith(a[i + 1], "--")) {
        out[[key]] <- a[i + 1]; i <- i + 2
      } else { out[[key]] <- "TRUE"; i <- i + 1 }
    } else i <- i + 1
  }
  out
}
args <- parse_args(commandArgs(trailingOnly = TRUE))

have <- function(p) requireNamespace(p, quietly = TRUE)

# ---- dependency self-check (used by the Python wrapper) ---------------------
if (!is.null(args[["check-deps"]])) {
  cat(sprintf("CMplot=%s\n", have("CMplot")))
  cat(sprintf("locuszoomr=%s\n", have("locuszoomr")))
  cat(sprintf("ensembldb=%s\n", have("ensembldb")))
  quit(status = 0)
}

read_tsv <- function(path) {
  if (ok_dt) as.data.frame(data.table::fread(path, sep = "\t",
                                             showProgress = FALSE))
  else read.delim(path, sep = "\t", stringsAsFactors = FALSE)
}

kind <- strsplit(if (is.null(args[["kind"]])) "manhattan,qq" else args[["kind"]],
                 ",")[[1]]
kind <- trimws(kind)
out_dir <- if (is.null(args[["out-dir"]])) "." else args[["out-dir"]]
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
prefix <- if (is.null(args[["prefix"]])) "homoeogwas" else args[["prefix"]]
fmt <- if (is.null(args[["format"]])) "pdf" else args[["format"]]
dpi <- if (is.null(args[["dpi"]])) 300 else as.numeric(args[["dpi"]])

strip_chr <- function(x) sub("^chr", "", x, ignore.case = TRUE)

written <- character(0)
status_code <- 0L

# ---- genome-wide via CMplot ------------------------------------------------
gw_kinds <- intersect(kind, c("manhattan", "circular", "qq", "density"))
if (length(gw_kinds) > 0) {
  if (!have("CMplot")) {
    stop("CMplot not installed: install.packages('CMplot')")
  }
  ss <- read_tsv(args[["sumstats"]])
  scol <- if (!is.null(args[["snp-col"]])) args[["snp-col"]] else "snp_id"
  ccol <- if (!is.null(args[["chrom-col"]])) args[["chrom-col"]] else "chrom"
  pcol <- if (!is.null(args[["pos-col"]])) args[["pos-col"]] else "pos"
  pvcol <- if (!is.null(args[["p-col"]])) args[["p-col"]] else "p"
  dat <- data.frame(SNP = as.character(ss[[scol]]),
                    Chromosome = as.character(ss[[ccol]]),
                    Position = as.numeric(ss[[pcol]]),
                    P = as.numeric(ss[[pvcol]]))
  dat <- dat[is.finite(dat$Position) & is.finite(dat$P), , drop = FALSE]
  m <- nrow(dat)
  gw <- 0.05 / max(m, 1)
  sugg <- 1 / max(m, 1)

  oldwd <- getwd(); setwd(out_dir); on.exit(setwd(oldwd), add = TRUE)
  before <- list.files(".")
  ptypes <- c(manhattan = "m", circular = "c", qq = "q", density = "d")
  for (gk in gw_kinds) {
    pt <- ptypes[[gk]]
    common <- list(Pmap = dat, plot.type = pt, file = fmt, dpi = dpi,
                   file.output = TRUE, verbose = FALSE,
                   file.name = paste0(prefix, ".", gk),
                   width = 14, height = 6)
    if (gk %in% c("manhattan", "circular")) {
      common$threshold <- c(gw, sugg)
      common$threshold.col <- c("#CB3E35", "#9A9A9A")
      common$threshold.lty <- c(1, 2)
      common$amplify <- FALSE
    }
    tryCatch(do.call(CMplot::CMplot, common),
             error = function(e) message("CMplot ", gk, " failed: ",
                                         conditionMessage(e)))
  }
  after <- setdiff(list.files("."), before)
  written <- c(written, file.path(out_dir, after))
  if (length(after) == 0) {
    message("CMplot produced no files for: ", paste(gw_kinds, collapse = ","))
    status_code <- 1L
  }
}

# ---- per-locus via locuszoomr ----------------------------------------------
if ("locus" %in% kind) {
  if (!have("locuszoomr")) {
    stop("locuszoomr not installed: install.packages('locuszoomr')")
  }
  loc <- read_tsv(args[["locus-tsv"]])
  names(loc)[names(loc) == "snp_id"] <- "rsid"
  if (!"r2_to_lead" %in% names(loc)) loc$r2_to_lead <- NA_real_
  lead_pos <- as.numeric(args[["lead-pos"]])
  lchr <- as.character(args[["lead-chrom"]])
  win_kb <- if (is.null(args[["window-kb"]])) 500 else
            as.numeric(args[["window-kb"]])

  edb <- NULL
  if (!is.null(args[["gff"]]) && have("ensembldb")) {
    cache <- if (!is.null(args[["ensdb-cache"]])) args[["ensdb-cache"]] else
             tempdir()
    dir.create(cache, recursive = TRUE, showWarnings = FALSE)
    org <- if (!is.null(args[["organism"]])) args[["organism"]] else "species"
    gver <- if (!is.null(args[["genome-version"]])) args[["genome-version"]] else
            "v1"
    ever <- if (!is.null(args[["ens-version"]])) args[["ens-version"]] else "1"
    sqlite <- file.path(cache, paste0(basename(args[["gff"]]), ".sqlite"))
    # ensDbFromGff requires a '##gff-version' header; many crop GFFs lack it,
    # so write a header-normalised plain copy into the cache when needed.
    norm_gff <- function(g) {
      con <- if (grepl("\\.gz$", g)) gzfile(g, "rt") else file(g, "rt")
      first <- tryCatch(readLines(con, n = 1), finally = close(con))
      if (length(first) && startsWith(first, "##gff-version")) return(g)
      dst <- file.path(cache, paste0(basename(g), ".hdr.gff3"))
      if (!file.exists(dst)) {
        inc <- if (grepl("\\.gz$", g)) gzfile(g, "rt") else file(g, "rt")
        outc <- file(dst, "wt")
        writeLines("##gff-version 3", outc)
        repeat {
          ln <- readLines(inc, n = 50000)
          if (length(ln) == 0) break
          writeLines(ln, outc)
        }
        close(inc); close(outc)
      }
      dst
    }
    edb <- tryCatch({
      if (!file.exists(sqlite)) {
        ensembldb::ensDbFromGff(gff = norm_gff(args[["gff"]]), outfile = sqlite,
                                organism = org, genomeVersion = gver,
                                version = ever)
      }
      ensembldb::EnsDb(sqlite)
    }, error = function(e) {
      message("EnsDb build failed (", conditionMessage(e), ")"); NULL })
  }
  if (is.null(edb)) {
    message("No usable gene annotation (EnsDb) for this locus; locuszoomr ",
            "needs one. Skipping R locus — use CMplot genome-wide figures ",
            "or the Python `homoeogwas locus` (which draws a gene track ",
            "directly from a GFF).")
    quit(status = 0)
  }

  # locuszoomr matches on its own seqname; align by stripping a 'chr' prefix
  loc$chrom <- strip_chr(as.character(loc$chrom))
  seqn <- strip_chr(lchr)
  outfile <- file.path(out_dir, paste0("locus_", prefix, "_",
                                       lchr, "_", round(lead_pos), ".", fmt))
  l <- tryCatch(
    locuszoomr::locus(data = loc, seqname = seqn,
                      xrange = c(lead_pos - win_kb * 1000,
                                 lead_pos + win_kb * 1000),
                      ens_db = edb, chrom = "chrom", pos = "pos",
                      p = "p", labs = "rsid", LD = "r2_to_lead"),
    error = function(e) { message("locus() failed: ",
                                  conditionMessage(e)); NULL })
  if (!is.null(l)) {
    if (fmt == "pdf") grDevices::pdf(outfile, width = 8, height = 7)
    else grDevices::png(outfile, width = 8, height = 7, units = "in",
                        res = dpi)
    ok <- TRUE
    tryCatch(locuszoomr::locus_plot(l),
             error = function(e) { message("locus_plot failed: ",
                                           conditionMessage(e)); ok <<- FALSE })
    grDevices::dev.off()
    if (ok) written <- c(written, outfile) else unlink(outfile)
  }
}

for (w in written) cat("WROTE\t", w, "\n", sep = "")
quit(status = status_code)
