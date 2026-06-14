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

# ---- distinctive HomoeoGWAS figures (ggplot2) ------------------------------
# These show what generic GWAS plots cannot: subgenome variance partition and
# the homoeolog-pair interaction estimand (invisible to single-locus tests).
dist_kinds <- intersect(kind, c("variance", "interaction", "marginal",
                                "burden", "triad", "network"))
if (length(dist_kinds) > 0) {
  if (!have("ggplot2")) stop("ggplot2 not installed: install.packages('ggplot2')")
  suppressWarnings(suppressMessages(library(ggplot2)))
  `%||%` <- function(a, b) if (is.null(a)) b else a
  trait <- args[["trait"]] %||% prefix
  topn <- as.integer(args[["top-n"]] %||% "6")
  SUBCOL <- c(A = "#1F577B", B = "#E07370", C = "#368650", D = "#FCBC10",
              homoeolog = "#A56BA7", residual = "#B8B8B8")
  theme_omic <- function() {
    theme_classic(base_size = 11) +
      theme(axis.line = element_line(linewidth = 0.4, colour = "#222222"),
            axis.ticks = element_line(linewidth = 0.4, colour = "#222222"),
            plot.title = element_text(face = "bold", size = 12),
            plot.subtitle = element_text(size = 9, colour = "#555555"),
            legend.key.size = grid::unit(0.8, "lines"))
  }
  ggw <- function(p, nm, w = 7, h = 5) {
    f <- file.path(out_dir, paste0(nm, "_", prefix, ".", fmt))
    suppressWarnings(suppressMessages(
      ggsave(f, p, width = w, height = h, dpi = dpi)))
    written <<- c(written, f)
  }

  if ("variance" %in% dist_kinds && requireNamespace("treemapify", quietly = TRUE)) {
    v <- read_tsv(args[["variance"]])    # component, pve, sigma2, kind, is_boundary
    v <- v[is.finite(v$sigma2) & v$sigma2 > 0, ]   # drop boundary (sigma2->0)
    if (nrow(v) == 0) {
      message("variance: all components at the boundary (sigma2->0); nothing ",
              "to draw — skipping")
    } else {
    v$component <- factor(v$component, levels = v$component)
    # earth palette: subgenomes warm earth tones, homoeolog kernel taupe-brown,
    # residual recessive greige
    EARTH <- c(A = "#8C6D31", B = "#C44E52", C = "#55A868", D = "#CCB974",
               homoeolog = "#937860", residual = "#C7C0B8")
    cols <- setNames(ifelse(as.character(v$component) %in% names(EARTH),
                            EARTH[as.character(v$component)], "#8C6D31"),
                     as.character(v$component))
    # σ² label: keep 1 decimal for values >= 1 (e.g. 279.9), but for small-scale
    # traits (e.g. seed length, σ²~0.005) show 2 significant figures so the label
    # never collapses to a misleading "0.0".
    fmt_sigma2 <- function(x) vapply(x, function(v) {
      if (!is.finite(v)) return("NA")
      if (abs(v) >= 1 || v == 0) return(sprintf("%.1f", v))
      d <- 1L - as.integer(floor(log10(abs(v))))   # 2 significant figures
      formatC(v, format = "f", digits = d)
    }, character(1))
    v$tlab <- sprintf("%s\n%.0f%%\nσ²=%s", v$component, 100 * v$pve, fmt_sigma2(v$sigma2))
    p <- ggplot(v, aes(area = sigma2, fill = component, label = tlab)) +
      treemapify::geom_treemap(colour = "white", size = 4, start = "topleft") +
      treemapify::geom_treemap_text(colour = "white", place = "centre",
                                    grow = FALSE, reflow = TRUE, size = 16,
                                    fontface = "bold", start = "topleft") +
      scale_fill_manual(values = cols, name = "Variance\ncomponent",
                        guide = guide_legend(reverse = TRUE)) +
      theme_minimal(base_size = 14) +
      theme(legend.title = element_text(face = "bold", size = 12),
            legend.text = element_text(size = 12),
            legend.key.size = grid::unit(1.2, "lines"),
            axis.text = element_blank(), axis.title = element_blank(),
            panel.grid = element_blank(), plot.margin = margin(10, 10, 10, 10))
    ggw(p, "variance", w = 7.6, h = 5.0)
    }
  }

  if (any(c("interaction", "marginal", "network") %in% dist_kinds)) {
    r <- read_tsv(args[["ranking"]])
    names(r)[2:3] <- c("gene_x", "gene_y")
    r$p_interaction <- as.numeric(r$p_interaction)
    r$neglog10p <- as.numeric(r$neglog10p)
    bonf <- 0.05 / nrow(r)
    # subgenome-pair label derived from the data (A×D cotton, A×B peanut, …)
    sx <- as.character(stats::na.omit(r$sub_x)[1])
    sy <- as.character(stats::na.omit(r$sub_y)[1])
    pair_lab <- paste0(sx, "×", sy)
    fmt_p <- function(x) formatC(as.numeric(x), format = "e", digits = 1)
  }

  if ("interaction" %in% dist_kinds) {
    has_pos <- "pos_x" %in% names(r) && "chrom_x" %in% names(r) &&
      !any(is.na(r$chrom_x) | trimws(as.character(r$chrom_x)) == "") &&
      all(is.finite(suppressWarnings(as.numeric(r$pos_x))))
    if (has_pos) {
      r$pos_x <- as.numeric(r$pos_x)
      # natural-sort anchor chromosomes: split subgenome prefix + number so
      # peanut "10" sorts after "2" (not after "1"), and cotton keeps A.. then D..
      chr_chr <- as.character(r$chrom_x)
      chr_num <- suppressWarnings(as.numeric(gsub("[^0-9]", "", chr_chr)))
      chr_pre <- gsub("[0-9]", "", chr_chr)
      ord_lvls <- unique(chr_chr[order(chr_pre, chr_num, chr_chr)])
      r$chrom_f <- factor(chr_chr, levels = ord_lvls)
      r <- r[order(r$chrom_f, r$pos_x), ]
      # cumulative genomic position with per-chromosome offset
      chr_max <- tapply(r$pos_x, r$chrom_f, max)
      chr_max[is.na(chr_max)] <- 0
      offset <- c(0, utils::head(cumsum(as.numeric(chr_max)), -1))
      names(offset) <- levels(r$chrom_f)
      r$x <- r$pos_x + offset[as.character(r$chrom_f)]
      ticks <- tapply(r$x, r$chrom_f, function(z) (min(z) + max(z)) / 2)
      tick_df <- data.frame(pos = as.numeric(ticks), lab = names(ticks))
      tick_df <- tick_df[is.finite(tick_df$pos), ]
      shade_idx <- as.integer(r$chrom_f) %% 2L     # alternating chromosome shade
      xlab <- paste0("genomic position (", sx, " subgenome anchor)")
      use_chrom_axis <- TRUE
    } else {
      r <- r[order(r$neglog10p, decreasing = FALSE), ]
      r$x <- seq_len(nrow(r))
      shade_idx <- rep(0L, nrow(r))
      tick_df <- NULL; use_chrom_axis <- FALSE
      xlab <- paste0(pair_lab, " homoeolog pair (rank)")
    }
    # significance: prefer the engine's bonferroni_sig flag, else recompute
    if ("bonferroni_sig" %in% names(r)) {
      # robust to numeric ("1"/"0") and logical ("TRUE"/"T"/"YES") spellings
      bs <- toupper(trimws(as.character(r$bonferroni_sig)))
      r$sig <- bs %in% c("1", "TRUE", "T", "YES")
    } else {
      r$sig <- r$p_interaction < bonf
    }
    # colour: alternating chromosome shade for non-sig, red highlight for sig
    r$col <- ifelse(r$sig, "sig", ifelse(shade_idx == 0L, "shadeA", "shadeB"))
    pal <- c(shadeA = "#9DB4C0", shadeB = "#5E7A8A", sig = "#CB3E35")
    # labels: sig pairs (cap topn); if none significant, label the single best
    # pair WITH its p so an honest-null plot still shows "best was p=…"
    sig_rows <- r[r$sig, , drop = FALSE]
    lab <- if (nrow(sig_rows) > 0) {
      utils::head(sig_rows[order(sig_rows$p_interaction), ], topn)
    } else {
      utils::head(r[order(r$p_interaction), ], 1)
    }
    lab$txt <- paste0(lab$gene_x, "×", lab$gene_y, "  p=", fmt_p(lab$p_interaction))
    p <- ggplot(r, aes(x, neglog10p)) +
      geom_point(aes(colour = col), size = 1.6, alpha = 0.85) +
      geom_hline(yintercept = -log10(bonf), colour = "#CB3E35",
                 linetype = 2, linewidth = 0.5) +
      scale_colour_manual(values = pal, guide = "none") +
      labs(x = xlab, y = expression(-log[10](p[interaction])),
           caption = paste0(pair_lab, " interaction scan · Bonferroni p=", fmt_p(bonf))) +
      theme_omic()
    if (use_chrom_axis && nrow(tick_df) > 0) {
      p <- p + scale_x_continuous(breaks = tick_df$pos, labels = tick_df$lab,
                                  expand = ggplot2::expansion(mult = 0.02))
      if (nrow(tick_df) > 12)
        p <- p + theme(axis.text.x = element_text(angle = 60, hjust = 1, size = 7))
    }
    if (nrow(lab) > 0) {
      if (requireNamespace("ggrepel", quietly = TRUE)) {
        p <- p + ggrepel::geom_text_repel(data = lab, aes(label = txt),
                   size = 2.6, colour = "#333333", min.segment.length = 0,
                   box.padding = 0.5, seed = 2026, max.overlaps = Inf)
      } else {
        p <- p + geom_text(data = lab, aes(label = txt), size = 2.6,
                   vjust = -0.7, colour = "#333333", check_overlap = TRUE)
      }
    }
    ggw(p, "interaction_manhattan", w = 9, h = 4.5)
  }

  if ("marginal" %in% dist_kinds) {
    m <- utils::head(r[order(r$p_interaction), ], topn)
    # subgenome labels are derived from the data (sub_x/sub_y) so the legend reads
    # "A×D" for cotton, "A×B" for peanut, etc. — never hard-coded.
    sx <- as.character(m$sub_x[1]); sy <- as.character(m$sub_y[1])
    lab_a <- paste0(sx, "-copy marginal")
    lab_d <- paste0(sy, "-copy marginal")
    lab_i <- paste0(sx, "×", sy, " interaction")
    pn <- paste0(m$gene_x, "×", m$gene_y)
    long <- data.frame(
      pair = factor(rep(pn, 3), levels = rev(pn)),
      test = factor(rep(c(lab_a, lab_d, lab_i), each = nrow(m)),
                    levels = c(lab_a, lab_d, lab_i)),
      nlp = c(as.numeric(m$neglog10p_marginal_x),
              as.numeric(m$neglog10p_marginal_y), m$neglog10p))
    p <- ggplot(long, aes(nlp, pair, colour = test)) +
      geom_line(aes(group = pair), colour = "#cccccc", linewidth = 0.6) +
      geom_point(size = 3) +
      geom_vline(xintercept = -log10(bonf), colour = "#CB3E35",
                 linetype = 2, linewidth = 0.4) +
      scale_colour_manual(values = setNames(c("#1F577B", "#FCBC10", "#CB3E35"),
                                            c(lab_a, lab_d, lab_i))) +
      labs(x = expression(-log[10](p)), y = NULL, colour = NULL) +
      theme_omic()
    ggw(p, "interaction_vs_marginal", w = 7.5, h = 0.45 * nrow(m) + 1.8)
  }

  if ("burden" %in% dist_kinds) {
    b <- read_tsv(args[["burdens"]])
    b <- b[b$pair_rank == 0, ]                   # top pair only
    b$bxv <- as.numeric(b$burden_x)
    b$byv <- as.numeric(b$burden_y)
    b$phenotype <- as.numeric(b$phenotype)
    b <- b[is.finite(b$bxv) & is.finite(b$byv) & is.finite(b$phenotype), ]
    sx <- as.character(b$sub_x[1]); sy <- as.character(b$sub_y[1])
    gx <- as.character(b$gene_x[1]); gy <- as.character(b$gene_y[1])

    # --- PRIMARY: fitted A×D interaction lines (ggplot2 only; always rendered) ---
    # The effect-based view: predicted phenotype vs the sx-copy burden at low/high
    # sy-copy burden. Diverging lines = interaction; small-PVE signals that vanish
    # in raw-group plots still show here. p is the OFFICIAL engine value matched
    # from the ranking TSV (falls back to the lm interaction term if absent).
    fmt_e <- function(x) formatC(as.numeric(x), format = "e", digits = 1)
    pint <- NA_real_
    if (!is.null(args[["ranking"]])) {
      rr <- tryCatch(read_tsv(args[["ranking"]]), error = function(e) NULL)
      if (!is.null(rr) && ncol(rr) >= 3) {
        names(rr)[2:3] <- c("gene_x", "gene_y")
        hit <- rr[as.character(rr$gene_x) == gx & as.character(rr$gene_y) == gy, ]
        if (nrow(hit) > 0) pint <- suppressWarnings(as.numeric(hit$p_interaction[1]))
      }
    }
    mfit <- stats::lm(phenotype ~ bxv * byv, data = b)
    if (!is.finite(pint)) {
      cf <- stats::coef(summary(mfit))
      if ("bxv:byv" %in% rownames(cf)) pint <- cf["bxv:byv", 4]
    }
    axseq <- seq(stats::quantile(b$bxv, .02), stats::quantile(b$bxv, .98), length = 100)
    lev <- stats::quantile(b$byv, c(.10, .90))
    nm  <- c(paste0(sy, " low"), paste0(sy, " high"))
    pf <- do.call(rbind, lapply(1:2, function(i) {
      nd <- data.frame(bxv = axseq, byv = as.numeric(lev[i]))
      pr <- stats::predict(mfit, nd, se.fit = TRUE)
      data.frame(bxv = axseq, grp = nm[i], fit = pr$fit,
                 lo = pr$fit - 1.96 * pr$se.fit, hi = pr$fit + 1.96 * pr$se.fit)
    }))
    pf$grp <- factor(pf$grp, levels = nm)
    fcols <- stats::setNames(c("#2C7FB8", "#C44E52"), nm)
    ann <- if (is.finite(pint)) {
      paste0(sx, "×", sy, " homoeolog interaction   p = ", fmt_e(pint))
    } else paste0(sx, "×", sy, " homoeolog interaction")
    pfit <- ggplot(pf, aes(bxv, fit, colour = grp, fill = grp)) +
      geom_ribbon(aes(ymin = lo, ymax = hi), alpha = 0.16, colour = NA) +
      geom_line(linewidth = 1.4) +
      scale_colour_manual(values = fcols, name = paste0(sy, "-copy burden")) +
      scale_fill_manual(values = fcols, name = paste0(sy, "-copy burden")) +
      annotate("text", x = min(axseq), y = Inf, hjust = 0, vjust = 1.8,
               size = 4.4, fontface = "bold", label = ann) +
      labs(x = paste0(sx, "-copy burden  (", gx, ")"),
           y = "predicted phenotype (95% CI)") +
      theme_omic()
    ggw(pfit, "burden_interaction", w = 7, h = 5)

    # --- SUPPLEMENTARY: raw-data group views (grafify) ---
    if (!requireNamespace("grafify", quietly = TRUE)) {
      message("burden: grafify not installed — skipping burden_violin/burden_bar ",
              "(install.packages('grafify'))")
    } else {
      # median-split each copy's burden into low/high; the subgenome id lives in
      # the factor LEVELS so grafify's tidy-eval column args stay species-agnostic.
      b$grpA <- factor(ifelse(b$bxv > stats::median(b$bxv),
                              paste0(sx, " high"), paste0(sx, " low")),
                       levels = c(paste0(sx, " low"), paste0(sx, " high")))
      b$grpD <- factor(ifelse(b$byv > stats::median(b$byv),
                              paste0(sy, " high"), paste0(sy, " low")),
                       levels = c(paste0(sy, " low"), paste0(sy, " high")))
      flab <- paste0(sy, "-copy burden")
      pv <- grafify::plot_4d_scatterviolin(data = b, xcol = grpA, ycol = phenotype,
                                           boxes = grpD, s_alpha = 0.5) +
        ggplot2::labs(x = NULL, y = "phenotype", fill = flab)
      ggw(pv, "burden_violin", w = 6.5, h = 5)
      pb <- grafify::plot_4d_scatterbar(data = b, xcol = grpA, ycol = phenotype,
                                        bars = grpD, ErrorType = "SEM", s_alpha = 0.4) +
        ggplot2::labs(x = NULL, y = "mean phenotype", fill = flab)
      ggw(pb, "burden_bar", w = 6.5, h = 5)
    }
  }

  if ("triad" %in% dist_kinds && !is.null(args[["triad"]])) {
    t <- read_tsv(args[["triad"]])
    pcols <- grep("^p_[A-Za-z0-9]{2}$", names(t), value = TRUE)
    pcols <- setdiff(pcols, "p_acat")
    if (length(pcols) >= 2) {
      gcols <- names(t)[2:4]
      t$p_acat <- as.numeric(t$p_acat)
      tt <- utils::head(t[order(t$p_acat), ], topn)
      tt$triad <- apply(tt[, gcols], 1, paste, collapse = "/")
      long <- do.call(rbind, lapply(pcols, function(pc) data.frame(
        triad = tt$triad, pair = sub("^p_", "", pc),
        nlp = -log10(pmax(as.numeric(tt[[pc]]), 1e-300)))))
      long$triad <- factor(long$triad, levels = rev(tt$triad))
      p <- ggplot(long, aes(nlp, triad, fill = pair)) +
        geom_col(position = position_dodge(width = 0.7), width = 0.65,
                 alpha = 0.92) +
        scale_fill_manual(values = c(AB = "#1F577B", AD = "#FCBC10",
                                     BD = "#E07370", AC = "#368650",
                                     CD = "#A56BA7")) +
        labs(x = expression(-log[10](p)), y = NULL, fill = "homoeolog pair",
             caption = "pairwise interaction within each triad (ACAT-ranked); longest bar = signal-carrying pair") +
        theme_omic()
      ggw(p, "triad_breakdown", w = 7.5, h = 0.5 * nrow(tt) + 1.8)
    }
  }

  if ("network" %in% dist_kinds) {
    if (!have("ggraph") || !have("igraph")) {
      message("network: ggraph/igraph not installed; skipping (the other ",
              "distinctive figures were produced)")
    } else {
      suppressWarnings(suppressMessages({library(ggraph); library(igraph)}))
      sg <- r[r$p_interaction < 0.05, ]
      if (nrow(sg) >= 1) {
        ed <- data.frame(from = paste0(sg$gene_x, "(", sg$sub_x, ")"),
                         to = paste0(sg$gene_y, "(", sg$sub_y, ")"),
                         w = sg$neglog10p)
        g <- igraph::graph_from_data_frame(ed, directed = FALSE)
        p <- ggraph(g, layout = "fr") +
          geom_edge_link(aes(width = w), colour = "#CB3E35", alpha = 0.6) +
          geom_node_point(size = 3, colour = "#1F577B") +
          geom_node_text(aes(label = name), size = 2.4, repel = TRUE) +
          scale_edge_width(range = c(0.3, 2), name = "-log10 p") +
          labs(title = paste0("Significant homoeolog-interaction network — ",
                              trait)) +
          theme_void()
        ggw(p, "interaction_network", w = 7, h = 6)
      }
    }
  }
}

for (w in written) cat("WROTE\t", w, "\n", sep = "")
quit(status = status_code)
