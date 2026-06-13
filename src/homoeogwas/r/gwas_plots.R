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
    v$tlab <- sprintf("%s\n%.0f%%\nσ²=%.1f", v$component, 100 * v$pve, v$sigma2)
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
  }

  if ("interaction" %in% dist_kinds) {
    has_pos <- "pos_x" %in% names(r) &&
      all(is.finite(suppressWarnings(as.numeric(r$pos_x))))
    if (has_pos) {
      r$pos_x <- as.numeric(r$pos_x)
      r <- r[order(as.character(r$chrom_x), r$pos_x), ]
      xlab <- "homoeolog pair (genomic order)"
    } else {
      r <- r[order(r$neglog10p, decreasing = FALSE), ]
      xlab <- "homoeolog pair (rank)"
    }
    r$x <- seq_len(nrow(r))
    r$sig <- r$p_interaction < bonf
    lab <- utils::head(r[order(r$p_interaction), ], topn)
    lab <- lab[lab$p_interaction < 0.05, ]
    p <- ggplot(r, aes(x, neglog10p)) +
      geom_point(aes(colour = sig), size = 1.5, alpha = 0.8) +
      geom_hline(yintercept = -log10(bonf), colour = "#CB3E35",
                 linewidth = 0.5) +
      scale_colour_manual(values = c(`FALSE` = "#9DB4C0", `TRUE` = "#CB3E35"),
                          guide = "none") +
      labs(title = paste0("Homoeolog-pair interaction scan — ", trait),
           subtitle = paste0("each point = one homoeolog pair; red line = ",
                             "Bonferroni (0.05/", nrow(r), ")"),
           x = xlab, y = expression(-log[10](p[interaction]))) +
      theme_omic()
    if (nrow(lab) > 0) {
      p <- p + geom_text(data = lab,
                         aes(label = paste0(gene_x, "×", gene_y)),
                         size = 2.5, vjust = -0.6, hjust = 0.5,
                         colour = "#333333")
    }
    ggw(p, "interaction_manhattan", w = 9, h = 4.5)
  }

  if ("marginal" %in% dist_kinds) {
    m <- utils::head(r[order(r$p_interaction), ], topn)
    pn <- paste0(m$gene_x, "×", m$gene_y)
    long <- data.frame(
      pair = factor(rep(pn, 3), levels = rev(pn)),
      test = rep(c("A-copy marginal", "D-copy marginal", "A×D interaction"),
                 each = nrow(m)),
      nlp = c(as.numeric(m$neglog10p_marginal_x),
              as.numeric(m$neglog10p_marginal_y), m$neglog10p))
    p <- ggplot(long, aes(nlp, pair, colour = test)) +
      geom_line(aes(group = pair), colour = "#cccccc", linewidth = 0.6) +
      geom_point(size = 3) +
      geom_vline(xintercept = -log10(bonf), colour = "#CB3E35",
                 linetype = 2, linewidth = 0.4) +
      scale_colour_manual(values = c("A-copy marginal" = "#1F577B",
                                     "D-copy marginal" = "#FCBC10",
                                     "A×D interaction" = "#CB3E35")) +
      labs(title = paste0("Interaction vs single-gene marginal — ", trait),
           subtitle = "signal lives in the A×D product, not in either copy alone",
           x = expression(-log[10](p)), y = NULL, colour = NULL) +
      theme_omic()
    ggw(p, "interaction_vs_marginal", w = 7.5, h = 0.45 * nrow(m) + 1.8)
  }

  if ("burden" %in% dist_kinds) {
    b <- read_tsv(args[["burdens"]])
    b <- b[b$pair_rank == 0, ]
    b$burden_x <- as.numeric(b$burden_x)
    b$burden_y <- as.numeric(b$burden_y)
    b$phenotype <- as.numeric(b$phenotype)
    qx <- stats::quantile(b$burden_x, probs = seq(0, 1, 0.25), na.rm = TRUE)
    qy <- stats::quantile(b$burden_y, probs = seq(0, 1, 0.25), na.rm = TRUE)
    b$bx <- cut(b$burden_x, unique(qx), include.lowest = TRUE)
    b$by <- cut(b$burden_y, unique(qy), include.lowest = TRUE)
    agg <- stats::aggregate(phenotype ~ bx + by, data = b, FUN = mean)
    gx <- b$gene_x[1]; gy <- b$gene_y[1]
    p <- ggplot(agg, aes(bx, by, fill = phenotype)) +
      geom_tile(colour = "white") +
      scale_fill_viridis_c() +
      labs(title = paste0("A×D burden interaction surface — ", trait),
           subtitle = paste0("top pair ", gx, " × ", gy,
                             "; phenotype rises only when both burdens combine"),
           x = paste0(gx, " burden (A-copy, quartile)"),
           y = paste0(gy, " burden (D-copy, quartile)"),
           fill = "mean\nphenotype") +
      theme_omic() +
      theme(axis.text.x = element_text(angle = 30, hjust = 1))
    ggw(p, "burden_surface", w = 6.5, h = 5)
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
        labs(title = paste0("Triad pairwise interaction breakdown — ", trait),
             subtitle = "which homoeolog pair within each triad carries the signal",
             x = expression(-log[10](p)), y = NULL, fill = "pair") +
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
