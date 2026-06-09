"""Publication-grade plotting for HomoeoGWAS.

This module owns every figure the tool emits. The headline figure is the
per-subgenome **variance-component / PVE** plot — a one-glance view of how
genetic variance partitions across the A/B/D (or A/C, A/D) subgenomes plus the
optional homoeolog Hadamard kernel and residual. No generic GWAS tool produces
it, so it doubles as the tool's visual signature. The companion figures
(subgenome-faceted Manhattan, stratified QQ with per-subgenome lambda, and a
lambda_GC QC bar) are styled to match.

Two entry points:

* :func:`make_all_plots` — orchestrator called by ``homoeogwas fit`` right
  after the scan, and by the ``plot`` subcommand after it reconstructs the same
  inputs from a finished run directory (no model refit).
* the per-figure functions, usable standalone.

Everything renders through the Agg backend with a publication ``rc_context`` so
global Matplotlib state is never mutated, and every figure is written as a PNG
preview plus editable PDF and SVG vectors.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# theme + palette
# ----------------------------------------------------------------------

DEFAULT_FORMATS: tuple[str, ...] = ("png", "pdf", "svg")
ALL_FIGURES: tuple[str, ...] = ("pve", "manhattan", "qq", "lambda")

# Okabe-Ito colourblind-safe palette, assigned to subgenomes by position.
_OKABE_ITO = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00",
              "#F0E442", "#000000"]
_HOM_COLOR = "#CC79A7"      # homoeolog Hadamard kernel (purple)
_RESID_COLOR = "#999999"    # residual variance (grey)
_GW_LINE = "#D55E00"        # genome-wide threshold (vermillion)
_SUGG_LINE = "#777777"      # suggestive threshold (grey)

_HOM_ALIASES = {"hom", "khom", "k_hom", "homoeolog", "hadamard"}

_PUB_RC = {
    "figure.dpi": 110,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "svg.fonttype": "none",   # keep text editable in Illustrator/Inkscape
    "pdf.fonttype": 42,       # embed TrueType, not Type-3
    "ps.fonttype": 42,
    "legend.frameon": False,
}


def _import_mpl():
    """Import matplotlib with the Agg backend, returning ``(plt, mpl)``."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt, matplotlib


def is_hom_kernel(name: str) -> bool:
    """True if ``name`` looks like the homoeolog Hadamard kernel."""
    return name.lower() in _HOM_ALIASES


def subgenome_palette(subg: Sequence[str]) -> dict[str, str]:
    """Map each subgenome label to a stable colourblind-safe colour."""
    return {sg: _OKABE_ITO[i % len(_OKABE_ITO)] for i, sg in enumerate(subg)}


def component_color(name: str, palette: Mapping[str, str]) -> str:
    """Colour for a variance component (subgenome, homoeolog, or residual)."""
    if name == "e":
        return _RESID_COLOR
    if is_hom_kernel(name):
        return _HOM_COLOR
    return palette.get(name, _RESID_COLOR)


def _lighten(hex_color: str, amount: float = 0.45) -> str:
    """Blend ``hex_color`` toward white by ``amount`` (0 = same, 1 = white)."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    r = int(r + (255 - r) * amount)
    g = int(g + (255 - g) * amount)
    b = int(b + (255 - b) * amount)
    return f"#{r:02x}{g:02x}{b:02x}"


def save_figure(fig, out_dir: Path, stem: str,
                formats: Sequence[str] = DEFAULT_FORMATS,
                dpi: int = 300) -> list[str]:
    """Write ``fig`` as ``stem.<fmt>`` for each requested format.

    PNG is a raster preview at ``dpi``; PDF/SVG are vectors (dpi ignored).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for fmt in formats:
        path = out_dir / f"{stem}.{fmt}"
        fig.savefig(path, format=fmt, dpi=dpi if fmt == "png" else None)
        written.append(str(path))
    return written


# ----------------------------------------------------------------------
# figure 1 (signature): per-subgenome variance components / PVE
# ----------------------------------------------------------------------

def plot_variance_components(sigma2: Mapping[str, float],
                             pve: Mapping[str, float],
                             subg: Sequence[str],
                             *,
                             boundary_components: Sequence[str] = (),
                             out_dir: Path,
                             prefix: str,
                             trait: str = "",
                             formats: Sequence[str] = DEFAULT_FORMATS,
                             dpi: int = 300) -> list[str]:
    """Signature figure: absolute variance (left) + stacked PVE (right).

    Boundary components (variance driven to zero by REML) are kept and hatched
    so a collapsed subgenome reads as informative rather than missing.
    """
    plt, _ = _import_mpl()
    palette = subgenome_palette(subg)
    boundary = set(boundary_components)

    # order: subgenomes (in declared order), homoeolog kernel(s), then residual
    kernel_keys = [k for k in sigma2 if k != "e"]
    ordered = [k for k in subg if k in kernel_keys]
    ordered += [k for k in kernel_keys if k not in ordered]
    comps = ordered + (["e"] if "e" in sigma2 else [])
    labels = ["residual" if c == "e" else
              ("homoeolog" if is_hom_kernel(c) else c) for c in comps]
    colors = [component_color(c, palette) for c in comps]

    with plt.rc_context(_PUB_RC):
        fig, (axv, axp) = plt.subplots(
            1, 2, figsize=(7.0, 3.4),
            gridspec_kw={"width_ratios": [1.5, 1.0]})

        # left: absolute variance bars
        xs = np.arange(len(comps))
        vals = [float(sigma2.get(c, 0.0)) for c in comps]
        bars = axv.bar(xs, vals, color=colors, edgecolor="white", linewidth=0.6)
        for c, bar in zip(comps, bars, strict=False):
            if c in boundary:
                bar.set_hatch("///")
                bar.set_edgecolor("#333333")
                axv.annotate("boundary", (bar.get_x() + bar.get_width() / 2,
                                          bar.get_height()),
                             ha="center", va="bottom", fontsize=6,
                             color="#333333", rotation=90, xytext=(0, 2),
                             textcoords="offset points")
        axv.set_xticks(xs)
        axv.set_xticklabels(labels, rotation=0)
        axv.set_ylabel(r"variance component $\sigma^2$")
        axv.set_title("Variance components")
        axv.margins(y=0.18)

        # right: 100% stacked PVE bar
        bottom = 0.0
        for c, lab, col in zip(comps, labels, colors, strict=False):
            frac = float(pve.get(c, 0.0))
            if frac <= 0:
                continue
            axp.bar(0, frac, bottom=bottom, width=0.6, color=col,
                    edgecolor="white", linewidth=0.6,
                    hatch="///" if c in boundary else None)
            if frac >= 0.04:    # only label readable segments
                axp.text(0, bottom + frac / 2, f"{lab}\n{frac * 100:.0f}%",
                         ha="center", va="center", fontsize=6,
                         color="white" if c != "e" else "#222222")
            bottom += frac
        axp.set_xlim(-0.6, 0.6)
        axp.set_ylim(0, 1.0)
        axp.set_xticks([])
        axp.set_ylabel("proportion of variance explained")
        axp.set_title("PVE partition")

        if trait:
            fig.suptitle(trait, fontsize=10, y=1.02)
        fig.tight_layout()
        paths = save_figure(fig, out_dir, f"variance_components_{prefix}",
                            formats, dpi)
        plt.close(fig)
    return paths


# ----------------------------------------------------------------------
# figure 2: subgenome-faceted Manhattan
# ----------------------------------------------------------------------

def _genomic_layout(df: pd.DataFrame):
    """Assign a cumulative x to each marker from its physical position.

    Chromosomes are laid out in first-appearance order (the sumstats are written
    in genomic order, so this avoids the lexicographic ``chr10 < chr2`` trap),
    and within a chromosome x is the base-pair coordinate so spacing is faithful
    to physical distance regardless of marker thinning.

    Returns ``(x, chrom_centers, chrom_labels, chrom_index)`` where
    ``chrom_index`` is the 0-based ordinal of each marker's chrom (for shading).
    """
    chrom = df["chrom"].astype(str).to_numpy()
    pos = df["pos"].to_numpy(dtype=np.float64)
    x = np.empty(len(df), dtype=np.float64)
    cidx = np.empty(len(df), dtype=np.int64)
    centers, labels = [], []
    offset = 0.0
    for j, ci in enumerate(dict.fromkeys(chrom)):   # genomic / input order
        msk = chrom == ci
        cpos = pos[msk]
        lo = float(cpos.min()) if cpos.size else 0.0
        hi = float(cpos.max()) if cpos.size else 1.0
        span = max(hi - lo, 1.0)
        x[msk] = offset + (cpos - lo)
        cidx[msk] = j
        centers.append(offset + span / 2.0)
        labels.append(ci)
        offset += span + max(span * 0.01, 1.0)      # small inter-chrom gap
    return x, centers, labels, cidx


def plot_manhattan(df: pd.DataFrame,
                   subg: Sequence[str],
                   *,
                   out_dir: Path,
                   prefix: str,
                   trait: str = "",
                   genomewide: float = 5e-8,
                   suggestive: float = 1e-5,
                   top_n: int = 5,
                   formats: Sequence[str] = DEFAULT_FORMATS,
                   dpi: int = 300) -> list[str]:
    """One Manhattan panel per subgenome, sharing the -log10(p) axis.

    Within each panel, chroms alternate between the subgenome's full and a
    lighter shade. Genome-wide (solid) and suggestive (dashed) lines are drawn,
    and the strongest ``top_n`` hits per panel are labelled ``chrom:pos``.
    """
    plt, _ = _import_mpl()
    if len(df) == 0:
        return []
    palette = subgenome_palette(subg)
    present = [s for s in subg if (df["subgenome"].astype(str) == s).any()]
    if not present:
        present = sorted(df["subgenome"].astype(str).unique())
    n_panel = len(present)

    gw_y = -math.log10(genomewide)
    sugg_y = -math.log10(suggestive)

    with plt.rc_context(_PUB_RC):
        # each subgenome holds different chromosomes, so panels do NOT share x:
        # every panel carries its own genomic layout and chrom ticks.
        fig, axes = plt.subplots(n_panel, 1, sharex=False,
                                 figsize=(11, 1.9 * n_panel + 0.6),
                                 squeeze=False)
        axes = axes[:, 0]
        ymax = 1.0
        panel_ticks: list[tuple] = []
        for ax, sg in zip(axes, present, strict=False):
            sub = df[df["subgenome"].astype(str) == sg]
            if len(sub) == 0:
                ax.set_visible(False)
                panel_ticks.append(([], []))
                continue
            p = np.clip(sub["p"].to_numpy(dtype=np.float64), 1e-300, 1.0)
            logp = -np.log10(p)
            x, centers, labels, cidx = _genomic_layout(sub)
            base = palette.get(sg, "#1f77b4")
            light = _lighten(base, 0.5)
            colors = np.where((cidx % 2) == 0, base, light)
            ax.scatter(x, logp, s=4, c=colors, linewidths=0, rasterized=True)
            ax.axhline(gw_y, ls="-", lw=0.8, color=_GW_LINE)
            ax.axhline(sugg_y, ls="--", lw=0.7, color=_SUGG_LINE)
            ax.set_ylabel(f"{sg}\n$-\\log_{{10}}p$")
            ax.margins(x=0.01)
            ymax = max(ymax, float(logp.max()) if logp.size else 1.0)
            panel_ticks.append((centers, labels))

            # label the strongest independent-ish hits
            if top_n > 0 and logp.size:
                idx = np.argsort(logp)[::-1][:top_n]
                seen_x: list[float] = []
                chrom_arr = sub["chrom"].astype(str).to_numpy()
                pos_arr = sub["pos"].to_numpy()
                for i in idx:
                    if logp[i] < sugg_y:
                        break
                    if any(abs(x[i] - sx) < 0.02 * (x.max() + 1)
                           for sx in seen_x):
                        continue
                    seen_x.append(x[i])
                    ax.annotate(f"{chrom_arr[i]}:{int(pos_arr[i])}",
                                (x[i], logp[i]), fontsize=6, rotation=45,
                                ha="left", va="bottom", xytext=(2, 2),
                                textcoords="offset points", color="#222222")
        # shared y-range; each visible panel keeps its own chrom ticks
        for ax, (centers, labels) in zip(axes, panel_ticks, strict=False):
            if not ax.get_visible():
                continue
            ax.set_ylim(0, ymax * 1.12)
            ax.set_xticks(centers)
            ax.set_xticklabels(labels, rotation=90, fontsize=6)
        axes[-1].set_xlabel("chromosome")
        title = f"{trait}" if trait else "Manhattan"
        fig.suptitle(title, fontsize=10, y=0.995)
        fig.tight_layout(rect=(0, 0, 1, 0.99))
        paths = save_figure(fig, out_dir, f"manhattan_{prefix}", formats, dpi)
        plt.close(fig)
    return paths


# ----------------------------------------------------------------------
# figure 3: stratified QQ with per-subgenome lambda + 95% band
# ----------------------------------------------------------------------

def _qq_ranks(logp_desc: np.ndarray, n_true: int,
              keep_p: float = 1e-3) -> np.ndarray:
    """True order-statistic ranks for a thinned, sorted-descending logp array.

    The plot frame keeps **every** marker with ``p <= keep_p`` plus a uniform
    genome-wide stride sample of the rest (see ``scan_summary``). So the tail
    (``-log10p >= -log10 keep_p``) is complete and gets exact ranks
    ``1..n_tail``; the retained bulk is one point per ``stride``-sized rank bin,
    so the i-th retained bulk marker is placed at its bin *midpoint* rank
    ``n_tail + (n_true-n_tail)*(i-0.5)/n_bulk`` rather than at the bin edges
    (a plain ``linspace`` would push the first bulk point too extreme and the
    last too null). Expected quantiles built from these ranks against the *true*
    ``n_true`` are therefore unbiased even though the frame is thinned.
    """
    m = logp_desc.size
    n_true = max(int(n_true), m)
    ranks = np.empty(m, dtype=np.float64)
    n_tail = int(np.count_nonzero(logp_desc >= -math.log10(keep_p)))
    n_tail = min(n_tail, m)
    ranks[:n_tail] = np.arange(1, n_tail + 1)
    n_bulk = m - n_tail
    if n_bulk > 0:
        if m == n_true:
            ranks[n_tail:] = np.arange(n_tail + 1, m + 1)
        else:
            i = np.arange(1, n_bulk + 1)
            ranks[n_tail:] = n_tail + (n_true - n_tail) * (i - 0.5) / n_bulk
    return ranks


def plot_qq(df: pd.DataFrame,
            subg: Sequence[str],
            lambda_by_level: Mapping[str, float],
            counts_by_level: Mapping[str, int] | None = None,
            *,
            out_dir: Path,
            prefix: str,
            trait: str = "",
            formats: Sequence[str] = DEFAULT_FORMATS,
            dpi: int = 300) -> list[str]:
    """Overlaid QQ: 'all' plus one curve per subgenome, each labelled with its
    lambda_GC, over a 95% all-marker null confidence band.

    ``counts_by_level`` gives the *true* (un-thinned) marker count per level so
    expected quantiles are computed against the real number of tests.
    """
    plt, _ = _import_mpl()
    if len(df) == 0:
        return []
    counts_by_level = counts_by_level or {}
    palette = subgenome_palette(subg)
    present = [s for s in subg if (df["subgenome"].astype(str) == s).any()]

    def _count(level: str, fallback: int) -> int:
        c = counts_by_level.get(level)
        return int(c) if c and c > 0 else fallback

    with plt.rc_context(_PUB_RC):
        fig, ax = plt.subplots(figsize=(4.8, 4.6))

        p_all = np.clip(df["p"].to_numpy(dtype=np.float64), 1e-300, 1.0)
        logp_all = np.sort(-np.log10(p_all))[::-1]
        n_all = _count("all", logp_all.size)
        ranks_all = _qq_ranks(logp_all, n_all)
        exp_all = -np.log10((ranks_all - 0.5) / n_all)

        # 95% all-marker null band at the displayed expected quantiles
        if logp_all.size > 1:
            try:
                from scipy.stats import beta as _beta
                lo = _beta.ppf(0.025, ranks_all, n_all - ranks_all + 1)
                hi = _beta.ppf(0.975, ranks_all, n_all - ranks_all + 1)
                ax.fill_between(exp_all, -np.log10(hi), -np.log10(lo),
                                color="#cccccc", alpha=0.6, lw=0,
                                label="95% null band (all)")
            except Exception:
                pass

        lim = (float(max(exp_all.max(), logp_all.max())) * 1.05
               if logp_all.size else 1.0)
        ax.plot([0, lim], [0, lim], ls="--", color="k", lw=0.9)

        lam_all = lambda_by_level.get("all")
        lab_all = ("all" + (f" (λ={lam_all:.3f})" if lam_all is not None
                            else ""))
        ax.scatter(exp_all, logp_all, s=5, color="#333333", label=lab_all,
                   rasterized=True)
        for sg in present:
            sub = df[df["subgenome"].astype(str) == sg]
            ps = np.clip(sub["p"].to_numpy(dtype=np.float64), 1e-300, 1.0)
            lp = np.sort(-np.log10(ps))[::-1]
            n_sg = _count(sg, lp.size)
            ranks = _qq_ranks(lp, n_sg)
            ex = -np.log10((ranks - 0.5) / n_sg)
            lam = lambda_by_level.get(sg)
            lab = sg + (f" (λ={lam:.3f})" if lam is not None else "")
            ax.scatter(ex, lp, s=5, color=palette.get(sg, "#1f77b4"),
                       label=lab, rasterized=True)

        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_xlabel(r"expected $-\log_{10}p$")
        ax.set_ylabel(r"observed $-\log_{10}p$")
        if trait:
            ax.set_title(trait)
        ax.legend(loc="upper left", fontsize=6)
        fig.tight_layout()
        paths = save_figure(fig, out_dir, f"qq_{prefix}", formats, dpi)
        plt.close(fig)
    return paths


# ----------------------------------------------------------------------
# figure 4: lambda_GC QC bar
# ----------------------------------------------------------------------

def plot_lambda_gc(lambda_df: pd.DataFrame,
                   subg: Sequence[str],
                   *,
                   out_dir: Path,
                   prefix: str,
                   trait: str = "",
                   formats: Sequence[str] = DEFAULT_FORMATS,
                   dpi: int = 300) -> list[str]:
    """QC bar of lambda_GC for 'all' and each subgenome, with a y=1 reference."""
    plt, _ = _import_mpl()
    if lambda_df is None or len(lambda_df) == 0:
        return []
    palette = subgenome_palette(subg)
    rows = lambda_df.copy()
    rows["level"] = rows["level"].astype(str)
    # 'all' first, then subgenomes in declared order, then any extras
    order = ["all"] + list(subg)
    rows["__o"] = rows["level"].apply(
        lambda x: order.index(x) if x in order else len(order))
    rows = rows.sort_values("__o").reset_index(drop=True)

    with plt.rc_context(_PUB_RC):
        fig, ax = plt.subplots(figsize=(0.7 * len(rows) + 2.0, 3.2))
        xs = np.arange(len(rows))
        colors = ["#333333" if lv == "all" else palette.get(lv, "#999999")
                  for lv in rows["level"]]
        vals = rows["lambda_gc"].to_numpy(dtype=np.float64)
        bars = ax.bar(xs, vals, color=colors, edgecolor="white", linewidth=0.6)
        ax.axhline(1.0, ls="--", lw=0.8, color="#777777")
        for bar, v in zip(bars, vals, strict=False):
            ax.annotate(f"{v:.3f}", (bar.get_x() + bar.get_width() / 2,
                                     bar.get_height()),
                        ha="center", va="bottom", fontsize=6, xytext=(0, 1),
                        textcoords="offset points")
        ax.set_xticks(xs)
        ax.set_xticklabels(rows["level"], rotation=0)
        ax.set_ylabel(r"genomic inflation $\lambda_{GC}$")
        ax.set_title(trait if trait else "Genomic control")
        ax.margins(y=0.18)
        fig.tight_layout()
        paths = save_figure(fig, out_dir, f"lambda_gc_{prefix}", formats, dpi)
        plt.close(fig)
    return paths


# ----------------------------------------------------------------------
# orchestrator
# ----------------------------------------------------------------------

def make_all_plots(plot_df: pd.DataFrame,
                   subg: Sequence[str],
                   *,
                   sigma2: Mapping[str, float],
                   pve: Mapping[str, float],
                   boundary_components: Sequence[str] = (),
                   lambda_df: pd.DataFrame | None = None,
                   trait: str = "",
                   out_dir: Path,
                   prefix: str,
                   formats: Sequence[str] = DEFAULT_FORMATS,
                   dpi: int = 300,
                   top_n: int = 5,
                   figures: Sequence[str] = ALL_FIGURES) -> list[str]:
    """Render the full figure set; return all written paths.

    ``lambda_df`` has columns ``scope/level/n_markers/lambda_gc`` (the same
    frame written to ``lambda_gc_<prefix>.tsv``); the per-level lambda values
    annotate the QQ curves and drive the QC bar.
    """
    out_dir = Path(out_dir)
    lam_by_level: dict[str, float] = {}
    counts_by_level: dict[str, int] = {}
    if lambda_df is not None and len(lambda_df):
        lam_by_level = {str(r["level"]): float(r["lambda_gc"])
                        for _, r in lambda_df.iterrows()}
        if "n_markers" in lambda_df.columns:
            counts_by_level = {str(r["level"]): int(r["n_markers"])
                               for _, r in lambda_df.iterrows()}

    paths: list[str] = []
    if "pve" in figures:
        paths += plot_variance_components(
            sigma2, pve, subg, boundary_components=boundary_components,
            out_dir=out_dir, prefix=prefix, trait=trait, formats=formats,
            dpi=dpi)
    if "manhattan" in figures:
        paths += plot_manhattan(
            plot_df, subg, out_dir=out_dir, prefix=prefix, trait=trait,
            top_n=top_n, formats=formats, dpi=dpi)
    if "qq" in figures:
        paths += plot_qq(
            plot_df, subg, lam_by_level, counts_by_level, out_dir=out_dir,
            prefix=prefix, trait=trait, formats=formats, dpi=dpi)
    if "lambda" in figures and lambda_df is not None:
        paths += plot_lambda_gc(
            lambda_df, subg, out_dir=out_dir, prefix=prefix, trait=trait,
            formats=formats, dpi=dpi)
    return paths


# ----------------------------------------------------------------------
# reconstruct a finished run for the `plot` subcommand (no refit)
# ----------------------------------------------------------------------

def _discover_summary(results_dir: Path, prefix: str | None) -> Path:
    """Locate ``summary_<prefix>.json`` inside a finished run directory."""
    results_dir = Path(results_dir)
    if prefix:
        cand = results_dir / f"summary_{prefix}.json"
        if not cand.exists():
            raise FileNotFoundError(f"no summary for prefix '{prefix}': {cand}")
        return cand
    found = sorted(results_dir.glob("summary_*.json"))
    if not found:
        raise FileNotFoundError(f"no summary_*.json in {results_dir}")
    if len(found) > 1:
        names = ", ".join(p.name for p in found)
        raise ValueError(
            f"multiple summaries in {results_dir} ({names}); pass --prefix")
    return found[0]


def _load_plot_df(sumstats_paths: Sequence[str],
                  max_points: int = 300_000,
                  keep_p: float = 1e-3,
                  chunksize: int = 500_000) -> pd.DataFrame:
    """Stream sumstats TSV(s) into a thinned plot frame in bounded memory.

    Two passes so the full file never lands in RAM: pass 1 counts finite
    background markers (``p > keep_p``) to fix a global stride; pass 2 keeps
    every strong marker (``p <= keep_p``, so top hits survive) plus every
    ``stride``-th background marker. Non-finite p-values are dropped in both
    passes. Background memory is bounded near ``max_points``; the complete
    strong tail is always retained (it must survive for an honest QQ/Manhattan).
    """
    cols = ["subgenome", "chrom", "pos", "p"]
    # pass 1: count finite background rows
    n_bg = 0
    for path in sumstats_paths:
        for ch in pd.read_csv(path, sep="\t", usecols=["p"],
                              chunksize=chunksize):
            pv = ch["p"].to_numpy(dtype=np.float64)
            n_bg += int(np.count_nonzero(np.isfinite(pv) & (pv > keep_p)))
    stride = (max(1, int(math.ceil(n_bg / max_points)))
              if (max_points and n_bg) else 1)

    # pass 2: keep all strong + every stride-th background (global counter)
    kept: list[pd.DataFrame] = []
    bg_counter = 0
    for path in sumstats_paths:
        for ch in pd.read_csv(path, sep="\t", usecols=cols,
                              chunksize=chunksize):
            pv = ch["p"].to_numpy(dtype=np.float64)
            finite = np.isfinite(pv)
            strong = finite & (pv <= keep_p)
            if strong.any():
                kept.append(ch.iloc[np.nonzero(strong)[0]])
            bg_pos = np.nonzero(finite & (pv > keep_p))[0]
            if bg_pos.size:
                sel = bg_pos[((bg_counter + np.arange(bg_pos.size))
                              % stride) == 0]
                if sel.size:
                    kept.append(ch.iloc[sel])
                bg_counter += bg_pos.size
    out = (pd.concat(kept, ignore_index=True) if kept
           else pd.DataFrame(columns=cols))
    return out[cols] if len(out) else out


def _resolve_sumstats(results_dir: Path,
                      sumstats: Sequence[str]) -> list[str]:
    """Resolve sumstats paths relative to the run directory.

    Tries, in order: an absolute path that exists, ``results_dir / <as-stored>``
    (preserving any relative subdir), then ``results_dir / <basename>`` as a
    portable fallback for runs moved off their original machine.
    """
    out: list[str] = []
    for s in sumstats:
        p = Path(s)
        if p.is_absolute() and p.exists():
            out.append(str(p))
            continue
        cand = results_dir / s
        if cand.exists():
            out.append(str(cand))
            continue
        out.append(str(results_dir / p.name))
    return out


def plot_from_results(results_dir: Path,
                      *,
                      prefix: str | None = None,
                      formats: Sequence[str] = DEFAULT_FORMATS,
                      dpi: int = 300,
                      top_n: int = 5,
                      max_points: int = 300_000,
                      figures: Sequence[str] = ALL_FIGURES,
                      update_summary: bool = True) -> list[str]:
    """Regenerate every figure from a finished run directory (no model refit).

    Reads ``summary_<prefix>.json`` for variance components, subgenomes and
    lambda, ``lambda_gc_<prefix>.tsv`` for the QC bar, and the sumstats TSV(s)
    for Manhattan/QQ. Returns written paths; optionally rewrites the summary's
    ``outputs.plots``.
    """
    import json

    results_dir = Path(results_dir)
    summary_path = _discover_summary(results_dir, prefix)
    summary = json.loads(summary_path.read_text())
    run_prefix = summary_path.stem[len("summary_"):]

    reml = summary.get("reml", {})
    sigma2 = reml.get("sigma2", {})
    pve = reml.get("pve", {})
    boundary = reml.get("boundary_components", []) or []
    subg = summary.get("subgenomes", [])
    trait = summary.get("trait", "")

    lam_tsv = results_dir / f"lambda_gc_{run_prefix}.tsv"
    if lam_tsv.exists():
        lambda_df = pd.read_csv(lam_tsv, sep="\t")
    else:
        lam_map = summary.get("lambda_gc", {})
        lambda_df = pd.DataFrame(
            [{"scope": "all" if k == "all" else "subgenome",
              "level": k, "n_markers": -1, "lambda_gc": v}
             for k, v in lam_map.items()])

    sumstats = _resolve_sumstats(
        results_dir, summary.get("outputs", {}).get("sumstats", []))
    plot_df = _load_plot_df(sumstats, max_points=max_points)

    paths = make_all_plots(
        plot_df, subg, sigma2=sigma2, pve=pve, boundary_components=boundary,
        lambda_df=lambda_df, trait=trait, out_dir=results_dir,
        prefix=run_prefix, formats=formats, dpi=dpi, top_n=top_n,
        figures=figures)

    if update_summary:
        summary.setdefault("outputs", {})["plots"] = paths
        summary_path.write_text(json.dumps(summary, indent=2))
    return paths
