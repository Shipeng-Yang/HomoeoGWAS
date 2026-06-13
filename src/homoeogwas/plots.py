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

# omicverse-inspired muted palette (sc_color / blue_color family), assigned to
# subgenomes by position. Chosen for a polished, desaturated look while staying
# as distinguishable as practical (it is no longer the strictly colourblind-safe
# Okabe-Ito set; the previous default was kept in git history if needed).
_SUBG_PALETTE = ["#1F577B", "#E07370", "#368650", "#FCBC10", "#5E4D9A",
                 "#01A0A7", "#D48F3E", "#941456"]
_HOM_COLOR = "#A56BA7"      # homoeolog Hadamard kernel (omicverse muted purple)
_RESID_COLOR = "#B8B8B8"    # residual variance (soft grey)
_GW_LINE = "#CB3E35"        # genome-wide threshold (omicverse muted red)
_SUGG_LINE = "#9A9A9A"      # suggestive threshold (grey)

_HOM_ALIASES = {"hom", "khom", "k_hom", "homoeolog", "hadamard"}

# optional: prettier non-overlapping hit labels when adjustText is installed.
try:                                    # pragma: no cover - optional dependency
    from adjustText import adjust_text as _adjust_text
except Exception:                       # pragma: no cover
    _adjust_text = None

_PUB_RC = {
    "figure.dpi": 110,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "sans-serif",
    # Arial-first stack (omicverse look); silently falls back to DejaVu Sans.
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans",
                        "Bitstream Vera Sans", "sans-serif"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.edgecolor": "#222222",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "axes.grid": False,       # grids are opt-in per-axis (bars only)
    "grid.color": "0.85",
    "grid.linewidth": 0.6,
    "grid.alpha": 0.7,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "svg.fonttype": "none",   # keep text editable in Illustrator/Inkscape
    "pdf.fonttype": 42,       # embed TrueType, not Type-3
    "ps.fonttype": 42,
    "legend.frameon": False,
    "legend.handlelength": 1.4,
    "legend.handletextpad": 0.4,
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
    """Map each subgenome label to a stable muted colour (by position)."""
    return {sg: _SUBG_PALETTE[i % len(_SUBG_PALETTE)]
            for i, sg in enumerate(subg)}


def _offset_spines(ax, outward: int = 10) -> None:
    """omicverse-style floating spines: drop top/right, push left/bottom out.

    Only offsets spines that are still visible, so panels that have already
    hidden a spine are left untouched.
    """
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        if ax.spines[side].get_visible():
            ax.spines[side].set_position(("outward", outward))
    ax.tick_params(direction="out")


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

        # left: absolute variance bars (faint y-grid behind, omicverse style)
        axv.set_axisbelow(True)
        axv.grid(axis="y", zorder=0)
        xs = np.arange(len(comps))
        vals = [float(sigma2.get(c, 0.0)) for c in comps]
        bars = axv.bar(xs, vals, color=colors, edgecolor="white", linewidth=0.6,
                       alpha=0.92, zorder=3)
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
                    edgecolor="white", linewidth=0.6, alpha=0.92,
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

        for ax in (axv, axp):
            _offset_spines(ax)

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
            ax.scatter(x, logp, s=5, c=colors, linewidths=0, alpha=0.65,
                       rasterized=True)
            ax.axhline(gw_y, ls="-", lw=1.4, color=_GW_LINE)
            ax.axhline(sugg_y, ls="--", lw=1.1, color=_SUGG_LINE)
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
                bbox = dict(boxstyle="round,pad=0.15", fc="white",
                            ec="none", alpha=0.6)
                texts = []
                for i in idx:
                    if logp[i] < sugg_y:
                        break
                    if any(abs(x[i] - sx) < 0.02 * (x.max() + 1)
                           for sx in seen_x):
                        continue
                    seen_x.append(x[i])
                    texts.append(ax.annotate(
                        f"{chrom_arr[i]}:{int(pos_arr[i])}", (x[i], logp[i]),
                        fontsize=6, rotation=45, ha="left", va="bottom",
                        xytext=(2, 2), textcoords="offset points",
                        color="#222222", bbox=bbox))
                # de-overlap labels when adjustText is available; otherwise the
                # manual offsets + white bbox above already keep them readable.
                if _adjust_text is not None and texts:   # pragma: no cover
                    try:
                        _adjust_text(texts, ax=ax, only_move={"text": "y"},
                                     arrowprops=dict(arrowstyle="->",
                                                     color="0.6", lw=0.5))
                    except Exception:
                        pass
        # shared y-range; each visible panel keeps its own chrom ticks
        for ax, (centers, labels) in zip(axes, panel_ticks, strict=False):
            if not ax.get_visible():
                continue
            ax.set_ylim(0, ymax * 1.12)
            ax.set_xticks(centers)
            ax.set_xticklabels(labels, rotation=90, fontsize=6)
            _offset_spines(ax)
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
                                color="#9A9A9A", alpha=0.18, lw=0,
                                label="95% null band (all)")
            except Exception:
                pass

        lim = (float(max(exp_all.max(), logp_all.max())) * 1.05
               if logp_all.size else 1.0)
        ax.plot([0, lim], [0, lim], ls="--", color="0.25", lw=1.1)

        lam_all = lambda_by_level.get("all")
        lab_all = ("all" + (f" (λ={lam_all:.3f})" if lam_all is not None
                            else ""))
        ax.scatter(exp_all, logp_all, s=6, color="#333333", label=lab_all,
                   alpha=0.7, linewidths=0, rasterized=True)
        for sg in present:
            sub = df[df["subgenome"].astype(str) == sg]
            ps = np.clip(sub["p"].to_numpy(dtype=np.float64), 1e-300, 1.0)
            lp = np.sort(-np.log10(ps))[::-1]
            n_sg = _count(sg, lp.size)
            ranks = _qq_ranks(lp, n_sg)
            ex = -np.log10((ranks - 0.5) / n_sg)
            lam = lambda_by_level.get(sg)
            lab = sg + (f" (λ={lam:.3f})" if lam is not None else "")
            ax.scatter(ex, lp, s=6, color=palette.get(sg, "#1F577B"),
                       label=lab, alpha=0.7, linewidths=0, rasterized=True)

        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_xlabel(r"expected $-\log_{10}p$")
        ax.set_ylabel(r"observed $-\log_{10}p$")
        if trait:
            ax.set_title(trait)
        _offset_spines(ax)
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
        ax.set_axisbelow(True)
        ax.grid(axis="y", zorder=0)
        xs = np.arange(len(rows))
        colors = ["#333333" if lv == "all" else palette.get(lv, "#B8B8B8")
                  for lv in rows["level"]]
        vals = rows["lambda_gc"].to_numpy(dtype=np.float64)
        bars = ax.bar(xs, vals, color=colors, edgecolor="white", linewidth=0.6,
                      alpha=0.92, zorder=3)
        ax.axhline(1.0, ls="--", lw=1.3, color="0.25", zorder=2)
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
        _offset_spines(ax)
        fig.tight_layout()
        paths = save_figure(fig, out_dir, f"lambda_gc_{prefix}", formats, dpi)
        plt.close(fig)
    return paths


# ----------------------------------------------------------------------
# figure 5: LocusZoom-style regional plot (per-hit zoom)
# ----------------------------------------------------------------------

# classic LocusZoom 5-bin LD-to-lead scheme, omicverse hex palette
_LD_BINS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
_LD_COLORS = ["#1f2f86", "#6fc2ec", "#5cba5c", "#f4a23b", "#e23b30"]
_LD_LABELS = ["0.0–0.2", "0.2–0.4", "0.4–0.6",
              "0.6–0.8", "0.8–1.0"]
_LD_NA_COLOR = "#9b9b9b"
_LEAD_COLOR = "#7b3fa0"       # lead-SNP purple diamond
_LOCUS_FLAT = "#3b6fb6"       # flat colour when no LD available
_GENE_FILL = "#2f6db4"
_GENE_EDGE = "#1b3f6e"


def _ld_bin_color(r2: float) -> str:
    """omicverse 5-bin colour for an r^2-to-lead value (NaN -> grey)."""
    if r2 is None or not math.isfinite(r2):
        return _LD_NA_COLOR
    for i in range(len(_LD_COLORS)):
        if r2 <= _LD_BINS[i + 1]:
            return _LD_COLORS[i]
    return _LD_COLORS[-1]


def _compute_ld_to_lead(genotype_prefix: str | Path,
                        snp_ids: Sequence[str],
                        lead_snp: str) -> dict[str, float]:
    """r^2 between each window SNP and the lead, read from a PLINK1 bed.

    Only the window columns are read (via bed_reader column indexing), so this
    is safe even on whole-genome panels. Returns ``{snp_id: r2}`` with the lead
    forced to 1.0; SNPs absent from the bed or with <3 shared non-missing
    samples get NaN. Returns ``{}`` (caller falls back) if the lead is not in
    the bed or the bed is unreadable.
    """
    from bed_reader import open_bed

    prefix = Path(genotype_prefix)
    bed_path = prefix.with_suffix(".bed")
    if not bed_path.exists():
        raise FileNotFoundError(f"no bed for LD: {bed_path}")
    want = {str(s) for s in snp_ids}
    want.add(str(lead_snp))
    with open_bed(str(bed_path), count_A1=True) as bed:
        sid = np.asarray(bed.sid, dtype=object).astype(str)
        sel = np.array([i for i, s in enumerate(sid) if s in want],
                       dtype=np.int64)
        if sel.size == 0:
            return {}
        sel_ids = sid[sel]
        if str(lead_snp) not in set(sel_ids):
            return {}
        g = bed.read(index=(np.s_[:], sel), dtype="float32")  # (n, k)
    lead_j = int(np.nonzero(sel_ids == str(lead_snp))[0][0])
    lead_col = g[:, lead_j]
    out: dict[str, float] = {}
    for j, sname in enumerate(sel_ids):
        x = g[:, j]
        mask = np.isfinite(x) & np.isfinite(lead_col)
        if int(mask.sum()) < 3:
            out[sname] = float("nan")
            continue
        xv, lv = x[mask], lead_col[mask]
        if xv.std() == 0.0 or lv.std() == 0.0:
            out[sname] = float("nan")
            continue
        r = float(np.corrcoef(xv, lv)[0, 1])
        out[sname] = r * r if math.isfinite(r) else float("nan")
    out[str(lead_snp)] = 1.0
    return out


def _parse_genes(genes) -> pd.DataFrame:
    """Normalise a gene source to columns ``gene/chrom/start/end/strand``.

    Accepts a DataFrame, a ``.tsv/.csv`` gene table, or a ``.gff/.gff3/.gtf``
    (optionally gzipped) from which only ``gene`` features are taken. Unknown
    strands default to ``+``. Returns an empty frame on any parse failure.
    """
    cols = ["gene", "chrom", "start", "end", "strand"]
    if genes is None:
        return pd.DataFrame(columns=cols)
    if isinstance(genes, pd.DataFrame):
        g = genes.copy()
        if "strand" not in g.columns:
            g["strand"] = "+"
        missing = [c for c in ("gene", "chrom", "start", "end") if
                   c not in g.columns]
        if missing:
            raise ValueError(f"gene table missing columns: {missing}")
        return g[cols].copy()

    path = Path(genes)
    suf = "".join(path.suffixes).lower()
    if suf.endswith((".tsv", ".csv")) or suf.endswith((".tsv.gz", ".csv.gz")):
        sep = "," if ".csv" in suf else "\t"
        g = pd.read_csv(path, sep=sep)
        return _parse_genes(g)

    # GFF/GTF: take gene features, pull a name from the attributes column
    import gzip
    import re

    opener = gzip.open if path.suffix == ".gz" else open
    rows: list[dict] = []

    def _gene_name(attrs: str, fallback: str) -> str:
        # prefer human-readable keys over the bare identifier, by priority
        for key in ("gene_name", "Name", "gene_id", "ID"):
            m = re.search(rf'{key}[ =]"?([^";]+)"?', attrs)
            if m:
                return m.group(1)
        return fallback

    with opener(path, "rt") as fh:
        for line in fh:
            if not line or line[0] == "#":
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9 or f[2] != "gene":
                continue
            rows.append({"gene": _gene_name(f[8], f"{f[0]}:{f[3]}"),
                         "chrom": f[0], "start": int(f[3]), "end": int(f[4]),
                         "strand": f[6] if f[6] in ("+", "-") else "+"})
    return pd.DataFrame(rows, columns=cols)


def _draw_gene_track(ax, genes_df: pd.DataFrame, x0_mb: float,
                     x1_mb: float) -> None:
    """Draw genes as packed strand-aware boxes (x in Mb) on ``ax``."""
    from matplotlib.patches import Rectangle

    rows_end: list[float] = []          # right edge (Mb) of last gene per row
    for _, gr in genes_df.sort_values("start").iterrows():
        s = float(gr["start"]) / 1e6
        e = float(gr["end"]) / 1e6
        pad = (x1_mb - x0_mb) * 0.02 + 1e-9
        row = next((r for r, re_ in enumerate(rows_end) if s > re_ + pad),
                   len(rows_end))
        if row == len(rows_end):
            rows_end.append(e)
        else:
            rows_end[row] = e
        y = -row
        ax.add_patch(Rectangle((s, y - 0.18), max(e - s, 1e-6), 0.36,
                               facecolor=_GENE_FILL, edgecolor=_GENE_EDGE,
                               linewidth=0.5, zorder=3))
        strand = str(gr["strand"])
        amark = ">" if strand == "+" else "<"
        ax.plot([e if strand == "+" else s], [y], marker=amark, ms=4,
                color=_GENE_EDGE, zorder=4)
        ax.text((s + e) / 2.0, y + 0.30, str(gr["gene"]), ha="center",
                va="bottom", fontsize=6.0, style="italic")
    ax.set_ylim(-(max(len(rows_end), 1)) + 0.4, 0.6)
    ax.set_yticks([])
    ax.set_ylabel("genes")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)


def plot_locus(df: pd.DataFrame,
               *,
               out_dir: Path,
               prefix: str,
               lead_snp: str | None = None,
               chrom: str | None = None,
               pos: int | None = None,
               subgenome: str | None = None,
               window_kb: int = 500,
               ld: Mapping[str, float] | None = None,
               genotype_prefix: str | Path | None = None,
               genes=None,
               trait: str = "",
               genomewide: float = 5e-8,
               suggestive: float = 1e-5,
               formats: Sequence[str] = DEFAULT_FORMATS,
               dpi: int = 300) -> list[str]:
    """LocusZoom-style regional plot around one hit.

    The window is centred on ``lead_snp`` (by id), else on ``chrom``/``pos``,
    else on the lowest-p marker (optionally within ``subgenome``), and spans
    ``+/- window_kb``. Points are coloured by r^2 to the lead when LD is
    available (``ld`` mapping, or computed from ``genotype_prefix``), otherwise
    flat with an explicit "LD unavailable" note. An optional gene-model track
    (``genes`` = DataFrame / TSV / GFF) is drawn below. Returns written paths.
    """
    plt, mpl = _import_mpl()
    if df is None or len(df) == 0:
        return []
    d = df.copy()
    d["chrom"] = d["chrom"].astype(str)
    d["snp_id"] = d["snp_id"].astype(str)
    has_sub = "subgenome" in d.columns
    if has_sub:
        d["subgenome"] = d["subgenome"].astype(str)

    # ---- choose the centre + lead ----
    if subgenome is not None and has_sub:
        d_sub = d[d["subgenome"] == str(subgenome)]
    else:
        d_sub = d
    if lead_snp is not None:
        row = d_sub[d_sub["snp_id"] == str(lead_snp)]
        if len(row) == 0:
            raise ValueError(f"lead_snp {lead_snp!r} not in sumstats")
        row = row.iloc[0]
        c_chrom, c_pos = str(row["chrom"]), int(row["pos"])
        if subgenome is None and has_sub:
            subgenome = str(row["subgenome"])
    elif chrom is not None and pos is not None:
        c_chrom, c_pos = str(chrom), int(pos)
    else:
        if len(d_sub) == 0:
            raise ValueError("no markers to pick a lead from")
        row = d_sub.loc[d_sub["p"].astype(float).idxmin()]
        c_chrom, c_pos = str(row["chrom"]), int(row["pos"])
        if subgenome is None and has_sub:
            subgenome = str(row["subgenome"])

    half = int(window_kb) * 1000
    win = d[(d["chrom"] == c_chrom)
            & (d["pos"].astype(np.int64) >= c_pos - half)
            & (d["pos"].astype(np.int64) <= c_pos + half)]
    if subgenome is not None and has_sub:
        win = win[win["subgenome"] == str(subgenome)]
    # drop non-finite p so -log10/ylim cannot become NaN
    win = win[np.isfinite(win["p"].astype(np.float64).to_numpy())]
    if len(win) == 0:
        raise ValueError(
            f"no markers in {c_chrom}:{c_pos}+/-{window_kb}kb")
    win = win.reset_index(drop=True)
    if lead_snp is None:
        lead_snp = str(win.loc[win["p"].astype(float).idxmin(), "snp_id"])

    # ---- LD to lead ----
    if ld is None and genotype_prefix is not None:
        import warnings
        try:
            ld = _compute_ld_to_lead(genotype_prefix,
                                     win["snp_id"].tolist(), lead_snp)
            if not ld:
                warnings.warn("LD: lead not found in genotypes; "
                              "drawing without r^2 colouring", stacklevel=2)
        except Exception as exc:                       # pragma: no cover
            warnings.warn(f"LD computation failed ({exc}); "
                          "drawing without r^2 colouring", stacklevel=2)
            ld = {}
    have_ld = bool(ld)

    genes_df = _parse_genes(genes)
    if len(genes_df):
        genes_df = genes_df[(genes_df["chrom"].astype(str) == c_chrom)
                            & (genes_df["end"].astype(np.int64)
                               >= c_pos - half)
                            & (genes_df["start"].astype(np.int64)
                               <= c_pos + half)]
    draw_genes = len(genes_df) > 0

    x = win["pos"].to_numpy(dtype=np.float64) / 1e6
    p = np.clip(win["p"].to_numpy(dtype=np.float64), 1e-300, 1.0)
    logp = -np.log10(p)
    snp_arr = win["snp_id"].to_numpy()
    if have_ld:
        colors = [_ld_bin_color(float(ld.get(s, float("nan"))))
                  for s in snp_arr]
    else:
        colors = [_LOCUS_FLAT] * len(snp_arr)
    lead_mask = snp_arr == str(lead_snp)

    with plt.rc_context(_PUB_RC):
        if draw_genes:
            fig, (ax, axg) = plt.subplots(
                2, 1, figsize=(9.0, 6.2), sharex=True,
                gridspec_kw={"height_ratios": [3.2, 1.0], "hspace": 0.12})
        else:
            fig, ax = plt.subplots(figsize=(9.0, 4.4))
            axg = None

        ax.scatter(x, logp, s=26, c=colors, edgecolors="black",
                   linewidths=0.35, alpha=0.9, zorder=3, rasterized=True)
        ax.axhline(-math.log10(genomewide), ls="-", lw=1.2, color=_GW_LINE,
                   zorder=2)
        ax.axhline(-math.log10(suggestive), ls="--", lw=1.0, color=_SUGG_LINE,
                   zorder=2)
        if lead_mask.any():
            ax.scatter(x[lead_mask], logp[lead_mask], marker="D", s=95,
                       c=_LEAD_COLOR, edgecolors="black", linewidths=0.6,
                       zorder=6)
            ax.annotate(str(lead_snp), (x[lead_mask][0], logp[lead_mask][0]),
                        fontsize=7, fontweight="bold", color="#4a2069",
                        xytext=(8, 6), textcoords="offset points")

        # r^2 legend (or LD-unavailable note)
        from matplotlib.patches import Patch
        if have_ld:
            handles = [Patch(facecolor=_LD_COLORS[i], edgecolor="black",
                             linewidth=0.3, label=_LD_LABELS[i])
                       for i in range(len(_LD_COLORS))][::-1]
            ax.legend(handles=handles, title=r"$r^2$", loc="upper left",
                      fontsize=6, title_fontsize=7)
        else:
            ax.legend(handles=[Patch(facecolor=_LOCUS_FLAT,
                                     label="LD unavailable")],
                      loc="upper left", fontsize=6)

        ax.set_ylim(0, max(float(logp.max()), 1.0) * 1.12)
        # force the x-range to the requested window so sparse loci read true
        ax.set_xlim((c_pos - half) / 1e6, (c_pos + half) / 1e6)
        ax.set_ylabel(r"$-\log_{10}p$")
        ttl = f"{trait} — {c_chrom}:{c_pos:,}" if trait else f"{c_chrom}:{c_pos:,}"
        ax.set_title(ttl)
        _offset_spines(ax)

        if draw_genes:
            _draw_gene_track(axg, genes_df, float(x.min()), float(x.max()))
            axg.set_xlabel(f"{c_chrom} position (Mb)")
            axg.xaxis.set_major_formatter(
                mpl.ticker.FuncFormatter(lambda v, _: f"{v:.2f}"))
        else:
            ax.set_xlabel(f"{c_chrom} position (Mb)")
            ax.xaxis.set_major_formatter(
                mpl.ticker.FuncFormatter(lambda v, _: f"{v:.2f}"))

        # gene track's custom spines/ylim are not tight_layout-compatible;
        # the gridspec hspace + savefig bbox="tight" handle spacing there.
        if not draw_genes:
            fig.tight_layout()
        safe = f"{c_chrom}_{c_pos}".replace(":", "_").replace("/", "_")
        paths = save_figure(fig, out_dir, f"locus_{prefix}_{safe}",
                            formats, dpi)
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


# ----------------------------------------------------------------------
# locus plots from a finished run (streamed, bounded memory)
# ----------------------------------------------------------------------

def _stream_pick_leads(sumstats_paths: Sequence[str], top_n: int,
                       subgenome: str | None = None,
                       chunksize: int = 500_000) -> list[dict]:
    """Stream sumstats and return up to ``top_n`` lowest-p, >1 Mb-apart leads.

    Memory is bounded by retaining only a buffer of the strongest candidates
    (``max(top_n*50, 500)`` rows) across the stream, then applying the 1 Mb
    spacing pass. The buffer is large enough that clustered top SNPs do not
    crowd out the next independent loci in practice; for a guaranteed-complete
    clumping use the full-sumstats path. Returns fewer than ``top_n`` leads if
    the buffer holds fewer independent loci.
    """
    cols = ["snp_id", "subgenome", "chrom", "pos", "p"]
    keep_n = max(top_n * 50, 500)
    best: pd.DataFrame | None = None
    for path in sumstats_paths:
        for ch in pd.read_csv(path, sep="\t",
                              usecols=lambda c: c in cols, chunksize=chunksize):
            pv = ch["p"].to_numpy(dtype=np.float64)
            ch = ch[np.isfinite(pv)]
            if subgenome is not None and "subgenome" in ch.columns:
                ch = ch[ch["subgenome"].astype(str) == str(subgenome)]
            if len(ch) == 0:
                continue
            best = ch if best is None else pd.concat([best, ch],
                                                     ignore_index=True)
            best = best.nsmallest(keep_n, "p")
    if best is None:
        return []
    best = best.sort_values("p").reset_index(drop=True)
    leads: list[dict] = []
    chosen: list[tuple[str, int]] = []
    for _, r in best.iterrows():
        if len(leads) >= top_n:
            break
        c, pp = str(r["chrom"]), int(r["pos"])
        if any(cc == c and abs(pp - pq) < 1_000_000 for cc, pq in chosen):
            continue
        chosen.append((c, pp))
        leads.append({"lead_snp": str(r["snp_id"]), "chrom": c, "pos": pp,
                      "subgenome": (str(r["subgenome"])
                                    if "subgenome" in best.columns
                                    else subgenome)})
    return leads


def _stream_find_snp(sumstats_paths: Sequence[str], snp_id: str,
                     chunksize: int = 500_000) -> dict | None:
    """Stream sumstats to locate one SNP's chrom/pos/subgenome by id."""
    for path in sumstats_paths:
        for ch in pd.read_csv(path, sep="\t", chunksize=chunksize):
            hit = ch[ch["snp_id"].astype(str) == str(snp_id)]
            if len(hit):
                r = hit.iloc[0]
                return {"lead_snp": str(snp_id), "chrom": str(r["chrom"]),
                        "pos": int(r["pos"]),
                        "subgenome": (str(r["subgenome"])
                                      if "subgenome" in ch.columns else None)}
    return None


def _stream_window(sumstats_paths: Sequence[str], chrom: str, center: int,
                   half: int, subgenome: str | None = None,
                   chunksize: int = 500_000) -> pd.DataFrame:
    """Stream sumstats and collect all rows within one locus window."""
    keep: list[pd.DataFrame] = []
    for path in sumstats_paths:
        for ch in pd.read_csv(path, sep="\t", chunksize=chunksize):
            sub = ch[ch["chrom"].astype(str) == str(chrom)]
            if len(sub) == 0:
                continue
            sp = sub["pos"].to_numpy(dtype=np.int64)
            sub = sub[(sp >= center - half) & (sp <= center + half)]
            if subgenome is not None and "subgenome" in sub.columns:
                sub = sub[sub["subgenome"].astype(str) == str(subgenome)]
            if len(sub):
                keep.append(sub)
    return (pd.concat(keep, ignore_index=True) if keep
            else pd.DataFrame(columns=["snp_id", "subgenome", "chrom",
                                       "pos", "p"]))


def plot_loci_from_results(results_dir: Path,
                           *,
                           prefix: str | None = None,
                           top_n: int = 3,
                           window_kb: int = 500,
                           lead_snp: str | None = None,
                           chrom: str | None = None,
                           pos: int | None = None,
                           subgenome: str | None = None,
                           genotype_template: str | None = None,
                           genotype: str | None = None,
                           genes=None,
                           formats: Sequence[str] = DEFAULT_FORMATS,
                           dpi: int = 300,
                           out_dir: Path | None = None) -> list[str]:
    """Emit LocusZoom-style plots for a finished run (no refit).

    Picks loci from ``--lead-snp`` / ``--chrom,--pos`` / the top ``top_n``
    genome-wide hits, streams each window out of the sumstats, resolves the
    per-subgenome genotype bed (``genotype`` override or ``genotype_template``
    with ``{subgenome}``) for r^2 colouring, and draws an optional gene track.
    """
    import json

    results_dir = Path(results_dir)
    summary_path = _discover_summary(results_dir, prefix)
    summary = json.loads(summary_path.read_text())
    run_prefix = summary_path.stem[len("summary_"):]
    trait = summary.get("trait", "")
    sumstats = _resolve_sumstats(
        results_dir, summary.get("outputs", {}).get("sumstats", []))
    out_dir = Path(out_dir) if out_dir is not None else results_dir
    half = int(window_kb) * 1000

    if lead_snp is None and (chrom is None) != (pos is None):
        raise ValueError("pass both chrom and pos (or neither)")

    if lead_snp is not None and (chrom is None or pos is None):
        found = _stream_find_snp(sumstats, lead_snp)
        if found is None:
            raise ValueError(f"lead_snp {lead_snp!r} not found in sumstats")
        leads = [found]
    elif lead_snp is not None or (chrom is not None and pos is not None):
        leads = [{"lead_snp": lead_snp, "chrom": str(chrom),
                  "pos": int(pos), "subgenome": subgenome}]
    else:
        leads = _stream_pick_leads(sumstats, top_n, subgenome=subgenome)

    paths: list[str] = []
    for lead in leads:
        c, p_ = lead["chrom"], lead["pos"]
        sg = lead.get("subgenome", subgenome)
        win = _stream_window(sumstats, str(c), int(p_), half, subgenome=sg)
        if len(win) == 0:
            continue
        gp = genotype
        if gp is None and genotype_template and sg:
            cand = genotype_template.format(subgenome=sg)
            if Path(cand).with_suffix(".bed").exists():
                gp = cand
        paths += plot_locus(
            win, out_dir=out_dir, prefix=run_prefix,
            lead_snp=lead.get("lead_snp"), chrom=str(c), pos=int(p_),
            subgenome=sg, window_kb=window_kb, genotype_prefix=gp,
            genes=genes, trait=trait, formats=formats, dpi=dpi)
    return paths
