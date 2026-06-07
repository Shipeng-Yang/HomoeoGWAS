"""H7 main figure (Fig 7) — the marker-density -> callable-pairs -> discovery chain.

Integrates the two H7 analyses into one figure:
  Part A (results/phase7/h7_marker_density.json):   density -> callable homoeolog pairs G (+ 4-species anchors)
  Part B (results/phase7/h7_partB_power_vs_density.json): density -> conditional power / FWER / discovery

Layout (2x2, SNP retention fraction increasing left->right on a shared log x-axis;
the left edge is the sparse/GBS-like regime, the right edge is full WGS density):
  A  density -> callable pairs G (uniform vs clustered thinning + feasibility bands) | right inset = 4 real-species anchors
  B  density -> conditional per-pair ACAT power vs global K_hom kernel  (PVE=0.10 benchmark; optional PVE=0.03 overlay)
  C  density -> empirical FWER (controlled near nominal 0.05)
  D  density -> realized discovery: coverage-adjusted discovery COLLAPSES while conditional per-pair power stays flat

Run:  python scripts/phase7/h7_fig_density_callable_power.py [out_dir]
Default out_dir = results/phase7/figures/ ; pass /tmp/... for a dry render.

FRAMING LOCKS (do not violate in any caption/heading/annotation; memory h7_marker_density_tool + h3_h4_acat_hardening):
  - Panel B/D power is CONDITIONAL on surviving (callable) pairs and is a HIGH-EFFECT PVE=0.10 benchmark; label it so.
  - per-pair > global K_hom holds in THIS per-pair-separating regime only; NOT a universal claim.
  - FWER is "controlled near nominal", never "exactly 0.05".
  - Feasibility bands are empirical design categories, not formal decision thresholds.
  - The conclusion is COVERAGE-limited discovery: 0.995 -> 0.467 -> 0.112, not a failure of the test.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results" / "phase7"
OUT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else (RES / "figures")

# ---------------------------------------------------------------- Okabe-Ito (colourblind-safe)
C_UNIFORM = "#0072B2"   # blue   — optimistic uniform thinning
C_CLUSTER = "#D55E00"   # vermil — pessimistic clustered thinning
C_PERPAIR = "#009E73"   # green  — per-pair ACAT test
C_KHOM = "#CC79A7"      # pink   — global K_hom kernel
C_DISCOV = "#222222"    # black  — coverage-adjusted discovery
C_DISC_OK = "#009E73"   # discovery outcome (filled)
C_DISC_NULL = "#999999" # null outcome (open)
BAND_GREYS = ["#efefef", "#e3e3e3", "#d6d6d6", "#c9c9c9"]  # null -> strong, low-tint

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8,
    "legend.fontsize": 6.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
})


def load(name):
    return json.loads((RES / name).read_text())


def partb_series(d):
    """Pull aligned arrays from a Part-B by_density block, sorted by descending density."""
    items = sorted(d["by_density"].values(), key=lambda v: -v["density"])
    dens = [v["density"] for v in items]
    return dict(
        density=dens,
        G=[v["G_callable"] for v in items],
        power_pp=[v["power_perpair"] for v in items],
        power_kh=[v["power_khom"] for v in items],
        fwer_pp=[v["fwer_perpair"] for v in items],
        fwer_kh=[v["fwer_khom"] for v in items],
        discovery=[v.get("coverage_adjusted_discovery") for v in items],  # primary PVE only
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    a = load("h7_marker_density.json")
    b = partb_series(load("h7_partB_power_vs_density.json"))
    rel_err = a["heldout_median_rel_error_min_snp3"]

    # optional weaker-signal overlay
    pve030 = RES / "h7_partB_power_vs_density_pve030.json"
    b2 = partb_series(load(pve030.name)) if pve030.exists() else None

    curve = a["thinning_curve"]["min_snp=3"]

    def flr(xs):  # log-safe floor (below ylim, off-view) so fill_between/plot never hit <=0
        return [max(float(x), 0.3) for x in xs]

    frac = [c["fraction"] for c in curve]
    g_uni = flr([c["G_uniform"] for c in curve])
    g_uni_lo = flr([c["G_uniform_ci"][0] for c in curve])
    g_uni_hi = flr([c["G_uniform_ci"][1] for c in curve])
    g_clu = flr([c["G_clustered"] for c in curve])
    g_clu_lo = flr([c["G_clustered_ci"][0] for c in curve])
    g_clu_hi = flr([c["G_clustered_ci"][1] for c in curve])
    bands = a["feasibility_bands"]
    anchors = a["species_anchors"]

    XLIM = (0.0085, 1.18)
    GTOP = 4500   # headroom so the top (wheat) anchor label clears the frame

    fig = plt.figure(figsize=(7.1, 5.7))
    outer = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.30,
                              height_ratios=[1, 1], width_ratios=[1, 1])
    # split top-left into curves + narrow anchor strip
    tl = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0, 0],
                                          width_ratios=[3.0, 1.05], wspace=0.08)
    axA = fig.add_subplot(tl[0, 0])
    axAn = fig.add_subplot(tl[0, 1], sharey=axA)
    axB = fig.add_subplot(outer[0, 1])
    axC = fig.add_subplot(outer[1, 0])
    axD = fig.add_subplot(outer[1, 1])

    # ---- shared feasibility bands on the G (log) axis, drawn in A and the anchor strip
    def draw_bands(ax, label=False):
        for i, bd in enumerate(bands):
            lo = max(bd["min_G"], 1)
            hi = bd["max_G"] if bd["max_G"] is not None else GTOP
            ax.axhspan(lo, hi, color=BAND_GREYS[i], zorder=0, lw=0)
            if label:
                short = {"underpowered/null-likely": "null", "borderline/risky": "borderline",
                         "discovery-feasible": "feasible", "strong-design": "strong"}[bd["label"]]
                ax.text(XLIM[1] * 0.97, (lo * hi) ** 0.5, short, ha="right", va="center",
                        fontsize=5.6, color="#555555", style="italic", zorder=1)

    # ===== Panel A: density -> callable pairs G
    draw_bands(axA, label=True)
    axA.fill_between(frac, g_uni_lo, g_uni_hi, color=C_UNIFORM, alpha=0.18, lw=0, zorder=2)
    axA.fill_between(frac, g_clu_lo, g_clu_hi, color=C_CLUSTER, alpha=0.18, lw=0, zorder=2)
    axA.plot(frac, g_uni, "-o", color=C_UNIFORM, ms=3, lw=1.4, label="uniform (optimistic)", zorder=4)
    axA.plot(frac, g_clu, "--s", color=C_CLUSTER, ms=3, lw=1.4, label="clustered (pessimistic)", zorder=4)
    axA.set_xscale("log")
    axA.set_yscale("log")
    axA.set_xlim(*XLIM)
    axA.set_ylim(0.8, GTOP)
    axA.set_xlabel("SNP retention fraction")
    axA.set_ylabel("callable homoeolog pairs $G$")
    axA.set_title("A   density → callable pairs", loc="left", fontweight="bold")
    axA.legend(loc="lower right", frameon=False, handlelength=1.6)
    axA.text(0.02, 0.04, f"empirical design bands\nheld-out rel-error {rel_err*100:.1f}%",
             transform=axA.transAxes, fontsize=5.8, color="#333333", va="bottom")

    # ===== anchor strip: 4 real species on the same G axis
    draw_bands(axAn)
    axAn.set_yscale("log")
    xs = list(range(len(anchors)))
    for x, an in zip(xs, anchors, strict=True):
        is_disc = an["outcome"].startswith("discovery")
        axAn.scatter([x], [an["G_callable"]], s=42, zorder=4,
                     marker="o" if is_disc else "o",
                     facecolor=C_DISC_OK if is_disc else "white",
                     edgecolor=C_DISC_OK if is_disc else C_DISC_NULL, linewidth=1.3)
        sp = an["species"].split()[0]
        axAn.annotate(f"{sp}\n{an['tech'].split('(')[0].strip()}", (x, an["G_callable"]),
                      textcoords="offset points", xytext=(0, 7 if is_disc else -13),
                      ha="center", fontsize=5.0, color="#333333")
    axAn.set_xlim(-0.7, len(anchors) - 0.3)
    axAn.set_xticks([])
    axAn.tick_params(labelleft=False)
    axAn.set_title("validation", loc="center", fontsize=7)
    # tiny legend for filled/open
    axAn.scatter([], [], s=40, marker="o", facecolor=C_DISC_OK, edgecolor=C_DISC_OK, label="discovery")
    axAn.scatter([], [], s=40, marker="o", facecolor="white", edgecolor=C_DISC_NULL, label="null")
    axAn.legend(loc="lower left", frameon=False, fontsize=5.4, handletextpad=0.2,
                borderpad=0.1, bbox_to_anchor=(-0.05, -0.02))

    # ===== Panel B: density -> conditional power
    axB.plot(b["density"], b["power_pp"], "-o", color=C_PERPAIR, ms=4, lw=1.6,
             label="per-pair ACAT", zorder=4)
    axB.plot(b["density"], b["power_kh"], "-s", color=C_KHOM, ms=4, lw=1.6,
             label="global $K_{hom}$", zorder=4)
    if b2:
        axB.plot(b2["density"], b2["power_pp"], "--o", color=C_PERPAIR, ms=3, lw=1.1,
                 alpha=0.6, label="per-pair (PVE 0.03)", zorder=3)
        axB.plot(b2["density"], b2["power_kh"], "--s", color=C_KHOM, ms=3, lw=1.1,
                 alpha=0.6, label="$K_{hom}$ (PVE 0.03)", zorder=3)
    axB.set_xscale("log")
    axB.set_xlim(*XLIM)
    axB.set_ylim(0, 1.05)
    axB.set_xlabel("SNP retention fraction")
    axB.set_ylabel("conditional power")
    axB.set_title("B   density → conditional power", loc="left", fontweight="bold")
    axB.legend(loc="center right", frameon=False, handlelength=1.6)
    bnote = ("solid PVE=0.10 (high-effect)\ndashed PVE=0.03 (weak); surviving pairs"
             if b2 else "high-effect PVE=0.10 benchmark\n(surviving pairs only)")
    axB.text(0.02, 0.04, bnote, transform=axB.transAxes, fontsize=5.8, color="#333333", va="bottom")

    # ===== Panel C: density -> FWER
    axC.axhline(0.05, color="#888888", lw=0.9, ls=":", zorder=1)
    axC.text(XLIM[0] * 1.15, 0.052, "nominal 0.05", fontsize=5.8, color="#666666", va="bottom")
    axC.plot(b["density"], b["fwer_pp"], "-o", color=C_PERPAIR, ms=4, lw=1.6, label="per-pair ACAT", zorder=4)
    axC.plot(b["density"], b["fwer_kh"], "-s", color=C_KHOM, ms=4, lw=1.6, label="global $K_{hom}$", zorder=4)
    axC.set_xscale("log")
    axC.set_xlim(*XLIM)
    axC.set_ylim(0, 0.1)
    axC.set_xlabel("SNP retention fraction")
    axC.set_ylabel("empirical FWER")
    axC.set_title("C   FWER controlled near nominal", loc="left", fontweight="bold")
    axC.legend(loc="upper right", frameon=False, handlelength=1.6)

    # ===== Panel D (punchline): discovery collapses while conditional power stays flat
    axD.plot(b["density"], b["power_pp"], "--o", color=C_PERPAIR, ms=4, lw=1.4,
             label="per-pair conditional power (PVE 0.10)", zorder=3)
    axD.plot(b["density"], b["discovery"], "-D", color=C_DISCOV, ms=4.5, lw=1.9,
             label="coverage-adjusted discovery", zorder=4)
    for x, y in zip(b["density"], b["discovery"], strict=True):
        axD.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(2, -10),
                     fontsize=6, color=C_DISCOV)
    axD.set_xscale("log")
    axD.set_xlim(*XLIM)
    axD.set_ylim(0, 1.05)
    axD.set_xlabel("SNP retention fraction")
    axD.set_ylabel("probability")
    axD.set_title("D   callable coverage limits realized discovery", loc="left", fontweight="bold")
    axD.legend(loc="lower right", frameon=False, handlelength=1.6)

    # ---- top chain annotation
    fig.text(0.5, 0.985,
             "marker density  →  callable-pair coverage  →  conditional power  →  realized discovery",
             ha="center", va="top", fontsize=7.5, color="#444444", style="italic")

    fig.subplots_adjust(top=0.93)
    for ext in ("pdf", "png"):
        p = OUT_DIR / f"h7_density_callable_power.{ext}"
        fig.savefig(p, dpi=300 if ext == "png" else None)
        print(f"wrote {p}")
    print(f"PVE=0.03 overlay: {'INCLUDED' if b2 else 'absent (PVE=0.10 only)'}")


if __name__ == "__main__":
    main()
