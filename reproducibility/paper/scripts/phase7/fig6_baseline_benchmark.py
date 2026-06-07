"""Fig 6 — homoeogwas interaction test vs baseline methods (matched candidate-pair benchmark).

Two panels, from the committed P2 benchmark:
  A  simulation power heatmap (results/phase7/baseline_benchmark.json):
       rows = 5 scenarios x 3 PVE (0.05/0.1/0.2), cols = 6 method families ORDERED BY ESTIMAND
       (homoeogwas, SNPxSNP, main-burden, single-gene, marginal, globalVC). Cell = power.
       The near-block-diagonal of bright cells = each method detects its OWN estimand.
  B  real-data recall (results/phase7/baseline_benchmark_realdata.json):
       two cotton hits; per-method -log10(P); ours is the only genome-wide detection.

Run:  python scripts/phase7/fig6_baseline_benchmark.py [out_dir]
Default out_dir = results/phase7/figures/.

FRAMING LOCKS (memory paper_software_first / results_retrospective_audit — honor in fig + caption):
  - Fill encodes power ONLY; ours is NOT given a brighter colour. ours/SNPxSNP get thin coloured
    OUTLINES + estimand tags so the panel reads as "different estimands", not "ours vs all".
  - ours wins ONLY the burden_product (target) estimand; it is NOT universally best.
  - SNPxSNP winning single_snp_pair is EXPECTED (its estimand); show it honestly.
  - mixed_sign ~0 for all = documented scope limit of any burden-product test.
  - additive_only: ours ~0 is the ANTI-CIRCULARITY result (does not turn main effects into epistasis);
    mainburden/singlegene winning here is expected (different estimand), NOT a loss for ours.
  - mispaired: ours = 0 = homoeolog-pair specificity (a feature).
  - globalVC = global screen, cannot localize; external GWAS omitted (different estimand), not because they lost.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results" / "phase7"
OUT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else (RES / "figures")

C_OURS = "#008C8C"     # teal   — targeted test outline / detection
C_SNPXSNP = "#7B61FF"  # purple — single-pair test outline
C_NEUTRAL = "#444444"

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

METHODS = ["ours", "snpxsnp", "mainburden", "singlegene", "marginal", "globalVC"]
MLABELS = ["homoeogwas\n(ours)", "SNP×SNP", "main\nburden", "single\ngene", "marginal\nGWAS", "global\nVC"]
SCEN = [
    ("burden_product", "Burden-product\ninteraction", "target estimand"),
    ("single_snp_pair", "Single SNP-pair\ninteraction", "off-model"),
    ("mixed_sign", "Mixed-sign\ninteraction", "scope limit"),
    ("additive_only", "Additive-only", "anti-circularity ctrl"),
    ("mispaired", "Mispaired\nhomoeolog", "specificity ctrl"),
]
PVES = ["0.05", "0.1", "0.2"]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bench = json.loads((RES / "baseline_benchmark.json").read_text())
    real = json.loads((RES / "baseline_benchmark_realdata.json").read_text())
    r = bench["results"]

    # power matrix: 15 rows (scenario x PVE) x 6 methods
    M = np.full((len(SCEN) * len(PVES), len(METHODS)), np.nan)
    row_labels = []
    for si, (scen, _, _) in enumerate(SCEN):
        for pi, p in enumerate(PVES):
            ri = si * len(PVES) + pi
            row_labels.append(p)
            cell = r[f"{scen}@pve{p}"]              # KeyError surfaces a schema typo (no silent blank)
            for mi, m in enumerate(METHODS):
                M[ri, mi] = cell[m]["power"]
    assert not np.isnan(M).any(), "blank power cell — check scenario/PVE/method keys"

    fig = plt.figure(figsize=(7.4, 6.3))
    outer = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.55, 1.0], wspace=0.34)
    axA = fig.add_subplot(outer[0, 0])
    rgs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[0, 1], hspace=0.42)
    axB1 = fig.add_subplot(rgs[0, 0])
    axB2 = fig.add_subplot(rgs[1, 0])

    # ===== Panel A: power heatmap
    im = axA.imshow(M, aspect="auto", cmap="cividis", vmin=0, vmax=1)
    axA.set_xticks(range(len(METHODS)))
    axA.set_xticklabels(MLABELS, fontsize=6.6)
    axA.set_yticks(range(len(row_labels)))
    axA.set_yticklabels(row_labels, fontsize=6.2)
    axA.tick_params(length=0)
    # annotate every cell (rigorous benchmark)
    for ri in range(M.shape[0]):
        for mi in range(M.shape[1]):
            v = M[ri, mi]
            if np.isnan(v):
                continue
            axA.text(mi, ri, f"{v:.2f}", ha="center", va="center", fontsize=5.3,
                     color="white" if v < 0.55 else "black")
    # scenario block separators + left group labels
    for si, (_, name, role) in enumerate(SCEN):
        y0 = si * 3 - 0.5
        if si > 0:
            axA.axhline(y0, color="white", lw=1.8)
        axA.text(-1.75, si * 3 + 1, name, ha="right", va="center", fontsize=6.4, fontweight="bold")
        axA.text(-1.75, si * 3 + 1.85, f"({role})", ha="right", va="center", fontsize=5.4,
                 style="italic", color="#666666")
    # estimand outlines for ours / snpxsnp + header tags
    axA.add_patch(Rectangle((-0.5, -0.5), 1, M.shape[0], fill=False, edgecolor=C_OURS,
                            lw=2.2, clip_on=False, zorder=5))
    axA.add_patch(Rectangle((0.5, -0.5), 1, M.shape[0], fill=False, edgecolor=C_SNPXSNP,
                            lw=2.0, clip_on=False, zorder=5))
    axA.text(0, -1.05, "targeted\ntest", ha="center", va="bottom", fontsize=5.8, color=C_OURS, fontweight="bold")
    axA.text(1, -1.05, "single-pair\ntest", ha="center", va="bottom", fontsize=5.8, color=C_SNPXSNP, fontweight="bold")
    axA.set_title("A   simulation power: each method detects its own estimand",
                  loc="left", fontweight="bold", pad=34)
    axA.text(-1.75, M.shape[0] - 0.2, "PVE per row", ha="right", va="top", fontsize=5.4, color="#888888")
    cb = fig.colorbar(im, ax=axA, fraction=0.045, pad=0.02)
    cb.set_label("power (FWER ≤ 0.05)", fontsize=6.5)
    cb.ax.tick_params(labelsize=6)

    # ===== Panel B: real-data recall, two cotton hits.
    # No binary "detected/not" fill (that would be circular): points are coloured by METHOD CLASS
    # (matching Panel A's estimand outlines), and the -log10 P magnitudes + reference lines tell
    # the story. ours' genome-wide candidate-pair significance is established in the cotton results.
    def mcolor(m):
        return C_OURS if m == "ours" else C_SNPXSNP if m == "snpxsnp" else C_NEUTRAL

    def forest(ax, hit, title):
        ys = list(range(len(METHODS)))[::-1]  # ours at top
        pmap = {
            "ours": hit["ours_interaction_p"],
            "snpxsnp": hit["snpxsnp_best_bonferroni_within_pair"],
            "mainburden": hit["mainburden_p"],
            "singlegene": hit["singlegene_min_p"],
            "marginal": hit["marginal_best_snp_p"],
            "globalVC": hit["globalVC_p"],
        }
        xs = {m: -np.log10(max(pmap[m], 1e-12)) for m in METHODS}
        for y, m in zip(ys, METHODS, strict=True):
            ax.scatter([xs[m]], [y], s=48, marker="D" if m == "globalVC" else "o",
                       facecolor=mcolor(m), edgecolor=mcolor(m), alpha=0.9, linewidth=1.0, zorder=4)
        ax.axvline(-np.log10(0.05), color="#999999", ls=":", lw=0.9, zorder=1)
        ax.text(-np.log10(0.05), len(METHODS) - 0.25, "nominal\nP=0.05", fontsize=5.0, color="#888888",
                ha="center", va="bottom")
        ax.set_yticks(ys)
        ax.set_yticklabels([lab.replace("\n", " ") for lab in MLABELS], fontsize=6.2)
        ax.set_xlim(-0.2, max(xs.values()) + 1.1)
        ax.set_xlabel("$-\\log_{10}P$", fontsize=7)
        ax.set_title(title, loc="left", fontsize=7.2, fontweight="bold")
        yy = {m: y for y, m in zip(ys, METHODS, strict=True)}
        ax.annotate("within-pair Bonferroni\n(not genome-wide)", (xs["snpxsnp"], yy["snpxsnp"]),
                    textcoords="offset points", xytext=(6, 0), fontsize=4.6, color=C_SNPXSNP, va="center")
        ax.annotate("non-localizing", (xs["globalVC"], yy["globalVC"]), textcoords="offset points",
                    xytext=(6, 0), fontsize=4.8, color="#888888", va="center")

    g = {h["key"]: h for h in real["hits"]}
    forest(axB1, g["Hit1_fiber_length"], "B   real cotton recall\nHit 1 fiber length  A06|D06")
    forest(axB2, g["Hit2_length_uniformity"], "Hit 2 length uniformity  A11|D11")
    # colour key (method class, matching Panel A); detection semantics live in the caption, not the fill
    fig.text(0.78, 0.012, "point colour = method class (teal = targeted, purple = single-pair, grey = other)",
             ha="center", va="bottom", fontsize=5.0, color="#666666")

    for ext in ("pdf", "png"):
        p = OUT_DIR / f"fig6_baseline_benchmark.{ext}"
        fig.savefig(p, dpi=300 if ext == "png" else None)
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
