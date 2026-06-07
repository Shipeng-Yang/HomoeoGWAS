"""Supplementary figure — real external published tools do not recall the two cotton homoeolog-pair
hits at genome-wide significance, while homoeogwas does.

Two panels (Hit1, Hit2): horizontal bars = best -log10(P) each external tool achieves in the hit
region, grouped by estimand class; dashed line = homoeogwas P (genome-wide candidate-pair discovery).
PLINK epistasis is annotated as candidate-pass / genome-wide-miss (resolution/multiplicity result).
"""
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/mnt/7302share/fast_ysp/U7_GWAS")
EB = ROOT / "results/phase7/ext_baseline"
OUT = ROOT / "results/phase7/figures"
OUT.mkdir(parents=True, exist_ok=True)

C_LOCUS = "#0072B2"   # single-locus marginal
C_BURDEN = "#E69F00"  # single-gene burden
C_EPI = "#7B61FF"     # SNP×SNP epistasis
C_OURS = "#008C8C"    # homoeogwas reference
ESTC = {"single-locus": C_LOCUS, "single-gene": C_BURDEN, "epistasis": C_EPI}

# display order (top->bottom) + estimand class + short label
TOOLS = [
    ("PLINK epistasis", "epistasis", "PLINK epistasis\n(SNP×SNP)"),
    ("regenie (single-variant)", "single-locus", "regenie\n(single-variant)"),
    ("GWASpoly", "single-locus", "GWASpoly\n(polyploid marker)"),
    ("regenie (gene-burden)", "single-gene", "regenie\n(gene burden)"),
    ("STAAR", "single-gene", "STAAR\n(STAAR-O)"),
    ("SKAT", "single-gene", "SKAT\n(SKAT-O)"),
]

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 8, "axes.spines.top": False,
    "axes.spines.right": False, "axes.titlesize": 8.5, "figure.dpi": 150, "savefig.bbox": "tight",
})


def main():
    d = json.loads((EB / "external_baselines.json").read_text())
    recs = {(r["tool"], r["hit"]): r for r in d["results"]}
    hits = [("Hit1_fiber_length", "Hit 1  fiber length  A06|D06"),
            ("Hit2_length_uniformity", "Hit 2  length uniformity  A11|D11")]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
    for ax, (hit, title) in zip(axes, hits, strict=True):
        ours_p = d["ours"][hit]["ours_p"]
        ys = list(range(len(TOOLS)))[::-1]
        ox = -math.log10(ours_p)
        for y, (tool, cls, _lab) in zip(ys, TOOLS, strict=True):
            r = recs[(tool, hit)]
            x = -math.log10(r["best_p"]) if r["best_p"] else 0
            ax.barh(y, x, color=ESTC[cls], alpha=0.85, height=0.62, zorder=3)
            ax.text(x + 0.08, y, f"P={r['best_p']:.2g}", va="center", fontsize=5.4, color="#333")
            if tool == "PLINK epistasis":
                ax.text(0.15, y - 0.46, "candidate-pass, genome-wide miss", va="top",
                        fontsize=5.0, color=C_EPI, style="italic")
        # homoeogwas reference line (vertical label along the line, mid-height)
        ax.axvline(ox, color=C_OURS, lw=1.6, ls="--", zorder=4)
        ax.text(ox - 0.12, 2.5, f"homoeogwas  P={ours_p:.1g}", color=C_OURS, fontsize=6.0,
                ha="center", va="center", rotation=90, fontweight="bold")
        ax.set_yticks(ys)
        ax.set_yticklabels([t[2] for t in TOOLS], fontsize=6.0)
        ax.set_xlim(0, ox + 1.3)
        ax.set_xlabel("best $-\\log_{10}P$ in hit region", fontsize=7)
        ax.set_title(title, loc="left", fontsize=7.4, fontweight="bold")

    # estimand legend
    from matplotlib.patches import Patch
    handles = [Patch(color=C_LOCUS, label="single-locus marginal"),
               Patch(color=C_BURDEN, label="single-gene burden/SKAT"),
               Patch(color=C_EPI, label="SNP×SNP epistasis")]
    axes[1].legend(handles=handles, loc="lower right", frameon=False, fontsize=5.6)
    fig.suptitle("External published tools do not recover the homoeolog-pair hits at genome-wide "
                 "significance; homoeogwas does", fontsize=8, y=1.02)
    for ext in ("pdf", "png"):
        p = OUT / f"figS_external_baselines.{ext}"
        fig.savefig(p, dpi=300 if ext == "png" else None)
        print("wrote", p)


if __name__ == "__main__":
    main()
