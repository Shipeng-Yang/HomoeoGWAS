"""Tier 1 Path F — K_hom ablation on real Phase 3 LOCO data.

Re-run a panel × trait LOCO scan twice (with and without K_hom in the
kinship stack), then compute recall@K against the panel's known-QTL
anchor set. Writes a per-panel ablation_recall.tsv.

Usage
-----
    # 1) Generate the with-K_hom sumstats (slow: 5min – 90min wall-clock)
    cp configs/runs/fit_horvath2020_plant_height_loco.yaml \\
       configs/runs/fit_horvath2020_plant_height_loco_with_khom.yaml
    # set kernels.include_hadamard: true in the copy
    homoeogwas fit --config configs/runs/fit_horvath2020_plant_height_loco_with_khom.yaml \\
                   --out-dir results/phase4/khom_tier1_F/horvath2020_plant_height/with_khom

    # 2) Run this script
    python scripts/phase4/khom_tier1_F_ablation.py \\
        --panel horvath2020 --trait plant_height \\
        --sumstats-without results/phase3/m3_1_loco/horvath2020/sumstats_plant_height_loco.tsv \\
        --sumstats-with results/phase4/khom_tier1_F/horvath2020_plant_height/with_khom/sumstats_plant_height_loco.tsv \\
        --known-qtl data/reference/rapeseed/known_qtl_rapeseed.tsv \\
        --out results/phase4/khom_tier1_F/horvath2020_plant_height/ablation_recall.tsv

The "with" sumstats path will not exist until you have re-run the LOCO scan
with K_hom enabled — see the cp/edit step in the comment above. The first
time you run this script for a new panel × trait, expect FileNotFoundError
on --sumstats-with; that is the cue to do the re-run.

Charter §2.1 Tier 1 F framing
-----------------------------
Result enters paper Supplementary ONLY. Cannot be used to promote K_hom
back to a main claim (see 5-item hard gate in charter §2.1).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Allow running from project root without install
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from homoeogwas.khom_tier1 import ablation_table  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--panel", required=True)
    p.add_argument("--trait", required=True)
    p.add_argument("--sumstats-without", type=Path, required=True,
                   help="Phase 3 LOCO sumstats WITHOUT K_hom (include_hadamard: false)")
    p.add_argument("--sumstats-with", type=Path, required=True,
                   help="Phase 4 ablation re-run sumstats WITH K_hom (include_hadamard: true)")
    p.add_argument("--known-qtl", type=Path, required=True,
                   help="data/reference/<panel>/known_qtl_*.tsv")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--k-grid", type=int, nargs="+",
                   default=[10, 50, 100, 500, 1000, 5000])
    p.add_argument("--window-bp", type=int, default=500_000)
    p.add_argument("--sumstats-chrom-col", default="chrom")
    p.add_argument("--sumstats-pos-col", default="pos")
    p.add_argument("--sumstats-p-col", default="p")
    p.add_argument("--known-chrom-col", default="logical_chrom")
    p.add_argument("--known-pos-col", default="pos")
    return p.parse_args()


TIER1_BANNER = (
    "[TIER 1 DEFENSIVE — supplementary only;\n"
    " charter §2.1 hard gate forbids promoting Tier 1 results to main claim.]"
)


def main() -> int:
    args = parse_args()
    print(TIER1_BANNER)
    for tag, path in [("without", args.sumstats_without),
                      ("with", args.sumstats_with),
                      ("known_qtl", args.known_qtl)]:
        if not path.exists():
            print(f"ERROR: --{tag} {path} does not exist", file=sys.stderr)
            return 2

    ss_without = pd.read_csv(args.sumstats_without, sep="\t")
    ss_with = pd.read_csv(args.sumstats_with, sep="\t")
    kn = pd.read_csv(args.known_qtl, sep="\t")
    print(f"[{args.panel}/{args.trait}] sumstats without={len(ss_without):,}, "
          f"with={len(ss_with):,}, known_qtl={len(kn):,}")

    tab = ablation_table(
        ss_without, ss_with, kn,
        k_grid=tuple(args.k_grid),
        window_bp=args.window_bp,
        sumstats_chrom_col=args.sumstats_chrom_col,
        sumstats_pos_col=args.sumstats_pos_col,
        sumstats_p_col=args.sumstats_p_col,
        known_chrom_col=args.known_chrom_col,
        known_pos_col=args.known_pos_col,
    )
    tab.insert(0, "trait", args.trait)
    tab.insert(0, "panel", args.panel)
    # TSV metadata flags so a downstream pipeline cannot
    # silently consume the row as paper-grade or main-claim evidence.
    tab["tier"] = "Tier1_F_ablation"
    tab["paper_section"] = "Supplementary"
    tab["promotable_to_main"] = False

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tab.to_csv(args.out, sep="\t", index=False)
    print(tab.to_string(index=False))
    print(f"\n[done] wrote {args.out}")

    # Simple consistency check & framing note
    mean_delta = tab["delta_recall"].mean()
    print(f"\nMean Δrecall across k_grid = {mean_delta:+.4f}")
    if mean_delta > 0:
        print("→ K_hom *added* power on this panel/trait; report in Supplementary.")
    elif mean_delta < 0:
        print("→ K_hom *hurt* power on this panel/trait; report in Supplementary.")
    else:
        print("→ Indistinguishable; report in Supplementary.")
    print("→ NOTE (charter §2.1): this result CANNOT promote K_hom to main claim;")
    print("  Tier 2 (synteny-aware local K_hom or epistasis-rich trait) required.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
