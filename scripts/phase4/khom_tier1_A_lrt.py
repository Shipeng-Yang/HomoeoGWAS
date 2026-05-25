"""Tier 1 Path A — Self-Liang (1987) boundary-aware LRT for K_hom.

Compares REML log-likelihoods of a null model (K_pool / per-sub kernels,
no K_hom) vs an alternative model (same kernels + K_hom), then applies
the 0.5·δ_0 + 0.5·χ²₁ mixture to compute a p-value that respects the
σ² ≥ 0 boundary constraint. Replaces the naive REML σ² estimate (pegged
at 0 in 4/4 Phase 5c panels) with an honest hypothesis test.

Usage
-----
This script consumes the GBLUP REML JSON files emitted by Phase 5c
(``results/phase5c/<panel>/<trait>/gblup_*.json``) which include the
``loglik`` field for each tier. If those JSONs are not yet on disk,
re-run scripts/phase5c/run_gblup_panel.py first.

    python scripts/phase4/khom_tier1_A_lrt.py \\
        --gblup-json results/phase5c/cotton_hebau/fiber_length_BLUE/gblup_cotton_hebau_fiber_length_BLUE.json \\
        --out results/phase4/khom_tier1_A/cotton_hebau_fiber_length_BLUE_lrt.tsv

For batch processing across all 4 panels:
    python scripts/phase4/khom_tier1_A_lrt.py \\
        --gblup-glob 'results/phase5c/*/*/gblup_*.json' \\
        --out results/phase4/khom_tier1_A/all_panels_lrt.tsv

Charter §2.1 Tier 1 A framing
-----------------------------
Result enters paper Supplementary ONLY. Even if LRT rejects σ²_h = 0 on
some panels, this is panel-specific evidence, NOT cross-panel universal;
the 5-item hard gate (charter §2.1) still applies for promotion.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from homoeogwas.khom_tier1 import self_liang_lrt  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--gblup-json", type=Path,
                     help="single GBLUP JSON (one panel × trait)")
    src.add_argument("--gblup-glob",
                     help="glob pattern for batch (e.g. results/phase5c/*/*/gblup_*.json)")
    p.add_argument("--null-tier", default="tier0",
                   help="key for null log-likelihood (default tier0 = K_pool only)")
    p.add_argument("--alt-tier", default="tier1",
                   help="key for alt log-likelihood (default tier1 = K_pool + K_hom)")
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def _extract_loglik(gblup: dict, tier: str) -> tuple[float | None, str]:
    """Look up tier's REML log-likelihood; returns (ll_or_None, source_note).

    Different gblup_*.json schemas live in the wild (some store per-fold,
    some store mean only); this scaffolding tries the common shapes.
    """
    if "tiers" in gblup and tier in gblup["tiers"]:
        t = gblup["tiers"][tier]
        if "mean_loglik" in t:
            return float(t["mean_loglik"]), "tiers.<tier>.mean_loglik"
        if "loglik" in t:
            return float(t["loglik"]), "tiers.<tier>.loglik"
        if "fold_logliks" in t:
            return float(sum(t["fold_logliks"]) / len(t["fold_logliks"])), \
                "tiers.<tier>.fold_logliks mean"
    return None, "NOT FOUND in any expected location — wire when impl audited"


def _process_one(path: Path, args: argparse.Namespace) -> dict:
    with path.open() as f:
        gb = json.load(f)
    panel = gb.get("panel") or path.parts[-3]
    trait = gb.get("trait") or path.parts[-2]
    ll_null, src_null = _extract_loglik(gb, args.null_tier)
    ll_alt, src_alt = _extract_loglik(gb, args.alt_tier)
    if ll_null is None or ll_alt is None:
        return {
            "panel": panel, "trait": trait,
            "ll_null": ll_null, "ll_alt": ll_alt,
            "lr_stat": None, "p_value": None, "boundary": None,
            "null_source": src_null, "alt_source": src_alt,
            "status": "MISSING_LOGLIK",
        }
    lrt = self_liang_lrt(ll_null=ll_null, ll_alt=ll_alt, n_vc_at_boundary=1)
    return {
        "panel": panel, "trait": trait,
        "ll_null": ll_null, "ll_alt": ll_alt,
        "lr_stat": lrt.lr_stat,
        "p_value": lrt.p_value,
        "boundary": lrt.boundary,
        "null_source": src_null, "alt_source": src_alt,
        "status": "BOUNDARY" if lrt.boundary else "OK",
    }


TIER1_BANNER = (
    "[TIER 1 DEFENSIVE — supplementary only;\n"
    " charter §2.1 hard gate forbids promoting Tier 1 results to main claim.]"
)


def main() -> int:
    args = parse_args()
    print(TIER1_BANNER)
    if args.gblup_json:
        paths = [args.gblup_json]
    else:
        paths = [Path(p) for p in sorted(glob.glob(args.gblup_glob))]
    if not paths:
        print("ERROR: no GBLUP JSONs matched", file=sys.stderr)
        return 2

    rows = [_process_one(p, args) for p in paths]
    df = pd.DataFrame(rows)
    df["tier"] = "Tier1_A_self_liang_LRT"
    df["paper_section"] = "Supplementary"
    df["promotable_to_main"] = False
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, sep="\t", index=False)
    print(df.to_string(index=False))
    print(f"\n[done] wrote {args.out}")

    n_boundary = int(df.get("boundary", pd.Series(dtype=bool)).fillna(False).sum())
    n_total = len(df)
    print(f"\nBoundary panels: {n_boundary}/{n_total}")
    print("→ NOTE (charter §2.1): individual panel sig MUST be reported case-by-case,")
    print("  NOT aggregated into 'across panels positive evidence' (banned-words list).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
