"""Tier 1 Path H' — spike-in epistasis detection power simulation.

For each (n, h2_epi) cell, simulate ``n_replicates`` phenotypes drawn from
y ~ N(0, h2_add·K_pool + h2_epi·K_hom + (1-h2_add-h2_epi)·I) and measure
the fraction where a boundary-aware LRT rejects H0: σ²_h = 0.

The output power-grid TSV is the "honest negative finding" Supplementary
evidence: at panel sizes 300-800 and ploidy 2-6 (our actual panels), how
much true cross-subgenome epistasis variance is needed for K_hom to even
be detectable?

Usage
-----
Single panel:
    python scripts/phase4/khom_tier1_H_simulation.py \\
        --kpool-npz data/processed/rapeseed/horvath/K_pool.npz \\
        --khom-npz data/processed/rapeseed/horvath/K_hom.npz \\
        --n-grid 200 300 400 500 \\
        --h2-epi-grid 0.0 0.05 0.10 0.20 0.30 \\
        --n-replicates 50 \\
        --out results/phase4/khom_tier1_H/horvath2020_spike_power.tsv

Quick smoke (random GRMs, no panel data needed):
    python scripts/phase4/khom_tier1_H_simulation.py --smoke \\
        --out results/phase4/khom_tier1_H/smoke.tsv

Charter §2.1 Tier 1 H' framing
------------------------------
Result enters paper Supplementary as detection-limit calibration; does NOT
promote K_hom to main claim. Power = 0 at realistic h2_epi reinforces the
"transparent negative finding" stance from charter §2.1.

Cost
----
~1000 REML fits per panel for the default 5×5 grid × 50 replicates ×
2 fits/replicate (null + alt). At ~1-3s per REML on n=500, expect
30-90 min per panel CPU. For Phase 4 paper-grade bump n_replicates to
200 and budget overnight.

NOTE: ``_run_one_replicate`` in src/homoeogwas/khom_tier1.py currently
returns a PLACEHOLDER REML output (the spike-in scaffolding ships before
the production wire-up). Once ``fit_multi_reml`` is audited and wired
in, this driver script will produce paper-grade power curves with no
script-side changes.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from homoeogwas.khom_tier1 import spike_in_power_grid  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--kpool-npz", type=Path,
                   help="K_pool kernel .npz (key 'K'); skip if --smoke")
    p.add_argument("--khom-npz", type=Path,
                   help="K_hom kernel .npz (key 'K'); skip if --smoke")
    p.add_argument("--n-grid", type=int, nargs="+", required=False)
    p.add_argument("--h2-epi-grid", type=float, nargs="+", required=False)
    p.add_argument("--h2-add", type=float, default=0.30)
    p.add_argument("--n-replicates", type=int, default=50)
    p.add_argument("--base-seed", type=int, default=2026)
    p.add_argument("--smoke", action="store_true",
                   help="Use random GRMs for end-to-end smoke (no panel data)")
    p.add_argument("--allow-placeholder", action="store_true",
                   help="Codex Q4 fix: must be set to run while khom_tier1._run_one_replicate "
                        "still uses PLACEHOLDER REML. Refuses to write a TSV otherwise. "
                        "Remove this flag once fit_multi_reml is wired in.")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--progress", action="store_true")
    return p.parse_args()


def _load_kernel(path: Path) -> np.ndarray:
    npz = np.load(path)
    if "K" in npz:
        return npz["K"]
    # Try common alternatives
    for key in ("kernel", "grm", "K_pool", "K_hom"):
        if key in npz:
            return npz[key]
    raise KeyError(f"{path}: no recognised kernel key in {list(npz.keys())}")


def _make_random_kernels(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n // 2))
    B = rng.standard_normal((n, n // 2))
    K_pool = (A @ A.T) / (n // 2)
    K_hom = (B @ B.T) / (n // 2)
    return K_pool, K_hom


TIER1_BANNER = (
    "[TIER 1 DEFENSIVE — supplementary only;\n"
    " charter §2.1 hard gate forbids promoting Tier 1 results to main claim.]"
)


def main() -> int:
    args = parse_args()
    print(TIER1_BANNER)

    # Codex Q4 hard gate — driver refuses to run with PLACEHOLDER REML unless
    # the operator explicitly opts in. Prevents PLACEHOLDER TSVs from being
    # silently consumed by the Phase 5 paper figure pipeline.
    from homoeogwas.khom_tier1 import _run_one_replicate  # noqa: F401
    if not args.allow_placeholder:
        print("ERROR: khom_tier1._run_one_replicate is PLACEHOLDER (scaffolding).",
              file=sys.stderr)
        print("       Pass --allow-placeholder to run anyway (TSV will be marked "
              "paper_value=False, backend=PLACEHOLDER).", file=sys.stderr)
        print("       Remove the gate once fit_multi_reml is wired in.",
              file=sys.stderr)
        return 3

    if args.smoke:
        K_pool, K_hom = _make_random_kernels(n=200, seed=args.base_seed)
        n_grid = args.n_grid or [50, 100, 150, 200]
        h2_epi_grid = args.h2_epi_grid or [0.0, 0.10, 0.30]
        print(f"[smoke] random kernels (n=200), grid {len(n_grid)} × {len(h2_epi_grid)}")
    else:
        if not args.kpool_npz or not args.khom_npz:
            print("ERROR: provide --kpool-npz and --khom-npz, or --smoke",
                  file=sys.stderr)
            return 2
        K_pool = _load_kernel(args.kpool_npz)
        K_hom = _load_kernel(args.khom_npz)
        if K_pool.shape != K_hom.shape:
            print(f"ERROR: K_pool {K_pool.shape} ≠ K_hom {K_hom.shape}",
                  file=sys.stderr)
            return 2
        n_max = K_pool.shape[0]
        n_grid = args.n_grid or sorted({n_max // 4, n_max // 2,
                                         (3 * n_max) // 4, n_max})
        h2_epi_grid = args.h2_epi_grid or [0.0, 0.05, 0.10, 0.20, 0.30]

    df = spike_in_power_grid(
        K_pool, K_hom,
        n_grid=tuple(n_grid),
        h2_epi_grid=tuple(h2_epi_grid),
        h2_add=args.h2_add,
        n_replicates=args.n_replicates,
        base_seed=args.base_seed,
        progress=args.progress,
        skip_normalization_check=args.smoke,   # smoke uses random kernels
    )
    df["tier"] = "Tier1_H_spike_in_simulation"
    df["paper_section"] = "Supplementary"
    df["promotable_to_main"] = False
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, sep="\t", index=False)
    print(df.to_string(index=False))
    print(f"\n[done] wrote {args.out}")
    print("\n→ NOTE (charter §2.1): power = 0 at the panel sizes / h2_epi we tested")
    print("  is detection-limit evidence — strengthens the negative finding, does NOT")
    print("  invalidate K_hom for synteny-aware/local revival (Tier 2).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
