#!/usr/bin/env python3
"""Tier 1 Path H' — spike-in epistasis power: DEPRECATED / REDIRECT (2026-06-02).

This driver previously called ``homoeogwas.khom_tier1.spike_in_power_grid``, which returned a
PLACEHOLDER (it discarded the simulated phenotype and fabricated σ²_h / LR statistics from a toy
h2_epi formula — never a real REML fit). That fake path was REMOVED from the package, so this
script no longer has anything real to run and is kept only as a redirect.

The genuine spike-in recoverability / detection-power analysis was carried out with real REML in
Phase 7:

    scripts/phase7/d2_khom_recoverability.py
    results/phase7/d2_recoverability/*.json

  Result: K_hom variance is barely recoverable even when it is the ground truth — weak power
  ~0.17–0.21 at PVE_hom=0.20, with ~8–10% upward bias under the null. This is the cited
  "K_hom is not variance-identifiable" evidence (charter §2.1, supplementary only).

To run a real boundary-corrected LRT on a simulated phenotype in-package:

    from homoeogwas.khom_tier1 import simulate_additive_plus_epistasis, SpikeInScenario
    from homoeogwas.diagnostics import compare_nested_reml, boundary_lrt
    y = simulate_additive_plus_epistasis(K_pool, K_hom, SpikeInScenario(n, h2_add, h2_epi))
    cmp = compare_nested_reml(y, X, {"pool": K_pool, "hom": K_hom},
                              model_specs={"pool+e": ["pool"], "pool+hom+e": ["pool", "hom"]})
    blt = boundary_lrt(cmp, "pool+e", "pool+hom+e")     # Self-Liang 0.5·χ²₀ + 0.5·χ²₁
"""
from __future__ import annotations

import sys

_MSG = __doc__
_REDIRECT = ("DEPRECATED: the placeholder spike-in power grid was removed. Use "
             "scripts/phase7/d2_khom_recoverability.py (real REML) instead.")


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if "-h" in argv or "--help" in argv:        # help is not an error -> exit 0
        sys.stdout.write(_MSG + "\n" + _REDIRECT + "\n")
        return 0
    sys.stderr.write(_MSG + "\n" + _REDIRECT + "\n")
    return 2                                     # nonzero so accidental runs fail loudly


if __name__ == "__main__":
    raise SystemExit(main())
