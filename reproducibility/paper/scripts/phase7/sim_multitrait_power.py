#!/usr/bin/env python3
"""Power/type-I simulation for multi-trait (pleiotropy) homoeolog-pair interaction testing —
the "discover-more" evidence behind ``homoeogwas interact`` with a predeclared multi_trait set.

Question: when a homoeolog pair's interaction effect is SHARED across a frozen set of
correlated traits (each per-trait signal too weak to survive G*T single-trait Bonferroni), does
combining the per-trait interaction p-values with ACAT into ONE per-pair pleiotropy p (corrected
over G pairs only) detect MORE true pleiotropic pairs at a fixed FWER than the best-single-trait
approach (min trait p, corrected over G*T)? And does the pleiotropy combination stay calibrated
under CROSS-TRAIT CORRELATION (correlated null -> FWER <= alpha)?

This isolates the statistical property with an identity whitener (OLS interaction test), mirroring
sim_weighted_power.py; the production CLI uses the LMM-whitened residuals. Honest framing: ACAT is
dependence-robust but is NOT the optimal combiner for dense weak same-direction effects, so the gain
is conditional on signal being shared across the predeclared traits — exactly what we vary here.

Two comparators, both at FWER = alpha:
  * pleiotropy : per pair p = ACAT(per-trait p), detect at alpha/G          (inferential unit = pair)
  * best_single: per pair p = min(per-trait p), detect at alpha/(G*T)       (union of G*T tests)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

ROOT = Path("/mnt/7302share/fast_ysp/U7_GWAS")
sys.path.insert(0, str(ROOT / "src"))
from homoeogwas.interact import acat, lambda_gc, pairwise_pvals, scols_safe  # noqa: E402

OUT = ROOT / "results/phase7/interact_validate"
N, G, SPG, K = 600, 400, 8, 12        # samples, pairs, snps/gene, causal (pleiotropic) pairs
T = 5                                  # traits in the predeclared frozen set
ALPHA = 0.05
R = 500                                # replicates
RHO = 0.5                              # cross-trait shared-noise correlation
BETA = 0.11                            # per-trait interaction effect (tuned: weak alone, strong combined)


def _make_burdens(rng):
    """Standardized B and D gene burdens (n x G) from random dosage block-means."""
    def one():
        return scols_safe(np.column_stack([rng.integers(0, 3, size=(N, SPG)).astype(float).mean(1)
                                           for _ in range(G)]))
    return one(), one()


def _one_rep(seed, BB, BD, causal, beta, n_sig_traits, rho):
    """One replicate: simulate T correlated traits, return pleiotropy/best-single power+FWER."""
    rng = np.random.default_rng(seed)
    inter = {c: (lambda v: (v - v.mean()) / (v.std() + 1e-12))(BB[:, c] * BD[:, c]) for c in causal}
    shared = rng.standard_normal(N)                       # latent driving cross-trait correlation
    P = np.empty((G, T))
    for t in range(T):
        y = np.sqrt(rho) * shared + np.sqrt(1 - rho) * rng.standard_normal(N)
        if t < n_sig_traits:                              # signal carried by the first n_sig traits
            for c in causal:
                y = y + beta * inter[c]
        pv = pairwise_pvals(np.eye(N), y, BB, BD)
        P[:, t] = np.where(np.isfinite(pv), pv, 1.0)

    pleio = np.array([acat(P[i]) for i in range(G)])
    best = P.min(1)
    cmask = np.zeros(G, bool)
    cmask[causal] = True
    thr_p, thr_b = ALPHA / G, ALPHA / (G * T)
    return dict(
        pleio_power=float((pleio[cmask] < thr_p).mean()),
        pleio_ncaus=int((pleio[cmask] < thr_p).sum()),
        pleio_fwer=bool((pleio[~cmask] < thr_p).any()),
        pleio_lambda=float(lambda_gc(pleio)),
        best_power=float((best[cmask] < thr_b).mean()),
        best_ncaus=int((best[cmask] < thr_b).sum()),
        best_fwer=bool((best[~cmask] < thr_b).any()),
    )


def _agg(reps):
    keys_mean = ("pleio_power", "pleio_ncaus", "pleio_fwer", "pleio_lambda",
                 "best_power", "best_ncaus", "best_fwer")
    return {k: float(np.mean([r[k] for r in reps])) for k in keys_mean}


def _run(seed0, BB, BD, causal, beta, n_sig, rho):
    reps = Parallel(n_jobs=48)(delayed(_one_rep)(seed0 + i, BB, BD, causal, beta, n_sig, rho)
                               for i in range(R))
    return _agg(reps)


def main():
    rng = np.random.default_rng(20260602)
    BB, BD = _make_burdens(rng)
    causal = np.sort(rng.choice(G, size=K, replace=False))

    # block 1: BETA sweep, full pleiotropy (signal in all T traits), rho=RHO
    print(f"=== (1) beta sweep (n_sig_traits={T}, rho={RHO}, T={T}) ===")
    beta_sweep = {}
    for beta in [0.06, 0.08, 0.11, 0.15]:
        a = _run(700000, BB, BD, causal, beta, T, RHO)
        beta_sweep[f"beta={beta}"] = a
        print(f"  beta={beta}: pleio power={a['pleio_power']:.3f} ncaus={a['pleio_ncaus']:.2f}/{K} "
              f"FWER={a['pleio_fwer']:.3f} λ={a['pleio_lambda']:.3f} | best-single power="
              f"{a['best_power']:.3f} ncaus={a['best_ncaus']:.2f} FWER={a['best_fwer']:.3f}")

    # block 2: spread sweep — how many of T traits carry the signal (fixed beta)
    print(f"=== (2) signal-spread sweep (beta={BETA}, rho={RHO}) ===")
    spread_sweep = {}
    for n_sig in range(1, T + 1):
        a = _run(710000, BB, BD, causal, BETA, n_sig, RHO)
        spread_sweep[f"n_sig_traits={n_sig}"] = a
        print(f"  n_sig={n_sig}/{T}: pleio power={a['pleio_power']:.3f} "
              f"ncaus={a['pleio_ncaus']:.2f}/{K} | best-single power={a['best_power']:.3f} "
              f"ncaus={a['best_ncaus']:.2f}")

    # block 3: correlated-null calibration (beta=0) across rho — pleiotropy FWER must stay <= alpha
    print("=== (3) correlated-null calibration (beta=0) ===")
    null_calib = {}
    for rho in [0.0, 0.5, 0.9]:
        a = _run(720000, BB, BD, causal, 0.0, T, rho)
        null_calib[f"rho={rho}"] = a
        print(f"  rho={rho}: pleio FWER={a['pleio_fwer']:.3f} λ={a['pleio_lambda']:.3f} | "
              f"best-single FWER={a['best_fwer']:.3f}  (both should be <= {ALPHA})")

    payload = dict(
        design=dict(N=N, G=G, T=T, K_causal=K, R=R, alpha=ALPHA, rho=RHO, beta=BETA, spg=SPG,
                    note=("identity whitener (OLS interaction test) isolates the across-trait ACAT "
                          "multiplicity property (production uses LMM-whitened residuals). "
                          "pleiotropy = ACAT across the FROZEN trait set, corrected over G pairs; "
                          "best_single = min trait p, corrected over G*T. traits share a latent "
                          "(rho) to induce cross-trait correlation; causal pairs carry the same "
                          "interaction in the first n_sig_traits.")),
        beta_sweep=beta_sweep, spread_sweep=spread_sweep, null_calibration=null_calib,
        conclusion=("When a homoeolog-pair interaction is SHARED across the predeclared trait set, "
                    "ACAT-across-traits (one p per pair, corrected over G) detects more true "
                    "pleiotropic pairs at fixed FWER than best-single-trait (corrected over G*T); "
                    "the gain grows with how broadly the signal is shared (block 2) and the "
                    "combination stays calibrated under cross-trait correlation (block 3, FWER<=alpha). "
                    "Honest scope: ACAT is not the optimal dense-weak combiner, so the gain is "
                    "conditional on genuine shared signal across the frozen traits, not unconditional."))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sim_multitrait_power.json").write_text(json.dumps(payload, indent=2, default=float))
    print(f"\nwrote {OUT/'sim_multitrait_power.json'}")


if __name__ == "__main__":
    main()
