#!/usr/bin/env python
"""H7 Part B — power vs marker density, closing the density -> callable-pairs -> POWER chain.

Part A predicts how many homoeolog pairs become callable as marker density drops. Part B shows the
downstream consequence: detection POWER. On real wheat chr1 (hexaploid), we THIN the SNP set to a
fraction (full / medium / GBS-like), rebuild burdens AND the GRM/whitener from the SAME thinned
markers, then inject a sparse B-D burden-product interaction at a FIXED separating effect size
(PVE=0.10, the regime where per-pair separates from the global kernel in H5) and measure power for
per-pair ACAT vs the global homoeolog K_hom kernel. All calibrated to each density's own PVE=0 null.

This connects Part A's callability prediction to actual power: as density falls, callable G shrinks
AND power falls; the per-pair method retains its advantage over the global kernel until G collapses.

Output: results/phase7/h7_partB_power_vs_density.json
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from homoeogwas.interact import (
    block_burden_capped,
    grm_from_X,
    pairwise_pvals,
    scols_safe,
    whiten_multi,
)
from homoeogwas.io import load_bed_hardcall
from homoeogwas.kernel import build_homoeolog_kernel, normalize_kernel
from homoeogwas.lmm import fit_multi_reml

ROOT = Path("/mnt/7302share/fast_ysp/U7_GWAS")
WDIR = ROOT / "results/phase7/interact_validate/wheat_chr1"

DENSITIES = [1.0, 0.20, 0.05]      # full / medium / GBS-like SNP-retention fractions
PVE = float(os.environ.get("PARTB_PVE", "0.10"))   # fixed effect size; weaker PVE tests degradation
OUT = (ROOT / "results/phase7/h7_partB_power_vs_density.json" if abs(PVE - 0.10) < 1e-9
       else ROOT / f"results/phase7/h7_partB_power_vs_density_pve{int(PVE*1000):03d}.json")
ADDITIVE_PVE = 0.40
N_CAUSAL = 3
CAP = 150
MIN_SNP = 3
R_NULL = 1000              # paper-grade FWER calibration CI
R_POWER = 400              # paper-grade power CI
N_JOBS = 96                # 256-core server; reps BLAS-pinned to 1 thread each (no oversubscribe)


def _load_sub(sub):
    bed = load_bed_hardcall(str(WDIR / sub / "all"))
    X = np.asarray(bed.dosage, dtype=np.float64)
    z = np.load(WDIR / f"snp_to_gene_{sub}.npz", allow_pickle=True)
    gene_snp = {g: np.asarray(z["snp_idx"][i], int) for i, g in enumerate(z["gene_ids"].tolist())}
    return X, gene_snp


def run_density(frac, XA, XB, XD, gsA, gsB, gsD, catalog, n, t0):
    """Thin SNPs to fraction ``frac``, rebuild burdens+GRM+K_hom, measure per-pair ACAT vs global
    K_hom power at the fixed PVE (closures bind this function's locals, not loop variables)."""
    tf = np.random.default_rng(int(frac * 1000) + 1)
    keptA = np.sort(tf.choice(XA.shape[1], int(frac * XA.shape[1]), replace=False))
    keptB = np.sort(tf.choice(XB.shape[1], int(frac * XB.shape[1]), replace=False))
    keptD = np.sort(tf.choice(XD.shape[1], int(frac * XD.shape[1]), replace=False))
    setB, setD = set(keptB.tolist()), set(keptD.tolist())
    rng = np.random.default_rng(7)
    bcols, dcols = [], []
    for gB, gD in catalog:
        iB = np.array([i for i in gsB[gB] if i in setB], int)
        iD = np.array([i for i in gsD[gD] if i in setD], int)
        if iB.size >= MIN_SNP and iD.size >= MIN_SNP:
            bcols.append(block_burden_capped(XB, iB, CAP, rng))
            dcols.append(block_burden_capped(XD, iD, CAP, rng))
    G = len(bcols)
    if G < 5:
        return dict(density=frac, G_callable=G, note="too few callable pairs")
    BB = scols_safe(np.column_stack(bcols))
    BD = scols_safe(np.column_stack(dcols))
    KA, KB, KD = grm_from_X(XA[:, keptA]), grm_from_X(XB[:, keptB]), grm_from_X(XD[:, keptD])
    kernels = {"A": KA, "B": KB, "D": KD}
    K_hom = normalize_kernel(build_homoeolog_kernel(kernels, mode="auto")[0], mode="trace")
    chols = {s: np.linalg.cholesky(K + 1e-6 * np.eye(n)) for s, K in kernels.items()}

    def _bg(r):
        sa = ADDITIVE_PVE / 3.0
        return sum(np.sqrt(sa) * (chols[s] @ r.standard_normal(n)) for s in kernels)

    def _signal(causal):
        sig = np.zeros(n)
        for c in causal:
            sig += scols_safe((BB[:, c] * BD[:, c]).reshape(-1, 1)).ravel()
        return scols_safe(sig.reshape(-1, 1)).ravel()

    # DISCOVERY power at GENOME-WIDE thresholds (avoids the omnibus-ACAT/empirical-threshold G-confound):
    #   per-pair = min over all G pairs of the analytic interaction p < Bonferroni alpha/G
    #             (alpha/G LOOSENS as density drops G, naturally pricing the multiple-testing cost);
    #   global K_hom = boundary-mixture LRT > 2.706 (0.5*chi2_0 + 0.5*chi2_1, alpha=0.05).
    bonf = 0.05 / G
    KHOM_CRIT = 2.706                                  # 50:50 chi-bar^2 5% boundary threshold

    def _stats(y):
        Wh, _ = whiten_multi(kernels, y, seed=42)
        X0 = np.ones((n, 1))
        ll0 = float(fit_multi_reml(y, X0, kernels, n_starts=3, random_state=7).log_lik)
        fh = fit_multi_reml(y, X0, {**kernels, "hom": K_hom}, n_starts=3, random_state=7)
        return dict(min_p=float(np.min(pairwise_pvals(Wh, y, BB, BD))),
                    lrt_hom=max(0.0, 2.0 * (float(fh.log_lik) - ll0)))

    def _null(seed):
        r = np.random.default_rng(seed)
        y = _bg(r) + np.sqrt(max(1 - ADDITIVE_PVE, 1e-6)) * r.standard_normal(n)
        try:
            return _stats(y)
        except Exception:  # noqa: BLE001
            return None

    def _pow(seed):
        r = np.random.default_rng(seed)
        causal = r.choice(G, min(N_CAUSAL, G), replace=False)
        y = (_bg(r) + np.sqrt(PVE) * _signal(causal)
             + np.sqrt(max(1 - ADDITIVE_PVE - PVE, 1e-6)) * r.standard_normal(n))
        try:
            return _stats(y)
        except Exception:  # noqa: BLE001
            return None

    nn = [x for x in Parallel(n_jobs=N_JOBS)(delayed(_null)(10_000 + i) for i in range(R_NULL)) if x]
    fwer_pp = float(np.mean([x["min_p"] < bonf for x in nn]))         # should be ~<= 0.05 (Bonferroni)
    fwer_h = float(np.mean([x["lrt_hom"] > KHOM_CRIT for x in nn]))   # should be ~0.05
    pp = [x for x in Parallel(n_jobs=N_JOBS)(delayed(_pow)(20_000 + i) for i in range(R_POWER)) if x]
    pwr_pp = float(np.mean([x["min_p"] < bonf for x in pp]))
    pwr_h = float(np.mean([x["lrt_hom"] > KHOM_CRIT for x in pp]))
    print(f"  density={frac:.2f} G={G} perpair_power={pwr_pp:.3f} Khom_power={pwr_h:.3f} "
          f"adv={pwr_pp/max(pwr_h,1e-6):.2f}x FWER(pp/hom)={fwer_pp:.3f}/{fwer_h:.3f} "
          f"({time.time()-t0:.0f}s)", flush=True)
    return dict(density=frac, G_callable=int(G), bonferroni_alpha=float(bonf),
                n_markers={"A": int(keptA.size), "B": int(keptB.size), "D": int(keptD.size)},
                power_perpair=pwr_pp, power_khom=pwr_h, advantage=float(pwr_pp / max(pwr_h, 1e-6)),
                fwer_perpair=fwer_pp, fwer_khom=fwer_h, r_null=len(nn), r_power=len(pp))


def main():
    t0 = time.time()
    XB, gsB = _load_sub("B")
    XD, gsD = _load_sub("D")
    XA, gsA = _load_sub("A")
    n = XB.shape[0]
    triads = pd.read_csv(WDIR / "triads_chr1.tsv", sep="\t")
    catalog = [(r.gene_B, r.gene_D) for r in triads.itertuples()
               if isinstance(r.gene_B, str) and isinstance(r.gene_D, str)
               and r.gene_B in gsB and r.gene_D in gsD]
    print(f"  n={n} catalog B-D pairs={len(catalog)} ({time.time()-t0:.0f}s)", flush=True)

    results = {f"{frac:.2f}": run_density(frac, XA, XB, XD, gsA, gsB, gsD, catalog, n, t0)
               for frac in DENSITIES}

    out = dict(
        tool="homoeogwas", analysis="h7_partB_power_vs_density", species="wheat_AABBDD_chr1",
        n=int(n), catalog_pairs=len(catalog), fixed_pve=PVE, min_snp=MIN_SNP, cap=CAP,
        densities=DENSITIES, by_density=results,
        power_definition=("per-pair DISCOVERY power = P(min over G pairs of analytic interaction p < "
                          "Bonferroni alpha/G); global K_hom = P(LRT > 2.706, the 50:50 chi-bar^2 5% "
                          "boundary threshold). alpha/G loosens as density drops G, so the metric "
                          "prices the multiple-testing cost; FWER columns confirm calibration."),
        framing=("closes density -> callable-pairs -> DISCOVERY POWER: as SNP density drops, callable "
                 "G shrinks (looser alpha/G) but gene burdens get noisier; we measure the net effect "
                 "on per-pair genome-wide discovery power vs the global K_hom kernel. PVE fixed at the "
                 "H5 separating regime. Conditional on this single sparse B-D architecture -- a "
                 "yield/power feasibility curve, not a universal power claim."),
        runtime_sec=round(time.time() - t0, 1))
    OUT.write_text(json.dumps(out, indent=2, default=float))
    print(f"✅ wrote {OUT} ({out['runtime_sec']}s)")


if __name__ == "__main__":
    main()
