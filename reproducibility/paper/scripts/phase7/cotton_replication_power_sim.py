#!/usr/bin/env python3
"""Real-genotype power for the cotton elongation replication (prereg s7, codex fix #2).

The analytic noncentral-t bound (power_bound.json) is optimistic: it assumes df=n-4 for an omnibus
with an estimated covariance, and ignores that the NDM8 panel has its own allele frequencies, LD and
burden variance. This measures power the honest way: inject the discovery-scale interaction into the
HBAU panel's OWN genotypes and kinship at the mapped elongation pair, and count how often the frozen
test (alpha=0.025, direction-concordant) recovers it. codex also notes that halving the interaction
COEFFICIENT quarters the PVE, so the winner's-curse row uses PVE/4, not PVE/2.

Reads NO interaction p-value of the real phenotype: only genotypes, gene spans and (for the null
variance split) the elongation phenotype's main-effect fit. The interaction of the real phenotype is
never evaluated here.
"""
from __future__ import annotations

import gzip
import json
import os
import re
import sys
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "16")

import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import orth

ROOT = Path("/mnt/7302share/fast_ysp/U7_GWAS")
sys.path.insert(0, str(ROOT / "src"))
from homoeogwas.interact import block_burden_capped, grm_from_X, rank_int  # noqa: E402
from homoeogwas.io import load_bed_hardcall  # noqa: E402
from homoeogwas.kernel import normalize_kernel  # noqa: E402

WGS = ROOT / "results/viz_preview/fig3_cotton_locus/rescue"
NDM8_GFF = Path("/mnt/nvme/cotton_hbau/cottongen/NDM8.gff3.gz")
PHENO = ROOT / "data/processed/cotton/pheno_m3_4_blue.tsv"
CAP, FLANK, MAF = 150, 2000, 0.01
PAIR = ("GhM_A01G0263", "GhM_D01G0230")     # frozen mapping of the elongation lead
ALPHA = 0.025                                # frozen in the prereg
DISC_PVE = 0.0153                            # from power_bound.json (t=4.39 at n=1245)


def _spans():
    sp = {}
    with gzip.open(NDM8_GFF, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.split("\t")
            if len(f) < 9 or f[2] != "gene":
                continue
            m = re.search(r"(GhM_[AD]\d\dG\d+)", f[8])
            if m and m.group(1) in PAIR:
                sp[m.group(1)] = (f[0], int(f[3]), int(f[4]))
    return sp


def _gene_dosage(bed, chrom, lo, hi):
    ch = np.asarray(bed.chrom).astype(str)
    po = np.asarray(bed.pos, dtype=np.int64)
    sel = np.where((ch == chrom) & (po >= lo - FLANK) & (po <= hi + FLANK))[0]
    X = np.asarray(bed.dosage, float)[:, sel]
    mu = np.nanmean(X, axis=0) / 2.0
    maf = np.minimum(mu, 1 - mu)
    return X[:, maf >= MAF]


def _power(Wh, L, bA, bD, pve, reps, rng, direction):
    n = bA.shape[0]
    xi = (bA * bD).ravel()
    sd = xi.std()
    if sd < 1e-12:
        return dict(power=None, note="interaction column constant")
    xi_s = (xi - xi.mean()) / sd
    one = (Wh @ np.ones(n)).reshape(-1, 1)
    Q = orth(np.column_stack([one, Wh @ bA, Wh @ bD]))
    z = (Wh @ xi_s) - Q @ (Q.T @ (Wh @ xi_s))
    ss = float((z ** 2).sum())
    E = L @ rng.standard_normal((n, reps))
    E /= E.std(0, keepdims=True)
    Y = np.sqrt(pve) * (direction * xi_s)[:, None] + np.sqrt(1 - pve) * E
    Y = np.apply_along_axis(rank_int, 0, Y)
    Yw = Wh @ Y
    df = n - Q.shape[1] - 1
    hits = 0
    for j in range(reps):
        yw = Yw[:, j]
        beta = float(z @ yw / ss)
        resid = yw - Q @ (Q.T @ yw) - beta * z
        se = float(np.sqrt((resid ** 2).sum() / df / ss))
        t = beta / se
        if np.sign(t) == direction and 2 * stats.t.sf(abs(t), df) < ALPHA:
            hits += 1
    return dict(power=hits / reps, df=df)


def main():
    rng = np.random.default_rng(21)
    sp = _spans()
    if len(sp) != 2:
        raise SystemExit(f"could not span both genes: {sp}")
    bedA = load_bed_hardcall(WGS / "A_wgs")
    bedD = load_bed_hardcall(WGS / "D_wgs")
    cA, lA, hA = sp[PAIR[0]]
    cD, lD, hD = sp[PAIR[1]]
    XA = _gene_dosage(bedA, cA, lA, hA)
    XD = _gene_dosage(bedD, cD, lD, hD)
    n = XA.shape[0]
    print(f"HBAU panel n={n}; elongation pair SNPs after MAF>={MAF}: A {XA.shape[1]}, D {XD.shape[1]}")
    bA = block_burden_capped(XA, np.arange(XA.shape[1]), CAP, rng, minor=True).reshape(-1, 1)
    bD = block_burden_capped(XD, np.arange(XD.shape[1]), CAP, rng, minor=True).reshape(-1, 1)

    # genome-wide kinship from a subsample of markers on each subgenome (phenotype-free)
    def _grm(bed, cap=60000):
        X = np.asarray(bed.dosage, float)
        idx = rng.choice(X.shape[1], size=min(cap, X.shape[1]), replace=False)
        K = grm_from_X(X[:, np.sort(idx)])
        return normalize_kernel(0.5 * (K + K.T), "trace")
    KA, KD = _grm(bedA), _grm(bedD)

    # null variance split from the elongation phenotype's MAIN-effect fit (no interaction term touched)
    ph = pd.read_csv(PHENO, sep="\t")
    samples = list(np.asarray(bedA.samples).astype(str))
    ph = ph.set_index(ph.columns[0]).reindex(samples)
    y = rank_int(np.asarray(ph["elongation_BLUE"], float))
    keep = np.isfinite(y)
    y = y[keep]
    KAk, KDk = KA[np.ix_(keep, keep)], KD[np.ix_(keep, keep)]
    bAk, bDk = bA[keep], bD[keep]
    m = keep.sum()
    # crude REML-free VC split by profiling a grid of (a,b); residual after main effects only
    Xmain = np.column_stack([np.ones(m), bAk, bDk])
    Pr = np.eye(m) - Xmain @ np.linalg.pinv(Xmain)
    ry = Pr @ y
    best, split = 1e18, (0.3, 0.3)
    for a in np.linspace(0.05, 0.7, 14):
        for b in np.linspace(0.05, 0.7, 14):
            V = a * KAk + b * KDk + (1 - a - b) * np.eye(m) if a + b < 0.95 else None
            if V is None:
                continue
            w, Qv = np.linalg.eigh(0.5 * (V + V.T))
            if w.min() < 1e-8:
                continue
            wh = (Qv * (1 / np.sqrt(w))) @ Qv.T
            r = wh @ ry
            nll = 0.5 * (np.log(w).sum() + m * np.log((r @ r) / m))
            if nll < best:
                best, split = nll, (a, b)
    a, b = split
    print(f"  null variance split on elongation (main-effects only): K_A {a:.2f}, K_D {b:.2f}, "
          f"e {1-a-b:.2f}  (n phenotyped {m})")
    V = a * KAk + b * KDk + (1 - a - b) * np.eye(m)
    w, Qv = np.linalg.eigh(0.5 * (V + V.T))
    w = np.clip(w, 1e-10, None)
    Wh = (Qv * (1 / np.sqrt(w))) @ Qv.T
    L = (Qv * np.sqrt(w)) @ Qv.T

    res = dict(prereg_blob="3b675e8ca696", amendment="24f820e", pair=list(PAIR),
               n_phenotyped=int(m), snps=[int(XA.shape[1]), int(XD.shape[1])],
               vc_split=dict(K_A=a, K_D=b, e=1 - a - b), alpha=ALPHA,
               discovery_pve=DISC_PVE, note="power on real HBAU genotypes; no interaction p read")
    print(f"\n  {'scenario':<34}{'PVE':>8}{'power @419':>12}")
    rows = [("discovery effect (PVE 1.53%)", DISC_PVE),
            ("coefficient halved -> PVE/4", DISC_PVE / 4),
            ("PVE/2 (for comparison w/ analytic)", DISC_PVE / 2)]
    res["power"] = {}
    for label, pve in rows:
        # direction sign is irrelevant to power magnitude; use -1 as a placeholder, symmetric here
        out = _power(Wh, L, bAk, bDk, pve, 3000, rng, direction=-1)
        pw = out["power"]
        res["power"][label] = pw
        print(f"  {label:<34}{pve:>8.4f}{(pw if pw is not None else float('nan')):>12.3f}")

    (ROOT / "results/phase7/cotton_replication_power_sim.json").write_text(json.dumps(res, indent=2))
    print("\nwrote results/phase7/cotton_replication_power_sim.json (no interaction p-value was read)")


if __name__ == "__main__":
    main()
