#!/usr/bin/env python3
"""Diagnostic (NOT a replication): why did the elongation pair not replicate in HBAU?

The primary reveal is a null (omniB p=0.37) and stays INCONCLUSIVE. This probes ONE candidate reason
-- the two burdens are built from incommensurable variant sets: the discovery panel (CottonGVD,
MAF>=0.1) had 10 A / 4 D SNPs at this pair, the replication (HBAU, full WGS) has 11 A / 67 D. A
gene-level burden over 4 common variants is a different statistic than one over 67. If matching the
variant set moves the HBAU interaction toward significance, non-transfer is (partly) methodological;
if it stays null, that points to a false positive or a genuinely absent effect.

Three levels of matching, coarse to fine, on the HBAU burden-product interaction (whitened t-test,
the interpretable core of omniB). This is post-hoc and cannot corroborate the discovery.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import orth

ROOT = Path("/mnt/7302share/fast_ysp/U7_GWAS")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "reproducibility/paper/scripts/phase7"))
from homoeogwas.interact import block_burden_capped, grm_from_X, rank_int  # noqa: E402
from homoeogwas.io import load_bed_hardcall  # noqa: E402
from homoeogwas.kernel import normalize_kernel  # noqa: E402
import cotton_cross_panel_ibs as ibs  # noqa: E402

b4 = importlib.util.module_from_spec(importlib.util.spec_from_file_location(
    "b4", str(ROOT / "reproducibility/paper/scripts/phase7/bio_pilot/04b_multi_trait_deploy.py")))
sys.modules["b4"] = b4
importlib.util.spec_from_file_location(
    "b4", str(ROOT / "reproducibility/paper/scripts/phase7/bio_pilot/04b_multi_trait_deploy.py")
).loader.exec_module(b4)

W = ROOT / "results/viz_preview/fig3_cotton_locus/rescue"
DISC = ROOT / "data/processed/cotton/cottongvd"
A_GENE_H, D_GENE_H = "GhM_A01G0263", "GhM_D01G0230"
A_GENE_C, D_GENE_C = "Gh_A01G025100", "Gh_D01G022200"


def _interaction_p(bA, bD, Wh, y, n):
    sA = (bA - bA.mean()) / (bA.std() + 1e-12)
    sD = (bD - bD.mean()) / (bD.std() + 1e-12)
    xi = sA * sD
    if xi.std() < 1e-9:
        return None, None
    Xd = np.column_stack([np.ones(n), sA, sD, (xi - xi.mean()) / xi.std()])
    Xw, yw = Wh @ Xd, Wh @ y
    beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    Q = orth(Xw[:, :3])
    resid = yw - Xw @ beta
    df = n - 4
    se = float(np.sqrt(float(resid @ resid) / df * np.linalg.pinv(Xw.T @ Xw)[3, 3]))
    t = float(beta[3] / se)
    return float(2 * stats.t.sf(abs(t), df)), float(beta[3])


def main():
    rng = np.random.default_rng(0)
    # HBAU genotypes at the pair
    g2s = {}
    for p in ("prep/snp_to_gene_A.npz", "prep/snp_to_gene_D.npz"):
        g2s.update(b4._load_snp_to_gene(W / p))
    bedA = load_bed_hardcall(W / "A_wgs")
    bedD = load_bed_hardcall(W / "D_wgs")
    XA = np.asarray(bedA.dosage, float)
    XD = np.asarray(bedD.dosage, float)
    iA_all = np.asarray(g2s[A_GENE_H], int)
    iD_all = np.asarray(g2s[D_GENE_H], int)

    ph = pd.read_csv(ROOT / "data/processed/cotton/pheno_m3_4_blue.tsv", sep="\t")
    ph = ph.set_index(ph.columns[0])
    sa = list(np.asarray(bedA.samples).astype(str))
    y = ph.reindex(sa)["elongation_BLUE"].to_numpy(float)
    keep = np.isfinite(y)
    y = rank_int(y[keep])
    n = int(keep.sum())
    KA = normalize_kernel(grm_from_X(XA[keep]), "trace")
    KD = normalize_kernel(grm_from_X(XD[keep]), "trace")
    Wh, _ = b4._whiten(KA, KD, y, seed=42)

    def maf(idx, X):
        mu = np.nanmean(X[keep][:, idx], axis=0) / 2
        return np.minimum(mu, 1 - mu)

    print(f"HBAU elongation, n={n}. Burden-product interaction p at increasing MAF floors:")
    print(f"{'MAF floor':>10}{'A SNPs':>8}{'D SNPs':>8}{'interaction p':>15}{'beta':>10}")
    rows = []
    for m in (0.01, 0.05, 0.10, 0.15, 0.20):
        iA = iA_all[maf(iA_all, XA) >= m]
        iD = iD_all[maf(iD_all, XD) >= m]
        if iA.size < 3 or iD.size < 3:
            print(f"{m:>10.2f}{iA.size:>8}{iD.size:>8}{'<3 SNP: n/a':>15}")
            continue
        bA = block_burden_capped(XA[keep][:, iA], np.arange(iA.size), 150, rng, minor=True)
        bD = block_burden_capped(XD[keep][:, iD], np.arange(iD.size), 150, rng, minor=True)
        p, beta = _interaction_p(bA, bD, Wh, y, n)
        print(f"{m:>10.2f}{iA.size:>8}{iD.size:>8}{p:>15.4g}{beta:>10.3f}")
        rows.append((m, int(iA.size), int(iD.size), p, beta))

    # fine match: lift the discovery burden SNP positions (CRI) to NDM8 and keep only HBAU SNPs there
    print("\nFine match -- HBAU burden restricted to the lifted discovery SNP positions:")
    blocks = ibs.build_liftover()
    g2s_c = {}
    for p in ("interact/snp_to_gene_A.npz", "interact/snp_to_gene_D.npz"):
        g2s_c.update(b4._load_snp_to_gene(DISC / p))
    cbimA = np.loadtxt(DISC / "A/all.bim", dtype=str, usecols=(0, 3))
    cbimD = np.loadtxt(DISC / "D/all.bim", dtype=str, usecols=(0, 3))
    hposA = {(str(c), int(p)) for c, p in zip(np.asarray(bedA.chrom).astype(str),
                                              np.asarray(bedA.pos, np.int64))}
    hposD = {(str(c), int(p)) for c, p in zip(np.asarray(bedD.chrom).astype(str),
                                              np.asarray(bedD.pos, np.int64))}

    def lifted_hbau_idx(disc_gene, cbim, g2s_disc, hpos, bed):
        cri_idx = np.asarray(g2s_disc[disc_gene], int)
        chrom = cbim[cri_idx, 0]
        cri_pos = cbim[cri_idx, 1].astype(np.int64)
        hit = []
        for c in np.unique(chrom):
            pos = np.sort(cri_pos[chrom == c])
            lift = ibs._lift_positions(blocks.get(c, []), pos)
            for cp, np_ in lift.items():
                if (c, np_) in hpos:
                    hit.append((c, np_))
        # map back to HBAU column index
        hp = {(str(cc), int(pp)): j for j, (cc, pp) in
              enumerate(zip(np.asarray(bed.chrom).astype(str), np.asarray(bed.pos, np.int64)))}
        return np.array([hp[k] for k in hit], int), len(cri_idx)

    jA, nA = lifted_hbau_idx(A_GENE_C, cbimA, g2s_c, hposA, bedA)
    jD, nD = lifted_hbau_idx(D_GENE_C, cbimD, g2s_c, hposD, bedD)
    print(f"  discovery A SNPs {nA} -> lifted+present in HBAU {jA.size}; "
          f"discovery D SNPs {nD} -> {jD.size}")
    if jA.size >= 3 and jD.size >= 3:
        bA = block_burden_capped(XA[keep][:, jA], np.arange(jA.size), 150, rng, minor=True)
        bD = block_burden_capped(XD[keep][:, jD], np.arange(jD.size), 150, rng, minor=True)
        p, beta = _interaction_p(bA, bD, Wh, y, n)
        print(f"  matched-variant interaction p = {p:.4g}  (beta {beta:.3f})")
    else:
        print("  too few lifted variants for a matched-set test (as expected when discovery used "
              "only 4 D-side common SNPs)")

    print("\nDISCOVERY reference: interaction p=1.23e-5 (CottonGVD, MAF>=0.1, 10 A / 4 D SNPs).")
    print("This is a post-hoc diagnostic; the primary reveal stays INCONCLUSIVE regardless.")


if __name__ == "__main__":
    main()
