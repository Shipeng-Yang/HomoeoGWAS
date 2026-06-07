#!/usr/bin/env python3
"""Task #2 (c): local LD / fine-mapping control for the wheat chr1 B-D interaction.

Pre-registered in reports/heb_causal_ladder_prereg.md (§4). Tests whether the chr1 B-D burden-product
interaction (days_to_emergence) survives conditioning on LOCAL additive structure, i.e. is NOT a
proxy for a strong neighbouring ADDITIVE locus. Uses the production interact engine's whitened-GLS
conditional diagnostic (`pair_conditional_diagnostics`): the interaction is tested in
``y_w ~ C + bB + bD + bB*bD`` for an increasing covariate block C (the interaction is already
conditional on the two focal main effects bB,bD in every model).

Conditional models (prereg §4):
  base : C = intercept (interaction | focal main effects)
  cond1: + focal A single-gene burden (the third homoeolog)
  cond2: + the STRONGEST neighbouring additive burden within +-1 Mb of the B and D hit genes
  cond3: + local burden PCs (chr1 B+D neighbouring-gene burdens, 80-90% variance)
  cond4: leave-block-out — drop the +-1 Mb local SNPs from the focal B,D burdens, refit
SURVIVE iff same sign AND p<0.05 AND <50% attenuation in the primary conditional (cond2/cond3).
Sensitivity windows: +-500 kb, +-2 Mb.
"""
from __future__ import annotations

import gzip
import json
import re
from pathlib import Path

import numpy as np

from homoeogwas.interact import (
    block_burden_capped,
    grm_from_X,
    pair_conditional_diagnostics,
    rank_int,
    whiten_multi,
)

ROOT = Path("/mnt/7302share/fast_ysp/U7_GWAS")
CHR1DIR = ROOT / "results/phase7/interact_validate/wheat_chr1"
GFF = ROOT / "data/reference/wheat/annotation/Triticum_aestivum.IWGSC.57.gff3.gz"
PHENO = ROOT / "data/processed/wheat/pheno_clean.tsv"
OUT = ROOT / "results/phase7/bio_wheat/heb_causal_ladder"
HIT = dict(A="TraesCS1A02G125400", B="TraesCS1B02G143800", D="TraesCS1D02G128400")
CAP = 150
_GID = re.compile(r"gene_id=([^;]+)")


def _load_sub(s):
    from homoeogwas.interact import _load_subgenome
    return _load_subgenome(str(CHR1DIR / s / "all"), str(CHR1DIR / f"snp_to_gene_{s}.npz"))


def _gene_starts(subchrom):
    """gene_id -> start bp for genes on a given chrom (e.g. '1B') from the IWGSC GFF (v1.1 ids)."""
    out = {}
    with gzip.open(GFF, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) < 9 or p[2] != "gene" or p[0] != subchrom:
                continue
            m = _GID.search(p[8])
            if m:
                out[m.group(1)] = int(p[3])
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("=== HEB causal ladder (c): local LD / fine-mapping control (chr1 B-D) ===")
    import pandas as pd

    sub = {s: _load_sub(s) for s in ("A", "B", "D")}
    samples = sub["A"].samples
    ph = pd.read_csv(PHENO, sep="\t").set_index("sample")
    trait = "days_to_emerg"
    valid = [s for s in samples if s in ph.index and pd.notna(ph.loc[s, trait])]
    sidx = np.array([samples.index(s) for s in valid])
    y = rank_int(np.array([float(ph.loc[s, trait]) for s in valid]))
    rng = np.random.default_rng(7)
    print(f"  n={len(valid)} samples")

    # whitener from the 3 chr1 subgenome GRMs (engine grm_from_X, restricted to valid samples)
    kernels = {s: grm_from_X(sub[s].X[sidx]) for s in ("A", "B", "D")}
    Wh, cv = whiten_multi(kernels, y, seed=42)

    def burden(s, g):
        return block_burden_capped(sub[s].X, sub[s].gene_snp[g], CAP, rng)[sidx]

    bA = burden("A", HIT["A"])
    bB = burden("B", HIT["B"])
    bD = burden("D", HIT["D"])

    def std(v):
        v = np.asarray(v, float)
        sd = v.std()
        return (v - v.mean()) / (sd if sd > 1e-12 else 1.0)

    bA, bB, bD = std(bA), std(bB), std(bD)
    n = len(valid)

    def diag(C):
        return pair_conditional_diagnostics(Wh, y, bB, bD, C=C)

    base = diag(None)
    p0, beta0 = base["interaction_p_wald"], base["interaction_beta"]
    print(f"  base B-D interaction: p={p0:.3g} beta={beta0:.4f} t={base['interaction_t']:.3f}")

    # neighbouring genes within +-W of the B and D hit start positions (GFF v1.1)
    startB = _gene_starts("1B")
    startD = _gene_starts("1D")
    posB0 = startB.get(HIT["B"])
    posD0 = startD.get(HIT["D"])

    def neighbours(s, starts, p0bp, W):
        """standardized burdens of genes within +-W bp (excl the hit), present in this subgenome."""
        cols, names = [], []
        for g, ps in starts.items():
            if g == HIT[s] or abs(ps - p0bp) > W or g not in sub[s].gene_snp:
                continue
            cols.append(std(burden(s, g)))
            names.append(g)
        return (np.column_stack(cols) if cols else np.empty((n, 0))), names

    # .bim SNP positions per subgenome (for local single-SNP / SNP-PC conditioning, Codex fix #5)
    def bim_pos(s):
        return np.array([int(ln.split("\t")[3]) for ln in (CHR1DIR / s / "all.bim").read_text().splitlines()])
    posSNP = {s: bim_pos(s) for s in ("B", "D")}

    def local_snps(s, pos0, W):
        """standardized (mean-imputed) local SNP dosage matrix within +-W bp of pos0 in subgenome s."""
        cols = np.where(np.abs(posSNP[s] - pos0) <= W)[0]
        if cols.size == 0:
            return np.empty((n, 0))
        M = sub[s].X[np.ix_(sidx, cols)].astype(float)
        mu = np.nanmean(M, 0)
        mu = np.where(np.isfinite(mu), mu, 0.0)
        M = np.where(np.isnan(M), mu, M)
        sd = M.std(0)
        Z = (M - mu) / np.where(sd > 1e-12, sd, 1.0)
        Z[:, sd <= 1e-12] = 0.0
        return Z

    yw_glob = Wh @ y
    one_w = Wh @ np.ones(n)

    def marginal_t(col):
        cj = Wh @ col
        X2 = np.column_stack([one_w, cj])
        bcoef, *_ = np.linalg.lstsq(X2, yw_glob, rcond=None)
        resid = yw_glob - X2 @ bcoef
        s2 = float(resid @ resid) / (n - 2)
        se = np.sqrt(max(s2 * np.linalg.pinv(X2.T @ X2)[1, 1], 1e-30))
        return abs(bcoef[1] / se)

    results = {}
    for label, W in (("primary_1Mb", 1_000_000), ("sens_500kb", 500_000), ("sens_2Mb", 2_000_000)):
        NB, nbB = neighbours("B", startB, posB0, W)
        ND, nbD = neighbours("D", startD, posD0, W)
        Nall = np.column_stack([NB, ND]) if (NB.size or ND.size) else np.empty((n, 0))
        n_neigh = Nall.shape[1]

        # cond1: + focal A burden
        C1 = np.column_stack([np.ones(n), bA])
        d1 = diag(C1)
        # cond2: + strongest neighbouring additive burden (max |whitened-GLS marginal t| vs y)
        cond2 = dict(skipped="no neighbours")
        if n_neigh:
            yw = Wh @ y
            tvals = []
            for j in range(n_neigh):
                cj = Wh @ Nall[:, j]
                X2 = np.column_stack([Wh @ np.ones(n), cj])
                bcoef, *_ = np.linalg.lstsq(X2, yw, rcond=None)
                resid = yw - X2 @ bcoef
                s2 = float(resid @ resid) / (n - 2)
                se = np.sqrt(max(s2 * np.linalg.pinv(X2.T @ X2)[1, 1], 1e-30))
                tvals.append(abs(bcoef[1] / se))
            jbest = int(np.argmax(tvals))
            C2 = np.column_stack([np.ones(n), bA, Nall[:, jbest]])
            d2 = diag(C2)
            cond2 = dict(p=d2["interaction_p_wald"], beta=d2["interaction_beta"], estimable=bool(d2["interaction_estimable"]),
                         strongest_neighbour_t=float(tvals[jbest]),
                         attenuation=float(1 - abs(d2["interaction_beta"]) / max(abs(beta0), 1e-12)))
        # cond3: + local burden PCs (80-90% variance of the neighbour burden matrix)
        cond3 = dict(skipped="no neighbours")
        if n_neigh:
            U, S, _ = np.linalg.svd(Nall - Nall.mean(0), full_matrices=False)
            varexp = np.cumsum(S ** 2) / np.sum(S ** 2)
            k = int(np.searchsorted(varexp, 0.85) + 1)
            k = min(k, max(n - 8, 1), n_neigh)
            PCs = (U[:, :k] * S[:k])
            C3 = np.column_stack([np.ones(n), bA, PCs])
            d3 = diag(C3)
            cond3 = dict(p=d3["interaction_p_wald"], beta=d3["interaction_beta"], estimable=bool(d3["interaction_estimable"]), n_local_pcs=k,
                         attenuation=float(1 - abs(d3["interaction_beta"]) / max(abs(beta0), 1e-12)))

        # cond5: + top local ADDITIVE SNPs (single-SNP, not gene burdens) near B & D, by |marginal t|
        SB, SD = local_snps("B", posB0, W), local_snps("D", posD0, W)
        Sall = np.column_stack([SB, SD]) if (SB.size or SD.size) else np.empty((n, 0))
        n_snp_local = Sall.shape[1]
        cond5 = dict(skipped="no local SNPs")
        cond6 = dict(skipped="no local SNPs")
        if n_snp_local:
            ts = np.array([marginal_t(Sall[:, j]) for j in range(n_snp_local)])
            ktop = min(10, n_snp_local)
            top = np.argsort(ts)[::-1][:ktop]
            C5 = np.column_stack([np.ones(n), bA, Sall[:, top]])
            d5 = diag(C5)
            cond5 = dict(p=d5["interaction_p_wald"], beta=d5["interaction_beta"], estimable=bool(d5["interaction_estimable"]), n_top_snps=int(ktop),
                         max_snp_marginal_t=float(ts.max()),
                         attenuation=float(1 - abs(d5["interaction_beta"]) / max(abs(beta0), 1e-12)))
            # cond6: + local SNP PCs (LD-pruned-equivalent: PCA of the local SNP matrix, 85% var)
            U2, S2, _ = np.linalg.svd(Sall - Sall.mean(0), full_matrices=False)
            ve = np.cumsum(S2 ** 2) / np.sum(S2 ** 2)
            ks = min(int(np.searchsorted(ve, 0.85) + 1), max(n - 8, 1), n_snp_local)
            C6 = np.column_stack([np.ones(n), bA, U2[:, :ks] * S2[:ks]])
            d6 = diag(C6)
            cond6 = dict(p=d6["interaction_p_wald"], beta=d6["interaction_beta"], estimable=bool(d6["interaction_estimable"]), n_snp_pcs=int(ks),
                         n_local_snps=int(n_snp_local),
                         attenuation=float(1 - abs(d6["interaction_beta"]) / max(abs(beta0), 1e-12)))

        results[label] = dict(window_bp=W, n_neighbours=n_neigh, n_neigh_B=NB.shape[1],
                              n_neigh_D=ND.shape[1], n_local_snps=n_snp_local,
                              cond1_plus_focalA=dict(p=d1["interaction_p_wald"], beta=d1["interaction_beta"]),
                              cond2_plus_strongest_neighbour=cond2, cond3_plus_local_pcs=cond3,
                              cond5_plus_top_local_snps=cond5, cond6_plus_local_snp_pcs=cond6)
        print(f"  [{label}] neigh={n_neigh} snp={n_snp_local} | cond1 p={d1['interaction_p_wald']:.3g} | "
              + (f"cond2 p={cond2['p']:.3g} | cond3 p={cond3['p']:.3g} | " if n_neigh else "")
              + (f"cond5(topSNP) p={cond5['p']:.3g} att={cond5['attenuation']:+.2f} | "
                 f"cond6(snpPC k={cond6['n_snp_pcs']}) p={cond6['p']:.3g} att={cond6['attenuation']:+.2f}"
                 if n_snp_local else "no local SNPs"))

    # leave-block-out note: leave-block-out is not a valid diagnostic for the focal burden itself (the
    # focal genes ARE in the block; a focal burden with its SNPs removed is undefined). Instead we test
    # whether the surrounding LOCAL ADDITIVE STRUCTURE explains the interaction, via neighbour burdens
    # (cond2), local burden PCs (cond3), top local single SNPs (cond5) and local SNP PCs (cond6).
    cond4 = dict(note=("leave-block-out is not a valid diagnostic for the focal burden (focal genes are "
                       "in the block); local additive structure is instead controlled by cond2 (neighbour "
                       "burdens), cond3 (local burden PCs), cond5 (top local single SNPs) and cond6 (local "
                       "SNP PCs)."))

    # survive verdict on the primary 1Mb window (now incl. single-SNP and SNP-PC conditioning)
    pr = results["primary_1Mb"]
    c2 = pr["cond2_plus_strongest_neighbour"]
    c3 = pr["cond3_plus_local_pcs"]
    c5 = pr["cond5_plus_top_local_snps"]
    c6 = pr["cond6_plus_local_snp_pcs"]
    def ok(c):
        return bool(c.get("estimable", True) and c.get("p", 1) < 0.05
                    and np.sign(c.get("beta", 0)) == np.sign(beta0) and c.get("attenuation", 1) < 0.5)
    conds = [c2, c3, c5, c6]
    verdict = ("SURVIVE" if all(ok(c) for c in conds) else
               "PARTLY_LD_CORRELATED" if any(c.get("p", 1) < 0.05 for c in conds) else "FAIL")
    print(f"  VERDICT(primary 1Mb, incl. single-SNP+SNP-PC) = {verdict}")

    payload = dict(task="heb_causal_ladder_c", trait=trait, n=n, prereg="reports/heb_causal_ladder_prereg.md",
                   base=dict(p=p0, beta=beta0, t=base["interaction_t"]),
                   windows=results, cond4_leave_block_out=cond4, verdict_primary_1Mb=verdict,
                   note=("interaction is conditional on focal bB,bD in every model; survive = same sign "
                         "& p<0.05 & <50% attenuation under local conditioning => B-D interaction is "
                         "not a neighbouring-additive proxy. chr5 unaffected (separate positive)."))
    fp = OUT / "heb_causal_c.json"
    fp.write_text(json.dumps(payload, indent=2, default=float))
    print(f"  wrote {fp}")


if __name__ == "__main__":
    main()
