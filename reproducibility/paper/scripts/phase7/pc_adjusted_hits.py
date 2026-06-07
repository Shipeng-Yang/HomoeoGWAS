#!/usr/bin/env python
"""H2 — PC-adjusted reproduction of the cotton/wheat homoeolog-pair hits.

Reviewer question answered: *do the discoveries survive standard genotype-PC adjustment, and
are they still "pair-only" (single-gene marginal null) once population structure enters the fixed
effects?* For each known hit we recompute, with the PRODUCTION engine
(``homoeogwas.interact``), the per-pair conditional diagnostics under n_pcs in {0, 5, 10}:
  - interaction Wald p (the headline hit p) and whether it is still estimable;
  - VIF / R2(int|marginals) (orthogonality of the product to the main effects);
  - single-gene burden-GLS marginal p (the 'single-gene-invisible' evidence).
PCs enter BOTH the subgenome-stratified null-LMM whitener mean model AND the per-pair GLS design;
n_pcs=0 reproduces the published covariate-free numbers bit-exactly. We additionally run one full
genome-context scan per species at n_pcs=10 with permutation to show genome-wide calibration
(lambda_perm) is preserved under PC adjustment (Freedman-Lane covariate-aware permutation).

Output: results/phase7/pc_adjusted_hits.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import homoeogwas.interact as I

ROOT = Path("/mnt/7302share/fast_ysp/U7_GWAS")
OUT = ROOT / "results/phase7/pc_adjusted_hits.json"
PC_LEVELS = [0, 5, 10]


def _load_pheno(path, sample_col, trait, samples):
    ph = pd.read_csv(ROOT / path, sep="\t").set_index(sample_col)
    valid = [s for s in samples if s in ph.index and pd.notna(ph.loc[s, trait])]
    sidx = np.array([samples.index(s) for s in valid])
    y_raw = np.array([float(ph.loc[s, trait]) for s in valid])
    return valid, sidx, y_raw


def _burden(sd, gene, sidx, cap, rng):
    return I.block_burden_capped(sd.X, sd.gene_snp[gene], cap, rng)[sidx]


def diagnostics_vs_pcs(subdata, subs, sidx, y_raw, gx, gy, sx, sy, grm_method, cap=150):
    """Per-hit conditional diagnostics across PC levels (whitener {all subgenomes}, burdens sx/sy)."""
    rng = np.random.default_rng(7)
    kernels = {s: I._build_grm(subdata[s], sidx, grm_method, 0.01) for s in subs}
    n_t = sidx.size
    bx = I.scols_safe(_burden(subdata[sx], gx, sidx, cap, rng).reshape(-1, 1)).ravel()
    by = I.scols_safe(_burden(subdata[sy], gy, sidx, cap, rng).reshape(-1, 1)).ravel()
    y = I.rank_int(y_raw)                                # INT primary
    rows = {}
    for k in PC_LEVELS:
        C, meta = I.build_covariate_block(kernels, n_t, n_pcs=k)
        Wh, cv = I.whiten_multi(kernels, y, X=C, seed=42)
        d = I.pair_conditional_diagnostics(Wh, y, bx, by, C=C)
        rows[f"n_pcs={k}"] = dict(
            interaction_p=d["interaction_p_wald"], interaction_t=d["interaction_t"],
            estimable=d["interaction_estimable"], n_covariates=d["n_covariates"],
            vif_int=d["vif_int"], r2_int_given_marginals=d["r2_int_given_marginals"],
            marginal_p_X=d["single_gene_marginal"]["p_X_alone"],
            marginal_p_Y=d["single_gene_marginal"]["p_Y_alone"],
            covariate_policy=meta["policy"])
    return rows


def main():
    t0 = time.time()
    out = {"tool": "homoeogwas", "analysis": "pc_adjusted_hits", "pc_levels": PC_LEVELS,
           "note": ("PCs enter the null-LMM whitener mean model AND the per-pair GLS design; "
                    "n_pcs=0 reproduces the covariate-free published numbers bit-exactly; "
                    "permutation under PCs is Freedman-Lane (residualize on C, permute residuals)."),
           "hits": {}}

    # ---- cotton Hit2 (PRIMARY) length_uniformity A11|D11 flank_same + Hit1 fiber A06|D06 body_same
    def cot_sub(npz):
        return {"A": I._load_subgenome("data/processed/cotton/A/all", str(ROOT / npz)),
                "D": I._load_subgenome("data/processed/cotton/D/all", str(ROOT / npz))}

    cot_body = cot_sub("results/phase7/bio_full/snp_to_gene_body.npz")
    samples_c = cot_body["A"].samples
    v, sidx, y = _load_pheno("data/processed/cotton/pheno_m3_4_blue.tsv", "sample",
                             "fiber_length_BLUE", samples_c)
    out["hits"]["cotton_Hit1_fiber_A06_D06_body"] = dict(
        species="cotton", role="secondary", trait="fiber_length_BLUE", n=len(v),
        diagnostics=diagnostics_vs_pcs(cot_body, ["A", "D"], sidx, y,
                                       "GhM_A06G1605", "GhM_D06G1557", "A", "D", "compute_grm_maf"))
    print(f"  cotton Hit1 done ({time.time()-t0:.0f}s)", flush=True)

    cot_flank = cot_sub("results/phase7/bio_full/snp_to_gene_flank2000bp.npz")
    v2, sidx2, y2 = _load_pheno("data/processed/cotton/pheno_m3_4_blue.tsv", "sample",
                                "length_uniformity_BLUE", cot_flank["A"].samples)
    out["hits"]["cotton_Hit2_length_uniformity_A11_D11_flank"] = dict(
        species="cotton", role="PRIMARY", trait="length_uniformity_BLUE", n=len(v2),
        diagnostics=diagnostics_vs_pcs(cot_flank, ["A", "D"], sidx2, y2,
                                       "GhM_A11G2420", "GhM_D11G2742", "A", "D", "compute_grm_maf"))
    print(f"  cotton Hit2 done ({time.time()-t0:.0f}s)", flush=True)

    # ---- wheat chr1 B-D (hexaploid whitener {K_A,K_B,K_D}, burdens B/D)
    wh = {s: I._load_subgenome(f"results/phase7/interact_validate/wheat_chr1/{s}/all",
                               str(ROOT / f"results/phase7/interact_validate/wheat_chr1/snp_to_gene_{s}.npz"))
          for s in ["A", "B", "D"]}
    vw, sidxw, yw = _load_pheno("data/processed/wheat/pheno_clean.tsv", "sample",
                                "days_to_emerg", wh["A"].samples)
    out["hits"]["wheat_chr1_BD_TraesCS1B02G143800_1D02G128400"] = dict(
        species="wheat", role="PRIMARY", trait="days_to_emerg", n=len(vw),
        diagnostics=diagnostics_vs_pcs(wh, ["A", "B", "D"], sidxw, yw,
                                       "TraesCS1B02G143800", "TraesCS1D02G128400", "B", "D",
                                       "grm_from_X"))
    print(f"  wheat BD done ({time.time()-t0:.0f}s)", flush=True)

    # ---- genome-context calibration under PC10: full scans with permutation (lambda_perm)
    out["calibration_under_pc10"] = {}
    pairs_flank = I._load_pairs(str(ROOT / "results/phase7/bio_full/pairs_flank_same.tsv"), ["A", "D"])
    r = I.run_pair_scan(cot_flank, pairs_flank, y2, sidx2, cap=150, transform="INT", perm_B=200,
                        n_jobs=16, pair_subs=("A", "D"), grm_method="compute_grm_maf",
                        covariates={"n_pcs": 10})
    out["calibration_under_pc10"]["cotton_flank_same"] = dict(
        G=r.G, pair_acat=r.pair_acat, pair_acat_emp=r.pair_acat_emp, min_p=r.min_p,
        lambda_gc_obs=r.lambda_gc_obs, lambda_gc_perm_median=r.lambda_gc_perm_median,
        n_sig=r.n_sig, bonferroni_alpha=r.bonferroni_alpha, covariates=r.covariates,
        hit_in_sig=any(tuple(h["pair"]) == ("GhM_A11G2420", "GhM_D11G2742") for h in r.sig))
    print(f"  cotton flank PC10 scan done ({time.time()-t0:.0f}s)", flush=True)

    triads = I._load_pairs(str(ROOT / "results/phase7/interact_validate/wheat_chr1/triads_chr1.tsv"),
                           ["A", "B", "D"])
    rt = I.run_triad_scan(wh, triads, yw, sidxw, cap=150, transform="INT", perm_B=100, n_jobs=16,
                          grm_method="grm_from_X", covariates={"n_pcs": 10})
    bd = rt["pairwise"]["BD"]
    out["calibration_under_pc10"]["wheat_chr1_triad"] = dict(
        G=rt["G"], triad_acat_omnibus=rt["triad_acat_omnibus"],
        triad_acat_omnibus_emp=rt["triad_acat_omnibus_emp"],
        BD_acat=bd["acat"], BD_min_p=bd["min_p"], BD_n_sig=bd["n_sig"],
        lambda_gc_perm_median=rt["lambda_gc_perm_median"], covariates=rt["covariates"],
        hit_in_BD_sig=any(h["triad"][1] == "TraesCS1B02G143800" for h in bd["sig"]))
    print(f"  wheat triad PC10 scan done ({time.time()-t0:.0f}s)", flush=True)

    out["runtime_sec"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(out, indent=2, default=float))
    print(f"✅ wrote {OUT} ({out['runtime_sec']}s)")


if __name__ == "__main__":
    main()
