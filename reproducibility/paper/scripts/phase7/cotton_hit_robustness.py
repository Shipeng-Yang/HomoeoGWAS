#!/usr/bin/env python3
"""P1: robustness hardening for the two cotton interaction hits (sample-LOO / MAF / SNP-LOO).

Targets reviewer attacks on the cotton AADD burden-product hits (n=419):
  Hit1 fiber_length     GhM_A06G1605 (body, 14 SNP) x GhM_D06G1557 (3 SNP)   p=2.14e-4
  Hit2 length_uniformity GhM_A11G2420 (flank,12 SNP) x GhM_D11G2742 (4 SNP)   p=4.75e-5
All hit-gene SNPs are common (MAF >= 0.051): the signal is not a rare-variant artifact.

Per hit, INT primary, reusing the production whitened GLS (pair_conditional_diagnostics):
1. SAMPLE LOO (primary, full pipeline): drop each accession -> subset {K_A,K_D} + trace-renorm,
   recompute INT on the 418, restandardize burdens on the 418, re-fit REML + re-whiten, re-run the
   GLS interaction. "Would the hit survive if this accession were absent?" Reports p_(-i)/beta_(-i)
   distribution, fraction still < alpha/G, sign stability, REML convergence/boundary, failed folds.
2. SAMPLE LOO (secondary, conditional GLS influence): keep the full-sample V_hat (fixed variance
   components) and full INT/burden scaling; for fold i use the EXACT subset covariance block
   V_hat[-i,-i] (NOT whitened-row deletion, which is invalid). DFBETA = beta_full - beta_(-i).
3. MAF sensitivity: SNP-MAF floor in {0, 0.05, 0.10}; rebuild burden, fixed full whitener; report
   p + retained SNP count (floor 0.10 may leave 0 SNPs -> "not estimable", not "lost signal").
4. LEAVE-ONE-SNP-OUT per gene + within-hit single-SNP-pair interaction scan (TOP_RISK: a 3-4-SNP
   D-gene "burden interaction" could be one SNP-pair in disguise).
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path("/mnt/7302share/fast_ysp/U7_GWAS")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/phase7"))
import d2_arm2_synteny_kernel as d2  # noqa: E402

from homoeogwas.interact import pair_conditional_diagnostics  # noqa: E402


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


b4 = _load(ROOT / "scripts/phase7/bio_pilot/04b_multi_trait_deploy.py", "b4")
OUT = ROOT / "results/phase7/cotton_hit_robustness.json"

HITS = [
    dict(key="Hit1_fiber_length", trait="fiber_length_BLUE", mode="body",
         gene_A="GhM_A06G1605", gene_D="GhM_D06G1557", G=125, expected_p=2.136e-4),
    dict(key="Hit2_length_uniformity", trait="length_uniformity_BLUE", mode="flank",
         gene_A="GhM_A11G2420", gene_D="GhM_D11G2742", G=372, expected_p=4.747e-5),
]


def _std1(v):
    v = np.asarray(v, float)
    sd = v.std(ddof=0)
    return (v - v.mean()) / (sd if sd > 1e-12 else 1.0)


def _burden(X, idx, rows=None):
    """Standardized gene burden (mean of column-standardized dosages) over `rows` (default all)."""
    Xr = X if rows is None else X[rows]
    return _std1(d2._block_burden(Xr, idx))


def _whiten_from_V(Vsub):
    w, Q = np.linalg.eigh(0.5 * (Vsub + Vsub.T))
    w = np.clip(w, 1e-10, None)
    return (Q * (1.0 / np.sqrt(w))) @ Q.T


def _maf(X, idx):
    mu = np.nanmean(X[:, idx], axis=0) / 2.0
    return np.minimum(mu, 1.0 - mu)


def harden(hit, D, samples, pheno, g2s):
    XA, XD, KA, KD, n = D["XA"], D["XD"], D["KA"], D["KD"], D["n"]
    idxA = np.asarray(g2s[hit["mode"]][hit["gene_A"]], int)
    idxD = np.asarray(g2s[hit["mode"]][hit["gene_D"]], int)
    y_raw = pheno[hit["trait"]]
    bonf = 0.05 / hit["G"]

    # ---- full-sample reference (reproduce the hit) ----
    yA = b4._rank_int(y_raw)
    bxA = _burden(XA, idxA)
    byA = _burden(XD, idxD)
    Wh_full, cv = b4._whiten(KA, KD, yA, seed=42)
    d_full = pair_conditional_diagnostics(Wh_full, yA, bxA, byA)
    p_full = d_full["interaction_p_wald"]
    beta_full = d_full["interaction_beta"]
    V_full = cv.get("A", 0.0) * KA + cv.get("D", 0.0) * KD + max(cv.get("e", 1e-6), 1e-6) * np.eye(n)

    # ---- 1. primary full-pipeline sample LOO ----
    p_loo, beta_loo, nfail, nbound = [], [], 0, 0
    for i in range(n):
        keep = np.delete(np.arange(n), i)
        KAi = KA[np.ix_(keep, keep)]
        KAi = KAi / (np.trace(KAi) / keep.size)
        KDi = KD[np.ix_(keep, keep)]
        KDi = KDi / (np.trace(KDi) / keep.size)
        yi = b4._rank_int(y_raw[keep])
        bxi = _burden(XA, idxA, keep)
        byi = _burden(XD, idxD, keep)
        try:
            Whi, cvi = b4._whiten(KAi, KDi, yi, seed=42)
            di = pair_conditional_diagnostics(Whi, yi, bxi, byi)
            p_loo.append(di["interaction_p_wald"])
            beta_loo.append(di["interaction_beta"])
            if min(cvi.get("A", 0.0), cvi.get("D", 0.0), cvi.get("e", 0.0)) <= 1e-8:
                nbound += 1
        except Exception:  # noqa: BLE001
            nfail += 1
    p_loo = np.array(p_loo)
    beta_loo = np.array(beta_loo)
    primary = dict(
        n_folds=int(p_loo.size), n_failed=int(nfail), n_boundary_vc=int(nbound),
        p_min=float(p_loo.min()), p_median=float(np.median(p_loo)), p_max=float(p_loo.max()),
        frac_below_alpha_over_G=float((p_loo < bonf).mean()),
        frac_below_0p01=float((p_loo < 0.01).mean()),
        any_fold_p_gt_0p01=bool((p_loo > 0.01).any()),
        beta_sign_stable=bool(np.all(np.sign(beta_loo) == np.sign(beta_full))),
        beta_min=float(beta_loo.min()), beta_max=float(beta_loo.max()))

    # ---- 2. conditional GLS influence LOO (fixed V_hat, exact subset block) ----
    p_cond, dfbeta = [], []
    for i in range(n):
        keep = np.delete(np.arange(n), i)
        Whi = _whiten_from_V(V_full[np.ix_(keep, keep)])
        di = pair_conditional_diagnostics(Whi, yA[keep], bxA[keep], byA[keep])
        p_cond.append(di["interaction_p_wald"])
        dfbeta.append(beta_full - di["interaction_beta"])
    p_cond = np.array(p_cond)
    dfbeta = np.array(dfbeta)
    amax = int(np.argmax(np.abs(dfbeta)))
    conditional = dict(
        p_min=float(p_cond.min()), p_median=float(np.median(p_cond)), p_max=float(p_cond.max()),
        frac_below_alpha_over_G=float((p_cond < bonf).mean()),
        max_abs_dfbeta=float(np.abs(dfbeta).max()),
        max_dfbeta_sample=str(samples[amax]),
        dfbeta_robust_z_of_max=float(np.abs(dfbeta[amax] - np.median(dfbeta))
                                     / (1.4826 * np.median(np.abs(dfbeta - np.median(dfbeta))) + 1e-30)))

    # ---- 3. MAF sensitivity ----
    mafA, mafD = _maf(XA, idxA), _maf(XD, idxD)
    maf_rows = []
    for floor in (0.0, 0.05, 0.10):
        kA = idxA[mafA >= floor]
        kD = idxD[mafD >= floor]
        if kA.size == 0 or kD.size == 0:
            maf_rows.append(dict(maf_floor=floor, n_snp_A=int(kA.size), n_snp_D=int(kD.size),
                                 p="not_estimable"))
            continue
        bx = _burden(XA, kA)
        by = _burden(XD, kD)
        dm = pair_conditional_diagnostics(Wh_full, yA, bx, by)
        maf_rows.append(dict(maf_floor=floor, n_snp_A=int(kA.size), n_snp_D=int(kD.size),
                             p=float(dm["interaction_p_wald"])))
    maf_block = dict(
        maf_A=dict(min=float(mafA.min()), median=float(np.median(mafA)), max=float(mafA.max())),
        maf_D=dict(min=float(mafD.min()), median=float(np.median(mafD)), max=float(mafD.max())),
        all_common_ge_0p05=bool(mafA.min() >= 0.05 and mafD.min() >= 0.05),
        floors=maf_rows)

    # ---- 4a. leave-one-SNP-out per gene (fixed full whitener) ----
    def _loso(idx_full, X_drop, fixed_burden, drop_is_A):
        out = []
        for j in range(idx_full.size):
            sub = np.delete(idx_full, j)
            if sub.size == 0:
                out.append(dict(dropped_snp=int(idx_full[j]), p="not_estimable"))
                continue
            bsub = _burden(X_drop, sub)
            bx, by = (bsub, fixed_burden) if drop_is_A else (fixed_burden, bsub)
            dd = pair_conditional_diagnostics(Wh_full, yA, bx, by)
            out.append(dict(dropped_snp=int(idx_full[j]), n_snp_remaining=int(sub.size),
                            p=float(dd["interaction_p_wald"])))
        return out
    loso_A = _loso(idxA, XA, byA, True)
    loso_D = _loso(idxD, XD, bxA, False)
    loso_pA = [r["p"] for r in loso_A if isinstance(r["p"], float)]
    loso_pD = [r["p"] for r in loso_D if isinstance(r["p"], float)]

    # ---- 4b. within-hit single-SNP-pair interaction scan ----
    # single-SNP burdens via the NaN-safe path (_burden mean-imputes missing genotypes)
    pair_ps = []
    for ja in idxA:
        bxj = _burden(XA, np.array([ja]))
        for jd in idxD:
            byj = _burden(XD, np.array([jd]))
            dp = pair_conditional_diagnostics(Wh_full, yA, bxj, byj)
            pair_ps.append(dict(snp_A=int(ja), snp_D=int(jd), p=float(dp["interaction_p_wald"])))
    pair_p_arr = np.array([r["p"] for r in pair_ps])
    best_pair = pair_ps[int(np.argmin(pair_p_arr))]

    # ---- verdict (3-level) ----
    # CORE robustness (must all hold). Note: the conditional-LOO DFBETA distribution is sharply
    # peaked at 0, so a robust-z of the max is always "extreme" and is NOT a useful gate — the
    # meaningful influence metric is |DFBETA| as a FRACTION of |beta_full| (reported below).
    dfbeta_frac = float(conditional["max_abs_dfbeta"] / (abs(beta_full) + 1e-30))
    best_pair_ratio = float(pair_p_arr.min() / max(p_full, 1e-300))
    core = dict(
        loo_ge90pct_sig=primary["frac_below_alpha_over_G"] >= 0.90,
        loo_never_nominal_null=primary["p_max"] < 0.05,
        beta_sign_stable=primary["beta_sign_stable"],
        maf_common_driven=maf_block["all_common_ge_0p05"],
        loso_no_single_snp_collapse=max(loso_pA + loso_pD) < 0.05,
    )
    caveats = {}
    # influence caveat = one accession MEANINGFULLY shifts beta (>=25%) OR weakens the worst LOO
    # fold toward null (p>0.01). Losing strict alpha/G in a rare fold while still p<<0.01 is
    # expected jitter, not influence sensitivity, so it does NOT trigger the caveat.
    if dfbeta_frac >= 0.25 or primary["p_max"] > 0.01:
        caveats["influence_sensitive"] = (
            f"max |DFBETA| = {dfbeta_frac:.0%} of beta and worst LOO p = {primary['p_max']:.2g} "
            f"(alpha/G = {bonf:.1e}): one accession ({conditional['max_dfbeta_sample']}) "
            "meaningfully shifts the estimate / weakens it toward nominal.")
    if best_pair_ratio <= 3.0:
        caveats["single_pair_concentration"] = (
            f"best single SNP-pair p ({pair_p_arr.min():.2g}) is within {best_pair_ratio:.1f}x of "
            f"the burden p ({p_full:.2g}): consistent with substantial single-pair concentration "
            "(the burden interaction is close to a one homoeologous-SNP-pair signal rather than a "
            "broadly distributed multi-SNP gene-level burden interaction).")
    if not all(core.values()):
        verdict = "REVIEW"
    elif caveats:
        verdict = "PASS_WITH_CAVEAT"
    else:
        verdict = "PASS"

    return dict(
        key=hit["key"], trait=hit["trait"], genes=[hit["gene_A"], hit["gene_D"]],
        n=int(n), G=int(hit["G"]), alpha_over_G=float(bonf),
        full_sample=dict(interaction_p=float(p_full), interaction_beta=float(beta_full),
                         reproduced_rel_err=float(abs(p_full - hit["expected_p"]) / hit["expected_p"])),
        sample_loo_primary=primary, sample_loo_conditional=conditional, maf_sensitivity=maf_block,
        leave_one_snp_out=dict(
            gene_A=dict(p_min=float(min(loso_pA)), p_max=float(max(loso_pA)), folds=loso_A),
            gene_D=dict(p_min=float(min(loso_pD)), p_max=float(max(loso_pD)), folds=loso_D),
            note="drop one SNP from one gene's burden, keep the partner gene's full burden"),
        single_snp_pair_scan=dict(
            n_pairs=len(pair_ps), p_min=float(pair_p_arr.min()),
            n_pairs_below_alpha_over_G=int((pair_p_arr < bonf).sum()),
            best_pair_to_burden_ratio=best_pair_ratio, best_pair=best_pair,
            note=("if best single SNP-pair p ~ the burden p, the 'gene burden interaction' is "
                  "effectively one SNP-pair; if the burden p is notably smaller, multiple SNPs "
                  "contribute (genuine gene-level burden interaction).")),
        dfbeta_frac_of_beta=dfbeta_frac, core_criteria=core, caveats=caveats, verdict=verdict)


def main():
    t0 = time.time()
    print("=== P1 cotton hit robustness (sample-LOO / MAF / SNP-LOO) ===")
    cfg = d2.PANELS["cotton_hebau"]
    D = d2._load(cfg)
    samples = b4._load_pheno_aligned(cfg)
    pheno = b4._load_pheno_all(cfg, samples)
    g2s = {"body": b4._load_snp_to_gene(ROOT / "results/phase7/bio_full/snp_to_gene_body.npz"),
           "flank": b4._load_snp_to_gene(ROOT / "results/phase7/bio_full/snp_to_gene_flank2000bp.npz")}
    results = []
    for hit in HITS:
        print(f"--- {hit['key']} ---", flush=True)
        r = harden(hit, D, samples, pheno, g2s)
        results.append(r)
        pr = r["sample_loo_primary"]
        print(f"  full p={r['full_sample']['interaction_p']:.3g} | LOO p[{pr['p_min']:.2g},"
              f"{pr['p_max']:.2g}] {100*pr['frac_below_alpha_over_G']:.0f}%<a/G "
              f"sign_stable={pr['beta_sign_stable']} | cond max|DFBETA|="
              f"{r['sample_loo_conditional']['max_abs_dfbeta']:.3g} | "
              f"best_single_pair_p={r['single_snp_pair_scan']['p_min']:.3g} | "
              f"VERDICT={r['verdict']} ({time.time()-t0:.0f}s)", flush=True)
    payload = dict(analysis="cotton_hit_robustness", date="2026-06-02", n_samples=int(D["n"]),
                   method=__doc__, hits=results)
    OUT.write_text(json.dumps(payload, indent=2, default=float))
    print(f"\nwrote {OUT} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
