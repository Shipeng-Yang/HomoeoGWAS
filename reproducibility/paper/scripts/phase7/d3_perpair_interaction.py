#!/usr/bin/env python3
"""Phase 7 D3 — per-homoeolog-pair targeted interaction test (candidate positive method).

D1/D2 showed dense omnibus burden kernels (global Hadamard K_hom, synteny-window
K_pair) cannot detect SPARSE gene-scale homoeolog interaction: the few causal
features are diluted among G null features (oracle true-feature kernel had
power~1.0, so the signal exists — the failure is the dense-aggregation framing).

Pivot: TEST EACH homoeolog pair individually and aggregate, exploiting
synteny to keep the test count ~G (not exhaustive M_A×M_D).

Method:
  per replicate:
    1. null LMM y ~ 1 + u + e, u~N(0, σ²_A K_A + σ²_D K_D)  (fit_multi_reml)
       -> V = σ²_A K_A + σ²_D K_D + σ²_e I ; whitening W = V^{-1/2}.
    2. per homoeolog pair g: GLS via whitening — OLS of W·y on W·[1, b_A,g, b_D,g,
       b_A,g·b_D,g]; t-test the INTERACTION coef (main effects retained -> guards
       against regional additive masquerading as interaction).
    3. aggregate p_g: ACAT (omnibus "any homoeolog interaction?") + minP/Bonferroni
       (localization "which pairs").
  Compare on identical sparse gene-block sims: per-pair ACAT vs global K_hom LRT
  vs omnibus K_pair LRT; wrong-pair (within-chrom shuffled D) negative control.
  Empirical null (PVE=0) sets calibrated thresholds.

Reuses the proven D2 arm-2 helpers (burdens, pairing, causal sim, kernels).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from d2_arm2_synteny_kernel import (  # noqa: E402
    ADDITIVE_PVE, MIN_SNP_PER_WIN, N_CAUSAL, PANELS,
    _block_burden, _causal_features, _chol, _load, _pair_kernel,
    _scale_vec, _shuffled_pair_kernel, _burden_matrix,
)
from homoeogwas.kernel import build_homoeolog_kernel, normalize_kernel  # noqa: E402
from homoeogwas.lmm import fit_multi_reml  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PVE_GRID = [0.05, 0.10, 0.20]


def _fixed_bins(X, chrom, pos, chrom_fmt, sub, n_chrom, bin_size):
    """Fixed-SNP-count bins (~bin_size SNPs) per chromosome; burden = mean std dosage.
    Returns (B n×nbins, labels[(chrom,rank)], win_snp[idx arrays], nbins_per_chrom)."""
    from d2_arm2_synteny_kernel import _scale_cols
    Xs = _scale_cols(X)
    cols, labels, win_snp, nbins = [], [], [], {}
    for k in range(1, n_chrom + 1):
        cn = chrom_fmt.format(sub=sub, k=k)
        idx = np.where(chrom == cn)[0]
        order = idx[np.argsort(pos[idx])]
        nb = max(1, int(round(len(order) / bin_size)))
        splits = np.array_split(order, nb)
        nbins[k] = len(splits)
        for r, sidx in enumerate(splits):
            cols.append(Xs[:, sidx].mean(1)); labels.append((cn, r)); win_snp.append(sidx)
    return np.column_stack(cols), labels, win_snp, nbins


def _rank_pair(labA, nbinsA, labD, nbinsD, n_chrom, chrom_fmt, subA, subD):
    """Pair A-bin (chrom k, rank i) with D-bin (chrom k, round(i·(nD-1)/(nA-1)))."""
    def idx_map(labels):
        m = {}
        for j, (cn, r) in enumerate(labels):
            m[(cn, r)] = j
        return m
    mA, mD = idx_map(labA), idx_map(labD)
    pairs = []
    for k in range(1, n_chrom + 1):
        cnA = chrom_fmt.format(sub=subA, k=k); cnD = chrom_fmt.format(sub=subD, k=k)
        nA, nD = nbinsA.get(k, 0), nbinsD.get(k, 0)
        if nA == 0 or nD == 0:
            continue
        for i in range(nA):
            rD = int(round(i * (nD - 1) / max(nA - 1, 1)))
            pairs.append((mA[(cnA, i)], mD[(cnD, rD)]))
    return pairs


def _acat(pvals):
    """Cauchy combination of p-values (robust to dependence)."""
    p = np.clip(np.asarray(pvals, float), 1e-15, 1 - 1e-15)
    t = np.mean(np.tan((0.5 - p) * np.pi))
    return float(0.5 - np.arctan(t) / np.pi)


_CHI2_MED1 = stats.chi2.ppf(0.5, 1)  # 0.4549364... median of 1-df chi-square


def _lambda_gc(pvals):
    """Genomic-control inflation factor from two-sided t-test p-values.

    Map p -> equivalent 1-df chi-square (z = Phi^{-1}(1-p/2), chi2 = z^2); do NOT
    use t^2/0.4549 since finite-sample t^2 ~ F(1, n-4), not chi2_1.
    lambda = median(chi2_obs) / qchisq(0.5, 1).
    """
    p = np.clip(np.asarray(pvals, float), 1e-300, 1.0)
    chi2 = stats.norm.isf(p / 2.0) ** 2
    chi2 = chi2[np.isfinite(chi2)]
    if chi2.size == 0:
        return float("nan")
    return float(np.median(chi2) / _CHI2_MED1)


def _whiten(KA, KD, y, X0):
    """V^{-1/2} from null LMM {K_A,K_D}; return (W, ok)."""
    res = fit_multi_reml(y, X0, {"A": KA, "D": KD}, n_starts=3, random_state=1)
    cv = res.component_var
    n = KA.shape[0]
    V = cv.get("A", 0.0) * KA + cv.get("D", 0.0) * KD + max(cv.get("e", 1e-6), 1e-6) * np.eye(n)
    w, Q = np.linalg.eigh(0.5 * (V + V.T))
    w = np.clip(w, 1e-10, None)
    return (Q * (1.0 / np.sqrt(w))) @ Q.T


def _perpair_pvals(Wh, y, BA_m, BD_m):
    """GLS interaction p-value per homoeolog pair (whitened OLS, t-test on b_A·b_D)."""
    n, G = BA_m.shape
    yw = Wh @ y
    one_w = Wh @ np.ones(n)
    BAw = Wh @ BA_m; BDw = Wh @ BD_m
    INTw = Wh @ (BA_m * BD_m)
    df = n - 4
    pv = np.empty(G)
    for g in range(G):
        Xw = np.column_stack([one_w, BAw[:, g], BDw[:, g], INTw[:, g]])
        beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
        resid = yw - Xw @ beta
        s2 = float(resid @ resid) / df
        XtX_inv = np.linalg.pinv(Xw.T @ Xw)
        se = np.sqrt(max(s2 * XtX_inv[3, 3], 1e-30))
        t = beta[3] / se
        pv[g] = 2.0 * stats.t.sf(abs(t), df)
    return pv


def _one_rep(KA, KD, K_hom, K_pair, cholA, cholD, BA_m, BD_m, BA_sh, BD_sh, Uc, pve, seed):
    rng = np.random.default_rng(seed)
    n = KA.shape[0]
    sa = ADDITIVE_PVE / 2
    y = np.sqrt(sa) * (cholA @ rng.standard_normal(n)) + np.sqrt(sa) * (cholD @ rng.standard_normal(n))
    if pve > 0:
        y += _scale_vec(Uc @ rng.standard_normal(Uc.shape[1])) * np.sqrt(pve)
    y += np.sqrt(max(1 - ADDITIVE_PVE - pve, 1e-6)) * rng.standard_normal(n)
    X0 = np.ones((n, 1)); rs = int(seed) % 100000
    try:
        Wh = _whiten(KA, KD, y, X0)
        p_correct = _perpair_pvals(Wh, y, BA_m, BD_m)
        p_wrong = _perpair_pvals(Wh, y, BA_sh, BD_sh)
        # omnibus kernel comparisons (LRT vs additive null)
        ll0 = float(fit_multi_reml(y, X0, {"A": KA, "D": KD}, n_starts=3, random_state=rs).log_lik)
        lrt = {}
        for tag, Kx in (("hom", K_hom), ("pair", K_pair)):
            f = fit_multi_reml(y, X0, {"A": KA, "D": KD, tag: Kx}, n_starts=3, random_state=rs)
            lrt[tag] = max(0.0, 2.0 * (float(f.log_lik) - ll0))
        return dict(ok=True, acat=_acat(p_correct), acat_wrong=_acat(p_wrong),
                    minp=float(p_correct.min()), minp_wrong=float(p_wrong.min()),
                    lrt_hom=lrt["hom"], lrt_pair=lrt["pair"])
    except Exception:  # noqa: BLE001
        return dict(ok=False)


def _null_calib_rep(KA, KD, cholA, cholD, BA_m, BD_m, probs, seed):
    """Lean PVE=0 rep for TAIL CALIBRATION: only correct-pair per-pair p + analytic
    ACAT p (skips wrong-pair scan + omnibus LRT REML fits, so we can afford 1000s of
    reps). Returns analytic acat_p, per-rep lambda_GC, tail counts, QQ quantiles, and
    REML variance-component hats (to flag boundary fits that bias the t-test tail)."""
    rng = np.random.default_rng(seed)
    n = KA.shape[0]
    sa = ADDITIVE_PVE / 2
    y = (np.sqrt(sa) * (cholA @ rng.standard_normal(n))
         + np.sqrt(sa) * (cholD @ rng.standard_normal(n))
         + np.sqrt(max(1 - ADDITIVE_PVE, 1e-6)) * rng.standard_normal(n))
    X0 = np.ones((n, 1))
    try:
        res = fit_multi_reml(y, X0, {"A": KA, "D": KD}, n_starts=3, random_state=int(seed) % 100000)
        cv = res.component_var
        V = (cv.get("A", 0.0) * KA + cv.get("D", 0.0) * KD
             + max(cv.get("e", 1e-6), 1e-6) * np.eye(n))
        w, Q = np.linalg.eigh(0.5 * (V + V.T))
        w = np.clip(w, 1e-10, None)
        Wh = (Q * (1.0 / np.sqrt(w))) @ Q.T
        p = _perpair_pvals(Wh, y, BA_m, BD_m)
        tail = {a: int((p <= a).sum()) for a in (0.05, 0.01, 0.005, 0.001)}
        return dict(ok=True, acat=_acat(p), lam=_lambda_gc(p), tail=tail, npairs=int(p.size),
                    qobs=np.quantile(p, probs),
                    sa=float(cv.get("A", 0.0)), sd=float(cv.get("D", 0.0)),
                    se=float(cv.get("e", 0.0)))
    except Exception:  # noqa: BLE001
        return dict(ok=False)


def _offset_partner_index(chrom_num, delta, rng):
    """Test-pairing transform: keep the A anchor, shift each pair's D partner by
    `delta` bins ALONG rank order WITHIN the same chromosome (boundary=drop, no wrap
    -> avoids non-physical chromosome-end adjacency). delta='random' -> fixed
    within-chrom permutation (= permuted-pair floor, matches acat_wrong machinery).
    Returns (off[G] partner pair-index or -1, valid[G] bool). off[g] indexes the pair
    whose D-burden becomes pair g's tested D partner; BD_m[:, off[g]] is that burden."""
    G = chrom_num.size
    off = np.full(G, -1, int)
    for k in np.unique(chrom_num):
        blk = np.where(chrom_num == k)[0]  # already A-rank order (pairs appended so)
        if delta == "random":
            off[blk] = rng.permutation(blk)
        else:
            d = int(delta)
            for j in range(blk.size):
                jj = j + d
                if 0 <= jj < blk.size:
                    off[blk[j]] = blk[jj]
    return off, off >= 0


def _offset_rep(KA, KD, cholA, cholD, BA_sub, BD_sub, Uc, pve, seed):
    """Lean rep for pairing sanity: causal stays at rank pairs (Uc fixed); test uses
    offset/permuted D partner (BA_sub, BD_sub). Returns analytic ACAT p (calibration
    established by the tail-calibration block, so power uses analytic acat_p<alpha, not
    empirical crit)."""
    rng = np.random.default_rng(seed)
    n = KA.shape[0]
    sa = ADDITIVE_PVE / 2
    y = np.sqrt(sa) * (cholA @ rng.standard_normal(n)) + np.sqrt(sa) * (cholD @ rng.standard_normal(n))
    if pve > 0:
        y += _scale_vec(Uc @ rng.standard_normal(Uc.shape[1])) * np.sqrt(pve)
    y += np.sqrt(max(1 - ADDITIVE_PVE - pve, 1e-6)) * rng.standard_normal(n)
    try:
        Wh = _whiten(KA, KD, y, np.ones((n, 1)))
        return dict(ok=True, acat=_acat(_perpair_pvals(Wh, y, BA_sub, BD_sub)))
    except Exception:  # noqa: BLE001
        return dict(ok=False)


def _binom_ci95(k, nn):
    """Wilson 95% CI for a binomial proportion (stable near 0, unlike normal approx)."""
    if nn == 0:
        return [0.0, 0.0]
    z = 1.959963984540054
    phat = k / nn
    denom = 1 + z * z / nn
    centre = (phat + z * z / (2 * nn)) / denom
    half = z * np.sqrt(phat * (1 - phat) / nn + z * z / (4 * nn * nn)) / denom
    return [float(max(0.0, centre - half)), float(min(1.0, centre + half))]


def run(panel, n_rep, null_rep, n_jobs, block_m, n_causal, bin_mode, bin_size,
        causal_kind, causal_pairing, out_dir, n_calib=0,
        pair_sweep=False, sweep_deltas=(0, 1, 2, 3, "random"), sweep_rep=150,
        arch_panel=False, arch_rep=150):
    cfg = PANELS[panel]
    print(f"=== D3 per-pair {panel} trait={cfg['trait']} block_m={block_m} "
          f"bin_mode={bin_mode} bin_size={bin_size} ===")
    t0 = time.time()
    D = _load(cfg)
    n = D["n"]
    if bin_mode == "fixed":
        BA, labA, winA, nbA = _fixed_bins(D["XA"], D["chromA"], D["posA"], cfg["chrom_fmt"],
                                          cfg["sub_A"], cfg["n_chrom"], bin_size)
        BD, labD, winD, nbD = _fixed_bins(D["XD"], D["chromD"], D["posD"], cfg["chrom_fmt"],
                                          cfg["sub_D"], cfg["n_chrom"], bin_size)
        pair_idx = _rank_pair(labA, nbA, labD, nbD, cfg["n_chrom"], cfg["chrom_fmt"],
                              cfg["sub_A"], cfg["sub_D"])
    else:
        BA, labA, winA = _burden_matrix(D["XA"], D["chromA"], D["posA"], cfg["chrom_fmt"],
                                        cfg["sub_A"], cfg["n_chrom"], 40, MIN_SNP_PER_WIN)
        BD, labD, winD = _burden_matrix(D["XD"], D["chromD"], D["posD"], cfg["chrom_fmt"],
                                        cfg["sub_D"], cfg["n_chrom"], 40, MIN_SNP_PER_WIN)
        pair_idx = _pair_kernel(BA, labA, BD, labD)[2]
    G = len(pair_idx)
    jA = [p[0] for p in pair_idx]; jD = [p[1] for p in pair_idx]
    BA_m = _scale_cols_safe(BA[:, jA]); BD_m = _scale_cols_safe(BD[:, jD])
    chrom_num = np.array([int("".join(c for c in labA[j][0] if c.isdigit())) for j in jA])
    # within-chrom shuffle of D partners (wrong-pair control)
    rng0 = np.random.default_rng(7)
    perm_wc = np.arange(G)
    for k in np.unique(chrom_num):
        idx = np.where(chrom_num == k)[0]; perm_wc[idx] = rng0.permutation(idx)
    BD_sh = BD_m[:, perm_wc]
    # omnibus synteny kernel from the SAME matched bin/window pair features
    KA, KD, K_hom = D["KA"], D["KD"], D["K_hom"]
    Zc = np.column_stack([_scale_vec(_scale_vec(BA_m[:, g]) * _scale_vec(BD_m[:, g]))
                          for g in range(G)])
    K_pair = normalize_kernel(Zc @ Zc.T / G, "trace")
    Z_true = Zc  # unused by gene_block causal; placeholder
    cholA, cholD = _chol(KA), _chol(KD)
    winA_m = [winA[p[0]] for p in pair_idx]; winD_m = [winD[p[1]] for p in pair_idx]
    chromsortA = {k: np.where(D["chromA"] == cfg["chrom_fmt"].format(sub=cfg["sub_A"], k=k))[0]
                  [np.argsort(D["posA"][np.where(D["chromA"] == cfg["chrom_fmt"].format(sub=cfg["sub_A"], k=k))[0]])]
                  for k in range(1, cfg["n_chrom"] + 1)}
    chromsortD = {k: np.where(D["chromD"] == cfg["chrom_fmt"].format(sub=cfg["sub_D"], k=k))[0]
                  [np.argsort(D["posD"][np.where(D["chromD"] == cfg["chrom_fmt"].format(sub=cfg["sub_D"], k=k))[0]])]
                  for k in range(1, cfg["n_chrom"] + 1)}
    Uc = _causal_features("gene_block", n_causal, D["XA"], D["XD"], winA_m, winD_m,
                          chrom_num, Z_true, rng0, chromsortA, chromsortD, block_m, "offset",
                          causal_kind, causal_pairing)
    print(f"  n={n} G={G} block_m={block_m} C={Uc.shape[1]} ({time.time()-t0:.1f}s)")

    # --- ARCHITECTURE PANEL: off-model architecture sensitivity at locked
    #     PVE=0.20. Iterates 3 causal architectures (on-model block burden-product /
    #     single-SNP-pair / mixed-sign residualized), reuses the DEPLOYED burden-product
    #     test (BA_m, BD_m) unchanged, reports power + projection diagnostic + realized
    #     PVE. Early returns: skips main grid/calib/sweep. The cor_with_burden_prod
    #     diagnostic separates "burden blind to existing signal" from "we accidentally
    #     built undetectable signal".
    if arch_panel:
        archs = [("on_model_concord", "interaction"),
                 ("single_snp_pair", "single_snp_pair"),
                 ("mixed_sign", "mixed_sign")]
        by_arch = {}
        for arch_name, kind in archs:
            diag = []
            rng_a = np.random.default_rng(11)  # fixed seed: same block positions across archs
            Uc_a = _causal_features("gene_block", n_causal, D["XA"], D["XD"], winA_m, winD_m,
                                    chrom_num, Z_true, rng_a, chromsortA, chromsortD,
                                    block_m, "offset", kind, "matched", diag_out=diag)
            sig = Parallel(n_jobs=n_jobs)(delayed(_offset_rep)(KA, KD, cholA, cholD, BA_m, BD_m,
                                           Uc_a, 0.20, 800000 + i) for i in range(arch_rep))
            nul = Parallel(n_jobs=n_jobs)(delayed(_offset_rep)(KA, KD, cholA, cholD, BA_m, BD_m,
                                           Uc_a, 0.0, 810000 + i) for i in range(arch_rep))
            aps = np.array([r["acat"] for r in sig if r.get("ok")])
            apn = np.array([r["acat"] for r in nul if r.get("ok")])
            ks, kn = int((aps < 0.05).sum()), int((apn < 0.05).sum())
            cors = np.array([d["cor_with_burden_prod"] for d in diag])
            rng_p = np.random.default_rng(999)
            pve_samples = [float((_scale_vec(Uc_a @ rng_p.standard_normal(Uc_a.shape[1]))
                                  * np.sqrt(0.20)).var()) for _ in range(50)]
            by_arch[arch_name] = dict(
                causal_kind=kind, C_used=int(Uc_a.shape[1]),
                power=float(ks / aps.size) if aps.size else None,
                power_ci95=_binom_ci95(ks, aps.size),
                type1=float(kn / apn.size) if apn.size else None,
                type1_ci95=_binom_ci95(kn, apn.size),
                cor_burdenprod_signed_mean=float(np.mean(cors)) if cors.size else None,
                cor_burdenprod_abs_mean=float(np.mean(np.abs(cors))) if cors.size else None,
                cor_burdenprod_range=[float(cors.min()), float(cors.max())] if cors.size else None,
                realized_pve_mean=float(np.mean(pve_samples)),
                realized_pve_sd=float(np.std(pve_samples)))
        bintag = f"{bin_mode}{bin_size}" if bin_mode == "fixed" else "win40"
        out_panel = dict(panel=panel, trait=cfg["trait"], n=n, G=G, block_m=block_m,
                         n_causal=n_causal, arch_rep=arch_rep, pve=0.20, alpha=0.05,
                         by_arch=by_arch,
                         note=("Architecture sensitivity at EQUALIZED total interaction variance "
                               "0.20 (via _scale_vec: causal signal norm forced equal across "
                               "architectures; this is a design equalization, NOT a natural "
                               "realized-PVE lockdown). Test = per-pair GLS burden-product + "
                               "ACAT, deployed form unchanged. on_model_concord: block burden-"
                               "product (truth==test form). single_snp_pair: 1 A-SNP × 1 D-SNP "
                               "within block; under cotton LD the empirical projection onto the "
                               "30-SNP burden is NOT the asymptotic 1/30 — varies block to block "
                               "(see cor_burdenprod_range). mixed_sign: balanced ±1 signed sums "
                               "residualized against plain burden, sample-level NEAR-orthogonal "
                               "(not mathematically strict) to gA*gD. cor_burdenprod_signed_mean "
                               "= <cor(causal_col, gA*gD)> over causal blocks; quantifies what "
                               "the deployed test can see. Honest claim: SCOPE LIMIT of the "
                               "burden-product test (NOT a claim that interaction detection is "
                               "generally invalid for these architectures — sign-aware/SKAT tests "
                               "could see mixed-sign signal). type-I is causal_kind-independent "
                               "under null (pve=0 -> Uc×0); identical numbers across archs are "
                               "an implementation sanity, not a separate calibration claim. "
                               "Within-panel comparison uses fixed seed=11 (same causal block "
                               "positions across archs, fair); absolute on_model_concord power "
                               "here is NOT directly comparable to the canonical run "
                               "(seed=7, different block positions, C=3 MC variance non-trivial)."))
        out_dir.mkdir(parents=True, exist_ok=True)
        pp = out_dir / f"{panel}_d3_arch_panel_m{block_m}_C{n_causal}_{bintag}.json"
        pp.write_text(json.dumps(out_panel, indent=2, default=float))
        print(f"  ARCH-PANEL (PVE0.20, analytic p<.05):")
        for a, v in by_arch.items():
            print(f"    {a:>18}: power={v['power']:.3f} CI{[round(x, 3) for x in v['power_ci95']]} "
                  f"type1={v['type1']:.3f} cor_bp={v['cor_burdenprod_signed_mean']:.3f} "
                  f"realized_pve={v['realized_pve_mean']:.3f}")
        print(f"  wrote {pp} ({time.time() - t0:.1f}s)")
        return

    def stat(reps, key):
        return np.array([r[key] for r in reps if r.get("ok") and key in r], float)

    out = {"panel": panel, "trait": cfg["trait"], "n": n, "G": G, "block_m": block_m,
           "n_rep": n_rep, "null_rep": null_rep, "grid": {}}
    # null
    nr = Parallel(n_jobs=n_jobs)(delayed(_one_rep)(KA, KD, K_hom, K_pair, cholA, cholD,
                                                   BA_m, BD_m, BA_m, BD_sh, Uc, 0.0, 90000 + r)
                                 for r in range(null_rep))
    crit = {"acat": np.nanpercentile(stat(nr, "acat"), 5),        # ACAT/minP: small=signif -> 5th pctile
            "minp": np.nanpercentile(stat(nr, "minp"), 5),
            "acat_wrong": np.nanpercentile(stat(nr, "acat_wrong"), 5),
            "lrt_hom": np.nanpercentile(stat(nr, "lrt_hom"), 95),
            "lrt_pair": np.nanpercentile(stat(nr, "lrt_pair"), 95)}
    out["empirical_crit"] = {k: float(v) for k, v in crit.items()}
    out["grid"]["0.00"] = dict(
        acat=float(np.mean(stat(nr, "acat") < crit["acat"])),
        minp=float(np.mean(stat(nr, "minp") < crit["minp"])),
        acat_wrong=float(np.mean(stat(nr, "acat_wrong") < crit["acat_wrong"])),
        hom=float(np.mean(stat(nr, "lrt_hom") > crit["lrt_hom"])),
        pair=float(np.mean(stat(nr, "lrt_pair") > crit["lrt_pair"])))
    print(f"  null power(~.05): {[(k, round(v, 3)) for k, v in out['grid']['0.00'].items()]}")
    for pve in PVE_GRID:
        reps = Parallel(n_jobs=n_jobs)(delayed(_one_rep)(KA, KD, K_hom, K_pair, cholA, cholD,
                                                        BA_m, BD_m, BA_m, BD_sh, Uc, pve,
                                                        10000 * int(pve * 100) + r)
                                       for r in range(n_rep))
        rec = dict(
            acat=float(np.mean(stat(reps, "acat") < crit["acat"])),
            minp=float(np.mean(stat(reps, "minp") < crit["minp"])),
            acat_wrong=float(np.mean(stat(reps, "acat_wrong") < crit["acat_wrong"])),
            hom=float(np.mean(stat(reps, "lrt_hom") > crit["lrt_hom"])),
            pair=float(np.mean(stat(reps, "lrt_pair") > crit["lrt_pair"])))
        out["grid"][f"{pve:.2f}"] = rec
        print(f"  PVE={pve:.2f} | PER-PAIR acat={rec['acat']:.2f} minp={rec['minp']:.2f} "
              f"|| omnibus hom={rec['hom']:.2f} pair={rec['pair']:.2f} "
              f"|| wrong-pair acat={rec['acat_wrong']:.2f}")
    # --- TAIL CALIBRATION: analytic ACAT p type-I @ multiple alpha +
    #     per-pair lambda_GC. type-I uses the ANALYTIC ACAT p (not empirical crit):
    #     empirical-percentile thresholds force type-I == alpha by construction
    #     (circular; would mask real tail miscalibration).
    if n_calib > 0:
        probs = np.unique(np.clip(np.logspace(np.log10(1.0 / G), 0.0, 300), 1e-9, 1.0))
        cr = Parallel(n_jobs=n_jobs)(
            delayed(_null_calib_rep)(KA, KD, cholA, cholD, BA_m, BD_m, probs, 500000 + r)
            for r in range(n_calib))
        cr = [r for r in cr if r.get("ok")]
        R = len(cr)
        acat_p = np.array([r["acat"] for r in cr])
        lam_rep = np.array([r["lam"] for r in cr])
        npairs = cr[0]["npairs"]
        typeI = {}
        for a in (0.05, 0.01, 0.005):
            k = int((acat_p <= a).sum())
            typeI[f"{a}"] = dict(type1=k / R, count=k, n_calib=R, ci95=_binom_ci95(k, R),
                                 mc_se_nominal=float(np.sqrt(a * (1 - a) / R)))
        pooled_tail = {f"{a}": sum(r["tail"][a] for r in cr) / (npairs * R)
                       for a in (0.05, 0.01, 0.005, 0.001)}
        qobs_mean = np.row_stack([r["qobs"] for r in cr]).mean(0)
        out["tail_calibration"] = dict(
            n_calib=R, G=G, npairs_per_rep=npairs,
            acat_analytic_typeI=typeI,
            lambda_gc_rep_median=float(np.median(lam_rep)),
            lambda_gc_rep_mean=float(np.mean(lam_rep)),
            lambda_gc_rep_iqr=[float(np.percentile(lam_rep, 25)), float(np.percentile(lam_rep, 75))],
            perpair_pooled_typeI=pooled_tail,
            sigma_hat_mean=dict(A=float(np.mean([r["sa"] for r in cr])),
                                D=float(np.mean([r["sd"] for r in cr])),
                                e=float(np.mean([r["se"] for r in cr]))),
            qq_probs=probs.tolist(), qq_obs_mean=qobs_mean.tolist())
        print(f"  TAIL-CALIB R={R} | ACAT analytic type-I: "
              f"{[(a, round(v['type1'], 4)) for a, v in typeI.items()]} | "
              f"per-pair lambda_GC med={np.median(lam_rep):.3f} "
              f"iqr=[{np.percentile(lam_rep, 25):.3f},{np.percentile(lam_rep, 75):.3f}] | "
              f"per-pair pooled type-I={[(a, round(v, 4)) for a, v in pooled_tail.items()]}")

    # --- PAIRING SANITY: decouple P_causal (fixed at rank pairs) from
    #     P_test (D partner shifted by delta within chrom). delta=0 -> oracle ceiling,
    #     'random' -> permuted floor (=acat_wrong), intermediate -> graceful decay vs
    #     knife-edge. Defuses "test-pairing==causal-pairing tautology". NOTE: only the
    #     D partner is offset (A anchor kept) -> "D-partner offset sensitivity", and
    #     small |delta| (~30 SNP/bin) may retain LD-driven power, not true robustness;
    #     judge by the random->floor decay signature, not a single delta. ---
    if pair_sweep:
        rng_perm = np.random.default_rng(7)  # match acat_wrong (perm_wc seed 7)
        sweep = {}
        pw0 = None
        for d in sweep_deltas:
            off, valid = _offset_partner_index(chrom_num, d, rng_perm)
            BA_sub = BA_m[:, valid]
            BD_sub = BD_m[:, off[valid]]
            sig = Parallel(n_jobs=n_jobs)(delayed(_offset_rep)(KA, KD, cholA, cholD, BA_sub,
                                          BD_sub, Uc, 0.20, 700000 + i) for i in range(sweep_rep))
            nul = Parallel(n_jobs=n_jobs)(delayed(_offset_rep)(KA, KD, cholA, cholD, BA_sub,
                                          BD_sub, Uc, 0.0, 710000 + i) for i in range(sweep_rep))
            aps = np.array([r["acat"] for r in sig if r.get("ok")])
            apn = np.array([r["acat"] for r in nul if r.get("ok")])
            ks, kn = int((aps < 0.05).sum()), int((apn < 0.05).sum())
            rec = dict(power=float(np.mean(aps < 0.05)), type1=float(np.mean(apn < 0.05)),
                       power_ci95=_binom_ci95(ks, aps.size), type1_ci95=_binom_ci95(kn, apn.size),
                       n_valid=int(valid.sum()), n_ok_sig=int(aps.size), n_ok_null=int(apn.size))
            sweep[str(d)] = rec
            if str(d) == "0":
                pw0 = rec["power"]
        for d in sweep:
            sweep[d]["retention"] = (sweep[d]["power"] / pw0) if pw0 else None
        out["pairing_sanity"] = dict(
            pve=0.20, alpha=0.05, sweep_rep=sweep_rep,
            deltas=[str(x) for x in sweep_deltas], by_delta=sweep,
            note=("D-partner offset sensitivity under FIXED causal rank architecture and "
                  "FIXED A anchor (NOT fully independent pairing re-sampling: A anchor "
                  "retained, so mispaired D bears only half the mismatch penalty -> "
                  "retention may be upper-biased). Small-|delta| retention also includes "
                  "local-LD / neighbour-tag effects (delta=1 ~30 SNP); the overall "
                  "random->floor decay is the core evidence, not any single delta. "
                  "P_causal=rank fixed; P_test=D-partner shift; power via analytic ACAT p<0.05."))
        print(f"  PAIRING-SANITY (PVE0.20, analytic p<.05): "
              f"{[(d, round(v['power'], 3), 'ret', round(v['retention'], 2), 't1', round(v['type1'], 3)) for d, v in sweep.items()]}")

    g = out["grid"]
    out["decisive_perpair_beats_omnibus"] = bool(any(
        g[p]["acat"] > g[p]["hom"] + 0.15 and g[p]["acat"] > g[p]["pair"] + 0.15
        and g[p]["acat"] > g[p]["acat_wrong"] + 0.15 for p in ("0.10", "0.20")))
    print(f"  DECISIVE per-pair ACAT >> omnibus & wrong-pair: {out['decisive_perpair_beats_omnibus']}")
    out_dir.mkdir(parents=True, exist_ok=True)
    bintag = f"{bin_mode}{bin_size}" if bin_mode == "fixed" else "win40"
    out["causal_kind"] = causal_kind; out["causal_pairing"] = causal_pairing
    ck = "" if (causal_kind == "interaction" and causal_pairing == "matched") else f"_{causal_kind}_{causal_pairing}"
    p = out_dir / f"{panel}_d3_perpair_m{block_m}_C{n_causal}_{bintag}{ck}.json"
    p.write_text(json.dumps(out, indent=2, default=float))
    print(f"  wrote {p} ({time.time()-t0:.1f}s)")


def _scale_cols_safe(M):
    mu = M.mean(0); sd = M.std(0, ddof=0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    return (M - mu) / sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", choices=list(PANELS), default="cotton_hebau")
    ap.add_argument("--n-rep", type=int, default=200)
    ap.add_argument("--null-rep", type=int, default=300)
    ap.add_argument("--n-jobs", type=int, default=40)
    ap.add_argument("--block-m", type=int, default=30)
    ap.add_argument("--n-causal", type=int, default=10)
    ap.add_argument("--bin-mode", choices=["window", "fixed"], default="window")
    ap.add_argument("--bin-size", type=int, default=30)
    ap.add_argument("--causal-kind",
                    choices=["interaction", "main_only", "single_snp_pair", "mixed_sign"],
                    default="interaction")
    ap.add_argument("--causal-pairing", choices=["matched", "noncorr"], default="matched")
    ap.add_argument("--n-calib", type=int, default=0,
                    help="lean PVE=0 reps for tail calibration (analytic ACAT type-I + lambda_GC); 0=off")
    ap.add_argument("--pair-sweep", action="store_true",
                    help="pairing sanity: P_causal=rank fixed, P_test=D-partner offset sweep")
    ap.add_argument("--sweep-deltas", default="0,1,2,3,random",
                    help="comma list of D-partner shifts (int or 'random'); must include 0 for retention")
    ap.add_argument("--sweep-rep", type=int, default=150)
    ap.add_argument("--arch-panel", action="store_true",
                    help="architecture sensitivity panel (on-model / single_snp_pair / mixed_sign); "
                         "early-returns, skips main grid/calib/sweep, writes a focused panel json")
    ap.add_argument("--arch-rep", type=int, default=150)
    ap.add_argument("--out-dir", default="results/phase7/d3_perpair")
    args = ap.parse_args()
    deltas = tuple(x if x == "random" else int(x) for x in args.sweep_deltas.split(","))
    run(args.panel, args.n_rep, args.null_rep, args.n_jobs, args.block_m, args.n_causal,
        args.bin_mode, args.bin_size, args.causal_kind, args.causal_pairing, ROOT / args.out_dir,
        n_calib=args.n_calib, pair_sweep=args.pair_sweep, sweep_deltas=deltas, sweep_rep=args.sweep_rep,
        arch_panel=args.arch_panel, arch_rep=args.arch_rep)


if __name__ == "__main__":
    main()
