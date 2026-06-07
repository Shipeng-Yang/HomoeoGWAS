#!/usr/bin/env python3
"""Phase 7 D2 arm-2 — synteny-aware homoeolog-pair kernel vs global K_hom.

The decisive "new method" test of the dilution hypothesis. D1+D2arm1 showed the
global Hadamard K_hom (=elementwise product of per-subgenome GRMs) is a LOW-POWER
instrument confounded with the residual. arm-2 asks: if the truth is SPARSE
homoeolog-pair interaction, does a SYNTENY-AWARE homoeolog-pair kernel recover it
where the global K_hom cannot?

Design (Codex arm-2 dual-plan):
  - chromosome-level homoeology A_k <-> D_k (cotton 13 pairs); split each into W
    equal-count windows; pair window w of A_k with window w of D_k -> G window-pairs.
  - per pair g: regional burden b_A,g = mean(standardized A-window dosage),
    b_D,g likewise; interaction feature z_g = scale(scale(b_A,g)*scale(b_D,g));
    K_pair = Z Z'/G  (reduced-rank additive×additive homoeolog epistasis kernel).
  - SHUFFLED control: permute D-window partners (across all D) -> K_pair_shuf.
  - simulate y = additive(K_A,K_D @0.4) + sparse pair signal (C causal window-pairs,
    s=Σβ_g z_g scaled to PVE_pair) + residual.
  - fit 4 models vs additive-only null: {+K_hom},{+K_pair},{+K_pair_shuf}; LRT power
    at empirical null (PVE_pair=0). Robustness: +K_mainpair (local burden main effects).

Decisive: at PVE_pair 0.10-0.20, power(K_pair) >= 2x power(K_hom) AND >> power(shuf).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from homoeogwas.grm import compute_grm
from homoeogwas.io import load_bed_hardcall
from homoeogwas.kernel import build_homoeolog_kernel, normalize_kernel
from homoeogwas.lmm import fit_multi_reml

ROOT = Path(__file__).resolve().parents[2]
PANELS = {
    "cotton_hebau": dict(bed_root="data/processed/cotton", bed_name="{sub}/all",
                         sub_A="A", sub_D="D", n_chrom=13, chrom_fmt="{sub}{k:02d}",
                         pheno="data/processed/cotton/pheno_m3_4_blue.tsv",
                         sample_col="sample", trait="fiber_length_BLUE", maf_min=0.01),
}
PVE_PAIR_GRID = [0.05, 0.10, 0.20]
ADDITIVE_PVE = 0.40
N_CAUSAL = 10
W_WINDOWS = 40
MIN_SNP_PER_WIN = 5


def _scale_cols(M):
    """Mean-impute missing genotypes, then z-score columns (zero-var -> 0)."""
    mu = np.nanmean(M, axis=0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    Mi = np.where(np.isnan(M), mu, M)
    sd = Mi.std(0, ddof=0)
    sd_safe = np.where(sd > 1e-12, sd, 1.0)
    Z = (Mi - mu) / sd_safe
    Z[:, sd <= 1e-12] = 0.0
    return Z


def _scale_vec(v):
    sd = v.std(ddof=0)
    return (v - v.mean()) / sd if sd > 1e-12 else np.zeros_like(v)


def _load(cfg):
    """Load aligned A/D dosages (trait samples) + trace-normed K_A,K_D,K_hom + BIM."""
    root = ROOT / cfg["bed_root"]
    bedA = load_bed_hardcall(root / cfg["bed_name"].format(sub=cfg["sub_A"]))
    bedD = load_bed_hardcall(root / cfg["bed_name"].format(sub=cfg["sub_D"]))
    sA, sD = list(np.asarray(bedA.samples)), list(np.asarray(bedD.samples))
    ref = sA if sA == sD else sorted(set(sA) & set(sD))
    ph = pd.read_csv(ROOT / cfg["pheno"], sep="\t").set_index(cfg["sample_col"])[cfg["trait"]]
    common = [s for s in ref if s in ph.index and pd.notna(ph.loc[s])]
    iA = np.array([sA.index(s) for s in common]); iD = np.array([sD.index(s) for s in common])
    XA = np.asarray(bedA.dosage, dtype=np.float64)[iA]      # n×M_A
    XD = np.asarray(bedD.dosage, dtype=np.float64)[iD]
    chromA = np.asarray(bedA.chrom).astype(str); posA = np.asarray(bedA.pos, dtype=np.int64)
    chromD = np.asarray(bedD.chrom).astype(str); posD = np.asarray(bedD.pos, dtype=np.int64)
    # additive GRMs (trace-normed) + global K_hom — match Table1/D1
    KA, _ = compute_grm(bedA, maf_min=cfg["maf_min"])
    KD, _ = compute_grm(bedD, maf_min=cfg["maf_min"])
    KA = KA[np.ix_(iA, iA)]; KD = KD[np.ix_(iD, iD)]
    grms = {"A": KA, "D": KD}
    K_hom_raw, hom_used = build_homoeolog_kernel(grms, mode="auto")
    kernels_add = {"A": normalize_kernel(KA, "trace"), "D": normalize_kernel(KD, "trace")}
    K_hom = normalize_kernel(K_hom_raw, "trace")
    return dict(n=len(common), XA=XA, XD=XD, chromA=chromA, posA=posA, chromD=chromD,
                posD=posD, KA=kernels_add["A"], KD=kernels_add["D"], K_hom=K_hom,
                hom_mode=hom_used, cfg=cfg)


def _burden_matrix(X, chrom, pos, chrom_fmt, sub, n_chrom, W, min_snp):
    """n×G_eff burden matrix + per-window SNP indices: equal-count W windows."""
    Xs = _scale_cols(X)        # standardize each SNP (mean-impute missing)
    cols, labels, win_snp = [], [], []
    for k in range(1, n_chrom + 1):
        cname = chrom_fmt.format(sub=sub, k=k)
        idx = np.where(chrom == cname)[0]
        order = idx[np.argsort(pos[idx])]
        for w, sidx in enumerate(np.array_split(order, W)):
            if sidx.size < min_snp:
                continue
            cols.append(Xs[:, sidx].mean(1))
            labels.append((cname, w))
            win_snp.append(sidx)
    B = np.column_stack(cols) if cols else np.zeros((X.shape[0], 0))
    return B, labels, win_snp


def _pair_kernel(BA, labA, BD, labD):
    """Match A/D windows by (chrom-index, window) -> z_g=scale(scale(bA)*scale(bD)); ZZ'/G."""
    # index D windows by (chrom_num, w)
    def keyset(labels):
        out = {}
        for j, (cn, w) in enumerate(labels):
            num = int("".join(ch for ch in cn if ch.isdigit()))
            out[(num, w)] = j
        return out
    dkey = keyset(labD)
    zcols, pair_idx = [], []
    for jA, (cn, w) in enumerate(labA):
        num = int("".join(ch for ch in cn if ch.isdigit()))
        jD = dkey.get((num, w))
        if jD is None:
            continue
        z = _scale_vec(_scale_vec(BA[:, jA]) * _scale_vec(BD[:, jD]))
        zcols.append(z); pair_idx.append((jA, jD))
    Z = np.column_stack(zcols)
    G = Z.shape[1]
    K = Z @ Z.T / G
    return K, Z, pair_idx


def _shuffled_pair_kernel(BA, BD_perm_cols):
    """K from z_g = scale(scale(bA_g)*scale(bD_perm_g)); BD already permuted columns."""
    zcols = [_scale_vec(_scale_vec(BA[:, j]) * _scale_vec(BD_perm_cols[:, j]))
             for j in range(BA.shape[1])]
    Z = np.column_stack(zcols); G = Z.shape[1]
    return Z @ Z.T / G


def _std_col(X, j):
    x = X[:, j].astype(float)
    mu = np.nanmean(x)
    x = np.where(np.isnan(x), mu if np.isfinite(mu) else 0.0, x)
    return _scale_vec(x)


def _pick_maf(X, idx, rng, lo=0.1, hi=0.9, tries=30):
    """Random SNP index from idx with MAF in [lo,hi]; None if none found."""
    if len(idx) == 0:
        return None
    for _ in range(tries):
        j = int(idx[rng.integers(len(idx))])
        maf = float(np.nanmean(X[:, j])) / 2.0
        if lo <= maf <= hi:
            return j
    return None


def _block_burden(X, idx):
    """Mean standardized dosage over an SNP block (gene-scale burden)."""
    return _scale_cols(X[:, idx]).mean(1)


def _causal_features(mode, C, XA, XD, winA_m, winD_m, chrom_num, Z_true, rng,
                     chromsortA=None, chromsortD=None, block_m=30, block_align="offset",
                     causal_kind="interaction", causal_pairing="matched", diag_out=None):
    """n×C causal feature matrix. window: K_pair's own window products;
    snp_corr: one A-SNP × one D-SNP within CORRESPONDING homoeolog windows;
    snp_noncorr: mismatched-chromosome SNP pair (specificity);
    gene_block: m contiguous SNPs (A-block × homoeologous D-block at matched relative
    position), placed at RANDOM positions OFFSET from the kernel's window grid -> the
    fair test of whether K_pair recovers region-scale signal it was NOT handed.

    gene_block causal_kind branches (for patch ③ architecture sensitivity):
      interaction (default, on-model): gA*gD where gA,gD are plain block burdens
        -> truth matches the burden-product test form (concordant, all-positive).
      main_only: gA+gD (additive control, no interaction).
      single_snp_pair (OFF-MODEL sparse): one MAF-filtered SNP in A-block × one in
        D-block; the test burden (mean over 30 SNPs) dilutes ~1/30, predicted power drop.
      mixed_sign (OFF-MODEL sign-discordant): balanced ±1 signed combos S_A,S_D over
        block SNPs, RESIDUALIZED against the plain burden direction so the burden-
        product test has ~zero projection on truth; sign-aware/SKAT could see it.
    diag_out (list, optional): per-causal-block dict with cor(causal_col, gA*gD) — the
    projection of the true signal onto the burden-product form the test sees."""
    if mode == "window":
        causal = rng.choice(Z_true.shape[1], size=min(C, Z_true.shape[1]), replace=False)
        return Z_true[:, causal]
    G = len(winA_m)
    cols = []
    if mode == "snp_corr":
        for g in rng.choice(G, size=min(C, G), replace=False):
            ai, di = _pick_maf(XA, winA_m[g], rng), _pick_maf(XD, winD_m[g], rng)
            if ai is not None and di is not None:
                cols.append(_std_col(XA, ai) * _std_col(XD, di))
    elif mode == "snp_noncorr":
        for _ in range(C):
            ga = int(rng.integers(G))
            others = np.where(chrom_num != chrom_num[ga])[0]
            gd = int(others[rng.integers(len(others))])
            ai, di = _pick_maf(XA, winA_m[ga], rng), _pick_maf(XD, winD_m[gd], rng)
            if ai is not None and di is not None:
                cols.append(_std_col(XA, ai) * _std_col(XD, di))
    elif mode == "gene_block" and block_align == "aligned":
        # causal block = first block_m SNPs of a kernel window (aligned to grid)
        for g in rng.choice(G, size=min(C, G), replace=False):
            bA, bD = winA_m[g], winD_m[g]
            if len(bA) < block_m or len(bD) < block_m:
                continue
            gA = _block_burden(XA, bA[:block_m]); gD = _block_burden(XD, bD[:block_m])
            cols.append(_scale_vec(gA) * _scale_vec(gD))
    elif mode == "gene_block":  # offset: random contiguous block, off the window grid
        ks = list(chromsortA.keys())
        for _ in range(C):
            kA = ks[int(rng.integers(len(ks)))]
            # D block: matched homoeologous chromosome (default) or a DIFFERENT
            # chromosome (non-homoeolog specificity control)
            if causal_pairing == "noncorr":
                kD = ks[int(rng.integers(len(ks)))]
                while kD == kA and len(ks) > 1:
                    kD = ks[int(rng.integers(len(ks)))]
            else:
                kD = kA
            iA, iD = chromsortA[kA], chromsortD[kD]
            if len(iA) < block_m or len(iD) < block_m:
                continue
            sA = int(rng.integers(0, len(iA) - block_m + 1))
            rel = sA / max(len(iA) - block_m, 1)
            sD = (int(rng.integers(0, len(iD) - block_m + 1)) if causal_pairing == "noncorr"
                  else int(round(rel * (len(iD) - block_m))))
            blkA = iA[sA:sA + block_m]; blkD = iD[sD:sD + block_m]
            # plain block burdens (the form the per-pair burden-product test sees)
            gA = _scale_vec(_block_burden(XA, blkA))
            gD = _scale_vec(_block_burden(XD, blkD))
            burden_prod = gA * gD
            if causal_kind == "single_snp_pair":
                ai = _pick_maf(XA, blkA, rng); di = _pick_maf(XD, blkD, rng)
                if ai is None or di is None:
                    continue
                ccol = _std_col(XA, ai) * _std_col(XD, di)
            elif causal_kind == "mixed_sign":
                m = block_m // 2
                signs_A = np.concatenate([np.ones(m), -np.ones(block_m - m)])
                signs_D = np.concatenate([np.ones(m), -np.ones(block_m - m)])
                rng.shuffle(signs_A); rng.shuffle(signs_D)
                ZA = _scale_cols(XA[:, blkA]); ZD = _scale_cols(XD[:, blkD])
                SA = ZA @ signs_A; SD = ZD @ signs_D
                # residualize signed combos against the plain burden direction so
                # cor(S_resid, burden) == 0 sample-wise (Codex requirement)
                SA -= SA.mean(); SD -= SD.mean()
                bA_dot = float(gA @ gA)
                bD_dot = float(gD @ gD)
                SA_resid = SA - (float(SA @ gA) / bA_dot) * gA if bA_dot > 1e-12 else SA
                SD_resid = SD - (float(SD @ gD) / bD_dot) * gD if bD_dot > 1e-12 else SD
                ccol = _scale_vec(SA_resid) * _scale_vec(SD_resid)
            else:  # interaction (gA*gD, on-model) or main_only (gA+gD)
                ccol = (gA + gD) if causal_kind == "main_only" else burden_prod
            cols.append(ccol)
            if diag_out is not None:
                # projection of true causal signal onto burden-product form (Codex
                # TOP_RISK: without this, "power dropped" can't separate "burden blind
                # to existing signal" from "we built a near-undetectable signal").
                bp_c = burden_prod - burden_prod.mean(); cc_c = ccol - ccol.mean()
                sd1, sd2 = bp_c.std(ddof=0), cc_c.std(ddof=0)
                cor = float((bp_c @ cc_c) / (sd1 * sd2 * bp_c.size)) if sd1 > 1e-12 and sd2 > 1e-12 else 0.0
                diag_out.append(dict(causal_kind=causal_kind, cor_with_burden_prod=cor))
    return np.column_stack(cols) if cols else np.zeros((XA.shape[0], 1))


def _chol(K):
    n = K.shape[0]
    for j in (1e-8, 1e-6, 1e-4):
        try:
            return np.linalg.cholesky(K + j * np.eye(n))
        except np.linalg.LinAlgError:
            continue
    w, V = np.linalg.eigh(0.5 * (K + K.T)); return V * np.sqrt(np.clip(w, 0, None))


def _one_rep(K, cholA, cholD, Uc, pve_pair, seed):
    """K: dict with A,D,hom,pair,shuf,shuf_wc,mainA,mainD. Uc: n×C causal feature
    matrix (window products or SNP-pair products). Fits 8 models."""
    rng = np.random.default_rng(seed)
    n = K["A"].shape[0]
    sa = ADDITIVE_PVE / 2
    y = np.sqrt(sa) * (cholA @ rng.standard_normal(n)) + np.sqrt(sa) * (cholD @ rng.standard_normal(n))
    if pve_pair > 0:
        s = _scale_vec(Uc @ rng.standard_normal(Uc.shape[1]))
        y += s * np.sqrt(pve_pair)
    y += np.sqrt(max(1 - ADDITIVE_PVE - pve_pair, 1e-6)) * rng.standard_normal(n)
    X = np.ones((n, 1)); rs = int(seed) % 100000
    try:
        ll_add = float(fit_multi_reml(y, X, {"A": K["A"], "D": K["D"]},
                                      n_starts=3, random_state=rs).log_lik)
        out = {"ok": True}
        # dilution test: extra kernel vs additive-only null
        for tag in ("hom", "pair", "shuf", "shuf_wc", "oracle"):
            f = fit_multi_reml(y, X, {"A": K["A"], "D": K["D"], tag: K[tag]},
                               n_starts=3, random_state=rs)
            out[f"lrt_{tag}"] = max(0.0, 2.0 * (float(f.log_lik) - ll_add))
            out[f"pve_{tag}"] = float(f.pve.get(tag, 0.0))
        # interaction-specificity: extra kernel vs (additive + local main effects) null
        base = {"A": K["A"], "D": K["D"], "mainA": K["mainA"], "mainD": K["mainD"]}
        ll_main = float(fit_multi_reml(y, X, base, n_starts=3, random_state=rs).log_lik)
        for tag in ("pair", "shuf_wc"):
            f = fit_multi_reml(y, X, {**base, tag: K[tag]}, n_starts=3, random_state=rs)
            out[f"lrt_{tag}_cond"] = max(0.0, 2.0 * (float(f.log_lik) - ll_main))
            out[f"pve_{tag}_cond"] = float(f.pve.get(tag, 0.0))
        return out
    except Exception:  # noqa: BLE001
        return {"ok": False}


def run(panel, n_rep, null_rep, n_jobs, truth_mode, tag, block_m, w_windows, block_align, out_dir):
    cfg = PANELS[panel]
    print(f"=== D2 arm-2 {panel} trait={cfg['trait']} ===")
    t0 = time.time()
    D = _load(cfg)
    n = D["n"]
    BA, labA, winA = _burden_matrix(D["XA"], D["chromA"], D["posA"], cfg["chrom_fmt"],
                                    cfg["sub_A"], cfg["n_chrom"], w_windows, MIN_SNP_PER_WIN)
    BD, labD, winD = _burden_matrix(D["XD"], D["chromD"], D["posD"], cfg["chrom_fmt"],
                                    cfg["sub_D"], cfg["n_chrom"], w_windows, MIN_SNP_PER_WIN)
    K_pair, Z_true, pair_idx = _pair_kernel(BA, labA, BD, labD)
    G = Z_true.shape[1]
    rng0 = np.random.default_rng(2026)
    jA_order = [p[0] for p in pair_idx]; jD_order = [p[1] for p in pair_idx]
    BA_m = BA[:, jA_order]; BD_m = BD[:, jD_order]                 # matched-order burdens
    chrom_num = np.array([int("".join(c for c in labA[j][0] if c.isdigit())) for j in jA_order])
    # local main-effect kernels (Bs Bs'/G), trace-normed
    def _main_kernel(B):
        Bs = _scale_cols(B)
        return normalize_kernel(Bs @ Bs.T / B.shape[1], "trace")
    K_mainA, K_mainD = _main_kernel(BA_m), _main_kernel(BD_m)
    # across-D shuffle + stricter within-chromosome shuffle
    perm = rng0.permutation(G)
    perm_wc = np.arange(G)
    for k in np.unique(chrom_num):
        idx = np.where(chrom_num == k)[0]
        perm_wc[idx] = rng0.permutation(idx)
    K_shuf = normalize_kernel(_shuffled_pair_kernel(BA_m, BD_m[:, perm]), "trace")
    K_shuf_wc = normalize_kernel(_shuffled_pair_kernel(BA_m, BD_m[:, perm_wc]), "trace")
    K_pair = normalize_kernel(K_pair, "trace")
    KA, KD, K_hom = D["KA"], D["KD"], D["K_hom"]
    cholA, cholD = _chol(KA), _chol(KD)
    K = {"A": KA, "D": KD, "hom": K_hom, "pair": K_pair, "shuf": K_shuf,
         "shuf_wc": K_shuf_wc, "mainA": K_mainA, "mainD": K_mainD}
    iu = np.triu_indices(n, 1)
    corr = lambda P, Q: float(np.corrcoef(P[iu], Q[iu])[0, 1])  # noqa: E731
    kcorr = dict(pair_hom=corr(K_pair, K_hom), pair_A=corr(K_pair, KA),
                 pair_mainA=corr(K_pair, K_mainA), pair_mainD=corr(K_pair, K_mainD),
                 pair_shuf=corr(K_pair, K_shuf), pair_shufwc=corr(K_pair, K_shuf_wc))
    winA_m = [winA[p[0]] for p in pair_idx]; winD_m = [winD[p[1]] for p in pair_idx]
    # per-chromosome SNP order (for gene_block contiguous placement)
    def _chromsort(chrom, pos, sub):
        out = {}
        for k in range(1, cfg["n_chrom"] + 1):
            cn = cfg["chrom_fmt"].format(sub=sub, k=k)
            idx = np.where(chrom == cn)[0]
            out[k] = idx[np.argsort(pos[idx])]
        return out
    chromsortA = _chromsort(D["chromA"], D["posA"], cfg["sub_A"])
    chromsortD = _chromsort(D["chromD"], D["posD"], cfg["sub_D"])
    Uc = _causal_features(truth_mode, N_CAUSAL, D["XA"], D["XD"], winA_m, winD_m,
                          chrom_num, Z_true, rng0, chromsortA, chromsortD, block_m, block_align)
    if Uc.shape[1] == 0 or np.allclose(Uc, 0.0):
        raise SystemExit(f"ERR: no causal features (truth_mode={truth_mode} W={w_windows} "
                         f"m={block_m} align={block_align}); window likely smaller than block_m "
                         f"(D-subgenome windows shrink at large W).")
    K["oracle"] = normalize_kernel(Uc @ Uc.T / Uc.shape[1], "trace")  # ceiling: true causal features
    print(f"  n={n} G_pairs={G} truth_mode={truth_mode} C_eff={Uc.shape[1]} hom_mode={D['hom_mode']} "
          f"kcorr={ {k: round(v, 3) for k, v in kcorr.items()} } ({time.time()-t0:.1f}s)")

    LRT_KEYS = ["hom", "pair", "shuf", "shuf_wc", "oracle", "pair_cond", "shuf_wc_cond"]

    def power(reps, key, crit):
        lrt = np.array([r[key] for r in reps if r.get("ok") and key in r], dtype=np.float64)
        return float(np.mean(lrt > crit)) if lrt.size else float("nan")

    def crit95(reps, key):
        lrt = np.array([r[key] for r in reps if r.get("ok") and key in r], dtype=np.float64)
        return float(np.nanpercentile(lrt, 95)) if lrt.size else float("nan")

    def med_pve(reps, key):
        v = [r[key] for r in reps if r.get("ok") and key in r]
        return float(np.nanmedian(v)) if v else float("nan")

    out = {"panel": panel, "trait": cfg["trait"], "n": n, "G_pairs": G,
           "hom_mode": D["hom_mode"], "kernel_corr": kcorr, "n_causal": int(Uc.shape[1]),
           "n_rep": n_rep, "null_rep": null_rep, "grid": {}}
    nullr = Parallel(n_jobs=n_jobs)(delayed(_one_rep)(K, cholA, cholD, Uc, 0.0, 80000 + r)
                                    for r in range(null_rep))
    crit = {t: crit95(nullr, f"lrt_{t}") for t in LRT_KEYS}
    out["empirical_crit95"] = crit
    out["grid"]["0.00"] = {t: dict(power=power(nullr, f"lrt_{t}", crit[t])) for t in LRT_KEYS}
    print(f"  null power(~.05): { {t: round(out['grid']['0.00'][t]['power'], 3) for t in LRT_KEYS} }")
    for pve in PVE_PAIR_GRID:
        reps = Parallel(n_jobs=n_jobs)(delayed(_one_rep)(K, cholA, cholD, Uc, pve,
                                                         10000 * int(pve * 100) + r)
                                       for r in range(n_rep))
        rec = {t: dict(power=power(reps, f"lrt_{t}", crit[t]),
                       pve_median=med_pve(reps, f"pve_{t}")) for t in LRT_KEYS}
        out["grid"][f"{pve:.2f}"] = rec
        print(f"  PVE={pve:.2f} | pair={rec['pair']['power']:.2f} hom={rec['hom']['power']:.2f} "
              f"shuf={rec['shuf']['power']:.2f} shufwc={rec['shuf_wc']['power']:.2f} "
              f"oracle={rec['oracle']['power']:.2f} || pair|main={rec['pair_cond']['power']:.2f} "
              f"shufwc|main={rec['shuf_wc_cond']['power']:.2f}")
    g = out["grid"]
    dilution = any(g[p]["pair"]["power"] >= 2 * max(g[p]["hom"]["power"], 1e-9)
                   and g[p]["pair"]["power"] > g[p]["shuf_wc"]["power"] + 0.15 for p in ("0.10", "0.20"))
    interaction = any(g[p]["pair_cond"]["power"] > g[p]["shuf_wc_cond"]["power"] + 0.15
                      and g[p]["pair_cond"]["power"] > 0.3 for p in ("0.10", "0.20"))
    out["verdict"] = dict(dilution_pair_beats_global=bool(dilution),
                          interaction_specific_beyond_maineffects=bool(interaction))
    print(f"  VERDICT: dilution={dilution}  interaction_specific={interaction}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out["truth_mode"] = truth_mode
    p = out_dir / f"{panel}_d2arm2_{tag}.json"
    p.write_text(json.dumps(out, indent=2, default=float))
    print(f"  wrote {p} ({time.time()-t0:.1f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", choices=list(PANELS), default="cotton_hebau")
    ap.add_argument("--n-rep", type=int, default=200)
    ap.add_argument("--null-rep", type=int, default=500)
    ap.add_argument("--n-jobs", type=int, default=40)
    ap.add_argument("--truth-mode",
                    choices=["window", "snp_corr", "snp_noncorr", "gene_block"], default="window")
    ap.add_argument("--block-m", type=int, default=30)
    ap.add_argument("--w-windows", type=int, default=40)
    ap.add_argument("--block-align", choices=["offset", "aligned"], default="offset")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out-dir", default="results/phase7/d2_arm2")
    args = ap.parse_args()
    if args.tag:
        tag = args.tag
    elif args.truth_mode == "gene_block":
        tag = f"gb_m{args.block_m}_W{args.w_windows}_{args.block_align}"
    else:
        tag = args.truth_mode
    run(args.panel, args.n_rep, args.null_rep, args.n_jobs, args.truth_mode, tag,
        args.block_m, args.w_windows, args.block_align, ROOT / args.out_dir)


if __name__ == "__main__":
    main()
