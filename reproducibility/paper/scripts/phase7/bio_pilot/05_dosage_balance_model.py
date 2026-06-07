#!/usr/bin/env python3
"""Phase D Step 1 (cotton pilot): homoeolog DOSAGE-BALANCE + COMPENSATION model.

Reparameterizes the abstract pairwise interaction into biologically interpretable,
orthogonal coordinates (gene-balance-hypothesis aligned):
    S = (z_A + z_D)/sqrt2   total-dosage axis  (joint dosage effect)
    I = (z_A - z_D)/sqrt2   imbalance axis      (which copy dominates; dosage-sensitivity)
    Q = z_A * z_D            non-linear epistasis (== the original b_A*b_D term)
Since z_A, z_D are standardized (var=1), cov(S,I)=0 -> S ⟂ I exactly; variance is
additively partitionable. Per-pair whitened GLS y_w ~ 1 + S + I + Q gives three near-
orthogonal partial t-tests. We then:
  - ACAT-aggregate each component across pairs -> "total-dosage / imbalance / epistasis
    architecture" omnibus per trait
  - per-component lambda_GC + B-perm empirical type-I (calibration)
  - architecture label per pair (which component dominates)
  - COMPENSATION classification for pairs with signal: buffering (sub-additive, Q opposes
    main) / complementary (super-additive, Q aligns) / antagonistic (b_A,b_D opposite sign)
  - re-interpret the known cotton hits (A06-D06 fiber_length, A11-D11 length_uniformity):
    are they dosage-driven, imbalance-driven, or epistasis-driven?

Reuses cotton loaders from bio_pilot 03/04b.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats

ROOT = Path("/mnt/7302share/fast_ysp/U7_GWAS")
sys.path.insert(0, str(ROOT / "scripts/phase7"))
sys.path.insert(0, str(ROOT / "scripts/phase7/bio_pilot"))
from d2_arm2_synteny_kernel import PANELS, _block_burden, _load  # noqa: E402
from d3_perpair_interaction import _acat, _binom_ci95, _lambda_gc  # noqa: E402

OUT = ROOT / "results/phase7/bio_full"
BIO = ROOT / "results/phase7/bio_full"
_S2 = np.sqrt(2.0)


def _load_snp_to_gene(path):
    z = np.load(path, allow_pickle=True)
    return {gid: np.asarray(idx, int)
            for gid, idx in zip(z["gene_ids"].tolist(), z["snp_idx"].tolist())}


def _load_pheno_aligned(cfg):
    from homoeogwas.io import load_bed_hardcall
    bedA = load_bed_hardcall(ROOT / cfg["bed_root"] / cfg["bed_name"].format(sub=cfg["sub_A"]))
    bedD = load_bed_hardcall(ROOT / cfg["bed_root"] / cfg["bed_name"].format(sub=cfg["sub_D"]))
    sA = list(np.asarray(bedA.samples)); sD = list(np.asarray(bedD.samples))
    ref = sA if sA == sD else sorted(set(sA) & set(sD))
    ph = pd.read_csv(ROOT / cfg["pheno"], sep="\t").set_index(cfg["sample_col"])
    common = [s for s in ref if s in ph.index and pd.notna(ph.loc[s, "fiber_length_BLUE"])]
    return common, ph


def _whiten(KA, KD, y, seed=42):
    from homoeogwas.lmm import fit_multi_reml
    res = fit_multi_reml(y, np.ones((KA.shape[0], 1)), {"A": KA, "D": KD},
                         n_starts=3, random_state=int(seed) % 100000)
    cv = res.component_var; n = KA.shape[0]
    V = (cv.get("A", 0.0) * KA + cv.get("D", 0.0) * KD
         + max(cv.get("e", 1e-6), 1e-6) * np.eye(n))
    w, Q = np.linalg.eigh(0.5 * (V + V.T))
    w = np.clip(w, 1e-10, None)
    return (Q * (1.0 / np.sqrt(w))) @ Q.T, cv


def _scols_safe(M):
    mu = M.mean(0); sd = M.std(0, ddof=0); sd = np.where(sd > 1e-12, sd, 1.0)
    return (M - mu) / sd


def _siq_decompose(Wh, y, ZA, ZD):
    """Per-pair S/I/Q decomposition. Returns dict of arrays (length G):
    pS,pI,pQ (component partial p), bS,bI,bQ (coefs), and classification fields."""
    n, G = ZA.shape
    yw = Wh @ y
    one_w = Wh @ np.ones(n)
    df = n - 4
    pS = np.empty(G); pI = np.empty(G); pQ = np.empty(G)
    bS = np.empty(G); bI = np.empty(G); bQ = np.empty(G)
    bA = np.empty(G); bD = np.empty(G)
    for g in range(G):
        zA = ZA[:, g]; zD = ZD[:, g]
        S = (zA + zD) / _S2; I = (zA - zD) / _S2; Qc = zA * zD
        Sw = Wh @ S; Iw = Wh @ I; Qw = Wh @ Qc
        X = np.column_stack([one_w, Sw, Iw, Qw])
        beta, *_ = np.linalg.lstsq(X, yw, rcond=None)
        resid = yw - X @ beta
        s2 = float(resid @ resid) / df
        XtX_inv = np.linalg.pinv(X.T @ X)
        se = np.sqrt(np.maximum(s2 * np.diag(XtX_inv), 1e-30))
        t = beta / se
        pS[g] = 2 * stats.t.sf(abs(t[1]), df)
        pI[g] = 2 * stats.t.sf(abs(t[2]), df)
        pQ[g] = 2 * stats.t.sf(abs(t[3]), df)
        bS[g], bI[g], bQ[g] = beta[1], beta[2], beta[3]
        # recover b_A, b_D main effects (S = (A+D)/√2, I = (A-D)/√2 ->
        #   A coef = (bS + bI)/√2, D coef = (bS - bI)/√2)
        bA[g] = (beta[1] + beta[2]) / _S2
        bD[g] = (beta[1] - beta[2]) / _S2
    return dict(pS=pS, pI=pI, pQ=pQ, bS=bS, bI=bI, bQ=bQ, bA=bA, bD=bD)


def _classify(rec, g, thr=0.05):
    """Architecture + compensation label for pair g."""
    pS, pI, pQ = rec["pS"][g], rec["pI"][g], rec["pQ"][g]
    comps = {"dosage_S": pS, "imbalance_I": pI, "epistasis_Q": pQ}
    dom = min(comps, key=comps.get)
    arch = dom if comps[dom] < thr else "none"
    # compensation direction (only meaningful if Q signal present)
    comp = "NA"
    if pQ < thr:
        bA, bD, bQ = rec["bA"][g], rec["bD"][g], rec["bQ"][g]
        if np.sign(bA) != np.sign(bD) and abs(bA) > 1e-9 and abs(bD) > 1e-9:
            comp = "antagonistic"
        elif np.sign(bQ) == np.sign(bA + bD):
            comp = "complementary_superadditive"
        else:
            comp = "buffering_subadditive"
    return arch, comp


def _perm_rep(KA, KD, ZA, ZD, y_shuf, seed):
    try:
        Wh, _ = _whiten(KA, KD, y_shuf, seed=seed)
        rec = _siq_decompose(Wh, y_shuf, ZA, ZD)
        return dict(ok=True,
                    acatS=_acat(rec["pS"]), acatI=_acat(rec["pI"]), acatQ=_acat(rec["pQ"]),
                    lamS=_lambda_gc(rec["pS"]), lamI=_lambda_gc(rec["pI"]), lamQ=_lambda_gc(rec["pQ"]))
    except Exception:  # noqa: BLE001
        return dict(ok=False)


def _rank_int(y):
    n = y.size; r = stats.rankdata(y, method="average")
    return stats.norm.ppf((r - 0.5) / n)


def run(panel, mode, trait, perm_B, n_jobs):
    t0 = time.time()
    cfg = PANELS[panel]
    pairs_fp = BIO / f"pairs_{mode}_same.tsv"
    npz_fp = BIO / ("snp_to_gene_body.npz" if mode == "body"
                    else list(BIO.glob("snp_to_gene_flank*bp.npz"))[0].name)
    pairs_df = pd.read_csv(pairs_fp, sep="\t")
    g2s = _load_snp_to_gene(npz_fp)
    D = _load(cfg); n = D["n"]; KA, KD = D["KA"], D["KD"]
    samples, ph = _load_pheno_aligned(cfg)
    a_genes = sorted(set(pairs_df["gene_A"])); d_genes = sorted(set(pairs_df["gene_D"]))
    BA_full = np.column_stack([_block_burden(D["XA"], g2s[g]) for g in a_genes])
    BD_full = np.column_stack([_block_burden(D["XD"], g2s[g]) for g in d_genes])
    a_idx = {g: i for i, g in enumerate(a_genes)}; d_idx = {g: i for i, g in enumerate(d_genes)}
    jA = [a_idx[g] for g in pairs_df["gene_A"]]; jD = [d_idx[g] for g in pairs_df["gene_D"]]
    gene_pairs = list(zip(pairs_df["gene_A"], pairs_df["gene_D"]))
    ZA = _scols_safe(BA_full[:, jA]); ZD = _scols_safe(BD_full[:, jD])
    G = ZA.shape[1]
    y_raw = np.array([float(ph.loc[s, trait]) for s in samples])
    print(f"=== Dosage-balance model: {panel} {mode}_same G={G} trait={trait} n={n} ===")

    results = {}
    for label, y in [("INT", _rank_int(y_raw)), ("raw", y_raw)]:
        Wh, cv = _whiten(KA, KD, y)
        rec = _siq_decompose(Wh, y, ZA, ZD)
        acat = dict(S=float(_acat(rec["pS"])), I=float(_acat(rec["pI"])), Q=float(_acat(rec["pQ"])))
        lam = dict(S=float(_lambda_gc(rec["pS"])), I=float(_lambda_gc(rec["pI"])), Q=float(_lambda_gc(rec["pQ"])))
        # architecture composition
        archs = [_classify(rec, g)[0] for g in range(G)]
        comps = [_classify(rec, g)[1] for g in range(G)]
        from collections import Counter
        arch_comp = dict(Counter(archs)); comp_comp = dict(Counter(c for c in comps if c != "NA"))
        bonf = 0.05 / G
        # per-component Bonferroni hits
        comp_hits = {}
        for cn, pv in [("S", rec["pS"]), ("I", rec["pI"]), ("Q", rec["pQ"])]:
            idx = [int(i) for i in np.argsort(pv) if pv[i] < bonf]
            comp_hits[cn] = [dict(pair=gene_pairs[i], p=float(pv[i]),
                                  bS=float(rec["bS"][i]), bI=float(rec["bI"][i]), bQ=float(rec["bQ"][i]),
                                  arch=_classify(rec, i)[0], comp=_classify(rec, i)[1]) for i in idx]
        # permutation null per component
        rng = np.random.default_rng(2027)
        perms = [rng.permutation(n) for _ in range(perm_B)]
        pr = Parallel(n_jobs=n_jobs)(delayed(_perm_rep)(KA, KD, ZA, ZD, y[perms[i]], 900000 + i)
                                     for i in range(perm_B))
        pr = [r for r in pr if r.get("ok")]; Bok = len(pr)
        emp = {}
        for cn in ("S", "I", "Q"):
            ap = np.array([r[f"acat{cn}"] for r in pr])
            lp = np.median([r[f"lam{cn}"] for r in pr])
            emp[cn] = dict(acat_emp=float((1 + int((ap <= acat[cn]).sum())) / (Bok + 1)),
                           lambda_perm_median=float(lp))
        results[label] = dict(acat=acat, acat_emp={k: emp[k]["acat_emp"] for k in emp},
                              lambda_obs=lam, lambda_perm={k: emp[k]["lambda_perm_median"] for k in emp},
                              arch_composition=arch_comp, comp_composition=comp_comp,
                              bonferroni_alpha=float(bonf), component_hits=comp_hits, perm_B=Bok,
                              sigma_hat={k: float(cv.get(k, 0.0)) for k in ("A", "D", "e")})
        print(f"  [{label}] ACAT  S={acat['S']:.4g}(emp {emp['S']['acat_emp']:.4g}) "
              f"I={acat['I']:.4g}(emp {emp['I']['acat_emp']:.4g}) Q={acat['Q']:.4g}(emp {emp['Q']['acat_emp']:.4g})")
        print(f"       λ_perm S={emp['S']['lambda_perm_median']:.3f} I={emp['I']['lambda_perm_median']:.3f} Q={emp['Q']['lambda_perm_median']:.3f}")
        print(f"       arch composition: {arch_comp}")
        print(f"       compensation (Q-sig pairs): {comp_comp}")
        for cn in ("S", "I", "Q"):
            for h in comp_hits[cn]:
                print(f"       {cn}-hit: {h['pair'][0]}|{h['pair'][1]} p={h['p']:.3g} arch={h['arch']} comp={h['comp']}")

    out = dict(panel=panel, mode=mode, trait=trait, n=n, G=G, results=results,
               note=("Homoeolog dosage-balance + compensation model. Per-pair whitened GLS "
                     "y~1+S+I+Q with S=(zA+zD)/√2 (total dosage), I=(zA-zD)/√2 (imbalance, "
                     "gene-balance-hypothesis axis), Q=zA*zD (non-linear epistasis). S⟂I "
                     "exactly (var(zA)=var(zD)=1 -> cov=0). Reparameterizes abstract "
                     "interaction into interpretable polyploid dosage mechanisms; reports "
                     "architecture composition (which component dominates each pair) + "
                     "compensation class (buffering/complementary/antagonistic) for Q-signal "
                     "pairs. Calibration: per-component ACAT empirical type-I (B-perm) + "
                     "lambda_GC. Low failure risk: produces architecture decomposition even "
                     "when few pairs pass Bonferroni."))
    fp = OUT / f"dosage_balance_{panel}_{mode}_{trait}.json"
    fp.write_text(json.dumps(out, indent=2, default=float))
    print(f"  wrote {fp} ({time.time()-t0:.1f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="cotton_hebau")
    ap.add_argument("--mode", choices=["body", "flank"], default="body")
    ap.add_argument("--trait", default="fiber_length_BLUE")
    ap.add_argument("--perm-B", type=int, default=2000)
    ap.add_argument("--n-jobs", type=int, default=48)
    args = ap.parse_args()
    run(args.panel, args.mode, args.trait, args.perm_B, args.n_jobs)


if __name__ == "__main__":
    main()
