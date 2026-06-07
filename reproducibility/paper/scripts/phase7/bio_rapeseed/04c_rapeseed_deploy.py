#!/usr/bin/env python3
"""Phase B rapeseed Step 4: multi-trait deployment with PER-TRAIT NA alignment.

Self-contained loader (doesn't extend d2_arm2 PANELS to keep cotton flow clean).
Per-trait alignment: for each trait, intersect non-NA samples with genotype samples
and rebuild K_A, K_C, B_A, B_C matrices on that subset. n varies per trait (297-405
non-NA out of 428). Same INT primary / raw sensitivity / B=2000 perm framework.
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
from d2_arm2_synteny_kernel import _block_burden  # noqa: E402
from d3_perpair_interaction import (  # noqa: E402
    _acat, _binom_ci95, _lambda_gc, _perpair_pvals,
)

ALL_TRAITS = ["winter_survival", "growth_habit", "bloom_50pct", "fall_vigor", "bloom",
              "ave_bloom_date", "stem_elongation", "growth_habit_2", "ave_growth_habit",
              "fall_stand", "plant_height"]

CFG = dict(
    bed_A=ROOT / "data/processed/rapeseed/horvath/A/all",
    bed_C=ROOT / "data/processed/rapeseed/horvath/C/all",
    pheno=ROOT / "data/processed/rapeseed/horvath/pheno_clean.tsv",
    sample_col="sample", maf_min=0.01,
)


def _load_rapeseed_genotypes():
    """Load A/C dosages + sample lists (intersect)."""
    from homoeogwas.io import load_bed_hardcall
    bedA = load_bed_hardcall(CFG["bed_A"])
    bedC = load_bed_hardcall(CFG["bed_C"])
    sA = list(np.asarray(bedA.samples)); sC = list(np.asarray(bedC.samples))
    ref = sA if sA == sC else sorted(set(sA) & set(sC))
    iA = np.array([sA.index(s) for s in ref]); iC = np.array([sC.index(s) for s in ref])
    XA = np.asarray(bedA.dosage, dtype=np.float64)[iA]
    XC = np.asarray(bedC.dosage, dtype=np.float64)[iC]
    return ref, XA, XC, bedA, bedC


def _build_kinships_per_subset(bedA, bedC, samples_in_subset, all_samples):
    """Subset K_A, K_C to a per-trait sample subset (trace-normed)."""
    from homoeogwas.grm import compute_grm
    from homoeogwas.kernel import normalize_kernel
    KA_full, _ = compute_grm(bedA, maf_min=CFG["maf_min"])
    KC_full, _ = compute_grm(bedC, maf_min=CFG["maf_min"])
    idx_in_full = np.array([all_samples.index(s) for s in samples_in_subset])
    KA = KA_full[np.ix_(idx_in_full, idx_in_full)]
    KC = KC_full[np.ix_(idx_in_full, idx_in_full)]
    return normalize_kernel(KA, "trace"), normalize_kernel(KC, "trace")


def _load_snp_to_gene(path):
    z = np.load(path, allow_pickle=True)
    return {gid: np.asarray(idx, int)
            for gid, idx in zip(z["gene_ids"].tolist(), z["snp_idx"].tolist())}


def _whiten(KA, KC, y, seed=1):
    from homoeogwas.lmm import fit_multi_reml
    res = fit_multi_reml(y, np.ones((KA.shape[0], 1)), {"A": KA, "C": KC},
                         n_starts=3, random_state=int(seed) % 100000)
    cv = res.component_var; n = KA.shape[0]
    V = (cv.get("A", 0.0) * KA + cv.get("C", 0.0) * KC
         + max(cv.get("e", 1e-6), 1e-6) * np.eye(n))
    w, Q = np.linalg.eigh(0.5 * (V + V.T))
    w = np.clip(w, 1e-10, None)
    return (Q * (1.0 / np.sqrt(w))) @ Q.T, cv


def _perpair_t_full(Wh, y, BA_m, BC_m):
    n, G = BA_m.shape
    yw = Wh @ y; one_w = Wh @ np.ones(n)
    BAw = Wh @ BA_m; BCw = Wh @ BC_m; INTw = Wh @ (BA_m * BC_m)
    df = n - 4
    pv = np.empty(G); tv = np.empty(G); bv = np.empty(G); sev = np.empty(G)
    for g in range(G):
        Xw = np.column_stack([one_w, BAw[:, g], BCw[:, g], INTw[:, g]])
        beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
        resid = yw - Xw @ beta
        s2 = float(resid @ resid) / df
        XtX_inv = np.linalg.pinv(Xw.T @ Xw)
        se = np.sqrt(max(s2 * XtX_inv[3, 3], 1e-30))
        t = beta[3] / se
        pv[g] = 2.0 * stats.t.sf(abs(t), df)
        tv[g] = t; bv[g] = beta[3]; sev[g] = se
    return pv, tv, bv, sev


def _perm_rep(KA, KC, BA_m, BC_m, y_shuffled, seed):
    try:
        Wh, _ = _whiten(KA, KC, y_shuffled, seed=seed)
        p = _perpair_pvals(Wh, y_shuffled, BA_m, BC_m)
        return dict(ok=True, acat=_acat(p), lam=_lambda_gc(p), minp=float(p.min()))
    except Exception:  # noqa: BLE001
        return dict(ok=False)


def _rank_int(y):
    n = y.size; ranks = stats.rankdata(y, method="average")
    return stats.norm.ppf((ranks - 0.5) / n)


def deploy(y, label, KA, KC, BA_m, BC_m, gene_pairs, n_jobs, B):
    n, G = BA_m.shape
    Wh, cv = _whiten(KA, KC, y, seed=42)
    pv, tv, bv, sev = _perpair_t_full(Wh, y, BA_m, BC_m)
    p_acat_obs = _acat(pv); minp_obs = float(pv.min()); lam_obs = _lambda_gc(pv)
    bonf = 0.05 / G
    sig_bonf_idx = [int(i) for i in np.argsort(pv) if pv[i] < bonf]
    rng = np.random.default_rng(2027)
    perms = [rng.permutation(n) for _ in range(B)]
    res = Parallel(n_jobs=n_jobs)(
        delayed(_perm_rep)(KA, KC, BA_m, BC_m, y[perms[i]], 900000 + i) for i in range(B))
    res = [r for r in res if r.get("ok")]
    Bok = len(res)
    p_perm = np.array([r["acat"] for r in res])
    minp_perm = np.array([r["minp"] for r in res])
    lam_perm = np.array([r["lam"] for r in res])
    p_acat_emp = (1 + int((p_perm <= p_acat_obs).sum())) / (Bok + 1)
    minp_emp = (1 + int((minp_perm <= minp_obs).sum())) / (Bok + 1)
    sig_bonf = []
    for i in sig_bonf_idx:
        a, c = gene_pairs[i]
        sig_bonf.append(dict(idx=int(i), gene_A=a, gene_C=c, p=float(pv[i]),
                             t=float(tv[i]), beta=float(bv[i]), se=float(sev[i])))
    top5_idx = np.argsort(pv)[:5]
    top5 = [dict(idx=int(i), gene_A=gene_pairs[i][0], gene_C=gene_pairs[i][1],
                 p=float(pv[i]), t=float(tv[i]), beta=float(bv[i])) for i in top5_idx]
    return dict(label=label, G=int(G), n=int(n), perm_B=int(Bok),
                sigma_hat=dict(A=float(cv.get("A", 0.0)), C=float(cv.get("C", 0.0)),
                               e=float(cv.get("e", 0.0))),
                p_acat_obs=float(p_acat_obs), p_acat_emp=float(p_acat_emp),
                minp_obs=minp_obs, minp_emp=float(minp_emp),
                lambda_gc_obs=float(lam_obs),
                lambda_gc_perm_median=float(np.median(lam_perm)),
                lambda_gc_perm_iqr=[float(np.percentile(lam_perm, 25)),
                                    float(np.percentile(lam_perm, 75))],
                bonferroni_alpha=float(bonf), n_sig_bonferroni=len(sig_bonf),
                sig_bonferroni=sig_bonf, top5=top5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["body", "flank"], required=True)
    ap.add_argument("--pair-class", choices=["same", "cross"], default="same")
    ap.add_argument("--traits", default=",".join(ALL_TRAITS))
    ap.add_argument("--perm-B", type=int, default=2000)
    ap.add_argument("--n-jobs", type=int, default=48)
    ap.add_argument("--bio-dir", default=str(ROOT / "results/phase7/bio_rapeseed"))
    args = ap.parse_args()
    bio = Path(args.bio_dir); t0 = time.time()

    pairs_fp = bio / f"pairs_{args.mode}_{args.pair_class}.tsv"
    if args.mode == "body":
        npz_fp = bio / "snp_to_gene_body.npz"
    else:
        cand = list(bio.glob("snp_to_gene_flank*bp.npz"))
        npz_fp = cand[0] if cand else bio / "snp_to_gene_flank2000bp.npz"
    print(f"=== Phase B rapeseed deploy mode={args.mode} class={args.pair_class} ===")
    pairs_df = pd.read_csv(pairs_fp, sep="\t")
    G = len(pairs_df)
    if G < 2:
        print(f"  ABORT: only {G} pairs."); return
    g2s = _load_snp_to_gene(npz_fp)

    # genotypes once
    samples_full, XA_full, XC_full, bedA, bedC = _load_rapeseed_genotypes()
    # phenotype
    ph = pd.read_csv(CFG["pheno"], sep="\t").set_index(CFG["sample_col"])
    print(f"  rapeseed genotype n_intersect={len(samples_full)} (A {XA_full.shape[1]} SNPs, C {XC_full.shape[1]} SNPs); "
          f"G={G} pairs ({time.time()-t0:.1f}s)")

    # build full burden once per subgenome (samples_full order)
    a_genes = sorted(set(pairs_df["gene_A"]))
    c_genes = sorted(set(pairs_df["gene_C"]))
    BA_full = np.column_stack([_block_burden(XA_full, g2s[g]) for g in a_genes])
    BC_full = np.column_stack([_block_burden(XC_full, g2s[g]) for g in c_genes])

    # GRM full -> trace-normed (subset per trait below)
    from homoeogwas.grm import compute_grm
    from homoeogwas.kernel import normalize_kernel
    KA_full_raw, _ = compute_grm(bedA, maf_min=CFG["maf_min"])
    KC_full_raw, _ = compute_grm(bedC, maf_min=CFG["maf_min"])
    # map bed sample order -> samples_full order
    sA_bed = list(np.asarray(bedA.samples)); sC_bed = list(np.asarray(bedC.samples))
    iA_full_in_bedA = np.array([sA_bed.index(s) for s in samples_full])
    iC_full_in_bedC = np.array([sC_bed.index(s) for s in samples_full])
    KA_full_aligned = KA_full_raw[np.ix_(iA_full_in_bedA, iA_full_in_bedA)]
    KC_full_aligned = KC_full_raw[np.ix_(iC_full_in_bedC, iC_full_in_bedC)]
    print(f"  K_A K_C built ({time.time()-t0:.1f}s)")

    traits = [t.strip() for t in args.traits.split(",") if t.strip()]
    by_trait = {}
    for t in traits:
        # per-trait NA alignment
        valid_samples = [s for s in samples_full if s in ph.index and pd.notna(ph.loc[s, t])]
        if len(valid_samples) < 50:
            print(f"  skip {t}: only {len(valid_samples)} non-NA samples"); continue
        # subset all matrices
        idx_in_full = np.array([samples_full.index(s) for s in valid_samples])
        n_t = len(valid_samples)
        KA_t = normalize_kernel(KA_full_aligned[np.ix_(idx_in_full, idx_in_full)], "trace")
        KC_t = normalize_kernel(KC_full_aligned[np.ix_(idx_in_full, idx_in_full)], "trace")
        a_idx = {g: i for i, g in enumerate(a_genes)}
        c_idx = {g: i for i, g in enumerate(c_genes)}
        jA = [a_idx[g] for g in pairs_df["gene_A"]]
        jC = [c_idx[g] for g in pairs_df["gene_C"]]
        gene_pairs = list(zip(pairs_df["gene_A"], pairs_df["gene_C"]))
        BA_sub = BA_full[idx_in_full][:, jA]
        BC_sub = BC_full[idx_in_full][:, jC]

        def _scols_safe(M):
            mu = M.mean(0); sd = M.std(0, ddof=0)
            sd = np.where(sd > 1e-12, sd, 1.0)
            return (M - mu) / sd
        BA_m = _scols_safe(BA_sub); BC_m = _scols_safe(BC_sub)
        y = np.array([float(ph.loc[s, t]) for s in valid_samples])
        if not np.all(np.isfinite(y)) or y.std() < 1e-9:
            print(f"  skip {t}: y degenerate"); continue
        print(f"  --- trait={t} n={n_t} y(mean={y.mean():.3f} sd={y.std():.3f}) ---")
        out_INT = deploy(_rank_int(y), "INT", KA_t, KC_t, BA_m, BC_m, gene_pairs, args.n_jobs, args.perm_B)
        out_raw = deploy(y, "raw", KA_t, KC_t, BA_m, BC_m, gene_pairs, args.n_jobs, args.perm_B)
        by_trait[t] = dict(INT_primary=out_INT, raw_sensitivity=out_raw, n=n_t)
        print(f"    INT p_acat_obs={out_INT['p_acat_obs']:.4f} emp={out_INT['p_acat_emp']:.4f} "
              f"λ_obs={out_INT['lambda_gc_obs']:.3f} λ_perm={out_INT['lambda_gc_perm_median']:.3f} "
              f"nsig={out_INT['n_sig_bonferroni']}")
        print(f"    raw p_acat_obs={out_raw['p_acat_obs']:.4f} emp={out_raw['p_acat_emp']:.4f} "
              f"λ_obs={out_raw['lambda_gc_obs']:.3f} nsig={out_raw['n_sig_bonferroni']}")

    out = dict(panel="rapeseed_horvath", mode=args.mode, pair_class=args.pair_class,
               G=G, perm_B=args.perm_B,
               traits_run=list(by_trait.keys()), by_trait=by_trait,
               note=("rapeseed (Brassica napus AACC, disomic allopolyploid) "
                     f"cross-species replication of cotton Phase A framework. Mode={args.mode}, "
                     f"pair_class={args.pair_class}, G={G}. Per-trait NA alignment "
                     "(each trait gets its own sample subset, K_A/K_C, burden matrices). "
                     "INT primary + raw sensitivity, B=2000 y-shuffle perm. Codex TOP_RISK: "
                     "rapeseed SNP density ~10x sparser than cotton + Brassica post-WGD "
                     "fractionation reduces clean same-num collinear pairs => method-level "
                     "calibration replication achievable; discovery-level depends on flowering "
                     "biology hitting the small retained gene set. Known QTL anchors (FLC/FT) "
                     "may or may not enter the burden-product framework depending on flank/SNP coverage."))
    fp = bio / f"deploy_{args.mode}_{args.pair_class}.json"
    fp.write_text(json.dumps(out, indent=2, default=float))
    print(f"\n  wrote {fp} ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
