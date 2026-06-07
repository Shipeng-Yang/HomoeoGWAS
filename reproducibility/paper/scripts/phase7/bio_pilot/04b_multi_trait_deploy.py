#!/usr/bin/env python3
"""Phase A Step 4: multi-trait deployment on full 13-chrom bio pair sets.

Reads a pairs.tsv (from 03b: pairs_{body,flank}_{same,cross}.tsv) and the matching
snp_to_gene_{body,flank}.npz; builds BA/BD bio burdens; deploys per-pair GLS+ACAT
on each requested trait under INT primary + raw sensitivity, with B=2000 y-shuffle
empirical perm null per trait per transform.

Output: one JSON aggregating all (trait, transform) results for the given (mode, pair_class).
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
from d2_arm2_synteny_kernel import (  # noqa: E402
    PANELS, _block_burden, _load,
)
from d3_perpair_interaction import (  # noqa: E402
    _acat, _binom_ci95, _lambda_gc, _perpair_pvals,
)

ALL_TRAITS = [
    "fiber_length_BLUE", "fiber_strength_BLUE", "micronaire_BLUE", "elongation_BLUE",
    "length_uniformity_BLUE", "maturity_BLUE", "spinning_consistency_BLUE",
    "boll_weight_BLUE", "lint_percentage_BLUE", "seed_index_BLUE", "lint_index_BLUE",
    "fiber_weight_per_boll_BLUE", "fiber_density_BLUE",
]


def _load_snp_to_gene(path):
    z = np.load(path, allow_pickle=True)
    return {gid: np.asarray(idx, int)
            for gid, idx in zip(z["gene_ids"].tolist(), z["snp_idx"].tolist())}


def _load_pheno_all(cfg, samples_order):
    """Load all traits aligned to samples_order; returns dict trait -> n-vector."""
    ph = pd.read_csv(ROOT / cfg["pheno"], sep="\t").set_index(cfg["sample_col"])
    return {t: np.array([float(ph.loc[s, t]) for s in samples_order]) for t in ALL_TRAITS}


def _load_pheno_aligned(cfg):
    """Replicate d2_arm2._load filtering order to recover the sample list used by D."""
    from homoeogwas.io import load_bed_hardcall
    bedA = load_bed_hardcall(ROOT / cfg["bed_root"] / cfg["bed_name"].format(sub=cfg["sub_A"]))
    bedD = load_bed_hardcall(ROOT / cfg["bed_root"] / cfg["bed_name"].format(sub=cfg["sub_D"]))
    sA = list(np.asarray(bedA.samples)); sD = list(np.asarray(bedD.samples))
    ref = sA if sA == sD else sorted(set(sA) & set(sD))
    primary = "fiber_length_BLUE"
    ph = pd.read_csv(ROOT / cfg["pheno"], sep="\t").set_index(cfg["sample_col"])
    common = [s for s in ref if s in ph.index and pd.notna(ph.loc[s, primary])]
    return common


def _whiten(KA, KD, y, seed=1):
    from homoeogwas.lmm import fit_multi_reml
    res = fit_multi_reml(y, np.ones((KA.shape[0], 1)), {"A": KA, "D": KD},
                         n_starts=3, random_state=int(seed) % 100000)
    cv = res.component_var
    n = KA.shape[0]
    V = (cv.get("A", 0.0) * KA + cv.get("D", 0.0) * KD
         + max(cv.get("e", 1e-6), 1e-6) * np.eye(n))
    w, Q = np.linalg.eigh(0.5 * (V + V.T))
    w = np.clip(w, 1e-10, None)
    return (Q * (1.0 / np.sqrt(w))) @ Q.T, cv


def _perpair_t_full(Wh, y, BA_m, BD_m):
    n, G = BA_m.shape
    yw = Wh @ y; one_w = Wh @ np.ones(n)
    BAw = Wh @ BA_m; BDw = Wh @ BD_m
    INTw = Wh @ (BA_m * BD_m)
    df = n - 4
    pv = np.empty(G); tv = np.empty(G); bv = np.empty(G); sev = np.empty(G)
    for g in range(G):
        Xw = np.column_stack([one_w, BAw[:, g], BDw[:, g], INTw[:, g]])
        beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
        resid = yw - Xw @ beta
        s2 = float(resid @ resid) / df
        XtX_inv = np.linalg.pinv(Xw.T @ Xw)
        se = np.sqrt(max(s2 * XtX_inv[3, 3], 1e-30))
        t = beta[3] / se
        pv[g] = 2.0 * stats.t.sf(abs(t), df)
        tv[g] = t; bv[g] = beta[3]; sev[g] = se
    return pv, tv, bv, sev


def _perm_rep(KA, KD, BA_m, BD_m, y_shuffled, seed):
    try:
        Wh, _ = _whiten(KA, KD, y_shuffled, seed=seed)
        p = _perpair_pvals(Wh, y_shuffled, BA_m, BD_m)
        return dict(ok=True, acat=_acat(p), lam=_lambda_gc(p), minp=float(p.min()))
    except Exception:  # noqa: BLE001
        return dict(ok=False)


def _rank_int(y):
    n = y.size
    ranks = stats.rankdata(y, method="average")
    return stats.norm.ppf((ranks - 0.5) / n)


def deploy(y, label, KA, KD, BA_m, BD_m, gene_pairs, n_jobs, B):
    n, G = BA_m.shape
    Wh, cv = _whiten(KA, KD, y, seed=42)
    pv, tv, bv, sev = _perpair_t_full(Wh, y, BA_m, BD_m)
    p_acat_obs = _acat(pv); minp_obs = float(pv.min()); lam_obs = _lambda_gc(pv)
    bonf = 0.05 / G
    sig_bonf_idx = [int(i) for i in np.argsort(pv) if pv[i] < bonf]

    rng = np.random.default_rng(2027)
    perms = [rng.permutation(n) for _ in range(B)]
    res = Parallel(n_jobs=n_jobs)(
        delayed(_perm_rep)(KA, KD, BA_m, BD_m, y[perms[i]], 900000 + i) for i in range(B))
    res = [r for r in res if r.get("ok")]
    Bok = len(res)
    p_perm = np.array([r["acat"] for r in res])
    minp_perm = np.array([r["minp"] for r in res])
    lam_perm = np.array([r["lam"] for r in res])
    p_acat_emp = (1 + int((p_perm <= p_acat_obs).sum())) / (Bok + 1)
    minp_emp = (1 + int((minp_perm <= minp_obs).sum())) / (Bok + 1)

    sig_bonf = []
    for i in sig_bonf_idx:
        a, d = gene_pairs[i]
        sig_bonf.append(dict(idx=int(i), gene_A=a, gene_D=d, p=float(pv[i]),
                             t=float(tv[i]), beta=float(bv[i]), se=float(sev[i])))
    top5_idx = np.argsort(pv)[:5]
    top5 = [dict(idx=int(i), gene_A=gene_pairs[i][0], gene_D=gene_pairs[i][1],
                 p=float(pv[i]), t=float(tv[i]), beta=float(bv[i])) for i in top5_idx]

    return dict(
        label=label, G=int(G), n=int(n), perm_B=int(Bok),
        sigma_hat=dict(A=float(cv.get("A", 0.0)), D=float(cv.get("D", 0.0)),
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
    ap.add_argument("--panel", default="cotton_hebau")
    ap.add_argument("--mode", choices=["body", "flank"], required=True)
    ap.add_argument("--pair-class", choices=["same", "cross"], default="same")
    ap.add_argument("--traits", default=",".join(ALL_TRAITS))
    ap.add_argument("--perm-B", type=int, default=2000)
    ap.add_argument("--n-jobs", type=int, default=48)
    ap.add_argument("--bio-dir", default=str(ROOT / "results/phase7/bio_full"))
    args = ap.parse_args()
    bio = Path(args.bio_dir); t0 = time.time()

    pairs_fp = bio / f"pairs_{args.mode}_{args.pair_class}.tsv"
    if args.mode == "body":
        npz_fp = bio / "snp_to_gene_body.npz"
    else:
        cand = list(bio.glob("snp_to_gene_flank*bp.npz"))
        npz_fp = cand[0] if cand else bio / "snp_to_gene_flank2000bp.npz"
    print(f"=== Phase A Step 4: deploy mode={args.mode} class={args.pair_class} ===")
    print(f"  pairs={pairs_fp.name}  snp_to_gene={npz_fp.name}")

    pairs_df = pd.read_csv(pairs_fp, sep="\t")
    G = len(pairs_df)
    if G < 5:
        print(f"  ABORT: only {G} pairs."); return
    g2s = _load_snp_to_gene(npz_fp)

    cfg = PANELS[args.panel]
    D = _load(cfg); n = D["n"]; KA, KD = D["KA"], D["KD"]
    samples = _load_pheno_aligned(cfg)
    assert len(samples) == n, f"sample-list len mismatch {len(samples)} vs {n}"
    pheno_dict = _load_pheno_all(cfg, samples)
    print(f"  cotton n={n}, G={G}; pheno loaded {len(pheno_dict)} traits ({time.time()-t0:.1f}s)")

    # build burden matrices for unique genes in the pair set
    a_genes = sorted(set(pairs_df["gene_A"]))
    d_genes = sorted(set(pairs_df["gene_D"]))
    BA_full = np.column_stack([_block_burden(D["XA"], g2s[g]) for g in a_genes])
    BD_full = np.column_stack([_block_burden(D["XD"], g2s[g]) for g in d_genes])
    a_idx = {g: i for i, g in enumerate(a_genes)}
    d_idx = {g: i for i, g in enumerate(d_genes)}
    jA = [a_idx[g] for g in pairs_df["gene_A"]]
    jD = [d_idx[g] for g in pairs_df["gene_D"]]
    gene_pairs = list(zip(pairs_df["gene_A"], pairs_df["gene_D"]))

    def _scols_safe(M):
        mu = M.mean(0); sd = M.std(0, ddof=0)
        sd = np.where(sd > 1e-12, sd, 1.0)
        return (M - mu) / sd
    BA_m = _scols_safe(BA_full[:, jA])
    BD_m = _scols_safe(BD_full[:, jD])
    print(f"  burdens built: BA={BA_full.shape}, BD={BD_full.shape}, paired={BA_m.shape} "
          f"({time.time()-t0:.1f}s)")

    traits = [t.strip() for t in args.traits.split(",") if t.strip()]
    by_trait = {}
    for t in traits:
        y = pheno_dict[t]
        print(f"\n  --- trait={t} (mean={y.mean():.3f} sd={y.std():.3f}) ---")
        out_INT = deploy(_rank_int(y), "INT", KA, KD, BA_m, BD_m, gene_pairs, args.n_jobs, args.perm_B)
        out_raw = deploy(y, "raw", KA, KD, BA_m, BD_m, gene_pairs, args.n_jobs, args.perm_B)
        by_trait[t] = dict(INT_primary=out_INT, raw_sensitivity=out_raw)
        print(f"    INT p_acat_obs={out_INT['p_acat_obs']:.4f} emp={out_INT['p_acat_emp']:.4f} "
              f"λ_obs={out_INT['lambda_gc_obs']:.3f} λ_perm={out_INT['lambda_gc_perm_median']:.3f} "
              f"nsig={out_INT['n_sig_bonferroni']}")
        print(f"    raw p_acat_obs={out_raw['p_acat_obs']:.4f} emp={out_raw['p_acat_emp']:.4f} "
              f"λ_obs={out_raw['lambda_gc_obs']:.3f} nsig={out_raw['n_sig_bonferroni']}")

    out = dict(panel=args.panel, mode=args.mode, pair_class=args.pair_class,
               n=n, G=G, perm_B=args.perm_B,
               traits_run=traits, by_trait=by_trait,
               note=("Phase A multi-trait deployment. Per-pair GLS+ACAT framework on "
                     f"{args.mode} bio bins, {args.pair_class}-number A-D synteny pairs. "
                     "Primary endpoint = INT-transformed y with B=2000 empirical permutation "
                     "p_acat_emp; raw retained as sensitivity. CAVEAT: y-shuffle perm breaks "
                     "kinship-phenotype coupling (LMM parametric bootstrap is the rigorous "
                     "replacement — Phase B). Bonferroni α/G per-trait for localization. "
                     "13-trait scan = multi-trait omnibus screen; do NOT cherry-pick "
                     "across traits without across-trait multiplicity adjustment."))
    fp = bio / f"deploy_{args.mode}_{args.pair_class}.json"
    fp.write_text(json.dumps(out, indent=2, default=float))
    print(f"\n  wrote {fp} ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
