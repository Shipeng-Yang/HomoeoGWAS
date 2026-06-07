#!/usr/bin/env python3
"""Phase 7 cotton DEFENSE: LMM parametric bootstrap guarding the two tetraploid hits.

y-shuffle permutation (04b) breaks the kinship-phenotype coupling; the rigorous null is
a parametric bootstrap that SIMULATES y from the fitted A/D subgenome LMM and re-runs the
WHOLE pipeline (REML refit + whitening + burden-product GLS). This asks: under the actual
fitted {K_A,K_D} covariance (no pair-marker effect), how often does the interaction test
reach the observed extremity?

Two endpoints (Codex):
  - pair-local: bootstrap the named hit pair's interaction p (mechanistic robustness).
  - genome-wide minP / ACAT over the SAME homoeolog-pair universe per replicate
    (post-selection calibration -> defeats the 'cherry-picked pair' circularity).
INT primary; raw = sensitivity only (Gaussian bootstrap on raw underestimates outlier
false positives). Reports exceedance counts, Monte-Carlo resolution 1/(B+1), and REML
fail rate. Does NOT fake e-5 precision: with B reps the smallest reportable p is 1/(B+1).
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

ROOT = Path("/mnt/7302share/fast_ysp/U7_GWAS")
sys.path.insert(0, str(ROOT / "scripts/phase7"))
sys.path.insert(0, str(ROOT / "scripts/phase7/bio_pilot"))
from d2_arm2_synteny_kernel import PANELS, _block_burden, _load  # noqa: E402
from d3_perpair_interaction import _acat  # noqa: E402
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "dep04b", str(ROOT / "scripts/phase7/bio_pilot/04b_multi_trait_deploy.py"))
_d4 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_d4)

OUT = ROOT / "results/phase7/bio_full"

# the two main tetraploid hits (mode, pair_class, trait, A gene, D gene)
HITS = [
    dict(name="hit1_fiber_length_A06D06", mode="body", pair_class="same",
         trait="fiber_length_BLUE", A="GhM_A06G1605", D="GhM_D06G1557"),
    dict(name="hit2_length_uniformity_A11D11", mode="flank", pair_class="same",
         trait="length_uniformity_BLUE", A="GhM_A11G2420", D="GhM_D11G2742"),
]


def _build_set(cfg, mode, pair_class):
    """Reproduce 04b burden/pair construction for a (mode, pair_class) set."""
    pairs_fp = OUT / f"pairs_{mode}_{pair_class}.tsv"
    if mode == "body":
        npz_fp = OUT / "snp_to_gene_body.npz"
    else:
        cand = list(OUT.glob("snp_to_gene_flank*bp.npz"))
        npz_fp = cand[0] if cand else OUT / "snp_to_gene_flank2000bp.npz"
    pairs_df = pd.read_csv(pairs_fp, sep="\t")
    g2s = _d4._load_snp_to_gene(npz_fp)
    D = _load(cfg)
    samples = _d4._load_pheno_aligned(cfg)
    assert len(samples) == D["n"], f"sample mismatch {len(samples)} vs {D['n']}"
    pheno = _d4._load_pheno_all(cfg, samples)
    a_genes = sorted(set(pairs_df["gene_A"])); d_genes = sorted(set(pairs_df["gene_D"]))
    BA_full = np.column_stack([_block_burden(D["XA"], g2s[g]) for g in a_genes])
    BD_full = np.column_stack([_block_burden(D["XD"], g2s[g]) for g in d_genes])
    a_idx = {g: i for i, g in enumerate(a_genes)}; d_idx = {g: i for i, g in enumerate(d_genes)}
    jA = [a_idx[g] for g in pairs_df["gene_A"]]; jD = [d_idx[g] for g in pairs_df["gene_D"]]
    gene_pairs = list(zip(pairs_df["gene_A"], pairs_df["gene_D"]))

    def _scols_safe(M):
        mu = M.mean(0); sd = M.std(0, ddof=0); sd = np.where(sd > 1e-12, sd, 1.0)
        return (M - mu) / sd
    BA_m = _scols_safe(BA_full[:, jA]); BD_m = _scols_safe(BD_full[:, jD])
    return D, BA_m, BD_m, gene_pairs, pheno


def _gls_intercept(V, y):
    """GLS intercept under V (1'V^-1 1)^-1 1'V^-1 y."""
    n = V.shape[0]
    Vi1 = np.linalg.solve(V, np.ones(n))
    return float((np.ones(n) @ np.linalg.solve(V, y)) / (np.ones(n) @ Vi1))


def _boot_rep(KA, KD, BA_m, BD_m, mu, L, hit_idx, seed):
    rng = np.random.default_rng(seed)
    y_star = mu + L @ rng.standard_normal(L.shape[0])
    try:
        Wh, _ = _d4._whiten(KA, KD, y_star, seed=seed % 100000)
        pv, *_ = _d4._perpair_t_full(Wh, y_star, BA_m, BD_m)
        if not np.all(np.isfinite(pv)):
            return dict(ok=False)
        return dict(ok=True, p_local=float(pv[hit_idx]), minp=float(pv.min()),
                    acat=float(_acat(pv)))
    except Exception:  # noqa: BLE001
        return dict(ok=False)


def _bootstrap(label, y, KA, KD, BA_m, BD_m, hit_idx, B, n_jobs):
    # observed
    Wh_obs, cv = _d4._whiten(KA, KD, y, seed=42)
    pv_obs, *_ = _d4._perpair_t_full(Wh_obs, y, BA_m, BD_m)
    p_local_obs = float(pv_obs[hit_idx]); minp_obs = float(pv_obs.min())
    acat_obs = float(_acat(pv_obs))
    # fitted null covariance V_hat from observed y's REML
    n = KA.shape[0]
    V = (cv.get("A", 0.0) * KA + cv.get("D", 0.0) * KD + max(cv.get("e", 1e-6), 1e-6) * np.eye(n))
    V = 0.5 * (V + V.T)
    mu = _gls_intercept(V, y)
    try:
        L = np.linalg.cholesky(V)
    except np.linalg.LinAlgError:
        w, Q = np.linalg.eigh(V); w = np.clip(w, 1e-12, None); L = Q * np.sqrt(w)
    mu_vec = np.full(n, mu)
    reps = Parallel(n_jobs=n_jobs)(
        delayed(_boot_rep)(KA, KD, BA_m, BD_m, mu_vec, L, hit_idx, 700000 + i) for i in range(B))
    ok = [r for r in reps if r.get("ok")]
    Bok = len(ok)
    fail_rate = 1.0 - Bok / B
    pl = np.array([r["p_local"] for r in ok]); mp = np.array([r["minp"] for r in ok])
    ac = np.array([r["acat"] for r in ok])
    exc_local = int((pl <= p_local_obs).sum())
    exc_minp = int((mp <= minp_obs).sum())
    exc_acat = int((ac <= acat_obs).sum())
    res = 1.0 / (Bok + 1)
    return dict(label=label, B_requested=B, B_ok=Bok, reml_fail_rate=round(fail_rate, 4),
                mc_resolution=res, sigma_hat=dict(A=cv.get("A", 0.0), D=cv.get("D", 0.0),
                                                  e=cv.get("e", 0.0)),
                p_local_obs=p_local_obs, exceed_local=exc_local,
                p_local_boot=(1 + exc_local) / (Bok + 1),
                minp_obs=minp_obs, exceed_minp=exc_minp,
                p_minp_boot=(1 + exc_minp) / (Bok + 1),
                acat_obs=acat_obs, exceed_acat=exc_acat,
                p_acat_boot=(1 + exc_acat) / (Bok + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B-int", type=int, default=20000)
    ap.add_argument("--B-raw", type=int, default=2000)
    ap.add_argument("--n-jobs", type=int, default=48)
    args = ap.parse_args()
    t0 = time.time()
    cfg = PANELS["cotton_hebau"]

    results = {}
    for hit in HITS:
        D, BA_m, BD_m, gene_pairs, pheno = _build_set(cfg, hit["mode"], hit["pair_class"])
        KA, KD = D["KA"], D["KD"]
        try:
            hit_idx = gene_pairs.index((hit["A"], hit["D"]))
        except ValueError:
            print(f"  {hit['name']}: PAIR NOT FOUND"); continue
        y_raw = pheno[hit["trait"]]
        y_int = _d4._rank_int(y_raw)
        G = len(gene_pairs)
        print(f"\n=== {hit['name']} | {hit['mode']}_{hit['pair_class']} G={G} "
              f"pair_idx={hit_idx} ({hit['A']}|{hit['D']}) ===")
        out_int = _bootstrap("INT", y_int, KA, KD, BA_m, BD_m, hit_idx, args.B_int, args.n_jobs)
        print(f"  INT  B_ok={out_int['B_ok']} fail={out_int['reml_fail_rate']} "
              f"| local p_obs={out_int['p_local_obs']:.2e} exceed={out_int['exceed_local']} "
              f"p_boot={out_int['p_local_boot']:.2e} | minP exceed={out_int['exceed_minp']} "
              f"p_boot={out_int['p_minp_boot']:.2e} | ACAT exceed={out_int['exceed_acat']} "
              f"p_boot={out_int['p_acat_boot']:.2e} ({time.time()-t0:.0f}s)")
        out_raw = _bootstrap("raw", y_raw, KA, KD, BA_m, BD_m, hit_idx, args.B_raw, args.n_jobs)
        print(f"  raw  B_ok={out_raw['B_ok']} (SENSITIVITY only; Gaussian bootstrap underestimates "
              f"raw outlier FP) | local p_boot={out_raw['p_local_boot']:.2e} "
              f"minP p_boot={out_raw['p_minp_boot']:.2e}")
        results[hit["name"]] = dict(config=hit, G=G, INT_primary=out_int, raw_sensitivity=out_raw)

    payload = dict(
        analysis="cotton LMM parametric bootstrap (defense of the 2 tetraploid hits)",
        null="y* = beta_hat*1 + L z, LL'=V_hat=sigmaA*K_A+sigmaD*K_D+sigmaE*I (REML on observed y); "
             "each rep re-fits REML + re-whitens + burden-product GLS (full pipeline).",
        endpoints=("pair-local (named hit interaction p) + genome-wide minP/ACAT over same "
                   "homoeolog-pair universe per replicate (post-selection calibration)."),
        framing=("INT primary; raw sensitivity only. Smallest reportable p = 1/(B+1); "
                 "0 exceedances => p_boot < resolution (concordant with analytic), confirming "
                 "the signal is anomalous under the FITTED kinship covariance, not merely under "
                 "label exchangeability. Plug-in V_hat (variance-component uncertainty not "
                 "integrated); Gaussian null (raw heavy tails not reproduced)."),
        results=results)
    fp = OUT / "cotton_bootstrap_defense.json"
    fp.write_text(json.dumps(payload, indent=2, default=float))
    print(f"\n  wrote {fp} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
