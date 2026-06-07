#!/usr/bin/env python3
"""Phase 7 bio-pilot Step 4: REAL phenotype deployment of per-pair GLS+ACAT on
fiber_length_BLUE over the A01-D01 18-pair bio pilot.

Two endpoints:
  primary  : omnibus analytic ACAT p_acat (single test, NO G correction)
  secondary: per-pair p_g — empirical minP threshold from B=2000 y-shuffle nulls
             + Bonferroni 0.05/G as transparent backup; BH-FDR exploratory only.
Empirical perm p: p_emp = (1 + #{p^perm <= p^obs}) / (B+1) (avoids zero).
Pheno: primary = raw BLUE (env-regressed); secondary = INT (rank-normal) sensitivity.
NOTE: simple y-shuffle breaks kinship-phenotype coupling — adequate for pilot, paper
main result would need LMM parametric bootstrap. KEY RISK: G=18 pilot null
does NOT distinguish "no biology" from "search space too small".
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
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "bio_step3", str(ROOT / "scripts/phase7/bio_pilot/03_run_d3_bio.py"))
_bio3 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_bio3)
_parse_collinearity = _bio3._parse_collinearity
_one_to_one_reciprocal = _bio3._one_to_one_reciprocal
_load_snp_to_gene = _bio3._load_snp_to_gene
_build_burden_matrix = _bio3._build_burden_matrix
_whiten = _bio3._whiten


def _load_pheno(cfg):
    """Re-derive phenotype y aligned with d2_arm2._load's 'common' filter (NA-aware)."""
    from homoeogwas.io import load_bed_hardcall
    bedA = load_bed_hardcall(ROOT / cfg["bed_root"] / cfg["bed_name"].format(sub=cfg["sub_A"]))
    bedD = load_bed_hardcall(ROOT / cfg["bed_root"] / cfg["bed_name"].format(sub=cfg["sub_D"]))
    sA = list(np.asarray(bedA.samples)); sD = list(np.asarray(bedD.samples))
    ref = sA if sA == sD else sorted(set(sA) & set(sD))
    ph = pd.read_csv(ROOT / cfg["pheno"], sep="\t").set_index(cfg["sample_col"])[cfg["trait"]]
    common = [s for s in ref if s in ph.index and pd.notna(ph.loc[s])]
    y = np.array([float(ph.loc[s]) for s in common])
    return y, common


def _perpair_t_full(Wh, y, BA_m, BD_m):
    """Same as _perpair_pvals but returns BOTH p_g and t_g per pair (for diagnostics)."""
    n, G = BA_m.shape
    yw = Wh @ y
    one_w = Wh @ np.ones(n); BAw = Wh @ BA_m; BDw = Wh @ BD_m
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
    """One y-shuffle perm: refit null LMM on shuffled y, compute p_acat^perm."""
    try:
        Wh, _ = _whiten(KA, KD, y_shuffled, seed=seed)
        p = _perpair_pvals(Wh, y_shuffled, BA_m, BD_m)
        return dict(ok=True, acat=_acat(p), lam=_lambda_gc(p), minp=float(p.min()))
    except Exception:  # noqa: BLE001
        return dict(ok=False)


def _rank_int(y):
    """Rank-based inverse normal transform (Blom)."""
    n = y.size
    ranks = stats.rankdata(y, method="average")
    return stats.norm.ppf((ranks - 0.5) / n)


def deploy(y, label, KA, KD, BA_m, BD_m, gene_pairs, n_jobs, B):
    """Full deployment: observed p_acat + per-pair table + empirical perm null."""
    n, G = BA_m.shape
    print(f"\n=== DEPLOY [{label}] G={G} n={n} ===")
    # observed
    Wh, cv = _whiten(KA, KD, y, seed=42)
    pv, tv, bv, sev = _perpair_t_full(Wh, y, BA_m, BD_m)
    p_acat_obs = _acat(pv)
    minp_obs = float(pv.min())
    lam_obs = _lambda_gc(pv)
    bonf = 0.05 / G
    sig_bonf = [(i, gene_pairs[i], float(pv[i]), float(tv[i]), float(bv[i]), float(sev[i]))
                for i in np.argsort(pv) if pv[i] < bonf]
    top5 = [(i, gene_pairs[i], float(pv[i]), float(tv[i]), float(bv[i]))
            for i in np.argsort(pv)[:5]]
    print(f"  σ̂ A={cv.get('A',0):.4f} D={cv.get('D',0):.4f} e={cv.get('e',0):.4f}")
    print(f"  p_acat={p_acat_obs:.4g}  minp={minp_obs:.4g}  λ_GC(pairs)={lam_obs:.3f}")
    print(f"  Bonferroni α/G={bonf:.4g}; n_sig={len(sig_bonf)}")
    print(f"  top 5 pairs by p:")
    for i, (a, d), p, t, b in top5:
        print(f"    {a} | {d} : p={p:.4g} t={t:.2f} β={b:.3g}")

    # permutation null
    rng = np.random.default_rng(2027)
    perms = [rng.permutation(n) for _ in range(B)]
    t0 = time.time()
    res = Parallel(n_jobs=n_jobs)(delayed(_perm_rep)(KA, KD, BA_m, BD_m, y[perms[i]],
                                                     900000 + i) for i in range(B))
    res = [r for r in res if r.get("ok")]
    Bok = len(res)
    p_perm = np.array([r["acat"] for r in res])
    minp_perm = np.array([r["minp"] for r in res])
    lam_perm = np.array([r["lam"] for r in res])
    # empirical p (right tail for ACAT/minP: smaller = more significant)
    p_acat_emp = (1 + int((p_perm <= p_acat_obs).sum())) / (Bok + 1)
    minp_emp = (1 + int((minp_perm <= minp_obs).sum())) / (Bok + 1)
    print(f"  PERM B={Bok} ({time.time()-t0:.1f}s):  p_acat_emp={p_acat_emp:.4g} "
          f"(obs={p_acat_obs:.4g})  minp_emp={minp_emp:.4g}  "
          f"λ_GC_perm med={np.median(lam_perm):.3f}")

    pair_table = []
    for i in range(G):
        a, d = gene_pairs[i]
        pair_table.append(dict(idx=int(i), gene_A=a, gene_D=d,
                               p=float(pv[i]), t=float(tv[i]),
                               beta=float(bv[i]), se=float(sev[i])))
    return dict(
        label=label, n=int(n), G=int(G),
        sigma_hat=dict(A=float(cv.get("A", 0.0)), D=float(cv.get("D", 0.0)),
                       e=float(cv.get("e", 0.0))),
        p_acat_obs=float(p_acat_obs), minp_obs=minp_obs, lambda_gc_obs=float(lam_obs),
        bonferroni_alpha=float(bonf), n_sig_bonferroni=len(sig_bonf),
        sig_bonferroni=[dict(idx=int(s[0]), gene_A=s[1][0], gene_D=s[1][1],
                             p=s[2], t=s[3], beta=s[4], se=s[5]) for s in sig_bonf],
        perm_B=int(Bok), p_acat_emp=float(p_acat_emp), minp_emp=float(minp_emp),
        lambda_gc_perm_median=float(np.median(lam_perm)),
        lambda_gc_perm_iqr=[float(np.percentile(lam_perm, 25)),
                            float(np.percentile(lam_perm, 75))],
        pair_table=pair_table)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="cotton_hebau")
    ap.add_argument("--bio-dir", default=str(ROOT / "results/phase7/bio_pilot"))
    ap.add_argument("--collinearity", default="/mnt/nvme/cotton_hbau/mcscanx_pilot/pilot.collinearity")
    ap.add_argument("--perm-B", type=int, default=2000)
    ap.add_argument("--n-jobs", type=int, default=48)
    args = ap.parse_args()
    bio = Path(args.bio_dir); t0 = time.time()
    print(f"=== Phase 7 bio-pilot Step 4: REAL TRAIT deployment (panel={args.panel}) ===")

    cfg = PANELS[args.panel]
    D = _load(cfg); n = D["n"]; KA, KD = D["KA"], D["KD"]
    y, samples = _load_pheno(cfg)
    assert len(y) == n, f"y len {len(y)} != n {n}"
    print(f"  trait={cfg['trait']}  n={n}  y mean={y.mean():.3f} sd={y.std():.3f} "
          f"range=[{y.min():.3f},{y.max():.3f}]  ({time.time()-t0:.1f}s)")

    raw = _parse_collinearity(args.collinearity)
    pairs_1to1 = _one_to_one_reciprocal(raw)
    g2s = _load_snp_to_gene(bio / "snp_to_gene.npz")
    pairs_kept = [(a, d) for a, d, _ in pairs_1to1 if a in g2s and d in g2s]
    a_genes = sorted({a for a, _ in pairs_kept}); d_genes = sorted({d for _, d in pairs_kept})
    a_idx = {g: i for i, g in enumerate(a_genes)}; d_idx = {g: i for i, g in enumerate(d_genes)}
    pair_idx_bio = [(a_idx[a], d_idx[d]) for a, d in pairs_kept]
    G = len(pair_idx_bio)
    print(f"  bio pairs G={G} (A={len(a_genes)} D={len(d_genes)})")

    BA_full = _build_burden_matrix(D["XA"], a_genes, g2s)
    BD_full = _build_burden_matrix(D["XD"], d_genes, g2s)
    jA = [p[0] for p in pair_idx_bio]; jD = [p[1] for p in pair_idx_bio]

    def _scols_safe(M):
        mu = M.mean(0); sd = M.std(0, ddof=0)
        sd = np.where(sd > 1e-12, sd, 1.0)
        return (M - mu) / sd
    BA_m = _scols_safe(BA_full[:, jA])
    BD_m = _scols_safe(BD_full[:, jD])
    gene_pairs = [(a_genes[ja], d_genes[jd]) for ja, jd in pair_idx_bio]

    # primary: raw BLUE; secondary: INT
    out_blue = deploy(y, "raw_BLUE", KA, KD, BA_m, BD_m, gene_pairs, args.n_jobs, args.perm_B)
    out_int = deploy(_rank_int(y), "INT", KA, KD, BA_m, BD_m, gene_pairs, args.n_jobs, args.perm_B)

    full = dict(
        panel=args.panel, trait=cfg["trait"], n=n, G=G,
        a_genes=a_genes, d_genes=d_genes, pair_idx=pair_idx_bio,
        primary=out_blue, sensitivity_INT=out_int,
        note=("Phase 7 bio-pilot real-trait deployment on A01-D01 18-pair set "
              "(MCScanX 1:1 best reciprocal collinear A-D pairs, GFF body-only burden). "
              "Primary endpoint = omnibus analytic ACAT p_acat (single test, no G "
              "correction). Empirical permutation null (B=2000 y-shuffles, refit null "
              "LMM each) gives p_acat_emp = (1+#{≤obs})/(B+1). Per-pair Bonferroni "
              "α/G=0.0028 transparent backup; BH-FDR not used at G=18 (too small for "
              "useful FDR). Sensitivity also reports INT-transformed y. "
              "CAVEATS: (1) y-shuffle breaks kinship-phenotype coupling — pilot only, "
              "paper main result requires LMM parametric bootstrap. (2) G=18 strict "
              "reciprocal subset has very low power; null does NOT distinguish 'no "
              "biology' from 'search space too small / effect too dilute'. (3) hebau "
              "panel is SNP-array, body-only covers ~8.5% A01+D01 genes — body-only "
              "selection bias likely. (4) any candidate pair from Bonferroni hit "
              "requires LOO / MAF strata / annotation / replication before causal "
              "claim. Do NOT over-interpret G=18 pilot as genome-wide "
              "biological conclusion."))
    out_path = bio / "d3_real_trait_deploy.json"
    out_path.write_text(json.dumps(full, indent=2, default=float))
    print(f"\n  wrote {out_path} ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
