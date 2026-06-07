#!/usr/bin/env python3
"""Phase C wheat Step 2 (pilot 1A-1B-1D): position-homology triad -> 3 pairwise
(A-B, A-D, B-D) burden-product GLS + ACAT, whitened by full 3-subgenome LMM
(K_A + K_B + K_D). emergence trait. INT primary + raw sensitivity, B=2000 perm.

triad = position-homology: within each of 1A/1B/1D, retained genes sorted by start;
aligned by relative rank (smallest subgenome count drives the triad count; the others
mapped by proportional rank). This is a synteny approximation (1A/1B/1D collinear),
NOT curated ortholog. burden SNP CAP applied (WGS density -> max 890 SNP/gene; cap
prevents mean-burden from being dominated by long LD blocks).

Each pairwise X-Y: GLS y_w ~ 1 + bX + bY + bX*bY, t-test interaction; whitening from
the FULL hexaploid null LMM {K_A,K_B,K_D} (Codex: never drop the 3rd subgenome).
triad-level ACAT over the 3 pairwise p = hexaploid omnibus.
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
from d3_perpair_interaction import _acat, _lambda_gc  # noqa: E402

OUT = ROOT / "results/phase7/bio_wheat"
PHENO = ROOT / "data/processed/wheat/pheno_clean.tsv"


def _block_burden_capped(X, snp_idx, cap, rng):
    """Mean of column-standardized dosages over a gene's SNPs; if > cap SNPs,
    subsample `cap` (deterministic by rng) to avoid WGS over-dilution."""
    idx = np.asarray(snp_idx, int)
    if idx.size > cap:
        idx = np.sort(rng.choice(idx, size=cap, replace=False))
    M = X[:, idx].astype(float)
    mu = np.nanmean(M, axis=0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    Mi = np.where(np.isnan(M), mu, M)
    sd = Mi.std(0, ddof=0)
    sd_safe = np.where(sd > 1e-12, sd, 1.0)
    Z = (Mi - mu) / sd_safe
    Z[:, sd <= 1e-12] = 0.0
    return Z.mean(1)


def _grm_from_X(X):
    """Additive GRM from a dosage matrix (mean-impute, standardize, XX'/m), trace-normed."""
    mu = np.nanmean(X, axis=0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    Xi = np.where(np.isnan(X), mu, X)
    sd = Xi.std(0, ddof=0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    Z = (Xi - mu) / sd
    K = Z @ Z.T / Z.shape[1]
    return K / (np.trace(K) / K.shape[0])


def _whiten3(KA, KB, KD, y, seed=42):
    from homoeogwas.lmm import fit_multi_reml
    res = fit_multi_reml(y, np.ones((KA.shape[0], 1)), {"A": KA, "B": KB, "D": KD},
                         n_starts=3, random_state=int(seed) % 100000)
    cv = res.component_var
    n = KA.shape[0]
    V = (cv.get("A", 0.0) * KA + cv.get("B", 0.0) * KB + cv.get("D", 0.0) * KD
         + max(cv.get("e", 1e-6), 1e-6) * np.eye(n))
    w, Q = np.linalg.eigh(0.5 * (V + V.T))
    w = np.clip(w, 1e-10, None)
    return (Q * (1.0 / np.sqrt(w))) @ Q.T, cv


def _scols_safe(M):
    mu = M.mean(0)
    sd = M.std(0, ddof=0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    return (M - mu) / sd


def _pairwise_pvals(Wh, y, BX, BY):
    """Per-pair interaction p (whitened GLS t-test on bX*bY)."""
    n, G = BX.shape
    yw = Wh @ y
    one_w = Wh @ np.ones(n)
    BXw = Wh @ BX
    BYw = Wh @ BY
    INTw = Wh @ (BX * BY)
    df = n - 4
    pv = np.empty(G)
    for g in range(G):
        Xw = np.column_stack([one_w, BXw[:, g], BYw[:, g], INTw[:, g]])
        beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
        resid = yw - Xw @ beta
        s2 = float(resid @ resid) / df
        se = np.sqrt(max(s2 * np.linalg.pinv(Xw.T @ Xw)[3, 3], 1e-30))
        pv[g] = 2.0 * stats.t.sf(abs(beta[3] / se), df)
    return pv


def _rank_int(y):
    n = y.size
    r = stats.rankdata(y, method="average")
    return stats.norm.ppf((r - 0.5) / n)


def _one_pair_deploy(Wh, y, Bdict, pair, gene_pairs):
    """Return per-pair p vector + acat + lambda for one pairwise (e.g. ('A','B'))."""
    sx, sy = pair
    pv = _pairwise_pvals(Wh, y, Bdict[sx], Bdict[sy])
    return pv


def _perm_rep_triad(KA, KB, KD, Bdict, y_shuf, seed):
    try:
        Wh, _ = _whiten3(KA, KB, KD, y_shuf, seed=seed)
        out = {}
        for pair in [("A", "B"), ("A", "D"), ("B", "D")]:
            pv = _pairwise_pvals(Wh, y_shuf, Bdict[pair[0]], Bdict[pair[1]])
            out[f"{pair[0]}{pair[1]}"] = dict(acat=_acat(pv), minp=float(pv.min()), lam=_lambda_gc(pv))
        # triad omnibus = ACAT over the 3 pairwise ACATs
        out["triad"] = _acat([out["AB"]["acat"], out["AD"]["acat"], out["BD"]["acat"]])
        return dict(ok=True, **out)
    except Exception:  # noqa: BLE001
        return dict(ok=False)


def deploy(y, label, KA, KB, KD, Bdict, gene_triads, n_jobs, B, full_sink=None):
    """``full_sink`` (optional dict): when provided, the FULL per-pairwise p-vector (aligned to
    ``gene_triads`` order) is stashed under ``full_sink[tag]`` as a side channel for the genome-wide
    ranking dump. The persisted JSON is unchanged (only top5 + summary), so this is a pure logging
    addition — the same p-values, just not discarded."""
    n = KA.shape[0]
    Wh, cv = _whiten3(KA, KB, KD, y, seed=42)
    res_pw = {}
    for pair in [("A", "B"), ("A", "D"), ("B", "D")]:
        tag = f"{pair[0]}{pair[1]}"
        pv = _pairwise_pvals(Wh, y, Bdict[pair[0]], Bdict[pair[1]])
        if full_sink is not None:
            full_sink[tag] = np.asarray(pv, float).copy()
        G = pv.size
        bonf = 0.05 / G
        sig_idx = [int(i) for i in np.argsort(pv) if pv[i] < bonf]
        res_pw[tag] = dict(acat=float(_acat(pv)), minp=float(pv.min()),
                           lambda_gc=float(_lambda_gc(pv)), G=int(G),
                           bonferroni_alpha=float(bonf), n_sig=len(sig_idx),
                           top5=[dict(triad_idx=int(i), genes=gene_triads[i], p=float(pv[i]))
                                 for i in np.argsort(pv)[:5]])
    triad_acat_obs = _acat([res_pw["AB"]["acat"], res_pw["AD"]["acat"], res_pw["BD"]["acat"]])

    # permutation null
    rng = np.random.default_rng(2027)
    perms = [rng.permutation(n) for _ in range(B)]
    pr = Parallel(n_jobs=n_jobs)(
        delayed(_perm_rep_triad)(KA, KB, KD, Bdict, y[perms[i]], 900000 + i) for i in range(B))
    pr = [r for r in pr if r.get("ok")]
    Bok = len(pr)
    triad_perm = np.array([r["triad"] for r in pr])
    triad_emp = (1 + int((triad_perm <= triad_acat_obs).sum())) / (Bok + 1)
    pw_emp = {}
    for tag in ("AB", "AD", "BD"):
        ap = np.array([r[tag]["acat"] for r in pr])
        lam_perm = np.median([r[tag]["lam"] for r in pr])
        pw_emp[tag] = dict(acat_emp=float((1 + int((ap <= res_pw[tag]["acat"]).sum())) / (Bok + 1)),
                           lambda_gc_perm_median=float(lam_perm))
    return dict(label=label, n=int(n), perm_B=int(Bok),
                sigma_hat=dict(A=float(cv.get("A", 0.0)), B=float(cv.get("B", 0.0)),
                               D=float(cv.get("D", 0.0)), e=float(cv.get("e", 0.0))),
                triad_acat_obs=float(triad_acat_obs), triad_acat_emp=float(triad_emp),
                pairwise=res_pw, pairwise_emp=pw_emp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["body", "flank"], default="flank")
    ap.add_argument("--trait", default="days_to_emerg")
    ap.add_argument("--cap", type=int, default=150, help="max SNP/gene for burden (WGS de-dilution)")
    ap.add_argument("--perm-B", type=int, default=2000)
    ap.add_argument("--n-jobs", type=int, default=48)
    ap.add_argument("--out-dir", default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out_dir)
    t0 = time.time()
    print(f"=== Phase C wheat deploy (1A-1B-1D triad, mode={args.mode}, cap={args.cap}, trait={args.trait}) ===")

    # load per-subgenome ETL
    sub_data = {}
    for s in ("A", "B", "D"):
        z = np.load(out / f"pilot_1{s}_{args.mode}.npz", allow_pickle=True)
        X = np.load(out / f"pilot_1{s}_{args.mode}_X.npy")
        gene_ids = z["gene_ids"].tolist()
        snp_idx = z["snp_idx"].tolist()
        starts = z["starts"]
        samples = z["samples"].tolist()
        # sort by start (already sorted in ETL, enforce)
        order = np.argsort(starts)
        gene_ids = [gene_ids[i] for i in order]
        snp_idx = [snp_idx[i] for i in order]
        sub_data[s] = dict(X=X, gene_ids=gene_ids, snp_idx=snp_idx, samples=samples)
        print(f"  1{s}: {len(gene_ids)} retained genes, X={X.shape}")

    # sample intersection (should be identical 1051 across subgenomes)
    common = sub_data["A"]["samples"]
    for s in ("B", "D"):
        assert sub_data[s]["samples"] == common, f"sample order mismatch {s}"

    # position-homology triad: smallest subgenome drives count; map others by proportional rank
    nA, nB, nD = (len(sub_data[s]["gene_ids"]) for s in ("A", "B", "D"))
    n_triad = min(nA, nB, nD)
    driver = min(("A", "B", "D"), key=lambda s: len(sub_data[s]["gene_ids"]))
    print(f"  triad: nA={nA} nB={nB} nD={nD} -> n_triad={n_triad} (driver=1{driver})")
    triad_idx = {"A": [], "B": [], "D": []}
    for i in range(n_triad):
        for s, ns in [("A", nA), ("B", nB), ("D", nD)]:
            triad_idx[s].append(int(round(i * (ns - 1) / max(n_triad - 1, 1))))
    gene_triads = [{s: sub_data[s]["gene_ids"][triad_idx[s][i]] for s in ("A", "B", "D")}
                   for i in range(n_triad)]

    # burdens (capped) for each subgenome's triad-selected genes
    rng = np.random.default_rng(7)
    Bdict = {}
    for s in ("A", "B", "D"):
        cols = []
        for i in range(n_triad):
            gi = triad_idx[s][i]
            cols.append(_block_burden_capped(sub_data[s]["X"], sub_data[s]["snp_idx"][gi], args.cap, rng))
        Bdict[s] = _scols_safe(np.column_stack(cols))
    print(f"  burdens built: each {Bdict['A'].shape} ({time.time()-t0:.1f}s)")

    # GRMs from gene-region SNPs per subgenome (pilot approximation)
    KA = _grm_from_X(sub_data["A"]["X"])
    KB = _grm_from_X(sub_data["B"]["X"])
    KD = _grm_from_X(sub_data["D"]["X"])
    print(f"  K_A/K_B/K_D built ({time.time()-t0:.1f}s)")

    # phenotype align
    ph = pd.read_csv(PHENO, sep="\t").set_index("sample")
    valid = [s for s in common if s in ph.index and pd.notna(ph.loc[s, args.trait])]
    idx_in = np.array([common.index(s) for s in valid])
    n_t = len(valid)
    KA_t = KA[np.ix_(idx_in, idx_in)]
    KB_t = KB[np.ix_(idx_in, idx_in)]
    KD_t = KD[np.ix_(idx_in, idx_in)]
    # trace-renorm subset
    KA_t = KA_t / (np.trace(KA_t) / n_t)
    KB_t = KB_t / (np.trace(KB_t) / n_t)
    KD_t = KD_t / (np.trace(KD_t) / n_t)
    Bd_t = {s: Bdict[s][idx_in] for s in ("A", "B", "D")}
    # re-standardize burdens on the subset
    Bd_t = {s: _scols_safe(Bd_t[s]) for s in ("A", "B", "D")}
    y_raw = np.array([float(ph.loc[s, args.trait]) for s in valid])
    print(f"  trait={args.trait} n={n_t} y(mean={y_raw.mean():.3f} sd={y_raw.std():.3f}) ({time.time()-t0:.1f}s)")

    out_INT = deploy(_rank_int(y_raw), "INT", KA_t, KB_t, KD_t, Bd_t, gene_triads, args.n_jobs, args.perm_B)
    out_raw = deploy(y_raw, "raw", KA_t, KB_t, KD_t, Bd_t, gene_triads, args.n_jobs, args.perm_B)

    for lab, o in [("INT", out_INT), ("raw", out_raw)]:
        print(f"  [{lab}] triad ACAT obs={o['triad_acat_obs']:.4g} emp={o['triad_acat_emp']:.4g} | "
              f"σ̂ A={o['sigma_hat']['A']:.3f} B={o['sigma_hat']['B']:.3f} D={o['sigma_hat']['D']:.3f} e={o['sigma_hat']['e']:.3f}")
        for tag in ("AB", "AD", "BD"):
            pw = o["pairwise"][tag]
            pe = o["pairwise_emp"][tag]
            print(f"    {tag}: ACAT_obs={pw['acat']:.4g} emp={pe['acat_emp']:.4g} "
                  f"λ_obs={pw['lambda_gc']:.3f} λ_perm={pe['lambda_gc_perm_median']:.3f} "
                  f"nsig(α/G={pw['bonferroni_alpha']:.1e})={pw['n_sig']}")

    full = dict(panel="wheat_watkins", mode=args.mode, trait=args.trait, cap=args.cap,
                n_triad=n_triad, perm_B=args.perm_B,
                INT_primary=out_INT, raw_sensitivity=out_raw,
                note=("wheat AABBDD pilot (1A-1B-1D, position-homology triad, NOT curated "
                      "ortholog — synteny approximation; collinear subgenomes). 3 pairwise "
                      "(A-B/A-D/B-D) burden-product GLS whitened by full hexaploid null LMM "
                      "{K_A,K_B,K_D}; triad ACAT over 3 pairwise = hexaploid omnibus. burden "
                      f"SNP cap={args.cap} (WGS density: SNP/gene up to 890; cap de-dilutes). "
                      "GRM from gene-region SNPs (pilot approx; genome-wide LD-pruned GRM = "
                      "full-run). Single trait (emergence) -> calibration stress test, NOT "
                      "discovery screen. INT primary + raw sensitivity, B=2000 y-shuffle perm. "
                      "Codex TOP_RISK: emergence biology weakly linked, result supports "
                      "calibration not 2nd positive discovery."))
    fp = out / f"deploy_wheat_pilot_{args.mode}.json"
    fp.write_text(json.dumps(full, indent=2, default=float))
    print(f"  wrote {fp} ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
