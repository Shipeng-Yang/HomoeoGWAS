#!/usr/bin/env python3
"""Task #4: genome-scale enrichment — do top homoeolog-INTERACTION triads enrich for expression
imbalance (HEB) / dosage class, after controlling for detectability?

LOCKED design: reports/enrichment_prereg.md (frozen before looking). This script ONLY consumes the
checksummed frozen ranking (no scan re-run). Axis-specific detectability matched-null, B=10000,
two-sided empirical p, BH-FDR over the wheat INT grid. Cotton HEB enrichment is DEFERRED (needs the
NDM8 GFF, task #3) and is NOT in the FDR family. chr5 A-D is the expression-balanced independent
positive and is NEVER invoked as convergence.
"""
from __future__ import annotations

import csv
import gzip
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path("/mnt/7302share/fast_ysp/U7_GWAS")
RANK_INT = ROOT / "results/phase7/bio_wheat/wheat_GW_ranking_flank_INT.tsv"
RANK_RAW = ROOT / "results/phase7/bio_wheat/wheat_GW_ranking_flank_raw.tsv"
TPM = ROOT / "data/raw/expression/wheat/tpm/Development_tpm.tsv.gz"
HCTRIADS = ROOT / "data/raw/expression/wheat/HCTriads.csv"
OUT = ROOT / "results/phase7/bio_wheat/enrichment"
B_PERM = 10000
TIERS = (0.01, 0.05, 0.10)
LEADS = {("TraesCS1A02G125400", "TraesCS1B02G143800", "TraesCS1D02G128400"),   # chr1 B-D
         ("TraesCS5A02G169500", "TraesCS5B02G166300", "TraesCS5D02G173900")}   # chr5 A-D


def _v10_to_v11(g):
    return g.replace("01G", "02G", 1) if "01G" in g else g


def _load_tpm_mean():
    means = {}
    with gzip.open(TPM, "rt") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\n").split("\t")
            vals = np.array([float(x) for x in parts[1:] if x != ""], dtype=float)
            means[parts[0]] = float(vals.mean()) if vals.size else 0.0
    return means


def _dist_cat(a, b, d):
    tot = a + b + d
    if tot <= 1e-9:
        return None, None
    frac = np.array([a / tot, b / tot, d / tot])
    dist = float(np.linalg.norm(frac - np.array([1 / 3, 1 / 3, 1 / 3])))
    ideals = {"balanced": [1 / 3, 1 / 3, 1 / 3], "A.dominant": [1, 0, 0], "B.dominant": [0, 1, 0],
              "D.dominant": [0, 0, 1], "A.suppressed": [0, 1 / 2, 1 / 2],
              "B.suppressed": [1 / 2, 0, 1 / 2], "D.suppressed": [1 / 2, 1 / 2, 0]}
    cat = min(ideals, key=lambda k: np.linalg.norm(frac - np.array(ideals[k])))
    return dist, cat


def _heb_background():
    """Per-triad HEB distance + percentile over the 17024 1:1:1 HCTriad background (Development mean
    TPM); returns {(A,B,D): (dist, percentile, imbalanced_bool)} (v11 gene IDs)."""
    means = _load_tpm_mean()
    import pandas as pd
    hc = pd.read_csv(HCTRIADS)
    hc = hc[hc["cardinality_abs"] == "1:1:1"]
    rows = []
    for _, r in hc.iterrows():
        a, b, d = _v10_to_v11(str(r["A"])), _v10_to_v11(str(r["B"])), _v10_to_v11(str(r["D"]))
        ta, tb, td = means.get(a, np.nan), means.get(b, np.nan), means.get(d, np.nan)
        if not np.all(np.isfinite([ta, tb, td])):
            continue
        dist, cat = _dist_cat(ta, tb, td)
        if dist is None:
            continue
        rows.append((a, b, d, dist, cat))
    dists = np.array([r[3] for r in rows])
    out = {}
    for a, b, d, dist, cat in rows:
        pct = float((dists < dist).mean())
        out[(a, b, d)] = (dist, pct, cat != "balanced")
    return out, len(rows)


def _load_ranking(path):
    rk = list(csv.DictReader(open(path), delimiter="\t"))
    for r in rk:
        for k in ("p_BD", "p_acat", "n_snp_A", "n_snp_B", "n_snp_D",
                  "gene_len_A", "gene_len_B", "gene_len_D"):
            r[k] = float(r[k])
    return rk


def _deciles(x):
    """full-universe decile (0..9) by rank (ties share)."""
    r = stats.rankdata(x, method="average")
    return np.clip(((r - 0.5) / x.size * 10).astype(int), 0, 9)


def enrich(score, snp_sum, len_sum, heb_pct, imbalanced, pct, rng, label):
    """One axis x tier contrast: stratified detectability-matched null on mean-HEB and fraction-
    imbalanced. Returns dict (or infeasible flag if any tier stratum lacks enough non-tier controls)."""
    n = score.size
    k = math.ceil(pct * n)
    order = np.argsort(-score, kind="stable")            # descending; deterministic tie handling
    tier = np.zeros(n, bool)
    tier[order[:k]] = True
    boundary_tie = bool(k < n and score[order[k - 1]] == score[order[k]])
    snp_dec = _deciles(snp_sum)
    len_dec = _deciles(len_sum)
    strat = snp_dec * 10 + len_dec
    nontier = ~tier
    # per-stratum tier counts + available non-tier pools
    strata_ids = np.unique(strat[tier])
    pools, need = {}, {}
    infeasible = []
    for s in strata_ids:
        need[s] = int((tier & (strat == s)).sum())
        pool = np.where(nontier & (strat == s))[0]
        pools[s] = pool
        if pool.size < need[s]:
            infeasible.append((int(s), need[s], int(pool.size)))
    tail = (heb_pct >= 0.9).astype(float)                # HEB top-decile (tail enrichment annotation)
    obs_heb = float(heb_pct[tier].mean())
    obs_imb = float(imbalanced[tier].mean())
    obs_tail = float(tail[tier].mean())
    if infeasible:
        return dict(label=label, tier_pct=pct, tier_n=int(k), infeasible=True,
                    offending_strata=infeasible, obs_mean_heb=obs_heb, obs_frac_imbalanced=obs_imb,
                    note="some tier strata lack enough non-tier controls -> locked null infeasible")
    # stratum diagnostics: rule out overmatching / degeneracy
    ratios = np.array([pools[s].size / need[s] for s in strata_ids])
    frac_sampled = float(k / sum(pools[s].size for s in strata_ids))
    diag = dict(n_occupied_strata=int(len(strata_ids)),
                pool_over_need_min=float(ratios.min()), pool_over_need_median=float(np.median(ratios)),
                pool_over_need_max=float(ratios.max()),
                n_strata_pool_lt_3x_need=int((ratios < 3).sum()),
                frac_nontier_controls_sampled_per_perm=frac_sampled)
    null_heb = np.empty(B_PERM)
    null_imb = np.empty(B_PERM)
    null_tail = np.empty(B_PERM)
    for b in range(B_PERM):
        idx = np.concatenate([rng.choice(pools[s], size=need[s], replace=False) for s in strata_ids])
        null_heb[b] = heb_pct[idx].mean()
        null_imb[b] = imbalanced[idx].mean()
        null_tail[b] = tail[idx].mean()

    def two_sided(obs, null):
        c = null.mean()
        p = (1 + int((np.abs(null - c) >= abs(obs - c)).sum())) / (1 + B_PERM)
        return float(p), float(c), float(obs - c), (float(obs / c) if c > 1e-12 else float("nan")), \
            float(np.percentile(null, 2.5)), float(np.percentile(null, 97.5))

    p_h, c_h, eff_h, er_h, lo_h, hi_h = two_sided(obs_heb, null_heb)
    p_i, c_i, eff_i, er_i, lo_i, hi_i = two_sided(obs_imb, null_imb)
    p_t, c_t, eff_t, er_t, lo_t, hi_t = two_sided(obs_tail, null_tail)
    # rank-sum sensitivity: tier HEB vs non-tier HEB
    rs = stats.mannwhitneyu(heb_pct[tier], heb_pct[nontier], alternative="greater")
    return dict(label=label, tier_pct=pct, tier_n=int(k), boundary_tie=boundary_tie, infeasible=False,
                heb_continuous=dict(obs_mean=obs_heb, null_mean=c_h, effect=eff_h, enrichment_ratio=er_h,
                                    perm_ci=[lo_h, hi_h], p_two_sided=p_h),
                dosage_binary=dict(obs_frac_imbalanced=obs_imb, null_mean=c_i, effect=eff_i,
                                   enrichment_ratio=er_i, perm_ci=[lo_i, hi_i], p_two_sided=p_i),
                heb_tail_top_decile=dict(obs_frac=obs_tail, null_mean=c_t, effect=eff_t,
                                         enrichment_ratio=er_t, perm_ci=[lo_t, hi_t], p_two_sided=p_t),
                ranksum_sensitivity=dict(U=float(rs.statistic), p_greater=float(rs.pvalue)),
                stratum_diagnostics=diag)


def run_axis_grid(rk, heb, seed, tag, drop_leads=False):
    keys = [(r["gene_A"], r["gene_B"], r["gene_D"]) for r in rk]
    keep = np.array([k in heb and (not drop_leads or k not in LEADS) for k in keys])
    rk2 = [r for r, kp in zip(rk, keep, strict=True) if kp]
    keys2 = [k for k, kp in zip(keys, keep, strict=True) if kp]
    heb_pct = np.array([heb[k][1] for k in keys2])
    imbalanced = np.array([float(heb[k][2]) for k in keys2])
    pBD = np.array([r["p_BD"] for r in rk2])
    pAC = np.array([r["p_acat"] for r in rk2])
    nsA = np.array([r["n_snp_A"] for r in rk2])
    nsB = np.array([r["n_snp_B"] for r in rk2])
    nsD = np.array([r["n_snp_D"] for r in rk2])
    glA = np.array([r["gene_len_A"] for r in rk2])
    glB = np.array([r["gene_len_B"] for r in rk2])
    glD = np.array([r["gene_len_D"] for r in rk2])
    bd_score = -np.log10(np.clip(pBD, 1e-300, 1.0))
    ac_score = -np.log10(np.clip(pAC, 1e-300, 1.0))
    axes = {"BD": (bd_score, nsB + nsD, glB + glD),
            "ACAT": (ac_score, nsA + nsB + nsD, glA + glB + glD)}
    out = {}
    rng = np.random.default_rng(seed)
    for ax, (sc, snp_sum, len_sum) in axes.items():
        for t in TIERS:
            res = enrich(sc, snp_sum, len_sum, heb_pct, imbalanced, t, rng, f"{tag}:{ax}:top{int(t*100)}")
            out[f"{ax}_top{int(t*100)}"] = res
            if not res.get("infeasible"):
                h = res["heb_continuous"]
                tl = res["heb_tail_top_decile"]
                sd = res["stratum_diagnostics"]
                print(f"  [{tag}] {ax} top{int(t*100)}% (n={res['tier_n']}): HEB obs={h['obs_mean']:.3f} "
                      f"null={h['null_mean']:.3f} eff={h['effect']:+.3f} p={h['p_two_sided']:.4g} | "
                      f"imb p={res['dosage_binary']['p_two_sided']:.4g} | "
                      f"tail(top-dec) obs={tl['obs_frac']:.3f} p={tl['p_two_sided']:.4g} | "
                      f"strata={sd['n_occupied_strata']} pool/need med={sd['pool_over_need_median']:.1f} "
                      f"fracSampled={sd['frac_nontier_controls_sampled_per_perm']:.2f}")
            else:
                print(f"  [{tag}] {ax} top{int(t*100)}%: INFEASIBLE {res['offending_strata'][:3]}")
    return out, int(keep.sum())


def _bh(pdict):
    items = [(k, v) for k, v in pdict.items() if v is not None]
    if not items:
        return {}
    ps = np.array([v for _, v in items])
    order = np.argsort(ps)
    m = len(ps)
    q = np.empty(m)
    prev = 1.0
    for rank, i in enumerate(order[::-1]):
        r = m - rank
        prev = min(prev, ps[i] * m / r)
        q[i] = prev
    return {items[i][0]: float(q[i]) for i in range(m)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("=== task #4 genome-scale enrichment (wheat; cotton deferred) ===")
    heb, n_bg = _heb_background()
    print(f"  HEB background triads (17024 universe expressed): {n_bg}")
    rk = _load_ranking(RANK_INT)
    print(f"  ranking (INT) triads: {len(rk)}")

    primary, n_join = run_axis_grid(rk, heb, seed=4001, tag="INT")
    print(f"  joined ranking triads with HEB: {n_join}/{len(rk)}")

    # BH-FDR over the wheat INT primary grid (axis x tier x {HEB, dosage})
    pgrid = {}
    for key, res in primary.items():
        if not res.get("infeasible"):
            pgrid[f"{key}:HEB"] = res["heb_continuous"]["p_two_sided"]
            pgrid[f"{key}:dosage"] = res["dosage_binary"]["p_two_sided"]
    fdr = _bh(pgrid)

    # sensitivities
    print("  -- sensitivity: raw-transform ranking --")
    rk_raw = _load_ranking(RANK_RAW)
    raw_sens, _ = run_axis_grid(rk_raw, heb, seed=4002, tag="raw")
    print("  -- sensitivity: leave-known-leads-out (drop chr1 + chr5 triads before tiering) --")
    lko, n_lko = run_axis_grid(rk, heb, seed=4003, tag="LKO", drop_leads=True)

    payload = dict(task="enrichment_wheat", prereg="reports/enrichment_prereg.md",
                   frozen_input="wheat_GW_ranking_flank_INT.tsv", B_perm=B_PERM,
                   heb_background_n=n_bg, ranking_n=len(rk), joined_n=n_join,
                   primary_INT=primary, bh_fdr_grid=fdr,
                   sensitivity_raw=raw_sens, sensitivity_leave_leads_out=lko,
                   note=("axis-specific detectability matched-null (BD: n_snp_B+D x gene_len_B+D "
                         "deciles; ACAT: triad sums); two-sided empirical p; BH-FDR over the wheat INT "
                         "grid only. Cotton HEB enrichment DEFERRED (NDM8 GFF, task #3) and excluded "
                         "from the FDR family. chr5 A-D = expression-balanced independent positive, "
                         "NOT convergence; leave-leads-out shows the enrichment is not lead-driven."))
    fp = OUT / "enrichment_wheat.json"
    fp.write_text(json.dumps(payload, indent=2, default=float))
    print(f"  headline top-5% BD x HEB: p={pgrid.get('BD_top5:HEB')} FDR-q={fdr.get('BD_top5:HEB')}")
    print(f"  wrote {fp}")


if __name__ == "__main__":
    main()
