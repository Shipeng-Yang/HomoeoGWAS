#!/usr/bin/env python3
"""Phase 7 bio-pilot Step 3: feed real gene-burden bins + MCScanX A-D collinear pairs
into the d3 per-pair GLS+ACAT test. Validates the bio pipeline end-to-end and confirms
the calibration (patch ①) still holds with bio bins.

Reuses d2_arm2._load (cotton panel), _scale_cols, _scale_vec, _block_burden, and
d3._perpair_pvals, _acat, _lambda_gc, _binom_ci95. Builds:
  - BA_bio (n × G_A_used): per-gene burden = mean std dosage over each gene's SNPs.
  - BD_bio (n × G_D_used): same on D side.
  - pair_idx_bio: list of (a_idx, d_idx) — MCScanX A-D collinear pairs, 1:1
    best-reciprocal (smallest e-value per A and per D) within blocks.

Outputs: results/phase7/bio_pilot/d3_bio_pilot.json (calib R + λ_GC + ACAT type-I +
                                                     one power@PVE0.20 point)."""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from scipy import stats

ROOT = Path("/mnt/7302share/fast_ysp/U7_GWAS")
sys.path.insert(0, str(ROOT / "scripts/phase7"))
from d2_arm2_synteny_kernel import (  # noqa: E402
    ADDITIVE_PVE, PANELS, _block_burden, _chol, _load, _scale_vec,
)
from d3_perpair_interaction import (  # noqa: E402
    _acat, _binom_ci95, _lambda_gc, _perpair_pvals,
)


def _parse_collinearity(path):
    """Return list of (a_gene, d_gene, e_value, alignment_idx) for inter-subgenome pairs."""
    pairs = []
    aln_idx = -1
    pat = re.compile(r"^\s*(\d+)-\s*\d+:\s+(\S+)\s+(\S+)\s+(\S+)")
    with open(path) as f:
        for line in f:
            if line.startswith("## Alignment"):
                aln_idx += 1
                continue
            m = pat.match(line)
            if not m:
                continue
            g1, g2, ev = m.group(2), m.group(3), m.group(4)
            # Determine which is A vs D from gene_id (GhM_A01G... / GhM_D01G...)
            a, d = (g1, g2) if "_A" in g1 and "_D" in g2 else (
                (g2, g1) if "_A" in g2 and "_D" in g1 else (None, None))
            if a is None:
                continue
            try:
                ev_f = float(ev)
            except ValueError:
                ev_f = 1.0
            pairs.append((a, d, ev_f, aln_idx))
    return pairs


def _one_to_one_reciprocal(pairs):
    """For each A gene keep its best (lowest e-value) D partner AND vice versa.
    Returns list of (a, d, e) keeping only mutual best pairs."""
    best_a = {}  # A gene -> (best e, d)
    best_d = {}
    for a, d, e, _ in pairs:
        if a not in best_a or e < best_a[a][0]:
            best_a[a] = (e, d)
        if d not in best_d or e < best_d[d][0]:
            best_d[d] = (e, a)
    out = []
    for a, (e, d) in best_a.items():
        if best_d.get(d, (None, None))[1] == a:
            out.append((a, d, e))
    return out


def _load_snp_to_gene(path):
    z = np.load(path, allow_pickle=True)
    return {gid: np.asarray(idx, int)
            for gid, idx in zip(z["gene_ids"].tolist(), z["snp_idx"].tolist())}


def _build_burden_matrix(X, gene_ids, gene_snp):
    """n × len(gene_ids); column j = block burden over gene_ids[j]'s SNP indices (mean of
    column-standardized dosages, like _block_burden)."""
    cols = [_block_burden(X, gene_snp[g]) for g in gene_ids]
    return np.column_stack(cols)


def _whiten(KA, KD, y, n_starts=3, seed=1):
    """V^{-1/2} from null LMM {K_A,K_D}. Local copy to avoid d3's signature dep on X0."""
    from homoeogwas.lmm import fit_multi_reml
    res = fit_multi_reml(y, np.ones((KA.shape[0], 1)), {"A": KA, "D": KD},
                         n_starts=n_starts, random_state=int(seed) % 100000)
    cv = res.component_var
    n = KA.shape[0]
    V = (cv.get("A", 0.0) * KA + cv.get("D", 0.0) * KD
         + max(cv.get("e", 1e-6), 1e-6) * np.eye(n))
    w, Q = np.linalg.eigh(0.5 * (V + V.T))
    w = np.clip(w, 1e-10, None)
    return (Q * (1.0 / np.sqrt(w))) @ Q.T, cv


def _calib_rep(KA, KD, cholA, cholD, BA_m, BD_m, seed):
    """Lean PVE=0 rep for tail calibration on bio bins."""
    rng = np.random.default_rng(seed)
    n = KA.shape[0]; sa = ADDITIVE_PVE / 2
    y = (np.sqrt(sa) * (cholA @ rng.standard_normal(n))
         + np.sqrt(sa) * (cholD @ rng.standard_normal(n))
         + np.sqrt(max(1 - ADDITIVE_PVE, 1e-6)) * rng.standard_normal(n))
    try:
        Wh, _ = _whiten(KA, KD, y, seed=seed)
        p = _perpair_pvals(Wh, y, BA_m, BD_m)
        return dict(ok=True, acat=_acat(p), lam=_lambda_gc(p), npairs=int(p.size),
                    tail={a: int((p <= a).sum()) for a in (0.05, 0.01, 0.005)})
    except Exception:  # noqa: BLE001
        return dict(ok=False)


def _power_rep(KA, KD, cholA, cholD, BA_m, BD_m, Uc, pve, seed):
    """Lean rep at given pve with bio causal Uc."""
    rng = np.random.default_rng(seed)
    n = KA.shape[0]; sa = ADDITIVE_PVE / 2
    y = (np.sqrt(sa) * (cholA @ rng.standard_normal(n))
         + np.sqrt(sa) * (cholD @ rng.standard_normal(n)))
    if pve > 0:
        y += _scale_vec(Uc @ rng.standard_normal(Uc.shape[1])) * np.sqrt(pve)
    y += np.sqrt(max(1 - ADDITIVE_PVE - pve, 1e-6)) * rng.standard_normal(n)
    try:
        Wh, _ = _whiten(KA, KD, y, seed=seed)
        return dict(ok=True, acat=_acat(_perpair_pvals(Wh, y, BA_m, BD_m)))
    except Exception:  # noqa: BLE001
        return dict(ok=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="cotton_hebau")
    ap.add_argument("--calib-rep", type=int, default=1000)
    ap.add_argument("--power-rep", type=int, default=150)
    ap.add_argument("--n-causal", type=int, default=3)
    ap.add_argument("--pve", type=float, default=0.20)
    ap.add_argument("--n-jobs", type=int, default=48)
    ap.add_argument("--bio-dir", default=str(ROOT / "results/phase7/bio_pilot"))
    ap.add_argument("--collinearity", default="/mnt/nvme/cotton_hbau/mcscanx_pilot/pilot.collinearity")
    args = ap.parse_args()

    bio = Path(args.bio_dir); t0 = time.time()
    print(f"=== bio-pilot Step 3: d3 on bio bins+pairs (panel={args.panel}) ===")

    # 1. parse MCScanX -> 1:1 reciprocal A-D pairs
    raw_pairs = _parse_collinearity(args.collinearity)
    pairs_1to1 = _one_to_one_reciprocal(raw_pairs)
    print(f"  MCScanX raw pairs={len(raw_pairs)} -> 1:1 reciprocal={len(pairs_1to1)}")

    # 2. load cotton dosages + kernels via d3 _load
    cfg = PANELS[args.panel]; D = _load(cfg)
    n = D["n"]
    KA, KD = D["KA"], D["KD"]
    cholA, cholD = _chol(KA), _chol(KD)
    print(f"  cotton: n={n}, KA/KD loaded ({time.time()-t0:.1f}s)")

    # 3. load snp_to_gene index (bim-row indices per gene)
    g2s = _load_snp_to_gene(bio / "snp_to_gene.npz")

    # 4. retain only pairs where both A and D gene are in g2s (i.e. retained at ETL)
    pairs_kept = [(a, d, e) for a, d, e in pairs_1to1 if a in g2s and d in g2s]
    a_genes = sorted({a for a, _, _ in pairs_kept})
    d_genes = sorted({d for _, d, _ in pairs_kept})
    a_idx = {g: i for i, g in enumerate(a_genes)}
    d_idx = {g: i for i, g in enumerate(d_genes)}
    pair_idx_bio = [(a_idx[a], d_idx[d]) for a, d, _ in pairs_kept]
    G = len(pair_idx_bio)
    print(f"  retained pairs (after ETL gene check)={G}; unique A={len(a_genes)} D={len(d_genes)}")
    if G < 5:
        print("  ABORT: too few bio pairs to test."); return

    # 5. build per-gene burden matrices (n × len(a_genes), n × len(d_genes))
    BA_full = _build_burden_matrix(D["XA"], a_genes, g2s)
    BD_full = _build_burden_matrix(D["XD"], d_genes, g2s)
    # gather paired burdens (standardize per-pair like d3 does)
    jA = [p[0] for p in pair_idx_bio]; jD = [p[1] for p in pair_idx_bio]

    def _scols_safe(M):
        mu = M.mean(0); sd = M.std(0, ddof=0)
        sd = np.where(sd > 1e-12, sd, 1.0)
        return (M - mu) / sd
    BA_m = _scols_safe(BA_full[:, jA])
    BD_m = _scols_safe(BD_full[:, jD])
    print(f"  BA_bio shape={BA_full.shape} BD_bio shape={BD_full.shape} "
          f"paired matrix={BA_m.shape} ({time.time()-t0:.1f}s)")

    # 6. TAIL CALIBRATION on bio bins (does patch① calibration hold?)
    cr = Parallel(n_jobs=args.n_jobs)(
        delayed(_calib_rep)(KA, KD, cholA, cholD, BA_m, BD_m, 600000 + r)
        for r in range(args.calib_rep))
    cr = [r for r in cr if r.get("ok")]
    R = len(cr)
    acat_p = np.array([r["acat"] for r in cr])
    lam_rep = np.array([r["lam"] for r in cr])
    npairs = cr[0]["npairs"]
    typeI = {}
    for a in (0.05, 0.01, 0.005):
        k = int((acat_p <= a).sum())
        typeI[f"{a}"] = dict(type1=k / R, count=k, ci95=_binom_ci95(k, R),
                             mc_se_nominal=float(np.sqrt(a * (1 - a) / R)))
    pooled_tail = {f"{a}": sum(r["tail"][a] for r in cr) / (npairs * R)
                   for a in (0.05, 0.01, 0.005)}
    print(f"  CALIB R={R} | ACAT analytic type-I: "
          f"{[(a, round(v['type1'], 4)) for a, v in typeI.items()]} | "
          f"per-pair λ_GC med={np.median(lam_rep):.4f} iqr=[{np.percentile(lam_rep,25):.3f},"
          f"{np.percentile(lam_rep,75):.3f}] | pooled type-I={pooled_tail}")

    # 7. POWER at PVE=0.20 with bio causal: pick C real A-D pairs, inject BA*BD interaction
    rng_c = np.random.default_rng(11)
    n_causal = min(args.n_causal, G)
    cidx = rng_c.choice(G, size=n_causal, replace=False)
    Uc = np.column_stack([
        _scale_vec(BA_m[:, c]) * _scale_vec(BD_m[:, c]) for c in cidx])
    sig = Parallel(n_jobs=args.n_jobs)(
        delayed(_power_rep)(KA, KD, cholA, cholD, BA_m, BD_m, Uc, args.pve, 700000 + i)
        for i in range(args.power_rep))
    nul = Parallel(n_jobs=args.n_jobs)(
        delayed(_power_rep)(KA, KD, cholA, cholD, BA_m, BD_m, Uc, 0.0, 710000 + i)
        for i in range(args.power_rep))
    aps = np.array([r["acat"] for r in sig if r.get("ok")])
    apn = np.array([r["acat"] for r in nul if r.get("ok")])
    ks, kn = int((aps < 0.05).sum()), int((apn < 0.05).sum())
    power = ks / aps.size; type1_pwrnull = kn / apn.size
    print(f"  POWER (bio causal, C={n_causal}, PVE={args.pve}): analytic p<.05 "
          f"power={power:.3f} CI{[round(x,3) for x in _binom_ci95(ks, aps.size)]} | "
          f"type1(null-y rerun)={type1_pwrnull:.3f}")

    # 8. write output
    out = dict(
        panel=args.panel, n=n, G_pairs=G, n_unique_A=len(a_genes), n_unique_D=len(d_genes),
        npairs_per_calib=npairs, calib_rep=R, power_rep=int(aps.size),
        tail_calibration=dict(
            acat_analytic_typeI=typeI,
            lambda_gc_rep_median=float(np.median(lam_rep)),
            lambda_gc_rep_iqr=[float(np.percentile(lam_rep, 25)), float(np.percentile(lam_rep, 75))],
            perpair_pooled_typeI=pooled_tail),
        power_check=dict(
            n_causal=int(n_causal), pve=args.pve, power=power,
            power_ci95=_binom_ci95(ks, aps.size),
            type1_with_powerseeds=type1_pwrnull),
        note=("bio-pilot A01-D01: per-gene burden over GFF body-only + MCScanX collinear "
              "1:1 reciprocal A-D pairs. Validates pipeline end-to-end; calibration check "
              "confirms patch① still holds with bio bins (λ_GC should be ~1 if test is "
              "well-calibrated on real gene-burden inputs). Pilot scope is small (G≈few "
              "dozen pairs) — power numbers are pipeline-validation, NOT scientific claim."))
    pp = bio / "d3_bio_pilot.json"
    pp.write_text(json.dumps(out, indent=2, default=float))
    print(f"  wrote {pp} ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
