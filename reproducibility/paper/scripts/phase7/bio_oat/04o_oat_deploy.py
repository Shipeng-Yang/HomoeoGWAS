#!/usr/bin/env python3
"""OAT Step 4: homoeolog-PAIR interaction deploy across 3 subgenome-pairs (A-C/A-D/C-D),
whitened by the full hexaploid null LMM {K_A,K_C,K_D}. Sparse-GBS pivot from strict
triads to pairwise units. Cotton AADD burden-product GLS applied to
each of the 3 subgenome-pairs of the hexaploid.

Multiplicity (predeclared):
  PRIMARY = INT transform + flank mode. Discovery threshold = 0.05/(G_total * n_traits)
  on the ANALYTIC per-pair interaction p (perm B too coarse for genome-wide tail; perm
  used for lambda/calibration only). raw transform + body mode = SENSITIVITY only.
Guardrails: sample-order asserts, NaN/inf p checks, per-trait N, lambda_perm, attrition,
per-(mode,transform) JSON + a master manifest/hit/calibration roll-up written by caller.
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
sys.path.insert(0, str(ROOT / "scripts/phase7/bio_wheat"))
from d3_perpair_interaction import _acat, _lambda_gc  # noqa: E402
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "wheat02", str(ROOT / "scripts/phase7/bio_wheat/02w_wheat_deploy.py"))
_w2 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_w2)

OUT = ROOT / "results/phase7/bio_oat"
PHENO = ROOT / "data/processed/oat/pheno_clean.tsv"
SUBS = ("A", "C", "D")
SUBPAIRS = (("A", "C"), ("A", "D"), ("C", "D"))  # maps onto wheat A-B/A-D/B-D with C<->B


def _psd_clip(K):
    """Project a symmetric matrix to nearest PSD (clip eigenvalues >=0), trace-normed.
    GBS GRMs can have tiny negative eigenvalues that fail fit_multi_reml PSD check."""
    Ks = 0.5 * (K + K.T)
    w, Q = np.linalg.eigh(Ks)
    w = np.clip(w, 1e-6, None)  # positive floor so trace-norm keeps min_eig > REML tol (1e-8)
    Kp = (Q * w) @ Q.T
    Kp = 0.5 * (Kp + Kp.T)
    return Kp / (np.trace(Kp) / Kp.shape[0])


def _load_sub(mode):
    """Per subgenome: full X, GRM, gene->burden-source (snp_idx), samples."""
    data = {}
    samples_ref = None
    for s in SUBS:
        X = np.load(OUT / f"X_{s}.npy")
        z = np.load(OUT / f"snp_to_gene_{s}_{mode}.npz", allow_pickle=True)
        gene_ids = z["gene_ids"].tolist()
        snp_idx = {g: np.asarray(z["snp_idx"][i], int) for i, g in enumerate(gene_ids)}
        samples = [str(x) for x in z["samples"].tolist()]
        if samples_ref is None:
            samples_ref = samples
        assert samples == samples_ref, f"sample order mismatch {s}"
        data[s] = dict(X=X, gene_ids=set(gene_ids), snp_idx=snp_idx, samples=samples)
    return data, samples_ref


def _callable_pairs(mode_data):
    """Per subgenome-pair, reciprocal-unique homoeolog pairs with both genes retained."""
    hp = pd.read_csv(OUT / "homoeolog_pairs.tsv", sep="\t")
    tag2pairs = {f"{a}{b}": [] for a, b in SUBPAIRS}
    for _, r in hp.iterrows():
        tag = r["sub_pair"]; sa, sb = tag[0], tag[1]
        if r["gene_x"] in mode_data[sa]["gene_ids"] and r["gene_y"] in mode_data[sb]["gene_ids"]:
            tag2pairs[tag].append((r["gene_x"], r["gene_y"]))
    return tag2pairs


def _build_burdens(mode_data, tag2pairs, cap, rng):
    """For each subgenome-pair tag, BX/BY (n x G) standardized burden matrices + gene lists."""
    fam = {}
    for (sa, sb) in SUBPAIRS:
        tag = f"{sa}{sb}"
        pairs = tag2pairs[tag]
        if not pairs:
            fam[tag] = None
            continue
        bx = np.column_stack([_w2._block_burden_capped(mode_data[sa]["X"],
                              mode_data[sa]["snp_idx"][x], cap, rng) for x, _ in pairs])
        by = np.column_stack([_w2._block_burden_capped(mode_data[sb]["X"],
                              mode_data[sb]["snp_idx"][y], cap, rng) for _, y in pairs])
        fam[tag] = dict(pairs=pairs, BX=_w2._scols_safe(bx), BY=_w2._scols_safe(by))
    return fam


def _perm_rep(KA, KC, KD, fam_sub, y_shuf, seed):
    try:
        Wh, _ = _w2._whiten3(KA, KC, KD, y_shuf, seed=seed)
        out = {}
        for tag, f in fam_sub.items():
            if f is None:
                continue
            pv = _w2._pairwise_pvals(Wh, y_shuf, f["BX"], f["BY"])
            out[tag] = dict(acat=float(_acat(pv)), lam=float(_lambda_gc(pv)))
        return dict(ok=True, **out)
    except Exception:  # noqa: BLE001
        return dict(ok=False)


def deploy_trait(y_raw, transform, KA, KC, KD, fam_sub, n_traits, G_total, n_jobs, perm_B):
    y = _w2._rank_int(y_raw) if transform == "INT" else y_raw.astype(float)
    Wh, cv = _w2._whiten3(KA, KC, KD, y, seed=42)
    fam_res = {}
    all_p = []
    for tag, f in fam_sub.items():
        if f is None:
            fam_res[tag] = dict(G=0)
            continue
        pv = _w2._pairwise_pvals(Wh, y, f["BX"], f["BY"])
        if not np.all(np.isfinite(pv)):
            pv = np.where(np.isfinite(pv), pv, 1.0)
        all_p.append(pv)
        order = np.argsort(pv)
        fam_res[tag] = dict(
            G=int(pv.size), acat=float(_acat(pv)), minp=float(pv.min()),
            lambda_gc=float(_lambda_gc(pv)),
            top=[dict(pair=f["pairs"][int(i)], p=float(pv[i])) for i in order[:5]])
    pooled = np.concatenate(all_p) if all_p else np.array([1.0])
    # PRIMARY genome-wide threshold: 0.05 / (G_total * n_traits)
    alpha_gw = 0.05 / max(G_total * n_traits, 1)
    hits = []
    for tag, f in fam_sub.items():
        if f is None:
            continue
        pv = _w2._pairwise_pvals(Wh, y, f["BX"], f["BY"])
        for i in np.where(pv < alpha_gw)[0]:
            hits.append(dict(sub_pair=tag, pair=f["pairs"][int(i)], p=float(pv[i])))

    res = dict(transform=transform, n=int(KA.shape[0]),
               sigma_hat=dict(A=float(cv.get("A", 0.0)), C=float(cv.get("B", 0.0)),
                              D=float(cv.get("D", 0.0)), e=float(cv.get("e", 0.0))),
               pooled_acat=float(_acat(pooled)), G_total=int(G_total),
               alpha_genomewide=float(alpha_gw), n_hits_gw=len(hits), hits_gw=hits,
               per_family=fam_res)

    # perm calibration (lambda_perm + empirical ACAT per family); coarse, sanity only
    if perm_B and perm_B > 0:
        n = KA.shape[0]
        prng = np.random.default_rng(2027)
        perms = [prng.permutation(n) for _ in range(perm_B)]
        pr = Parallel(n_jobs=n_jobs)(
            delayed(_perm_rep)(KA, KC, KD, fam_sub, y[perms[i]], 900000 + i) for i in range(perm_B))
        pr = [r for r in pr if r.get("ok")]
        cal = {}
        for tag, f in fam_sub.items():
            if f is None:
                continue
            ap = np.array([r[tag]["acat"] for r in pr if tag in r])
            lam = np.array([r[tag]["lam"] for r in pr if tag in r])
            obs = fam_res[tag]["acat"]
            cal[tag] = dict(acat_emp=float((1 + int((ap <= obs).sum())) / (len(ap) + 1)),
                            lambda_perm_median=float(np.median(lam)) if lam.size else None)
        res["perm_B"] = len(pr)
        res["calibration"] = cal
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["body", "flank"], default="flank")
    ap.add_argument("--cap", type=int, default=150)
    ap.add_argument("--perm-B", type=int, default=0, help=">0 enables perm calibration (primary only)")
    ap.add_argument("--n-jobs", type=int, default=48)
    ap.add_argument("--traits", default="ALL")
    args = ap.parse_args()
    t0 = time.time()

    mode_data, samples = _load_sub(args.mode)
    tag2pairs = _callable_pairs(mode_data)
    rng = np.random.default_rng(7)
    fam_sub = _build_burdens(mode_data, tag2pairs, args.cap, rng)
    G_total = sum(f["G"] if f else 0 for f in
                  [({"G": len(v)} if v else None) for v in tag2pairs.values()])
    G_total = sum(len(v) for v in tag2pairs.values())
    print(f"=== OAT deploy mode={args.mode} | callable pairs: "
          + " ".join(f"{t}={len(p)}" for t, p in tag2pairs.items())
          + f" | G_total={G_total} ===")

    # GRMs from full per-subgenome genotype
    K = {s: _w2._grm_from_X(mode_data[s]["X"]) for s in SUBS}
    print(f"  GRMs built ({time.time()-t0:.1f}s)")

    ph = pd.read_csv(PHENO, sep="\t")
    idcol = ph.columns[0]
    ph = ph.set_index(idcol)
    traits = list(ph.columns) if args.traits == "ALL" else args.traits.split(",")

    results = {}
    for trait in traits:
        valid = [s for s in samples if s in ph.index and pd.notna(ph.loc[s, trait])]
        if len(valid) < 50:
            results[trait] = dict(skipped=f"n={len(valid)}<50")
            continue
        idx = np.array([samples.index(s) for s in valid])
        n_t = len(valid)
        Ksub = {}
        for s in SUBS:
            Kt = K[s][np.ix_(idx, idx)]
            Ksub[s] = _psd_clip(Kt / (np.trace(Kt) / n_t))
        # subset + re-standardize burdens on valid samples
        fam_t = {}
        for tag, f in fam_sub.items():
            if f is None:
                fam_t[tag] = None
                continue
            fam_t[tag] = dict(pairs=f["pairs"], BX=_w2._scols_safe(f["BX"][idx]),
                              BY=_w2._scols_safe(f["BY"][idx]))
        y_raw = np.array([float(ph.loc[s, trait]) for s in valid])
        is_primary = (args.mode == "flank")
        pB = args.perm_B if is_primary else 0
        out_INT = deploy_trait(y_raw, "INT", Ksub["A"], Ksub["C"], Ksub["D"],
                               fam_t, len(traits), G_total, args.n_jobs, pB)
        out_raw = deploy_trait(y_raw, "raw", Ksub["A"], Ksub["C"], Ksub["D"],
                               fam_t, len(traits), G_total, args.n_jobs, 0)
        results[trait] = dict(n=n_t, INT_primary=out_INT, raw_sensitivity=out_raw)
        flag = "  <== GW HIT" if out_INT["n_hits_gw"] else ""
        print(f"  {trait}: n={n_t} INT pooled_acat={out_INT['pooled_acat']:.3g} "
              f"min_fam_acat={min((v.get('acat',1) for v in out_INT['per_family'].values() if 'acat' in v), default=1):.3g} "
              f"GWhits={out_INT['n_hits_gw']}{flag} ({time.time()-t0:.1f}s)")

    payload = dict(species="Avena_sativa_AACCDD", panel="OLD_rahman2025_737",
                   mode=args.mode, cap=args.cap, n_traits=len(traits), G_total=G_total,
                   callable_pairs={t: len(p) for t, p in tag2pairs.items()},
                   primary=(args.mode == "flank"),
                   alpha_genomewide=0.05 / max(G_total * len(traits), 1),
                   framing=("hexaploid homoeolog-PAIR interaction (A-C/A-D/C-D), full {K_A,K_C,K_D} "
                            "whitening. Sparse GBS -> strict 3-way triads ~0 callable, so pairwise "
                            "(2-of-3 retained) is the unit. PRIMARY=INT+flank; raw/body=sensitivity. "
                            "Discovery=analytic per-pair p vs Bonferroni 0.05/(G_total*n_traits); "
                            "perm=calibration(lambda) only. If G_total small -> underpowered "
                            "exploratory scan, NOT 'oat genome-wide null'."),
                   results=results)
    fp = OUT / f"deploy_oat_{args.mode}.json"
    fp.write_text(json.dumps(payload, indent=2, default=float))
    print(f"  wrote {fp} ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
