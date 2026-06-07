#!/usr/bin/env python3
"""Phase D task-2: dosage-balance (S/I/Q) decomposition for wheat, per pairwise.

Extends the cotton dosage-balance model to hexaploid wheat. For each of the 3
pairwise comparisons (A-B, A-D, B-D), reparameterize each triad's two burdens into:
    S = (zX+zY)/sqrt2  total dosage
    I = (zX-zY)/sqrt2  imbalance (gene-balance axis)
    Q = zX*zY          non-linear epistasis
whitened by full hexaploid LMM {K_A,K_B,K_D}. Report per-component ACAT + lambda +
B-perm empirical type-I + architecture composition, and re-interpret the chr1 B-D
hit and chr5 A-D hit: are they dosage(S)/imbalance(I)/epistasis(Q) driven?

Reuses 04w genome-wide ETL (extract+burden+GRM) and 05 dosage helpers.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats

ROOT = Path("/mnt/7302share/fast_ysp/U7_GWAS")
OUT = ROOT / "results/phase7/bio_wheat"
PHENO = ROOT / "data/processed/wheat/pheno_clean.tsv"
HCTRIADS = ROOT / "data/raw/expression/wheat/HCTriads.csv"
sys.path.insert(0, str(ROOT / "scripts/phase7"))
from d3_perpair_interaction import _acat, _lambda_gc  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "w4", str(ROOT / "scripts/phase7/bio_wheat/04w_wheat_genomewide.py"))
_w4 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_w4)
_w2 = _w4._w2
_S2 = np.sqrt(2.0)

HITS = {"chr1_BD": ("B", "D", "TraesCS1B02G143800", "TraesCS1D02G128400"),
        "chr5_AD": ("A", "D", "TraesCS5A02G169500", "TraesCS5D02G173900")}


def _siq_pvals(Wh, y, ZX, ZY):
    n, G = ZX.shape
    yw = Wh @ y; one_w = Wh @ np.ones(n); df = n - 4
    pS = np.empty(G); pI = np.empty(G); pQ = np.empty(G)
    for g in range(G):
        zX = ZX[:, g]; zY = ZY[:, g]
        S = (zX + zY) / _S2; I = (zX - zY) / _S2; Q = zX * zY
        X = np.column_stack([one_w, Wh @ S, Wh @ I, Wh @ Q])
        beta, *_ = np.linalg.lstsq(X, yw, rcond=None)
        resid = yw - X @ beta; s2 = float(resid @ resid) / df
        se = np.sqrt(np.maximum(s2 * np.diag(np.linalg.pinv(X.T @ X)), 1e-30))
        t = beta / se
        pS[g] = 2 * stats.t.sf(abs(t[1]), df)
        pI[g] = 2 * stats.t.sf(abs(t[2]), df)
        pQ[g] = 2 * stats.t.sf(abs(t[3]), df)
    return pS, pI, pQ


def _perm(KA, KB, KD, ZX, ZY, yshuf, seed):
    try:
        Wh, _ = _w2._whiten3(KA, KB, KD, yshuf, seed=seed)
        pS, pI, pQ = _siq_pvals(Wh, yshuf, ZX, ZY)
        return dict(ok=True, aS=_acat(pS), aI=_acat(pI), aQ=_acat(pQ))
    except Exception:  # noqa: BLE001
        return dict(ok=False)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="flank")
    ap.add_argument("--trait", default="days_to_emerg")
    ap.add_argument("--perm-B", type=int, default=2000)
    ap.add_argument("--n-jobs", type=int, default=48)
    args = ap.parse_args()
    t0 = time.time()
    print("=== wheat dosage-balance (S/I/Q) per pairwise ===")

    rng = np.random.default_rng(7)
    burdens = {}; grms = {}; samples = None
    for s in ("A", "B", "D"):
        genes = _w4._parse_gff(s, 2000)
        g2b, K, smp = _w4._extract_and_burden(s, genes, args.mode, 2000, 150, 3, rng)
        burdens[s] = g2b; grms[s] = K; samples = smp
        print(f"  {s}: {len(g2b)} genes ({time.time()-t0:.1f}s)")
    common = samples

    hc = pd.read_csv(HCTRIADS); hc = hc[hc["cardinality_abs"] == "1:1:1"]
    kept = []
    for _, r in hc.iterrows():
        a = _w4._v10_to_v11(str(r["A"])); b = _w4._v10_to_v11(str(r["B"])); d = _w4._v10_to_v11(str(r["D"]))
        if a in burdens["A"] and b in burdens["B"] and d in burdens["D"]:
            kept.append(dict(A=a, B=b, D=d))
    G = len(kept)
    print(f"  {G} triads ({time.time()-t0:.1f}s)")

    ph = pd.read_csv(PHENO, sep="\t").set_index("sample")
    valid = [s for s in common if s in ph.index and pd.notna(ph.loc[s, args.trait])]
    idx = np.array([common.index(s) for s in valid]); n = len(valid)
    KA = grms["A"][np.ix_(idx, idx)]; KB = grms["B"][np.ix_(idx, idx)]; KD = grms["D"][np.ix_(idx, idx)]
    KA = KA / (np.trace(KA) / n); KB = KB / (np.trace(KB) / n); KD = KD / (np.trace(KD) / n)
    Z = {s: _w2._scols_safe(np.column_stack([burdens[s][t[s]] for t in kept]))[idx] for s in ("A", "B", "D")}
    Z = {s: _w2._scols_safe(Z[s]) for s in ("A", "B", "D")}
    y = _w2._rank_int(np.array([float(ph.loc[s, args.trait]) for s in valid]))
    Wh, cv = _w2._whiten3(KA, KB, KD, y, seed=42)
    print(f"  n={n} G={G} ({time.time()-t0:.1f}s)")

    out = {}
    rng2 = np.random.default_rng(2027)
    perms = [rng2.permutation(n) for _ in range(args.perm_B)]
    for px, py in [("A", "B"), ("A", "D"), ("B", "D")]:
        tag = px + py
        ZX, ZY = Z[px], Z[py]
        pS, pI, pQ = _siq_pvals(Wh, y, ZX, ZY)
        acat = dict(S=float(_acat(pS)), I=float(_acat(pI)), Q=float(_acat(pQ)))
        lam = dict(S=float(_lambda_gc(pS)), I=float(_lambda_gc(pI)), Q=float(_lambda_gc(pQ)))
        bonf = 0.05 / G
        archs = []
        for g in range(G):
            comps = {"S": pS[g], "I": pI[g], "Q": pQ[g]}
            dom = min(comps, key=comps.get)
            archs.append(dom if comps[dom] < bonf else "none")
        pr = Parallel(n_jobs=args.n_jobs)(delayed(_perm)(KA, KB, KD, ZX, ZY, y[perms[i]], 900000 + i)
                                          for i in range(args.perm_B))
        pr = [r for r in pr if r.get("ok")]; B = len(pr)
        emp = {}
        for c, ac in (("S", "aS"), ("I", "aI"), ("Q", "aQ")):
            arr = np.array([r[ac] for r in pr])
            emp[c] = float((1 + int((arr <= acat[c]).sum())) / (B + 1))
        out[tag] = dict(acat=acat, acat_emp=emp, lambda_obs=lam,
                        arch=dict(Counter(archs)), bonf=float(bonf), G=G)
        print(f"  {tag}: ACAT S={acat['S']:.3g}(emp {emp['S']:.3g}) I={acat['I']:.3g}(emp {emp['I']:.3g}) "
              f"Q={acat['Q']:.3g}(emp {emp['Q']:.3g}) | arch={dict(Counter(archs))}")

    # re-interpret hits: which component drives them
    print("\n  === hit decomposition ===")
    hit_decomp = {}
    gidx = {(t["A"], t["B"], t["D"]): i for i, t in enumerate(kept)}
    for name, (px, py, gx, gy) in HITS.items():
        # find the triad containing this pair
        found = None
        for i, t in enumerate(kept):
            if t[px] == gx and t[py] == gy:
                found = i; break
        if found is None:
            print(f"  {name}: triad not in retained set"); continue
        pS, pI, pQ = _siq_pvals(Wh, y, Z[px][:, [found]], Z[py][:, [found]])
        comp = {"S_dosage": float(pS[0]), "I_imbalance": float(pI[0]), "Q_epistasis": float(pQ[0])}
        dom = min(comp, key=comp.get)
        hit_decomp[name] = dict(pair=f"{px}-{py}", **comp, dominant=dom)
        print(f"  {name} ({px}-{py}): S(dosage) p={pS[0]:.3g}  I(imbalance) p={pI[0]:.3g}  "
              f"Q(epistasis) p={pQ[0]:.3g}  -> dominant={dom}")

    full = dict(panel="wheat", mode=args.mode, trait=args.trait, n=n, G=G,
                pairwise=out, hit_decomposition=hit_decomp,
                note=("wheat dosage-balance S/I/Q decomposition per pairwise (A-B/A-D/B-D), "
                      "hexaploid-whitened. S=total dosage, I=imbalance (gene-balance axis), "
                      "Q=epistasis. Re-interprets the GWAS hits: which dosage mechanism "
                      "drives each. Pairs with the cotton dosage-balance model (05_)."))
    (OUT / f"wheat_dosage_balance_{args.mode}.json").write_text(json.dumps(full, indent=2, default=float))
    print(f"  wrote {OUT/f'wheat_dosage_balance_{args.mode}.json'} ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
