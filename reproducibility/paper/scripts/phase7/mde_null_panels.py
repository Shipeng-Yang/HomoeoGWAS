#!/usr/bin/env python
"""H4 — minimum detectable effect (MDE) for the calibrated-null species (rapeseed, oat).

Reframes the rapeseed/oat nulls as *calibrated detection boundaries* rather than "no biology".
For the per-pair whitened-GLS interaction t-test, the smallest homoeolog-pair epistatic effect a
panel can detect at a target power, expressed as a fraction of (whitened) phenotypic variance, is

    MDE_PVE(power) = NCP^2 / (NCP^2 + n),   NCP = t_crit(alpha/2, df) + z_power,

where ``alpha`` is the panel's preregistered genome-wide threshold and df = n - 4 (intercept + the
two main effects + interaction). This is scale-free: it depends only on (n, multiplicity, power),
so it is a property of the *design*, not of the (null) result. We report it for every callable set,
then EMPIRICALLY VALIDATE the formula on the real oat whitened design (inject a signal of exactly
MDE_PVE into the realized whitened interaction regressor and confirm detection power ~= target).

Framing (locked): nulls are conditional whitened-scale 80%-power detection limits on pair-epistasis
PVE at the preregistered threshold, NOT evidence of absence (non-detection does not rule out effects
above the MDE with certainty).

Output: results/phase7/mde_null_panels.json
"""
from __future__ import annotations

import glob
import json
import time
from pathlib import Path

import numpy as np
from scipy import stats

import homoeogwas.interact as I

ROOT = Path("/mnt/7302share/fast_ysp/U7_GWAS")
OUT = ROOT / "results/phase7/mde_null_panels.json"


def _ncp_exact(t_crit: float, df: int, power: float) -> float:
    """Solve the EXACT two-sided noncentral-t power equation for the noncentrality at the boundary:
    find ncp with P(|T'(df,ncp)| > t_crit) = power. Falls back to the normal-tail approximation
    ncp ~ t_crit + z_power if the root solve fails."""
    from scipy.optimize import brentq

    def pw(ncp):
        return float(stats.nct.sf(t_crit, df, ncp) + stats.nct.cdf(-t_crit, df, ncp)) - power
    try:
        lo, hi = 0.0, t_crit + 20.0
        if pw(lo) > 0 or pw(hi) < 0:
            raise ValueError
        return float(brentq(pw, lo, hi, xtol=1e-4))
    except Exception:   # noqa: BLE001
        return float(t_crit + stats.norm.isf(1.0 - power))


def mde_pve(n: int, alpha: float, power: float, n_params: int = 4) -> dict:
    """Conditional whitened-scale ``power``-detection limit for a two-sided interaction t-test at
    threshold ``alpha``: the min interaction PVE = ncp^2/(ncp^2+n) using the EXACT noncentral-t
    noncentrality (with the normal-tail ncp = t_crit + z_power reported for cross-check)."""
    df = max(n - n_params, 1)
    t_crit = float(stats.t.isf(alpha / 2.0, df))      # critical value (exact t)
    ncp = _ncp_exact(t_crit, df, power)               # exact noncentral-t noncentrality
    ncp_approx = float(t_crit + stats.norm.isf(1.0 - power))
    pve = ncp * ncp / (ncp * ncp + n)
    return dict(n=int(n), alpha=float(alpha), df=int(df), power=float(power), t_crit=t_crit,
                ncp=float(ncp), ncp_normal_approx=ncp_approx, mde_pve=float(pve))


def analytic_table():
    """MDE per callable set for rapeseed (4 sets x per-trait n) + oat (3 pair-sets, n=737)."""
    rows = {"rapeseed": {}, "oat": {}}

    # rapeseed: each deploy_*.json has G and by_trait[t].INT_primary.n; use median trait n
    for f in sorted(glob.glob(str(ROOT / "results/phase7/bio_rapeseed/deploy_*.json"))):
        d = json.load(open(f))
        setname = f.split("deploy_")[1].replace(".json", "")
        G = int(d["G"])
        ns = [v["INT_primary"]["n"] for v in d["by_trait"].values() if "INT_primary" in v]
        if not ns or G < 1:
            continue
        n_med = int(np.median(ns))
        n_traits = len(d["traits_run"])
        a_within = 0.05 / G                            # within-trait genome-wide (per set)
        a_across = 0.05 / (G * n_traits)               # across all traits in this set
        rows["rapeseed"][setname] = dict(
            G=G, n_median=n_med, n_traits=n_traits,
            within_trait=dict(alpha=a_within, **{k: v for k, v in mde_pve(n_med, a_within, 0.8).items()
                                                 if k in ("mde_pve", "ncp")},
                              mde_pve_50=mde_pve(n_med, a_within, 0.5)["mde_pve"]),
            across_trait=dict(alpha=a_across, mde_pve_80=mde_pve(n_med, a_across, 0.8)["mde_pve"]))

    # oat: G_total=69 (AC/AD/CD), n=737, preregistered genome-wide alpha
    od = json.load(open(ROOT / "results/phase7/bio_oat/deploy_oat_flank.json"))
    n_oat = int(od["results"]["BIO1"]["n"]) if "results" in od else 737
    G_total = int(od["G_total"])
    a_oat = float(od["alpha_genomewide"])              # 0.05/(G_total*n_traits)
    rows["oat"]["flank_all"] = dict(
        G_total=G_total, callable_pairs=od["callable_pairs"], n=n_oat, n_traits=od["n_traits"],
        preregistered=dict(alpha=a_oat, mde_pve_80=mde_pve(n_oat, a_oat, 0.8)["mde_pve"],
                           mde_pve_50=mde_pve(n_oat, a_oat, 0.5)["mde_pve"]),
        within_trait=dict(alpha=0.05 / G_total,
                          mde_pve_80=mde_pve(n_oat, 0.05 / G_total, 0.8)["mde_pve"]))
    return rows


def validate_oat(set_name="AD", pve_target=None, R=500, alpha=None):
    """Empirically confirm the analytic MDE on the REAL oat whitened design: inject a signal of
    exactly ``pve_target`` into each pair's realized whitened interaction regressor and measure
    detection power at ``alpha`` (should be ~= the 0.8 used to derive pve_target)."""
    sub_map = {"A": "A", "C": "C", "D": "D"}
    X = {s: np.load(ROOT / f"results/phase7/bio_oat/X_{s}.npy", mmap_mode="r") for s in sub_map}
    gene_snp = {}
    for s in sub_map:
        z = np.load(ROOT / f"results/phase7/bio_oat/snp_to_gene_{s}_flank.npz", allow_pickle=True)
        gene_snp[s] = {g: np.asarray(z["snp_idx"][i], int) for i, g in enumerate(z["gene_ids"].tolist())}
    import pandas as pd
    pairs = pd.read_csv(ROOT / "results/phase7/bio_oat/homoeolog_pairs.tsv", sep="\t")
    pairs = pairs[pairs["sub_pair"] == set_name]
    sx, sy = set_name[0], set_name[1]
    n = X[sx].shape[0]
    rng = np.random.default_rng(11)

    # build whitener once from the three-subgenome GRMs (grm_from_X), null LMM on a standardized y
    kernels = {s: I.grm_from_X(np.asarray(X[s])) for s in sub_map}
    y0 = I.rank_int(rng.standard_normal(n))
    Wh, _ = I.whiten_multi(kernels, y0, seed=42)

    cap = 150
    powers = []
    used = 0
    for _, row in pairs.iterrows():
        gx, gy = row["gene_x"], row["gene_y"]
        if gx not in gene_snp[sx] or gy not in gene_snp[sy]:
            continue
        bx = I.scols_safe(I.block_burden_capped(np.asarray(X[sx]), gene_snp[sx][gx], cap, rng).reshape(-1, 1)).ravel()
        by = I.scols_safe(I.block_burden_capped(np.asarray(X[sy]), gene_snp[sy][gy], cap, rng).reshape(-1, 1)).ravel()
        # realized whitened design columns; residualize the interaction on [1,bX,bY] (whitened)
        Cw = Wh @ np.ones((n, 1))
        cX, cY = Wh @ bx, Wh @ by
        cI = Wh @ (bx * by)
        Xm = np.column_stack([Cw, cX, cY])
        b, *_ = np.linalg.lstsq(Xm, cI, rcond=None)
        c = cI - Xm @ b
        nc = float(np.linalg.norm(c))
        if nc < 1e-8:
            continue
        c = c / nc                                     # unit-norm interaction direction (df=1)
        Xdes = np.column_stack([Cw, cX, cY, c])
        XtXinv = np.linalg.pinv(Xdes.T @ Xdes)
        gamma = np.sqrt(pve_target * n / (1.0 - pve_target))   # signal scale for target PVE
        df = n - 4
        hits = 0
        for _ in range(R):
            yw = gamma * c + rng.standard_normal(n)    # inject signal of PVE=gamma^2/(gamma^2+n)
            beta, *_ = np.linalg.lstsq(Xdes, yw, rcond=None)
            resid = yw - Xdes @ beta
            s2 = float(resid @ resid) / df
            se = np.sqrt(max(s2 * XtXinv[3, 3], 1e-30))
            p = 2.0 * stats.t.sf(abs(beta[3] / se), df)
            if p < alpha:
                hits += 1
        powers.append(hits / R)
        used += 1
    return dict(set=set_name, n=int(n), pairs_used=used, pve_target=float(pve_target),
                alpha=float(alpha), R=R, power_median=float(np.median(powers)) if powers else None,
                power_min=float(np.min(powers)) if powers else None,
                power_max=float(np.max(powers)) if powers else None)


def main():
    t0 = time.time()
    out = {"tool": "homoeogwas", "analysis": "mde_null_panels",
           "definition": ("MDE_PVE(power) = ncp^2/(ncp^2+n), ncp = EXACT two-sided noncentral-t "
                          "noncentrality at threshold alpha (normal-tail t_crit+z_power reported as "
                          "cross-check); interaction PVE = fraction of whitened phenotypic variance; "
                          "df=n-4. CONDITIONAL on the fitted whitening/model design."),
           "framing": ("nulls are CONDITIONAL whitened-scale 80%-power DETECTION LIMITS for "
                       "homoeolog-pair epistasis PVE at the preregistered threshold: non-detection "
                       "does not rule out effects above the MDE with certainty, but bounds the "
                       "panel's sensitivity — NOT evidence of absence of epistasis."),
           "analytic": analytic_table()}
    print(f"  analytic table done ({time.time()-t0:.0f}s)", flush=True)

    # empirical validation: inject at the oat preregistered MDE_PVE(0.8) and confirm power ~= 0.8
    oat = out["analytic"]["oat"]["flank_all"]
    pve80 = oat["preregistered"]["mde_pve_80"]
    a_oat = oat["preregistered"]["alpha"]
    out["empirical_validation_oat_AD"] = validate_oat("AD", pve_target=pve80, R=500, alpha=a_oat)
    print(f"  oat empirical validation done ({time.time()-t0:.0f}s)", flush=True)

    out["runtime_sec"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(out, indent=2, default=float))
    print(f"wrote {OUT} ({out['runtime_sec']}s)")


if __name__ == "__main__":
    main()
