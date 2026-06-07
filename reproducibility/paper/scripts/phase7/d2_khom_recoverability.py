#!/usr/bin/env python3
"""Phase 7 D2 arm-1 — K_hom recoverability control (best case).

Question: if the EXACT fitted global K_hom is the true data-generating
covariance, can REML recover its variance component under the real GRMs /
sample size / kernel geometry?  If a component explaining 20% of phenotypic
variance is NOT recoverable when simulated from the kernel itself, the global
Hadamard K_hom is not fit-for-purpose (the flat D1 profiles then reflect
non-identifiability, not biological absence) — motivating a synteny-aware
homoeolog-pair kernel (D2 arm-2).

Design (Codex D2 dual-plan + arm-1 codex-check fixes):
  - real trace-normalized K_A,(K_B,)K_D and K_hom (= Phase-5c tier2 kernels).
  - pre-sim geometry: decompose K_hom ≈ a·I + Σ b·K_add + residual; residual-
    norm fraction = the identifiable part.
  - simulate y = Σ g_sub + g_hom + e; ADDITIVE_PVE configurable (set 0 for the
    no-additive arm that isolates K_hom-vs-residual identifiability); true
    PVE_hom swept; σ²_e fills the rest; realized PVE per component recorded
    (sim sanity).
  - fit fit_multi_reml{additive..,K_hom}; record FULL pve decomposition per rep
    (leakage of σ²_h into K_A/K_D/residual), boundary rate, and
    2·(logL_full − logL_null) with σ²_h fixed at 0.
  - null (true PVE_hom=0) run with more reps -> stable empirical 5% LRT crit.

Verdict @ true PVE_hom=0.20 (Codex thresholds):
  unrecoverable : median PVE_hat<0.03, boundary>70%, calibrated power<0.2
  weak          : median 0.05-0.12, power 0.2-0.5
  recoverable   : median 0.12-0.28, boundary<40%, power>0.6
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from homoeogwas.grm import compute_grm
from homoeogwas.io import load_bed_hardcall
from homoeogwas.kernel import build_homoeolog_kernel, normalize_kernel
from homoeogwas.lmm import fit_multi_reml

ROOT = Path(__file__).resolve().parents[2]

PANELS = {
    "cotton_hebau": dict(bed_root="data/processed/cotton", bed_name="{sub}/all",
                         subgenomes=["A", "D"],
                         pheno="data/processed/cotton/pheno_m3_4_blue.tsv",
                         sample_col="sample", trait="fiber_length_BLUE", maf_min=0.01),
    "oat_old_rahman2025": dict(bed_root="data/processed/avena_sativa_logical",
                               bed_name="{sub}/all", subgenomes=["A", "C", "D"],
                               pheno="data/processed/oat/pheno_clean.tsv",
                               sample_col="sample_id", trait="BIO12", maf_min=0.01),
}

PVE_HOM_GRID = [0.05, 0.10, 0.20, 0.30]   # signal points (null=0 run separately)


def _load_kernels(cfg):
    bed_root = ROOT / cfg["bed_root"]
    grms, samp = {}, {}
    for sg in cfg["subgenomes"]:
        bed = load_bed_hardcall(bed_root / cfg["bed_name"].format(sub=sg))
        K, _ = compute_grm(bed, maf_min=cfg["maf_min"])
        grms[sg] = K
        samp[sg] = list(np.asarray(bed.samples))
    ref = samp[cfg["subgenomes"][0]]
    for sg in cfg["subgenomes"][1:]:
        if samp[sg] != ref:
            ref = sorted(set(ref) & set(samp[sg]))
    for sg in cfg["subgenomes"]:
        loc = samp[sg]; idx = np.asarray([loc.index(s) for s in ref], dtype=np.int64)
        grms[sg] = grms[sg][np.ix_(idx, idx)]
    ph = pd.read_csv(ROOT / cfg["pheno"], sep="\t").set_index(cfg["sample_col"])[cfg["trait"]]
    common = [s for s in ref if s in ph.index and pd.notna(ph.loc[s])]
    keep = np.asarray([ref.index(s) for s in common], dtype=np.int64)
    grms = {sg: K[np.ix_(keep, keep)] for sg, K in grms.items()}
    K_hom_raw, hom_used = build_homoeolog_kernel(grms, mode="auto")
    kernels = {sg: normalize_kernel(K, mode="trace") for sg, K in grms.items()}
    kernels["K_hom"] = normalize_kernel(K_hom_raw, mode="trace")
    return kernels, len(common), hom_used


def _geometry_decomp(kernels):
    add = [k for k in kernels if k != "K_hom"]
    n = kernels["K_hom"].shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    cen = lambda K: H @ K @ H  # noqa: E731
    iu = np.triu_indices(n, 0)
    tgt = cen(kernels["K_hom"])[iu]
    basis = [cen(np.eye(n))[iu]] + [cen(kernels[k])[iu] for k in add]
    A = np.column_stack([np.ones_like(tgt)] + basis)
    coef, *_ = np.linalg.lstsq(A, tgt, rcond=None)
    resid = tgt - A @ coef
    return dict(resid_norm_frac=float(np.linalg.norm(resid) / np.linalg.norm(tgt)),
                coef_I=float(coef[1]),
                coef_add={k: float(c) for k, c in zip(add, coef[2:], strict=True)})


def _chol(K):
    n = K.shape[0]
    for j in (1e-8, 1e-6, 1e-4):
        try:
            return np.linalg.cholesky(K + j * np.eye(n))
        except np.linalg.LinAlgError:
            continue
    w, V = np.linalg.eigh(0.5 * (K + K.T))
    return V * np.sqrt(np.clip(w, 0, None))


def _one_rep(kernels, chols, add_keys, pve_hom, add_pve, seed):
    rng = np.random.default_rng(seed)
    n = next(iter(kernels.values())).shape[0]
    sig_add = (add_pve / len(add_keys)) if add_keys else 0.0
    comps = {}
    y = np.zeros(n)
    for k in add_keys:
        g = np.sqrt(sig_add) * (chols[k] @ rng.standard_normal(n)) if sig_add > 0 else np.zeros(n)
        comps[k] = g; y += g
    g_h = np.sqrt(pve_hom) * (chols["K_hom"] @ rng.standard_normal(n)) if pve_hom > 0 else np.zeros(n)
    comps["K_hom"] = g_h; y += g_h
    sig_e = max(1.0 - add_pve - pve_hom, 1e-6)
    e = np.sqrt(sig_e) * rng.standard_normal(n)
    y += e
    vy = float(np.var(y, ddof=1))
    realized = {k: float(np.var(g, ddof=1) / vy) for k, g in comps.items()}
    realized["e"] = float(np.var(e, ddof=1) / vy)
    X = np.ones((n, 1))
    try:
        full = fit_multi_reml(y, X, kernels, n_starts=3, random_state=int(seed) % 100000)
        null = fit_multi_reml(y, X, kernels, bounds={"K_hom": (0.0, 0.0)},
                              n_starts=3, random_state=int(seed) % 100000)
        pve_full = {k: float(full.pve.get(k, 0.0)) for k in list(kernels) + ["e"]}
        boundary = "K_hom" in full.boundary_components
        lrt = max(0.0, 2.0 * (float(full.log_lik) - float(null.log_lik)))
        return dict(pve_hat=float(full.pve.get("K_hom", 0.0)), boundary=bool(boundary),
                    lrt=lrt, ok=True, pve_full=pve_full, realized=realized)
    except Exception:  # noqa: BLE001
        return dict(pve_hat=float("nan"), boundary=True, lrt=float("nan"), ok=False,
                    pve_full={}, realized=realized)


def _summarize(res, true_pve, crit):
    pve_hat = np.array([r["pve_hat"] for r in res], dtype=np.float64)
    boundary = np.array([r["boundary"] for r in res], dtype=bool)
    lrt = np.array([r["lrt"] for r in res], dtype=np.float64)
    ok = np.array([r["ok"] for r in res], dtype=bool)
    fin = np.isfinite(pve_hat)
    # leakage: mean fitted PVE per component
    comp_keys = sorted({k for r in res for k in r["pve_full"]})
    leak = {k: float(np.nanmean([r["pve_full"].get(k, np.nan) for r in res])) for k in comp_keys}
    realized = {k: float(np.nanmean([r["realized"].get(k, np.nan) for r in res]))
                for k in sorted({k for r in res for k in r["realized"]})}
    power = float(np.mean(lrt[np.isfinite(lrt)] > crit)) if np.isfinite(crit) else float("nan")
    rmse = float(np.sqrt(np.nanmean((pve_hat[fin] - true_pve) ** 2))) if fin.any() else float("nan")
    return dict(
        true_pve_hom=true_pve, n_ok=int(ok.sum()),
        pve_hat_median=float(np.nanmedian(pve_hat)), pve_hat_mean=float(np.nanmean(pve_hat)),
        pve_hat_iqr=[float(np.nanpercentile(pve_hat, 25)), float(np.nanpercentile(pve_hat, 75))],
        rmse=rmse, bias=float(np.nanmean(pve_hat[fin]) - true_pve) if fin.any() else float("nan"),
        frac_above_half_true=float(np.mean(pve_hat[fin] > 0.5 * true_pve)) if (true_pve > 0 and fin.any()) else float("nan"),
        boundary_rate=float(np.mean(boundary)), lrt_median=float(np.nanmedian(lrt)),
        calibrated_power=power, leakage_fitted_pve=leak, realized_sim_pve=realized)


def run_panel(panel, n_rep, null_rep, n_jobs, add_pve, tag, out_dir):
    cfg = PANELS[panel]
    print(f"=== D2 arm-1 {panel} trait={cfg['trait']} add_pve={add_pve} tag={tag} ===")
    t0 = time.time()
    kernels, n, hom_used = _load_kernels(cfg)
    add_keys = [k for k in kernels if k != "K_hom"]
    chols = {k: _chol(K) for k, K in kernels.items()}
    geom = _geometry_decomp(kernels)
    print(f"  n={n} hom_mode={hom_used} resid_norm_frac={geom['resid_norm_frac']:.3f} "
          f"({time.time()-t0:.1f}s)")

    out = {"panel": panel, "trait": cfg["trait"], "n": n, "hom_mode": hom_used,
           "additive_pve": add_pve, "n_rep": n_rep, "null_rep": null_rep,
           "geometry": geom, "grid": {}}

    # null (PVE_hom=0) with more reps -> stable 5% crit
    null_res = Parallel(n_jobs=n_jobs)(
        delayed(_one_rep)(kernels, chols, add_keys, 0.0, add_pve, 70_000 + r)
        for r in range(null_rep))
    null_lrt = np.array([r["lrt"] for r in null_res], dtype=np.float64)
    crit = float(np.nanpercentile(null_lrt[np.isfinite(null_lrt)], 95))
    out["grid"]["0.00"] = _summarize(null_res, 0.0, crit)
    out["empirical_crit95"] = crit
    print(f"  PVE_hom=0.00 (null,{null_rep}): crit95={crit:.2f} "
          f"boundary={out['grid']['0.00']['boundary_rate']:.2f} "
          f"power={out['grid']['0.00']['calibrated_power']:.2f} "
          f"leak={ {k: round(v,3) for k,v in out['grid']['0.00']['leakage_fitted_pve'].items()} }")

    for pve in PVE_HOM_GRID:
        res = Parallel(n_jobs=n_jobs)(
            delayed(_one_rep)(kernels, chols, add_keys, pve, add_pve, 10_000 * int(pve * 100) + r)
            for r in range(n_rep))
        rec = _summarize(res, pve, crit)
        out["grid"][f"{pve:.2f}"] = rec
        print(f"  PVE_hom={pve:.2f}: med={rec['pve_hat_median']:.3f} mean={rec['pve_hat_mean']:.3f} "
              f"IQR={[round(x,3) for x in rec['pve_hat_iqr']]} bd={rec['boundary_rate']:.2f} "
              f"pw={rec['calibrated_power']:.2f} frac>½tru={rec['frac_above_half_true']:.2f} "
              f"leak={ {k: round(v,3) for k,v in rec['leakage_fitted_pve'].items()} }")

    g20 = out["grid"].get("0.20", {})
    med, br, pw = g20.get("pve_hat_median", 0), g20.get("boundary_rate", 1), g20.get("calibrated_power", 0)
    out["verdict_at_pve20"] = ("unrecoverable" if (med < 0.03 and br > 0.70 and pw < 0.2)
                               else "recoverable" if (med > 0.12 and br < 0.40 and pw > 0.6)
                               else "weak")
    print(f"  VERDICT @ PVE=0.20: {out['verdict_at_pve20']}")
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{panel}_d2arm1_{tag}.json"
    p.write_text(json.dumps(out, indent=2, default=float))
    print(f"  wrote {p}  ({time.time()-t0:.1f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", choices=list(PANELS), default="cotton_hebau")
    ap.add_argument("--n-rep", type=int, default=200)
    ap.add_argument("--null-rep", type=int, default=500)
    ap.add_argument("--n-jobs", type=int, default=32)
    ap.add_argument("--additive-pve", type=float, default=0.40)
    ap.add_argument("--tag", default="withadd")
    ap.add_argument("--out-dir", default="results/phase7/d2_recoverability")
    args = ap.parse_args()
    run_panel(args.panel, args.n_rep, args.null_rep, args.n_jobs, args.additive_pve,
              args.tag, ROOT / args.out_dir)


if __name__ == "__main__":
    main()
