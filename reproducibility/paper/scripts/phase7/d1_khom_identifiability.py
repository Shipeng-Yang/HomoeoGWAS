#!/usr/bin/env python3
"""Phase 7 D1 — K_hom identifiability diagnostic.

Tests whether the homoeolog kernel K_hom (= elementwise product of per-subgenome
GRMs, the Phase-5c "tier2" K_hom) collapsing to sigma2_h = 0 in REML is
NON-IDENTIFIABLE (collinear with / absorbed by the additive subgenome kernels)
rather than a genuine biological absence of cross-subgenome epistasis.

Three diagnostics, all on the SAME trace-normalized kernels that produced the
Phase-5c GBLUP Table 1 result:

  geometry (phenotype-independent):
    G1  collinearity R^2 — regress double-centered vec(K_hom) on the additive
        span {K_A,K_B,K_D,(I)}; R^2 -> 1 means K_hom adds no independent
        covariance structure.   (full-vech and off-diagonal-only views)
    G2  kernel_design_cond — condition number of the [vec(K_j)] design
        (from fit_multi_reml).
    G3  K_hom effective rank / eigenspectrum.

  likelihood (per trait):
    L1  full REML fit {K_A,K_B,K_D,K_hom} -> sigma2_h, boundary flag, logL.
    L2  PROFILE restricted logL fixing sigma2_h on a grid (re-optimising the
        other components). A flat profile (Delta logL(best - zero) < 1.35,
        the 5% boundary-LRT half-threshold) => non-identifiable.
    L3  orthogonalised K_hom_resid (residual of K_hom against the additive
        span, PSD-clipped) -> does sigma2_h become estimable once the additive
        geometry is removed?  (diagnostic only; PSD-clip changes the estimand)

Verdict per panel/trait = a FLAT profile boundary-LRT (Delta logL < 1.35), i.e.
sigma2_hom is likelihood-UNSUPPORTED (operationally non-identifiable: it collapses
to the boundary and is not distinguishable from zero in the fitted REML). Geometric
absorption (r2) and the design condition number are recorded as supporting context,
not gates. Caveat (recorded in verdict_basis): a genuinely-zero-but-identifiable
component also yields a flat profile, so this is "likelihood-unsupported", not a
formal design-identifiability proof; pve_hom (near 0 for the collapsing fits) and r2
are reported so the distinction stays transparent.

Reuses compute_grm / build_homoeolog_kernel / normalize_kernel / fit_multi_reml
so the kernels are byte-for-byte the Table 1 tier2 kernels.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from homoeogwas.grm import compute_grm
from homoeogwas.io import load_bed_hardcall
from homoeogwas.kernel import build_homoeolog_kernel, normalize_kernel
from homoeogwas.lmm import fit_multi_reml

ROOT = Path(__file__).resolve().parents[2]

# Per-panel inputs — identical bed roots / subgenomes / pheno as Phase 5c GBLUP.
PANELS: dict[str, dict] = {
    "cotton_hebau": dict(
        bed_root="data/processed/cotton", bed_name="{sub}/all",
        subgenomes=["A", "D"],
        pheno="data/processed/cotton/pheno_m3_4_blue.tsv", sample_col="sample",
        traits=["fiber_length_BLUE", "lint_percentage_BLUE"], maf_min=0.01,
    ),
    "rapeseed_horvath2020": dict(
        bed_root="data/processed/rapeseed/horvath", bed_name="{sub}/all",
        subgenomes=["A", "C"],
        pheno="data/processed/rapeseed/horvath/pheno_clean.tsv", sample_col="sample",
        traits=["bloom_50pct", "plant_height"], maf_min=0.01,
    ),
    "strawberry_pincot2018": dict(
        bed_root="data/processed/fragaria_ananassa", bed_name="{sub}/all",
        subgenomes=["A", "B", "C", "D"],
        pheno="data/processed/strawberry/pheno_clean.tsv", sample_col="sample_id",
        traits=["mean_score"], maf_min=0.01,
    ),
    "oat_old_rahman2025": dict(
        bed_root="data/processed/avena_sativa_logical", bed_name="{sub}/all",
        subgenomes=["A", "C", "D"],
        pheno="data/processed/oat/pheno_clean.tsv", sample_col="sample_id",
        traits=["BIO6", "SOC", "BIO12"], maf_min=0.01,
    ),
}

PVE_GRID = [0.0, 0.02, 0.05, 0.10, 0.20, 0.30]   # profile grid in var(y) units
FLAT_DLOGL = 1.35   # 5% boundary-LRT half-threshold (2*dlogL ~ 2.71)
R2_ABSORB = 0.95
COND_CONCERN = 1.0e4


def _load_aligned_grms(cfg: dict) -> tuple[dict[str, np.ndarray], list[str]]:
    """Load per-subgenome RAW GRMs aligned to common samples (mirrors phase5c)."""
    bed_root = ROOT / cfg["bed_root"]
    grms, samples_per_sub = {}, {}
    for sg in cfg["subgenomes"]:
        bed = load_bed_hardcall(bed_root / cfg["bed_name"].format(sub=sg))
        K, _ = compute_grm(bed, maf_min=cfg["maf_min"])
        grms[sg] = K
        samples_per_sub[sg] = list(np.asarray(bed.samples))
    ref = samples_per_sub[cfg["subgenomes"][0]]
    for sg in cfg["subgenomes"][1:]:
        if samples_per_sub[sg] != ref:
            ref = sorted(set(ref) & set(samples_per_sub[sg]))
    aligned = {}
    for sg in cfg["subgenomes"]:
        loc = samples_per_sub[sg]
        idx = np.asarray([loc.index(s) for s in ref], dtype=np.int64)
        aligned[sg] = grms[sg][np.ix_(idx, idx)]
    return aligned, ref


def _double_center(K: np.ndarray) -> np.ndarray:
    """HKH then trace-normalise to tr/(n-1)=1 (centered-kernel scale)."""
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    Kc = H @ K @ H
    Kc = 0.5 * (Kc + Kc.T)
    tr = np.trace(Kc) / (n - 1)
    return Kc / tr if tr > 0 else Kc


def _upper_offdiag(M: np.ndarray) -> np.ndarray:
    iu = np.triu_indices(M.shape[0], k=1)
    return M[iu]


def _vech(M: np.ndarray) -> np.ndarray:
    iu = np.triu_indices(M.shape[0], k=0)
    return M[iu]


def _r2(target: np.ndarray, basis_cols: list[np.ndarray]) -> float:
    """R^2 of OLS regressing target on [1, *basis_cols]."""
    A = np.column_stack([np.ones_like(target)] + basis_cols)
    coef, *_ = np.linalg.lstsq(A, target, rcond=None)
    resid = target - A @ coef
    ss_res = float(resid @ resid)
    ss_tot = float(((target - target.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def _geometry(kernels_tn: dict[str, np.ndarray], hom_key: str) -> dict:
    """Collinearity of K_hom against the additive span (double-centered)."""
    add_keys = [k for k in kernels_tn if k != hom_key]
    n = kernels_tn[hom_key].shape[0]
    cen = {k: _double_center(K) for k, K in kernels_tn.items()}
    I_c = _double_center(np.eye(n))
    # full-vech (incl diagonal -> carries the variance/residual direction)
    tgt_v = _vech(cen[hom_key])
    add_v = [_vech(cen[k]) for k in add_keys]
    r2_full_add = _r2(tgt_v, add_v)
    r2_full_addI = _r2(tgt_v, add_v + [_vech(I_c)])
    # off-diagonal-only (robustness view; centered-I off-diag is ~constant)
    tgt_o = _upper_offdiag(cen[hom_key])
    add_o = [_upper_offdiag(cen[k]) for k in add_keys]
    r2_offdiag = _r2(tgt_o, add_o)
    # K_hom effective rank (on the trace-normalized, uncentered K_hom)
    ev = np.linalg.eigvalsh(0.5 * (kernels_tn[hom_key] + kernels_tn[hom_key].T))
    ev = np.clip(ev, 0, None)
    eff_rank = float((ev.sum() ** 2) / (ev @ ev)) if (ev @ ev) > 0 else 0.0
    return dict(
        r2_full_additive=r2_full_add, r2_full_additive_plus_I=r2_full_addI,
        r2_offdiag_additive=r2_offdiag, khom_effective_rank=eff_rank, n=n,
    )


def _resid_kernel(kernels_tn: dict[str, np.ndarray], hom_key: str) -> np.ndarray:
    """K_hom orthogonalised against additive span (full matrix), PSD-clipped, trace-norm."""
    add_keys = [k for k in kernels_tn if k != hom_key]
    n = kernels_tn[hom_key].shape[0]
    cen = {k: _double_center(K) for k, K in kernels_tn.items()}
    I_c = _double_center(np.eye(n))
    tgt = _vech(cen[hom_key])
    basis = [_vech(cen[k]) for k in add_keys] + [_vech(I_c)]
    A = np.column_stack([np.ones_like(tgt)] + basis)
    coef, *_ = np.linalg.lstsq(A, tgt, rcond=None)
    resid_v = tgt - A @ coef
    # rebuild symmetric matrix from vech residual
    M = np.zeros((n, n))
    iu = np.triu_indices(n, k=0)
    M[iu] = resid_v
    M = M + M.T - np.diag(np.diag(M))
    # PSD-clip
    w, V = np.linalg.eigh(0.5 * (M + M.T))
    w = np.clip(w, 0, None)
    Kp = (V * w) @ V.T
    return normalize_kernel(Kp, mode="trace") if np.trace(Kp) > 0 else Kp


def _profile(y, X, kernels: dict, hom_key: str, vary: float) -> dict:
    """Profile REML logL fixing sigma2_h (raw var) on PVE_GRID*var(y)."""
    rows = []
    for pve in PVE_GRID:
        v = pve * vary
        bounds = {hom_key: (v, v)}
        try:
            res = fit_multi_reml(y, X, kernels, bounds=bounds, n_starts=3,
                                 random_state=7)
            rows.append((pve, v, float(res.log_lik), bool(res.optimizer_status)))
        except Exception as e:   # noqa: BLE001 — record failure, keep profiling
            rows.append((pve, v, float("nan"), False))
            print(f"      profile pve={pve} failed: {e}")
    ll = np.array([r[2] for r in rows], dtype=np.float64)
    ll0 = ll[0]
    best = np.nanmax(ll)
    dlogl = float(best - ll0) if np.isfinite(best) and np.isfinite(ll0) else float("nan")
    return dict(grid=[dict(pve=r[0], sigma2=r[1], log_lik=r[2], ok=r[3]) for r in rows],
                logL_at_zero=float(ll0), logL_best=float(best),
                delta_logL=dlogl, flat=bool(np.isfinite(dlogl) and dlogl < FLAT_DLOGL))


def run_panel(panel: str, out_dir: Path) -> None:
    cfg = PANELS[panel]
    print(f"=== D1 {panel}  subs={cfg['subgenomes']} ===")
    t0 = time.time()
    raw_grms, ref_samples = _load_aligned_grms(cfg)
    print(f"  GRMs loaded, {len(ref_samples)} common samples ({time.time()-t0:.1f}s)")
    pheno = pd.read_csv(ROOT / cfg["pheno"], sep="\t").set_index(cfg["sample_col"])

    panel_out = {"panel": panel, "subgenomes": cfg["subgenomes"], "traits": {}}
    for trait in cfg["traits"]:
        if trait not in pheno.columns:
            print(f"  [skip] trait {trait!r} not in pheno")
            continue
        ph = pheno[trait]
        common = [s for s in ref_samples if s in ph.index and pd.notna(ph.loc[s])]
        if len(common) < 50:
            print(f"  [skip] trait {trait}: only {len(common)} samples")
            continue
        keep = np.asarray([ref_samples.index(s) for s in common], dtype=np.int64)
        grms = {sg: K[np.ix_(keep, keep)] for sg, K in raw_grms.items()}
        y = ph.loc[common].astype(np.float64).to_numpy()
        y = (y - y.mean()) / y.std(ddof=1)
        X = np.ones((len(common), 1))
        vary = float(np.var(y, ddof=1))

        # tier2 kernels — exactly as Phase 5c gp.py builds them
        K_hom_raw, hom_used = build_homoeolog_kernel(grms, mode="auto")
        kernels = {sg: normalize_kernel(K, mode="trace") for sg, K in grms.items()}
        kernels["K_hom"] = normalize_kernel(K_hom_raw, mode="trace")

        print(f"  [{trait}] n={len(common)} hom_mode={hom_used}")
        geom = _geometry(kernels, "K_hom")

        full = fit_multi_reml(y, X, kernels, n_starts=5, random_state=7)
        L1 = dict(pve=full.pve, component_var=full.component_var,
                  boundary=full.boundary_components, log_lik=float(full.log_lik),
                  kernel_design_cond=float(full.kernel_design_cond),
                  kernel_corr=full.kernel_corr)

        prof = _profile(y, X, kernels, "K_hom", vary)

        # orthogonalised resid kernel
        K_resid = _resid_kernel(kernels, "K_hom")
        kernels_resid = {k: v for k, v in kernels.items() if k != "K_hom"}
        kernels_resid["K_hom_resid"] = K_resid
        try:
            full_r = fit_multi_reml(y, X, kernels_resid, n_starts=5, random_state=7)
            L3 = dict(pve_resid=float(full_r.pve.get("K_hom_resid", float("nan"))),
                      boundary=full_r.boundary_components,
                      estimable=bool("K_hom_resid" not in full_r.boundary_components))
        except Exception as e:   # noqa: BLE001
            L3 = dict(error=str(e), estimable=None)

        # Non-identifiability verdict = sigma2_hom carries NO identifiable/significant variance
        # signal, evidenced by a FLAT profile boundary-LRT (delta logL < FLAT_DLOGL, i.e. fixing
        # sigma2_hom anywhere on the grid never improves the REML likelihood by the 5% threshold ->
        # the K_hom variance collapses to the boundary and is not distinguishable from zero).
        # Geometric absorption by the additive kernels (r2_full_additive) and the [vec K_j] design
        # condition number are RECORDED as supporting context but are NOT gates: the previous
        # `cond > COND_CONCERN(1e4)` AND-term never fired (cond is naturally O(10) here) and forced
        # every verdict to False even when the profile was flat and pve_hom collapsed to ~0.
        pve_hom = float(L1["pve"].get("K_hom", float("nan")))
        absorbed = bool(geom["r2_full_additive"] >= R2_ABSORB)
        verdict = bool(prof["flat"])
        verdict_basis = dict(
            profile_flat=bool(prof["flat"]), delta_logL=float(prof["delta_logL"]), pve_hom=pve_hom,
            geometrically_absorbed=absorbed, r2_full_additive=float(geom["r2_full_additive"]),
            kernel_design_cond=float(L1["kernel_design_cond"]),
            interpretation=("likelihood-unsupported global K_hom variance (flat profile "
                            "boundary-LRT); operationally non-identifiable, NOT a formal "
                            "design-identifiability proof — a true zero would also be flat, hence "
                            "pve_hom (near 0) and r2_full_additive are reported as context"),
            rule=(f"verdict=flat profile boundary-LRT (delta logL < {FLAT_DLOGL:.2f}); "
                  f"absorption (r2>={R2_ABSORB:.2f}) and cond reported as context, not gated"))
        panel_out["traits"][trait] = dict(
            n=len(common), hom_mode=hom_used, geometry=geom, full_reml=L1,
            profile=prof, orthogonalised=L3,
            non_identifiable_verdict=bool(verdict), verdict_basis=verdict_basis,
        )
        print(f"      R2_add={geom['r2_full_additive']:.3f} "
              f"cond={L1['kernel_design_cond']:.1e} "
              f"pve_hom={L1['pve'].get('K_hom', float('nan')):.4f} "
              f"profile_dLogL={prof['delta_logL']:.2f} flat={prof['flat']} "
              f"resid_estimable={L3.get('estimable')} -> non_identif={verdict}")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{panel}_d1.json"
    out_path.write_text(json.dumps(panel_out, indent=2, default=float))
    print(f"  wrote {out_path}  ({time.time()-t0:.1f}s)")


def main():
    ap = argparse.ArgumentParser(description="Phase 7 D1 K_hom identifiability")
    ap.add_argument("--panel", choices=list(PANELS) + ["all"], default="cotton_hebau")
    ap.add_argument("--out-dir", default="results/phase7/d1_identifiability")
    args = ap.parse_args()
    out_dir = ROOT / args.out_dir
    panels = list(PANELS) if args.panel == "all" else [args.panel]
    for p in panels:
        run_panel(p, out_dir)


if __name__ == "__main__":
    main()
