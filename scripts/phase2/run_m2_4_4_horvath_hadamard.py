#!/usr/bin/env python3
"""Phase 2 M2.4.4 — Horvath2020 homoeolog Hadamard kernel inclusion.

Adds the homoeolog Hadamard kernel K_hom = G_A ⊙ G_C as a third random-effect
kernel and tests whether it carries phenotypic variance beyond the additive
A+C subgenome model — the direct test of the paper's Core "homoeolog
co-association" claim:

    y = X β + u_A + u_C + u_hom + e

Headline contrast: A+C+e -> A+C+hom+e  (does K_hom add signal over additive?).

Identifiability caveat: K_hom is the elementwise product of A/C GRMs and may
correlate with them. The summary grades kernel_design_cond; when the variance
split is weakly identified, the trustworthy evidence is the A+C -> A+C+hom
LRT / bootstrap, NOT the PVE_hom point estimate.

acceptance gates the *machinery*; whether K_hom is significant is the
scientific RESULT (reported, not gated).

Usage:
  acceptance : python run_m2_4_4_horvath_hadamard.py --B 200 --n-jobs 8
  paper-grade: python run_m2_4_4_horvath_hadamard.py --B 1000 --n-jobs 32 \
                      --n-starts 50 --run-null-calibration
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from homoeogwas import hadamard_kernel, normalize_kernel  # noqa: E402
from homoeogwas.diagnostics import (  # noqa: E402
    boundary_lrt_table,
    bootstrap_lrt_table,
    compare_nested_reml,
    pve_sensitivity_grid,
)
from homoeogwas.lmm import fit_multi_reml  # noqa: E402

HEADLINE = ("A+C+e", "A+C+hom+e")     # does K_hom add over additive A+C?
FULL_MODEL = "A+C+hom+e"
MULTISTART_GRID = (1, 5, 10, 20, 50)


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o)}")


def _ident_grade(cond: float) -> str:
    """Grade the 3-kernel variance-split identifiability by design cond number."""
    if cond < 1e4:
        return "interpretable"
    if cond < 1e7:
        return "cautious"
    return "weakly-identified"


def load_inputs(grm_npz: Path, pheno_tsv: Path, trait: str):
    """Load RAW GRMs + phenotype; build raw + canonical {A, C, hom} kernels."""
    if not grm_npz.exists():
        sys.exit(f"ERR: M2.1 GRM artifact missing: {grm_npz}")
    if not pheno_tsv.exists():
        sys.exit(f"ERR: pheno_clean.tsv missing: {pheno_tsv}")
    npz = np.load(grm_npz, allow_pickle=True)
    G_A = np.asarray(npz["G_A"], dtype=np.float64)
    G_C = np.asarray(npz["G_C"], dtype=np.float64)
    samples = list(np.asarray(npz["samples"]))
    pheno = pd.read_csv(pheno_tsv, sep="\t").set_index("sample")
    if trait not in pheno.columns:
        sys.exit(f"ERR: trait {trait!r} not in pheno; have {list(pheno.columns)}")
    overlap = [s for s in samples if s in pheno.index and not pd.isna(pheno.loc[s, trait])]
    n = len(overlap)
    if n < 50:
        sys.exit(f"ERR: too few samples {n} after inner-join on {trait!r}")
    idx = {s: i for i, s in enumerate(samples)}
    sel = np.array([idx[s] for s in overlap], dtype=np.int64)
    G_A_sub = G_A[np.ix_(sel, sel)]
    G_C_sub = G_C[np.ix_(sel, sel)]
    # homoeolog Hadamard kernel from RAW GRMs (Schur product theorem -> PSD)
    K_hom_raw = hadamard_kernel({"A": G_A_sub, "C": G_C_sub})
    raw_kernels = {"A": G_A_sub, "C": G_C_sub, "hom": K_hom_raw}
    y = pheno.loc[overlap, trait].astype(np.float64).to_numpy()
    X = np.ones((n, 1), dtype=np.float64)
    meta = {
        "trait": trait, "input_grm_npz": str(grm_npz), "input_pheno_tsv": str(pheno_tsv),
        "n_analysis": n, "phenotype_var": float(np.var(y, ddof=1)),
        "raw_kernel_trace": {k: float(np.trace(K)) for k, K in raw_kernels.items()},
        "K_hom_min_eig": float(np.linalg.eigvalsh(K_hom_raw).min()),
    }
    return y, X, raw_kernels, meta


def make_canonical_kernels(raw_kernels: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Trace-normalize each raw kernel -> canonical kernels for Step 2-4."""
    return {name: normalize_kernel(K, mode="trace") for name, K in raw_kernels.items()}


def multistart_sweep(y, X, kernels, seed) -> pd.DataFrame:
    """Refit the full 3-kernel model at a grid of n_starts; report stability."""
    rows = []
    for ns in MULTISTART_GRID:
        res = fit_multi_reml(y, X, kernels, n_starts=ns, random_state=seed)
        spread = (float(np.max(res.all_start_log_lik) - np.min(res.all_start_log_lik))
                  if len(res.all_start_log_lik) > 1 else 0.0)
        rows.append({
            "n_starts": ns,
            "log_lik": res.log_lik,
            "best_start": res.best_start,
            "all_start_log_lik_spread": spread,
            "pve_A": res.pve["A"], "pve_C": res.pve["C"],
            "pve_hom": res.pve["hom"], "pve_e": res.pve["e"],
            "optimizer_status": bool(res.optimizer_status),
        })
    return pd.DataFrame(rows)


def make_plots(out_dir, trait, full_pve, kernel_corr, sweep):
    paths = []
    # 1) full-model PVE bar
    fig, ax = plt.subplots(figsize=(5, 4))
    keys = ["A", "C", "hom", "e"]
    vals = [full_pve[k] for k in keys]
    bars = ax.bar(keys, vals, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#999999"])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("PVE")
    ax.set_title(f"M2.4.4 A+C+hom PVE — Horvath2020 {trait}")
    fig.tight_layout()
    p = out_dir / f"pve_bar_{trait}.png"
    fig.savefig(p, dpi=120, bbox_inches="tight"); plt.close(fig)
    paths.append(str(p))

    # 2) 3-kernel correlation heatmap
    names = ["A", "C", "hom"]
    M = np.array([[kernel_corr[a][b] for b in names] for a in names])
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(3)); ax.set_xticklabels(names)
    ax.set_yticks(range(3)); ax.set_yticklabels(names)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=9)
    ax.set_title(f"M2.4.4 kernel correlation — {trait}")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    p = out_dir / f"kernel_corr_{trait}.png"
    fig.savefig(p, dpi=120, bbox_inches="tight"); plt.close(fig)
    paths.append(str(p))

    # 3) multi-start sweep: log_lik + PVE_hom vs n_starts
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(sweep["n_starts"], sweep["log_lik"], "o-", color="#1f77b4")
    ax1.set_xlabel("n_starts"); ax1.set_ylabel("full-model log_lik", color="#1f77b4")
    ax1.set_xscale("log")
    ax2 = ax1.twinx()
    ax2.plot(sweep["n_starts"], sweep["pve_hom"], "s--", color="#2ca02c")
    ax2.set_ylabel("PVE_hom", color="#2ca02c")
    ax1.set_title(f"M2.4.4 multi-start stability — {trait}")
    fig.tight_layout()
    p = out_dir / f"multistart_sweep_{trait}.png"
    fig.savefig(p, dpi=120, bbox_inches="tight"); plt.close(fig)
    paths.append(str(p))
    return paths


def main():
    ap = argparse.ArgumentParser(description="M2.4.4 Horvath2020 Hadamard kernel inclusion")
    ap.add_argument("--trait", default="plant_height")
    ap.add_argument("--B", type=int, default=200, help="bootstrap reps for headline contrast")
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--n-starts", type=int, default=20,
                    help="multi-start count for canonical fits (paper: 50)")
    ap.add_argument("--min-success-rate", type=float, default=0.95)
    ap.add_argument("--run-null-calibration", action="store_true",
                    help="also null-calibrate the A+C+e -> A+C+hom+e contrast (slow)")
    ap.add_argument("--null-n-sim", type=int, default=1000)
    ap.add_argument("--grm-npz", default=None)
    ap.add_argument("--pheno-tsv", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--skip-bootstrap", action="store_true")
    ap.add_argument("--skip-plots", action="store_true")
    args = ap.parse_args()

    grm_npz = Path(args.grm_npz) if args.grm_npz else ROOT / "results/phase2/m2_1/horvath2020/grm_A_C.npz"
    pheno_tsv = Path(args.pheno_tsv) if args.pheno_tsv else ROOT / "data/processed/rapeseed/horvath/pheno_clean.tsv"
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "results/phase2/m2_4_4/horvath2020"
    out_dir.mkdir(parents=True, exist_ok=True)
    trait = args.trait
    t0 = time.time()

    print(f"=== M2.4.4 Horvath2020 Hadamard kernel inclusion — trait={trait} ===")
    y, X, raw_kernels, meta = load_inputs(grm_npz, pheno_tsv, trait)
    n = meta["n_analysis"]
    print(f"  n={n}  var(y)={meta['phenotype_var']:.3f}")
    print(f"  RAW kernel trace: " +
          " ".join(f"{k}={v:.1f}" for k, v in meta['raw_kernel_trace'].items()))
    print(f"  K_hom min eig = {meta['K_hom_min_eig']:.3g}")

    canonical = make_canonical_kernels(raw_kernels)

    acceptance: list[dict] = []
    observations: list[dict] = []

    def check(name, ok, msg=""):
        acceptance.append({"check": name, "passed": bool(ok), "message": msg})
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {msg}" if msg else ""))

    def observe(name, value, msg=""):
        observations.append({"observation": name, "value": value, "message": msg})
        print(f"  [obs ] {name} = {value}" + (f" — {msg}" if msg else ""))

    # ---- Step 2: nested REML comparison (J=3 exhaustive => 8 models) --
    print("\n[Step 2] compare_nested_reml (exhaustive, J=3)")
    cmp = compare_nested_reml(
        y, X, canonical, strategy="exhaustive",
        fit_kwargs={"n_starts": args.n_starts, "random_state": args.seed},
    )
    lik = cmp.likelihood_table
    lik_path = out_dir / f"nested_reml_{trait}.tsv"
    lik.to_csv(lik_path, sep="\t", index=False)
    print(f"  wrote {lik_path}  ({len(lik)} models)")
    full_fit = cmp.fits[FULL_MODEL]
    full_pve = {k: float(full_fit.pve[k]) for k in ("A", "C", "hom", "e")}
    cond = float(full_fit.kernel_design_cond)
    grade = _ident_grade(cond)
    print(f"  full {FULL_MODEL}: log_lik={full_fit.log_lik:.4f}  "
          f"PVE={ {k: round(v, 4) for k, v in full_pve.items()} }")
    print(f"  kernel_corr A-C={full_fit.kernel_corr['A']['C']:.3f} "
          f"A-hom={full_fit.kernel_corr['A']['hom']:.3f} "
          f"C-hom={full_fit.kernel_corr['C']['hom']:.3f}")
    print(f"  kernel_design_cond={cond:.3e}  identifiability={grade}")

    check("step2_8_models", len(lik) == 8, f"models={len(lik)}")
    check("step2_full_present", FULL_MODEL in cmp.fits)
    check("step2_loglik_all_finite", bool(np.all(np.isfinite(lik["log_lik"]))))
    check("step2_full_loglik_is_max",
          bool(full_fit.log_lik >= lik["log_lik"].max() - 1e-6))
    check("step2_full_pve_sums_to_1", abs(sum(full_pve.values()) - 1.0) < 1e-9)
    check("step2_full_optimizer_converged", bool(full_fit.optimizer_status))
    observe("identifiability_grade", grade, f"kernel_design_cond={cond:.3e}")

    # ---- Step 3: boundary-corrected LRT (default 7 contrasts) ---------
    print("\n[Step 3] boundary_lrt_table (default, J=3 => 7 contrasts)")
    lrt = boundary_lrt_table(cmp, "default")
    lrt_path = out_dir / f"lrt_table_{trait}.tsv"
    lrt.to_csv(lrt_path, sep="\t", index=False)
    print(f"  wrote {lrt_path}  ({len(lrt)} contrasts)")
    for r in lrt.itertuples():
        print(f"    {r.null_model}→{r.alt_model}: T={r.lrt:.3g} p_boundary={r.p_boundary:.3g}")
    check("step3_lrt_7_contrasts", len(lrt) == 7, f"rows={len(lrt)}")
    check("step3_lrt_nonneg", bool((lrt["lrt"] >= 0).all()))
    check("step3_lrt_all_converged", bool(lrt["both_converged"].all()))

    # ---- headline contrast: K_hom over additive A+C ------------------
    hl = lrt[(lrt.null_model == HEADLINE[0]) & (lrt.alt_model == HEADLINE[1])]
    check("headline_contrast_present", len(hl) == 1)
    hl_row = hl.iloc[0]
    headline_T = float(hl_row["lrt"])
    headline_p = float(hl_row["p_boundary"])
    hl.to_csv(out_dir / f"headline_lrt_{trait}.tsv", sep="\t", index=False)
    print(f"\n[headline] {HEADLINE[0]} → {HEADLINE[1]}  "
          f"T={headline_T:.4g}  p_boundary={headline_p:.4g}")

    # ---- Step 4: parametric bootstrap of the headline contrast -------
    headline_boot_p = None
    if args.skip_bootstrap:
        print("\n[Step 4] SKIPPED (--skip-bootstrap)")
    else:
        print(f"\n[Step 4] bootstrap headline contrast  B={args.B}  n_jobs={args.n_jobs}")
        boot = bootstrap_lrt_table(
            cmp, pairs=[HEADLINE], y=y, X=X, kernels=canonical,
            B=args.B, seed=args.seed, n_jobs=args.n_jobs,
            require_converged=True, min_success_rate=args.min_success_rate,
            on_failure="raise",
        )
        boot.to_csv(out_dir / f"bootstrap_headline_{trait}.tsv", sep="\t", index=False)
        br = boot.iloc[0]
        headline_boot_p = float(br["bootstrap_p"])
        print(f"  bootstrap_p={headline_boot_p:.4g}  B_usable={br['B_usable']}  "
              f"success_rate={br['success_rate']:.3f}")
        check("step4_bootstrap_success_rate",
              bool(br["success_rate"] >= args.min_success_rate),
              f"{br['success_rate']:.3f}")
        check("step4_bootstrap_p_finite", bool(np.isfinite(headline_boot_p)))

    # ---- multi-start stability sweep ---------------------------------
    print(f"\n[multi-start sweep] n_starts ∈ {MULTISTART_GRID}")
    sweep = multistart_sweep(y, X, canonical, args.seed)
    sweep.to_csv(out_dir / f"multistart_sweep_{trait}.tsv", sep="\t", index=False)
    for r in sweep.itertuples():
        print(f"    n_starts={r.n_starts:2d}  log_lik={r.log_lik:.5f}  "
              f"best_start={r.best_start}  PVE_hom={r.pve_hom:.4f}")
    ll_20 = float(sweep[sweep.n_starts == 20]["log_lik"].iloc[0])
    ll_50 = float(sweep[sweep.n_starts == 50]["log_lik"].iloc[0])
    check("multistart_n20_sufficient", (ll_50 - ll_20) < 1e-4,
          f"ll(50)-ll(20)={ll_50 - ll_20:.3g}")
    pve_hom_drift = float(sweep["pve_hom"].max() - sweep["pve_hom"].min())
    observe("pve_hom_multistart_drift", round(pve_hom_drift, 5),
            "PVE_hom spread across n_starts grid")

    # ---- Step 5: PVE sensitivity grid (RAW kernels, J=3) -------------
    print("\n[Step 5] pve_sensitivity_grid (RAW kernels, J=3)")
    sens = pve_sensitivity_grid(
        y, X, raw_kernels,
        reference=("raw", "trace", args.n_starts),
        n_starts_grid=tuple(sorted({1, args.n_starts})),
        random_state=args.seed,
    )
    sens.table.to_csv(out_dir / f"sensitivity_table_{trait}.tsv", sep="\t", index=False)
    sens.drift_table.to_csv(out_dir / f"sensitivity_drift_{trait}.tsv", sep="\t", index=False)
    glob_gen = sens.drift_table[(sens.drift_table.axis == "global")
                                & (sens.drift_table.is_genetic)]
    max_drift = float(glob_gen["pve_range"].max())
    print(f"  global genetic PVE drift = {max_drift:.3g}")
    check("step5_sensitivity_all_converged", bool(sens.table["optimizer_status"].all()))
    check("step5_no_upper_bound_hit", not bool(sens.table["genetic_upper_bound_hit"].any()))
    # Engine stability at the acceptance n_starts: n_starts=1 cells are a
    # deliberate local-optimum stress and may differ; the high-n_starts cells
    # must agree across y-transform / kernel-norm.
    big_ns = max(sorted({1, args.n_starts}))
    big_cells = sens.table[sens.table["n_starts"] == big_ns]
    ns_stable_drift = float(max(
        big_cells[f"pve_{k}"].max() - big_cells[f"pve_{k}"].min()
        for k in ("A", "C", "hom")
    ))
    check(f"step5_n{big_ns}_cells_stable", ns_stable_drift < 1e-3,
          f"genetic PVE drift across norms/y at n_starts={big_ns} = {ns_stable_drift:.3g}")
    # Per-axis attribution: if global drift >> every single-axis drift, the
    # spread is an n_starts=1 local-optimum / interaction effect, not genuine
    # preprocessing sensitivity of the decomposition.
    axis_drift = {}
    for ax in ("y_mode", "kernel_norm", "n_starts"):
        sub = sens.drift_table[(sens.drift_table["axis"] == ax)
                               & sens.drift_table["is_genetic"]]
        axis_drift[ax] = float(sub["pve_range"].max()) if len(sub) else float("nan")
    print(f"  per-axis genetic drift: "
          + " ".join(f"{k}={v:.2g}" for k, v in axis_drift.items()))
    observe("sensitivity_drift_by_axis", {k: round(v, 6) for k, v in axis_drift.items()},
            f"single-axis genetic PVE drift (others at reference); global "
            f"{max_drift:.3g} >> any single axis ⇒ n_starts=1 local-optimum / "
            "interaction effect, not preprocessing sensitivity")

    # ---- optional: null calibration of the headline contrast ---------
    null_cal = None
    if args.run_null_calibration:
        print(f"\n[null calibration] A+C+e → A+C+hom+e  n_sim={args.null_n_sim}")
        from homoeogwas.calibration import (
            run_null_lrt_calibration, scenario_from_reduced_fit,
        )
        scen = scenario_from_reduced_fit(
            "hom_inclusion", y, X, canonical,
            null_model=HEADLINE[0], alt_model=HEADLINE[1],
            n_starts=args.n_starts, random_state=args.seed,
        )
        cal = run_null_lrt_calibration(
            X, canonical, scen, n_sim=args.null_n_sim, seed=args.seed,
            fit_kwargs={"n_starts": args.n_starts}, n_jobs=args.n_jobs,
        )
        cal.type1_table.to_csv(out_dir / f"null_calibration_{trait}.tsv", sep="\t", index=False)
        null_cal = cal.type1_table.to_dict(orient="records")
        for _, rr in cal.type1_table.iterrows():
            print(f"    {rr['method']:11s} α={rr['alpha']:.2f}  "
                  f"type-I={rr['type1_rate']:.4f}  {rr['verdict']}")

    # ---- scientific verdict on K_hom inclusion -----------------------
    sig_boundary = headline_p < 0.05
    sig_boot = (headline_boot_p is not None) and (headline_boot_p < 0.05)
    if args.skip_bootstrap:
        khom_verdict = ("Hadamard increment detected (p_boundary<0.05)" if sig_boundary
                        else "no detectable Hadamard increment over additive A+C")
    else:
        if sig_boundary and sig_boot:
            khom_verdict = "Hadamard increment detected (LRT + bootstrap concordant)"
        elif not sig_boundary and not sig_boot:
            khom_verdict = "no detectable Hadamard increment over additive A+C"
        else:
            khom_verdict = "ambiguous (LRT / bootstrap disagree)"
    observe("K_hom_inclusion_verdict", khom_verdict,
            f"headline p_boundary={headline_p:.3g}"
            + ("" if headline_boot_p is None else f", bootstrap_p={headline_boot_p:.3g}"))

    # ---- summary json ------------------------------------------------
    all_passed = all(c["passed"] for c in acceptance)
    summary = {
        "script": "run_m2_4_4_horvath_hadamard.py",
        "milestone": "M2.4.4",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "runtime_sec": round(time.time() - t0, 1),
        "seed": args.seed, "B": args.B, "n_jobs": args.n_jobs, "n_starts": args.n_starts,
        **meta,
        "out_dir": str(out_dir),
        "full_model": {
            "name": FULL_MODEL,
            "log_lik": float(full_fit.log_lik),
            "pve": full_pve,
            "sigma2": {k: float(v) for k, v in full_fit.sigma2.items()},
            "boundary_components": list(full_fit.boundary_components),
            "kernel_corr": full_fit.kernel_corr,
            "kernel_design_cond": cond,
            "identifiability_grade": grade,
            "best_start": int(full_fit.best_start),
        },
        "headline": {
            "contrast": f"{HEADLINE[0]} -> {HEADLINE[1]}",
            "lrt": headline_T,
            "p_boundary": headline_p,
            "bootstrap_p": headline_boot_p,
        },
        "lrt_table": lrt.to_dict(orient="records"),
        "multistart_sweep": sweep.to_dict(orient="records"),
        "sensitivity_global_genetic_drift": max_drift,
        "null_calibration": null_cal,
        "acceptance": acceptance,
        "acceptance_all_passed": all_passed,
        "observations": observations,
        "scientific_finding": khom_verdict,
    }

    plot_paths = []
    if not args.skip_plots:
        print("\n[plots]")
        plot_paths = make_plots(out_dir, trait, full_pve, full_fit.kernel_corr, sweep)
        for p in plot_paths:
            print(f"  wrote {p}")
        summary["plots"] = plot_paths

    summary_path = out_dir / f"m2_4_4_summary_{trait}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=_json_default)
    print(f"\nwrote {summary_path}")

    n_pass = sum(c["passed"] for c in acceptance)
    print(f"\nacceptance: {n_pass}/{len(acceptance)} checks passed  "
          f"(runtime {summary['runtime_sec']}s)")
    print(f"scientific finding: K_hom inclusion → {khom_verdict}")
    if all_passed:
        print("✅ M2.4.4 Horvath Hadamard-kernel acceptance PASS")
    else:
        failed = [c["check"] for c in acceptance if not c["passed"]]
        print(f"❌ M2.4.4 acceptance FAIL — {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
