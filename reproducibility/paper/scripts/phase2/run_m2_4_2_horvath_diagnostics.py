#!/usr/bin/env python3
"""Phase 2 M2.4.2 — Horvath2020 end-to-end variance-component diagnostics.

Integrates Step 2-5 of M2.4.2 on a single trait:

  Step 2  compare_nested_reml   -> nested_reml_{trait}.tsv
  Step 3  boundary_lrt_table    -> lrt_table_{trait}.tsv
  Step 4  bootstrap_lrt_table   -> bootstrap_lrt_table_{trait}.tsv
  Step 5  pve_sensitivity_grid  -> sensitivity_table_{trait}.tsv
                                 + sensitivity_drift_{trait}.tsv
  + m2_4_2_summary_{trait}.json, diagnostic plots, acceptance assertions.

Kernel discipline (the one fatal-if-wrong rule):
  - Step 2-4 use TRACE-NORMALIZED canonical kernels (the M2.4.1 convention).
  - Step 5 takes the RAW GRMs; pve_sensitivity_grid applies trace / frobenius /
    none itself, so passing pre-normalized kernels would make the "none" cell
    meaningless. The two kernel sets are kept in separate variables and the
    summary records both traces.

Usage:
  acceptance : python run_m2_4_2_horvath_diagnostics.py --B 200  --n-jobs 8
  paper-grade: python run_m2_4_2_horvath_diagnostics.py --B 1000 --n-jobs 32
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

from homoeogwas import normalize_kernel  # noqa: E402
from homoeogwas.diagnostics import (  # noqa: E402
    bootstrap_lrt_table,
    boundary_lrt_table,
    compare_nested_reml,
    pve_sensitivity_grid,
)

# ---------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------

def _json_default(o):
    """Make numpy scalars / arrays JSON-serializable."""
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o)}")


def load_inputs(grm_npz: Path, pheno_tsv: Path, trait: str):
    """Load RAW GRMs + phenotype, inner-join on trait non-NA.

    Returns (y, X, raw_kernels, meta) where raw_kernels = {"A": G_A, "C": G_C}
    are the *un-normalized* VanRaden GRMs subset to the analysis samples.
    """
    if not grm_npz.exists():
        sys.exit(f"ERR: M2.1 GRM artifact missing: {grm_npz}")
    if not pheno_tsv.exists():
        sys.exit(f"ERR: pheno_clean.tsv missing: {pheno_tsv}")

    npz = np.load(grm_npz, allow_pickle=True)
    G_A = np.asarray(npz["G_A"], dtype=np.float64)
    G_C = np.asarray(npz["G_C"], dtype=np.float64)
    samples = list(np.asarray(npz["samples"]))
    n_kernel = len(samples)

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
    y = pheno.loc[overlap, trait].astype(np.float64).to_numpy()
    X = np.ones((n, 1), dtype=np.float64)

    meta = {
        "trait": trait,
        "input_grm_npz": str(grm_npz),
        "input_pheno_tsv": str(pheno_tsv),
        "n_kernel_samples": n_kernel,
        "n_analysis": n,
        "p": 1,
        "phenotype_mean": float(np.mean(y)),
        "phenotype_var": float(np.var(y, ddof=1)),
        "raw_kernel_trace": {"A": float(np.trace(G_A_sub)),
                             "C": float(np.trace(G_C_sub))},
    }
    return y, X, {"A": G_A_sub, "C": G_C_sub}, meta


def make_canonical_kernels(raw_kernels: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Trace-normalize each RAW GRM -> canonical kernels for Step 2-4."""
    return {name: normalize_kernel(K, mode="trace") for name, K in raw_kernels.items()}


# ---------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------

def _neglog10(p) -> float:
    return float(-np.log10(max(float(p), 1e-300)))


def make_plots(out_dir: Path, trait: str, full_pve: dict, lrt_table: pd.DataFrame,
               boot_table: pd.DataFrame | None, sens_table: pd.DataFrame) -> list[str]:
    """Write diagnostic PNGs; return list of paths."""
    paths: list[str] = []

    # 1) full-model PVE bar
    fig, ax = plt.subplots(figsize=(4.5, 4))
    keys = list(full_pve.keys())
    vals = [full_pve[k] for k in keys]
    bars = ax.bar(keys, vals, color=["#1f77b4", "#ff7f0e", "#999999"][:len(keys)])
    for b, v in zip(bars, vals, strict=True):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("PVE")
    ax.set_title(f"M2.4.2 full-model PVE — Horvath2020 {trait}")
    fig.tight_layout()
    p = out_dir / f"pve_bar_{trait}.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    paths.append(str(p))

    # 2) LRT significance per contrast: -log10(p)
    fig, ax = plt.subplots(figsize=(7, 4))
    contrasts = [f"{r.null_model}→{r.alt_model}" for r in lrt_table.itertuples()]
    x = np.arange(len(contrasts))
    series = [("p_naive", lrt_table["p_naive"], "#aaaaaa"),
              ("p_boundary", lrt_table["p_boundary"], "#1f77b4")]
    if boot_table is not None:
        series.append(("bootstrap_p", boot_table["bootstrap_p"], "#d62728"))
    w = 0.8 / len(series)
    for i, (label, col, color) in enumerate(series):
        ax.bar(x + i * w, [_neglog10(v) for v in col], width=w, label=label, color=color)
    ax.set_xticks(x + w * (len(series) - 1) / 2)
    ax.set_xticklabels(contrasts, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("-log10(p)")
    ax.set_title(f"M2.4.2 LRT significance — Horvath2020 {trait}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = out_dir / f"lrt_p_{trait}.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    paths.append(str(p))

    # 3) sensitivity: PVE_A across all grid cells (flat line = robust)
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = [f"{r.y_mode}/{r.kernel_norm}/{r.n_starts}" for r in sens_table.itertuples()]
    x = np.arange(len(labels))
    ax.plot(x, sens_table["pve_A"], "o-", color="#1f77b4", label="PVE_A")
    ax.plot(x, sens_table["pve_C"], "s-", color="#ff7f0e", label="PVE_C")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("PVE")
    ax.set_title(f"M2.4.2 PVE sensitivity grid — Horvath2020 {trait}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = out_dir / f"sensitivity_grid_{trait}.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    paths.append(str(p))

    return paths


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="M2.4.2 Horvath2020 end-to-end diagnostics")
    ap.add_argument("--trait", default="plant_height")
    ap.add_argument("--B", type=int, default=200,
                    help="bootstrap replicates (acceptance 200, paper 1000)")
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--n-starts", type=int, default=10,
                    help="multi-start count for canonical fits (bootstrap inherits it)")
    ap.add_argument("--min-success-rate", type=float, default=0.95)
    ap.add_argument("--grm-npz", default=None)
    ap.add_argument("--pheno-tsv", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--skip-bootstrap", action="store_true",
                    help="skip Step 4 (the slow step)")
    ap.add_argument("--skip-plots", action="store_true")
    args = ap.parse_args()

    grm_npz = Path(args.grm_npz) if args.grm_npz else ROOT / "results/phase2/m2_1/horvath2020/grm_A_C.npz"
    pheno_tsv = Path(args.pheno_tsv) if args.pheno_tsv else ROOT / "data/processed/rapeseed/horvath/pheno_clean.tsv"
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "results/phase2/m2_4_2/horvath2020"
    out_dir.mkdir(parents=True, exist_ok=True)
    trait = args.trait
    t0 = time.time()

    print(f"=== M2.4.2 Horvath2020 diagnostics — trait={trait} ===")
    y, X, raw_kernels, meta = load_inputs(grm_npz, pheno_tsv, trait)
    n = meta["n_analysis"]
    print(f"  inner-join n={n}  var(y)={meta['phenotype_var']:.3f}")
    print(f"  RAW GRM trace: A={meta['raw_kernel_trace']['A']:.2f} "
          f"C={meta['raw_kernel_trace']['C']:.2f}  (n={n})")

    # Canonical (trace-normalized) kernels for Step 2-4
    canonical = make_canonical_kernels(raw_kernels)
    kernel_names = list(canonical.keys())
    full_model = "+".join(kernel_names) + "+e"
    meta["canonical_kernel_trace"] = {k: float(np.trace(canonical[k])) for k in kernel_names}
    print("  canonical trace-norm: " +
          " ".join(f"{k}={meta['canonical_kernel_trace'][k]:.2f}" for k in kernel_names))

    acceptance: list[dict] = []

    def check(name: str, ok: bool, msg: str = ""):
        acceptance.append({"check": name, "passed": bool(ok), "message": msg})
        flag = "PASS" if ok else "FAIL"
        print(f"  [{flag}] {name}" + (f" — {msg}" if msg else ""))

    check("inner_join_n>=50", n >= 50, f"n={n}")
    for k in kernel_names:
        tr = meta["canonical_kernel_trace"][k]
        check(f"canonical_trace_{k}≈n", abs(tr - n) < 1e-6 * max(n, 1.0), f"trace={tr:.4f} n={n}")

    # ---- Step 2: nested REML comparison ------------------------------
    print("\n[Step 2] compare_nested_reml (exhaustive)")
    cmp = compare_nested_reml(
        y, X, canonical, strategy="exhaustive",
        fit_kwargs={"n_starts": args.n_starts, "random_state": args.seed},
    )
    lik = cmp.likelihood_table
    lik_path = out_dir / f"nested_reml_{trait}.tsv"
    lik.to_csv(lik_path, sep="\t", index=False)
    print(f"  wrote {lik_path}  ({len(lik)} models)")
    full_fit = cmp.fits[full_model]
    full_pve = {k: float(full_fit.pve[k]) for k in kernel_names + ["e"]}
    print(f"  full model {full_model}: log_lik={full_fit.log_lik:.4f}  "
          f"PVE={ {k: round(v, 4) for k, v in full_pve.items()} }")

    check("step2_full_model_present", full_model in cmp.fits)
    check("step2_likelihood_all_finite", bool(np.all(np.isfinite(lik["log_lik"]))))
    check("step2_full_loglik_is_max",
          bool(full_fit.log_lik >= lik["log_lik"].max() - 1e-6),
          f"full log_lik={full_fit.log_lik:.4f}")
    check("step2_full_pve_sums_to_1", abs(sum(full_pve.values()) - 1.0) < 1e-9)
    check("step2_full_optimizer_converged", bool(full_fit.optimizer_status))

    # ---- Step 3: boundary-corrected LRT ------------------------------
    print("\n[Step 3] boundary_lrt_table (default 5 contrasts)")
    lrt = boundary_lrt_table(cmp, "default")
    lrt_path = out_dir / f"lrt_table_{trait}.tsv"
    lrt.to_csv(lrt_path, sep="\t", index=False)
    print(f"  wrote {lrt_path}  ({len(lrt)} contrasts)")
    for r in lrt.itertuples():
        print(f"    {r.null_model}→{r.alt_model}: T={r.lrt:.3g} "
              f"p_boundary={r.p_boundary:.3g}")
    check("step3_lrt_5_contrasts", len(lrt) == 5, f"rows={len(lrt)}")
    check("step3_lrt_nonneg", bool((lrt["lrt"] >= 0).all()))
    check("step3_lrt_all_converged", bool(lrt["both_converged"].all()))
    full_vs_null = lrt[(lrt["null_model"] == "e") & (lrt["alt_model"] == full_model)]
    check("step3_full_vs_null_significant",
          bool(len(full_vs_null) == 1 and full_vs_null.iloc[0]["p_boundary"] < 1e-10),
          f"p_boundary={full_vs_null.iloc[0]['p_boundary']:.3g}" if len(full_vs_null) else "missing")

    # ---- Step 4: parametric bootstrap LRT ----------------------------
    boot = None
    boot_path = None
    if args.skip_bootstrap:
        print("\n[Step 4] SKIPPED (--skip-bootstrap)")
    else:
        print(f"\n[Step 4] bootstrap_lrt_table  B={args.B}  n_jobs={args.n_jobs}")
        boot = bootstrap_lrt_table(
            cmp, "default", y=y, X=X, kernels=canonical,
            B=args.B, seed=args.seed, n_jobs=args.n_jobs,
            require_converged=True, min_success_rate=args.min_success_rate,
            on_failure="raise", adjust_method="bh",
        )
        boot_path = out_dir / f"bootstrap_lrt_table_{trait}.tsv"
        boot.to_csv(boot_path, sep="\t", index=False)
        print(f"  wrote {boot_path}")
        for r in boot.itertuples():
            print(f"    {r.null_model}→{r.alt_model}: bootstrap_p={r.bootstrap_p:.4g} "
                  f"q_bh={r.bootstrap_q_bh:.4g} B_usable={r.B_usable} "
                  f"success={r.success_rate:.3f}")
        check("step4_bootstrap_5_contrasts", len(boot) == 5, f"rows={len(boot)}")
        check("step4_bootstrap_B_requested", bool((boot["B_requested"] == args.B).all()))
        check("step4_bootstrap_success_rate",
              bool((boot["success_rate"] >= args.min_success_rate).all()),
              f"min={boot['success_rate'].min():.3f}")
        check("step4_bootstrap_B_usable_positive", bool((boot["B_usable"] > 0).all()))
        check("step4_bootstrap_p_finite", bool(np.all(np.isfinite(boot["bootstrap_p"]))))

    # ---- Step 5: PVE sensitivity grid (RAW kernels!) -----------------
    print("\n[Step 5] pve_sensitivity_grid (RAW GRMs in)")
    n_starts_grid = tuple(sorted({1, args.n_starts}))
    sens = pve_sensitivity_grid(
        y, X, raw_kernels,
        n_starts_grid=n_starts_grid,
        reference=("raw", "trace", args.n_starts),
        random_state=args.seed,
    )
    sens_path = out_dir / f"sensitivity_table_{trait}.tsv"
    drift_path = out_dir / f"sensitivity_drift_{trait}.tsv"
    sens.table.to_csv(sens_path, sep="\t", index=False)
    sens.drift_table.to_csv(drift_path, sep="\t", index=False)
    print(f"  wrote {sens_path}  ({len(sens.table)} cells)")
    print(f"  wrote {drift_path}")
    glob_genetic = sens.drift_table[
        (sens.drift_table["axis"] == "global") & (sens.drift_table["is_genetic"])
    ]
    max_drift = float(glob_genetic["pve_range"].max())
    ref_row = sens.table[sens.table["is_reference"]].iloc[0]
    print(f"  global genetic PVE drift = {max_drift:.3g}  "
          f"reference={sens.reference_cell}")

    check("step5_sensitivity_n_cells",
          len(sens.table) == 2 * 3 * len(n_starts_grid),
          f"cells={len(sens.table)}")
    check("step5_sensitivity_all_converged", bool(sens.table["optimizer_status"].all()))
    check("step5_no_upper_bound_hit", not bool(sens.table["genetic_upper_bound_hit"].any()))
    check("step5_reference_unique", int(sens.table["is_reference"].sum()) == 1)
    check("step5_global_genetic_drift<0.01", max_drift < 0.01, f"drift={max_drift:.3g}")
    ref_vs_full = max(abs(float(ref_row[f"pve_{k}"]) - full_pve[k]) for k in kernel_names)
    check("step5_reference≈step2_full", ref_vs_full < 1e-3,
          f"max|Δ|={ref_vs_full:.3g}")

    # ---- summary json ------------------------------------------------
    all_passed = all(c["passed"] for c in acceptance)
    summary = {
        "script": "run_m2_4_2_horvath_diagnostics.py",
        "milestone": "M2.4.2",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "runtime_sec": round(time.time() - t0, 1),
        "seed": args.seed,
        "B": args.B,
        "n_jobs": args.n_jobs,
        "n_starts": args.n_starts,
        "skip_bootstrap": args.skip_bootstrap,
        **meta,
        "out_dir": str(out_dir),
        "full_model": {
            "name": full_model,
            "log_lik": float(full_fit.log_lik),
            "sigma2": {k: float(v) for k, v in full_fit.sigma2.items()},
            "pve": full_pve,
            "optimizer_status": bool(full_fit.optimizer_status),
            "n_iter": int(full_fit.n_iter),
            "boundary_components": list(full_fit.boundary_components),
            "kernel_corr": full_fit.kernel_corr,
        },
        "lrt_boundary": lrt[["null_model", "alt_model", "lrt", "p_naive",
                             "p_boundary"]].to_dict(orient="records"),
        "bootstrap": (None if boot is None else
                      boot[["null_model", "alt_model", "bootstrap_p", "bootstrap_q_bh",
                            "B_usable", "mcse", "success_rate"]].to_dict(orient="records")),
        "sensitivity": {
            "reference_cell": sens.reference_cell,
            "n_cells": len(sens.table),
            "global_genetic_pve_drift": max_drift,
            "all_cells_converged": bool(sens.table["optimizer_status"].all()),
            "any_upper_bound_hit": bool(sens.table["genetic_upper_bound_hit"].any()),
            "reference_pve": {k: float(ref_row[f"pve_{k}"]) for k in kernel_names + ["e"]},
        },
        "acceptance": acceptance,
        "acceptance_all_passed": all_passed,
        "outputs": {
            "nested_reml": str(lik_path),
            "lrt_table": str(lrt_path),
            "bootstrap_lrt_table": (None if boot_path is None else str(boot_path)),
            "sensitivity_table": str(sens_path),
            "sensitivity_drift": str(drift_path),
        },
    }

    # ---- plots -------------------------------------------------------
    if not args.skip_plots:
        print("\n[plots]")
        plot_paths = make_plots(out_dir, trait, full_pve, lrt, boot, sens.table)
        for p in plot_paths:
            print(f"  wrote {p}")
        summary["outputs"]["plots"] = plot_paths

    summary_path = out_dir / f"m2_4_2_summary_{trait}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=_json_default)
    print(f"\nwrote {summary_path}")

    # ---- verdict -----------------------------------------------------
    n_pass = sum(c["passed"] for c in acceptance)
    print(f"\nacceptance: {n_pass}/{len(acceptance)} checks passed  "
          f"(runtime {summary['runtime_sec']}s)")
    if all_passed:
        print("✅ M2.4.2 Horvath diagnostics acceptance PASS")
    else:
        failed = [c["check"] for c in acceptance if not c["passed"]]
        print(f"❌ M2.4.2 acceptance FAIL — {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
