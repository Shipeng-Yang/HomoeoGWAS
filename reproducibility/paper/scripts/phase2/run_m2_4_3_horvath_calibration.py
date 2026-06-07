#!/usr/bin/env python3
"""Phase 2 M2.4.3 — Horvath2020 null-simulation calibration of the
multi-kernel variance-component LRT.

Validates that the M2.4.2 boundary-corrected LRT controls type-I error under
H0 (a variance component truly 0). Three null scenarios, σ² drawn from the
*reduced* (null) model fitted on the real trait:

  S1  global null      true σ²_A=σ²_C=0   contrast e   -> A+C+e   df=2
  S2  C null given A   true σ²_C=0        contrast A+e -> A+C+e   df=1
  S3  A null given C   true σ²_A=0        contrast C+e -> A+C+e   df=1

S2/S3 are the realistic correlated-kernel nulls (Horvath A/C corr ~0.69),
where the asymptotic Stram-Lee binomial chi-bar weights are only approximate.

Three p-values are calibrated per replicate: p_naive (χ²), p_mixture
(boundary-corrected), and — if --bootstrap-B > 0 — bootstrap_p.

Acceptance gates the *machinery* (success / convergence / structure); whether
p_mixture is calibrated vs anti-conservative is the scientific RESULT and is
reported, not gated.

Usage:
  acceptance : python run_m2_4_3_horvath_calibration.py --n-sim 1000 --n-jobs 8
  paper-grade: python run_m2_4_3_horvath_calibration.py --n-sim 5000 --n-jobs 32 \
                      --bootstrap-B 199 --bootstrap-n-sim 200
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
from homoeogwas.calibration import (  # noqa: E402
    _chibar_quantile,
    _chibar_weights,
    run_null_lrt_calibration,
    scenario_from_reduced_fit,
)

# (label, null_model, alt_model, description)
SCENARIOS = [
    ("S1_global_null", "e", "A+C+e", "true sigma2_A=sigma2_C=0; test e -> A+C+e (df=2)"),
    ("S2_C_null_given_A", "A+e", "A+C+e", "true sigma2_C=0; test A+e -> A+C+e (df=1)"),
    ("S3_A_null_given_C", "C+e", "A+C+e", "true sigma2_A=0; test C+e -> A+C+e (df=1)"),
]


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


def load_inputs(grm_npz: Path, pheno_tsv: Path, trait: str):
    """Load RAW GRMs + phenotype, inner-join; return (y, X, canonical_kernels, meta)."""
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
    # Canonical trace-normalized kernels (M2.4.2 convention)
    K_A = normalize_kernel(G_A[np.ix_(sel, sel)], mode="trace")
    K_C = normalize_kernel(G_C[np.ix_(sel, sel)], mode="trace")
    y = pheno.loc[overlap, trait].astype(np.float64).to_numpy()
    X = np.ones((n, 1), dtype=np.float64)
    meta = {
        "trait": trait, "input_grm_npz": str(grm_npz), "input_pheno_tsv": str(pheno_tsv),
        "n_analysis": n, "phenotype_var": float(np.var(y, ddof=1)),
    }
    return y, X, {"A": K_A, "C": K_C}, meta


def make_plots(out_dir: Path, trait: str, results: dict) -> list[str]:
    """type-I calibration bar + LRT chi-bar QQ panels."""
    paths: list[str] = []

    # 1) empirical type-I @ alpha=0.05 with Wilson CI, nominal line
    fig, ax = plt.subplots(figsize=(7.5, 4))
    scen_names = list(results.keys())
    methods = ["p_naive", "p_mixture"]
    if any(results[s]["has_bootstrap"] for s in scen_names):
        methods.append("bootstrap_p")
    colors = {"p_naive": "#aaaaaa", "p_mixture": "#1f77b4", "bootstrap_p": "#d62728"}
    x = np.arange(len(scen_names))
    w = 0.8 / len(methods)
    for i, m in enumerate(methods):
        rates, los, his = [], [], []
        for s in scen_names:
            t1 = results[s]["type1"]
            row = t1[(t1.method == m) & (t1.alpha == 0.05)]
            if len(row):
                r = row.iloc[0]
                rates.append(r["type1_rate"])
                los.append(r["type1_rate"] - r["ci_lo"])
                his.append(r["ci_hi"] - r["type1_rate"])
            else:
                rates.append(np.nan)
                los.append(0)
                his.append(0)
        ax.bar(x + i * w, rates, width=w, yerr=[los, his], capsize=3,
               label=m, color=colors[m])
    ax.axhline(0.05, ls="--", color="k", lw=1, label="nominal α=0.05")
    ax.set_xticks(x + w * (len(methods) - 1) / 2)
    ax.set_xticklabels(scen_names, fontsize=8)
    ax.set_ylabel("empirical type-I rate @ α=0.05")
    ax.set_title(f"M2.4.3 LRT null calibration — Horvath2020 {trait}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = out_dir / f"calibration_type1_{trait}.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    paths.append(str(p))

    # 2) LRT statistic QQ vs theoretical chi-bar mixture
    fig, axes = plt.subplots(1, len(scen_names), figsize=(4.2 * len(scen_names), 4))
    if len(scen_names) == 1:
        axes = [axes]
    for ax, s in zip(axes, scen_names, strict=True):
        rep = results[s]["replicate"]
        T = np.sort(rep.loc[rep["success"], "lrt"].to_numpy(dtype=np.float64))
        df = int(results[s]["df_added"])
        w_chi = _chibar_weights(df)
        m = len(T)
        pp = (np.arange(1, m + 1) - 0.5) / m
        theo = np.array([_chibar_quantile(q, w_chi) for q in pp])
        lim = max(float(theo.max()), float(T.max()), 1.0) * 1.05
        ax.plot([0, lim], [0, lim], ls="--", color="k", lw=1)
        ax.scatter(theo, T, s=8, color="#1f77b4", alpha=0.5)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_xlabel(f"theoretical chi-bar (df={df})")
        ax.set_ylabel("empirical LRT")
        ax.set_title(s, fontsize=9)
    fig.suptitle(f"M2.4.3 LRT chi-bar QQ — Horvath2020 {trait}", fontsize=11)
    fig.tight_layout()
    p = out_dir / f"calibration_qq_{trait}.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    paths.append(str(p))
    return paths


def main():
    ap = argparse.ArgumentParser(description="M2.4.3 Horvath2020 null-LRT calibration")
    ap.add_argument("--trait", default="plant_height")
    ap.add_argument("--n-sim", type=int, default=1000,
                    help="boundary-LRT replicates per scenario (paper: 2000-5000)")
    ap.add_argument("--n-starts", type=int, default=10)
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--bootstrap-B", type=int, default=0,
                    help="bootstrap replicates per sim; 0 = skip bootstrap calibration")
    ap.add_argument("--bootstrap-n-sim", type=int, default=200,
                    help="replicate count for the (expensive) bootstrap calibration")
    ap.add_argument("--grm-npz", default=None)
    ap.add_argument("--pheno-tsv", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--skip-plots", action="store_true")
    args = ap.parse_args()

    grm_npz = Path(args.grm_npz) if args.grm_npz else ROOT / "results/phase2/m2_1/horvath2020/grm_A_C.npz"
    pheno_tsv = Path(args.pheno_tsv) if args.pheno_tsv else ROOT / "data/processed/rapeseed/horvath/pheno_clean.tsv"
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "results/phase2/m2_4_3/horvath2020"
    out_dir.mkdir(parents=True, exist_ok=True)
    trait = args.trait
    t0 = time.time()

    print(f"=== M2.4.3 Horvath2020 null-LRT calibration — trait={trait} ===")
    y, X, kernels, meta = load_inputs(grm_npz, pheno_tsv, trait)
    n = meta["n_analysis"]
    print(f"  n={n}  var(y)={meta['phenotype_var']:.3f}  n_sim={args.n_sim}  "
          f"n_starts={args.n_starts}  bootstrap_B={args.bootstrap_B}")

    acceptance: list[dict] = []
    observations: list[dict] = []

    def check(name, ok, msg=""):
        acceptance.append({"check": name, "passed": bool(ok), "message": msg})
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {msg}" if msg else ""))

    def observe(name, value, msg=""):
        """Report-only: a p-value behaviour / scientific result, never gated."""
        observations.append({"observation": name, "value": value, "message": msg})
        print(f"  [obs ] {name} = {value}" + (f" — {msg}" if msg else ""))

    results: dict[str, dict] = {}
    type1_all: list[pd.DataFrame] = []
    tail_all: list[pd.DataFrame] = []
    summary_scen: dict[str, dict] = {}

    for label, null_m, alt_m, desc in SCENARIOS:
        print(f"\n[{label}] {desc}")
        scen = scenario_from_reduced_fit(
            label, y, X, kernels, null_model=null_m, alt_model=alt_m,
            n_starts=args.n_starts, random_state=args.seed, description=desc,
        )
        print(f"  reduced-fit σ²_true: { {k: round(v, 4) for k, v in scen.sigma2_true.items()} }")
        regularity = ("residual-boundary (Stram-Lee mixture approximate)"
                      if scen.metadata.get("residual_boundary") else "regular")
        if scen.metadata.get("residual_boundary"):
            print(f"  residual-boundary null: reduced fit σ²_e at bound, "
                  f"boundary={scen.metadata.get('reduced_fit_boundary_components')} "
                  f"residual_pve={scen.metadata.get('reduced_fit_residual_pve'):.3g} "
                  "— empirical type-I valid, analytic mixture approximate")

        # boundary-only calibration (large n_sim)
        res = run_null_lrt_calibration(
            X, kernels, scen,
            n_sim=args.n_sim, seed=args.seed,
            fit_kwargs={"n_starts": args.n_starts}, n_jobs=args.n_jobs,
        )
        print(f"  boundary calibration: success={res.n_sim_success}/{res.n_sim_requested} "
              f"converged_rate={res.converged_rate:.3f}")

        rep = res.replicate_table.copy()
        type1 = res.type1_table.copy()
        tail = res.tail_inflation.copy()
        has_bootstrap = False

        # optional bootstrap calibration (smaller n_sim)
        if args.bootstrap_B > 0:
            print(f"  bootstrap calibration: n_sim={args.bootstrap_n_sim} B={args.bootstrap_B}")
            res_b = run_null_lrt_calibration(
                X, kernels, scen,
                n_sim=args.bootstrap_n_sim, seed=args.seed + 1,
                fit_kwargs={"n_starts": args.n_starts}, n_jobs=args.n_jobs,
                bootstrap_B=args.bootstrap_B,
            )
            t1b = res_b.type1_table
            t1b = t1b[t1b.method == "bootstrap_p"].copy()
            type1 = pd.concat([type1, t1b], ignore_index=True)
            res_b.replicate_table.to_csv(
                out_dir / f"replicate_bootstrap_{label}_{trait}.tsv", sep="\t", index=False)
            has_bootstrap = True

        rep.to_csv(out_dir / f"replicate_{label}_{trait}.tsv", sep="\t", index=False)
        type1_all.append(type1)
        tail_all.append(tail)
        results[label] = {
            "type1": type1, "replicate": rep, "df_added": res.df_added,
            "has_bootstrap": has_bootstrap,
        }

        # per-scenario console + acceptance (machinery only)
        for _, r in type1.iterrows():
            print(f"    {r['method']:11s} α={r['alpha']:.2f}  "
                  f"type-I={r['type1_rate']:.4f} CI=[{r['ci_lo']:.4f},{r['ci_hi']:.4f}]  "
                  f"{r['verdict']}")
        sr = res.n_sim_success / res.n_sim_requested
        check(f"{label}_success_rate>=0.95", sr >= 0.95, f"{sr:.3f}")
        check(f"{label}_converged_rate>=0.95", res.converged_rate >= 0.95,
              f"{res.converged_rate:.3f}")
        expected_df = 2 if label == "S1_global_null" else 1
        check(f"{label}_df_added=={expected_df}", res.df_added == expected_df)
        # p_naive behaviour is a scientific observation, not a machinery gate.
        naive = type1[type1.method == "p_naive"]
        naive_ok = bool((naive["verdict"].isin(["calibrated", "conservative"])).all())
        observe(f"{label}_p_naive_not_anticonservative", naive_ok,
                "χ² for a boundary test is expected conservative (report-only)")
        if res.e_boundary_rate > 0:
            observe(f"{label}_e_boundary_rate", round(res.e_boundary_rate, 3),
                    "fraction of replicates with σ²_e at boundary")

        def _verdict(method, type1=type1):
            row = type1[(type1.method == method) & (type1.alpha == 0.05)]
            return row.iloc[0]["verdict"] if len(row) else None

        summary_scen[label] = {
            "description": desc,
            "sigma2_true": scen.sigma2_true,
            "df_added": res.df_added,
            "boundary_regularity": regularity,
            "reduced_fit_boundary_components": scen.metadata.get("reduced_fit_boundary_components", []),
            "reduced_fit_residual_pve": scen.metadata.get("reduced_fit_residual_pve"),
            "n_sim_success": res.n_sim_success,
            "converged_rate": res.converged_rate,
            "clip_rate": res.clip_rate,
            "e_boundary_rate": res.e_boundary_rate,
            "jitter_used": res.jitter_used,
            "type1": type1.to_dict(orient="records"),
            "tail_inflation": tail.to_dict(orient="records"),
            "p_naive_verdict_at_0.05": _verdict("p_naive"),
            "p_mixture_verdict_at_0.05": _verdict("p_mixture"),
        }

    # combined tables
    type1_combined = pd.concat(type1_all, ignore_index=True)
    tail_combined = pd.concat(tail_all, ignore_index=True)
    type1_combined.to_csv(out_dir / f"type1_summary_{trait}.tsv", sep="\t", index=False)
    tail_combined.to_csv(out_dir / f"tail_inflation_{trait}.tsv", sep="\t", index=False)
    print(f"\nwrote type1_summary_{trait}.tsv / tail_inflation_{trait}.tsv")

    # plots
    plot_paths: list[str] = []
    if not args.skip_plots:
        plot_paths = make_plots(out_dir, trait, results)
        for p in plot_paths:
            print(f"wrote {p}")

    # summary json
    all_passed = all(c["passed"] for c in acceptance)
    summary = {
        "script": "run_m2_4_3_horvath_calibration.py",
        "milestone": "M2.4.3",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "runtime_sec": round(time.time() - t0, 1),
        "seed": args.seed, "n_sim": args.n_sim, "n_starts": args.n_starts,
        "n_jobs": args.n_jobs, "bootstrap_B": args.bootstrap_B,
        **meta,
        "out_dir": str(out_dir),
        "scenarios": summary_scen,
        "acceptance": acceptance,
        "acceptance_all_passed": all_passed,
        "observations": observations,
        "scientific_finding": {
            label: summary_scen[label]["p_mixture_verdict_at_0.05"]
            for label in summary_scen
        },
    }
    summary_path = out_dir / f"m2_4_3_summary_{trait}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=_json_default)
    print(f"wrote {summary_path}")

    # verdict
    n_pass = sum(c["passed"] for c in acceptance)
    print(f"\nacceptance: {n_pass}/{len(acceptance)} checks passed  "
          f"(runtime {summary['runtime_sec']}s)")
    print("scientific finding (p_mixture @ α=0.05): " +
          "  ".join(f"{k}={v}" for k, v in summary["scientific_finding"].items()))
    if all_passed:
        print("M2.4.3 Horvath null-LRT calibration acceptance PASS")
    else:
        failed = [c["check"] for c in acceptance if not c["passed"]]
        print(f"M2.4.3 acceptance FAIL — {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
