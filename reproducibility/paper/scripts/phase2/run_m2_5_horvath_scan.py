#!/usr/bin/env python3
"""Phase 2 M2.5 — Horvath2020 per-SNP association scan (GPU-accelerated).

End-to-end EMMAX/P3D-style scan under the subgenome-stratified LMM:

  1. genome-wide multi-kernel REML on A+C  ->  σ²_A, σ²_C, σ²_e
  2. build the fixed-V projection P
  3. scan every A- and C-subgenome marker  (batched P @ G; CPU or GPU backend)
  4. per-SNP TSV + Manhattan + QQ + λ_GC   <- Phase 2 exit gate (QQ / λ_GC)

v1: in-memory BED scan, single GPU (or CPU). Wheat-scale streaming + 2-GPU +
LOCO are M2.5-v2. proximal contamination is not corrected (standard EMMAX).

Run on the GPU env for the GPU backend:
  ~/miniconda3/envs/polygwas-gpu/bin/python run_m2_5_horvath_scan.py
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

from homoeogwas import fit_multi_reml, normalize_kernel  # noqa: E402
from homoeogwas.io import load_bed_hardcall  # noqa: E402
from homoeogwas.scan import (  # noqa: E402
    build_scan_context,
    lambda_gc,
    scan_snps,
)

SUBGENOMES = ("A", "C")


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


def _concat_results(results: list) -> dict:
    """Concatenate the per-subgenome ScanResult arrays into plain arrays."""
    keys = ("snp_id", "chrom", "pos", "beta", "se", "chi2", "p",
            "n_obs", "maf", "call_rate")
    return {k: np.concatenate([getattr(r, k) for r in results]) for k in keys}


def make_plots(out_dir: Path, trait: str, cols: dict, subg: np.ndarray):
    paths = []
    p = np.clip(cols["p"].astype(np.float64), 1e-300, 1.0)
    logp = -np.log10(p)

    # Manhattan — markers laid out per chromosome, coloured by subgenome
    chrom = cols["chrom"].astype(str)
    order = np.lexsort((cols["pos"], chrom))
    chrom_o, logp_o, sub_o = chrom[order], logp[order], subg[order]
    uchrom = list(dict.fromkeys(chrom_o))
    xpos = np.zeros(len(chrom_o))
    offset = 0.0
    ticks = []
    tick_lab = []
    fig, ax = plt.subplots(figsize=(11, 4))
    for ci in uchrom:
        msk = chrom_o == ci
        k = int(msk.sum())
        xpos[msk] = offset + np.arange(k)
        ticks.append(offset + k / 2)
        tick_lab.append(ci)
        color = "#1f77b4" if sub_o[msk][0] == "A" else "#ff7f0e"
        ax.scatter(xpos[msk], logp_o[msk], s=4, c=color)
        offset += k + max(1, k // 50)
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_lab, rotation=90, fontsize=6)
    ax.set_ylabel("-log10(p)")
    ax.set_title(f"M2.5 Manhattan — Horvath2020 {trait} (A=blue, C=orange)")
    ax.axhline(-np.log10(5e-8), ls="--", lw=0.8, color="red")
    fig.tight_layout()
    mp = out_dir / f"manhattan_{trait}.png"
    fig.savefig(mp, dpi=130, bbox_inches="tight")
    plt.close(fig)
    paths.append(str(mp))

    # QQ plot
    obs = np.sort(logp)[::-1]
    exp = -np.log10((np.arange(1, len(obs) + 1) - 0.5) / len(obs))
    fig, ax = plt.subplots(figsize=(4.6, 4.5))
    lim = max(float(obs.max()), float(exp.max())) * 1.05
    ax.plot([0, lim], [0, lim], ls="--", color="k", lw=1)
    ax.scatter(exp, obs, s=5, color="#1f77b4")
    ax.set_xlabel("expected -log10(p)")
    ax.set_ylabel("observed -log10(p)")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_title(f"M2.5 QQ — Horvath2020 {trait}  λ_GC={lambda_gc(cols['chi2']):.3f}")
    fig.tight_layout()
    qp = out_dir / f"qq_{trait}.png"
    fig.savefig(qp, dpi=130, bbox_inches="tight")
    plt.close(fig)
    paths.append(str(qp))
    return paths


def main():
    ap = argparse.ArgumentParser(description="M2.5 Horvath2020 per-SNP LMM scan")
    ap.add_argument("--trait", default="plant_height")
    ap.add_argument("--backend", default="auto", choices=["auto", "cpu", "gpu"])
    ap.add_argument("--n-starts", type=int, default=10)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--maf-min", type=float, default=0.01)
    ap.add_argument("--call-rate-min", type=float, default=0.9)
    ap.add_argument("--batch-size", type=int, default=20000)
    ap.add_argument("--grm-npz", default=None)
    ap.add_argument("--pheno-tsv", default=None)
    ap.add_argument("--geno-root", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--skip-plots", action="store_true")
    args = ap.parse_args()

    grm_npz = Path(args.grm_npz) if args.grm_npz else ROOT / "results/phase2/m2_1/horvath2020/grm_A_C.npz"
    pheno_tsv = Path(args.pheno_tsv) if args.pheno_tsv else ROOT / "data/processed/rapeseed/horvath/pheno_clean.tsv"
    geno_root = Path(args.geno_root) if args.geno_root else ROOT / "data/processed/rapeseed/horvath"
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "results/phase2/m2_5/horvath2020"
    out_dir.mkdir(parents=True, exist_ok=True)
    trait = args.trait
    t0 = time.time()

    print(f"=== M2.5 Horvath2020 per-SNP LMM scan — trait={trait} ===")
    for f in (grm_npz, pheno_tsv):
        if not f.exists():
            sys.exit(f"ERR: missing input {f}")

    # ---- inputs: GRM + phenotype, inner-join -------------------------
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
        sys.exit(f"ERR: too few samples {n}")
    idx = {s: i for i, s in enumerate(samples)}
    sel = np.array([idx[s] for s in overlap], dtype=np.int64)
    K_A = normalize_kernel(G_A[np.ix_(sel, sel)], mode="trace")
    K_C = normalize_kernel(G_C[np.ix_(sel, sel)], mode="trace")
    y = pheno.loc[overlap, trait].astype(np.float64).to_numpy()
    X = np.ones((n, 1), dtype=np.float64)
    sample_ids = np.array(overlap, dtype=object)
    print(f"  n={n}  var(y)={np.var(y, ddof=1):.3f}")

    acceptance: list[dict] = []

    def check(name, ok, msg=""):
        acceptance.append({"check": name, "passed": bool(ok), "message": msg})
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {msg}" if msg else ""))

    # ---- genome-wide REML -> variance components --------------------
    print("\n[1] genome-wide multi-kernel REML (A+C)")
    reml = fit_multi_reml(y, X, {"A": K_A, "C": K_C},
                          n_starts=args.n_starts, random_state=args.seed)
    print(f"  σ²: { {k: round(v, 4) for k, v in reml.sigma2.items()} }")
    print(f"  PVE: { {k: round(v, 4) for k, v in reml.pve.items()} }")
    check("reml_converged", bool(reml.optimizer_status))
    check("reml_pve_sums_to_1", abs(sum(reml.pve.values()) - 1.0) < 1e-9)

    # ---- fixed-V projection -----------------------------------------
    print("\n[2] build_scan_context (fixed-V projection P)")
    ctx = build_scan_context(y, X, {"A": K_A, "C": K_C}, reml.sigma2,
                             sample_ids=sample_ids)
    print(f"  P {ctx.P.shape}  jitter_used={ctx.jitter_used:.2g}")

    # ---- per-SNP scan over both subgenomes --------------------------
    print(f"\n[3] per-SNP scan  backend={args.backend}  "
          f"maf>={args.maf_min}  call_rate>={args.call_rate_min}")
    results, subg_tags, backend_used = [], [], None
    cpu_check_chi2 = {}
    for sg in SUBGENOMES:
        bed_prefix = geno_root / sg / "all"
        if not bed_prefix.with_suffix(".bed").exists():
            sys.exit(f"ERR: genotype BED missing: {bed_prefix}.bed")
        geno = load_bed_hardcall(bed_prefix)
        res = scan_snps(ctx, geno, backend=args.backend,
                        batch_size=args.batch_size,
                        maf_min=args.maf_min, call_rate_min=args.call_rate_min)
        backend_used = res.backend_used
        print(f"  {sg}: {res.n_input} markers -> {res.n_kept} kept "
              f"(filtered { {k: v for k, v in res.filter_counts.items() if v} })")
        results.append(res)
        subg_tags.append(np.full(res.n_kept, sg, dtype=object))
        # GPU correctness gate: re-scan on CPU and compare
        if backend_used == "gpu":
            res_cpu = scan_snps(ctx, geno, backend="cpu",
                                batch_size=args.batch_size,
                                maf_min=args.maf_min, call_rate_min=args.call_rate_min)
            cpu_check_chi2[sg] = float(np.max(np.abs(
                np.sort(res.chi2) - np.sort(res_cpu.chi2))))

    cols = _concat_results(results)
    subg = np.concatenate(subg_tags)
    n_kept = len(cols["p"])
    print(f"  total kept markers: {n_kept}")

    check("scan_has_markers", n_kept > 0, f"n_kept={n_kept}")
    check("scan_p_in_unit_interval",
          bool(np.all((cols["p"] >= 0) & (cols["p"] <= 1))))
    check("scan_chi2_nonneg", bool(np.all(cols["chi2"] >= 0)))
    if backend_used == "gpu":
        max_gpu_cpu = max(cpu_check_chi2.values()) if cpu_check_chi2 else 0.0
        check("gpu_matches_cpu", max_gpu_cpu < 1e-6,
              f"max|Δχ²|={max_gpu_cpu:.3g}")
    else:
        print("  [note] GPU backend not used; gpu_matches_cpu check skipped")

    # ---- genomic inflation ------------------------------------------
    lam_all = lambda_gc(cols["chi2"])
    lam_sub = {sg: lambda_gc(cols["chi2"][subg == sg]) for sg in SUBGENOMES}
    print(f"\n[4] λ_GC overall = {lam_all:.4f}  "
          + "  ".join(f"λ_{sg}={lam_sub[sg]:.4f}" for sg in SUBGENOMES))
    check("lambda_gc_finite", np.isfinite(lam_all), f"λ_GC={lam_all:.4f}")

    # ---- outputs -----------------------------------------------------
    df = pd.DataFrame({
        "snp_id": cols["snp_id"], "subgenome": subg, "chrom": cols["chrom"],
        "pos": cols["pos"], "beta": cols["beta"], "se": cols["se"],
        "chi2": cols["chi2"], "p": cols["p"], "n_obs": cols["n_obs"],
        "maf": cols["maf"], "call_rate": cols["call_rate"],
    }).sort_values(["subgenome", "chrom", "pos"]).reset_index(drop=True)
    scan_tsv = out_dir / f"scan_{trait}.tsv"
    df.to_csv(scan_tsv, sep="\t", index=False)
    print(f"  wrote {scan_tsv}  ({len(df)} markers)")

    top = df.nsmallest(10, "p")[["snp_id", "subgenome", "chrom", "pos", "beta", "p"]]
    print("  top 10 markers by p:")
    for r in top.itertuples():
        print(f"    {r.snp_id:18s} {r.subgenome} {r.chrom}:{r.pos}  "
              f"beta={r.beta:+.3f}  p={r.p:.3e}")

    plot_paths = []
    if not args.skip_plots:
        plot_paths = make_plots(out_dir, trait, cols, subg)
        for pth in plot_paths:
            print(f"  wrote {pth}")

    all_passed = all(c["passed"] for c in acceptance)
    summary = {
        "script": "run_m2_5_horvath_scan.py", "milestone": "M2.5",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "runtime_sec": round(time.time() - t0, 1),
        "trait": trait, "n": n, "seed": args.seed,
        "backend_requested": args.backend, "backend_used": backend_used,
        "maf_min": args.maf_min, "call_rate_min": args.call_rate_min,
        "reml_sigma2": {k: float(v) for k, v in reml.sigma2.items()},
        "reml_pve": {k: float(v) for k, v in reml.pve.items()},
        "n_markers_input": int(sum(r.n_input for r in results)),
        "n_markers_kept": n_kept,
        "filter_counts": {sg: results[i].filter_counts
                          for i, sg in enumerate(SUBGENOMES)},
        "lambda_gc": lam_all,
        "lambda_gc_by_subgenome": lam_sub,
        "gpu_vs_cpu_max_abs_chi2_diff": cpu_check_chi2 or None,
        "top_markers": top.to_dict(orient="records"),
        "acceptance": acceptance,
        "acceptance_all_passed": all_passed,
        "outputs": {"scan_tsv": str(scan_tsv), "plots": plot_paths},
    }
    summary_path = out_dir / f"m2_5_summary_{trait}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=_json_default)
    print(f"\nwrote {summary_path}")

    n_pass = sum(c["passed"] for c in acceptance)
    print(f"\nacceptance: {n_pass}/{len(acceptance)} checks passed  "
          f"(runtime {summary['runtime_sec']}s, backend={backend_used})")
    if all_passed:
        print("M2.5 Horvath per-SNP scan acceptance PASS")
    else:
        failed = [c["check"] for c in acceptance if not c["passed"]]
        print(f"M2.5 acceptance FAIL — {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
