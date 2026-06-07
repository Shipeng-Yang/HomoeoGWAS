#!/usr/bin/env python3
"""Phase 2 M2.4.5 — wheat Watkins A+B+D pilot (the scale anchor).

End-to-end HomoeoGWAS at hexaploid scale: 1,051-sample Watkins panel, three
subgenomes (A 34.2M / B 43.4M / D 8.9M post-QC SNPs). Produces the Phase 2
exit deliverable — a runtime / peak-RAM / peak-VRAM table — plus the
subgenome-stratified variance decomposition.

Pipeline:
  preflight  closeout OK gate + inputs present
  join       A/B/D psam ∩ pheno(days_to_emerg) non-missing  -> analysis samples
  prune      plink2 --indep-pairwise per subgenome           -> pruned BED
  grm        compute_grm on pruned BED                       -> G_A / G_B / G_D
  kernels    trace-normalized A/B/D + K_hom_ABD = G_A⊙G_B⊙G_D
  reml       J=3 additive (8 models) + J=4 Hadamard (16 models) + boundary LRT
  canary     GPU per-SNP scan on a pruned BED  (fills peak VRAM)
  profile    stage_profile.tsv  (runtime / peak RAM / peak VRAM)

Scope: M2.4.5 is the scale/runtime pilot on a *pruned* GRM marker set; the
full 34M-SNP streaming per-SNP scan, 2-GPU data-parallel and LOCO are M2.5-v2.
Known-QTL recovery is deferred to that full scan.

Run in the GPU env (has the full data stack + torch):
  ~/miniconda3/envs/polygwas-gpu/bin/python run_m2_4_5_wheat_pilot.py
"""
from __future__ import annotations

import argparse
import json
import resource
import subprocess
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
from homoeogwas.diagnostics import boundary_lrt_table, compare_nested_reml  # noqa: E402
from homoeogwas.grm import compute_grm  # noqa: E402
from homoeogwas.io import load_bed_hardcall  # noqa: E402
from homoeogwas.scan import build_scan_context, lambda_gc, scan_snps  # noqa: E402

PLINK2 = str(Path.home() / ".local/share/mamba/envs/polygwas-cpu/bin/plink2")
SUBGENOMES = ("A", "B", "D")
J3_FULL = "A+B+D+e"
J4_FULL = "A+B+D+hom+e"
HEADLINE = ("A+B+D+e", "A+B+D+hom+e")     # does K_hom add over additive A+B+D?


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


def _peak_rss_gb() -> float:
    """Cumulative peak RSS of this process (ru_maxrss is KB on Linux)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 ** 2)


class Stage:
    """Time a pipeline stage and record runtime + cumulative peak RSS."""
    def __init__(self, rows: list, name: str, **meta):
        self.rows, self.name, self.meta = rows, name, meta

    def __enter__(self):
        self.t0 = time.time()
        print(f"\n[{self.name}] ...")
        return self

    def __exit__(self, *exc):
        rt = time.time() - self.t0
        row = {"stage": self.name, "runtime_sec": round(rt, 2),
               "process_peak_rss_gb": round(_peak_rss_gb(), 3), **self.meta}
        self.rows.append(row)
        print(f"[{self.name}] done in {rt:.1f}s  (process peak RSS {row['process_peak_rss_gb']:.2f} GB)")
        return False


def run_plink(args: list[str]) -> float | None:
    """Run plink2 (wrapped in /usr/bin/time -v for max RSS). Returns RSS in GB."""
    use_time = Path("/usr/bin/time").exists()
    cmd = (["/usr/bin/time", "-v"] if use_time else []) + [PLINK2] + args
    r = subprocess.run(cmd, capture_output=True, text=True)
    rss_gb = None
    if use_time:
        for line in r.stderr.splitlines():
            if "Maximum resident set size" in line:
                try:
                    rss_gb = int(line.split(":")[-1].strip()) / (1024.0 ** 2)
                except ValueError:
                    pass
    if r.returncode != 0:
        raise RuntimeError(f"plink2 failed (rc={r.returncode}):\n{r.stderr[-2000:]}")
    return rss_gb


def main():
    ap = argparse.ArgumentParser(description="M2.4.5 wheat Watkins A+B+D pilot")
    ap.add_argument("--trait", default="days_to_emerg")
    ap.add_argument("--n-starts", type=int, default=10)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--prune-window", default="1000kb")
    ap.add_argument("--prune-r2", type=float, default=0.2)
    ap.add_argument("--maf-min", type=float, default=0.01)
    ap.add_argument("--plink-threads", type=int, default=16)
    ap.add_argument("--canary-subgenome", default="D",
                    help="which pruned BED the GPU canary scan runs on")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--skip-plots", action="store_true")
    args = ap.parse_args()

    wheat_dir = ROOT / "data/processed/wheat"
    pheno_tsv = wheat_dir / "pheno_clean.tsv"
    qc_summary = ROOT / "results/phase1/wheat_watkins/closeout/qc_summary.tsv"
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "results/phase2/m2_4_5/wheat_watkins"
    prune_dir = out_dir / "prune"
    pruned_bed_dir = out_dir / "pruned"
    for d in (out_dir, prune_dir, pruned_bed_dir):
        d.mkdir(parents=True, exist_ok=True)
    trait = args.trait
    t_start = time.time()
    profile: list[dict] = []
    acceptance: list[dict] = []
    observations: list[dict] = []

    def check(name, ok, msg=""):
        acceptance.append({"check": name, "passed": bool(ok), "message": msg})
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {msg}" if msg else ""))

    def observe(name, value, msg=""):
        observations.append({"observation": name, "value": value, "message": msg})
        print(f"  [obs ] {name} = {value}" + (f" — {msg}" if msg else ""))

    print(f"=== M2.4.5 wheat Watkins A+B+D pilot — trait={trait} ===")

    # ---- preflight ---------------------------------------------------
    with Stage(profile, "preflight"):
        if not qc_summary.exists():
            sys.exit(f"ERR: closeout qc_summary missing: {qc_summary}")
        qc = pd.read_csv(qc_summary, sep="\t")
        # Success format = per-entity table; require *valid* wheat_A/B/D rows
        # (positive n_samples & n_snps, no "fail" note) so a FAILED or partial
        # closeout cannot slip through.
        qc_rows = ({str(r.entity): r for r in qc.itertuples()}
                   if "entity" in qc.columns else {})
        snp_counts: dict[str, int] = {}
        sample_counts: dict[str, int] = {}
        closeout_ok = True
        for sg in SUBGENOMES:
            r = qc_rows.get(f"wheat_{sg}")
            row_ok = False
            if r is not None:
                try:                       # n_snps/n_samples are object/float (NA rows)
                    ns = int(float(r.n_samples))
                    nv = int(float(r.n_snps))
                    note = str(getattr(r, "notes", "")).lower()
                    row_ok = ns > 0 and nv > 0 and "fail" not in note
                    if row_ok:
                        sample_counts[sg], snp_counts[sg] = ns, nv
                except (ValueError, TypeError):
                    row_ok = False
            closeout_ok = closeout_ok and row_ok
        check("closeout_status_ok", closeout_ok,
              f"wheat_A/B/D rows valid; post-QC SNPs={snp_counts}" if closeout_ok
              else "qc_summary missing valid wheat_A/B/D rows (FAILED/partial closeout)")
        if not closeout_ok:
            sys.exit("ERR: wheat closeout not complete/valid — aborting M2.4.5")
        pgen_ok = all((wheat_dir / sg / "all.pgen").exists() for sg in SUBGENOMES)
        check("abd_pgen_present", pgen_ok)
        if not pgen_ok:
            sys.exit("ERR: A/B/D pgen not all present")
        if not pheno_tsv.exists():
            sys.exit(f"ERR: pheno missing: {pheno_tsv}")
        if not Path(PLINK2).exists():
            sys.exit(f"ERR: plink2 not found: {PLINK2}")
        print(f"  closeout OK; post-QC SNPs: {snp_counts}")

    # ---- sample join -------------------------------------------------
    with Stage(profile, "sample_join"):
        psam_ids = {}
        for sg in SUBGENOMES:
            ps = pd.read_csv(wheat_dir / sg / "all.psam", sep="\t")
            iid_col = ps.columns[0]          # '#IID'
            psam_ids[sg] = list(ps[iid_col].astype(str))
        common = set(psam_ids["A"]) & set(psam_ids["B"]) & set(psam_ids["D"])
        same_order = (psam_ids["A"] == psam_ids["B"] == psam_ids["D"])
        check("abd_psam_same_id_set",
              len(common) == len(psam_ids["A"]) == len(psam_ids["B"]) == len(psam_ids["D"]),
              f"|A|={len(psam_ids['A'])} |B|={len(psam_ids['B'])} |D|={len(psam_ids['D'])} "
              f"common={len(common)}")
        pheno = pd.read_csv(pheno_tsv, sep="\t").set_index("sample")
        if trait not in pheno.columns:
            sys.exit(f"ERR: trait {trait!r} not in pheno; have {list(pheno.columns)}")
        # analysis = psam-common ∩ pheno non-missing trait, in psam(A) order
        pheno_ok = {s for s in pheno.index if not pd.isna(pheno.loc[s, trait])}
        analysis = [s for s in psam_ids["A"] if s in common and s in pheno_ok]
        n = len(analysis)
        y = pheno.loc[analysis, trait].astype(np.float64).to_numpy()
        X = np.ones((n, 1), dtype=np.float64)
        keep_path = out_dir / "analysis_samples.tsv"
        pd.DataFrame({"#IID": analysis}).to_csv(keep_path, sep="\t", index=False)
        print(f"  n_analysis={n}  var(y)={np.var(y, ddof=1):.3f}  "
              f"(psam_common={len(common)}, psam_same_order={same_order})")
        check("n_analysis>=750", n >= 750, f"n={n}")

    # ---- LD prune + pruned BED per subgenome ------------------------
    pruned: dict[str, dict] = {}
    for sg in SUBGENOMES:
        with Stage(profile, f"prune_{sg}", subgenome=sg) as st:
            pfile = str(wheat_dir / sg / "all")
            prune_out = str(prune_dir / sg)
            rss1 = run_plink([
                "--pfile", pfile, "--keep", str(keep_path),
                "--indep-pairwise", args.prune_window, "1", str(args.prune_r2),
                "--threads", str(args.plink_threads), "--out", prune_out,
            ])
            n_pruned = sum(1 for _ in open(f"{prune_out}.prune.in"))
            bed_out = str(pruned_bed_dir / sg)
            rss2 = run_plink([
                "--pfile", pfile, "--keep", str(keep_path),
                "--extract", f"{prune_out}.prune.in", "--make-bed",
                "--threads", str(args.plink_threads), "--out", bed_out,
            ])
            pruned[sg] = {"bed": bed_out, "n_pruned": n_pruned}
            st.meta["n_pruned"] = n_pruned
            st.meta["plink_peak_rss_gb"] = round(max(filter(None, [rss1, rss2, 0])), 3)
            print(f"  {sg}: {n_pruned} pruned SNPs")
    for sg in SUBGENOMES:
        check(f"prune_{sg}_in_range", 50_000 <= pruned[sg]["n_pruned"] <= 2_000_000,
              f"{pruned[sg]['n_pruned']} pruned")

    # ---- per-subgenome GRM from pruned BED --------------------------
    raw_grm: dict[str, np.ndarray] = {}
    with Stage(profile, "compute_grm_ABD"):
        for sg in SUBGENOMES:
            chunk = load_bed_hardcall(pruned[sg]["bed"])
            # reorder rows to the canonical analysis-sample order
            pos = {s: i for i, s in enumerate(np.asarray(chunk.samples, dtype=str))}
            if set(pos) != set(analysis):
                sys.exit(f"ERR: pruned BED {sg} samples != analysis set")
            order = np.array([pos[s] for s in analysis], dtype=np.int64)
            from homoeogwas.io import GenoChunk
            sub = GenoChunk(
                samples=np.asarray(analysis, dtype=object),
                variant_ids=chunk.variant_ids, chrom=chunk.chrom, pos=chunk.pos,
                dosage=chunk.dosage[order, :],
            )
            G, info = compute_grm(sub, maf_min=args.maf_min)
            raw_grm[sg] = G
            print(f"  G_{sg}: n={info['n_samples']} m_used={info['n_variants_used']} "
                  f"trace={info['trace']:.1f}")
    for sg in SUBGENOMES:
        G = raw_grm[sg]
        check(f"grm_{sg}_finite_symmetric",
              bool(np.all(np.isfinite(G)) and np.allclose(G, G.T, atol=1e-8)))

    # ---- kernels: canonical (trace-norm) A/B/D + K_hom_ABD ----------
    with Stage(profile, "build_kernels"):
        raw_kernels = dict(raw_grm)
        raw_kernels["hom"] = hadamard_kernel({sg: raw_grm[sg] for sg in SUBGENOMES})
        canonical = {k: normalize_kernel(K, mode="trace") for k, K in raw_kernels.items()}
        hom_min_eig = float(np.linalg.eigvalsh(canonical["hom"]).min())
        for k in ("A", "B", "D", "hom"):
            tr = float(np.trace(canonical[k]))
            check(f"kernel_{k}_trace≈n", abs(tr - n) < 1e-6 * n, f"trace={tr:.3f}")
        check("K_hom_psd", hom_min_eig > -1e-6, f"min_eig={hom_min_eig:.3g}")

    # ---- REML: J=3 additive + J=4 Hadamard --------------------------
    with Stage(profile, "reml_j3_additive"):
        kern_j3 = {sg: canonical[sg] for sg in SUBGENOMES}
        cmp3 = compare_nested_reml(y, X, kern_j3, strategy="exhaustive",
                                   fit_kwargs={"n_starts": args.n_starts,
                                               "random_state": args.seed})
        lrt3 = boundary_lrt_table(cmp3, "default")
        full3 = cmp3.fits[J3_FULL]
        pve3 = {k: float(full3.pve[k]) for k in ("A", "B", "D", "e")}
        print(f"  J=3 additive PVE: { {k: round(v, 4) for k, v in pve3.items()} }")
        check("j3_8_models", len(cmp3.likelihood_table) == 8)
        check("j3_full_pve_sum1", abs(sum(pve3.values()) - 1.0) < 1e-9)
        check("j3_loglik_finite", bool(np.all(np.isfinite(cmp3.likelihood_table["log_lik"]))))
        check("j3_all_converged", bool(cmp3.likelihood_table["optimizer_status"].all()),
              "all 8 nested fits converged")
        check("j3_lrt_converged",
              bool(lrt3["both_converged"].all()) and not bool(lrt3["clipped"].any()),
              "boundary LRT contrasts converged, none clipped")

    with Stage(profile, "reml_j4_hadamard"):
        cmp4 = compare_nested_reml(y, X, canonical, strategy="exhaustive",
                                   fit_kwargs={"n_starts": args.n_starts,
                                               "random_state": args.seed})
        lrt4 = boundary_lrt_table(cmp4, "default")
        full4 = cmp4.fits[J4_FULL]
        pve4 = {k: float(full4.pve[k]) for k in ("A", "B", "D", "hom", "e")}
        cond4 = float(full4.kernel_design_cond)
        print(f"  J=4 PVE: { {k: round(v, 4) for k, v in pve4.items()} }  "
              f"design_cond={cond4:.2e}")
        check("j4_16_models", len(cmp4.likelihood_table) == 16)
        check("j4_all_converged", bool(cmp4.likelihood_table["optimizer_status"].all()),
              "all 16 nested fits converged")
        check("j4_lrt_converged",
              bool(lrt4["both_converged"].all()) and not bool(lrt4["clipped"].any()),
              "boundary LRT contrasts converged, none clipped")
        hl = lrt4[(lrt4.null_model == HEADLINE[0]) & (lrt4.alt_model == HEADLINE[1])]
        check("j4_headline_present", len(hl) == 1)
        headline_T = float(hl.iloc[0]["lrt"]) if len(hl) else float("nan")
        headline_p = float(hl.iloc[0]["p_boundary"]) if len(hl) else float("nan")
        khom_verdict = ("Hadamard increment detected" if headline_p < 0.05
                        else "no detectable Hadamard increment over additive A+B+D")
        observe("K_hom_inclusion_verdict", khom_verdict,
                f"headline T={headline_T:.3g} p_boundary={headline_p:.3g}")
        observe("j4_design_cond", f"{cond4:.3e}")

    lrt3.to_csv(out_dir / f"lrt_j3_{trait}.tsv", sep="\t", index=False)
    lrt4.to_csv(out_dir / f"lrt_j4_{trait}.tsv", sep="\t", index=False)
    cmp3.likelihood_table.to_csv(out_dir / f"nested_j3_{trait}.tsv", sep="\t", index=False)
    cmp4.likelihood_table.to_csv(out_dir / f"nested_j4_{trait}.tsv", sep="\t", index=False)

    # ---- GPU canary: per-SNP scan on a pruned BED -------------------
    canary = {}
    with Stage(profile, "gpu_canary_scan", subgenome=args.canary_subgenome) as st:
        try:
            import torch
            torch.cuda.reset_peak_memory_stats()
        except Exception:  # noqa: BLE001
            torch = None
        ctx = build_scan_context(y, X, kern_j3, full3.sigma2,
                                 sample_ids=np.asarray(analysis, dtype=object))
        geno = load_bed_hardcall(pruned[args.canary_subgenome]["bed"])
        scan_res = scan_snps(ctx, geno, backend="auto",
                             maf_min=args.maf_min, call_rate_min=0.9)
        vram_gb = None
        if torch is not None and torch.cuda.is_available():
            vram_gb = torch.cuda.max_memory_allocated() / (1024.0 ** 3)
        st.meta["peak_vram_gb"] = round(vram_gb, 4) if vram_gb is not None else None
        st.meta["backend"] = scan_res.backend_used
        st.meta["n_markers"] = scan_res.n_kept
        canary = {"backend": scan_res.backend_used, "n_markers": scan_res.n_kept,
                  "lambda_gc": lambda_gc(scan_res.chi2), "peak_vram_gb": vram_gb,
                  "subgenome": args.canary_subgenome}
        print(f"  canary: {scan_res.n_kept} markers on {args.canary_subgenome}, "
              f"backend={scan_res.backend_used}, λ_GC={canary['lambda_gc']:.4f}, "
              f"peak VRAM={vram_gb if vram_gb is None else round(vram_gb,3)} GB")
    check("gpu_canary_backend_gpu", canary.get("backend") == "gpu",
          f"backend={canary.get('backend')}")
    check("gpu_canary_vram_measured", canary.get("peak_vram_gb") is not None)

    # ---- profiling table + summary ----------------------------------
    total_rt = time.time() - t_start
    profile.append({"stage": "TOTAL", "runtime_sec": round(total_rt, 2),
                    "process_peak_rss_gb": round(_peak_rss_gb(), 3)})
    prof_df = pd.DataFrame(profile)
    prof_path = out_dir / f"stage_profile_{trait}.tsv"
    prof_df.to_csv(prof_path, sep="\t", index=False)
    print(f"\nwrote {prof_path}")
    print(prof_df.to_string(index=False))

    if not args.skip_plots:
        fig, ax = plt.subplots(figsize=(5.5, 4))
        keys = ["A", "B", "D", "hom", "e"]
        vals = [pve4[k] for k in keys]
        ax.bar(keys, vals, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#999999"])
        for i, v in enumerate(vals):
            ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("PVE")
        ax.set_title(f"M2.4.5 wheat A+B+D+hom PVE — {trait} (n={n})")
        fig.tight_layout()
        fig.savefig(out_dir / f"pve_bar_{trait}.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out_dir / f'pve_bar_{trait}.png'}")

    all_passed = all(c["passed"] for c in acceptance)
    summary = {
        "script": "run_m2_4_5_wheat_pilot.py", "milestone": "M2.4.5",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "trait": trait, "n_analysis": n, "seed": args.seed,
        "total_runtime_sec": round(total_rt, 1),
        "peak_rss_gb": round(_peak_rss_gb(), 3),
        "peak_vram_gb": canary.get("peak_vram_gb"),
        "post_qc_snp_counts": snp_counts,
        "n_pruned": {sg: pruned[sg]["n_pruned"] for sg in SUBGENOMES},
        "reml_j3": {"pve": pve3, "sigma2": {k: float(v) for k, v in full3.sigma2.items()},
                    "log_lik": float(full3.log_lik)},
        "reml_j4": {"pve": pve4, "sigma2": {k: float(v) for k, v in full4.sigma2.items()},
                    "log_lik": float(full4.log_lik), "kernel_design_cond": cond4,
                    "kernel_corr": full4.kernel_corr},
        "headline": {"contrast": f"{HEADLINE[0]} -> {HEADLINE[1]}",
                     "lrt": headline_T, "p_boundary": headline_p},
        "gpu_canary": canary,
        "stage_profile": profile,
        "acceptance": acceptance, "acceptance_all_passed": all_passed,
        "observations": observations,
        "scope_note": "pruned-GRM scale pilot; full 34M-SNP scan / 2-GPU / LOCO = M2.5-v2",
    }
    summary_path = out_dir / f"m2_4_5_summary_{trait}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=_json_default)
    print(f"wrote {summary_path}")

    n_pass = sum(c["passed"] for c in acceptance)
    print(f"\nacceptance: {n_pass}/{len(acceptance)} checks passed  "
          f"(total runtime {total_rt:.0f}s, peak RSS {_peak_rss_gb():.1f} GB, "
          f"peak VRAM {canary.get('peak_vram_gb')})")
    if all_passed:
        print("✅ M2.4.5 wheat Watkins pilot acceptance PASS")
    else:
        failed = [c["check"] for c in acceptance if not c["passed"]]
        print(f"❌ M2.4.5 acceptance FAIL — {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
