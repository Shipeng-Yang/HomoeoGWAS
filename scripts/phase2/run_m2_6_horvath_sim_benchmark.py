#!/usr/bin/env python3
"""Phase 2 M2.6 — Horvath2020 simulation benchmark (Phase 2 EXIT hard gate).

Per-subgenome causal-SNP injection; power-vs-FDR curves for HomoeoGWAS vs
the GEMMA / regenie main-bucket baselines (+ a pooled-GRM EMMAX ablation).

Scientific claim under test
---------------------------
When the true allopolyploid background has unequal subgenome polygenic
variance (σ²_A ≠ σ²_C), the subgenome-stratified multi-kernel LMM has higher
discovery power at matched empirical FDR than single pooled-GRM methods.

Arms (see homoeogwas.sim.default_arms):
  S1_C_dominant / S2_A_dominant  — stratified, exit-gate evidence
  S3_pooled / S4_balanced        — honesty arms (HomoeoGWAS must not lose)
  S5_null                        — calibration (no causal SNPs)
  S6_hadamard                    — secondary homoeolog-interaction stress arm

Subcommands (also chained by `run`):
  prepare        merged A+C BED + GEMMA kinship + kernels + marker metadata
  simulate       causal sets + simulated phenotypes for every arm/replicate
  run-homoeo     HomoeoGWAS multi-kernel scan + pooled-GRM EMMAX ablation
  run-baselines  GEMMA + regenie per replicate (parallel)
  summarize      power/FDR curves, paired tests, figures, summary.json

Everything runs under the CPU env (no GPU needed at Horvath scale):
  ~/.local/share/mamba/envs/polygwas-cpu/bin/python run_m2_6_horvath_sim_benchmark.py run
"""
from __future__ import annotations
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# Force BLAS/OpenMP to a single thread BEFORE importing numpy — this both
# keeps the HomoeoGWAS runtime comparison fair vs the single-threaded
# baselines (regenie --threads 1, GEMMA) and prevents thread oversubscription
# when the replicate ProcessPool fans out. Hard-set (not setdefault) so an
# inherited OMP_NUM_THREADS>1 cannot silently un-level the runtime gate.
# Must precede `import numpy`.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from homoeogwas import fit_multi_reml, hadamard_kernel, normalize_kernel  # noqa: E402
from homoeogwas.io import load_bed_hardcall  # noqa: E402
from homoeogwas.scan import build_scan_context, scan_snps  # noqa: E402
from homoeogwas import sim  # noqa: E402

GEMMA = os.path.expanduser("~/.local/share/mamba/envs/polygwas-cpu/bin/gemma")
REGENIE = os.path.expanduser("~/.local/share/mamba/envs/polygwas-cpu/bin/regenie")
SUBGENOMES = ("A", "C")

# replicate counts by mode
MODE_REPS = {
    "smoke": {"S1_C_dominant": 3, "S2_A_dominant": 3, "S3_pooled": 3,
              "S4_balanced": 3, "S5_null": 3, "S6_hadamard": 3},
    "production": {"S1_C_dominant": 100, "S2_A_dominant": 100, "S3_pooled": 100,
                   "S4_balanced": 50, "S5_null": 50, "S6_hadamard": 50},
}
# p-value grid for the power-FDR curve
CURVE_THRESHOLDS = np.logspace(-2.0, -8.0, 49)
EXIT_GATE_ARMS = ("S1_C_dominant", "S2_A_dominant")   # stratified evidence


# =====================================================================
# small helpers
# =====================================================================

def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(f"not JSON serializable: {type(o)}")


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _nanmean(a, axis=None):
    """np.nanmean that returns NaN (no warning) when a slice is all-NaN.

    Null-arm power columns are all-NaN by construction (power undefined with
    no causal SNPs); silence the harmless 'Mean of empty slice' warning.
    """
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(a, axis=axis)


def _arm_table(arm: sim.SimArm) -> dict:
    return {"name": arm.name, "kind": arm.kind, "h_qtl_A": arm.h_qtl_A,
            "h_qtl_C": arm.h_qtl_C, "h_poly_A": arm.h_poly_A,
            "h_poly_C": arm.h_poly_C, "h_poly_pooled": arm.h_poly_pooled,
            "h_poly_hom": arm.h_poly_hom, "h_e": arm.h_e,
            "description": arm.description}


# =====================================================================
# prepare — merged BED, kinship, kernels, marker metadata
# =====================================================================

def cmd_prepare(args) -> None:
    out = Path(args.out_dir)
    prep = out / "prepare"
    prep.mkdir(parents=True, exist_ok=True)
    grm_npz = Path(args.grm_npz)
    geno_root = Path(args.geno_root)

    merged_bed = prep / "merged.bed"
    if merged_bed.exists() and not args.force:
        _log(f"prepare: merged BED exists, skip ({merged_bed})")
        # even on the skip path, re-validate that the reused merged BED still
        # matches the current GRM / subgenome BEDs in sample order.
        cached = pd.read_csv(prep / "samples.tsv", sep="\t", dtype=str)
        cached_iid = cached["iid"].to_numpy().astype(object)
        grm_s = np.asarray(np.load(grm_npz, allow_pickle=True)["samples"],
                           dtype=object)
        if not np.array_equal(cached_iid, grm_s):
            sys.exit("ERR: cached prepare/samples.tsv != grm_npz samples — "
                     "re-run prepare with --force (inputs changed)")
        for sg in SUBGENOMES:
            bed_s = np.asarray(
                load_bed_hardcall(geno_root / sg / "all").samples, dtype=object)
            if not np.array_equal(cached_iid, bed_s):
                sys.exit(f"ERR: cached prepare/samples.tsv != {sg} BED samples "
                         "— re-run prepare with --force (inputs changed)")
        # also re-validate the cached merged BED was built with the current
        # common-QC thresholds (a stale universe would break BH denominators).
        qp_path = prep / "qc_params.json"
        if not qp_path.exists():
            sys.exit("ERR: prepare/qc_params.json missing — re-run prepare "
                     "with --force")
        qp = json.loads(qp_path.read_text())
        if (qp.get("common_maf_min") != args.common_maf_min
                or qp.get("common_call_rate_min") != args.common_call_rate_min):
            sys.exit(
                f"ERR: cached merged BED built with common-QC "
                f"(maf>={qp.get('common_maf_min')}, "
                f"call_rate>={qp.get('common_call_rate_min')}) != current args "
                f"(maf>={args.common_maf_min}, "
                f"call_rate>={args.common_call_rate_min}) — re-run with --force")
    else:
        _log("prepare: loading per-subgenome genotypes")
        from bed_reader import to_bed
        grm_samples = np.asarray(
            np.load(grm_npz, allow_pickle=True)["samples"], dtype=object)
        dos, meta_rows, sid_all, chrom_num_all, pos_all = [], [], [], [], []
        samples_ref = None
        # deterministic chrom -> numeric map (A01..A10, C01..C09 -> 1..19)
        chrom_map: dict[str, int] = {}
        n_total = n_kept = 0
        for sg in SUBGENOMES:
            geno = load_bed_hardcall(geno_root / sg / "all")
            g_samples = np.asarray(geno.samples, dtype=object)
            if samples_ref is None:
                samples_ref = g_samples
            elif not np.array_equal(samples_ref, g_samples):
                sys.exit(f"ERR: sample order mismatch between subgenomes at {sg}")
            # common-QC keep mask -> ONE shared marker universe for every
            # method, so GEMMA/regenie/HomoeoGWAS test the same SNPs and the
            # per-method BH denominators match (avoids QC-universe mismatch).
            _Z, keep, _maf, _cr = sim.standardize_dosage(
                geno.dosage, maf_min=args.common_maf_min,
                call_rate_min=args.common_call_rate_min)
            n_total += int(keep.size)
            n_kept += int(keep.sum())
            kept_idx = np.where(keep)[0]
            for ci in sorted(set(str(geno.chrom[j]) for j in kept_idx)):
                if ci not in chrom_map:
                    chrom_map[ci] = len(chrom_map) + 1
            dos.append(np.asarray(geno.dosage, dtype=np.float64)[:, kept_idx])
            for j in kept_idx:
                oc = str(geno.chrom[j])
                merged_id = f"{sg}{j}"
                meta_rows.append({
                    "merged_id": merged_id, "subgenome": sg,
                    "orig_snp_id": str(geno.variant_ids[j]),
                    "orig_chrom": oc, "pos": int(geno.pos[j]),
                    "numeric_chrom": chrom_map[oc]})
                sid_all.append(merged_id)
                chrom_num_all.append(str(chrom_map[oc]))
                pos_all.append(int(geno.pos[j]))
        # hard assertion: the GRM, the A BED and the C BED must share one
        # analysis-sample order — kernels, simulated y and genotypes are all
        # row-aligned by position, so a mismatch would silently corrupt y.
        if not np.array_equal(samples_ref, grm_samples):
            sys.exit("ERR: grm_npz samples != genotype BED samples — the "
                     "analysis sample order must be identical across the GRM "
                     "and both subgenome BEDs")
        merged = np.concatenate(dos, axis=1)
        n_var = merged.shape[1]
        _log(f"prepare: common-QC kept {n_kept}/{n_total} markers "
             f"(MAF>={args.common_maf_min}, "
             f"call_rate>={args.common_call_rate_min}) -> writing merged BED")
        to_bed(
            str(merged_bed), merged,
            properties={
                "fid": ["0"] * len(samples_ref),
                "iid": list(samples_ref),
                "sid": sid_all,
                "chromosome": chrom_num_all,
                "bp_position": pos_all,
                "allele_1": ["A"] * n_var,
                "allele_2": ["B"] * n_var,
            },
            count_A1=True,
        )
        pd.DataFrame(meta_rows).to_csv(prep / "marker_meta.tsv", sep="\t", index=False)
        pd.DataFrame({"fid": ["0"] * len(samples_ref),
                      "iid": list(samples_ref)}).to_csv(
            prep / "samples.tsv", sep="\t", index=False)
        with open(prep / "chrom_map.json", "w") as fh:
            json.dump(chrom_map, fh, indent=2)
        # record the QC parameters the merged BED was built with, so a later
        # skip-path reuse can detect a stale universe (changed thresholds).
        with open(prep / "qc_params.json", "w") as fh:
            json.dump({"common_maf_min": args.common_maf_min,
                       "common_call_rate_min": args.common_call_rate_min,
                       "n_common": int(n_var)}, fh, indent=2)
        _log(f"prepare: wrote merged BED ({n_var} variants) + marker_meta")

    # --- kernels --------------------------------------------------------
    kern_npz = prep / "kernels.npz"
    if kern_npz.exists() and not args.force:
        _log("prepare: kernels.npz exists, skip")
    else:
        _log("prepare: building per-subgenome kernels")
        g = np.load(grm_npz, allow_pickle=True)
        K_A = normalize_kernel(np.asarray(g["G_A"], dtype=np.float64), "trace")
        K_C = normalize_kernel(np.asarray(g["G_C"], dtype=np.float64), "trace")
        K_sum = normalize_kernel(K_A + K_C, "trace")
        K_hom = normalize_kernel(hadamard_kernel({"A": K_A, "C": K_C}), "trace")
        np.savez(kern_npz, K_A=K_A, K_C=K_C, K_sum=K_sum, K_hom=K_hom,
                 samples=np.asarray(g["samples"], dtype=object))
        _log(f"prepare: wrote {kern_npz}")

    # --- GEMMA kinship --------------------------------------------------
    gdir = prep / "gemma"
    gdir.mkdir(exist_ok=True)
    kinship = gdir / "kinship.cXX.txt"
    if kinship.exists() and not args.force:
        _log("prepare: GEMMA kinship exists, skip")
    else:
        _log("prepare: computing GEMMA kinship (one-time)")
        samples = pd.read_csv(prep / "samples.tsv", sep="\t", dtype=str)
        # dummy non-constant phenotype so GEMMA keeps all individuals
        dummy = np.random.default_rng(0).standard_normal(len(samples))
        _link(merged_bed, gdir / "dummy.bed")
        _link(prep / "merged.bim", gdir / "dummy.bim")
        _write_fam(gdir / "dummy.fam", samples["fid"].tolist(),
                   samples["iid"].tolist(), dummy)
        rc, logtxt = _run([GEMMA, "-bfile", str(gdir / "dummy"), "-gk", "1",
                           "-o", "kinship", "-outdir", str(gdir)])
        (gdir / "kinship.run.log").write_text(logtxt)
        if rc != 0 or not kinship.exists():
            sys.exit(f"ERR: GEMMA kinship failed (rc={rc}); see {gdir}/kinship.run.log")
        _log(f"prepare: wrote {kinship}")

    # --- config ---------------------------------------------------------
    cfg = {
        "milestone": "M2.6", "panel": "horvath2020",
        "geno_root": str(geno_root), "grm_npz": str(grm_npz),
        "out_dir": str(out), "seed": args.seed,
        "subgenomes": list(SUBGENOMES),
        "curve_thresholds": CURVE_THRESHOLDS.tolist(),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    _log("prepare: DONE")


def _link(src: Path, dst: Path) -> None:
    """Hard-link src->dst (immutable shared file); overwrite if present."""
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _write_fam(path: Path, fid, iid, pheno) -> None:
    """Write a PLINK .fam with the given column-6 phenotype."""
    with open(path, "w") as fh:
        for f, i, y in zip(fid, iid, pheno):
            fh.write(f"{f} {i} 0 0 0 {y}\n")


def _run(cmd: list[str], timeout: int = 1800) -> tuple[int, str]:
    """Run a command, capture combined output."""
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout + "\n" + p.stderr
    except subprocess.TimeoutExpired as e:
        return 124, f"TIMEOUT after {timeout}s\n{e}"


# =====================================================================
# simulate — causal sets + phenotypes
# =====================================================================

def cmd_simulate(args) -> None:
    out = Path(args.out_dir)
    prep = out / "prepare"
    arms = _selected_arms(args)
    reps = _rep_counts(args)
    _log(f"simulate: arms={list(arms)}  reps={reps}")

    # standardized genotypes; keep_causal = causal-SNP eligibility universe
    Z, _keep_common, keep, meta, samples = _load_standardized(args)
    kern = np.load(prep / "kernels.npz", allow_pickle=True)
    # hard assertion: kernels (-> u draws) and genotypes (-> η draws) must be
    # row-aligned to the same analysis-sample order, or y would be corrupt.
    if not np.array_equal(samples, np.asarray(kern["samples"], dtype=object)):
        sys.exit("ERR: kernels.npz samples != genotype BED samples — re-run "
                 "prepare; analysis-sample order must match")
    factors = {
        "A": sim.kernel_factor(np.asarray(kern["K_A"], dtype=np.float64)),
        "C": sim.kernel_factor(np.asarray(kern["K_C"], dtype=np.float64)),
        "pooled": sim.kernel_factor(np.asarray(kern["K_sum"], dtype=np.float64)),
        "hom": sim.kernel_factor(np.asarray(kern["K_hom"], dtype=np.float64)),
    }
    all_arms = sim.default_arms()
    manifest_rows = []

    for arm_name in arms:
        arm = all_arms[arm_name]
        n_rep = reps[arm_name]
        # global arm index -> seed is stable regardless of the --arms subset
        arm_gidx = _arm_idx(arm_name)
        sim_dir = out / "sim" / arm_name
        sim_dir.mkdir(parents=True, exist_ok=True)
        q_A = 0 if arm.kind == "null" else args.q_causal
        q_C = 0 if arm.kind == "null" else args.q_causal
        for r in range(n_rep):
            npz_path = sim_dir / f"rep_{r:04d}.npz"
            if npz_path.exists() and not args.force:
                continue
            rng = np.random.default_rng(
                np.random.SeedSequence([args.seed, arm_gidx, r]))
            # causal selection
            idx_A = sim.select_causal_snps(
                rng, np.where(keep["A"])[0], meta["A"]["chrom"], meta["A"]["pos"],
                q_A, min_sep_bp=args.min_sep_bp)
            idx_C = sim.select_causal_snps(
                rng, np.where(keep["C"])[0], meta["C"]["chrom"], meta["C"]["pos"],
                q_C, min_sep_bp=args.min_sep_bp)
            causal = _make_causal_set(idx_A, idx_C, meta)
            Zc_A = Z["A"][:, idx_A] if q_A else np.zeros((Z["A"].shape[0], 0))
            Zc_C = Z["C"][:, idx_C] if q_C else np.zeros((Z["C"].shape[0], 0))
            pheno = sim.simulate_phenotype(rng, arm, causal, Zc_A, Zc_C, factors)
            np.savez(
                npz_path, y=pheno.y,
                causal_id=causal.causal_id, subgenome=causal.subgenome,
                snp_id=causal.snp_id, chrom=causal.chrom, pos=causal.pos,
                marker_index=causal.marker_index, effect=pheno.causal.effect,
                realized_var=json.dumps(pheno.realized_var))
            manifest_rows.append({
                "arm": arm_name, "rep": r, "n_causal": causal.n_causal,
                **{f"rv_{k}": v for k, v in pheno.realized_var.items()}})
        _log(f"simulate: {arm_name} — {n_rep} replicates ready")

    if manifest_rows:
        man = out / "sim" / "manifest.tsv"
        df = pd.DataFrame(manifest_rows)
        if man.exists():
            df = pd.concat([pd.read_csv(man, sep="\t"), df]).drop_duplicates(
                ["arm", "rep"], keep="last")
        df.sort_values(["arm", "rep"]).to_csv(man, sep="\t", index=False)
    _log("simulate: DONE")


def _make_causal_set(idx_A, idx_C, meta) -> sim.CausalSet:
    sub, sid, chrom, pos, midx = [], [], [], [], []
    for sg, idx in (("A", idx_A), ("C", idx_C)):
        for j in idx:
            sub.append(sg)
            sid.append(str(meta[sg]["snp_id"][j]))
            chrom.append(str(meta[sg]["chrom"][j]))
            pos.append(int(meta[sg]["pos"][j]))
            midx.append(int(j))
    q = len(sub)
    return sim.CausalSet(
        causal_id=np.arange(q, dtype=np.int64),
        subgenome=np.array(sub, dtype=object), snp_id=np.array(sid, dtype=object),
        chrom=np.array(chrom, dtype=object), pos=np.array(pos, dtype=np.int64),
        marker_index=np.array(midx, dtype=np.int64),
        effect=np.zeros(q, dtype=np.float64))


def _load_standardized(args):
    """Standardized per-subgenome genotype matrices + QC masks + metadata.

    ``Z`` is standardized over the common-QC universe (MAF >= common_maf_min)
    so truth-window LD is computed over every marker any method can test.
    ``keep_causal`` is the stricter MAF >= causal_maf_min subset from which
    causal SNPs are drawn (common, well-powered variants).

    Returns (Z, keep_common, keep_causal, meta, samples) — samples is the (n,)
    analysis-sample order, asserted identical across subgenomes.
    """
    geno_root = Path(args.geno_root)
    Z, keep_common, keep_causal, meta = {}, {}, {}, {}
    samples = None
    for sg in SUBGENOMES:
        geno = load_bed_hardcall(geno_root / sg / "all")
        g_samples = np.asarray(geno.samples, dtype=object)
        if samples is None:
            samples = g_samples
        elif not np.array_equal(samples, g_samples):
            sys.exit(f"ERR: sample order mismatch between subgenomes at {sg}")
        z, k, maf, _cr = sim.standardize_dosage(
            geno.dosage, maf_min=args.common_maf_min,
            call_rate_min=args.common_call_rate_min)
        Z[sg] = z
        keep_common[sg] = k
        keep_causal[sg] = k & (maf >= args.causal_maf_min)
        meta[sg] = {"snp_id": np.asarray(geno.variant_ids, dtype=object),
                    "chrom": np.asarray(geno.chrom, dtype=object),
                    "pos": np.asarray(geno.pos, dtype=np.int64)}
    return Z, keep_common, keep_causal, meta, samples


# =====================================================================
# run-homoeo — HomoeoGWAS multi-kernel scan + pooled-GRM EMMAX ablation
# =====================================================================

_GENO_CACHE: dict = {}
_KERN_CACHE: dict = {}


def _get_geno(geno_root: str):
    if "geno" not in _GENO_CACHE:
        _GENO_CACHE["geno"] = {
            sg: load_bed_hardcall(Path(geno_root) / sg / "all")
            for sg in SUBGENOMES}
    return _GENO_CACHE["geno"]


def _get_kernels(prep: str):
    if "kern" not in _KERN_CACHE:
        k = np.load(Path(prep) / "kernels.npz", allow_pickle=True)
        _KERN_CACHE["kern"] = {
            "A": np.asarray(k["K_A"], dtype=np.float64),
            "C": np.asarray(k["K_C"], dtype=np.float64),
            "sum": np.asarray(k["K_sum"], dtype=np.float64),
            "samples": np.asarray(k["samples"], dtype=object)}
    return _KERN_CACHE["kern"]


def _homoeo_worker(task: dict) -> dict:
    """Run HomoeoGWAS (A+C) and pooled-GRM EMMAX for one (arm, rep)."""
    geno = _get_geno(task["geno_root"])
    kern = _get_kernels(task["prep"])
    npz = np.load(task["sim_npz"], allow_pickle=True)
    y = np.asarray(npz["y"], dtype=np.float64)
    n = y.shape[0]
    X = np.ones((n, 1), dtype=np.float64)
    sample_ids = kern["samples"]
    out = {"arm": task["arm"], "rep": task["rep"], "runtime": {}, "status": {}}

    for method, kernels in (
        ("homoeo", {"A": kern["A"], "C": kern["C"]}),
        ("pooled_emmax", {"pooled": kern["sum"]}),
    ):
        t0 = time.time()
        try:
            reml = fit_multi_reml(y, X, kernels, n_starts=task["n_starts"],
                                  random_state=task["seed"])
            ctx = build_scan_context(y, X, kernels, reml.sigma2,
                                     sample_ids=sample_ids)
            frames = []
            for sg in SUBGENOMES:
                res = scan_snps(ctx, geno[sg], backend="cpu",
                                maf_min=task["scan_maf_min"], call_rate_min=0.90)
                frames.append(pd.DataFrame({
                    "subgenome": sg, "snp_id": res.snp_id, "chrom": res.chrom,
                    "pos": res.pos, "beta": res.beta, "chi2": res.chi2,
                    "p": res.p}))
            df = pd.concat(frames, ignore_index=True)
            dt = time.time() - t0
            _write_sumstats(task["out_dir"], method, task["arm"], task["rep"], df)
            out["runtime"][method] = dt
            out["status"][method] = "ok"
        except Exception as e:  # noqa: BLE001
            out["runtime"][method] = float("nan")
            out["status"][method] = f"FAIL: {type(e).__name__}: {e}"
    return out


def cmd_run_homoeo(args) -> None:
    out = Path(args.out_dir)
    tasks = _build_tasks(args, methods_dir=("homoeo", "pooled_emmax"))
    if not tasks:
        _log("run-homoeo: nothing to do (all sumstats present; use --force)")
        return
    payload = [{
        "arm": a, "rep": r, "geno_root": args.geno_root,
        "prep": str(out / "prepare"), "sim_npz": str(out / "sim" / a / f"rep_{r:04d}.npz"),
        "out_dir": str(out), "n_starts": args.n_starts,
        "scan_maf_min": args.scan_maf_min,
        "seed": int(np.random.SeedSequence([args.seed, 9, _arm_idx(a), r]).generate_state(1)[0]),
    } for (a, r) in tasks]
    _log(f"run-homoeo: {len(payload)} (arm,rep) tasks, n_jobs={args.n_jobs}")
    results = _map_tasks(_homoeo_worker, payload, args.n_jobs)
    _append_runtime(out, results)
    fails = [r for r in results if any(v.startswith("FAIL") for v in r["status"].values())]
    _log(f"run-homoeo: DONE ({len(results)} tasks, {len(fails)} with failures)")
    for r in fails[:5]:
        _log(f"  FAIL arm={r['arm']} rep={r['rep']}: {r['status']}")


# =====================================================================
# run-baselines — GEMMA + regenie per replicate
# =====================================================================

def _baseline_worker(task: dict) -> dict:
    """Run GEMMA + regenie for one (arm, rep)."""
    out_dir = Path(task["out_dir"])
    prep = Path(task["prep"])
    work = out_dir / "baseline_work" / task["arm"] / f"rep_{task['rep']:04d}"
    work.mkdir(parents=True, exist_ok=True)
    npz = np.load(task["sim_npz"], allow_pickle=True)
    y = np.asarray(npz["y"], dtype=np.float64)
    samples = pd.read_csv(prep / "samples.tsv", sep="\t", dtype=str)
    meta = pd.read_csv(prep / "marker_meta.tsv", sep="\t", dtype={"merged_id": str})
    res = {"arm": task["arm"], "rep": task["rep"], "runtime": {}, "status": {}}

    # ---- GEMMA --------------------------------------------------------
    t0 = time.time()
    try:
        _link(prep / "merged.bed", work / "g.bed")
        _link(prep / "merged.bim", work / "g.bim")
        _write_fam(work / "g.fam", samples["fid"].tolist(),
                   samples["iid"].tolist(), y)
        # -maf 0.005 (below the 0.01 common-universe floor) disables GEMMA's
        # own MAF filter so it reports the full shared marker set; the merged
        # BED is already common-QC'd, so every method tests an identical set.
        rc, log = _run([GEMMA, "-bfile", str(work / "g"),
                        "-k", str(prep / "gemma" / "kinship.cXX.txt"),
                        "-lmm", "4", "-maf", "0.005",
                        "-o", "gemma", "-outdir", str(work)])
        (work / "gemma.log").write_text(log)
        assoc = work / "gemma.assoc.txt"
        if rc == 0 and assoc.exists():
            df = pd.read_csv(assoc, sep="\t")
            df = df.rename(columns={"rs": "merged_id", "p_wald": "p"})
            df = _join_meta(df[["merged_id", "p"]], meta)
            _write_sumstats(task["out_dir"], "gemma", task["arm"], task["rep"], df)
            res["status"]["gemma"] = "ok"
        else:
            res["status"]["gemma"] = f"FAIL rc={rc}"
    except Exception as e:  # noqa: BLE001
        res["status"]["gemma"] = f"FAIL: {type(e).__name__}: {e}"
    res["runtime"]["gemma"] = time.time() - t0

    # ---- regenie ------------------------------------------------------
    t0 = time.time()
    try:
        ph = work / "pheno.txt"
        with open(ph, "w") as fh:
            fh.write("FID IID Y\n")
            for f, i, v in zip(samples["fid"], samples["iid"], y):
                fh.write(f"{f} {i} {v}\n")
        s1 = work / "step1"
        rc1, log1 = _run([REGENIE, "--step", "1", "--bed", str(prep / "merged"),
                          "--phenoFile", str(ph), "--phenoCol", "Y", "--qt",
                          "--bsize", "1000", "--lowmem",
                          "--lowmem-prefix", str(work / "rg_tmp"),
                          "--threads", "1", "--minMAC", "5", "--out", str(s1)])
        s2 = work / "step2"
        rc2, log2 = _run([REGENIE, "--step", "2", "--bed", str(prep / "merged"),
                          "--phenoFile", str(ph), "--phenoCol", "Y", "--qt",
                          "--pred", str(work / "step1_pred.list"),
                          "--bsize", "400", "--threads", "1", "--minMAC", "5",
                          "--out", str(s2)])
        (work / "regenie.log").write_text(log1 + "\n=== step2 ===\n" + log2)
        reg_file = work / "step2_Y.regenie"
        if rc1 == 0 and rc2 == 0 and reg_file.exists():
            df = pd.read_csv(reg_file, sep=r"\s+", engine="python", comment="#")
            df = df.rename(columns={"ID": "merged_id"})
            df["p"] = 10.0 ** (-df["LOG10P"].astype(np.float64))
            df = _join_meta(df[["merged_id", "p"]], meta)
            _write_sumstats(task["out_dir"], "regenie", task["arm"], task["rep"], df)
            res["status"]["regenie"] = "ok"
        else:
            res["status"]["regenie"] = f"FAIL rc1={rc1} rc2={rc2}"
    except Exception as e:  # noqa: BLE001
        res["status"]["regenie"] = f"FAIL: {type(e).__name__}: {e}"
    res["runtime"]["regenie"] = time.time() - t0

    if not task.get("keep_temp", False):
        for p in ("g.bed", "g.bim", "rg_tmp"):
            for f in work.glob(p + "*"):
                try:
                    f.unlink()
                except OSError:
                    pass
    return res


def _join_meta(df: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Map merged variant ids back to (subgenome, orig snp_id, chrom, pos).

    Every row of ``df`` must map — an unmatched merged_id means a parser or
    chromosome-remap mismatch, so raise rather than silently drop rows.
    """
    j = df.merge(meta, on="merged_id", how="inner")
    if len(j) != len(df):
        missing = sorted(set(df["merged_id"]) - set(meta["merged_id"]))
        raise ValueError(
            f"_join_meta: {len(df) - len(j)} of {len(df)} sumstats rows have a "
            f"merged_id absent from marker_meta (e.g. {missing[:5]}) — "
            "parser / chromosome-remap mismatch")
    return pd.DataFrame({
        "subgenome": j["subgenome"], "snp_id": j["orig_snp_id"],
        "chrom": j["orig_chrom"], "pos": j["pos"].astype(np.int64),
        "p": j["p"].astype(np.float64)})


def cmd_run_baselines(args) -> None:
    out = Path(args.out_dir)
    tasks = _build_tasks(args, methods_dir=("gemma", "regenie"))
    if not tasks:
        _log("run-baselines: nothing to do (all sumstats present; use --force)")
        return
    payload = [{
        "arm": a, "rep": r, "out_dir": str(out), "prep": str(out / "prepare"),
        "sim_npz": str(out / "sim" / a / f"rep_{r:04d}.npz"),
        "keep_temp": args.keep_temp,
    } for (a, r) in tasks]
    _log(f"run-baselines: {len(payload)} (arm,rep) tasks, n_jobs={args.n_jobs}")
    results = _map_tasks(_baseline_worker, payload, args.n_jobs)
    _append_runtime(out, results)
    fails = [r for r in results if any("FAIL" in v for v in r["status"].values())]
    _log(f"run-baselines: DONE ({len(results)} tasks, {len(fails)} with failures)")
    for r in fails[:5]:
        _log(f"  FAIL arm={r['arm']} rep={r['rep']}: {r['status']}")


# =====================================================================
# shared task / runtime plumbing
# =====================================================================

def _arm_idx(name: str) -> int:
    return list(sim.default_arms()).index(name)


def _selected_arms(args) -> list[str]:
    allnames = list(sim.default_arms())
    if args.arms:
        sel = [a.strip() for a in args.arms.split(",") if a.strip()]
        bad = [a for a in sel if a not in allnames]
        if bad:
            sys.exit(f"ERR: unknown arms {bad}; choose from {allnames}")
        return sel
    return allnames


def _rep_counts(args) -> dict[str, int]:
    base = dict(MODE_REPS[args.mode])
    if args.n_reps is not None:
        base = {k: args.n_reps for k in base}
    return {a: base[a] for a in _selected_arms(args)}


def _build_tasks(args, methods_dir) -> list[tuple[str, int]]:
    """(arm,rep) pairs still needing any of methods_dir sumstats."""
    out = Path(args.out_dir)
    reps = _rep_counts(args)
    tasks = []
    for arm, n_rep in reps.items():
        for r in range(n_rep):
            need = any(not (out / "sumstats" / m / arm / f"rep_{r:04d}.tsv").exists()
                       for m in methods_dir)
            if need or args.force:
                tasks.append((arm, r))
    return tasks


def _write_sumstats(out_dir, method, arm, rep, df: pd.DataFrame) -> None:
    d = Path(out_dir) / "sumstats" / method / arm
    d.mkdir(parents=True, exist_ok=True)
    cols = ["subgenome", "snp_id", "chrom", "pos", "p"]
    df[cols].to_csv(d / f"rep_{rep:04d}.tsv", sep="\t", index=False)


def _map_tasks(fn, payload, n_jobs):
    if n_jobs <= 1 or len(payload) == 1:
        return [fn(t) for t in payload]
    results = []
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        for res in ex.map(fn, payload):
            results.append(res)
    return results


def _append_runtime(out: Path, results: list[dict]) -> None:
    rows = []
    for r in results:
        for method, dt in r["runtime"].items():
            rows.append({"arm": r["arm"], "rep": r["rep"], "method": method,
                         "runtime_sec": dt, "status": r["status"].get(method, "")})
    if not rows:
        return
    path = out / "runtime.tsv"
    df = pd.DataFrame(rows)
    if path.exists():
        df = pd.concat([pd.read_csv(path, sep="\t"), df]).drop_duplicates(
            ["arm", "rep", "method"], keep="last")
    df.sort_values(["arm", "method", "rep"]).to_csv(path, sep="\t", index=False)


# =====================================================================
# summarize — power/FDR curves, paired tests, figures
# =====================================================================

METHODS = ("homoeo", "pooled_emmax", "gemma", "regenie")
BASELINES = ("gemma", "regenie")


def cmd_summarize(args) -> None:
    from scipy.stats import wilcoxon

    out = Path(args.out_dir)
    arms = _selected_arms(args)
    reps = _rep_counts(args)
    metrics = out / "metrics"
    metrics.mkdir(parents=True, exist_ok=True)
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    Z, _keep_common, _keep_causal, meta, _samples = _load_standardized(args)
    geno_meta = {sg: {"snp_id": meta[sg]["snp_id"], "chrom": meta[sg]["chrom"],
                      "pos": meta[sg]["pos"]} for sg in SUBGENOMES}
    rng = np.random.default_rng(args.seed + 777)

    # the shared marker universe — every method is intersected to this set so
    # the per-method BH denominators / FDR are computed on identical markers.
    mm = pd.read_csv(Path(args.out_dir) / "prepare" / "marker_meta.tsv",
                     sep="\t", dtype={"orig_snp_id": str})
    common_mi = pd.MultiIndex.from_arrays(
        [mm["subgenome"].to_numpy(), mm["orig_snp_id"].astype(str).to_numpy()])
    n_common = len(common_mi)
    _log(f"summarize: common marker universe = {n_common} markers")

    acceptance = []

    def check(name, ok, msg=""):
        acceptance.append({"check": name, "passed": bool(ok), "message": msg})
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {msg}" if msg else ""))

    curve_rows, perrep_rows, paired_rows, score_rows = [], [], [], []
    marker_counts = {m: [] for m in METHODS}
    arm_summaries = {}

    for arm_name in arms:
        n_rep = reps[arm_name]
        all_arms = sim.default_arms()
        arm = all_arms[arm_name]
        _log(f"summarize: arm {arm_name} ({n_rep} reps)")
        # per-rep×threshold power/fdr per method ; per-rep BH power
        T = len(CURVE_THRESHOLDS)
        pw = {m: np.full((n_rep, T), np.nan) for m in METHODS}
        fd = {m: np.full((n_rep, T), np.nan) for m in METHODS}
        bh_pw = {m: np.full(n_rep, np.nan) for m in METHODS}
        bh_fd = {m: np.full(n_rep, np.nan) for m in METHODS}
        present = {m: np.zeros(n_rep, dtype=bool) for m in METHODS}

        for r in range(n_rep):
            sim_npz = out / "sim" / arm_name / f"rep_{r:04d}.npz"
            if not sim_npz.exists():
                continue
            npz = np.load(sim_npz, allow_pickle=True)
            causal = sim.CausalSet(
                causal_id=np.asarray(npz["causal_id"], dtype=np.int64),
                subgenome=np.asarray(npz["subgenome"], dtype=object),
                snp_id=np.asarray(npz["snp_id"], dtype=object),
                chrom=np.asarray(npz["chrom"], dtype=object),
                pos=np.asarray(npz["pos"], dtype=np.int64),
                marker_index=np.asarray(npz["marker_index"], dtype=np.int64),
                effect=np.asarray(npz["effect"], dtype=np.float64))
            truth = sim.build_truth_windows(
                causal, geno_meta, Z, window_bp=args.truth_window_bp,
                r2_min=args.truth_r2_min)
            for m in METHODS:
                ss_path = out / "sumstats" / m / arm_name / f"rep_{r:04d}.tsv"
                if not ss_path.exists():
                    continue
                df = pd.read_csv(ss_path, sep="\t")
                # intersect to the shared marker universe (identical SNP set
                # and BH denominator across every method)
                df_mi = pd.MultiIndex.from_arrays(
                    [df["subgenome"].to_numpy(),
                     df["snp_id"].astype(str).to_numpy()])
                df = df[df_mi.isin(common_mi)].reset_index(drop=True)
                ss = {"subgenome": df["subgenome"].to_numpy(),
                      "snp_id": df["snp_id"].astype(str).to_numpy(),
                      "chrom": df["chrom"].astype(str).to_numpy(),
                      "pos": df["pos"].to_numpy(),
                      "p": df["p"].to_numpy()}
                present[m][r] = True
                marker_counts[m].append(len(df))
                curve = sim.power_fdr_curve(
                    ss, truth, CURVE_THRESHOLDS,
                    collapse_window_bp=args.truth_window_bp)
                for ti, ev in enumerate(curve):
                    pw[m][r, ti] = ev.power
                    fd[m][r, ti] = ev.fdp
                thr = sim.bh_threshold(ss["p"], args.bh_q)
                ev = sim.evaluate_threshold(ss, truth, thr,
                                            collapse_window_bp=args.truth_window_bp)
                bh_pw[m][r] = ev.power
                bh_fd[m][r] = ev.fdp
                perrep_rows.append({
                    "arm": arm_name, "rep": r, "method": m,
                    "n_markers": len(df),
                    "bh_power": ev.power, "bh_fdp": ev.fdp,
                    "bh_n_sig": ev.n_sig, "bh_n_loci": ev.n_loci,
                    "bh_tp_loci": ev.n_tp_loci, "bh_fp_loci": ev.n_fp_loci,
                    "bh_dup_loci": ev.n_duplicate_loci})

        # ---- aggregate curves -----------------------------------------
        mean_pw = {m: (_nanmean(pw[m], axis=0) if present[m].any()
                       else np.full(T, np.nan)) for m in METHODS}
        mean_fd = {m: (_nanmean(fd[m], axis=0) if present[m].any()
                       else np.full(T, np.nan)) for m in METHODS}
        for m in METHODS:
            if not present[m].any():
                continue
            for ti, t in enumerate(CURVE_THRESHOLDS):
                curve_rows.append({
                    "arm": arm_name, "method": m, "threshold": float(t),
                    "mean_power": float(mean_pw[m][ti]),
                    "mean_fdr": float(mean_fd[m][ti])})

        # ---- scalar scores --------------------------------------------
        scores = {}
        for m in METHODS:
            if not present[m].any():
                continue
            paf = sim.power_at_fdr(mean_pw[m], mean_fd[m], args.target_fdr)
            pauc = sim.partial_auc(mean_pw[m], mean_fd[m], fdr_cap=0.10)
            n_ok = int(present[m].sum())
            scores[m] = {
                "power_at_fdr": paf, "pauc_0.10": pauc,
                "bh_power_mean": float(_nanmean(bh_pw[m])),
                "bh_fdp_mean": float(_nanmean(bh_fd[m])),
                "n_reps": n_ok}
            score_rows.append({"arm": arm_name, "method": m, **scores[m]})

        # ---- paired tests: HomoeoGWAS vs each comparator ---------------
        arm_paired = {}
        if present["homoeo"].any():
            for comp in ("pooled_emmax", "gemma", "regenie"):
                if not present[comp].any():
                    continue
                both = present["homoeo"] & present[comp]
                if both.sum() < 3:
                    continue
                rec = {"arm": arm_name, "comparator": comp,
                       "n_paired": int(both.sum())}
                if arm.kind != "null":
                    a = bh_pw["homoeo"][both]
                    b = bh_pw[comp][both]
                    fa = bh_fd["homoeo"][both]
                    fb = bh_fd[comp][both]
                    bs = sim.paired_bootstrap(a, b, rng=rng, n_boot=args.n_boot)
                    try:
                        w = wilcoxon(a, b, alternative="greater",
                                     zero_method="zsplit")
                        wp = float(w.pvalue)
                    except ValueError:
                        wp = float("nan")
                    rec.update({
                        "metric": "bh_power",
                        "homoeo_mean": float(np.mean(a)),
                        "comparator_mean": float(np.mean(b)),
                        "delta": bs["delta"], "boot_ci_lo": bs["ci_lo"],
                        "boot_ci_hi": bs["ci_hi"], "boot_p": bs["p_value"],
                        "wilcoxon_p": wp,
                        "homoeo_fdp_mean": float(np.mean(fa)),
                        "comparator_fdp_mean": float(np.mean(fb))})
                # runtime (always)
                rt = _runtime_arrays(out, arm_name, both, "homoeo", comp)
                if rt is not None:
                    a_rt, b_rt = rt
                    bs_rt = sim.paired_bootstrap(b_rt, a_rt, rng=rng,
                                                 n_boot=args.n_boot)
                    rec.update({
                        "homoeo_runtime_mean": float(np.mean(a_rt)),
                        "comparator_runtime_mean": float(np.mean(b_rt)),
                        "runtime_delta_baseline_minus_homoeo": bs_rt["delta"],
                        "runtime_boot_p": bs_rt["p_value"]})
                paired_rows.append(rec)
                arm_paired[comp] = rec

        arm_summaries[arm_name] = {
            "arm": _arm_table(arm), "n_reps": n_rep, "scores": scores,
            "paired": arm_paired}

        # ---- figures --------------------------------------------------
        _plot_power_fdr(figs / f"power_fdr_{arm_name}.png", arm_name,
                        mean_pw, mean_fd, present)
        if arm.kind != "null":
            _plot_power_box(figs / f"power_box_{arm_name}.png", arm_name,
                            bh_pw, present)

    # ---- write metric tables ------------------------------------------
    pd.DataFrame(curve_rows).to_csv(metrics / "power_fdr_curve.tsv",
                                    sep="\t", index=False)
    pd.DataFrame(perrep_rows).to_csv(metrics / "per_replicate.tsv",
                                     sep="\t", index=False)
    pd.DataFrame(score_rows).to_csv(metrics / "method_scores.tsv",
                                    sep="\t", index=False)
    if paired_rows:
        pd.DataFrame(paired_rows).to_csv(metrics / "paired_tests.tsv",
                                         sep="\t", index=False)
    _plot_runtime(out, figs / "runtime_box.png")

    # ---- exit-gate verdict --------------------------------------------
    verdict = _exit_gate(arm_summaries)
    _log("=== EXIT-GATE VERDICT ===")
    for line in verdict["lines"]:
        print("  " + line)

    # ---- acceptance checks --------------------------------------------
    print("\nacceptance:")
    # every requested (arm, method) replicate must have produced a sumstats
    for arm_name in arms:
        s = arm_summaries.get(arm_name, {})
        sc = s.get("scores", {})
        exp = reps[arm_name]
        missing = {m: exp - sc.get(m, {}).get("n_reps", 0) for m in METHODS}
        check(f"{arm_name}_all_replicates_complete",
              all(v == 0 for v in missing.values()),
              f"expected {exp}/method; missing={ {k: v for k, v in missing.items() if v} }")
    # no failed external-binary / homoeo runs in runtime.tsv
    rt_path = out / "runtime.tsv"
    if rt_path.exists():
        rt = pd.read_csv(rt_path, sep="\t")
        rt = rt[rt["arm"].isin(arms)]
        bad = rt[rt["status"].astype(str).str.contains("FAIL", na=False)]
        check("no_failed_method_runs", len(bad) == 0,
              f"{len(bad)} failed runs" if len(bad)
              else "all method runs ok")
    # every method/arm/rep must test EXACTLY the common universe — identical
    # BH denominators, no silent FDR mismatch.
    mk = {m: marker_counts[m] for m in METHODS if marker_counts[m]}
    if mk:
        all_counts = [c for v in mk.values() for c in v]
        lo, hi = min(all_counts), max(all_counts)
        check("common_marker_universe",
              lo == n_common and hi == n_common,
              f"marker counts span [{lo},{hi}]; all must equal "
              f"common={n_common}")
    if "S5_null" in arms:
        nullsc = arm_summaries.get("S5_null", {}).get("scores", {})
        h_fdp = nullsc.get("homoeo", {}).get("bh_fdp_mean", float("nan"))
        check("null_arm_homoeo_fdp_controlled",
              np.isfinite(h_fdp) and h_fdp <= 0.20,
              f"homoeo null BH-FDP={h_fdp:.3f} (<=0.20)")
    check("exit_gate_evaluated", verdict["evaluated"], verdict["headline"])
    # the exit-gate *verdict* is the scientific result (in summary.json /
    # printed above), not a pipeline-integrity check — production runs decide
    # storyline from verdict["passed"]; it is not asserted here.

    all_pass = all(c["passed"] for c in acceptance)
    summary = {
        "milestone": "M2.6", "script": "run_m2_6_horvath_sim_benchmark.py",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": args.mode, "arms": arms, "rep_counts": reps,
        "params": {"q_causal": args.q_causal, "min_sep_bp": args.min_sep_bp,
                   "common_maf_min": args.common_maf_min,
                   "common_call_rate_min": args.common_call_rate_min,
                   "causal_maf_min": args.causal_maf_min,
                   "scan_maf_min": args.scan_maf_min,
                   "n_starts": args.n_starts, "n_common_markers": n_common,
                   "bh_q": args.bh_q, "target_fdr": args.target_fdr,
                   "truth_window_bp": args.truth_window_bp,
                   "truth_r2_min": args.truth_r2_min, "n_boot": args.n_boot,
                   "seed": args.seed},
        "arm_summaries": arm_summaries,
        "exit_gate": verdict,
        "acceptance": acceptance,
        "acceptance_all_passed": all_pass,
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, default=_json_default))
    _log(f"summarize: wrote {out / 'summary.json'}")
    n_pass = sum(c["passed"] for c in acceptance)
    print(f"\nacceptance: {n_pass}/{len(acceptance)} checks passed")
    if all_pass:
        print("✅ M2.6 simulation benchmark acceptance PASS")
    else:
        failed = [c["check"] for c in acceptance if not c["passed"]]
        print(f"❌ M2.6 acceptance FAIL — {failed}")
        sys.exit(1)


def _runtime_arrays(out: Path, arm: str, mask: np.ndarray,
                    method_a: str, method_b: str):
    """Per-replicate paired runtime arrays (method_a, method_b) for `mask` reps."""
    path = out / "runtime.tsv"
    if not path.exists():
        return None
    rt = pd.read_csv(path, sep="\t")
    rt = rt[rt["arm"] == arm]
    reps = np.where(mask)[0]
    a_vals, b_vals = [], []
    for r in reps:
        ra = rt[(rt["rep"] == r) & (rt["method"] == method_a)]["runtime_sec"]
        rb = rt[(rt["rep"] == r) & (rt["method"] == method_b)]["runtime_sec"]
        if len(ra) and len(rb) and np.isfinite(ra.iloc[0]) and np.isfinite(rb.iloc[0]):
            a_vals.append(float(ra.iloc[0]))
            b_vals.append(float(rb.iloc[0]))
    if len(a_vals) < 3:
        return None
    return np.array(a_vals), np.array(b_vals)


def _exit_gate(arm_summaries: dict, *, fdp_cap: float = 0.10,
               fdp_margin: float = 0.03) -> dict:
    """Decide the Phase 2 exit gate: HomoeoGWAS beats >=2 main-bucket baselines.

    Evaluated per stratified exit-gate arm (S1, S2). For the paper-grade claim
    the win must hold in *every* evaluated unequal-variance arm — not be
    cherry-picked from the favourable mirror. A power win additionally requires
    HomoeoGWAS's BH-FDP to stay controlled (<= fdp_cap) and not exceed the
    baseline's by more than fdp_margin: extra power must not buy false
    discoveries. A runtime win requires HomoeoGWAS faster than both baselines
    (paired bootstrap p < 0.05).
    """
    lines: list[str] = []
    arms_eval: list[str] = []
    power_win_arm: dict[str, bool] = {}
    runtime_win_arm: dict[str, bool] = {}
    for arm_name in EXIT_GATE_ARMS:
        s = arm_summaries.get(arm_name)
        if not s or not s.get("paired"):
            continue
        paired = s["paired"]
        if not all(b in paired for b in BASELINES):
            continue
        arms_eval.append(arm_name)
        praw = {b: paired[b].get("boot_p", float("nan")) for b in BASELINES}
        deltas = {b: paired[b].get("delta", float("nan")) for b in BASELINES}
        h_fdp = {b: paired[b].get("homoeo_fdp_mean", float("nan"))
                 for b in BASELINES}
        c_fdp = {b: paired[b].get("comparator_fdp_mean", float("nan"))
                 for b in BASELINES}
        if all(np.isfinite(v) for v in praw.values()):
            padj = sim.holm_correct(praw)
            sig = all(padj[b] < 0.05 and deltas[b] > 0 for b in BASELINES)
            fdp_ok = all(
                np.isfinite(h_fdp[b]) and h_fdp[b] <= fdp_cap
                and (not np.isfinite(c_fdp[b])
                     or h_fdp[b] <= c_fdp[b] + fdp_margin)
                for b in BASELINES)
            power_win = bool(sig and fdp_ok)
        else:
            padj, sig, fdp_ok, power_win = {}, False, False, False
        power_win_arm[arm_name] = power_win
        rt_p = {b: paired[b].get("runtime_boot_p", float("nan"))
                for b in BASELINES}
        rt_d = {b: paired[b].get("runtime_delta_baseline_minus_homoeo",
                                 float("nan")) for b in BASELINES}
        runtime_win = all(
            np.isfinite(rt_p[b]) and rt_p[b] < 0.05
            and np.isfinite(rt_d[b]) and rt_d[b] > 0 for b in BASELINES)
        runtime_win_arm[arm_name] = bool(runtime_win)
        _nan = float("nan")
        padj_s = ", ".join(f"{b}:{padj.get(b, _nan):.3g}" for b in BASELINES)
        delta_s = ", ".join(f"{b}:{deltas[b]:+.3f}" for b in BASELINES)
        rtp_s = ", ".join(f"{b}:{rt_p[b]:.3g}" for b in BASELINES)
        lines.append(
            f"{arm_name}: power_win={power_win} (Holm-p [{padj_s}], "
            f"deltas [{delta_s}], fdp_ok={fdp_ok}); "
            f"runtime_win={runtime_win} (boot-p [{rtp_s}])")

    evaluated = len(arms_eval) > 0
    both_arms = set(arms_eval) >= set(EXIT_GATE_ARMS)
    # the Phase 2 exit claim requires the win in BOTH unequal-variance arms
    # (S1 AND S2) — a single-arm result is reported but does not pass.
    power_pass = both_arms and all(power_win_arm[a] for a in EXIT_GATE_ARMS)
    runtime_pass = both_arms and all(runtime_win_arm[a] for a in EXIT_GATE_ARMS)
    passed = bool(power_pass or runtime_pass)
    if not evaluated:
        headline = "not evaluated (need S1/S2 paired vs both baselines)"
        lines.append(headline)
    elif not both_arms:
        headline = (f"NOT PASSED — only {'+'.join(arms_eval)} evaluated; the "
                    "exit claim needs both S1 and S2")
    else:
        if power_pass and runtime_pass:
            headline = "PASS — power & runtime win over GEMMA & regenie (S1+S2)"
        elif power_pass:
            headline = "PASS — power win over GEMMA & regenie (S1+S2)"
        elif runtime_pass:
            headline = "PASS — runtime win over GEMMA & regenie (S1+S2)"
        else:
            headline = "no significant win over >=2 baselines (S1+S2)"
    return {"evaluated": evaluated, "passed": passed, "both_arms": both_arms,
            "power_pass": power_pass, "runtime_pass": runtime_pass,
            "arms_evaluated": arms_eval, "headline": headline, "lines": lines}


# =====================================================================
# plotting
# =====================================================================

_COLORS = {"homoeo": "#d62728", "pooled_emmax": "#9467bd",
           "gemma": "#1f77b4", "regenie": "#2ca02c"}
_LABELS = {"homoeo": "HomoeoGWAS", "pooled_emmax": "pooled-EMMAX",
           "gemma": "GEMMA", "regenie": "regenie"}


def _plot_power_fdr(path, arm, mean_pw, mean_fd, present):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    for m in METHODS:
        if not present[m].any():
            continue
        fdr, pwr = mean_fd[m], mean_pw[m]
        ok = np.isfinite(fdr) & np.isfinite(pwr)
        order = np.argsort(fdr[ok])
        ax.plot(fdr[ok][order], pwr[ok][order], "-o", ms=3, lw=1.3,
                color=_COLORS[m], label=_LABELS[m])
    ax.axvline(0.05, ls="--", lw=0.8, color="gray")
    ax.set_xlabel("empirical FDR")
    ax.set_ylabel("power")
    ax.set_xlim(-0.02, 0.6)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"M2.6 power vs FDR — {arm}")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _plot_power_box(path, arm, bh_pw, present):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    data, labels, colors = [], [], []
    for m in METHODS:
        if not present[m].any():
            continue
        vals = bh_pw[m][present[m]]
        vals = vals[np.isfinite(vals)]
        if vals.size:
            data.append(vals)
            labels.append(_LABELS[m])
            colors.append(_COLORS[m])
    if data:
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showmeans=True)
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.6)
    ax.set_ylabel(f"power @ BH q≤0.05")
    ax.set_title(f"M2.6 per-replicate power — {arm}")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _plot_runtime(out: Path, path):
    rt_file = out / "runtime.tsv"
    if not rt_file.exists():
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rt = pd.read_csv(rt_file, sep="\t")
    rt = rt[np.isfinite(rt["runtime_sec"])]
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    data, labels, colors = [], [], []
    for m in METHODS:
        vals = rt[rt["method"] == m]["runtime_sec"].to_numpy()
        if vals.size:
            data.append(vals)
            labels.append(_LABELS[m])
            colors.append(_COLORS[m])
    if data:
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showmeans=True)
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.6)
    ax.set_ylabel("per-replicate runtime (s, single-thread)")
    ax.set_title("M2.6 runtime by method (all arms)")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# run — chain everything
# =====================================================================

def cmd_run(args) -> None:
    cmd_prepare(args)
    cmd_simulate(args)
    cmd_run_homoeo(args)
    cmd_run_baselines(args)
    cmd_summarize(args)


# =====================================================================
# CLI
# =====================================================================

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="M2.6 Horvath2020 simulation benchmark")
    ap.add_argument("subcommand",
                    choices=["prepare", "simulate", "run-homoeo",
                             "run-baselines", "summarize", "run"])
    ap.add_argument("--out-dir",
                    default=str(ROOT / "results/phase2/m2_6_sim/horvath2020"))
    ap.add_argument("--geno-root",
                    default=str(ROOT / "data/processed/rapeseed/horvath"))
    ap.add_argument("--grm-npz",
                    default=str(ROOT / "results/phase2/m2_1/horvath2020/grm_A_C.npz"))
    ap.add_argument("--mode", choices=["smoke", "production"], default="smoke")
    ap.add_argument("--arms", default=None,
                    help="comma-separated subset of arm names")
    ap.add_argument("--n-reps", type=int, default=None,
                    help="override replicate count for every arm")
    ap.add_argument("--q-causal", type=int, default=4,
                    help="causal SNPs per subgenome (pre-registered: 4)")
    ap.add_argument("--min-sep-bp", type=int, default=1_000_000)
    ap.add_argument("--common-maf-min", type=float, default=0.01,
                    help="MAF floor for the shared marker universe — every "
                         "method scans & BH-corrects over exactly this set")
    ap.add_argument("--common-call-rate-min", type=float, default=0.95,
                    help="call-rate floor for the shared marker universe "
                         "(>= GEMMA's default -miss 0.05 so no method drops "
                         "common-universe markers)")
    ap.add_argument("--causal-maf-min", type=float, default=0.05,
                    help="stricter MAF floor for causal-SNP eligibility "
                         "(causal SNPs are common, well-powered variants)")
    ap.add_argument("--scan-maf-min", type=float, default=0.01)
    ap.add_argument("--n-starts", type=int, default=5,
                    help="REML optimizer starts")
    ap.add_argument("--truth-window-bp", type=int, default=500_000)
    ap.add_argument("--truth-r2-min", type=float, default=0.20)
    ap.add_argument("--bh-q", type=float, default=0.05,
                    help="per-replicate BH q-level for the headline metric")
    ap.add_argument("--target-fdr", type=float, default=0.05,
                    help="empirical-FDR level for Power@FDR")
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260522)
    ap.add_argument("--keep-temp", action="store_true")
    ap.add_argument("--force", action="store_true")
    return ap


def main() -> None:
    args = build_parser().parse_args()
    dispatch = {
        "prepare": cmd_prepare, "simulate": cmd_simulate,
        "run-homoeo": cmd_run_homoeo, "run-baselines": cmd_run_baselines,
        "summarize": cmd_summarize, "run": cmd_run,
    }
    t0 = time.time()
    dispatch[args.subcommand](args)
    _log(f"[{args.subcommand}] finished in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
