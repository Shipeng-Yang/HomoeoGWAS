#!/usr/bin/env python3
"""Phase 2 M2.5-v2 — wheat Watkins full per-SNP scan, 2-GPU data-parallel.

The genuine hexaploid GWAS: streams all 86.6M post-QC wheat SNPs (A 34.2M /
B 43.4M / D 8.9M) through the fixed-V score test, one GPU per worker, and
produces per-subgenome result TSVs + Manhattan / QQ / λ_GC.

Modes
-----
  build        sample-join + pruned-GRM + A/B/D REML  ->  scan_context.npz
  worker       load context, stream-scan assigned subgenomes  ->  scan_<sg>.tsv.gz
  postprocess  aggregate scan_<sg>.tsv.gz  ->  λ_GC / Manhattan / QQ / summary
  run          (default) build -> launch 2 GPU workers (0:A,D  1:B) -> postprocess

2-GPU: no NVLink, no NCCL — two independent processes pinned via
CUDA_VISIBLE_DEVICES, each owning a private copy of the (tiny) projection P.
LOCO and proximal-contamination correction are deferred to Phase 3; this is a
standard EMMAX/P3D genome-wide scan with one global fixed V.

Run in the GPU env:  ~/miniconda3/envs/polygwas-gpu/bin/python run_m2_5_v2_wheat_scan.py
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from homoeogwas import fit_multi_reml, normalize_kernel  # noqa: E402
from homoeogwas.grm import compute_grm  # noqa: E402
from homoeogwas.io import GenoChunk, load_bed_hardcall  # noqa: E402
from homoeogwas.scan import ScanContext, lambda_gc, scan_bed_stream  # noqa: E402

SUBGENOMES = ("A", "B", "D")
GPU_ASSIGN = {0: ("A", "D"), 1: ("B",)}     # near-balanced: 43.1M vs 43.4M SNP
WHEAT_CHROM_ORDER = [f"chr{i}{s}" for i in range(1, 8) for s in ("A", "B", "D")]
_CHI2_1_MEDIAN = 0.4549364231195724


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o)}")


# build: scan context (sample-join + pruned-GRM + REML)

def build_context(args, out_dir: Path) -> dict:
    """Build the fixed-V projection from the M2.4.5 pruned GRMs; save npz."""
    wheat_dir = ROOT / "data/processed/wheat"
    pruned_dir = ROOT / "results/phase2/m2_4_5/wheat_watkins/pruned"
    pheno = pd.read_csv(wheat_dir / "pheno_clean.tsv", sep="\t").set_index("sample")
    trait = args.trait
    if trait not in pheno.columns:
        sys.exit(f"ERR: trait {trait!r} not in wheat pheno")

    # analysis samples = A/B/D psam ∩ pheno non-missing, in psam(A) order
    psam = {sg: list(pd.read_csv(wheat_dir / sg / "all.psam", sep="\t").iloc[:, 0]
                     .astype(str)) for sg in SUBGENOMES}
    common = set(psam["A"]) & set(psam["B"]) & set(psam["D"])
    ok = {s for s in pheno.index if not pd.isna(pheno.loc[s, trait])}
    analysis = [s for s in psam["A"] if s in common and s in ok]
    n = len(analysis)
    if n < 750:
        sys.exit(f"ERR: only {n} analysis samples")
    y = pheno.loc[analysis, trait].astype(np.float64).to_numpy()
    X = np.ones((n, 1), dtype=np.float64)

    # per-subgenome GRM from the M2.4.5 pruned BEDs
    canonical = {}
    for sg in SUBGENOMES:
        bed = pruned_dir / sg
        if not bed.with_suffix(".bed").exists():
            sys.exit(f"ERR: M2.4.5 pruned BED missing: {bed}.bed (run M2.4.5 first)")
        chunk = load_bed_hardcall(bed)
        pos = {s: i for i, s in enumerate(np.asarray(chunk.samples, dtype=str))}
        order = np.array([pos[s] for s in analysis], dtype=np.int64)
        sub = GenoChunk(samples=np.asarray(analysis, dtype=object),
                        variant_ids=chunk.variant_ids, chrom=chunk.chrom,
                        pos=chunk.pos, dosage=chunk.dosage[order, :])
        G, _ = compute_grm(sub, maf_min=args.maf_min)
        canonical[sg] = normalize_kernel(G, mode="trace")
        print(f"  G_{sg} built (n={n})")

    reml = fit_multi_reml(y, X, canonical, n_starts=args.n_starts,
                          random_state=args.seed)
    print(f"  REML σ²: { {k: round(v, 4) for k, v in reml.sigma2.items()} }")

    from homoeogwas.scan import build_scan_context
    ctx = build_scan_context(y, X, canonical, reml.sigma2,
                             sample_ids=np.asarray(analysis, dtype=object))
    npz = out_dir / "scan_context.npz"
    np.savez(npz, P=ctx.P, Py=ctx.Py, sample_ids=np.asarray(analysis, dtype=object),
             n=n, p=1, kernel_names=np.asarray(SUBGENOMES, dtype=object),
             sigma2_json=json.dumps(reml.sigma2), jitter_used=ctx.jitter_used)
    print(f"  wrote {npz}")
    return {"n_analysis": n, "trait": trait, "reml_sigma2": dict(reml.sigma2),
            "reml_pve": dict(reml.pve), "context_npz": str(npz)}


def load_context(npz_path: Path) -> ScanContext:
    """Reconstruct a ScanContext from a build-mode npz."""
    d = np.load(npz_path, allow_pickle=True)
    return ScanContext(
        P=np.ascontiguousarray(d["P"]), Py=np.ascontiguousarray(d["Py"]),
        n=int(d["n"]), p=int(d["p"]),
        sigma2=json.loads(str(d["sigma2_json"])),
        kernel_names=list(d["kernel_names"]),
        sample_ids=np.asarray(d["sample_ids"], dtype=object),
        jitter_used=float(d["jitter_used"]), dtype="float64",
    )


# worker: stream-scan assigned subgenomes on one GPU

def run_worker(args, out_dir: Path) -> dict:
    ctx = load_context(out_dir / "scan_context.npz")
    subg = [s.strip() for s in args.subgenomes.split(",") if s.strip()]
    wheat_bed = ROOT / "data/processed/wheat_bed"
    cuda = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    results = []
    for sg in subg:
        bed = wheat_bed / sg / "all"
        if not bed.with_suffix(".bed").exists():
            sys.exit(f"ERR: wheat BED missing: {bed}.bed")
        out_tsv = out_dir / f"scan_{sg}.tsv.gz"
        print(f"[worker CUDA={cuda}] scanning {sg} -> {out_tsv}")
        t0 = time.time()
        summ = scan_bed_stream(ctx, bed, out_tsv, backend=args.backend,
                               chunk_size=args.chunk_size, maf_min=args.maf_min,
                               call_rate_min=args.call_rate_min, subgenome=sg)
        rt = time.time() - t0
        print(f"[worker CUDA={cuda}] {sg}: {summ.n_input} -> {summ.n_kept} kept "
              f"in {rt:.0f}s ({summ.backend_used})")
        results.append({"subgenome": sg, "n_input": summ.n_input,
                        "n_kept": summ.n_kept, "backend": summ.backend_used,
                        "runtime_sec": round(rt, 1),
                        "filter_counts": summ.filter_counts})
    prof = {"cuda_visible_devices": cuda, "subgenomes": subg, "results": results}
    with open(out_dir / f"worker_profile_{'_'.join(subg)}.json", "w") as f:
        json.dump(prof, f, indent=2, default=_json_default)
    return prof


# postprocess: λ_GC / Manhattan / QQ / summary

def _scan_file_pass(tsv_gz: Path, thin: int = 50):
    """One streaming pass over a scan TSV.gz. Returns:
      n           — total markers
      chi2        — full χ² array (for λ_GC)
      chrom_chi2  — {chrom: χ² array} (for per-chromosome λ_GC)
      plot_df     — all p<1e-3 rows + a 1/thin background sample (for Manhattan)
    """
    chi2_parts: list[np.ndarray] = []
    chrom_chi2: dict[str, list[np.ndarray]] = {}
    plot_parts: list[pd.DataFrame] = []
    n = 0
    for chunk in pd.read_csv(tsv_gz, sep="\t",
                             usecols=["chrom", "pos", "chi2", "p"],
                             chunksize=1_000_000):
        n += len(chunk)
        chi2_parts.append(chunk["chi2"].to_numpy(dtype=np.float64))
        for ch, grp in chunk.groupby("chrom"):
            chrom_chi2.setdefault(str(ch), []).append(
                grp["chi2"].to_numpy(dtype=np.float64))
        sig = chunk[chunk["p"] < 1e-3]
        bg = chunk.iloc[::thin]
        plot_parts.append(pd.concat([sig, bg]).drop_duplicates())
    chi2 = np.concatenate(chi2_parts) if chi2_parts else np.zeros(0)
    chrom_chi2 = {c: np.concatenate(v) for c, v in chrom_chi2.items()}
    plot = (pd.concat(plot_parts, ignore_index=True) if plot_parts
            else pd.DataFrame(columns=["chrom", "pos", "chi2", "p"]))
    return n, chi2, chrom_chi2, plot


def _manhattan(plot_df: pd.DataFrame, out_path: Path, trait: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    chrom = plot_df["chrom"].astype(str).to_numpy()
    present = [c for c in WHEAT_CHROM_ORDER if c in set(chrom)]
    rank = {c: i for i, c in enumerate(present)}
    mask = np.array([c in rank for c in chrom])
    chrom = chrom[mask]
    pos = plot_df["pos"].to_numpy(dtype=np.int64)[mask]
    logp = -np.log10(np.clip(plot_df["p"].to_numpy(dtype=np.float64)[mask],
                             1e-300, 1.0))
    span = int(pos.max()) + 1 if pos.size else 1
    order = np.argsort(np.array([rank[c] for c in chrom], dtype=np.int64) * span + pos)
    chrom, logp = chrom[order], logp[order]
    x = np.arange(chrom.size)
    colors = ["#1f77b4" if rank[c] % 2 == 0 else "#9ecae1" for c in chrom]
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.scatter(x, logp, s=2, c=colors)
    ticks, labels = [], []
    for c in present:
        idx = np.where(chrom == c)[0]
        if idx.size:
            ticks.append(float(idx.mean()))
            labels.append(c.replace("chr", ""))
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.axhline(-np.log10(5e-8), ls="--", color="red", lw=0.8)
    ax.set_ylabel("-log10(p)")
    ax.set_title(f"M2.5-v2 wheat Watkins Manhattan — {trait} (thinned)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _qq(chi2: np.ndarray, out_path: Path, trait: str, lam: float):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats
    p = np.clip(stats.chi2.sf(chi2, df=1), 1e-300, 1.0)
    obs = np.sort(-np.log10(p))[::-1]
    exp = -np.log10((np.arange(1, obs.size + 1) - 0.5) / obs.size)
    step = max(1, obs.size // 200_000)         # thin the scatter for rendering
    fig, ax = plt.subplots(figsize=(4.6, 4.5))
    lim = max(float(obs[0]), float(exp[-1])) * 1.05 if obs.size else 1.0
    ax.plot([0, lim], [0, lim], ls="--", color="k", lw=1)
    ax.scatter(exp[::step], obs[::step], s=3, color="#1f77b4")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("expected -log10(p)")
    ax.set_ylabel("observed -log10(p)")
    ax.set_title(f"M2.5-v2 wheat QQ — {trait}  λ_GC={lam:.3f}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def postprocess(args, out_dir: Path) -> dict:
    trait = args.trait
    per_sg: dict[str, dict] = {}
    all_chi2: list[np.ndarray] = []
    plot_parts: list[pd.DataFrame] = []
    lam_rows: list[dict] = []
    chrom_lam: list[dict] = []
    for sg in SUBGENOMES:
        tsv = out_dir / f"scan_{sg}.tsv.gz"
        if not tsv.exists():
            print(f"  [skip] {tsv.name} not present (partial / canary run)")
            continue
        n, chi2, chrom_chi2, plot = _scan_file_pass(tsv)
        lam_sg = lambda_gc(chi2)
        per_sg[sg] = {"n_markers": n, "lambda_gc": lam_sg}
        lam_rows.append({"scope": "subgenome", "level": sg,
                         "n_markers": n, "lambda_gc": lam_sg})
        for ch, cc in sorted(chrom_chi2.items()):
            chrom_lam.append({"scope": "chrom", "level": ch,
                              "n_markers": int(cc.size), "lambda_gc": lambda_gc(cc)})
        all_chi2.append(chi2)
        plot["subgenome"] = sg
        plot_parts.append(plot)
        print(f"  {sg}: {n} markers, λ_GC={lam_sg:.4f}")

    if not all_chi2:
        sys.exit("ERR: no scan_<sg>.tsv.gz found — run workers first")
    overall = np.concatenate(all_chi2)
    lam_all = lambda_gc(overall)
    lam_rows = ([{"scope": "all", "level": "all",
                  "n_markers": int(overall.size), "lambda_gc": lam_all}]
                + lam_rows + chrom_lam)
    pd.DataFrame(lam_rows).to_csv(out_dir / f"lambda_gc_{trait}.tsv",
                                  sep="\t", index=False)

    plot_df = pd.concat(plot_parts, ignore_index=True)
    _manhattan(plot_df, out_dir / f"manhattan_{trait}.png", trait)
    _qq(overall, out_dir / f"qq_{trait}.png", trait, lam_all)
    print(f"  overall λ_GC={lam_all:.4f}  total markers={overall.size}")
    return {"lambda_gc_all": lam_all, "per_subgenome": per_sg,
            "n_markers_total": int(overall.size)}


# orchestrate

def main():
    ap = argparse.ArgumentParser(description="M2.5-v2 wheat full per-SNP scan")
    ap.add_argument("--mode", default="run",
                    choices=["run", "build", "worker", "postprocess"])
    ap.add_argument("--trait", default="days_to_emerg")
    ap.add_argument("--subgenomes", default="A,B,D",
                    help="worker mode: comma-list of subgenomes to scan")
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--chunk-size", type=int, default=200_000)
    ap.add_argument("--n-starts", type=int, default=10)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--maf-min", type=float, default=0.01)
    ap.add_argument("--call-rate-min", type=float, default=0.9)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--canary-subgenome", default=None,
                    help="run mode: scan only this subgenome on GPU 0 (fast canary)")
    args = ap.parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "results/phase2/m2_5_v2/wheat_watkins"
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    if args.mode == "build":
        meta = build_context(args, out_dir)
        print(json.dumps(meta, indent=2, default=_json_default))
        return
    if args.mode == "worker":
        run_worker(args, out_dir)
        return
    if args.mode == "postprocess":
        res = postprocess(args, out_dir)
        print(json.dumps(res, indent=2, default=_json_default))
        return

    # ---- run: build -> 2-GPU workers -> postprocess ------------------
    print("=== M2.5-v2 wheat full scan — orchestrate ===")
    print("\n[build] scan context")
    build_meta = build_context(args, out_dir)

    if args.canary_subgenome:
        assign = {0: (args.canary_subgenome,)}
        print(f"\n[canary] scanning only {args.canary_subgenome} on GPU 0")
    else:
        assign = GPU_ASSIGN

    print(f"\n[workers] GPU assignment: {assign}")
    procs = []
    self_py = sys.executable
    for gpu, sgs in assign.items():
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
        cmd = [self_py, str(Path(__file__).resolve()),
               "--mode", "worker", "--subgenomes", ",".join(sgs),
               "--trait", args.trait, "--chunk-size", str(args.chunk_size),
               "--maf-min", str(args.maf_min),
               "--call-rate-min", str(args.call_rate_min),
               "--out-dir", str(out_dir), "--backend", args.backend]
        log = out_dir / f"worker_gpu{gpu}.log"
        print(f"  launch GPU{gpu} {sgs} -> {log}")
        procs.append((gpu, subprocess.Popen(cmd, env=env,
                      stdout=open(log, "w"), stderr=subprocess.STDOUT)))
    failed = []
    for gpu, p in procs:
        if p.wait() != 0:
            failed.append(gpu)
    if failed:
        sys.exit(f"ERR: worker(s) on GPU {failed} failed — see worker_gpu*.log")

    print("\n[postprocess]")
    post = postprocess(args, out_dir)

    runtime = time.time() - t0
    acceptance = []

    def check(name, ok, msg=""):
        acceptance.append({"check": name, "passed": bool(ok), "message": msg})
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {msg}" if msg else ""))

    scanned = [sg for sg in SUBGENOMES
               if (out_dir / f"scan_{sg}.tsv.gz").exists()]
    check("scan_outputs_present", len(scanned) >= 1, f"subgenomes={scanned}")
    check("lambda_gc_finite", np.isfinite(post["lambda_gc_all"]),
          f"λ_GC={post['lambda_gc_all']:.4f}")
    check("markers_scanned", post["n_markers_total"] > 0,
          f"{post['n_markers_total']} markers")
    all_passed = all(c["passed"] for c in acceptance)

    summary = {
        "script": "run_m2_5_v2_wheat_scan.py", "milestone": "M2.5-v2",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "trait": args.trait, "total_runtime_sec": round(runtime, 1),
        "canary": args.canary_subgenome,
        "gpu_assignment": {str(k): v for k, v in assign.items()},
        **build_meta, **post,
        "acceptance": acceptance, "acceptance_all_passed": all_passed,
    }
    with open(out_dir / f"m2_5_v2_summary_{args.trait}.json", "w") as f:
        json.dump(summary, f, indent=2, default=_json_default)
    print(f"\nwrote {out_dir / f'm2_5_v2_summary_{args.trait}.json'}")
    print(f"\ntotal runtime {runtime:.0f}s  λ_GC={post['lambda_gc_all']:.4f}  "
          f"markers={post['n_markers_total']}")
    if all_passed:
        print("M2.5-v2 wheat full scan acceptance PASS")
    else:
        sys.exit("M2.5-v2 acceptance FAIL")


if __name__ == "__main__":
    main()
