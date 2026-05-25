"""Phase 5c Week 2 — panel-general GBLUP CV driver.

Runs the Tier 0 / Tier 1 / Tier 2 GBLUP CV pipeline on any of the four
HomoeoGWAS panels and writes a paper Table 1 row.

Usage example:
    python scripts/phase5c/run_gblup_panel.py \
        --panel wheat_watkins --trait days_to_emerg \
        --subgenomes A,B,D \
        --bed-root results/phase2/m2_4_5/wheat_watkins/pruned \
        --bed-name "{sub}"                            \
        --pheno data/processed/wheat/pheno_clean.tsv \
        --sample-col sample \
        --out-dir results/phase5c/wheat_watkins/days_to_emerg \
        --n-repeats 20 --n-folds 5
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from homoeogwas.gp import run_cv_gblup
from homoeogwas.grm import compute_grm
from homoeogwas.io import GenoChunk, load_bed_hardcall


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", required=True)
    ap.add_argument("--trait", required=True)
    ap.add_argument("--subgenomes", required=True,
                    help="comma-separated subgenome IDs, e.g. A,B,D or A,D")
    ap.add_argument("--bed-root", required=True)
    ap.add_argument("--bed-name", default="{sub}/all",
                    help="bed prefix template; substitute {sub}. "
                         "e.g. '{sub}/all' for the std layout, '{sub}' if "
                         "bed prefix is just <root>/A, <root>/B …")
    ap.add_argument("--pheno", required=True)
    ap.add_argument("--sample-col", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--n-repeats", type=int, default=20)
    ap.add_argument("--n-starts", type=int, default=5)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--maf-min", type=float, default=0.01)
    ap.add_argument("--tiers", default="tier0,tier1,tier2")
    return ap.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    subgenomes = [s.strip() for s in args.subgenomes.split(",") if s.strip()]
    tiers = tuple(t.strip() for t in args.tiers.split(",") if t.strip())

    t_start = time.time()
    print(f"=== GBLUP CV — panel={args.panel} trait={args.trait}  "
          f"subgenomes={subgenomes}  tiers={tiers} ===")
    bed_root = Path(args.bed_root)

    # Load each subgenome BED + compute GRM (in memory)
    beds_obj = {}
    grms = {}
    snp_counts = {}
    samples_per_sub = {}
    for sg in subgenomes:
        bed_prefix = bed_root / args.bed_name.format(sub=sg)
        print(f"  [sub {sg}] {bed_prefix}")
        ts = time.time()
        bed = load_bed_hardcall(bed_prefix)
        K, info = compute_grm(bed, maf_min=args.maf_min)
        beds_obj[sg] = bed
        grms[sg] = K
        samples_per_sub[sg] = np.asarray(bed.samples)
        m_kept = info.get("n_variants_kept", bed.dosage.shape[1]) or bed.dosage.shape[1]
        snp_counts[sg] = int(m_kept)
        print(f"    n_smp={K.shape[0]}  m_kept={m_kept}  "
              f"GRM trace/n={np.trace(K)/K.shape[0]:.3f}  ({time.time()-ts:.1f}s)")

    # Verify all sub samples align (intersect if needed)
    ref_samples = list(samples_per_sub[subgenomes[0]])
    for sg in subgenomes[1:]:
        if list(samples_per_sub[sg]) != ref_samples:
            common = sorted(set(ref_samples) & set(samples_per_sub[sg].tolist()))
            print(f"  WARN: sub {sg} sample order differs — intersect to {len(common)} samples")
            ref_samples = sorted(set(ref_samples) & set(samples_per_sub[sg].tolist()))
    n_full = len(ref_samples)

    # Map per-sub GRM to common-sample ordering (if intersect happened)
    aligned_grms = {}
    for sg in subgenomes:
        s_local = samples_per_sub[sg].tolist()
        idx = np.asarray([s_local.index(s) for s in ref_samples], dtype=np.int64)
        aligned_grms[sg] = grms[sg][np.ix_(idx, idx)]
    grms = aligned_grms

    # Build K_pool by horizontal concat of dosages
    concat_dosage = np.hstack([beds_obj[sg].dosage for sg in subgenomes])
    # If samples are aligned across subs (typical), can use ref_samples; else first sub
    concat_chunk = GenoChunk(
        samples=beds_obj[subgenomes[0]].samples,
        variant_ids=np.concatenate([beds_obj[sg].variant_ids for sg in subgenomes]),
        chrom=np.concatenate([beds_obj[sg].chrom for sg in subgenomes]),
        pos=np.concatenate([beds_obj[sg].pos for sg in subgenomes]),
        dosage=concat_dosage,
    )
    ts = time.time()
    K_pool, info_pool = compute_grm(concat_chunk, maf_min=args.maf_min)
    n_pool_snp = info_pool.get("n_variants_kept", concat_dosage.shape[1]) or concat_dosage.shape[1]
    print(f"  K_pool: n_snp={n_pool_snp}  ({time.time()-ts:.1f}s)")
    # align K_pool to ref_samples
    s_local = list(beds_obj[subgenomes[0]].samples)
    idx = np.asarray([s_local.index(s) for s in ref_samples], dtype=np.int64)
    K_pool = K_pool[np.ix_(idx, idx)]

    # Pheno
    pheno = pd.read_csv(args.pheno, sep="\t")
    pheno = pheno.set_index(args.sample_col)
    common = [s for s in ref_samples if s in pheno.index]
    print(f"  pheno overlap: {len(common)}/{n_full}")
    if not common:
        raise SystemExit("ERR: no overlap between BED samples and pheno")
    keep = np.asarray([ref_samples.index(s) for s in common], dtype=np.int64)
    grms = {sg: K[np.ix_(keep, keep)] for sg, K in grms.items()}
    K_pool = K_pool[np.ix_(keep, keep)]
    y = pheno.loc[common, args.trait].to_numpy(dtype=np.float64)
    n_nan = int(np.isnan(y).sum())
    if n_nan:
        print(f"  drop {n_nan} samples with missing {args.trait}")
        finite = ~np.isnan(y)
        keep2 = np.where(finite)[0]
        y = y[keep2]
        grms = {sg: K[np.ix_(keep2, keep2)] for sg, K in grms.items()}
        K_pool = K_pool[np.ix_(keep2, keep2)]
    X = np.ones((len(y), 1))
    n_final = len(y)
    print(f"  final n={n_final}  y mean={y.mean():.3f}  var={y.var():.3f}")

    # Run CV
    print(f"\n=== run_cv_gblup n_folds={args.n_folds} n_repeats={args.n_repeats} ===")
    ts = time.time()
    res = run_cv_gblup(
        y, X, grms,
        tiers=tiers,
        k_pool=K_pool,
        snp_counts=snp_counts,
        n_folds=args.n_folds, n_repeats=args.n_repeats,
        seed=args.seed, n_starts=args.n_starts,
        panel=args.panel, trait=args.trait, verbose=True,
    )
    print(f"\nCV runtime {time.time()-ts:.1f}s")

    # Summary print
    print(f"\n=== Tier summary ===")
    for tname, ts_obj in res.tiers.items():
        print(f"  {tname:5}  r²={ts_obj.mean_r2:.4f}  r={ts_obj.mean_r:.4f}  "
              f"top10_enrich={ts_obj.mean_top10_enrichment:.2f}  "
              f"CI=[{ts_obj.ci_r2_low:+.4f},{ts_obj.ci_r2_high:+.4f}]")
    print(f"\n=== Δ vs tier0 ===")
    for tname, d in res.delta_vs_tier0.items():
        sig = "**" if d["significant_95"] else "  "
        print(f"  {tname:5} {sig} Δr²={d['delta_r2_mean']:+.4f}  "
              f"CI=[{d['ci_lo']:+.4f},{d['ci_hi']:+.4f}]")

    # Persist
    out_path = out_dir / f"gblup_{args.panel}_{args.trait}.json"
    with out_path.open("w") as f:
        json.dump({
            "panel": args.panel, "trait": args.trait,
            "n_samples": n_final, "subgenomes": subgenomes,
            "snp_counts": snp_counts,
            "n_pool_snp": n_pool_snp,
            "n_folds": args.n_folds, "n_repeats": args.n_repeats,
            "result": res.to_dict(),
        }, f, indent=2, default=float)
    print(f"\nwrote {out_path}")
    print(f"total elapsed: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
