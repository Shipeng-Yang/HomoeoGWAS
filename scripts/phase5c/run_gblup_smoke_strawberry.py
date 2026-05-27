"""Phase 5c Week 1 smoke test — GBLUP CV on strawberry (Pincot 2018).

Quick (n_repeats=3) sanity run to verify the pipeline end-to-end on real
panel data before scaling up to all 4 panels with n_repeats=20.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from homoeogwas.gp import run_cv_gblup
from homoeogwas.grm import compute_grm
from homoeogwas.io import GenoChunk, load_bed_hardcall

ROOT = Path(__file__).resolve().parents[2]
BED_ROOT = ROOT / "data/processed/fragaria_ananassa"
PHENO_TSV = ROOT / "data/processed/strawberry/pheno_clean.tsv"
OUT_DIR = ROOT / "results/phase5c/strawberry_smoke"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    t0 = time.time()
    print("=== Phase 5c Week 1 strawberry smoke ===")

    pheno = pd.read_csv(PHENO_TSV, sep="\t")
    pheno = pheno.set_index("sample_id")
    print(f"  pheno rows: {len(pheno)}")

    grms: dict[str, np.ndarray] = {}
    sample_ids_per_sub: list[list[str]] = []
    snp_counts: dict[str, int] = {}
    beds_obj: dict[str, object] = {}
    for sg in ("A", "B", "C", "D"):
        bed_prefix = BED_ROOT / sg / "all"
        print(f"  [sub {sg}] load BED + compute GRM ...")
        ts = time.time()
        bed = load_bed_hardcall(bed_prefix)
        beds_obj[sg] = bed
        K, info = compute_grm(bed, maf_min=0.05)
        grms[sg] = K
        sample_ids_per_sub.append(list(bed.samples))
        n_smp, n_snp_raw = bed.dosage.shape
        n_snp_kept = info.get("n_variants_kept", n_snp_raw) or n_snp_raw
        snp_counts[sg] = int(n_snp_kept)
        print(f"    n_samples={n_smp}  n_snp_kept={n_snp_kept}  "
              f"GRM trace/n={np.trace(K)/n_smp:.3f}  ({time.time()-ts:.1f}s)")

    # Verify sample order identical across subgenomes
    ref_samples = sample_ids_per_sub[0]
    for sg, sids in zip(("A","B","C","D"), sample_ids_per_sub, strict=True):
        if sids != ref_samples:
            raise RuntimeError(f"sub {sg} sample order differs from sub A")
    print(f"  4 GRMs aligned on {len(ref_samples)} samples")

    # Align phenotype on BED sample order
    common = [s for s in ref_samples if s in pheno.index]
    if len(common) != len(ref_samples):
        print(f"  WARN: {len(ref_samples) - len(common)} BED samples missing in pheno")
    keep = [ref_samples.index(s) for s in common]
    keep = np.asarray(keep, dtype=np.int64)
    grms = {sg: K[np.ix_(keep, keep)] for sg, K in grms.items()}
    y = pheno.loc[common, "mean_score"].to_numpy(dtype=np.float64)
    X = np.ones((len(y), 1))
    n = len(y)
    print(f"  aligned n={n}  y mean={y.mean():.3f}  var={y.var():.3f}")

    # Build K_pool from CONCATENATED dosage (proper rrBLUP single-K baseline)
    print("\n=== build K_pool from concatenated dosage (proper rrBLUP) ===")
    concat_dosage = np.hstack([beds_obj[sg].dosage for sg in "ABCD"])
    concat_chunk = GenoChunk(
        samples=beds_obj["A"].samples,
        variant_ids=np.concatenate([beds_obj[sg].variant_ids for sg in "ABCD"]),
        chrom=np.concatenate([beds_obj[sg].chrom for sg in "ABCD"]),
        pos=np.concatenate([beds_obj[sg].pos for sg in "ABCD"]),
        dosage=concat_dosage,
    )
    K_pool, info_pool = compute_grm(concat_chunk, maf_min=0.05)
    K_pool = K_pool[np.ix_(keep, keep)]
    print(f"  K_pool n_snp = {info_pool.get('n_variants_kept', concat_dosage.shape[1])}")

    # Quick CV
    print("\n=== run_cv_gblup (n_folds=5, n_repeats=3) ===")
    ts = time.time()
    res = run_cv_gblup(
        y, X, grms,
        tiers=("tier0", "tier1", "tier2"),
        k_pool=K_pool,
        snp_counts=snp_counts,
        n_folds=5, n_repeats=3, seed=2026, n_starts=3,
        panel="strawberry_pincot2018", trait="mean_score",
        verbose=True,
    )
    print(f"  total CV runtime {time.time()-ts:.1f}s")

    # Summary
    print("\n=== Tier summary ===")
    for tname, ts_obj in res.tiers.items():
        print(f"  {tname:5}  r²={ts_obj.mean_r2:.4f}  r={ts_obj.mean_r:.4f}  "
              f"top10_enrich={ts_obj.mean_top10_enrichment:.2f}  "
              f"CI=[{ts_obj.ci_r2_low:.4f},{ts_obj.ci_r2_high:.4f}]")
    print("\n=== Δ vs tier0 ===")
    for tname, d in res.delta_vs_tier0.items():
        sig = "**" if d["significant_95"] else "  "
        print(f"  {tname:5} {sig} Δr²={d['delta_r2_mean']:+.4f}  "
              f"CI=[{d['ci_lo']:+.4f},{d['ci_hi']:+.4f}]")

    # Persist
    out_path = OUT_DIR / "gblup_smoke_summary.json"
    with out_path.open("w") as f:
        json.dump(res.to_dict(), f, indent=2, default=float)
    print(f"\nwrote {out_path}")
    print(f"total elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
