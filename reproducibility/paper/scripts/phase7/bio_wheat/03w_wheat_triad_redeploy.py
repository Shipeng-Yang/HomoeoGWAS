#!/usr/bin/env python3
"""Phase C wheat Step 3: re-deploy with CURATED 1:1:1 triads (HCTriads.csv,
Ramirez-Gonzalez 2018) replacing the position-rank proxy triads. Fixes Codex's
top framing risk. Reuses 02w deploy/GLS/ACAT/perm machinery verbatim.

ID mapping: HCTriads uses RefSeq v1.0 gene IDs (TraesCS7A01G243100); our wheat uses
v1.1 (TraesCS1A02G050600). Only the build tag differs (01 -> 02); numeric part is
identical. We map by substituting the build tag. Keep only triads whose A/B/D genes
fall on 1A/1B/1D and are all in our retained (>=3 SNP) gene sets.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/mnt/7302share/fast_ysp/U7_GWAS")
OUT = ROOT / "results/phase7/bio_wheat"
PHENO = ROOT / "data/processed/wheat/pheno_clean.tsv"
HCTRIADS = ROOT / "data/raw/expression/wheat/HCTriads.csv"

# load 02w as a module (numeric filename -> importlib)
_spec = importlib.util.spec_from_file_location(
    "wheat02", str(ROOT / "scripts/phase7/bio_wheat/02w_wheat_deploy.py"))
_w2 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_w2)


def _v10_to_v11(gid):
    """TraesCS7A01G243100 -> TraesCS7A02G243100 (build tag 01->02)."""
    # pattern: TraesCS<chr><build>G<num>; build is the 2 digits after chr letter
    i = gid.find("G", 8)  # 'TraesCS7A01G...' -> find the 'G' after build digits
    # locate '01' just before the final G-block: chr is e.g. '7A', then '01', then 'G'
    # robust: replace the '01' that immediately precedes 'G<digits>'
    if "01G" in gid:
        return gid.replace("01G", "02G", 1)
    return gid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["body", "flank"], default="flank")
    ap.add_argument("--trait", default="days_to_emerg")
    ap.add_argument("--cap", type=int, default=150)
    ap.add_argument("--perm-B", type=int, default=2000)
    ap.add_argument("--n-jobs", type=int, default=48)
    args = ap.parse_args()
    t0 = time.time()
    print(f"=== wheat CURATED-triad redeploy (1A-1B-1D, mode={args.mode}, trait={args.trait}) ===")

    # load per-subgenome ETL (same as 02w)
    sub_data = {}
    for s in ("A", "B", "D"):
        z = np.load(OUT / f"pilot_1{s}_{args.mode}.npz", allow_pickle=True)
        X = np.load(OUT / f"pilot_1{s}_{args.mode}_X.npy")
        gene_ids = z["gene_ids"].tolist(); snp_idx = z["snp_idx"].tolist()
        starts = z["starts"]; samples = z["samples"].tolist()
        order = np.argsort(starts)
        gene_ids = [gene_ids[i] for i in order]; snp_idx = [snp_idx[i] for i in order]
        sub_data[s] = dict(X=X, gene_ids=gene_ids, snp_idx=snp_idx, samples=samples,
                           gid2idx={g: i for i, g in enumerate(gene_ids)})
        print(f"  1{s}: {len(gene_ids)} retained genes")
    common = sub_data["A"]["samples"]

    # load curated 1:1:1 triads, restrict to 1A/1B/1D, map v1.0->v1.1, keep retained
    hc = pd.read_csv(HCTRIADS)
    hc = hc[hc["cardinality_abs"] == "1:1:1"].copy()
    kept = []
    for _, r in hc.iterrows():
        a = _v10_to_v11(str(r["A"])); b = _v10_to_v11(str(r["B"])); d = _v10_to_v11(str(r["D"]))
        # must be on chr1 of each subgenome AND in our retained sets
        if not (a.startswith("TraesCS1A") and b.startswith("TraesCS1B") and d.startswith("TraesCS1D")):
            continue
        if a in sub_data["A"]["gid2idx"] and b in sub_data["B"]["gid2idx"] and d in sub_data["D"]["gid2idx"]:
            kept.append((a, b, d))
    n_triad = len(kept)
    # how many chr1 triads existed before the retained filter
    chr1_total = sum(1 for _, r in hc.iterrows()
                     if _v10_to_v11(str(r["A"])).startswith("TraesCS1A")
                     and _v10_to_v11(str(r["B"])).startswith("TraesCS1B")
                     and _v10_to_v11(str(r["D"])).startswith("TraesCS1D"))
    print(f"  HCTriads 1:1:1 on chr1(A/B/D): {chr1_total}; retained (all 3 genes have >=3 SNP): {n_triad}")
    if n_triad < 10:
        print("  ABORT: too few curated triads with genotype coverage."); return

    gene_triads = [{"A": a, "B": b, "D": d} for a, b, d in kept]
    triad_idx = {s: [sub_data[s]["gid2idx"][g[i]] for g in kept]
                 for i, s in enumerate(("A", "B", "D"))}

    # burdens (capped), GRMs, phenotype align — reuse 02w helpers
    rng = np.random.default_rng(7)
    Bdict = {}
    for s in ("A", "B", "D"):
        cols = [_w2._block_burden_capped(sub_data[s]["X"], sub_data[s]["snp_idx"][gi], args.cap, rng)
                for gi in triad_idx[s]]
        Bdict[s] = _w2._scols_safe(np.column_stack(cols))
    KA = _w2._grm_from_X(sub_data["A"]["X"]); KB = _w2._grm_from_X(sub_data["B"]["X"])
    KD = _w2._grm_from_X(sub_data["D"]["X"])
    print(f"  burdens + GRMs built ({time.time()-t0:.1f}s)")

    ph = pd.read_csv(PHENO, sep="\t").set_index("sample")
    valid = [s for s in common if s in ph.index and pd.notna(ph.loc[s, args.trait])]
    idx_in = np.array([common.index(s) for s in valid]); n_t = len(valid)
    KA_t = KA[np.ix_(idx_in, idx_in)]; KB_t = KB[np.ix_(idx_in, idx_in)]; KD_t = KD[np.ix_(idx_in, idx_in)]
    KA_t = KA_t / (np.trace(KA_t) / n_t); KB_t = KB_t / (np.trace(KB_t) / n_t); KD_t = KD_t / (np.trace(KD_t) / n_t)
    Bd_t = {s: _w2._scols_safe(Bdict[s][idx_in]) for s in ("A", "B", "D")}
    y_raw = np.array([float(ph.loc[s, args.trait]) for s in valid])
    print(f"  trait={args.trait} n={n_t} G={n_triad} ({time.time()-t0:.1f}s)")

    out_INT = _w2.deploy(_w2._rank_int(y_raw), "INT", KA_t, KB_t, KD_t, Bd_t, gene_triads, args.n_jobs, args.perm_B)
    out_raw = _w2.deploy(y_raw, "raw", KA_t, KB_t, KD_t, Bd_t, gene_triads, args.n_jobs, args.perm_B)
    for lab, o in [("INT", out_INT), ("raw", out_raw)]:
        print(f"  [{lab}] triad ACAT obs={o['triad_acat_obs']:.4g} emp={o['triad_acat_emp']:.4g} "
              f"| σ̂ A={o['sigma_hat']['A']:.3f} B={o['sigma_hat']['B']:.3f} D={o['sigma_hat']['D']:.3f}")
        for tag in ("AB", "AD", "BD"):
            pw = o["pairwise"][tag]; pe = o["pairwise_emp"][tag]
            print(f"    {tag}: ACAT_obs={pw['acat']:.4g} emp={pe['acat_emp']:.4g} "
                  f"λ_obs={pw['lambda_gc']:.3f} λ_perm={pe['lambda_gc_perm_median']:.3f} nsig={pw['n_sig']}")

    full = dict(panel="wheat_watkins", mode=args.mode, trait=args.trait, cap=args.cap,
                triad_source="HCTriads_curated_1to1to1", n_triad=n_triad, chr1_total=chr1_total,
                perm_B=args.perm_B, INT_primary=out_INT, raw_sensitivity=out_raw,
                note=("wheat AABBDD CURATED-triad redeploy (1A-1B-1D). Triads = Ramirez-Gonzalez "
                      "2018 HCTriads.csv 1:1:1 (v1.0 IDs mapped 01->02 to our v1.1), restricted to "
                      "chr1 and our >=3-SNP retained genes. REPLACES the position-rank proxy triads "
                      "(Codex top framing risk). Same 3-pairwise + hexaploid-whitened GLS + triad "
                      "ACAT + B-perm as 02w. Single trait (emergence) = calibration check on real "
                      "homoeolog units."))
    fp = OUT / f"deploy_wheat_CURATED_{args.mode}.json"
    fp.write_text(json.dumps(full, indent=2, default=float))
    print(f"  wrote {fp} ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
