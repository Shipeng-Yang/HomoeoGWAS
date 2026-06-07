#!/usr/bin/env python3
"""M3.4 species-general M3.2 known-QTL recovery + novel-locus extraction.

Wraps the wheat-only run_m3_2_wheat_qtl.py logic to accept ANY panel:
  - LOCO sumstats from `homoeogwas fit` in-memory mode
    (single TSV: results/phase3/m3_1_loco_v2/<panel>/<trait>/sumstats_<trait>_loco.tsv)
  - any known_qtl TSV (data/reference/<species>/known_qtl_<species>.tsv)
  - any subgenome list
  - any chrom-order convention

For cotton (assembly mismatch flagged earlier), gene_pos may not
match the panel exactly; the script still reports closest-lead per QTL,
but acceptance gates here only require the pipeline to complete; the
biological hit interpretation needs assembly verification.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
P_GENOME_WIDE = 5.0e-8
P_SUGGESTIVE = 5.0e-5
P_RECOVERY_RELAXED = 1.0e-4     # smaller panels: relaxed gate for known QTL hit
KNOWN_WINDOW_BP = 500_000
NOVEL_EXCLUSION_BP = 1_000_000
CLUMP_KB = 1_000_000


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def load_loco_sumstats(path: Path) -> pd.DataFrame:
    """Read CLI in-memory mode sumstats (single TSV, no gz)."""
    df = pd.read_csv(path, sep="\t")
    df["chrom"] = df["chrom"].astype(str)
    return df


def scan_known_windows(sumstats: pd.DataFrame, qtls: pd.DataFrame,
                         p_relaxed: float = P_RECOVERY_RELAXED) -> pd.DataFrame:
    """For each QTL, find lead SNP within ±window_bp."""
    rows = []
    for _, q in qtls.iterrows():
        wbp = int(q["window_bp"])
        qp = int(q["qtl_pos"])
        w = sumstats[(sumstats["chrom"] == q["chrom"]) &
                     (sumstats["pos"] >= qp - wbp) &
                     (sumstats["pos"] <= qp + wbp)]
        if len(w) == 0:
            rows.append({
                "qtl_name": q["qtl_name"], "qtl_family": q["qtl_family"],
                "gene_symbol": q["gene_symbol"], "gene_id": q["gene_id"],
                "chrom": q["chrom"], "qtl_pos": qp,
                "window_start": qp - wbp, "window_end": qp + wbp,
                "lead_snp": None, "lead_pos": None, "lead_dist_to_qtl": None,
                "lead_chi2": None, "lead_p": None,
                "n_snps_in_window": 0,
                "n_snps_in_window_p5e8": 0,
                "n_snps_in_window_p5e5": 0,
                "recovered_p5e5": False, "recovered_p5e8": False,
                "recovered_p_relaxed": False,
            })
            continue
        lead = w.loc[w["chi2"].idxmax()]
        rows.append({
            "qtl_name": q["qtl_name"], "qtl_family": q["qtl_family"],
            "gene_symbol": q["gene_symbol"], "gene_id": q["gene_id"],
            "chrom": q["chrom"], "qtl_pos": qp,
            "window_start": qp - wbp, "window_end": qp + wbp,
            "lead_snp": str(lead["snp_id"]),
            "lead_pos": int(lead["pos"]),
            "lead_dist_to_qtl": int(lead["pos"] - qp),
            "lead_chi2": float(lead["chi2"]),
            "lead_p": float(lead["p"]),
            "n_snps_in_window": int(len(w)),
            "n_snps_in_window_p5e8": int((w["p"] < P_GENOME_WIDE).sum()),
            "n_snps_in_window_p5e5": int((w["p"] < P_SUGGESTIVE).sum()),
            "recovered_p5e5": bool(lead["p"] < P_SUGGESTIVE),
            "recovered_p5e8": bool(lead["p"] < P_GENOME_WIDE),
            "recovered_p_relaxed": bool(lead["p"] < p_relaxed),
        })
    return pd.DataFrame(rows).sort_values(["chrom", "qtl_pos"]).reset_index(drop=True)


def collect_significant_seeds(sumstats: pd.DataFrame, p_thresh: float) -> pd.DataFrame:
    return sumstats[sumstats["p"] < p_thresh].sort_values("p").reset_index(drop=True)


def physical_clump(seeds: pd.DataFrame, clump_kb: int = CLUMP_KB) -> pd.DataFrame:
    if len(seeds) == 0:
        return seeds.assign(clump_size=[])
    leads = []
    remaining = seeds.sort_values("p").reset_index(drop=True)
    while len(remaining) > 0:
        lead = remaining.iloc[0]
        ch = lead["chrom"]
        po = int(lead["pos"])
        within = ((remaining["chrom"] == ch) &
                  (remaining["pos"] >= po - clump_kb) &
                  (remaining["pos"] <= po + clump_kb))
        clump = remaining.loc[within]
        leads.append({**lead.to_dict(), "clump_size": int(len(clump))})
        remaining = remaining.loc[~within].reset_index(drop=True)
    return pd.DataFrame(leads).sort_values(["chrom", "pos"]).reset_index(drop=True)


def annotate_known_distance(leads: pd.DataFrame, qtls: pd.DataFrame,
                              exclusion_bp: int = NOVEL_EXCLUSION_BP) -> pd.DataFrame:
    if len(leads) == 0:
        return leads.assign(nearest_known_qtl=[], dist_to_nearest_known_bp=[],
                              in_known_window_1mb=[])
    nearest, dist, inwin = [], [], []
    for _, r in leads.iterrows():
        chr_qtls = qtls[qtls["chrom"] == r["chrom"]]
        if len(chr_qtls) == 0:
            nearest.append(None)
            dist.append(None)
            inwin.append(False)
            continue
        d = (chr_qtls["qtl_pos"] - int(r["pos"])).abs()
        i = d.idxmin()
        nearest.append(str(chr_qtls.loc[i, "qtl_name"]))
        dist.append(int(d[i]))
        inwin.append(bool(d[i] <= exclusion_bp))
    return leads.assign(nearest_known_qtl=nearest,
                          dist_to_nearest_known_bp=dist,
                          in_known_window_1mb=inwin)


def main():
    ap = argparse.ArgumentParser(description="M3.4 species M3.2 QTL recovery")
    ap.add_argument("--panel", required=True)
    ap.add_argument("--trait", required=True)
    ap.add_argument("--sumstats", required=True,
                    help="LOCO sumstats TSV (CLI in-memory mode output)")
    ap.add_argument("--known-qtl", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"=== M3.2 species QTL recovery — panel={args.panel} trait={args.trait} ===")

    sumstats = load_loco_sumstats(Path(args.sumstats))
    qtls = pd.read_csv(args.known_qtl, sep="\t")
    qtls["chrom"] = qtls["chrom"].astype(str)
    print(f"[1] {len(sumstats)} sumstats SNPs, {len(qtls)} known QTLs")

    print(f"[2] known-QTL window scan (±{KNOWN_WINDOW_BP//1000} kb)")
    known_hits = scan_known_windows(sumstats, qtls)
    n_recov_p5e5 = int(known_hits["recovered_p5e5"].fillna(False).sum())
    n_recov_p5e8 = int(known_hits["recovered_p5e8"].fillna(False).sum())
    n_recov_relaxed = int(known_hits["recovered_p_relaxed"].fillna(False).sum())
    print(f"  recovered: {n_recov_relaxed}/{len(qtls)} at p<1e-4 (relaxed), "
          f"{n_recov_p5e5} at p<5e-5, {n_recov_p5e8} at p<5e-8")

    print(f"[3] genome-wide significant seeds (p<{P_GENOME_WIDE:.0e})")
    seeds = collect_significant_seeds(sumstats, P_GENOME_WIDE)
    print(f"  {len(seeds)} significant seeds")

    print(f"[4] physical clumping (±{CLUMP_KB//1000} kb)")
    leads = physical_clump(seeds)
    print(f"  {len(leads)} independent leads")

    leads = annotate_known_distance(leads, qtls)
    novel = leads[(leads["p"] < P_GENOME_WIDE) &
                  (~leads["in_known_window_1mb"])].copy() if len(leads) else leads
    print(f"  novel candidates (p<5e-8, dist>{NOVEL_EXCLUSION_BP//1000}kb to known): "
          f"{len(novel)}")

    # lambda_gc
    lam_gc = float(np.median(sumstats["chi2"].dropna()) / 0.4549364231195724)
    print(f"[5] λ_GC = {lam_gc:.4f}")

    # outputs
    out_hits = out_dir / "known_qtl_hits.tsv"
    out_leads = out_dir / "all_significant_leads.tsv"
    out_novel = out_dir / "new_locus_candidates.tsv"
    known_hits.to_csv(out_hits, sep="\t", index=False)
    leads.to_csv(out_leads, sep="\t", index=False)
    novel.to_csv(out_novel, sep="\t", index=False)
    print(f"  wrote {out_hits}")
    print(f"  wrote {out_leads}")
    print(f"  wrote {out_novel}")

    acceptance = []
    def check(name, ok, msg=""):
        acceptance.append({"check": name, "passed": bool(ok), "message": msg})
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {msg}" if msg else ""))
    print("\n[6] acceptance gates")
    check("lambda_gc_in_range", 0.80 <= lam_gc <= 1.20,
          f"λ_GC={lam_gc:.4f}")
    check("at_least_1_known_recovery_relaxed", n_recov_relaxed >= 1,
          f"{n_recov_relaxed}/{len(qtls)} known QTL at p<1e-4 (small panel relaxed)")
    check("scan_complete", len(sumstats) > 0,
          f"{len(sumstats)} sumstats rows")
    all_passed = all(c["passed"] for c in acceptance)

    summary = {
        "panel": args.panel, "trait": args.trait,
        "n_sumstats_snps": int(len(sumstats)),
        "n_known_qtls": int(len(qtls)),
        "n_known_recovered_p5e5": n_recov_p5e5,
        "n_known_recovered_p5e8": n_recov_p5e8,
        "n_known_recovered_p_relaxed": n_recov_relaxed,
        "n_significant_seeds": int(len(seeds)),
        "n_independent_leads": int(len(leads)),
        "n_novel_candidates": int(len(novel)),
        "lambda_gc": lam_gc,
        "thresholds": {
            "p_genome_wide": P_GENOME_WIDE, "p_suggestive": P_SUGGESTIVE,
            "known_window_bp": KNOWN_WINDOW_BP,
            "novel_exclusion_bp": NOVEL_EXCLUSION_BP, "clump_kb": CLUMP_KB,
        },
        "runtime_sec": round(time.time() - t0, 1),
        "outputs": {"known_qtl_hits": str(out_hits),
                     "all_significant_leads": str(out_leads),
                     "new_locus_candidates": str(out_novel)},
        "acceptance": acceptance, "acceptance_all_passed": all_passed,
    }
    summary_path = out_dir / f"m3_2_summary_{args.trait}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=_json_default)
    print(f"\nwrote {summary_path}")
    n_pass = sum(c["passed"] for c in acceptance)
    print(f"acceptance: {n_pass}/{len(acceptance)} gates "
          f"(runtime {summary['runtime_sec']}s)")
    if all_passed:
        print(f"M3.2 {args.panel}/{args.trait} acceptance PASS")
        return 0
    print(f"M3.2 {args.panel}/{args.trait} acceptance FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
