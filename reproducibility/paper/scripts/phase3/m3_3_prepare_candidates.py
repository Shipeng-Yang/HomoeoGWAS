#!/usr/bin/env python3
"""Phase 3 M3.3 — prepare candidate SNP set for DL-prior scoring.

Streams the LOCO sumstats once to extract all SNPs at p<P_MAX (default
1e-3), force-includes the 85 M3.2 novel candidate leads, the 13 known-QTL
gene sentinels (±100 kb), and any plink2 LD partners (r²≥0.2 within
±1 Mb of each STRICT_HIGH_CONFIDENCE lead). Harmonises with the per-SNP
BIM (REF/ALT) and the IWGSC v1.0 FASTA (REF base check).

Inputs:
  results/phase3/m3_1_loco_v2/wheat_watkins/scan_loco_{A,B,D}.tsv.gz
  results/phase3/m3_2_qtl/wheat_watkins/novel_loci_tiered.tsv
  data/reference/wheat/known_qtl_wheat.tsv
  data/processed/wheat_bed/{A,B,D}/all.bim
  /mnt/nvme/wheat_ref/iwgsc_refseqv1.0_all_chromosomes/<chrom>.fsa
                                                       (or single multi-fasta)

Outputs in results/phase3/m3_3_dl_prior/wheat_watkins/:
  candidates.tsv.gz      one row per SNP to score, with source tags
  candidates_summary.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
P_MAX_DEFAULT = 1.0e-3
LD_WINDOW_KB = 1000                             # ±1 Mb LD expansion per lead
LD_MIN_R2 = 0.20
KNOWN_SENTINEL_FLANK_BP = 100_000
PLINK2_MEMORY_MB = 8000


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serialisable: {type(o)}")


def stream_significant(loco_paths: dict[str, Path], p_max: float
                        ) -> pd.DataFrame:
    """One pass per subgenome; keep rows with p < p_max."""
    parts: list[pd.DataFrame] = []
    for sg, path in loco_paths.items():
        n_keep = 0
        for chunk in pd.read_csv(
                path, sep="\t",
                usecols=["snp_id","subgenome","chrom","pos",
                         "beta","se","chi2","p"],
                chunksize=1_000_000, low_memory=False):
            m = chunk["p"] < p_max
            if m.any():
                parts.append(chunk.loc[m].copy())
                n_keep += int(m.sum())
        print(f"  {sg}: {n_keep} SNPs at p<{p_max:.0e}")
    if not parts:
        return pd.DataFrame(columns=["snp_id","subgenome","chrom","pos",
                                       "beta","se","chi2","p"])
    return pd.concat(parts, ignore_index=True).sort_values(
        ["chrom","pos"]).reset_index(drop=True)


def expand_ld_partners(
    leads: pd.DataFrame, full_bed_root: Path, plink2_bin: str,
    out_dir: Path, ld_window_kb: int = LD_WINDOW_KB,
    min_r2: float = LD_MIN_R2,
) -> pd.DataFrame:
    """For each lead, plink2 --r2-unphased lead vs window; collect partners r²≥min_r2.

    Cached at {out_dir}/ld_partners_<chrom>_<pos>.vcor; SHA file of (lead_snp,
    window, min_r2) tuples for invalidation.
    """
    import hashlib
    rows: list[dict] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for _, lead in leads.iterrows():
        sg = lead["subgenome"]
        chrom = str(lead["chrom"])
        snp = str(lead["snp_id"])
        bfile = full_bed_root / sg / "all"
        out_prefix = out_dir / f"part_{snp}"
        vcor = Path(str(out_prefix) + ".vcor")
        sha_payload = f"{snp}|{ld_window_kb}|{min_r2}".encode()
        sha = hashlib.sha256(sha_payload).hexdigest()
        sha_path = out_dir / f"part_{snp}.sha256"
        cached = (vcor.exists() and vcor.stat().st_size > 0
                  and sha_path.exists() and sha_path.read_text().strip() == sha)
        if not cached:
            cmd = [plink2_bin,
                   "--bfile", str(bfile),
                   "--chr", chrom,
                   "--r2-unphased",
                   "--ld-snp", snp,
                   "--ld-window", "10000000",
                   "--ld-window-kb", str(ld_window_kb),
                   "--ld-window-r2", str(min_r2),
                   "--memory", str(PLINK2_MEMORY_MB),
                   "--threads", "4",
                   "--out", str(out_prefix)]
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True)
                sha_path.write_text(sha)
            except subprocess.CalledProcessError as e:
                print(f"  WARN: plink2 LD {snp}: {e.stderr[-200:] if e.stderr else 'fail'}")
                continue
        if not vcor.exists() or vcor.stat().st_size == 0:
            continue
        df = pd.read_csv(vcor, sep="\t")
        def _col(prefer, df=df):
            for p in prefer:
                for c in df.columns:
                    if c.upper() == p:
                        return c
            return None
        id_b = _col(["ID_B"]) or _col(["VARIANT_B"])
        pos_b = _col(["POS_B"]) or _col(["BP_B"])
        r2 = _col(["UNPHASED_R2", "R2"])
        if id_b is None or r2 is None:
            continue
        for _, r in df.iterrows():
            rows.append({
                "lead_snp": snp,
                "partner_snp": str(r[id_b]),
                "partner_pos": int(r[pos_b]) if pos_b else 0,
                "chrom": chrom,
                "subgenome": sg,
                "r2": float(r[r2]),
            })
    return pd.DataFrame(rows)


def load_bim_alleles(full_bed_root: Path,
                      snp_id_filter: set[str] | None = None
                      ) -> dict[str, tuple[str,str,str,int]]:
    """{snp_id -> (chrom, A1, A2, pos)} from the 3 subgenome BIMs.

    Vectorised; iterating 83M rows in Python takes >10 min, so we read each
    BIM with pandas, filter to ``snp_id_filter`` (when given), then use
    ``DataFrame.to_dict('index')`` for O(N) dict build over the filtered set.
    """
    out: dict[str, tuple[str,str,str,int]] = {}
    for sg in ("A","B","D"):
        bim = full_bed_root / sg / "all.bim"
        df = pd.read_csv(bim, sep="\t", header=None,
                          names=["chrom","snp_id","cm","pos","a1","a2"],
                          usecols=["chrom","snp_id","pos","a1","a2"],
                          dtype={"snp_id": str, "chrom": str,
                                 "a1": str, "a2": str, "pos": np.int64})
        if snp_id_filter is not None:
            df = df[df["snp_id"].isin(snp_id_filter)]
        for snp_id, chrom, a1, a2, pos in zip(
                df["snp_id"].to_numpy(),
                df["chrom"].to_numpy(),
                df["a1"].to_numpy(),
                df["a2"].to_numpy(),
                df["pos"].to_numpy(), strict=True):
            out[str(snp_id)] = (str(chrom), str(a1), str(a2), int(pos))
    return out


def harmonize_with_fasta(
    cand: pd.DataFrame, fasta_path: Path, allele_map: dict,
) -> pd.DataFrame:
    """Add ref_fasta, alt_fasta, ref_match (bool) using pysam.FastaFile."""
    import pysam
    fa = pysam.FastaFile(str(fasta_path))
    refs: list[str | None] = []
    alts: list[str | None] = []
    matches: list[str] = []
    for _, r in cand.iterrows():
        sid = str(r["snp_id"])
        if sid not in allele_map:
            refs.append(None)
            alts.append(None)
            matches.append("BIM_MISSING")
            continue
        ch, a1, a2, pos = allele_map[sid]
        try:
            fasta_base = fa.fetch(ch, pos - 1, pos).upper()
        except (KeyError, ValueError):
            refs.append(None)
            alts.append(None)
            matches.append("FASTA_KEYERR")
            continue
        if fasta_base not in ("A","C","G","T"):
            refs.append(None)
            alts.append(None)
            matches.append("FASTA_NON_ACGT")
            continue
        # plink convention: A1 is the minor allele (counted), A2 is reference
        # but the BIM may not match the actual fasta REF. So we pick the
        # allele that matches FASTA as ref_fasta, the other as alt_fasta.
        a1u, a2u = a1.upper(), a2.upper()
        if fasta_base == a1u and a2u in ("A","C","G","T"):
            refs.append(a1u)
            alts.append(a2u)
            matches.append("A1_IS_REF")
        elif fasta_base == a2u and a1u in ("A","C","G","T"):
            refs.append(a2u)
            alts.append(a1u)
            matches.append("A2_IS_REF")
        else:
            refs.append(fasta_base)
            alts.append(None)
            matches.append("REF_MISMATCH")
    fa.close()
    cand = cand.copy()
    cand["ref_fasta"] = refs
    cand["alt_fasta"] = alts
    cand["ref_match_status"] = matches
    return cand


def known_qtl_sentinels(known_path: Path, full_bed_root: Path,
                         flank_bp: int = KNOWN_SENTINEL_FLANK_BP,
                         per_qtl: int = 10) -> pd.DataFrame:
    qtls = pd.read_csv(known_path, sep="\t")
    out: list[dict] = []
    for sg in ("A","B","D"):
        sg_q = qtls[qtls["chrom"].str.endswith(sg)]
        if len(sg_q) == 0:
            continue
        bim = pd.read_csv(full_bed_root / sg / "all.bim", sep="\t", header=None,
                           names=["chrom","snp_id","cm","pos","a1","a2"],
                           usecols=["chrom","snp_id","pos"])
        for _, q in sg_q.iterrows():
            mask = ((bim["chrom"] == q["chrom"])
                    & (bim["pos"] >= q["qtl_pos"] - flank_bp)
                    & (bim["pos"] <= q["qtl_pos"] + flank_bp))
            window = bim.loc[mask]
            if len(window) == 0:
                continue
            step = max(1, len(window) // per_qtl)
            picks = window.iloc[::step].head(per_qtl)
            for _, p in picks.iterrows():
                out.append({
                    "snp_id": str(p["snp_id"]),
                    "subgenome": sg,
                    "chrom": str(p["chrom"]),
                    "pos": int(p["pos"]),
                    "qtl_name": q["qtl_name"],
                })
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser(description="M3.3 prepare candidate SNPs")
    ap.add_argument("--trait", default="days_to_emerg")
    ap.add_argument("--out-dir", default=str(
        ROOT / "results/phase3/m3_3_dl_prior/wheat_watkins"))
    ap.add_argument("--loco-dir", default=str(
        ROOT / "results/phase3/m3_1_loco_v2/wheat_watkins"))
    ap.add_argument("--m3-2-dir", default=str(
        ROOT / "results/phase3/m3_2_qtl/wheat_watkins"))
    ap.add_argument("--known-qtl", default=str(
        ROOT / "data/reference/wheat/known_qtl_wheat.tsv"))
    ap.add_argument("--full-bed-root", default=str(
        ROOT / "data/processed/wheat_bed"))
    ap.add_argument("--fasta", default=
        "/mnt/nvme/wheat_ref/iwgsc_refseqv1.0_all_chromosomes/"
        "iwgsc_refseqv1.0_all_chromosomes.fa")
    ap.add_argument("--plink2", default=
        "/home/yys05/.local/share/mamba/envs/polygwas-cpu/bin/plink2")
    ap.add_argument("--p-max", type=float, default=P_MAX_DEFAULT)
    ap.add_argument("--ld-leads-tier", default="STRICT_HIGH_CONFIDENCE",
                    choices=["STRICT_HIGH_CONFIDENCE", "HIGH_CONFIDENCE", "ALL"],
                    help="which novel leads get LD expansion")
    ap.add_argument("--pilot", action="store_true",
                    help="pilot mode: skip p<p_max stream + LD expansion, "
                         "use only forced leads + known QTL sentinels")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print(f"=== M3.3 prepare candidates — trait={args.trait} ===")
    if args.pilot:
        print("  PILOT mode: only forced leads + known-QTL sentinels")

    # 1. Forced novel leads
    novel = pd.read_csv(Path(args.m3_2_dir) / "novel_loci_tiered.tsv", sep="\t")
    novel_subset = novel.copy()
    if args.ld_leads_tier == "STRICT_HIGH_CONFIDENCE":
        ld_leads = novel[novel["strict_high_confidence"].eq(True)].copy()
    elif args.ld_leads_tier == "HIGH_CONFIDENCE":
        ld_leads = novel[novel["tier"] == "HIGH_CONFIDENCE"].copy()
    else:
        ld_leads = novel.copy()
    print(f"[1] novel leads: {len(novel)} total, LD-expand {len(ld_leads)} "
          f"({args.ld_leads_tier})")
    forced_leads = novel_subset[["lead_snp","subgenome","chrom","pos","beta","se","chi2","p"]].rename(
        columns={"lead_snp": "snp_id"})
    forced_leads["source"] = "novel_lead"

    # 2. Known QTL sentinels
    print(f"[2] known QTL sentinels (±{KNOWN_SENTINEL_FLANK_BP//1000}kb)")
    sentinels = known_qtl_sentinels(
        Path(args.known_qtl), Path(args.full_bed_root))
    print(f"  picked {len(sentinels)} sentinel SNPs across "
          f"{sentinels['qtl_name'].nunique()} QTLs")
    sentinels["source"] = "known_qtl_sentinel"

    # 3. Significant SNPs (p < p_max)
    loco_paths = {
        sg: Path(args.loco_dir) / f"scan_loco_{sg}.tsv.gz" for sg in "ABD"
    }
    if args.pilot:
        sig = pd.DataFrame()
        print("[3] (pilot) skipping p<p_max significant pass")
    else:
        print(f"[3] streaming LOCO sumstats: p < {args.p_max:.0e}")
        sig = stream_significant(loco_paths, args.p_max)
        sig["source"] = "loco_p_max"
        print(f"  {len(sig)} SNPs at p<{args.p_max:.0e}")

    # 4. LD partners around STRICT_HIGH_CONFIDENCE leads
    if args.pilot or len(ld_leads) == 0:
        ld_partners = pd.DataFrame()
        print("[4] (pilot or no LD leads) skipping LD expansion")
    else:
        print(f"[4] LD expansion (r²≥{LD_MIN_R2}, ±{LD_WINDOW_KB} kb) for "
              f"{len(ld_leads)} {args.ld_leads_tier} leads")
        ld_partners = expand_ld_partners(
            ld_leads.rename(columns={"lead_snp": "snp_id"}),
            Path(args.full_bed_root), args.plink2, out_dir / "ld_cache")
        print(f"  {len(ld_partners)} partner pairs collected")

    # 5. Merge into unique SNP set
    pieces = []
    if len(sig) > 0:
        pieces.append(sig[["snp_id","subgenome","chrom","pos","beta","se","chi2","p","source"]])
    if len(forced_leads) > 0:
        pieces.append(forced_leads)
    if len(sentinels) > 0:
        pieces.append(sentinels[["snp_id","subgenome","chrom","pos","source"]].assign(
            beta=np.nan, se=np.nan, chi2=np.nan, p=np.nan))
    if len(ld_partners) > 0:
        # add unique partner SNPs
        partners_df = ld_partners[["partner_snp","chrom","subgenome","partner_pos","r2"]].rename(
            columns={"partner_snp":"snp_id","partner_pos":"pos"})
        partners_df = partners_df.assign(
            beta=np.nan, se=np.nan, chi2=np.nan, p=np.nan, source="ld_partner")
        pieces.append(partners_df[["snp_id","subgenome","chrom","pos","beta","se","chi2","p","source"]])
    all_cand = pd.concat(pieces, ignore_index=True)
    # collapse duplicates: keep min(p) and combine source tags
    all_cand["p_safe"] = all_cand["p"].fillna(1.0)
    agg = all_cand.sort_values("p_safe").drop_duplicates("snp_id", keep="first")
    # rebuild source as semicolon-joined unique values per snp
    src_map = all_cand.groupby("snp_id")["source"].apply(
        lambda s: ";".join(sorted(set(s)))).to_dict()
    agg["source"] = agg["snp_id"].map(src_map)
    agg = agg.drop(columns=["p_safe"]).sort_values(["chrom","pos"]).reset_index(drop=True)
    print(f"[5] merged unique candidates: {len(agg)} (sources: "
          f"{agg['source'].value_counts().head(8).to_dict()})")

    # 6. Add BIM alleles + FASTA REF harmonisation
    print(f"[6] BIM alleles + FASTA REF check (fasta={args.fasta})")
    snp_filter = set(agg["snp_id"].astype(str))
    allele_map = load_bim_alleles(Path(args.full_bed_root),
                                    snp_id_filter=snp_filter)
    print(f"  loaded {len(allele_map)}/{len(snp_filter)} BIM alleles")
    if not Path(args.fasta).exists():
        print(f"  ERR: fasta missing: {args.fasta} — please unzip first")
        sys.exit(2)
    final = harmonize_with_fasta(agg, Path(args.fasta), allele_map)
    n_match = int((final["ref_match_status"].isin(["A1_IS_REF","A2_IS_REF"])).sum())
    print(f"  REF match: {n_match}/{len(final)} "
          f"({100*n_match/max(len(final),1):.1f}%)")
    print(f"  mismatch breakdown: {final['ref_match_status'].value_counts().to_dict()}")

    # 7. Save
    out_path = out_dir / "candidates.tsv.gz"
    final.to_csv(out_path, sep="\t", index=False, compression="gzip")
    print(f"[7] wrote {out_path}")

    summary = {
        "trait": args.trait,
        "pilot": args.pilot,
        "p_max": args.p_max,
        "ld_leads_tier": args.ld_leads_tier,
        "ld_window_kb": LD_WINDOW_KB,
        "ld_min_r2": LD_MIN_R2,
        "n_total_candidates": int(len(final)),
        "n_p_lt_pmax": int(len(sig)),
        "n_forced_leads": int(len(forced_leads)),
        "n_sentinels": int(len(sentinels)),
        "n_ld_partners": int(len(ld_partners)),
        "ref_match_breakdown": final["ref_match_status"].value_counts().to_dict(),
        "n_ref_matched": n_match,
        "source_breakdown": final["source"].value_counts().head(20).to_dict(),
        "runtime_sec": round(time.time() - t0, 1),
        "outputs": {"candidates": str(out_path)},
    }
    summary_path = out_dir / "candidates_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=_json_default)
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
