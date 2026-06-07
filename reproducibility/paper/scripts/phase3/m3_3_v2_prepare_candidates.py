#!/usr/bin/env python3
"""M3.3 v2 — panel-general candidate SNP preparation for DL-prior scoring.

Generalises scripts/phase3/m3_3_prepare_candidates.py (wheat-only) to any
species/panel. Differences:
  - subgenome set comes from --subgenomes (e.g. "A,C" for rapeseed,
    "A,D" for cotton, "A,B,D" for wheat)
  - LOCO sumstats expected as the homoeogwas CLI in-memory mode output:
    a single TSV (non-gzipped), columns
        snp_id  subgenome  chrom  pos  beta  se  chi2  p  n_obs  maf  call_rate
  - leads TSV is M3.2's `new_locus_candidates.tsv` (panel-general output);
    wheat-style `novel_loci_tiered.tsv` (with strict_high_confidence col)
    is also accepted automatically
  - FASTA chrom names may differ from panel-chrom names; --chrom-map TSV
    (panel_chrom, fasta_chrom, subgenome) is used at the FASTA fetch step

Outputs (same schema as wheat v1, in --out-dir):
  candidates.tsv.gz       one row per SNP to score
  candidates_summary.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

P_MAX_DEFAULT = 1.0e-3
LD_WINDOW_KB = 1000
LD_MIN_R2 = 0.20
KNOWN_SENTINEL_FLANK_BP = 100_000
PLINK2_MEMORY_MB = 8000
SENTINELS_PER_QTL = 10


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


def load_inmemory_sumstats(path: Path, p_max: float) -> pd.DataFrame:
    """Load CLI in-memory mode sumstats (single TSV) and filter p<p_max."""
    print(f"  reading {path}")
    df = pd.read_csv(path, sep="\t",
                     usecols=["snp_id","subgenome","chrom","pos",
                              "beta","se","chi2","p"],
                     dtype={"snp_id": str, "chrom": str},
                     low_memory=False)
    n0 = len(df)
    df = df[df["p"] < p_max].copy()
    print(f"  {n0} SNPs total → {len(df)} at p<{p_max:.0e}")
    return df.sort_values(["chrom","pos"]).reset_index(drop=True)


def load_leads(path: Path, ld_leads_tier: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (all_leads, ld_expansion_leads).
    Accepts either v2 wheat-style novel_loci_tiered.tsv with
    strict_high_confidence/tier columns, or panel-general new_locus_candidates.tsv.
    """
    df = pd.read_csv(path, sep="\t")
    # Determine schema:
    has_strict_col = "strict_high_confidence" in df.columns
    has_tier_col = "tier" in df.columns
    has_lead_snp = "lead_snp" in df.columns
    if has_lead_snp:
        df = df.rename(columns={"lead_snp": "snp_id"})
    elif "snp_id" not in df.columns:
        raise ValueError(f"leads TSV {path} missing lead_snp/snp_id column")

    if ld_leads_tier == "STRICT_HIGH_CONFIDENCE" and has_strict_col:
        ld = df[df["strict_high_confidence"].eq(True)].copy()
    elif ld_leads_tier == "HIGH_CONFIDENCE" and has_tier_col:
        ld = df[df["tier"] == "HIGH_CONFIDENCE"].copy()
    elif ld_leads_tier == "ALL":
        ld = df.copy()
    else:
        # fallback: panel-general leads have no tier => use all (e.g. cotton/horvath)
        print(f"  WARN: leads TSV has no '{ld_leads_tier}' column; using ALL leads")
        ld = df.copy()
    return df, ld


def expand_ld_partners(leads: pd.DataFrame, bed_root: Path, plink2_bin: str,
                       out_dir: Path, ld_window_kb: int = LD_WINDOW_KB,
                       min_r2: float = LD_MIN_R2) -> pd.DataFrame:
    rows: list[dict] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for _, lead in leads.iterrows():
        sg = lead["subgenome"]
        chrom = str(lead["chrom"])
        snp = str(lead["snp_id"])
        bfile = bed_root / sg / "all"
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
                "lead_snp": snp, "partner_snp": str(r[id_b]),
                "partner_pos": int(r[pos_b]) if pos_b else 0,
                "chrom": chrom, "subgenome": sg, "r2": float(r[r2]),
            })
    return pd.DataFrame(rows)


def load_bim_alleles(bed_root: Path, subgenomes: list[str],
                     snp_id_filter: set[str] | None = None):
    out: dict[str, tuple[str,str,str,int]] = {}
    for sg in subgenomes:
        bim = bed_root / sg / "all.bim"
        if not bim.exists():
            print(f"  WARN: missing BIM {bim}; skipping subgenome {sg}")
            continue
        df = pd.read_csv(bim, sep="\t", header=None,
                         names=["chrom","snp_id","cm","pos","a1","a2"],
                         usecols=["chrom","snp_id","pos","a1","a2"],
                         dtype={"snp_id": str, "chrom": str,
                                "a1": str, "a2": str, "pos": np.int64})
        if snp_id_filter is not None:
            df = df[df["snp_id"].isin(snp_id_filter)]
        for snp_id, chrom, a1, a2, pos in zip(
                df["snp_id"].to_numpy(), df["chrom"].to_numpy(),
                df["a1"].to_numpy(), df["a2"].to_numpy(),
                df["pos"].to_numpy(), strict=True):
            out[str(snp_id)] = (str(chrom), str(a1), str(a2), int(pos))
    return out


def harmonize_with_fasta(cand: pd.DataFrame, fasta_path: Path,
                         allele_map: dict, chrom_map: dict) -> pd.DataFrame:
    """Add ref_fasta, alt_fasta, ref_match_status using pysam.
    chrom_map: panel_chrom -> fasta_chrom (e.g. chrA01 -> NC_027757.2).

    FASTA fetch uses BIM-derived (chrom,pos), but the
    cand DataFrame's chrom/pos comes from sumstats/leads/sentinels — if they
    disagree (plink2 normalisation, VCF re-encode, etc.) we'd silently score
    the wrong locus. We assert agreement here so the failure is loud.
    """
    import pysam
    fa = pysam.FastaFile(str(fasta_path))
    refs, alts, matches, palindromes = [], [], [], []
    n_pos_disagree = 0
    COMP = {"A": "T", "T": "A", "C": "G", "G": "C"}
    for _, r in cand.iterrows():
        sid = str(r["snp_id"])
        if sid not in allele_map:
            refs.append(None)
            alts.append(None)
            matches.append("BIM_MISSING")
            palindromes.append(False)
            continue
        panel_ch, a1, a2, pos = allele_map[sid]
        # Sanity: cand and BIM should agree on (chrom, pos) for this snp_id
        cand_ch = str(r["chrom"])
        cand_pos = int(r["pos"]) if pd.notna(r["pos"]) else -1
        if cand_ch != panel_ch or (cand_pos > 0 and cand_pos != pos):
            n_pos_disagree += 1
            refs.append(None)
            alts.append(None)
            matches.append("BIM_CAND_DISAGREE")
            palindromes.append(False)
            continue
        fasta_ch = chrom_map.get(panel_ch, panel_ch)
        try:
            fasta_base = fa.fetch(fasta_ch, pos - 1, pos).upper()
        except (KeyError, ValueError):
            refs.append(None)
            alts.append(None)
            matches.append("FASTA_KEYERR")
            palindromes.append(False)
            continue
        if fasta_base not in ("A","C","G","T"):
            refs.append(None)
            alts.append(None)
            matches.append("FASTA_NON_ACGT")
            palindromes.append(False)
            continue
        a1u, a2u = a1.upper(), a2.upper()
        both_acgt = a1u in COMP and a2u in COMP
        is_pal = both_acgt and ({a1u, a2u} == {"A", "T"} or {a1u, a2u} == {"C", "G"})
        # Direct match (priority): FASTA + strand base equals one VCF allele.
        # Palindromic A/T or C/G SNPs always resolve here (ref=fasta_base is
        # correct for scoring even though the design strand is unknowable).
        if fasta_base == a1u and a2u in COMP:
            refs.append(a1u)
            alts.append(a2u)
            matches.append("A1_IS_REF")
        elif fasta_base == a2u and a1u in COMP:
            refs.append(a2u)
            alts.append(a1u)
            matches.append("A2_IS_REF")
        # Reverse-complement: array allele recorded on the strand opposite the
        # FASTA. ref stays = fasta_base (+ strand, so seq[center]==ref holds);
        # the + strand alt is the complement of the OTHER VCF allele. beta/se/p
        # are untouched — this is allele-frame harmonisation, not an effect flip.
        # NOTE: beta stays tied to the ORIGINAL GWAS effect
        # allele. This is safe because the only downstream consumer (the DL-prior
        # fusion m3_3_fuse_and_evaluate.py) ranks by |z(-log10 p)| + beta_w*z(|LLR|),
        # which is sign-agnostic in the GWAS term and uses |LLR| — neither
        # reinterprets beta as the (RC-harmonised) ALT effect. Any future code
        # that reads beta as an allele-specific effect MUST re-derive the effect
        # allele from ref_fasta/alt_fasta, not assume a1.
        elif both_acgt and COMP[fasta_base] == a1u:
            refs.append(fasta_base)
            alts.append(COMP[a2u])
            matches.append("A1_IS_REF_RC")
        elif both_acgt and COMP[fasta_base] == a2u:
            refs.append(fasta_base)
            alts.append(COMP[a1u])
            matches.append("A2_IS_REF_RC")
        else:
            refs.append(fasta_base)
            alts.append(None)
            matches.append("REF_MISMATCH")
        palindromes.append(is_pal)
    fa.close()
    cand = cand.copy()
    cand["ref_fasta"] = refs
    cand["alt_fasta"] = alts
    cand["ref_match_status"] = matches
    cand["is_palindromic"] = palindromes
    cand["fasta_chrom"] = cand["chrom"].map(lambda c: chrom_map.get(str(c), str(c)))
    if n_pos_disagree:
        print(f"  WARN: {n_pos_disagree} candidates dropped (BIM_CAND_DISAGREE) — "
              f"cand chrom/pos did not match BIM lookup for that snp_id")
    return cand


def known_qtl_sentinels_general(known_path: Path, bed_root: Path,
                                subgenomes: list[str],
                                flank_bp: int = KNOWN_SENTINEL_FLANK_BP,
                                per_qtl: int = SENTINELS_PER_QTL,
                                require_qtl_chrom: bool = True) -> pd.DataFrame:
    """Pick sentinel SNPs from BIMs around each known QTL chrom/pos.
    Skips QTL rows with chrom=='NA' (DROPPED) or qtl_pos<0.
    """
    qtls = pd.read_csv(known_path, sep="\t")
    qtls = qtls[(qtls["chrom"].astype(str) != "NA") & (qtls["qtl_pos"] >= 0)].copy()
    qtls["chrom"] = qtls["chrom"].astype(str)
    print(f"  {len(qtls)} known QTLs with non-dropped coords")
    out: list[dict] = []
    for sg in subgenomes:
        bim = bed_root / sg / "all.bim"
        if not bim.exists():
            continue
        bim_df = pd.read_csv(bim, sep="\t", header=None,
                             names=["chrom","snp_id","cm","pos","a1","a2"],
                             usecols=["chrom","snp_id","pos"],
                             dtype={"chrom": str, "snp_id": str,
                                    "pos": np.int64})
        for _, q in qtls.iterrows():
            mask = ((bim_df["chrom"] == q["chrom"])
                    & (bim_df["pos"] >= int(q["qtl_pos"]) - flank_bp)
                    & (bim_df["pos"] <= int(q["qtl_pos"]) + flank_bp))
            window = bim_df.loc[mask]
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


def load_chrom_map(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    df = pd.read_csv(path, sep="\t")
    return dict(zip(df["panel_chrom"].astype(str),
                    df["fasta_chrom"].astype(str), strict=True))


def main():
    ap = argparse.ArgumentParser(description="M3.3 v2 panel-general candidate prep")
    ap.add_argument("--panel", required=True, help="e.g. cotton_hebau or horvath2020")
    ap.add_argument("--trait", required=True)
    ap.add_argument("--subgenomes", required=True,
                    help='comma-separated, e.g. "A,D" (cotton), "A,C" (rapeseed), "A,B,D" (wheat)')
    ap.add_argument("--sumstats", required=True,
                    help="LOCO sumstats TSV (homoeogwas CLI in-memory mode output)")
    ap.add_argument("--leads-tsv", required=True,
                    help="M3.2 leads TSV (new_locus_candidates.tsv or wheat-v2 novel_loci_tiered.tsv)")
    ap.add_argument("--known-qtl", required=True)
    ap.add_argument("--bed-root", required=True,
                    help="root containing <subgenome>/all.bim e.g. data/processed/cotton")
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--chrom-map", default="",
                    help="TSV with panel_chrom→fasta_chrom (used at FASTA fetch); blank=identity")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--plink2", default=str(
        Path.home() / ".local/share/mamba/envs/polygwas-cpu/bin/plink2"))
    ap.add_argument("--p-max", type=float, default=P_MAX_DEFAULT)
    ap.add_argument("--ld-leads-tier", default="ALL",
                    choices=["STRICT_HIGH_CONFIDENCE","HIGH_CONFIDENCE","ALL"])
    ap.add_argument("--ld-window-kb", type=int, default=LD_WINDOW_KB,
                    help="LD-partner window (±kb) around each lead. Tighten for "
                         "dense WGS panels (e.g. rice 4.7M SNP) to keep the "
                         "candidate set tractable/comparable. Default 1000.")
    ap.add_argument("--ld-min-r2", type=float, default=LD_MIN_R2,
                    help="LD-partner r² threshold. Default 0.20; raise (e.g. 0.5) "
                         "for dense panels.")
    ap.add_argument("--pilot", action="store_true",
                    help="pilot mode: skip p<p_max stream + LD expansion")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"=== M3.3 v2 prepare — panel={args.panel} trait={args.trait} ===")

    subgenomes = [s.strip() for s in args.subgenomes.split(",") if s.strip()]
    bed_root = Path(args.bed_root)
    chrom_map = load_chrom_map(Path(args.chrom_map)) if args.chrom_map else {}
    print(f"  subgenomes={subgenomes}  bed_root={bed_root}  chrom_map_entries={len(chrom_map)}")

    # 1. leads
    print(f"[1] leads ({args.ld_leads_tier})")
    all_leads, ld_leads = load_leads(Path(args.leads_tsv), args.ld_leads_tier)
    print(f"  total leads: {len(all_leads)}, LD-expand: {len(ld_leads)}")
    forced_leads = all_leads[
        [c for c in ["snp_id","subgenome","chrom","pos","beta","se","chi2","p"]
         if c in all_leads.columns]].copy()
    forced_leads["source"] = "novel_lead"

    # 2. known QTL sentinels
    print(f"[2] known QTL sentinels (±{KNOWN_SENTINEL_FLANK_BP//1000}kb)")
    sentinels = known_qtl_sentinels_general(
        Path(args.known_qtl), bed_root, subgenomes)
    n_qtl_with_sent = sentinels["qtl_name"].nunique() if len(sentinels) else 0
    print(f"  {len(sentinels)} sentinel SNPs across {n_qtl_with_sent} QTLs")
    if len(sentinels):
        sentinels["source"] = "known_qtl_sentinel"

    # 3. significant SNPs (p<p_max) from in-memory sumstats
    if args.pilot:
        sig = pd.DataFrame()
        print("[3] (pilot) skip p<p_max stream")
    else:
        print(f"[3] in-memory sumstats filter (p<{args.p_max:.0e})")
        sig = load_inmemory_sumstats(Path(args.sumstats), args.p_max)
        if len(sig):
            sig["source"] = "loco_p_max"

    # 4. LD partners
    if args.pilot or len(ld_leads) == 0:
        ld_partners = pd.DataFrame()
        print("[4] (pilot or no LD leads) skipping LD expansion")
    else:
        print(f"[4] LD expansion (r²≥{args.ld_min_r2}, ±{args.ld_window_kb}kb) for {len(ld_leads)} leads")
        ld_partners = expand_ld_partners(
            ld_leads, bed_root, args.plink2, out_dir / "ld_cache",
            ld_window_kb=args.ld_window_kb, min_r2=args.ld_min_r2)
        print(f"  {len(ld_partners)} partner pairs")

    # 5. merge
    pieces = []
    if len(sig):
        pieces.append(sig[["snp_id","subgenome","chrom","pos","beta","se","chi2","p","source"]])
    if len(forced_leads):
        # forced_leads may lack some cols; align
        for c in ["snp_id","subgenome","chrom","pos","beta","se","chi2","p","source"]:
            if c not in forced_leads.columns:
                forced_leads[c] = np.nan if c not in ("snp_id","subgenome","chrom","source") else ""
        pieces.append(forced_leads[["snp_id","subgenome","chrom","pos","beta","se","chi2","p","source"]])
    if len(sentinels):
        pieces.append(sentinels[["snp_id","subgenome","chrom","pos","source"]].assign(
            beta=np.nan, se=np.nan, chi2=np.nan, p=np.nan))
    if len(ld_partners):
        partners_df = ld_partners[["partner_snp","chrom","subgenome","partner_pos","r2"]].rename(
            columns={"partner_snp":"snp_id","partner_pos":"pos"})
        partners_df = partners_df.assign(
            beta=np.nan, se=np.nan, chi2=np.nan, p=np.nan, source="ld_partner")
        pieces.append(partners_df[["snp_id","subgenome","chrom","pos","beta","se","chi2","p","source"]])

    if not pieces:
        print("  no candidates assembled (empty inputs)")
        return 1
    all_cand = pd.concat(pieces, ignore_index=True)
    all_cand["p_safe"] = all_cand["p"].fillna(1.0)
    all_cand["snp_id"] = all_cand["snp_id"].astype(str)
    # collapse: keep min(p) per snp; union sources
    agg = all_cand.sort_values("p_safe").drop_duplicates("snp_id", keep="first")
    src_map = all_cand.groupby("snp_id")["source"].apply(
        lambda s: ";".join(sorted(set(s)))).to_dict()
    agg["source"] = agg["snp_id"].map(src_map)
    agg = agg.drop(columns=["p_safe"]).sort_values(["chrom","pos"]).reset_index(drop=True)
    print(f"[5] merged unique: {len(agg)}")
    print(f"  source breakdown: {agg['source'].value_counts().head(8).to_dict()}")

    # 6. BIM alleles + FASTA REF
    print(f"[6] BIM+FASTA harmonisation (fasta={args.fasta})")
    snp_filter = set(agg["snp_id"].astype(str))
    allele_map = load_bim_alleles(bed_root, subgenomes, snp_id_filter=snp_filter)
    print(f"  loaded {len(allele_map)}/{len(snp_filter)} BIM alleles")
    if not Path(args.fasta).exists():
        print(f"  ERR: fasta missing: {args.fasta}")
        return 2
    final = harmonize_with_fasta(agg, Path(args.fasta), allele_map, chrom_map)
    SCORABLE = ["A1_IS_REF", "A2_IS_REF", "A1_IS_REF_RC", "A2_IS_REF_RC"]
    n_direct = int(final["ref_match_status"].isin(["A1_IS_REF", "A2_IS_REF"]).sum())
    n_match = int(final["ref_match_status"].isin(SCORABLE).sum())
    n_rc = n_match - n_direct
    n_pal = int(final.get("is_palindromic", pd.Series(dtype=bool)).sum())
    print(f"  REF match (scorable): {n_match}/{len(final)} "
          f"({100*n_match/max(len(final),1):.1f}%)  "
          f"[direct {n_direct} + RC {n_rc}; palindromic {n_pal}]")
    print(f"  status breakdown: {final['ref_match_status'].value_counts().to_dict()}")

    # 7. save
    out_path = out_dir / "candidates.tsv.gz"
    final.to_csv(out_path, sep="\t", index=False, compression="gzip")
    print(f"[7] wrote {out_path}")

    summary = {
        "panel": args.panel, "trait": args.trait, "pilot": args.pilot,
        "subgenomes": subgenomes, "p_max": args.p_max,
        "ld_leads_tier": args.ld_leads_tier,
        "ld_window_kb": LD_WINDOW_KB, "ld_min_r2": LD_MIN_R2,
        "n_total_candidates": int(len(final)),
        "n_p_lt_pmax": int(len(sig)),
        "n_forced_leads": int(len(forced_leads)),
        "n_sentinels": int(len(sentinels)),
        "n_ld_partners": int(len(ld_partners)),
        "ref_match_breakdown": final["ref_match_status"].value_counts().to_dict(),
        "n_ref_matched": n_match,
        "n_ref_matched_direct": n_direct,
        "n_ref_matched_rc": n_rc,
        "n_palindromic": n_pal,
        "source_breakdown": final["source"].value_counts().head(20).to_dict(),
        "runtime_sec": round(time.time() - t0, 1),
        "outputs": {"candidates": str(out_path)},
    }
    summary_path = out_dir / "candidates_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=_json_default)
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
