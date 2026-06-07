#!/usr/bin/env python3
"""Phase A Step 3: parse union .collinearity -> audit table -> 4 final pair sets
(body/flank2kb × same_number/cross_number). Implements the "within-block-best
+ reciprocal" filter with full audit columns.

audit_table.tsv columns:
  gene_A, gene_D, chr_A, chr_D, block_id, e_value, bit_score, pair_class,
  within_A_rank (across all blocks, by e_value), within_D_rank,
  reciprocal_best_flag, within_block_best_flag, final_1to1_flag,
  reason_dropped (if not final), retained_body, retained_flank
4 final pair tsv files: pairs_{body,flank}_{same,cross}.tsv with subset of cols.
"""
from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/mnt/7302share/fast_ysp/U7_GWAS")
DEFAULT_OUT = ROOT / "results/phase7/bio_full"
DEFAULT_COLL = "/mnt/nvme/cotton_hbau/mcscanx_full/union.collinearity"
DEFAULT_BLAST = "/mnt/nvme/cotton_hbau/mcscanx_full/union.blast"


def _parse_collinearity(path):
    """Return rows of (block_id, gene1, gene2, e_value, header_pair) for inter-subgenome
    alignments only (header_pair like 'aa01&dd01'). For each line we also keep which side
    is A vs D via the gene_id prefix (GhM_A.../GhM_D...)."""
    rows = []
    block_id = None
    block_inter = False
    header_pair = ""
    pat_pair = re.compile(r"^\s*(\d+)-\s*\d+:\s+(\S+)\s+(\S+)\s+(\S+)")
    pat_aln = re.compile(r"^## Alignment (\d+):.*?(\S+)&(\S+)")
    with open(path) as f:
        for line in f:
            m = pat_aln.match(line)
            if m:
                block_id = int(m.group(1))
                ch1, ch2 = m.group(2), m.group(3)
                block_inter = (ch1.startswith("aa") and ch2.startswith("dd")) or \
                              (ch1.startswith("dd") and ch2.startswith("aa"))
                header_pair = f"{ch1}&{ch2}"
                continue
            if not block_inter:
                continue
            mm = pat_pair.match(line)
            if not mm:
                continue
            g1, g2, ev = mm.group(2), mm.group(3), mm.group(4)
            a, d = (g1, g2) if "_A" in g1 and "_D" in g2 else (
                (g2, g1) if "_A" in g2 and "_D" in g1 else (None, None))
            if a is None:
                continue
            try:
                ev_f = float(ev)
            except ValueError:
                ev_f = 1.0
            rows.append(dict(block_id=block_id, header_pair=header_pair,
                             gene_A=a, gene_D=d, e_value=ev_f))
    return rows


def _load_bit_scores(blast_path):
    """Return dict (qry,sbj) -> max bit_score from BLAST tabular outfmt 6."""
    bs = {}
    with open(blast_path) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 12:
                continue
            q, s = p[0], p[1]
            try:
                b = float(p[11])
            except ValueError:
                continue
            k = (q, s) if q < s else (s, q)  # unordered key
            if k not in bs or b > bs[k]:
                bs[k] = b
    return bs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--collinearity", default=DEFAULT_COLL)
    ap.add_argument("--blast", default=DEFAULT_BLAST)
    args = ap.parse_args()
    out = Path(args.out_dir)

    print("=== Phase A Step 3: pair audit + final 1:1 sets ===")
    raw = _parse_collinearity(args.collinearity)
    print(f"  raw inter-subgenome collinear pair lines: {len(raw)}")
    bs = _load_bit_scores(args.blast)
    for r in raw:
        k = (r["gene_A"], r["gene_D"]) if r["gene_A"] < r["gene_D"] else (r["gene_D"], r["gene_A"])
        r["bit_score"] = float(bs.get(k, np.nan))

    # load gene meta to get chrom + retained flags
    gmeta = pd.read_csv(out / "genes.tsv", sep="\t").set_index("gene_id")
    for r in raw:
        r["chr_A"] = gmeta.loc[r["gene_A"], "chrom"]
        r["chr_D"] = gmeta.loc[r["gene_D"], "chrom"]
        r["pair_class"] = ("same_number" if r["chr_A"][1:] == r["chr_D"][1:] else "cross_number")
        r["retained_body"] = int(gmeta.loc[r["gene_A"], "retained_body"]
                                  and gmeta.loc[r["gene_D"], "retained_body"])
        r["retained_flank"] = int(gmeta.loc[r["gene_A"], "retained_flank"]
                                   and gmeta.loc[r["gene_D"], "retained_flank"])
        r["n_snp_body_A"] = int(gmeta.loc[r["gene_A"], "n_snp_body"])
        r["n_snp_body_D"] = int(gmeta.loc[r["gene_D"], "n_snp_body"])
        r["n_snp_flank_A"] = int(gmeta.loc[r["gene_A"], "n_snp_flank"])
        r["n_snp_flank_D"] = int(gmeta.loc[r["gene_D"], "n_snp_flank"])

    # within-block-best flags (per block, for each A, is this its best-e D in this block;
    # for each D, is this its best-e A in this block)
    block_groups = defaultdict(list)
    for i, r in enumerate(raw):
        block_groups[r["block_id"]].append(i)
    for bid, idxs in block_groups.items():
        block_rows = [raw[i] for i in idxs]
        # for each A in block, find min e_value entry
        a_best = {}; d_best = {}
        for ri in idxs:
            r = raw[ri]
            if r["gene_A"] not in a_best or r["e_value"] < raw[a_best[r["gene_A"]]]["e_value"]:
                a_best[r["gene_A"]] = ri
            if r["gene_D"] not in d_best or r["e_value"] < raw[d_best[r["gene_D"]]]["e_value"]:
                d_best[r["gene_D"]] = ri
        for ri in idxs:
            r = raw[ri]
            r["within_block_best_A"] = int(a_best[r["gene_A"]] == ri)
            r["within_block_best_D"] = int(d_best[r["gene_D"]] == ri)
            r["within_block_best_flag"] = int(r["within_block_best_A"] and r["within_block_best_D"])

    # global within-A-rank, within-D-rank (across all blocks, by e_value ascending)
    a_groups = defaultdict(list); d_groups = defaultdict(list)
    for i, r in enumerate(raw):
        a_groups[r["gene_A"]].append(i)
        d_groups[r["gene_D"]].append(i)
    for a, idxs in a_groups.items():
        sorted_i = sorted(idxs, key=lambda i: raw[i]["e_value"])
        for rk, ri in enumerate(sorted_i):
            raw[ri]["within_A_rank"] = rk + 1
    for d, idxs in d_groups.items():
        sorted_i = sorted(idxs, key=lambda i: raw[i]["e_value"])
        for rk, ri in enumerate(sorted_i):
            raw[ri]["within_D_rank"] = rk + 1

    # reciprocal_best across blocks: within_A_rank==1 AND within_D_rank==1
    for r in raw:
        r["reciprocal_best_flag"] = int(r["within_A_rank"] == 1 and r["within_D_rank"] == 1)
        # final_1to1 := within-block-best AND reciprocal-best
        r["final_1to1_flag"] = int(r["within_block_best_flag"] and r["reciprocal_best_flag"])
        # reason_dropped
        if r["final_1to1_flag"]:
            r["reason_dropped"] = ""
        elif not r["within_block_best_flag"]:
            if not r["within_block_best_A"]:
                r["reason_dropped"] = "within_block: same A has better D"
            else:
                r["reason_dropped"] = "within_block: same D has better A"
        elif r["within_A_rank"] > 1:
            r["reason_dropped"] = "non-reciprocal: same A has better D in another block"
        elif r["within_D_rank"] > 1:
            r["reason_dropped"] = "non-reciprocal: same D has better A in another block"
        else:
            r["reason_dropped"] = "tie/ambiguous"

    audit = pd.DataFrame(raw)
    cols = ["block_id", "header_pair", "gene_A", "gene_D", "chr_A", "chr_D", "pair_class",
            "e_value", "bit_score", "within_A_rank", "within_D_rank",
            "within_block_best_A", "within_block_best_D", "within_block_best_flag",
            "reciprocal_best_flag", "final_1to1_flag", "reason_dropped",
            "retained_body", "retained_flank",
            "n_snp_body_A", "n_snp_body_D", "n_snp_flank_A", "n_snp_flank_D"]
    audit = audit[cols]
    audit.to_csv(out / "audit_table.tsv", sep="\t", index=False)
    print(f"  audit_table.tsv: {len(audit)} raw pairs")

    # summary stats
    n_raw = len(audit)
    n_same = int((audit["pair_class"] == "same_number").sum())
    n_cross = int((audit["pair_class"] == "cross_number").sum())
    n_wbb = int(audit["within_block_best_flag"].sum())
    n_rcp = int(audit["reciprocal_best_flag"].sum())
    n_1to1 = int(audit["final_1to1_flag"].sum())
    reasons = audit[audit["final_1to1_flag"] == 0]["reason_dropped"].value_counts().to_dict()
    print(f"  raw {n_raw}  same={n_same}  cross={n_cross}")
    print(f"  within_block_best={n_wbb}  reciprocal_best={n_rcp}  final_1to1={n_1to1}")
    print(f"  reason_dropped: {reasons}")

    # 4 final pair sets: {body, flank} × {same, cross}
    final = audit[audit["final_1to1_flag"] == 1].copy()
    for mode, ret_col in [("body", "retained_body"), ("flank", "retained_flank")]:
        for cls in ["same_number", "cross_number"]:
            sub = final[(final[ret_col] == 1) & (final["pair_class"] == cls)].copy()
            tag = "same" if cls == "same_number" else "cross"
            fp = out / f"pairs_{mode}_{tag}.tsv"
            sub.to_csv(fp, sep="\t", index=False)
            print(f"  pairs_{mode}_{tag}.tsv: G={len(sub)}")

    # manifest
    mans = []
    for mode in ["body", "flank"]:
        for cls in ["same", "cross"]:
            fp = out / f"pairs_{mode}_{cls}.tsv"
            df = pd.read_csv(fp, sep="\t")
            mans.append(dict(mode=mode, pair_class=cls, n_pairs=len(df),
                             n_unique_A=df["gene_A"].nunique() if len(df) else 0,
                             n_unique_D=df["gene_D"].nunique() if len(df) else 0,
                             path=str(fp.relative_to(ROOT))))
    pd.DataFrame(mans).to_csv(out / "manifest.tsv", sep="\t", index=False)
    print(f"  manifest.tsv written.")
    print("=== DONE ===")


if __name__ == "__main__":
    main()
