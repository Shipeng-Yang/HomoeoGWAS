#!/usr/bin/env python3
"""Phase B rapeseed Step 3: parse union .collinearity -> audit + 4 final pair sets.
Adapted from cotton 03b: rapeseed gene_id (LOCxxx) doesn't carry A/C label, so
subgenome is determined from chrom (chrA0i vs chrC0i) using genes.tsv lookup.
same_number = chrA0i + chrC0i (i in 1..9). chrA10 has no C10 -> goes to cross_number.
"""
from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/mnt/7302share/fast_ysp/U7_GWAS")
DEFAULT_OUT = ROOT / "results/phase7/bio_rapeseed"
DEFAULT_COLL = "/mnt/nvme/rapeseed_mcscanx/union.collinearity"
DEFAULT_BLAST = "/mnt/nvme/rapeseed_mcscanx/union.blast"


def _parse_collinearity(path, gmeta):
    """Return rows of inter-subgenome pair lines (A-C). gmeta: gene_id -> chrom/sub."""
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
                block_inter = (ch1.startswith("aa") and ch2.startswith("cc")) or \
                              (ch1.startswith("cc") and ch2.startswith("aa"))
                header_pair = f"{ch1}&{ch2}"
                continue
            if not block_inter:
                continue
            mm = pat_pair.match(line)
            if not mm:
                continue
            g1, g2, ev = mm.group(2), mm.group(3), mm.group(4)
            if g1 not in gmeta.index or g2 not in gmeta.index:
                continue
            sub1 = gmeta.at[g1, "sub"]; sub2 = gmeta.at[g2, "sub"]
            a, c = (g1, g2) if sub1 == "A" and sub2 == "C" else (
                (g2, g1) if sub2 == "A" and sub1 == "C" else (None, None))
            if a is None:
                continue
            try:
                ev_f = float(ev)
            except ValueError:
                ev_f = 1.0
            rows.append(dict(block_id=block_id, header_pair=header_pair,
                             gene_A=a, gene_C=c, e_value=ev_f))
    return rows


def _load_bit_scores(path):
    bs = {}
    with open(path) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 12:
                continue
            try:
                b = float(p[11])
            except ValueError:
                continue
            k = (p[0], p[1]) if p[0] < p[1] else (p[1], p[0])
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
    print("=== Phase B rapeseed Step 3: pair audit + final 1:1 sets ===")

    gmeta = pd.read_csv(out / "genes.tsv", sep="\t").set_index("gene_id")
    raw = _parse_collinearity(args.collinearity, gmeta)
    print(f"  raw inter-subgenome (A-C) collinear pair lines: {len(raw)}")
    bs = _load_bit_scores(args.blast)
    for r in raw:
        k = (r["gene_A"], r["gene_C"]) if r["gene_A"] < r["gene_C"] else (r["gene_C"], r["gene_A"])
        r["bit_score"] = float(bs.get(k, np.nan))
        r["chr_A"] = gmeta.at[r["gene_A"], "chrom"]
        r["chr_C"] = gmeta.at[r["gene_C"], "chrom"]
        a_num = r["chr_A"][4:]  # chrA01 -> 01
        c_num = r["chr_C"][4:]
        r["pair_class"] = "same_number" if a_num == c_num else "cross_number"
        r["retained_body"] = int(gmeta.at[r["gene_A"], "retained_body"]
                                  and gmeta.at[r["gene_C"], "retained_body"])
        r["retained_flank"] = int(gmeta.at[r["gene_A"], "retained_flank"]
                                   and gmeta.at[r["gene_C"], "retained_flank"])
        r["n_snp_body_A"] = int(gmeta.at[r["gene_A"], "n_snp_body"])
        r["n_snp_body_C"] = int(gmeta.at[r["gene_C"], "n_snp_body"])
        r["n_snp_flank_A"] = int(gmeta.at[r["gene_A"], "n_snp_flank"])
        r["n_snp_flank_C"] = int(gmeta.at[r["gene_C"], "n_snp_flank"])

    # within-block-best (A best D / D best A within same block)
    block_groups = defaultdict(list)
    for i, r in enumerate(raw):
        block_groups[r["block_id"]].append(i)
    for bid, idxs in block_groups.items():
        a_best, c_best = {}, {}
        for ri in idxs:
            r = raw[ri]
            if r["gene_A"] not in a_best or r["e_value"] < raw[a_best[r["gene_A"]]]["e_value"]:
                a_best[r["gene_A"]] = ri
            if r["gene_C"] not in c_best or r["e_value"] < raw[c_best[r["gene_C"]]]["e_value"]:
                c_best[r["gene_C"]] = ri
        for ri in idxs:
            r = raw[ri]
            r["within_block_best_A"] = int(a_best[r["gene_A"]] == ri)
            r["within_block_best_C"] = int(c_best[r["gene_C"]] == ri)
            r["within_block_best_flag"] = int(r["within_block_best_A"] and r["within_block_best_C"])

    # global ranks
    a_groups, c_groups = defaultdict(list), defaultdict(list)
    for i, r in enumerate(raw):
        a_groups[r["gene_A"]].append(i); c_groups[r["gene_C"]].append(i)
    for a, idxs in a_groups.items():
        s = sorted(idxs, key=lambda i: raw[i]["e_value"])
        for rk, ri in enumerate(s):
            raw[ri]["within_A_rank"] = rk + 1
    for c, idxs in c_groups.items():
        s = sorted(idxs, key=lambda i: raw[i]["e_value"])
        for rk, ri in enumerate(s):
            raw[ri]["within_C_rank"] = rk + 1
    for r in raw:
        r["reciprocal_best_flag"] = int(r["within_A_rank"] == 1 and r["within_C_rank"] == 1)
        r["final_1to1_flag"] = int(r["within_block_best_flag"] and r["reciprocal_best_flag"])
        if r["final_1to1_flag"]:
            r["reason_dropped"] = ""
        elif not r["within_block_best_flag"]:
            r["reason_dropped"] = ("within_block: same A has better C" if not r["within_block_best_A"]
                                    else "within_block: same C has better A")
        elif r["within_A_rank"] > 1:
            r["reason_dropped"] = "non-reciprocal: same A better C in another block"
        elif r["within_C_rank"] > 1:
            r["reason_dropped"] = "non-reciprocal: same C better A in another block"
        else:
            r["reason_dropped"] = "tie/ambiguous"

    audit = pd.DataFrame(raw)
    cols = ["block_id", "header_pair", "gene_A", "gene_C", "chr_A", "chr_C", "pair_class",
            "e_value", "bit_score", "within_A_rank", "within_C_rank",
            "within_block_best_A", "within_block_best_C", "within_block_best_flag",
            "reciprocal_best_flag", "final_1to1_flag", "reason_dropped",
            "retained_body", "retained_flank",
            "n_snp_body_A", "n_snp_body_C", "n_snp_flank_A", "n_snp_flank_C"]
    audit = audit[cols]
    audit.to_csv(out / "audit_table.tsv", sep="\t", index=False)
    print(f"  audit_table.tsv: {len(audit)} raw pairs")

    n_raw = len(audit)
    print(f"  raw {n_raw}  same={int((audit['pair_class']=='same_number').sum())}  "
          f"cross={int((audit['pair_class']=='cross_number').sum())}")
    print(f"  within_block_best={int(audit['within_block_best_flag'].sum())}  "
          f"reciprocal_best={int(audit['reciprocal_best_flag'].sum())}  "
          f"final_1to1={int(audit['final_1to1_flag'].sum())}")
    print(f"  reason_dropped: {audit[audit['final_1to1_flag']==0]['reason_dropped'].value_counts().to_dict()}")

    final = audit[audit["final_1to1_flag"] == 1].copy()
    for mode, ret_col in [("body", "retained_body"), ("flank", "retained_flank")]:
        for cls in ["same_number", "cross_number"]:
            sub = final[(final[ret_col] == 1) & (final["pair_class"] == cls)].copy()
            tag = "same" if cls == "same_number" else "cross"
            sub.to_csv(out / f"pairs_{mode}_{tag}.tsv", sep="\t", index=False)
            print(f"  pairs_{mode}_{tag}.tsv: G={len(sub)}")
    print("=== DONE ===")


if __name__ == "__main__":
    main()
