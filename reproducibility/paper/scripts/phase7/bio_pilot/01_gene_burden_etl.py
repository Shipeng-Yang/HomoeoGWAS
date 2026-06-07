#!/usr/bin/env python3
"""Phase 7 bio-pilot Step 1: GFF -> per-gene SNP indices -> retained genes (>=3 SNP)
+ MCScanX inputs for the A01-D01 pilot pair.

Maps cotton hebau bim SNPs (A01, D01) onto NDM8 gene bodies (body-only, primary).
Builds:
  - genes.tsv: gene table (chrom, gene_id, start, end, strand, sub, n_snp)
  - snp_to_gene.npz: per-retained-gene SNP indices into the subgenome's bim (variable-length)
  - retained_pep.fa: subset PEP for retained genes, MCScanX-labeled ("aa01_..."/"dd01_...")
  - mcscanx.gff: MCScanX 4-col format (sp_chrom_label, gene_id_labeled, start, end)
ID convention: PEP header = "GhM_<chrom>G<n>.1" (transcript) -> gene_id strips ".1"
(verified 1:1 across 80124 entries).
"""
from __future__ import annotations

import argparse
import bisect
import gzip
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path("/mnt/7302share/fast_ysp/U7_GWAS")
GFF = Path("/mnt/nvme/cotton_hbau/cottongen/NDM8.gff3.gz")
PEP = Path("/mnt/nvme/cotton_hbau/cottongen/NDM8.pep.fa")
BIM_A = ROOT / "data/processed/cotton/A/all.bim"
BIM_D = ROOT / "data/processed/cotton/D/all.bim"
OUT = ROOT / "results/phase7/bio_pilot"

_ID_RE = re.compile(r"ID=([^;]+)")


def _parse_gff_genes(chroms):
    """Return list of (chrom, gene_id, start, end, strand) for `gene` features on chroms."""
    rows = []
    with gzip.open(GFF, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) < 9 or p[2] != "gene" or p[0] not in chroms:
                continue
            m = _ID_RE.search(p[8])
            if not m:
                continue
            rows.append((p[0], m.group(1), int(p[3]), int(p[4]), p[6]))
    return rows


def _parse_bim_snps(bim_path, chroms):
    """Return dict chrom -> (positions array, snp_idx-in-bim array)."""
    by_chrom = {c: ([], []) for c in chroms}
    with open(bim_path) as f:
        for i, line in enumerate(f):
            p = line.rstrip("\n").split("\t")
            c = p[0]
            if c in by_chrom:
                by_chrom[c][0].append(int(p[3]))
                by_chrom[c][1].append(i)
    return {c: (np.array(v[0], int), np.array(v[1], int)) for c, v in by_chrom.items()}


def _map_snps_to_genes(genes_on_chrom, pos_arr):
    """genes_on_chrom: list[(gene_id, start, end)] (sorted by start).
    pos_arr: sorted SNP positions (within chrom).
    Returns dict gene_id -> list[snp_array_idx] (indices into pos_arr).

    Body-only: SNP at pos p assigned to gene if start <= p <= end. SNP may map to
    multiple overlapping genes (rare in plants); each gets the index."""
    starts = [g[1] for g in genes_on_chrom]
    gene2snps = {g[0]: [] for g in genes_on_chrom}
    # for each snp, find all genes whose start <= pos; walk backward while end >= pos
    for si, pos in enumerate(pos_arr):
        j = bisect.bisect_right(starts, pos) - 1  # last gene with start <= pos
        while j >= 0:
            gid, gs, ge = genes_on_chrom[j]
            if gs > pos:  # safety
                j -= 1; continue
            if ge >= pos:
                gene2snps[gid].append(si)
            # genes are sorted by start; earlier genes with start <= pos may still have end >= pos
            # (overlap). Walk back, but stop if we've gone far (heuristic bound by typical gene size).
            if pos - gs > 200_000:  # cotton genes rarely > 200kb; safe early-stop
                break
            j -= 1
    return gene2snps


def _extract_pep(pep_path, retained_gene_ids):
    """Yield (gene_id, seq) for headers whose stripped (.1) gene_id is in retained set."""
    cur_gid, cur_seq = None, []
    with open(pep_path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if cur_gid is not None and cur_gid in retained_gene_ids:
                    yield cur_gid, "".join(cur_seq)
                tid = line[1:].split()[0]
                gid = tid[:-2] if tid.endswith(".1") else tid
                cur_gid = gid
                cur_seq = []
            else:
                cur_seq.append(line)
    if cur_gid is not None and cur_gid in retained_gene_ids:
        yield cur_gid, "".join(cur_seq)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chrom-pairs", default="A01:D01",
                    help="comma-list of A:D pairs to ETL (pilot: A01:D01)")
    ap.add_argument("--min-snp", type=int, default=3,
                    help="min SNP per gene (primary=3; sensitivity=5)")
    ap.add_argument("--out-dir", default=str(OUT))
    args = ap.parse_args()

    pairs = [tuple(p.split(":")) for p in args.chrom_pairs.split(",")]
    a_chroms = [a for a, _ in pairs]; d_chroms = [d for _, d in pairs]
    all_chroms = set(a_chroms + d_chroms)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    print(f"=== bio-pilot ETL: pairs={pairs} min_snp={args.min_snp} ===")
    # 1. parse GFF
    raw_genes = _parse_gff_genes(all_chroms)
    print(f"  GFF: {len(raw_genes)} genes on {sorted(all_chroms)}")

    # 2. parse BIM (A subgenome bim and D subgenome bim separately)
    a_snps = _parse_bim_snps(BIM_A, a_chroms)
    d_snps = _parse_bim_snps(BIM_D, d_chroms)
    for c in a_chroms:
        print(f"  BIM A {c}: {a_snps[c][0].size} SNPs")
    for c in d_chroms:
        print(f"  BIM D {c}: {d_snps[c][0].size} SNPs")

    # 3. map SNPs to genes, per chrom
    gene_rows = []  # full output rows
    gene_snp_indices = {}  # gene_id -> snp idx-in-bim list
    sub_map = {}  # gene_id -> "A"|"D"
    for chrom in sorted(all_chroms):
        genes_c = [(g[1], g[2], g[3]) for g in raw_genes if g[0] == chrom]
        genes_c.sort(key=lambda x: x[1])
        sub = "A" if chrom in a_chroms else "D"
        pos_arr, snp_idx_arr = (a_snps if sub == "A" else d_snps)[chrom]
        # ensure positions are sorted (BIM may be, but enforce)
        order = np.argsort(pos_arr)
        pos_sorted = pos_arr[order]; snp_idx_sorted = snp_idx_arr[order]
        g2s_local = _map_snps_to_genes(genes_c, pos_sorted)
        for gid, gs, ge in genes_c:
            local = g2s_local[gid]
            # convert local (pos-sorted) idx -> bim idx
            bim_idx = snp_idx_sorted[local].tolist() if local else []
            n = len(bim_idx)
            strand = next((g[4] for g in raw_genes if g[1] == gid), ".")
            gene_rows.append(dict(chrom=chrom, gene_id=gid, start=gs, end=ge, strand=strand,
                                  sub=sub, n_snp=n))
            if n > 0:
                gene_snp_indices[gid] = bim_idx
                sub_map[gid] = sub

    # 4. retained: n_snp >= min_snp
    retained = [g for g in gene_rows if g["n_snp"] >= args.min_snp]
    retained_ids = {g["gene_id"] for g in retained}
    print(f"  GENES total={len(gene_rows)} with-snp={sum(1 for g in gene_rows if g['n_snp']>0)} "
          f"retained(>={args.min_snp})={len(retained)} "
          f"[A={sum(1 for g in retained if g['sub']=='A')} D={sum(1 for g in retained if g['sub']=='D')}]")

    # 5. write outputs
    with (out / "genes.tsv").open("w") as f:
        f.write("chrom\tgene_id\tstart\tend\tstrand\tsub\tn_snp\tretained\n")
        for g in gene_rows:
            f.write(f"{g['chrom']}\t{g['gene_id']}\t{g['start']}\t{g['end']}\t"
                    f"{g['strand']}\t{g['sub']}\t{g['n_snp']}\t"
                    f"{int(g['gene_id'] in retained_ids)}\n")
    # snp indices: save as object array (variable-length per gene), keyed by gene_id
    keys = list(gene_snp_indices.keys())
    # filter to retained only
    keys = [k for k in keys if k in retained_ids]
    arrs = np.array([np.array(gene_snp_indices[k], int) for k in keys], dtype=object)
    subs = np.array([sub_map[k] for k in keys])
    np.savez(out / "snp_to_gene.npz", gene_ids=np.array(keys), snp_idx=arrs, sub=subs,
             allow_pickle=True)
    print(f"  wrote genes.tsv ({len(gene_rows)} rows), snp_to_gene.npz ({len(keys)} retained genes)")

    # 6. MCScanX .gff: 4-col "sp_chrom_label\tgene_id\tstart\tend"
    #    MCScanX uses first 2 chars of sp_chrom as species code -> we encode subgenome as
    #    species: A->"aa", D->"dd"; chrom label = "aa01"/"dd01" so it cleanly separates.
    with (out / "mcscanx.gff").open("w") as f:
        for g in retained:
            code = "aa" if g["sub"] == "A" else "dd"
            sp_chrom = f"{code}{g['chrom'][1:]}"  # A01 -> aa01, D01 -> dd01
            f.write(f"{sp_chrom}\t{g['gene_id']}\t{g['start']}\t{g['end']}\n")
    print(f"  wrote mcscanx.gff ({len(retained)} labeled gene entries)")

    # 7. retained PEP subset
    pep_out = out / "retained_pep.fa"
    n_pep = 0
    with pep_out.open("w") as fout:
        for gid, seq in _extract_pep(PEP, retained_ids):
            fout.write(f">{gid}\n{seq}\n")
            n_pep += 1
    print(f"  wrote retained_pep.fa ({n_pep} sequences; expect == retained count)")

    # 8. summary json
    summ = dict(pairs=pairs, min_snp=args.min_snp, n_gene_total=len(gene_rows),
                n_gene_with_snp=sum(1 for g in gene_rows if g["n_snp"] > 0),
                n_retained=len(retained),
                n_retained_A=sum(1 for g in retained if g["sub"] == "A"),
                n_retained_D=sum(1 for g in retained if g["sub"] == "D"),
                n_pep_extracted=n_pep,
                snp_count_quantiles_all=dict(
                    p10=int(np.percentile([g["n_snp"] for g in gene_rows if g["n_snp"] > 0], 10)),
                    p50=int(np.percentile([g["n_snp"] for g in gene_rows if g["n_snp"] > 0], 50)),
                    p90=int(np.percentile([g["n_snp"] for g in gene_rows if g["n_snp"] > 0], 90))),
                snp_count_quantiles_retained=dict(
                    p10=int(np.percentile([g["n_snp"] for g in retained], 10)),
                    p50=int(np.percentile([g["n_snp"] for g in retained], 50)),
                    p90=int(np.percentile([g["n_snp"] for g in retained], 90))))
    (out / "etl_summary.json").write_text(json.dumps(summ, indent=2))
    print(f"  ETL DONE. summary: {summ}")


if __name__ == "__main__":
    main()
