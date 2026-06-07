#!/usr/bin/env python3
"""Phase A Step 1 (rewrite): full 13-chrom-pair ETL + dual-mode (body-only and ±2kb).

Codex Phase A key design: body and ±2kb SHARE one MCScanX synteny backbone (computed
on the UNION of body-retained and flank-retained PEP). This decouples "window changes
SNP assignment" from "window changes which genes are syntenic" — without it, body vs
±2kb differences confound ETL changes with biology (Codex TOP_RISK).

Outputs (all global, mode columns embedded):
  genes.tsv             — 80k rows: gene_id, chrom, start, end, strand, sub,
                          n_snp_body, n_snp_flank2kb, retained_body, retained_flank2kb
  snp_to_gene_body.npz  — gene_ids + per-gene SNP indices (body-only retained ≥min_snp)
  snp_to_gene_flank2kb.npz
  union_pep.fa          — PEP for genes retained under EITHER mode (MCScanX backbone)
  union_mcscanx.gff     — MCScanX 4-col: sp_chrom_label(aa01/dd01) gene_id start end
  etl_summary.json
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
OUT = ROOT / "results/phase7/bio_full"

_ID_RE = re.compile(r"ID=([^;]+)")
A_CHROMS = [f"A{i:02d}" for i in range(1, 14)]
D_CHROMS = [f"D{i:02d}" for i in range(1, 14)]
ALL_CHROMS = set(A_CHROMS + D_CHROMS)


def _parse_gff_genes():
    rows = []
    with gzip.open(GFF, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) < 9 or p[2] != "gene" or p[0] not in ALL_CHROMS:
                continue
            m = _ID_RE.search(p[8])
            if not m:
                continue
            rows.append((p[0], m.group(1), int(p[3]), int(p[4]), p[6]))
    return rows


def _parse_bim_snps(bim_path, chroms):
    by_chrom = {c: ([], []) for c in chroms}
    with open(bim_path) as f:
        for i, line in enumerate(f):
            p = line.rstrip("\n").split("\t")
            if p[0] in by_chrom:
                by_chrom[p[0]][0].append(int(p[3])); by_chrom[p[0]][1].append(i)
    return {c: (np.array(v[0], int), np.array(v[1], int)) for c, v in by_chrom.items()}


def _map_snps_to_windows(genes_on_chrom_ext, pos_sorted, snp_idx_sorted):
    """genes_on_chrom_ext: list[(gene_id, window_start, window_end)] sorted by window_start.
    pos_sorted: SNP positions (ascending) on the chrom; snp_idx_sorted: bim indices in same order.
    Returns dict gene_id -> bim-row idx list (SNPs falling in [w_start, w_end])."""
    starts = [g[1] for g in genes_on_chrom_ext]
    out = {g[0]: [] for g in genes_on_chrom_ext}
    for si, pos in enumerate(pos_sorted):
        j = bisect.bisect_right(starts, pos) - 1
        # walk back across overlapping windows; bound walk-back by ±2kb world ≈ 1MB max realistic
        steps = 0
        while j >= 0 and steps < 2000:
            gid, ws, we = genes_on_chrom_ext[j]
            if pos > we and pos - ws > 1_000_000:
                break
            if ws <= pos <= we:
                out[gid].append(int(snp_idx_sorted[si]))
            j -= 1; steps += 1
    return out


def _extract_pep_subset(pep_path, retained_ids):
    """Yield (gene_id, seq) for retained gene IDs (PEP transcript-id "gene_id.1")."""
    cur_gid, cur_seq = None, []
    with open(pep_path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if cur_gid is not None and cur_gid in retained_ids:
                    yield cur_gid, "".join(cur_seq)
                tid = line[1:].split()[0]
                cur_gid = tid[:-2] if tid.endswith(".1") else tid
                cur_seq = []
            else:
                cur_seq.append(line)
    if cur_gid is not None and cur_gid in retained_ids:
        yield cur_gid, "".join(cur_seq)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flank-bp", type=int, default=2000,
                    help="symmetric flank for the ±flank window mode (primary uses both 0 and this)")
    ap.add_argument("--min-snp", type=int, default=3)
    ap.add_argument("--out-dir", default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    print(f"=== Phase A Step 1: full 13-chrom ETL (body-only AND ±{args.flank_bp}bp) ===")

    raw_genes = _parse_gff_genes()
    print(f"  GFF: {len(raw_genes)} genes on A01-A13/D01-D13 (scaffolds dropped)")

    a_snps = _parse_bim_snps(BIM_A, A_CHROMS)
    d_snps = _parse_bim_snps(BIM_D, D_CHROMS)
    total_a = sum(v[0].size for v in a_snps.values())
    total_d = sum(v[0].size for v in d_snps.values())
    print(f"  BIM: A {total_a} SNPs, D {total_d} SNPs across 13×2 chroms")

    # build per-chrom indexes once, then run both modes
    chrom_genes = {c: [] for c in ALL_CHROMS}
    for c, gid, gs, ge, strand in raw_genes:
        chrom_genes[c].append((gid, gs, ge, strand))

    # mapping output: for each mode, gene_id -> list of bim indices
    map_body = {}
    map_flank = {}
    for c in sorted(ALL_CHROMS):
        gs_list = sorted(chrom_genes[c], key=lambda x: x[1])
        sub = "A" if c in A_CHROMS else "D"
        pos_arr, snp_idx_arr = (a_snps if sub == "A" else d_snps)[c]
        order = np.argsort(pos_arr)
        pos_sorted = pos_arr[order]; snp_idx_sorted = snp_idx_arr[order]
        # body-only windows
        ext_body = [(g[0], g[1], g[2]) for g in gs_list]
        m_body = _map_snps_to_windows(ext_body, pos_sorted, snp_idx_sorted)
        # flank ±flank-bp (re-sort by extended start)
        ext_flank = [(g[0], max(1, g[1] - args.flank_bp), g[2] + args.flank_bp) for g in gs_list]
        ext_flank.sort(key=lambda x: x[1])
        m_flank = _map_snps_to_windows(ext_flank, pos_sorted, snp_idx_sorted)
        map_body.update(m_body); map_flank.update(m_flank)

    # build genes.tsv with both n_snp columns + retained flags
    gene_meta = {}
    for c, gid, gs, ge, strand in raw_genes:
        sub = "A" if c in A_CHROMS else "D"
        n_b = len(map_body.get(gid, []))
        n_f = len(map_flank.get(gid, []))
        gene_meta[gid] = dict(gene_id=gid, chrom=c, start=gs, end=ge, strand=strand, sub=sub,
                              n_snp_body=n_b, n_snp_flank=n_f,
                              retained_body=int(n_b >= args.min_snp),
                              retained_flank=int(n_f >= args.min_snp))
    with (out / "genes.tsv").open("w") as f:
        f.write("gene_id\tchrom\tstart\tend\tstrand\tsub\tn_snp_body\tn_snp_flank\t"
                "retained_body\tretained_flank\n")
        for g in gene_meta.values():
            f.write(f"{g['gene_id']}\t{g['chrom']}\t{g['start']}\t{g['end']}\t{g['strand']}\t"
                    f"{g['sub']}\t{g['n_snp_body']}\t{g['n_snp_flank']}\t"
                    f"{g['retained_body']}\t{g['retained_flank']}\n")

    retained_body = [gid for gid, g in gene_meta.items() if g["retained_body"]]
    retained_flank = [gid for gid, g in gene_meta.items() if g["retained_flank"]]
    union_retained = set(retained_body) | set(retained_flank)

    def _summ(retained, src):
        a = [gid for gid in retained if gene_meta[gid]["sub"] == "A"]
        d = [gid for gid in retained if gene_meta[gid]["sub"] == "D"]
        return dict(total=len(retained), A=len(a), D=len(d))

    print(f"  RETAINED ≥{args.min_snp} SNP:")
    print(f"    body-only:   {_summ(retained_body, 'body')}")
    print(f"    ±{args.flank_bp}bp:  {_summ(retained_flank, 'flank')}")
    print(f"    union (MCScanX backbone): {len(union_retained)} genes "
          f"[A={sum(1 for g in union_retained if gene_meta[g]['sub']=='A')} "
          f"D={sum(1 for g in union_retained if gene_meta[g]['sub']=='D')}]")

    # snp_to_gene npz per mode (only retained genes)
    def _save_map(name, retained, source):
        keys = [gid for gid in retained]
        arrs = np.array([np.array(source[gid], int) for gid in keys], dtype=object)
        subs = np.array([gene_meta[gid]["sub"] for gid in keys])
        chroms = np.array([gene_meta[gid]["chrom"] for gid in keys])
        np.savez(out / name, gene_ids=np.array(keys), snp_idx=arrs, sub=subs, chrom=chroms,
                 allow_pickle=True)
    _save_map("snp_to_gene_body.npz", retained_body, map_body)
    _save_map(f"snp_to_gene_flank{args.flank_bp}bp.npz", retained_flank, map_flank)

    # union PEP and MCScanX gff (MCScanX format: sp_chrom_label\tgene_id\tstart\tend)
    n_pep = 0
    with (out / "union_pep.fa").open("w") as fout:
        for gid, seq in _extract_pep_subset(PEP, union_retained):
            fout.write(f">{gid}\n{seq}\n"); n_pep += 1
    with (out / "union_mcscanx.gff").open("w") as f:
        for gid in union_retained:
            g = gene_meta[gid]
            code = "aa" if g["sub"] == "A" else "dd"
            sp = f"{code}{g['chrom'][1:]}"
            f.write(f"{sp}\t{gid}\t{g['start']}\t{g['end']}\n")
    print(f"  wrote union_pep.fa ({n_pep} sequences) + union_mcscanx.gff")

    # diagnostics (SNP-per-gene quantiles for retained sets)
    def _quants(arr):
        arr = np.asarray(arr)
        return dict(p10=int(np.percentile(arr, 10)), p50=int(np.percentile(arr, 50)),
                    p90=int(np.percentile(arr, 90)), max=int(arr.max()))

    summ = dict(
        flank_bp=args.flank_bp, min_snp=args.min_snp,
        n_genes_total=len(gene_meta),
        n_genes_with_snp_body=sum(1 for g in gene_meta.values() if g["n_snp_body"] > 0),
        n_genes_with_snp_flank=sum(1 for g in gene_meta.values() if g["n_snp_flank"] > 0),
        retained_body=_summ(retained_body, ""),
        retained_flank=_summ(retained_flank, ""),
        union_retained=len(union_retained),
        snp_per_gene_body_retained=_quants([gene_meta[g]["n_snp_body"] for g in retained_body]),
        snp_per_gene_flank_retained=_quants([gene_meta[g]["n_snp_flank"] for g in retained_flank]),
        bim_total_A=total_a, bim_total_D=total_d,
        pct_genes_with_snp_body=round(
            sum(1 for g in gene_meta.values() if g["n_snp_body"] > 0) / len(gene_meta) * 100, 2),
        pct_genes_with_snp_flank=round(
            sum(1 for g in gene_meta.values() if g["n_snp_flank"] > 0) / len(gene_meta) * 100, 2))
    (out / "etl_summary.json").write_text(json.dumps(summ, indent=2))
    print(f"  ETL SUMMARY: body coverage {summ['pct_genes_with_snp_body']}% / "
          f"±{args.flank_bp}bp coverage {summ['pct_genes_with_snp_flank']}%")
    print(f"  DONE -> {out}")


if __name__ == "__main__":
    main()
