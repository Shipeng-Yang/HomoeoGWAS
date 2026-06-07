#!/usr/bin/env python3
"""Phase 7 OAT (Avena sativa AACCDD 6n) Step 1: in-memory gene-burden ETL.

4th species / 2nd hexaploid. GBS-sparse panel (~373k SNP total) -> load the cleaned
logical merged PLINK1 bed entirely in memory, partition SNP columns by subgenome
(chrom suffix A/C/D), map SNPs to gene windows (body + ±flank), and emit per-subgenome
genotype + gene maps + a union PEP/GFF backbone for de-novo MCScanX triad building.

Reference: OT3098 v2 (Ensembl). Chrom names 1A..7D match merged.bim. gene_id like
AVESA.00001b.r3.1Ag0000001 (subgenome encoded but we use the GFF chrom column).
Proteins were pre-extracted with gffread (transcript-keyed); we pick the longest
protein per gene as canonical.
"""
from __future__ import annotations

import argparse
import bisect
import gzip
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/mnt/7302share/fast_ysp/U7_GWAS")
GFF = Path("/mnt/nvme/oat_raw/oat_ot3098_v2.gff3")  # gunzipped Ensembl OT3098 v2
PEP = Path("/mnt/nvme/oat_raw/oat_ot3098_v2.pep.fa")  # gffread -y (transcript-keyed)
MERGED = ROOT / "data/processed/avena_sativa_logical/merged"  # PLINK1 bed/bim/fam, chrom 1A..7D
OUT = ROOT / "results/phase7/bio_oat"
SUBS = ("A", "C", "D")
CHROMS = {s: [f"{i}{s}" for i in range(1, 8)] for s in SUBS}
ALL_CHROMS = {c for s in SUBS for c in CHROMS[s]}
MCSX_CODE = {"A": "aa", "C": "cc", "D": "dd"}

_GENE_ID_RE = re.compile(r"gene_id=([^;]+)")
_ID_RE = re.compile(r"ID=([^;]+)")
_PARENT_RE = re.compile(r"Parent=([^;]+)")


def _parse_gff():
    """Return (genes, gene2canonprot).
    genes: list of (gene_id, chrom, start, end, strand, sub).
    gene2canonprot: gene_id -> transcript_id of longest CDS (canonical protein)."""
    genes = []
    mrna2gene = {}
    cds_len = defaultdict(int)
    with open(GFF) as f:
        for line in f:
            if line.startswith("#"):
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) < 9 or p[0] not in ALL_CHROMS:
                continue
            ft = p[2]
            if ft == "gene":
                if "biotype=protein_coding" not in p[8]:
                    continue
                m = _GENE_ID_RE.search(p[8])
                if not m:
                    continue
                sub = p[0][-1]
                genes.append((m.group(1), p[0], int(p[3]), int(p[4]), p[6], sub))
            elif ft == "mRNA":
                mi = _ID_RE.search(p[8]); mp = _PARENT_RE.search(p[8])
                if mi and mp:
                    tid = mi.group(1).replace("transcript:", "")
                    gid = mp.group(1).replace("gene:", "")
                    mrna2gene[tid] = gid
            elif ft == "CDS":
                mp = _PARENT_RE.search(p[8])
                if mp:
                    tid = mp.group(1).replace("transcript:", "")
                    cds_len[tid] += int(p[4]) - int(p[3]) + 1
    # canonical protein = longest-CDS transcript per gene
    gene2trans = defaultdict(list)
    for tid, gid in mrna2gene.items():
        gene2trans[gid].append(tid)
    gene2canonprot = {g: max(ts, key=lambda t: cds_len[t]) for g, ts in gene2trans.items()}
    return genes, gene2canonprot


def _load_pep_by_transcript():
    """gffread pep keyed by 'transcript:AVESA....1Ag0000001.1' -> {transcript_id: seq}."""
    seqs, name, buf = {}, None, []
    with open(PEP) as f:
        for line in f:
            if line.startswith(">"):
                if name is not None:
                    seqs[name] = "".join(buf)
                name = line[1:].split()[0].replace("transcript:", "")
                buf = []
            else:
                buf.append(line.strip().rstrip("*"))
    if name is not None:
        seqs[name] = "".join(buf)
    return seqs


def _map_snps_to_windows(genes_ext, pos_sorted, idx_sorted):
    """genes_ext: list (gid, win_start, win_end) sorted by win_start.
    Returns {gid: [col_idx...]} where col_idx index into the subgenome X matrix."""
    starts = [g[1] for g in genes_ext]
    out = {g[0]: [] for g in genes_ext}
    for si, pos in enumerate(pos_sorted):
        j = bisect.bisect_right(starts, pos) - 1
        steps = 0
        while j >= 0 and steps < 3000:
            gid, ws, we = genes_ext[j]
            if pos > we and pos - ws > 2_000_000:
                break
            if ws <= pos <= we:
                out[gid].append(int(idx_sorted[si]))
            j -= 1
            steps += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flank-bp", type=int, default=2000)
    ap.add_argument("--min-snp", type=int, default=3)
    ap.add_argument("--out-dir", default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    print(f"=== OAT AACCDD ETL (body + ±{args.flank_bp}bp, min_snp={args.min_snp}) ===")

    genes, gene2canonprot = _parse_gff()
    print(f"  GFF: {len(genes)} protein-coding genes on 21 chroms "
          + " ".join(f"{s}={sum(1 for g in genes if g[5]==s)}" for s in SUBS))

    from homoeogwas.io import load_bed_hardcall
    bed = load_bed_hardcall(str(MERGED))
    X_all = np.asarray(bed.dosage, dtype=np.float32)  # (n_samples, n_variants)
    samples = [str(s) for s in np.asarray(bed.samples)]
    # bim: chrom + pos, in column order matching X_all
    bim_chrom, bim_pos = [], []
    with open(f"{MERGED}.bim") as f:
        for line in f:
            pp = line.rstrip("\n").split("\t")
            bim_chrom.append(pp[0]); bim_pos.append(int(pp[3]))
    bim_chrom = np.array(bim_chrom); bim_pos = np.array(bim_pos)
    print(f"  genotype: {X_all.shape} samples={len(samples)} total_SNP={X_all.shape[1]}")

    # group genes by chrom
    chrom_genes = defaultdict(list)
    for gid, chrom, gs, ge, strand, sub in genes:
        chrom_genes[chrom].append((gid, gs, ge, strand, sub))
    gene_meta = {}

    retained_union = set()
    for sub in SUBS:
        # subgenome column selection (all SNPs on this subgenome's chroms)
        sel = np.where(np.isin(bim_chrom, CHROMS[sub]))[0]
        Xs = np.ascontiguousarray(X_all[:, sel])
        sub_chrom = bim_chrom[sel]; sub_pos = bim_pos[sel]
        n_snp = sel.size
        print(f"  [{sub}] subgenome SNPs={n_snp}")
        np.save(out / f"X_{sub}.npy", Xs)

        for mode in ("body", "flank"):
            gene2col = {}
            for chrom in CHROMS[sub]:
                cmask = np.where(sub_chrom == chrom)[0]
                if cmask.size == 0:
                    continue
                cpos = sub_pos[cmask]; order = np.argsort(cpos)
                pos_sorted = cpos[order]; idx_sorted = cmask[order]  # idx into Xs
                gl = chrom_genes.get(chrom, [])
                if mode == "body":
                    ext = [(g[0], g[1], g[2]) for g in gl]
                else:
                    ext = [(g[0], max(1, g[1] - args.flank_bp), g[2] + args.flank_bp) for g in gl]
                ext.sort(key=lambda x: x[1])
                m = _map_snps_to_windows(ext, pos_sorted, idx_sorted)
                gene2col.update(m)
            # retained genes for this (sub, mode)
            gids = []; idxs = []; starts = []
            for chrom in CHROMS[sub]:
                for g in chrom_genes.get(chrom, []):
                    gid = g[0]; cols = gene2col.get(gid, [])
                    nb = len(cols)
                    meta = gene_meta.setdefault(gid, dict(gene_id=gid, chrom=chrom, start=g[1],
                                                          end=g[2], strand=g[3], sub=sub,
                                                          n_body=0, n_flank=0))
                    meta["n_body" if mode == "body" else "n_flank"] = nb
                    if nb >= args.min_snp:
                        gids.append(gid); idxs.append(np.array(cols, int)); starts.append(g[1])
            order = np.argsort(starts)
            gids = [gids[i] for i in order]
            idxs = [idxs[i] for i in order]
            starts = np.array(starts)[order] if len(starts) else np.array([], int)
            np.savez(out / f"snp_to_gene_{sub}_{mode}.npz",
                     gene_ids=np.array(gids), snp_idx=np.array(idxs, dtype=object),
                     starts=starts, samples=np.array(samples), allow_pickle=True)
            retained_union.update(gids)
            print(f"      mode={mode}: retained {len(gids)} genes")

    del X_all

    # gene metadata table
    with (out / "genes.tsv").open("w") as f:
        f.write("gene_id\tchrom\tstart\tend\tstrand\tsub\tn_snp_body\tn_snp_flank\tcanon_protein\n")
        for g in gene_meta.values():
            f.write(f"{g['gene_id']}\t{g['chrom']}\t{g['start']}\t{g['end']}\t{g['strand']}\t"
                    f"{g['sub']}\t{g['n_body']}\t{g['n_flank']}\t{gene2canonprot.get(g['gene_id'],'')}\n")

    # union PEP (canonical protein per retained gene, keyed by gene_id) + MCScanX gff
    pep = _load_pep_by_transcript()
    n_pep = 0
    with (out / "union_pep.fa").open("w") as fo:
        for gid in retained_union:
            tid = gene2canonprot.get(gid)
            if tid and tid in pep and pep[tid]:
                fo.write(f">{gid}\n{pep[tid]}\n"); n_pep += 1
    with (out / "union_mcscanx.gff").open("w") as f:
        for gid in retained_union:
            g = gene_meta[gid]
            n = g["chrom"][:-1]  # '1A' -> '1'
            sp = f"{MCSX_CODE[g['sub']]}{n}"  # aa1 / cc1 / dd1
            f.write(f"{sp}\t{gid}\t{g['start']}\t{g['end']}\n")
    print(f"  union retained={len(retained_union)} pep_written={n_pep}")

    summ = dict(species="Avena_sativa_AACCDD", flank_bp=args.flank_bp, min_snp=args.min_snp,
                n_genes=len(gene_meta), n_samples=len(samples),
                retained_union=len(retained_union), n_pep=n_pep,
                retained_per_sub_mode={
                    f"{s}_{m}": int(sum(1 for g in gene_meta.values()
                                        if g["sub"] == s and g[f"n_{m}"] >= args.min_snp))
                    for s in SUBS for m in ("body", "flank")})
    (out / "etl_summary.json").write_text(json.dumps(summ, indent=2))
    print(f"  DONE -> {out}\n  retained per sub/mode: {summ['retained_per_sub_mode']}")


if __name__ == "__main__":
    main()
