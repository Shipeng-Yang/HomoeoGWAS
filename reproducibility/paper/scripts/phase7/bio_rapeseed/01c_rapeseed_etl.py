#!/usr/bin/env python3
"""Phase B Step 1 (rapeseed): full A01-A10 / C01-C09 ETL + dual-mode body/±2kb.

Cross-species replication adaptation of cotton Phase A 01b. Key differences:
  - chrom map: BIM `chrA01`/`chrC01` <-> GFF `NC_027757.2`/`NC_027767.2` via TSV
  - NCBI GFF v2.0 ID chain: protein_id (NP_/XP_) -> CDS.Parent (mRNA) -> mRNA.Parent (gene)
    -> gene_id stripped of "gene-" prefix (e.g. "LOC106345161")
  - PEP from NCBI GCF_000686985.2_protein.faa.gz (123465 proteins / 98062 genes:
    multiple transcripts per gene -> we take ONE canonical protein per gene
    = first-encountered CDS protein_id with longest CDS sum)
  - 11 traits with per-trait NA: alignment handled in 04c, not here
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
import pandas as pd

ROOT = Path("/mnt/7302share/fast_ysp/U7_GWAS")
GFF = ROOT / "data/reference/rapeseed/GCF_000686985.2_Bra_napus_v2.0_genomic.gff.gz"
PEP = Path("/mnt/nvme/rapeseed_mcscanx/protein.faa")
CHROM_MAP = ROOT / "data/reference/rapeseed/chrom_map_horvath_to_ncbi.tsv"
BIM_A = ROOT / "data/processed/rapeseed/horvath/A/all.bim"
BIM_C = ROOT / "data/processed/rapeseed/horvath/C/all.bim"
OUT = ROOT / "results/phase7/bio_rapeseed"

_ID_GENE_RE = re.compile(r"ID=gene-([^;]+)")
_ID_RNA_RE = re.compile(r"ID=rna-([^;]+)")
_PARENT_RNA_RE = re.compile(r"Parent=rna-([^;]+)")
_PARENT_GENE_RE = re.compile(r"Parent=gene-([^;]+)")
_PROTID_RE = re.compile(r"protein_id=([^;]+)")
A_CHROMS = [f"chrA{i:02d}" for i in range(1, 11)]
C_CHROMS = [f"chrC{i:02d}" for i in range(1, 10)]
ALL_PANEL_CHROMS = set(A_CHROMS + C_CHROMS)


def _load_chrom_map():
    df = pd.read_csv(CHROM_MAP, sep="\t")
    return {row.fasta_chrom: row.panel_chrom for row in df.itertuples()}


def _parse_gff_full():
    """Parse rapeseed GFF -> (gene_table, gene2protein primary).
    gene_table rows: (panel_chrom, gene_id_LOC, start, end, strand, sub).
    gene2protein: gene_id -> protein_id (largest cumulative CDS length per gene)."""
    nc2panel = _load_chrom_map()
    genes = []  # (panel_chrom, gene_id, start, end, strand, sub)
    rna2gene = {}  # transcript ID like XM_013814943.2 -> gene_id LOC...
    cds_len = defaultdict(int)  # protein_id -> total CDS span (sum of CDS line lengths)
    cds2rna = {}  # protein_id -> rna transcript ID (XM_)
    with gzip.open(GFF, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) < 9:
                continue
            nc = p[0]
            if nc not in nc2panel:
                continue
            panel_c = nc2panel[nc]
            sub = "A" if panel_c.startswith("chrA") else "C"
            ft = p[2]
            if ft == "gene":
                m = _ID_GENE_RE.search(p[8])
                if not m or "gene_biotype=protein_coding" not in p[8]:
                    continue
                genes.append((panel_c, m.group(1), int(p[3]), int(p[4]), p[6], sub))
            elif ft == "mRNA":
                m_rna = _ID_RNA_RE.search(p[8]); m_par = _PARENT_GENE_RE.search(p[8])
                if m_rna and m_par:
                    rna2gene[m_rna.group(1)] = m_par.group(1)
            elif ft == "CDS":
                m_pid = _PROTID_RE.search(p[8]); m_par = _PARENT_RNA_RE.search(p[8])
                if m_pid and m_par:
                    cds2rna[m_pid.group(1)] = m_par.group(1)
                    cds_len[m_pid.group(1)] += int(p[4]) - int(p[3]) + 1
    # build gene -> best protein (largest cumulative CDS length)
    gene2prots = defaultdict(list)
    for pid, rna in cds2rna.items():
        g = rna2gene.get(rna)
        if g:
            gene2prots[g].append(pid)
    gene2protein = {}
    for g, prots in gene2prots.items():
        gene2protein[g] = max(prots, key=lambda p: cds_len[p])
    return genes, gene2protein, cds_len


def _parse_bim_snps(bim_path, chroms):
    by_chrom = {c: ([], []) for c in chroms}
    with open(bim_path) as f:
        for i, line in enumerate(f):
            p = line.rstrip("\n").split("\t")
            if p[0] in by_chrom:
                by_chrom[p[0]][0].append(int(p[3])); by_chrom[p[0]][1].append(i)
    return {c: (np.array(v[0], int), np.array(v[1], int)) for c, v in by_chrom.items()}


def _map_snps_to_windows(genes_ext, pos_sorted, snp_idx_sorted):
    starts = [g[1] for g in genes_ext]
    out = {g[0]: [] for g in genes_ext}
    for si, pos in enumerate(pos_sorted):
        j = bisect.bisect_right(starts, pos) - 1
        steps = 0
        while j >= 0 and steps < 2000:
            gid, ws, we = genes_ext[j]
            if pos > we and pos - ws > 1_000_000:
                break
            if ws <= pos <= we:
                out[gid].append(int(snp_idx_sorted[si]))
            j -= 1; steps += 1
    return out


def _extract_pep_subset(pep_path, retained_proteins_to_genes):
    """Yield (gene_id, seq) for proteins in the retained set; gene_id from crosswalk.
    PEP header: '>NP_xxx.1 ...' or '>XP_xxx.1 ...'."""
    cur_pid, cur_seq = None, []
    with open(pep_path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if cur_pid is not None and cur_pid in retained_proteins_to_genes:
                    yield retained_proteins_to_genes[cur_pid], "".join(cur_seq)
                cur_pid = line[1:].split()[0]
                cur_seq = []
            else:
                cur_seq.append(line)
    if cur_pid is not None and cur_pid in retained_proteins_to_genes:
        yield retained_proteins_to_genes[cur_pid], "".join(cur_seq)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flank-bp", type=int, default=2000)
    ap.add_argument("--min-snp", type=int, default=3)
    ap.add_argument("--out-dir", default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    print(f"=== Phase B rapeseed ETL: 13(=10A+9C)-chrom dual-mode (body + ±{args.flank_bp}bp) ===")

    raw_genes, gene2protein, _ = _parse_gff_full()
    print(f"  GFF: {len(raw_genes)} protein-coding genes on chrA01-A10/chrC01-C09 "
          f"({sum(1 for g in raw_genes if g[5]=='A')} A / {sum(1 for g in raw_genes if g[5]=='C')} C)")
    print(f"  gene->protein crosswalk: {len(gene2protein)} genes mapped to a canonical protein")

    a_snps = _parse_bim_snps(BIM_A, A_CHROMS)
    c_snps = _parse_bim_snps(BIM_C, C_CHROMS)
    total_a = sum(v[0].size for v in a_snps.values())
    total_c = sum(v[0].size for v in c_snps.values())
    print(f"  BIM: A {total_a} SNPs, C {total_c} SNPs")

    # per-chrom: body / flank window maps
    chrom_genes = {c: [] for c in ALL_PANEL_CHROMS}
    for c, gid, gs, ge, strand, sub in raw_genes:
        chrom_genes[c].append((gid, gs, ge, strand))
    map_body, map_flank = {}, {}
    for c in sorted(ALL_PANEL_CHROMS):
        gs_list = sorted(chrom_genes[c], key=lambda x: x[1])
        sub = "A" if c in A_CHROMS else "C"
        pos_arr, snp_idx_arr = (a_snps if sub == "A" else c_snps)[c]
        order = np.argsort(pos_arr)
        pos_sorted = pos_arr[order]; snp_idx_sorted = snp_idx_arr[order]
        ext_body = [(g[0], g[1], g[2]) for g in gs_list]
        m_body = _map_snps_to_windows(ext_body, pos_sorted, snp_idx_sorted)
        ext_flank = [(g[0], max(1, g[1] - args.flank_bp), g[2] + args.flank_bp) for g in gs_list]
        ext_flank.sort(key=lambda x: x[1])
        m_flank = _map_snps_to_windows(ext_flank, pos_sorted, snp_idx_sorted)
        map_body.update(m_body); map_flank.update(m_flank)

    gene_meta = {}
    for c, gid, gs, ge, strand, sub in raw_genes:
        n_b = len(map_body.get(gid, [])); n_f = len(map_flank.get(gid, []))
        gene_meta[gid] = dict(gene_id=gid, chrom=c, start=gs, end=ge, strand=strand, sub=sub,
                              n_snp_body=n_b, n_snp_flank=n_f,
                              retained_body=int(n_b >= args.min_snp),
                              retained_flank=int(n_f >= args.min_snp),
                              protein_id=gene2protein.get(gid, ""))

    with (out / "genes.tsv").open("w") as f:
        f.write("gene_id\tprotein_id\tchrom\tstart\tend\tstrand\tsub\tn_snp_body\tn_snp_flank\t"
                "retained_body\tretained_flank\n")
        for g in gene_meta.values():
            f.write(f"{g['gene_id']}\t{g['protein_id']}\t{g['chrom']}\t{g['start']}\t{g['end']}\t"
                    f"{g['strand']}\t{g['sub']}\t{g['n_snp_body']}\t{g['n_snp_flank']}\t"
                    f"{g['retained_body']}\t{g['retained_flank']}\n")

    retained_body = [gid for gid, g in gene_meta.items() if g["retained_body"]]
    retained_flank = [gid for gid, g in gene_meta.items() if g["retained_flank"]]
    union_retained = set(retained_body) | set(retained_flank)
    print(f"  RETAINED ≥{args.min_snp} SNP:")
    print(f"    body: total={len(retained_body)} A={sum(1 for g in retained_body if gene_meta[g]['sub']=='A')} "
          f"C={sum(1 for g in retained_body if gene_meta[g]['sub']=='C')}")
    print(f"    ±{args.flank_bp}bp: total={len(retained_flank)} A={sum(1 for g in retained_flank if gene_meta[g]['sub']=='A')} "
          f"C={sum(1 for g in retained_flank if gene_meta[g]['sub']=='C')}")
    print(f"    union (MCScanX backbone): {len(union_retained)}")

    def _save_map(name, retained, source):
        keys = [gid for gid in retained]
        arrs = np.array([np.array(source[gid], int) for gid in keys], dtype=object)
        subs = np.array([gene_meta[gid]["sub"] for gid in keys])
        chroms = np.array([gene_meta[gid]["chrom"] for gid in keys])
        np.savez(out / name, gene_ids=np.array(keys), snp_idx=arrs, sub=subs, chrom=chroms,
                 allow_pickle=True)
    _save_map("snp_to_gene_body.npz", retained_body, map_body)
    _save_map(f"snp_to_gene_flank{args.flank_bp}bp.npz", retained_flank, map_flank)

    # union PEP: protein_id -> gene_id reverse lookup (only retained genes)
    prot_to_gene_retained = {gene_meta[g]["protein_id"]: g for g in union_retained
                              if gene_meta[g]["protein_id"]}
    n_pep = 0
    with (out / "union_pep.fa").open("w") as fout:
        for gid, seq in _extract_pep_subset(PEP, prot_to_gene_retained):
            fout.write(f">{gid}\n{seq}\n"); n_pep += 1
    # union MCScanX gff (label: aaXX for A subgenome, ccXX for C; chrom 2-digit suffix)
    with (out / "union_mcscanx.gff").open("w") as f:
        for gid in union_retained:
            g = gene_meta[gid]
            code = "aa" if g["sub"] == "A" else "cc"
            sp = f"{code}{g['chrom'][4:]}"  # chrA01 -> aa01, chrC01 -> cc01
            f.write(f"{sp}\t{gid}\t{g['start']}\t{g['end']}\n")
    print(f"  wrote union_pep.fa ({n_pep} seqs; expected ~{len(union_retained)} retained genes) + union_mcscanx.gff")

    summ = dict(flank_bp=args.flank_bp, min_snp=args.min_snp,
                n_genes_total=len(gene_meta),
                n_genes_with_snp_body=sum(1 for g in gene_meta.values() if g["n_snp_body"] > 0),
                n_genes_with_snp_flank=sum(1 for g in gene_meta.values() if g["n_snp_flank"] > 0),
                pct_genes_with_snp_body=round(
                    sum(1 for g in gene_meta.values() if g["n_snp_body"] > 0) / len(gene_meta) * 100, 2),
                pct_genes_with_snp_flank=round(
                    sum(1 for g in gene_meta.values() if g["n_snp_flank"] > 0) / len(gene_meta) * 100, 2),
                retained_body_total=len(retained_body), retained_flank_total=len(retained_flank),
                union_retained=len(union_retained), n_pep_extracted=n_pep,
                bim_total_A=total_a, bim_total_C=total_c)
    (out / "etl_summary.json").write_text(json.dumps(summ, indent=2))
    print(f"  ETL SUMMARY: body coverage {summ['pct_genes_with_snp_body']}% / "
          f"±{args.flank_bp}bp coverage {summ['pct_genes_with_snp_flank']}%")
    print(f"  DONE -> {out}")


if __name__ == "__main__":
    main()
