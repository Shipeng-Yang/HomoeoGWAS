#!/usr/bin/env python3
"""OAT Step 2a: build FULL-proteome MCScanX backbone (all 64461 genes, not just
retained), so de-novo homoeolog triads are computed on genome-wide synteny and only
THEN intersected with SNP-retained burden genes (mirrors wheat's genome-wide curated
HCTriads logic). Writes full_pep.fa (canonical protein per gene, keyed by gene_id)
and full_mcscanx.gff (sp code aa/cc/dd + chrom number)."""
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

ROOT = Path("/mnt/7302share/fast_ysp/U7_GWAS")
GFF = Path("/mnt/nvme/oat_raw/oat_ot3098_v2.gff3")
PEP = Path("/mnt/nvme/oat_raw/oat_ot3098_v2.pep.fa")
OUT = ROOT / "results/phase7/bio_oat"
SUBS = ("A", "C", "D")
CHROMS = {c for s in SUBS for c in (f"{i}{s}" for i in range(1, 8))}
MCSX_CODE = {"A": "aa", "C": "cc", "D": "dd"}

_GENE_ID_RE = re.compile(r"gene_id=([^;]+)")
_ID_RE = re.compile(r"ID=([^;]+)")
_PARENT_RE = re.compile(r"Parent=([^;]+)")


def main():
    genes = {}  # gid -> (chrom, start, end, sub)
    mrna2gene = {}
    cds_len = defaultdict(int)
    with open(GFF) as f:
        for line in f:
            if line.startswith("#"):
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) < 9 or p[0] not in CHROMS:
                continue
            ft = p[2]
            if ft == "gene":
                if "biotype=protein_coding" not in p[8]:
                    continue
                m = _GENE_ID_RE.search(p[8])
                if m:
                    genes[m.group(1)] = (p[0], int(p[3]), int(p[4]), p[0][-1])
            elif ft == "mRNA":
                mi = _ID_RE.search(p[8]); mp = _PARENT_RE.search(p[8])
                if mi and mp:
                    mrna2gene[mi.group(1).replace("transcript:", "")] = mp.group(1).replace("gene:", "")
            elif ft == "CDS":
                mp = _PARENT_RE.search(p[8])
                if mp:
                    cds_len[mp.group(1).replace("transcript:", "")] += int(p[4]) - int(p[3]) + 1
    print(f"genes={len(genes)} transcripts={len(mrna2gene)}")

    gene2trans = defaultdict(list)
    for tid, gid in mrna2gene.items():
        gene2trans[gid].append(tid)
    gene2canon = {g: max(ts, key=lambda t: cds_len[t]) for g, ts in gene2trans.items()}

    # load pep transcripts
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

    n_pep = 0
    with (OUT / "full_pep.fa").open("w") as fo:
        for gid in genes:
            tid = gene2canon.get(gid)
            if tid and seqs.get(tid):
                fo.write(f">{gid}\n{seqs[tid]}\n"); n_pep += 1
    with (OUT / "full_mcscanx.gff").open("w") as f:
        for gid, (chrom, gs, ge, sub) in genes.items():
            sp = f"{MCSX_CODE[sub]}{chrom[:-1]}"
            f.write(f"{sp}\t{gid}\t{gs}\t{ge}\n")
    print(f"wrote full_pep.fa ({n_pep} seqs) + full_mcscanx.gff ({len(genes)} genes) -> {OUT}")


if __name__ == "__main__":
    main()
