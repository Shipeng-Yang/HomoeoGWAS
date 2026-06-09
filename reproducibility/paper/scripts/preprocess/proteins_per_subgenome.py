#!/usr/bin/env python
"""Make per-subgenome protein FASTAs for `homoeogwas prep-homoeologs --method
diamond-rbh`, from a genome FASTA + GFF.

Steps:
  1. run ``gffread -y`` to get per-transcript proteins (external; pass the binary);
  2. collapse to one longest protein per gene (gene id = transcript id minus a
     trailing ``.tN``, or the GFF Parent if you adapt this);
  3. subset to the genes in each ``genes_<S>.tsv`` from ``prep-snps``;
  4. strip ``.``/``*`` (stop/gap) characters that DIAMOND rejects;
  5. write ``prot_<S>.faa`` with gene-id headers (matching the snp_to_gene NPZ).

Usage:
  proteins_per_subgenome.py --gffread <bin> --genome g.fa --gff genes.gff \
      --genes-template prep/genes_{S}.tsv --subgenomes A,B,C --out-dir prot/
"""
from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path


def read_fasta(path):
    h, seq = None, []
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                if h:
                    yield h, "".join(seq)
                h = line[1:].split()[0]
                seq = []
            else:
                seq.append(line.strip())
    if h:
        yield h, "".join(seq)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gffread", required=True, help="path to gffread binary")
    ap.add_argument("--genome", required=True)
    ap.add_argument("--gff", required=True)
    ap.add_argument("--genes-template", required=True,
                    help="genes_{S}.tsv template from prep-snps (use {S})")
    ap.add_argument("--subgenomes", required=True, help="comma list e.g. A,B,C")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tx-strip", default=r"\.t\d+$",
                    help="regex to turn a transcript id into a gene id")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tx_faa = out / "proteins_tx.faa"
    if not tx_faa.exists():
        subprocess.run([args.gffread, "-y", str(tx_faa), "-g", args.genome,
                        args.gff], check=True)

    # longest protein per gene, cleaned of stop/gap chars
    best: dict[str, str] = {}
    rx = re.compile(args.tx_strip)
    for tid, seq in read_fasta(tx_faa):
        gene = rx.sub("", tid)
        seq = seq.replace(".", "").replace("*", "")
        if gene not in best or len(seq) > len(best[gene]):
            best[gene] = seq

    import pandas as pd
    for s in args.subgenomes.split(","):
        s = s.strip()
        genes = set(pd.read_csv(args.genes_template.replace("{S}", s),
                                sep="\t", dtype={"gene_id": str})["gene_id"])
        n = 0
        with open(out / f"prot_{s}.faa", "w") as fh:
            for g in genes:
                if g in best:
                    fh.write(f">{g}\n{best[g]}\n")
                    n += 1
        print(f"[{s}] wrote {n}/{len(genes)} gene proteins -> prot_{s}.faa")


if __name__ == "__main__":
    main()
