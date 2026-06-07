#!/usr/bin/env python
"""Strawberry ASE -> quartet (Step C1a): extract ASE-pair protein sequences.

Dataset S8 (Fan 2022) = 12,503 homoeolog ASE pairs scored on the FL15.89
reference genotype (4 reps), F12 (F-haplotype) vs Bea (B-haplotype) diploid
references. ratio_m = continuous F-vs-B expression-bias ratio; padjust =
significance. This is reference-genotype ASE, NOT a 196-individual population
trait; framing is an external functional-annotation layer on the Camarosa
quartets, same caveat level as the wheat CS expression atlas.

ASE gene ids match F12/Bea protein headers 100% after stripping '-mRNA-N'.
We write the F-copy and B-copy protein FASTAs (longest isoform per gene) so they
can be diamond-blastp'd to Camarosa for sequence-based ortholog mapping (F12
coords are NOT aligned to Camarosa: only ~34% land in a gene span).
"""
import gzip
import re
from pathlib import Path

import pandas as pd

PROJ = Path("/mnt/7302share/fast_ysp/U7_GWAS")
AD = Path("/mnt/a080share/ysp/草莓/711之前的/NP文章eQTL")
ASE = AD / "List of allele-specific expressed genes..csv"
OUT = PROJ / "results/phase7/bio_strawberry"
OUT.mkdir(parents=True, exist_ok=True)


def load_prots(fa):
    """gene_id (header up to -mRNA-N) -> longest protein seq."""
    best = {}
    cur, buf = None, []
    with gzip.open(fa, "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                if cur is not None:
                    g = re.sub(r"-mRNA-\d+$", "", cur)
                    s = "".join(buf).replace(".", "").replace("*", "")
                    if g not in best or len(s) > len(best[g]):
                        best[g] = s
                cur = line[1:].split()[0]
                buf = []
            else:
                buf.append(line.strip())
        if cur is not None:
            g = re.sub(r"-mRNA-\d+$", "", cur)
            s = "".join(buf).replace(".", "").replace("*", "")
            if g not in best or len(s) > len(best[g]):
                best[g] = s
    return best


df = pd.read_csv(ASE)
df = df[["F12 gene", "Bea gene", "pvalue", "padjust", "chr_ID", "CHR",
         "ratio_m"]].dropna(subset=["F12 gene", "Bea gene"])
df.to_csv(OUT / "ase_pairs.tsv", sep="\t", index=False)
print(f"[ase] {len(df)} ASE homoeolog pairs")

f12 = load_prots(AD / "genomea_anno/F12.standard.proteins.fasta.gz")
bea = load_prots(AD / "genomea_anno/Bea.standard.proteins.fasta.gz")
print(f"[prot] F12 {len(f12)} genes, Bea {len(bea)} genes")

for tag, col, db in [("F", "F12 gene", f12), ("B", "Bea gene", bea)]:
    ids = sorted(set(df[col].astype(str)))
    n = 0
    with open(OUT / f"ase_{tag}_pep.fa", "w") as out:
        for gid in ids:
            seq = db.get(gid)
            if not seq:
                continue
            out.write(f">{gid}\n")
            for i in range(0, len(seq), 60):
                out.write(seq[i:i + 60] + "\n")
            n += 1
    print(f"[{tag}] {n}/{len(ids)} ASE {tag}-copy genes written -> ase_{tag}_pep.fa")
print(f"[done] -> {OUT}")
