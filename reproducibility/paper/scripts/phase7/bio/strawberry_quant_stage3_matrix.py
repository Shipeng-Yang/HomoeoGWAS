#!/usr/bin/env python
"""Stage 3: assemble gene x sample expression matrix from salmon quants.

Reads results/phase7/bio_strawberry/salmon_quant/<RUN>/quant.sf for every run in
the manifest, sums transcript NumReads (and TPM) to the gene level (transcript id
= <gene>.tN), and writes a gene x sample matrix. Technical replicates (same
genotype, e.g. the FL15_89_25 reference's 4 runs) are summed at the count level.

For population HEB we work with WITHIN-quartet proportions per sample (Stage 4),
which are ratios inside one library and so are robust to library-size differences;
cross-sample DESeq size factors are emitted too for any cross-gene analysis.
Runs only after Stage 2 (needs quant.sf files).
"""
import csv
from collections import defaultdict
from pathlib import Path

PROJ = Path("/mnt/7302share/fast_ysp/U7_GWAS")
IN = PROJ / "results/phase7/bio_strawberry"
QUANT = IN / "salmon_quant"
MANIFEST = PROJ / "data/raw/strawberry/rnaseq_PRJNA787565/sample_manifest.tsv"


def gene_of(txid):
    return txid.rsplit(".t", 1)[0] if ".t" in txid else txid


# run -> genotype (for tech-rep merge)
run2geno = {}
with open(MANIFEST) as fh:
    for r in csv.DictReader(fh, delimiter="\t"):
        run2geno[r["run"]] = r["genotype"]

# read each quant.sf -> gene counts
geno_counts = defaultdict(lambda: defaultdict(float))  # genotype -> gene -> reads
geno_tpm = defaultdict(lambda: defaultdict(float))
genes = set()
n_ok = 0
for run, geno in run2geno.items():
    sf = QUANT / run / "quant.sf"
    if not sf.exists():
        continue
    with open(sf) as fh:
        next(fh)
        for line in fh:
            c = line.rstrip("\n").split("\t")
            g = gene_of(c[0])
            geno_counts[geno][g] += float(c[4])   # NumReads
            geno_tpm[geno][g] += float(c[3])       # TPM (summed to gene)
            genes.add(g)
    n_ok += 1
print(f"[matrix] {n_ok} runs read -> {len(geno_counts)} genotypes, {len(genes)} genes")

genotypes = sorted(geno_counts)
genes = sorted(genes)


def write_matrix(path, table):
    with open(path, "w") as o:
        w = csv.writer(o, delimiter="\t")
        w.writerow(["gene", *genotypes])
        for g in genes:
            w.writerow([g, *[f"{table[s].get(g, 0.0):.3f}" for s in genotypes]])


write_matrix(IN / "expr_counts_gene_x_genotype.tsv", geno_counts)
write_matrix(IN / "expr_tpm_gene_x_genotype.tsv", geno_tpm)
print(f"[done] -> expr_counts/expr_tpm matrices ({len(genes)} x {len(genotypes)})")
