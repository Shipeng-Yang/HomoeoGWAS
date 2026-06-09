# Worked example: homoeolog interaction in an allohexaploid

A full `prep-snps → prep-homoeologs → interact` run on **Jerusalem artichoke**
(*Helianthus tuberosus*, allohexaploid **AABBCC**, 2n = 6x = 102 = 17 base
chromosomes × 3 subgenomes), tuber-quality traits.

## Starting material

- A reference GFF with `gene` features, chromosomes named `CM060351.1 …`
  (NCBI/GenBank style).
- Per-subgenome PLINK BEDs `subgenome_{A,B,C}.{bed,bim,fam}` (the genotypes
  split by subgenome; ~0.5 M SNPs each).
- The genome (or per-subgenome) **protein FASTA**, here extracted from the
  assembly with `gffread -y` and reduced to one longest protein per gene
  (header = gene id, matching the GFF `ID=`).
- A phenotype TSV (`IID` + trait columns).

## The subgenome map

The 102 chromosomes group into 3 subgenomes, 34 each; the base chromosome each
descends from is the `base_group` (Chr1 … Chr17):

```
chrom        subgenome    base_group
CM060351.1   A            Chr1
CM060352.1   A            Chr1
CM060353.1   C            Chr1
CM060355.1   B            Chr1
...
```

## Step 1 — SNP → gene

```
homoeogwas prep-snps \
  --gff Helianthus_tuberosus.genes.ncbi.gff \
  --subgenome-map ja_subgenome_map.tsv \
  --bed A=geno/subgenome_A --bed B=geno/subgenome_B --bed C=geno/subgenome_C \
  --feature gene --id-attr ID --flank-bp 2000 --min-snp 2 \
  --out-dir interact_prep
```

Result (real run): ~10.3–10.6 k genes per subgenome carry ≥ 2 SNPs (≈ 56 k of
each subgenome's ~0.5 M SNPs fall within a gene ± 2 kb — sparse, as expected for
GBS). Writes `snp_to_gene_{A,B,C}.npz` and `genes_{A,B,C}.tsv`.

## Step 2 — homoeolog triads (DIAMOND RBH, base-group restricted)

Proteins for the SNP-carrying genes were split per subgenome, then:

```
homoeogwas prep-homoeologs --mode triad --subgenomes A,B,C \
  --genes interact_prep/genes_{S}.tsv \
  --method diamond-rbh \
  --proteins A=prot_A.faa --proteins B=prot_B.faa --proteins C=prot_C.faa \
  --restrict-base-group --subgenome-map ja_subgenome_map.tsv --threads 32 \
  --out interact_prep/triads.tsv
```

`--restrict-base-group` keeps only A/B/C triples whose three genes sit on the
same base chromosome (Chr1…Chr17), so paralogs on non-homologous chromosomes are
not mistaken for homoeologs. Output `triads.tsv` has columns
`gene_A, gene_B, gene_C`.

> For a publication-grade analysis, build orthogroups with a dedicated tool
> (OrthoFinder / synteny) and feed them via `--from-table` instead; DIAMOND RBH
> here is the reproducible convenience baseline.

## Step 3 — interaction scan

```yaml
interact:
  mode: triad
  subgenomes: [A, B, C]
  genotype:    {A: geno/subgenome_A, B: geno/subgenome_B, C: geno/subgenome_C}
  snp_to_gene: {A: interact_prep/snp_to_gene_A.npz,
                B: interact_prep/snp_to_gene_B.npz,
                C: interact_prep/snp_to_gene_C.npz}
  triads: interact_prep/triads.tsv
  phenotype: pheno_tuber_quality.tsv
  sample_col: IID
  trait: number_INT
  burden: {cap: 150, min_snp: 2}
  grm: {method: grm_from_X}
  calibration: {perm_B: 200}
outputs: {out_dir: results_interact/number_INT}
```

```
homoeogwas interact -c interact_number_INT.yaml --n-jobs 32
```

This tests, for each homoeolog triad, whether the A/B/C gene burdens interact
(burden-product) in shaping the trait, with an ACAT omnibus over triads and
permutation calibration — a question the standard per-SNP `fit` scan cannot ask.

The complete driver used to produce this example is in the project's
reproducibility scripts.
