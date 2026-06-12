# Worked example: homoeolog interaction in an allo-octoploid (strawberry)

A full `prep-snps → prep-homoeologs → interact` run on cultivated **strawberry**
(*Fragaria × ananassa*, allo-octoploid **AABBCCDD**, 2n = 8x = 56 = 7 base
chromosomes × 4 subgenomes A/B/C/D). It also shows the **≥4-subgenome rule**:
the interaction test is run over 2-/3-subgenome subsets, never a single 4-way
product (AGENTS.md §5).

## Starting material

- a reference GFF with `gene` features (NIHHS Seolhyang assembly);
- per-subgenome PLINK BEDs `subgenome_{A,B,C,D}` (genotypes split by subgenome);
- a protein FASTA derived from the genome + GFF (see *Preparing proteins* in
  `docs/interact_inputs.md`);
- a phenotype TSV.

## Subgenome map

28 chromosomes → 4 subgenomes, 7 each; the base chromosome (1…7) is `base_group`.
The `chrom` column must match **both** the GFF and the `.bim` (rename one side
first if they differ):

```
chrom   subgenome   base_group
1A      A           1
1B      B           1
1C      C           1
1D      D           1
2A      A           2
...
```

## Step 1 — SNP → gene (all four subgenomes)

```
homoeogwas prep-snps \
  --gff genes.gff --subgenome-map sgmap.tsv \
  --bed A=geno/subgenome_A --bed B=geno/subgenome_B \
  --bed C=geno/subgenome_C --bed D=geno/subgenome_D \
  --feature gene --id-attr ID --flank-bp 2000 --min-snp 2 \
  --out-dir interact_prep
```

`prep-snps` is fully N-subgenome — you get `snp_to_gene_{A,B,C,D}.npz` and
`genes_{A,B,C,D}.tsv`. Note that a SNP-array panel can be very uneven across
subgenomes (a probe set designed on a diploid progenitor skews SNPs toward one
subgenome), which limits how many homoeolog groups carry SNPs in the others —
report this rather than over-interpreting a sparse subgenome.

## Step 2 — homoeolog table for a subset

For 4 subgenomes you build the gene table and run the test over **subsets**.
Triad over A/B/C (or any 3), and/or pairwise over any 2:

```
homoeogwas prep-homoeologs --mode triad --subgenomes A,B,C \
  --genes interact_prep/genes_{S}.tsv --method diamond-rbh \
  --proteins A=prot_A.faa --proteins B=prot_B.faa --proteins C=prot_C.faa \
  --restrict-base-group --subgenome-map sgmap.tsv --diamond /path/to/diamond-2.1.x \
  --out interact_prep/triads_ABC.tsv
```

## Step 3 — interaction scan (per subset)

```yaml
interact:
  mode: triad
  subgenomes: [A, B, C]
  genotype:    {A: geno/subgenome_A, B: geno/subgenome_B, C: geno/subgenome_C}
  snp_to_gene: {A: interact_prep/snp_to_gene_A.npz,
                B: interact_prep/snp_to_gene_B.npz,
                C: interact_prep/snp_to_gene_C.npz}
  triads: interact_prep/triads_ABC.tsv
  phenotype: pheno.tsv
  sample_col: sample_id
  trait: my_trait
  burden: {cap: 150, min_snp: 2}
  grm: {method: grm_from_X}
  calibration: {perm_B: 200}
outputs: {out_dir: results_interact/ABC}
```

```
homoeogwas interact -c interact_ABC.yaml --n-jobs 16
```

Repeat for the other subsets (e.g. `A,B,D`) or pairwise (`A,B`), then aggregate
the ACAT omnibus p-values across subsets. A single 4-way product is deliberately
not offered — at realistic sample sizes it is far too sparse.

The MCP server / agent skill does this subset orchestration automatically: give
it the four subgenomes and it runs the supported 2-/3-way subsets for you.
