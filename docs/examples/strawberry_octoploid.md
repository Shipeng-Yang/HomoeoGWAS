# Worked example: unified group omniB in allo-octoploid strawberry

Cultivated strawberry (*Fragaria × ananassa*) is an allo-octoploid with four
subgenomes (AABBCCDD). HomoeoGWAS represents each A/B/C/D homoeolog group as a
four-node graph and derives its six pair edges: AB, AC, AD, BC, BD and CD. The
group statistic is ACAT over those edge omniB p-values. It is not a fourth-order
interaction coefficient.

## Inputs

- reference GFF with chromosome names matching each PLINK `.bim`;
- per-subgenome PLINK prefixes `geno/subgenome_{A,B,C,D}`;
- a fingerprint-bound `snp_to_gene_<S>.npz` for each subgenome;
- one wide table with `group_id,gene_A,gene_B,gene_C,gene_D`;
- phenotype TSV with string sample identifiers.

Build the mappings for all four subgenomes in one call:

```bash
homoeogwas prep-snps \
  --gff genes.gff --subgenome-map sgmap.tsv \
  --bed A=geno/subgenome_A --bed B=geno/subgenome_B \
  --bed C=geno/subgenome_C --bed D=geno/subgenome_D \
  --feature gene --id-attr ID --flank-bp 2000 --min-snp 3 \
  --out-dir interact_prep
```

For publication work, build the wide group table from curated synteny or an
orthology table. Each row must contain exactly one gene copy per subgenome and a
stable unique `group_id`.

## Canonical interaction config

Agents generate this config as
`<out_dir>/configs/interact.generated.group.omnib.yaml`; users normally provide
only the biological paths above.

```yaml
interact:
  mode: group
  subgenomes: [A, B, C, D]
  groups: interact_prep/groups_ABCD.tsv
  statistic: omniB
  hypothesis_unit: group
  subset_order: 2
  family_scope: primary_only
  primary_transform: INT
  primary_multiplicity: bootstrap_minp
  genotype:
    A: geno/subgenome_A
    B: geno/subgenome_B
    C: geno/subgenome_C
    D: geno/subgenome_D
  snp_to_gene:
    A: interact_prep/snp_to_gene_A.npz
    B: interact_prep/snp_to_gene_B.npz
    C: interact_prep/snp_to_gene_C.npz
    D: interact_prep/snp_to_gene_D.npz
  phenotype: pheno.tsv
  sample_col: sample_id
  trait: my_trait
  burden: {cap: 150, min_snp: 3, maf_min: 0.01, n_pc: 3}
  grm: {method: grm_from_X, maf_min: 0.01, scope: all_subgenomes}
  calibration: {method: bootstrap, B: 2000, seed: 2026}
outputs:
  out_dir: results_interact/strawberry_group
  full_ranking: true
```

Validate before the long run, then audit the result:

```bash
homoeogwas validate -c results_interact/strawberry_group/configs/interact.generated.group.omnib.yaml
homoeogwas interact -c results_interact/strawberry_group/configs/interact.generated.group.omnib.yaml --n-jobs 16
homoeogwas audit results_interact/strawberry_group
```

With `hypothesis_unit: group`, a formal hit supports “encoding-robust omnibus
pairwise interaction evidence within a homoeolog group”. The six edge and three
component p-values localize that evidence; they are not six additional
discoveries. Neither a group hit nor an edge hit establishes a physical complex,
a causal mechanism, or an A×B×C×D interaction.
