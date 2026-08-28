# Building the inputs for `homoeogwas interact`

`homoeogwas interact` runs a gene-resolution homoeolog-pair / triad interaction
test. The default statistic is **omniB**, an encoding-robust ACAT combination of
minor-allele burden-product, PC1×PC1 and low-rank kernel-Hadamard components,
calibrated by a kinship-preserving parametric bootstrap. The legacy
REF-oriented burden-product + permutation path remains available by explicit
configuration. Both need two preprocessed inputs per analysis that the rest of
the pipeline does not produce:

1. a **`snp_to_gene` NPZ** per subgenome — which SNPs belong to which gene;
2. a **master homoeolog-group TSV** — one `group_id` plus one `gene_<S>` column
   per subgenome. Legacy pair and triad tables are adapted automatically.

Two subcommands build them from standard files, so you never have to hand-craft
either:

```
homoeogwas prep-snps        # -> snp_to_gene_<S>.npz  + genes_<S>.tsv
homoeogwas prep-homoeologs  # -> triads.tsv (or pairs.tsv)
```

Everything is keyed off one small **subgenome map** you supply once.

The workflow is crop-agnostic — the only thing that changes between species is the
subgenome map and how many subgenomes you list.

---

## What you need (any crop)

| input | used by | notes |
|---|---|---|
| **reference GFF/GTF** with `gene` features | prep-snps | chromosome names must match the `.bim`; gene id via `--id-attr` |
| **per-subgenome PLINK BED** (`.bed/.bim/.fam`) | prep-snps, interact | your genotypes split by subgenome (`homoeogwas split` produces these) |
| **subgenome map TSV** (`chrom,subgenome[,base_group]`) | both | see below; one row per chromosome |
| **phenotype TSV** (`sample_id` + trait columns) | interact | |
| **per-subgenome protein FASTA** | prep-homoeologs `--method diamond-rbh` only | headers = gene ids matching the GFF; usually extracted from the **genome FASTA + GFF** (see *Preparing proteins* below) |
| **an orthology table** | prep-homoeologs `--from-table` only | OrthoFinder/synteny/etc.; the alternative to proteins+DIAMOND |

So the only "extra" file beyond a standard GWAS setup is either a protein FASTA
(which you derive from your genome FASTA + GFF) or an orthology table — you do
**not** need the genome FASTA itself for `prep-snps`/`interact`, only to make the
proteins for the DIAMOND path.

`prep-snps` binds every `snp_to_gene_<S>.npz` to the exact source `.bim`
using a SHA-256 fingerprint. `interact` refuses a legacy/unverified NPZ or a
mapping built from a different BIM/variant order; rerun `prep-snps` with the
analysis BED to repair it.

## Ploidy — one group engine, different graph sizes

| copies in each group (`k`) | unique pair edges `C(k,2)` | default primary unit | example |
|---:|---:|---|---|
| 2 | 1 | `edge` | cotton, rapeseed |
| 3 | 3 | `group` | wheat |
| 4 | 6 | `group` | strawberry |
| 1 | — | homoeolog test N/A | diploid rice |

**Octoploid and beyond (≥4 subgenomes).** List all four subgenomes once. The
canonical engine derives the six pair edges from each A/B/C/D row, scores them
under one all-subgenome null and bootstrap stream, then ACAT-combines them for
group primary. It does not create an `A:B:C:D` design column. Do not run six
direction-specific analyses and repair multiplicity afterwards.

**Diploids** have no homoeologs, so the cross-subgenome interaction test does not
apply directly. Two things still work for a diploid:

- `prep-snps` is fully generic (one genome): give a subgenome map that labels
  every chromosome with a single group, and you still get a `snp_to_gene` NPZ +
  gene table — useful for gene-based work and for `fit` (which handles diploids
  via a single kernel).
- If you want a **gene×gene epistasis** test between *paralog* pairs in one
  diploid genome, point both subgenomes at the *same* BED and NPZ
  (`genotype: {A: genome, B: genome}`, `snp_to_gene: {A: npz, B: npz}`) and give
  `prep-homoeologs --from-table` a table of paralog pairs (or build it with
  `--method diamond-rbh` on the genome's proteins against themselves). `interact
  --mode pairwise` then tests those pairs exactly like homoeolog pairs.

---

## The subgenome map (the "chromosome groups" file)

A TSV that says which chromosome belongs to which subgenome. One row per
chromosome in your reference.

| column | required | meaning |
|---|---|---|
| `chrom` | yes | chromosome name **exactly as it appears in BOTH the GFF and the PLINK `.bim`** |
| `subgenome` | yes | subgenome label (e.g. `A`, `B`, `C`) |
| `base_group` | optional | the ancestral/base chromosome a homoeolog set descends from (e.g. `Chr1`); enables `--restrict-base-group` |

Example (allohexaploid AABBDD, one row per chromosome):

```
chrom   subgenome    base_group
chr1A   A            1
chr1B   B            1
chr1D   D            1
chr2A   A            2
chr2B   B            2
...
```

If your GFF and `.bim` use different chromosome names, rename one of them first
so the `chrom` column matches both. Column names are configurable
(`--chrom-col`, `--subgenome-col`, `--base-group-col`).

---

## Step 1 — `prep-snps`

Assigns each SNP to the gene(s) whose body (± a flank) contains it, and writes
the NPZ the engine loads plus a human-readable gene table.

```
homoeogwas prep-snps \
  --gff genes.gff3 \
  --subgenome-map subgenome_map.tsv \
  --bed A=path/to/subgenome_A \
  --bed B=path/to/subgenome_B \
  --bed C=path/to/subgenome_C \
  --feature gene --id-attr ID \
  --flank-bp 2000 --min-snp 2 \
  --out-dir interact_prep
```

- `--bed SUB=PREFIX` — per-subgenome PLINK prefix (expects `.bed/.bim/.fam`),
  repeat once per subgenome. Only the `.bim` is read here.
- `--feature` / `--id-attr` — GFF feature row type and the column-9 attribute
  used as the gene id (GFF3 `ID=` or GTF `gene_id "..."`).
- `--flank-bp` — bp added on each side of a gene when assigning SNPs (0 = inside
  gene body only).
- `--min-snp` — a gene enters the burden **NPZ** (and is marked `callable=1` in
  `genes_<S>.tsv`) only with at least this many assigned SNPs. It does **not**
  restrict the gene-id universe: `genes_<S>.tsv` lists every gene regardless, so
  homoeolog pairing stays defined on the full genome (callability is a separate
  downstream gate, never a determinant of which genes are homoeologs).

**Outputs** (in `--out-dir`):

- `snp_to_gene_<S>.npz` — `gene_ids` (1-D string array) and `snp_idx` (object
  array; `snp_idx[i]` is the **0-based row indices into that subgenome's
  `.bim`** = the BED dosage-column indices) for gene `gene_ids[i]`.
- `genes_<S>.tsv` — `gene_id, subgenome, chrom, start, end, strand, n_snp,
  callable`. This is the **full gene-id universe** (every gene, including
  `n_snp=0`); `callable` = the gene is in the burden NPZ (`n_snp >= --min-snp`).
  `prep-homoeologs` defines homology over this full universe and gates testability
  on `callable`.

> The stored SNP indices are 0-based `.bim` row numbers, **not** SNP IDs or
> coordinates — this is what `interact` indexes into the genotype matrix.

---

## Step 2 — `prep-homoeologs`

Builds the `gene_<S>` TSV. Two ways, pick one.

### 2a. From your own orthology table (`--from-table`, recommended)

If you already have orthogroups (OrthoFinder, synteny, curated), convert them:

```
# long format: two columns mapping each gene to its orthogroup
homoeogwas prep-homoeologs --mode triad --subgenomes A,B,C \
  --genes interact_prep/genes_{S}.tsv \
  --from-table orthogroups.tsv --table-format long \
  --gene-col gene --group-col group \
  --out interact_prep/triads.tsv
```

- `long`: columns `gene`, `group` (one row per gene). Genes are grouped by
  `group`; a group must resolve to **exactly one** gene per subgenome.
- `wide`: a `group` column plus one column per subgenome holding
  comma-separated gene lists.

Homology and callability are kept strictly separate (this is what prevents a
SNP-rich paralog from being substituted for a SNP-poor true homoeolog):

- A group with **>1 candidate** in any subgenome is genuine 1:many ambiguity →
  **dropped and recorded** (never collapsed by picking the SNP-richest member).
- A resolved 1:1 pair/triad is emitted only if **every copy is `callable`** (and
  `>= --min-snp-pair`); otherwise the whole group is **dropped and recorded**.
- `--min-snp-pair` (default 1) is the per-copy callability gate.
- Coverage (`tested / total` true groups) and the drop breakdown are printed, and
  a per-group `<out>.audit.tsv` (status + `n_snp` per copy) is written, so the
  fraction of true homoeolog pairs that are untestable in a SNP-sparse panel is
  an explicit, honest number — not a silent source of mispairings.

### 2b. Compute with DIAMOND reciprocal best hits (`--method diamond-rbh`)

**Preparing the protein FASTA** (from a genome FASTA + the GFF). DIAMOND needs
one protein per gene whose header is the gene id used in `genes_<S>.tsv`:

```
# 1. extract per-transcript proteins
gffread -y proteins_tx.faa -g genome.fa genes.gff
# 2. keep the longest protein per gene, rename header to the gene id, and split
#    per subgenome (genes on each subgenome's chromosomes); also strip '.'/'*'
#    stop/gap characters, which DIAMOND rejects.
```

A ready-made
[protein splitter](https://github.com/Shipeng-Yang/HomoeoGWAS-reproducibility/blob/main/scripts/preprocess/proteins_per_subgenome.py)
that does steps 2–3 (longest isoform per gene, subset to `genes_<S>.tsv`, clean
`.`/`*`) is maintained in the separate reproducibility repository. Then:

If you have per-subgenome protein FASTAs (headers = gene ids matching
`genes_<S>.tsv`):

```
homoeogwas prep-homoeologs --mode triad --subgenomes A,B,C \
  --genes interact_prep/genes_{S}.tsv \
  --method diamond-rbh \
  --proteins A=prot_A.faa --proteins B=prot_B.faa --proteins C=prot_C.faa \
  --restrict-base-group --subgenome-map subgenome_map.tsv \
  --threads 16 \
  --out interact_prep/triads.tsv
```

- A triad is kept when the three subgenome pairs agree (A↔B, B↔C, A↔C all
  reciprocal-best and consistent).
- `--restrict-base-group` (needs `base_group` in the subgenome map) keeps only
  homoeologs that share a base chromosome — important for autopolyploids, where
  it stops paralogs on non-homologous chromosomes from being mistaken for
  homoeologs.
- DIAMOND RBH is a convenience baseline; for publication, prefer a dedicated
  orthology/synteny workflow and feed it via `--from-table`.
- Protein sequences must not contain `.`/`*` (stop/gap) characters — strip them
  first (DIAMOND errors otherwise). Some DIAMOND builds (e.g. 2.2.0) deadlock on
  `makedb`; 2.1.x works — pass a known-good binary with `--diamond`.

**Output**: `triads.tsv` with columns `gene_A, gene_B, gene_C` (or `gene_A,
gene_B` for `--mode pairwise`). These gene ids match the NPZ exactly.

---

## Step 3 — run `interact`

Point an `interact` config at the files just built:

```yaml
interact:
  mode: group
  subgenomes: [A, B, C]
  groups: interact_prep/groups_ABC.tsv
  statistic: omniB
  hypothesis_unit: group
  subset_order: 2
  family_scope: primary_only
  primary_transform: INT
  primary_multiplicity: bootstrap_minp
  genotype:    {A: geno/subgenome_A, B: geno/subgenome_B, C: geno/subgenome_C}
  snp_to_gene: {A: interact_prep/snp_to_gene_A.npz,
                B: interact_prep/snp_to_gene_B.npz,
                C: interact_prep/snp_to_gene_C.npz}
  phenotype: pheno.tsv
  sample_col: IID
  trait: my_trait
  burden: {cap: 150, min_snp: 3, maf_min: 0.01, n_pc: 3}
  grm: {method: grm_from_X, maf_min: 0.01, scope: all_subgenomes}
  calibration: {method: bootstrap, B: 2000, seed: 2026}
outputs:
  out_dir: results_interact/my_trait
  full_ranking: true
```

```
homoeogwas interact -c interact.yaml --n-jobs 16
```

For canonical `mode: group`, `statistic: omniB` runs, `--n-jobs` is the
maximum number of POSIX worker processes. Large prepared NumPy arrays are
inherited read-only through fork copy-on-write, and every child limits native
BLAS/OpenMP libraries to one thread. Use the installed `homoeogwas interact`
entry point for parallel runs: it applies OpenBLAS/OpenMP/MKL/NumExpr limits
before NumPy/SciPy import. Direct library calls with an already oversized
native pool are refused for `n_jobs > 1`. The JSON result records requested and
effective jobs, backend, inner-thread limit and observed worker PIDs under
`results.INT.model_diagnostics.parallel_execution`; this execution metadata is
not part of the statistical/checkpoint identity. A platform without `fork`
runs serially and records the fallback reason instead of silently using an
ineffective Python thread pool.

A complete worked example on an allo-octoploid (strawberry, AABBCCDD,
2n=8x=56) — one group family with six derived pair edges — is in
[`examples/strawberry_octoploid.md`](examples/strawberry_octoploid.md).

### Interpreting omniB output

`interact_<trait>.json` reports the combined omniB p-value. Top and significant
units also carry:

- `component_p.minor_burden`;
- `component_p.pc1`;
- `component_p.kernel_hadamard`;
- `smallest_component` and `smallest_component_p`.

Canonical group runs always write the full primary ranking (and also accept
`outputs.full_ranking: true` explicitly). The file
`interact_<trait>_ranking_<mode>_<transform>.tsv` contains these fields for
every declared edge or group, including explicit non-estimable rows. Group
records retain the driving edge/component as descriptive localization.

The smallest component and non-primary edge/group layer are descriptive
localization, not another multiple-testing-corrected discovery. An omniB hit
must be called an **omnibus
interaction** unless a burden-specific estimand and threshold were
prespecified. To reproduce a legacy burden analysis, say so explicitly:

```yaml
interact:
  statistic: burden
  primary_transform: INT
  primary_multiplicity: bonferroni
  calibration: {method: permutation, perm_B: 2000}
```

The canonical formal CLI route emits INT only. A raw-scale sensitivity analysis
must be a separate, explicitly noninferential run and cannot inherit INT
rejection fields. Historical `mode: pairwise`/`pairs` and `mode: triad`/`triads`
omniB configs remain readable and normalize internally to group mode.

### Experimental conditional A×B×D test

For an allohexaploid triad, `statistic: triad3` tests a distinct, explicitly
third-order estimand:

```text
y ~ covariates + A + B + D + A:B + A:D + B:D + A:B:D
```

Only the `A:B:D` coefficient is tested. All main effects and pairwise
interactions remain in the model, so a strong AB, AD or BD signal cannot be
relabelled as three-way evidence. The implementation uses centered/scaled
minor-allele burdens and a kinship-preserving parametric bootstrap:

```yaml
interact:
  mode: triad
  subgenomes: [A, B, D]
  statistic: triad3
  primary_transform: INT
  primary_multiplicity: bootstrap_minp
  # genotype, snp_to_gene, triads, phenotype, sample_col and trait as above
  burden: {cap: 150, min_snp: 2, maf_min: 0.01}
  grm: {method: grm_from_X}
  calibration: {method: bootstrap, B: 2000, seed: 2026}
```

This path is experimental and is allowed only for exactly three subgenomes.
Its ranking reports `p_threeway`, `target_residual_ratio` and `target_sd`.
The kinship-preserving bootstrap min-P result is the only formal discovery
decision. Formal inference requires `B >= 999`; `B: 2000` is recommended for
a final run. Faster engineering checks may set
`calibration: {method: bootstrap, B: 99, seed: 2026, qa_only: true}`. A
QA-only run estimates the null tail but deliberately emits no formal
`n_sig`/`sig`.
The analytic Bonferroni count is retained only as a candidate screen and is
reported separately as `analytic_screen_n`/`analytic_screen_sig`.
`target_residual_ratio` measures how much of the raw three-way column remains
after projecting out all lower-order terms; very small values flag weak
identification. The ranking and formal hit records also report
`target_information_max_fraction`,
`target_information_top10_fraction` and
`target_information_effective_n`. These outcome-independent diagnostics show
whether the residualised three-way target is supported broadly across samples
or almost entirely by a few rare genotype combinations. A formal hit with
maximum sample fraction above 0.10 or information effective n below 20 is
flagged for case-deletion review. A significant result means statistical third-order
non-additivity conditional on the declared burden model. It does not, by
itself, show that the three homoeolog products form a physical complex.

After a run, use `homoeogwas audit <results-dir>` to produce a JSON/TSV/Markdown
record of estimability, calibration, component interpretation and replication
status.

---

## Pitfalls

- **Gene-id consistency is everything.** The GFF `--id-attr`, the protein FASTA
  headers, and any orthology table must use the same gene ids. `prep-homoeologs`
  validates ids against `genes_<S>.tsv` and drops/errors on mismatches.
- **0-based `.bim` indexing.** `snp_to_gene` stores `.bim` row numbers; do not
  re-sort the BED after building it.
- **Autopolyploids** often carry several chromosome copies per subgenome per
  base chromosome — set `base_group` and use `--restrict-base-group`.
- DIAMOND results depend on the binary/version and thresholds; record them.
