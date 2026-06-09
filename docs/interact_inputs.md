# Building the inputs for `homoeogwas interact`

`homoeogwas interact` runs a gene-resolution homoeolog-pair / triad
burden-product interaction test. It needs two preprocessed inputs per analysis
that the rest of the pipeline does not produce:

1. a **`snp_to_gene` NPZ** per subgenome — which SNPs belong to which gene;
2. a **homoeolog `gene_<S>` TSV** — which genes are homoeologs across the
   subgenomes (pairs for 2 subgenomes, triads for 3).

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

## Ploidy — pick the mode for your crop

| ploidy / genome | subgenomes | `interact --mode` | example |
|---|---|---|---|
| allotetraploid AADD / AACC | 2 | `pairwise` | cotton, rapeseed |
| allohexaploid AABBDD / AABBCC | 3 | `triad` | wheat, Jerusalem artichoke |
| allo-octoploid AABBCCDD | 4 | `pairwise`/`triad` over subsets | strawberry — run over 2-/3-subgenome subsets (see below) |
| **diploid** | 1 | — homoeolog test N/A | rice, etc. |

**Octoploid and beyond (≥4 subgenomes).** `prep-snps` and `prep-homoeologs
--from-table` are fully N-subgenome — list all four in `--subgenomes` and you
get four NPZs and a `gene_A,gene_B,gene_C,gene_D` table. The interaction *test*
is run as `pairwise` over the C(4,2)=6 subgenome pairs or `triad` over the
C(4,3)=4 triples (each `interact` run names the 2–3 subgenomes in its config;
extra `gene_<S>` columns in the table are ignored). This is deliberate, not a
gap: a single 4-way burden product is far too sparse/low-power at realistic
sample sizes — the same reason the variance-component side falls back to a
pairwise-mean homoeolog kernel at 4 subgenomes. So decompose into 2-/3-way
tests and aggregate (e.g. ACAT) across them.

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

Example (allohexaploid AABBCC, NCBI-style chrom names):

```
chrom        subgenome    base_group
CM060351.1   A            Chr1
CM060352.1   A            Chr1
CM060353.1   C            Chr1
CM060355.1   B            Chr1
CM060357.1   A            Chr2
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
- `--min-snp` — drop genes with fewer than this many assigned SNPs.

**Outputs** (in `--out-dir`):

- `snp_to_gene_<S>.npz` — `gene_ids` (1-D string array) and `snp_idx` (object
  array; `snp_idx[i]` is the **0-based row indices into that subgenome's
  `.bim`** = the BED dosage-column indices) for gene `gene_ids[i]`.
- `genes_<S>.tsv` — `gene_id, subgenome, chrom, start, end, strand, n_snp`. This
  is the **authoritative gene-id universe** that `prep-homoeologs` validates
  against, so the two files always agree on gene ids.

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
  `group`; one gene per subgenome per group is emitted (the gene with the most
  callable SNPs wins ties).
- `wide`: a `group` column plus one column per subgenome holding
  comma-separated gene lists.

Only genes present in `genes_<S>.tsv` are used; a group missing any required
subgenome is skipped.

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

A ready-made splitter that does steps 2–3 (longest isoform per gene, subset to
`genes_<S>.tsv`, clean `.`/`*`) is in the repo's reproducibility scripts. Then:

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
  mode: triad
  subgenomes: [A, B, C]
  genotype:    {A: geno/subgenome_A, B: geno/subgenome_B, C: geno/subgenome_C}
  snp_to_gene: {A: interact_prep/snp_to_gene_A.npz,
                B: interact_prep/snp_to_gene_B.npz,
                C: interact_prep/snp_to_gene_C.npz}
  triads: interact_prep/triads.tsv
  phenotype: pheno.tsv
  sample_col: IID
  trait: my_trait
  burden: {cap: 150, min_snp: 2}
  grm: {method: grm_from_X}
  calibration: {perm_B: 200}
outputs: {out_dir: results_interact/my_trait}
```

```
homoeogwas interact -c interact.yaml --n-jobs 16
```

A complete worked example on an allohexaploid (Jerusalem artichoke, AABBCC,
2n=6x=102) is in
[`examples/jerusalem_artichoke_hexaploid.md`](examples/jerusalem_artichoke_hexaploid.md).

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
