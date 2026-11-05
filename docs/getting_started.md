# Getting Started

## Install

```bash
# CPU
python -m pip install "homoeogwas==2.1.3"
# + GPU extras for the per-SNP scan
python -m pip install "homoeogwas[gpu]==2.1.3"
```

The same release can be installed from GitHub:
`python -m pip install "homoeogwas @ https://github.com/Shipeng-Yang/HomoeoGWAS/archive/refs/tags/v2.1.3.tar.gz"`.

From a checkout (for development):

```bash
git clone https://github.com/Shipeng-Yang/HomoeoGWAS.git
cd HomoeoGWAS
pip install -e ".[dev]"
```

**Python**: 3.10+ (3.11 recommended; 3.12 tested). **OS**: Linux (Ubuntu
22.04/24.04 verified); macOS works CPU-only. The GPU stack needs CUDA 12.1. For
VCF QC/splitting, have `plink2 ≥ 2.0` and `bcftools ≥ 1.18` on `$PATH`. If
`pysam`/`bed-reader` fail to build under pip on an HPC node, install via conda
instead.

## Verify the install (≈2 s)

```bash
homoeogwas demo --keep
```

This synthesises a tiny allotetraploid dataset, runs a full fit, prints the
acceptance checks, and lists the outputs — an end-to-end self-test. The
generated dataset and config live in `demo_run/` (see
[`examples/minimal/`](https://github.com/Shipeng-Yang/HomoeoGWAS/tree/main/examples/minimal)).

## Run on your own data

```bash
homoeogwas validate -c my_run.yaml          # check config + input paths first
homoeogwas fit -c my_run.yaml -o results/my_run
```

The full input/output contract — genotype (per-subgenome PLINK `.bed`),
phenotype TSV/CSV, config schema, and every output file — is documented in
**[Inputs & Outputs](io.md)**. The quickest template is the generated
`demo_run/demo.yaml`.

## Run the homoeolog-group interaction test

The interaction workflow needs two prepared inputs besides the genotypes and
phenotype: which SNPs belong to which gene, and which genes form a homoeolog
group. The full guide with options is **[Interaction Inputs](interact_inputs.md)**.

```bash
# 1. SNP-to-gene blocks for each subgenome (gene body ± flank)
homoeogwas prep-snps --gff genes.gff3 --subgenome-map subgenome_map.tsv \
    --bed A=geno/subgenome_A --bed B=geno/subgenome_B \
    --feature gene --id-attr ID --flank-bp 2000 --out-dir interact_prep

# 2. Homoeolog groups (one gene copy per subgenome per row)
homoeogwas prep-homoeologs --mode pairwise --subgenomes A,B \
    --genes interact_prep/genes_{S}.tsv \
    --from-table orthogroups.tsv --table-format long \
    --gene-col gene --group-col group --out interact_prep/pairs.tsv

# 3. The experiment-wide interaction test (group-omniB + bootstrap minimum-P)
homoeogwas validate -c interact.yaml
homoeogwas interact -c interact.yaml --n-jobs 16
```

The result is a ranked table of homoeolog groups with experiment-wide adjusted
P values, plus a JSON run record ([Inputs & Outputs](io.md#homoeogwas-interact-interaction-workflow)).
`homoeogwas audit` then labels the result as computationally valid, an internal
discovery, or in need of replication.

## Add a new species (no framework code change)

```bash
# 1. Copy a species template and edit subgenomes / chrom_map / reference / paths
cp configs/species/wheat_aestivum.yaml configs/species/myspecies.yaml

# 2. Split a panel VCF into per-subgenome genotype sets
homoeogwas split \
    --species-yaml configs/species/myspecies.yaml \
    --vcf myspecies_all.vcf.gz \
    --out-dir myspecies_geno/

# 3. Point a run config at the per-subgenome sets and fit
homoeogwas fit -c myspecies_run.yaml
```

`n_subgenomes ∈ {2, 3, 4+}` is supported. For `n = 2..3`, `K_hom` uses the full
Hadamard product; for `n ≥ 4` (e.g. strawberry octoploid) it auto-falls back to
the pairwise-mean kernel to avoid off-diagonal rank collapse.

## CLI subcommands

| Command | Purpose |
|---|---|
| `homoeogwas fit -c <cfg>` | subgenome-stratified mixed model: variance partition and per-marker scan |
| `homoeogwas interact -c <cfg>` | homoeolog-group interaction test with experiment-wide error control |
| `homoeogwas prep-snps` | build SNP-to-gene blocks for `interact` from a GFF and the genotypes |
| `homoeogwas prep-homoeologs` | build the homoeolog group table for `interact` |
| `homoeogwas validate -c <cfg>` | check config schema and input paths without computing |
| `homoeogwas audit <result.json>` | label a finished result: computationally valid, internal discovery or needs replication |
| `homoeogwas follow-up` | stability and evidence follow-up of significant groups |
| `homoeogwas plot` / `locus` / `rplot` | redraw figures, regional plots or R figures from a finished run |
| `homoeogwas predict -c <cfg>` | cross-validated subgenome-stratified genomic prediction |
| `homoeogwas split` | split a panel VCF into per-subgenome genotype sets |
| `homoeogwas demo` | generate a tiny dataset and run an end-to-end fit (self-test) |
| `homoeogwas design` | parametric pre-flight for sequencing depth |
| `homoeogwas registry` | validate or run a cross-species run registry |
| `homoeogwas mcp` | MCP server for AI assistants (`pip install homoeogwas[mcp]`) |

`fit` flags: `-c/--config` (required), `-o/--out-dir`, `--backend {cpu,gpu,auto}`,
`--dry-run`, `--force`. Each run writes a `resolved_config.yaml` recording the
exact settings used, so re-runs are reproducible.

## Next

→ **[Inputs & Outputs](io.md)** — the I/O contract
→ **[Algorithm](algorithm.md)** — the statistical model
→ **[API Reference](api.md)** — module-level docs
