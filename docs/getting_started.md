# Getting Started

## Install

```bash
pip install homoeogwas            # CPU
pip install "homoeogwas[gpu]"     # + GPU extras (per-SNP scan + DL prior), transformers pinned 4.46.3
```

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
phenotype TSV, config schema, and every output file — is documented in
**[Inputs & Outputs](io.md)**. The quickest template is the generated
`demo_run/demo.yaml`.

## Add a new species (no framework code change)

```bash
# 1. Copy a species template and edit subgenomes / chrom_map / reference / paths
cp configs/species/wheat_aestivum.yaml configs/species/myspecies.yaml

# 2. Split a panel VCF into per-subgenome genotype sets
homoeogwas split \
    --species configs/species/myspecies.yaml \
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
| `homoeogwas fit -c <cfg>` | run the subgenome-stratified LMM GWAS |
| `homoeogwas validate -c <cfg>` | check config schema + input paths without computing |
| `homoeogwas demo` | generate a tiny dataset and run an end-to-end fit (self-test) |
| `homoeogwas split --species <yaml> --vcf <in>` | split a VCF into per-subgenome sets |
| `homoeogwas interact` | homoeolog-pair interaction scan |

`fit` flags: `-c/--config` (required), `-o/--out-dir`, `--backend {cpu,gpu,auto}`,
`--dry-run`, `--force`. Each run writes a `resolved_config.yaml` recording the
exact settings used, so re-runs are reproducible.

## Next

→ **[Inputs & Outputs](io.md)** — the I/O contract
→ **[Algorithm](algorithm.md)** — the statistical model
→ **[API Reference](api.md)** — module-level docs
