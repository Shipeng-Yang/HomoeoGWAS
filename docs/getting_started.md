# Getting Started

## Install

```bash
git clone https://github.com/U7-GWAS/homoeogwas.git
cd homoeogwas

# CPU install
pip install -e ".[dev]"

# Optional GPU extras (per-SNP scan + DL prior inference)
pip install -e ".[gpu]"   # transformers locked at 4.46.3
```

**Python**: 3.10+ (3.11 recommended; 3.12 tested).
**OS**: Linux (verified Ubuntu 22.04 / 24.04). macOS works for CPU-only; GPU stack needs CUDA 12.1.
**External tools**: `plink2 ≥ 2.0`, `bcftools ≥ 1.18` on `$PATH` for VCF QC and splitting.

## Run a single trait

```bash
homoeogwas fit --config configs/runs/fit_horvath2020_plant_height_loco.yaml
```

Outputs land in `results/phase3/m3_1_loco/<panel>/<trait>/`:
- `sumstats_<trait>_loco.tsv` — per-SNP χ², β, SE, p, effect_allele_freq
- `summary_<trait>.json` — λ_GC, n_analysis, runtime, top hits
- `manhattan_<trait>.png` / `qq_<trait>.png` — diagnostic plots
- `lambda_gc_<trait>.tsv` — per-chrom λ_GC

## Add a new species (zero code change)

```bash
# 1. Copy a template
cp configs/species/wheat_aestivum.yaml configs/species/myspecies.yaml

# 2. Edit subgenomes (id + chroms), reference fasta/gff, panel paths
# 3. Build a chrom_map TSV: panel_chrom → fasta_chrom → subgenome
vim configs/species/myspecies.yaml

# 4. Split the panel VCF into per-subgenome pgens
homoeogwas split \
    --species configs/species/myspecies.yaml \
    --vcf data/raw/myspecies/all.vcf.gz \
    --out-dir data/processed/myspecies/
```

The framework supports `n_subgenomes ∈ {2, 3, 4+}`. For `n=2..3`, K_hom uses full Hadamard; for `n ≥ 4` (e.g. strawberry octoploid), K_hom auto-falls back to pairwise-mean (`pairwise_mean_kernel`) to avoid sparse off-diagonal rank collapse.

## CLI flag reference (≤ 5 flags per `homoeogwas fit`)

| Flag | Meaning |
|---|---|
| `--config <path>` | Required — YAML run config (panel + trait + LOCO + backend) |
| `--out-dir <path>` | Override default output dir |
| `--backend {cpu,gpu,auto}` | Per-SNP scan backend (default `auto`) |
| `--dry-run` | Validate config + show resolved paths without running |
| `--force` | Overwrite existing outputs |

This satisfies charter §3.3 release item "YAML 配置驱动,主流程命令行 ≤5 个 flag".

## Where outputs go

```
results/
├── phase2/          # core LMM development (M2.* milestones)
├── phase3/          # LOCO + known QTL + DL prior (M3.* milestones)
├── phase5c/         # GBLUP Table 1 (4 panel scope-conditional)
└── phase5d/         # oat blind-run (in progress)
```

Each run writes its own `resolved_config.yaml` recording the actual config used (after defaults / env-var expansion), so re-runs are reproducible.

## Next

→ **[Algorithm](algorithm.md)** for the mathematical model
→ **[API Reference](api.md)** for module-level docs
