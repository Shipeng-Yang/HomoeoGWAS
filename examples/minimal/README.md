# Minimal example

A tiny, fully synthetic allotetraploid dataset that exercises the complete
HomoeoGWAS input → fit → output contract in a few seconds. Use it to verify your
install and to see exactly what files go in and come out.

## One command (recommended)

```bash
homoeogwas demo --out demo_run --keep
```

This generates the dataset, runs an end-to-end fit, prints the acceptance checks,
and lists the outputs. Drop `--keep` to auto-clean afterwards.

## Or step by step

```bash
python make_demo_data.py demo_data          # writes demo_data/{geno,pheno.tsv,demo.yaml}
homoeogwas validate -c demo_data/demo.yaml   # schema + input-path check
homoeogwas fit -c demo_data/demo.yaml        # run; outputs under demo_data/demo_out/
```

## What gets generated

```
demo_data/
├── geno/A/all.{bed,bim,fam}   # subgenome A — 120 samples × 250 SNPs (2 chroms A1,A2)
├── geno/B/all.{bed,bim,fam}   # subgenome B — 120 samples × 250 SNPs (2 chroms B1,B2)
├── pheno.tsv                  # columns: sample, y   (quantitative trait, planted QTL)
└── demo.yaml                  # the run config (absolute paths, so it runs anywhere)
```

All values are deterministic (fixed seed) — there is **no real biology** here, the
point is the I/O contract. To use your own data, copy `demo.yaml` and point
`phenotype.path` / the genotype `*_bed_prefix_template` at your PLINK
`.bed/.bim/.fam` per subgenome (or a VCF — see `docs/`).

## Expected output (under `demo_out/`)

| file | what |
|---|---|
| `sumstats_demo.tsv` | per-SNP association table (chrom, pos, beta, se, p, …) |
| `summary_demo.json` | run summary: REML σ²/PVE per subgenome, λ_GC, acceptance checks |
| `lambda_gc_demo.tsv` | genomic-control λ |
| `manhattan_demo.png`, `qq_demo.png` | diagnostic plots |
| `resolved_config.yaml` | the fully resolved config actually used |
| `analysis_samples.tsv` | samples that passed phenotype↔genotype join |
