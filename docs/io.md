# Inputs & outputs

This page is the I/O contract for `homoeogwas fit`: what you provide, and what
you get back. The fastest way to see it concretely is `homoeogwas demo --keep`
(see [Getting Started](getting_started.md)).

## Inputs

### 1. Genotypes — per subgenome

HomoeoGWAS is subgenome-stratified: genotypes are supplied **one set per
subgenome**, in PLINK1 binary format (`.bed` + `.bim` + `.fam`). The config
references them with a template where `{subgenome}` is substituted by each entry
of `panel.subgenomes`:

```yaml
genotype:
  scan_bed_prefix_template: data/myspecies/{subgenome}/all   # → .../A/all.bed, .../B/all.bed
  grm:
    source: bed
    bed_prefix_template: data/myspecies/{subgenome}/all
    maf_min: 0.01
```

- Values are hard-call dosages `0/1/2`; missing calls are mean-imputed at scan time.
- A VCF can be converted first with `homoeogwas split` (genome → per-subgenome
  sets) or `homoeogwas.io.vcf_to_bed`.
- Subgenomes must be distinguishable — by homoeologous chromosome naming
  (`A1, A2, …, B1, …`) or an explicit `chrom_map` in the species config.

#### Biallelic SV, PAV and haplotype pseudo-markers

The same BED interface can carry non-SNP features, but their encoding must be
declared and bound to an exact marker manifest:

```yaml
genotype:
  scan_bed_prefix_template: data/panel/{subgenome}/features
  marker_encoding: binary_presence_0_2
  marker_manifest_template: data/panel/{subgenome}/marker_manifest.tsv
  grm:
    source: bed
    bed_prefix_template: data/panel/{subgenome}/features
    # These paths equal the scan paths, so the GRM inherits the scan contract.
```

The manifest is TSV/CSV with one row for every BIM variant and required columns
`variant_id,marker_type`; `marker_type` is one of `SNP`, `INDEL`, `SV`, `PAV`,
or `HAPLOTYPE`. IDs must match the corresponding BIM exactly and uniquely.

Accepted encodings are:

- `diploid_0_1_2` (default);
- `binary_presence_0_2` for biallelic PAV/SV absence/presence, encoded as
  **0/2**, not 0/1;
- `haplotype_dosage_0_1_2`; and
- `mixed_biallelic_0_1_2`.

The 0/2 requirement preserves the allele-frequency and GRM denominator used by
the diploidised model. For this encoding, the reported `maf` is the frequency
of the encoded presence allele, not the population frequency of a molecular
copy-number state. Raw or multi-allelic copy number, continuous dosage and
graph-path states are not accepted as if they were ordinary SNPs: convert them
to validated biallelic hypotheses, or use a copy-number-specific model. A
non-SNP association remains evidence for the encoded feature; it does not by
itself identify a molecular mechanism.

The scan and GRM inputs have separate contracts:

- when `grm.bed_prefix_template` resolves to the same BEDs as
  `scan_bed_prefix_template`, the GRM inherits the scan encoding and manifest;
- a distinct GRM BED defaults to standard diploid `0/1/2` SNP-like dosages and
  does not inherit the scan-feature manifest;
- a distinct non-standard GRM must declare
  `genotype.grm.marker_encoding` and
  `genotype.grm.marker_manifest_template` explicitly.

For example, binary PAV features can be scanned while an independent,
LD-pruned SNP panel supplies the GRM:

```yaml
genotype:
  scan_bed_prefix_template: data/panel/{subgenome}/pav_features
  marker_encoding: binary_presence_0_2
  marker_manifest_template: data/panel/{subgenome}/pav_manifest.tsv
  grm:
    source: bed
    bed_prefix_template: data/panel/{subgenome}/pruned_snps
    maf_min: 0.01
```

If the independent GRM itself uses non-standard features, add its own
`marker_encoding` and exact `marker_manifest_template` under `grm`. Both `fit`
and `predict` run the same preflight checks before constructing kernels.

### 2. Phenotype — one TSV or CSV

```yaml
phenotype:
  path: pheno.tsv
  sample_col: sample     # column holding sample IDs (must match .fam IIDs)
  trait: y               # column to analyse
```

The phenotype is a tab- or comma-separated table with a sample-ID column and
one column per trait. Sample IDs are always read as strings and matched to the
genotype `.fam` IID; only the intersection is analysed.

### 3. Config — one YAML

Required blocks: `panel.subgenomes`, `phenotype.{path,sample_col,trait}`,
`genotype.scan_bed_prefix_template`. Common optional blocks: `kernels`
(`normalize`, `include_hadamard`), `reml` (`n_starts`, `seed`), `scan`
(`mode`, `backend`, `maf_min`, `call_rate_min`, `loco`), `outputs`
(`out_dir`, `prefix`). Validate before running:

```bash
homoeogwas validate -c your_config.yaml     # schema + path preflight, no compute
```

See `examples/minimal/demo.yaml` for a complete minimal config, and
`configs/species/*.yaml` for species templates.

## Outputs (`outputs.out_dir`)

| file | contents |
|---|---|
| `sumstats_<prefix>.tsv` | per-SNP association: subgenome, chrom, pos, allele, beta, se, chi2, **p** (calibrated), maf, call-rate |
| `summary_<prefix>.json` | REML σ²/PVE per subgenome, λ_GC, acceptance checks, runtime, resolved settings |
| `lambda_gc_<prefix>.tsv` | genomic-control λ (overall and per subgenome) |
| `manhattan_<prefix>.png`, `qq_<prefix>.png` | diagnostic plots |
| `resolved_config.yaml` | the fully resolved config actually used (provenance) |
| `analysis_samples.tsv` | samples retained after the phenotype↔genotype join |

The `p` column in `sumstats` is ready for downstream Manhattan/QQ plotting and
multiple-testing control. `summary_<prefix>.json` is the machine-readable record
of the run (variance components, calibration, and whether every acceptance check
passed).

## Common pitfalls

- **Sample IDs don't match** between `.fam` and the phenotype `sample_col` → the
  join is empty. The run prints matched/missing counts; fix the ID convention.
- **Relative paths** in the config resolve against your current working
  directory. Use absolute paths (as `homoeogwas demo` does) or run from the
  config's directory.
- **Indistinguishable subgenomes** → provide homoeolog chromosome naming or a
  `chrom_map`; the homoeolog kernel needs to know which copy is which.
