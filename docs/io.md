# Inputs & outputs

HomoeoGWAS has two analysis workflows that share the same genotypes and phenotypes:

| workflow | command | question it answers | main result |
|---|---|---|---|
| Additive | `homoeogwas fit` | Which markers are associated with the trait, and how much phenotypic variance does each subgenome explain? | per-marker association table, subgenome PVE |
| Interaction | `homoeogwas interact` | Which homoeolog groups (one gene copy per subgenome) carry a genetic interaction, with the error rate controlled over all groups of the experiment? | ranked group table with experiment-wide adjusted P values |

The fastest way to see the additive workflow concretely is `homoeogwas demo --keep`
(see [Getting Started](getting_started.md)); every file listed below for `fit` is
produced by the demo. Interaction inputs are prepared with `prep-snps` and
`prep-homoeologs` ([Interaction Inputs](interact_inputs.md)).

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

### 4. Interaction inputs — gene blocks and homoeolog groups

`homoeogwas interact` uses the same per-subgenome genotypes and phenotype table,
plus two prepared inputs (step-by-step guide: [Interaction Inputs](interact_inputs.md)):

| input | made by | contents |
|---|---|---|
| `snp_to_gene_<S>.npz` + `genes_<S>.tsv` | `homoeogwas prep-snps` | for each subgenome `<S>`, which SNPs belong to which gene (gene body ± flank); `genes_<S>.tsv` lists `gene_id, subgenome, chrom, start, end, strand, n_snp`. The NPZ is bound to its genotype file by SHA-256. |
| groups table (`pairs.tsv`, `triads.tsv`, …) | `homoeogwas prep-homoeologs` | one row per homoeolog group, one column per subgenome copy (`gene_A`, `gene_B`, …); two to four copies per group. Built from reciprocal best protein hits or from an existing orthology table. |

Every group in the table is one hypothesis of the experiment-wide family, so the
table should be fixed before any phenotype is analysed.

## Outputs

All files are written to `outputs.out_dir`; `<prefix>` is `outputs.prefix` and
`<trait>` the analysed phenotype column. Figures are written as PDF (vector), SVG
and PNG.

### `homoeogwas fit` — additive workflow

| file | contents |
|---|---|
| `sumstats_<prefix>.tsv` | one row per tested marker: `snp_id`, `subgenome`, `chrom`, `pos`, `beta` (effect of one allele copy on the analysed phenotype scale), `se`, `chi2` (1-d.f. GLS statistic), `p`, `n_obs`, `maf`, `call_rate` |
| `summary_<prefix>.json` | the run record: REML variance components (`reml.sigma2`), subgenome PVE (`reml.pve`, and `reml.pve_uncertainty` when a bootstrap was requested), log-likelihood and optimizer status, boundary components, marker filtering counts (`scan.filter_counts`), LOCO setting, λ_GC, resolved settings, software version, runtime and every acceptance check (`acceptance`, `acceptance_all_passed`) |
| `lambda_gc_<prefix>.tsv` | genomic inflation factor overall and per subgenome (`scope`, `level`, `n_markers`, `lambda_gc`) |
| `report_card_summary_<prefix>.json` | the numbers to quote (n, PVE, λ_GC), audit flags with severity, and the statements a report must carry |
| `analysis_samples.tsv` | the samples retained after joining phenotype and genotypes |
| `resolved_config.yaml` | the fully resolved configuration actually used |
| `manhattan_<prefix>.{pdf,svg,png}` | Manhattan plot coloured by subgenome |
| `qq_<prefix>.{pdf,svg,png}` | Q–Q plot of the marker P values |
| `lambda_gc_<prefix>.{pdf,svg,png}` | λ_GC overall and per subgenome |
| `variance_components_<prefix>.{pdf,svg,png}` | PVE of each subgenome kernel and the residual |
| `pve_uncertainty_<prefix>.{pdf,svg,png}` | when `reml.pve_bootstrap` is set: subgenome PVE with bootstrap fits, interval and the probability that each subgenome explains the most variance |
| `figures_<prefix>.html` | one page showing every figure with a short description |

Regional plots around chosen loci are drawn afterwards with
`homoeogwas locus`, and all figures can be regenerated from a finished run with
`homoeogwas plot` (matplotlib) or `homoeogwas rplot` (R).

### `homoeogwas interact` — interaction workflow

| file | contents |
|---|---|
| `interact_<trait>.json` | the run record: number of planned, valid and unestimable groups; experiment-wide test (statistic, bootstrap null model, B, seed, family-wise level, threshold and rejected groups); variance components of the null model; interaction-scan λ; full provenance (software version, input hashes, settings) |
| `interact_<trait>_ranking_group_<transform>.tsv` | one row per homoeolog group, ranked by P value (columns below) |
| `provenance/` | launch and input attestations of formal runs |
| `figures/interaction_scan.{pdf,svg,png}` | group −log10 P along the genome at the position of every copy; significant groups highlighted, their copies joined; experiment-wide threshold |
| `figures/interaction_evidence.{pdf,svg,png}` | for each significant group (or the top groups when none is significant): −log10 P of every encoding on every edge, with the edge P |
| `figures/interaction_calibration.{pdf,svg,png}` | bootstrap distribution of the family minimum P with the observed minimum and the threshold; Q–Q plot of the group P values |
| `figures/interaction_null_variance.{pdf,svg,png}` | `smooth_pc4` runs: per-sample variance weights against the first two genomic eigenvectors |
| `figures/figures.html` | one page showing every figure with a short description |

Columns of the ranking table:

| column | meaning |
|---|---|
| `rank`, `hypothesis_id`, `group_id` | rank by `p_interaction` and the group identifier |
| `gene_A`, `gene_B`, … | the gene copy of each subgenome |
| `p_interaction` | group-omniB P value: the Cauchy combination of the edge P values, each combining the burden, PC1 and kernel encodings |
| `p_adjusted_bootstrap_minp` | experiment-wide adjusted P value from the bootstrap minimum-P null |
| `primary_sig` | 1 when the group is rejected at the family-wise level |
| `driving_edge`, `driving_component`, `driving_component_p` | the copy pair and encoding with the strongest evidence, and its P value (to localise the signal; not part of the family-wise test) |
| `p_unestimable` | 1 when the group could not be tested |

The result JSON also records, for the figures and for your own checks, the
bootstrap family minima (`model_diagnostics.bootstrap_fwer.null_minima`), the
encoding P values of every edge of the reported groups (`edge_components`) and,
for `smooth_pc4`, the per-sample variance profile
(`model_diagnostics.null_variance_profile`). Redraw the figures at any time with
`homoeogwas plot <out_dir>`; the scan figure reads gene positions from the
genotype `.bim` files and SNP-to-gene maps named in the run config.

### `homoeogwas follow-up` — significant groups

Besides its tables, `follow-up` writes `followup_allele_combinations.*` (for the
driving edge of each significant group, the raw mean phenotype of the four
combinations of PC1-defined genotype classes of the two copies, relative to the
low/low class, with the additive expectation; descriptive, not adjusted for
relatedness), `followup_robustness.*` (nominal P of each group in every
accession- and environment-deletion reanalysis, with the full-data P) and
`followup_figures.html`.

### Audit and report card

`homoeogwas audit` checks a finished `fit` or `interact` result against its
acceptance contract, and the report card (`report_card_<result>.json`) lists the
numbers to quote, the audit flags and the statements a report must carry. Use it
as the source for the results paragraph of a paper.

## Common pitfalls

- **Sample IDs don't match** between `.fam` and the phenotype `sample_col` → the
  join is empty. The run prints matched/missing counts; fix the ID convention.
- **Relative paths** in the config resolve against your current working
  directory. Use absolute paths (as `homoeogwas demo` does) or run from the
  config's directory.
- **Indistinguishable subgenomes** → provide homoeolog chromosome naming or a
  `chrom_map`; the homoeolog kernel needs to know which copy is which.
