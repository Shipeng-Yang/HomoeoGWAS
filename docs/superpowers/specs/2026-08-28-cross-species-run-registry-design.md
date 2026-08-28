# Cross-Species Run Registry Design

**Date:** 2026-08-28  
**Status:** Approved architecture (option 1)  
**Scope:** Production orchestration around the existing HomoeoGWAS workflow engine

## 1. Goal

Provide one species-independent registry that freezes high-level biological inputs for
wheat, rapeseed, cotton, peanut and future polyploids, generates canonical HomoeoGWAS
configs, validates before execution, resumes safely, audits completed runs and writes one
cross-species status index.

The registry does not introduce a new association statistic. Species names and ploidy are
metadata; `workflow.run_gwas()` and `workflow.run_interaction()` remain the executable
statistical workflow.

## 2. Product boundary

The registry owns:

- relative-path resolution and schema validation;
- deterministic run identity;
- dispatch to the existing breeder-level workflow API;
- resume/skip/fail-closed decisions;
- atomic per-run state and cross-run JSON/TSV/Markdown indexes;
- migration records for historical results.

It does not own:

- GWAS or omniB mathematics;
- SNP-to-gene or homoeolog inference;
- stability or functional evidence analysis;
- conversion of legacy results into newly claimed canonical results.

## 3. Registry schema

The tracked input is YAML with `registry_version: 1` and an ordered `runs` list.

```yaml
registry_version: 1
name: current_polyploid_interactions
index_dir: results/registry/current_polyploid_interactions
runs:
  - id: rapeseed.flowering.group-omnib.v1
    kind: interaction
    species: Brassica napus
    panel: Wu-2019 flowering panel
    subgenomes: [A, C]
    phenotype: data/phenotype.tsv
    sample_col: sample
    trait: flowering_time
    out_dir: results/rapeseed/flowering_time
    bed_prefixes: {A: data/geno/A, C: data/geno/C}
    snp_to_gene: {A: data/maps/A.npz, C: data/maps/C.npz}
    groups: data/groups/rapeseed_AC.tsv
    hypothesis_unit: edge
    family_scope: primary_only
    bootstrap_B: 2000
    n_jobs: 32
```

`kind` is `gwas`, `interaction`, or `historical`.

- `gwas` contains the minimal BED GWAS inputs from `AGENTS.md`.
- `interaction` contains the canonical group omniB inputs. `statistic`, transform,
  multiplicity and `subset_order` are deliberately not configurable in registry v1.
- `historical` contains `result_root`, `analysis_shape`, and an immutable result inventory.
  It is indexed and audited as historical evidence but never dispatched as a new run.

All paths are resolved relative to the registry YAML. Stored identities use normalized
absolute paths plus file identities, while generated configs retain resolved paths.

## 4. Validation

Registry validation fails before any long run when:

- run IDs are blank or duplicated;
- subgenomes contain fewer than two labels for interaction or duplicate labels;
- a mapping does not contain every declared subgenome;
- an interaction entry omits `groups`;
- `hypothesis_unit` is not `edge` or `group`;
- `family_scope` is not `primary_only` or `joint`;
- `bootstrap_B < 19` or `n_jobs < 1`;
- a four-copy entry requests anything other than pair-edge group omniB;
- a `historical` entry is missing its result root or analysis-shape declaration.

Input existence, sample identifiers, BIM/NPZ fingerprints and interaction schema are still
validated by the existing workflow and `homoeogwas validate`; the registry must not duplicate
or weaken those checks.

## 5. Immutable run identity

Each run receives a SHA-256 over a canonical JSON payload containing:

- registry schema version and run ID;
- kind, species, panel, subgenomes, trait and sample column;
- resolved input paths;
- canonical interaction choices (`group`, `omniB`, `INT`, `bootstrap_minp`, subset order 2);
- hypothesis unit, family scope, bootstrap count and seed;
- content SHA-256 and size for small text inputs;
- `.bed/.bim/.fam` paths and BIM/FAM content identities;
- NPZ path and file identity;
- installed HomoeoGWAS version.

Large BED bytes are not rehashed. Their BIM/FAM identities plus the mandatory SNP-to-gene
NPZ-to-BIM binding provide the genotype identity used by the executable validation.

## 6. State and resume

Each non-historical output root receives `registry_run.json`, written atomically. Its states are:

- `PLANNED`: schema and identity accepted;
- `RUNNING`: dispatch began;
- `COMPLETE`: workflow and audit returned successfully;
- `FAILED`: a command failed, with the actionable reason retained;
- `BLOCKED_IDENTITY_MISMATCH`: an existing state belongs to a different canonical identity;
- `SKIPPED_COMPLETE`: resume found a complete matching run;
- `DRY_RUN`: generated/validated plan only.

Resume rules:

1. Matching `COMPLETE` is skipped and summarized.
2. Matching incomplete state is dispatched again; the existing omniB checkpoint layer owns
   resampling continuation.
3. Different identity in the same output root fails closed and does not overwrite files.
4. `--no-resume` rejects any existing registry state.
5. A historical entry is always read-only.

## 7. Dispatch and outputs

Public commands:

```bash
homoeogwas registry validate -c analyses/current_species.yaml
homoeogwas registry run -c analyses/current_species.yaml --resume
homoeogwas registry run -c analyses/current_species.yaml --only rapeseed.flowering.group-omnib.v1
```

The runner calls only:

- `workflow.run_gwas(...)` for `kind: gwas`;
- `workflow.run_interaction(..., statistic="omniB")` for `kind: interaction`;
- `audit`/summary readers for `kind: historical`.

The registry index directory contains:

- `run_index.json`: complete machine-readable record;
- `run_index.tsv`: one row per run;
- `run_index.md`: breeder-facing status and biological summary;
- `registry.resolved.yaml`: resolved registry with secrets excluded (the schema has no secrets).

Index rows include species, panel, trait, subgenomes, kind, identity, status, output root,
audit status, declared primary unit, tested family counts, significant count and next action.

## 8. Historical migration policy

The current positive and negative analyses are registered without rewriting provenance:

- canonical rapeseed group omniB: production result;
- wheat Route-B group result: historical positive fixture until canonical public-shape rerun;
- cotton FibLen/FibElo pairwise results: historical positive/negative fixtures;
- peanut hundred-seed-weight and seed-length: historical negative fixtures.

Historical fixtures prove numerical migration and paper traceability. They may not be labelled
as canonical group-registry runs unless their generated config, family hash and audit meet the
current contract.

## 9. Failure semantics

- A no-discovery run is `COMPLETE`, not failed.
- Missing optional environment metadata is not a registry failure.
- Failed formal audit prevents `COMPLETE`.
- An unreadable historical result is `FAILED_HISTORICAL_AUDIT` and remains read-only.
- Registry execution continues to independent later runs unless `--fail-fast` is supplied.

## 10. Acceptance criteria

1. The same registry code dry-runs two-, three- and four-copy interaction entries.
2. Generated interaction configs are canonical group omniB with one family and subset order 2.
3. Four copies produce six pair edges through the existing engine and never a four-way coefficient.
4. Matching complete state resumes by skipping; identity mismatch fails closed.
5. Historical entries are indexed without dispatch or provenance rewriting.
6. Unit tests contain no wheat/rapeseed/cotton/peanut branches in execution code.
7. All generated configs live below each run's `configs/` directory and are validated first.

