# Cross-species run registry

The registry fixes biological inputs and execution status for several species without
introducing species-specific algorithms. Every new interaction entry is translated into the
same canonical group omniB config and validated before computation.

## Commands

```bash
homoeogwas registry validate -c analyses/cross_species_interaction_inventory.yaml
homoeogwas registry run -c analyses/cross_species_interaction_inventory.yaml --dry-run
homoeogwas registry run -c analyses/cross_species_interaction_inventory.yaml \
  --only rapeseed.flowering.canonical-group-omnib.v1 --resume
```

Users provide biological paths and labels in the registry. HomoeoGWAS generates the lower-level
run YAML below each output root; users are not asked to write it.

## One statistical path

Two-copy, three-copy and four-copy homoeolog groups all use pair edges as the primitive: one,
three and six edges respectively. The declared primary unit is either an edge or the group ACAT
over its pair edges. There is no direct four-way coefficient and no post-run union of separate
direction-level alpha families.

## Resume safety

`registry_run.json` binds an output root to its input and method identity. A matching completed
run is skipped. A matching partial run is dispatched again so the formal resampling checkpoint
can continue. If a phenotype, BIM, FAM, groups table, mapping or method declaration changes,
HomoeoGWAS refuses to overwrite the old root. Use a new `out_dir`.

The registry writes `run_index.json`, `run_index.tsv` and `run_index.md` under `index_dir`.
A scientifically valid no-discovery run is complete; it is not a software failure.
All index formats are published by same-directory atomic replacement. Unexpected runner
exceptions are recorded as `FAILED` and independent runs continue unless `--fail-fast` is set.

## Historical entries

`kind: historical` preserves existing wheat, cotton, peanut and earlier rapeseed results as
read-only numerical/provenance fixtures. These entries are not dispatched or silently renamed as
current canonical analyses. A canonical rerun uses a separate `kind: interaction` entry and a
new output root.

Historical status is fail-closed: each entry declares every `interact_*.json` and `audit/*.json`
artifact under `artifact_inventory`, including its relative path, role, byte size and SHA-256.
The observed directory must match that frozen inventory exactly. The audit must have a recognized
non-invalid status and bind the declared result by source path or embedded output hash; declared
expected discovery/planned/valid counts are checked when present. A missing, unreadable, changed or
unbound fixture becomes `FAILED_HISTORICAL_AUDIT`, not an apparently successful historical record.

## Repair messages

- Identity mismatch: keep the old result and choose a new output root.
- Missing PLINK files: repair the prefix so `.bed/.bim/.fam` all exist.
- NPZ/BIM mismatch: rerun `homoeogwas prep-snps` against the analysis BED.
- GFF/BIM chromosome mismatch: rename one side so labels match exactly.
- Missing environment metadata: the formal run remains valid; environment deletion is unavailable.
