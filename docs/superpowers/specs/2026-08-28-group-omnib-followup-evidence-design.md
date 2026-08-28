# Group omniB Stability and Evidence Follow-up Design

**Date:** 2026-08-28  
**Status:** Approved architecture (option 1)  
**Scope:** Candidate-only follow-up for canonical two-or-more-copy group omniB results

## 1. Goal

Turn a completed canonical group omniB result into an auditable candidate dossier containing:

- exact formal-hit replay;
- independent-region clustering;
- deterministic material-deletion sensitivity;
- environment deletion when valid metadata exist;
- generic annotation, expression, QTL and functional-evidence joins;
- manuscript-safe evidence tiers and claim boundaries.

This command never creates discoveries and never recalibrates the formal family.

## 2. Public command

```bash
homoeogwas follow-up RESULTS_DIR \
  --material-folds 20 --n-jobs 32 \
  [--environment-col environment] \
  [--evidence evidence.generated.yaml]
```

The command locates the generated canonical config and complete INT ranking inside
`RESULTS_DIR`. Explicit `--config` and `--ranking` overrides are accepted for migrated results.

If there are no `primary_sig == 1` rows, it writes a successful
`NO_FORMAL_DISCOVERY` summary and stops before candidate annotation.

## 3. Supported formal input

The primary product path requires:

- normalized `interact.mode: group`;
- `statistic: omniB`;
- `primary_transform: INT`;
- `primary_multiplicity: bootstrap_minp`;
- `subset_order: 2`;
- one `groups` table and verified SNP-to-gene mappings;
- `hypothesis_unit: edge` or `group`.

Legacy pair/triad rankings can be inventoried but are not silently replayed by the canonical
command. They remain historical follow-up fixtures until migrated.

## 4. Frozen-feature replay

The formal observed run creates one prepared omniB context from the complete analysis cohort:

- exact master group and expanded unique-edge order;
- cohort MAF gate;
- seeded cap selections;
- minor-allele burden, PC1 and kernel-Hadamard feature blocks;
- all-subgenome trace-normalized GRMs;
- phenotype INT and null-model policy.

Follow-up calls the same `omnib_family` preparation code with zero bootstrap columns, then scores
the observed phenotype through those prepared features. The replay must satisfy:

- candidate hypothesis IDs match exactly;
- maximum absolute raw-P difference is at most `1e-10` and relative tolerance `1e-6`;
- the evidence-driving edge/component matches when present;
- formal family and estimability counts match the ranking/provenance.

Any mismatch stops before deletion analyses.

## 5. Deletion sensitivity protocol

Deletion is candidate-only internal sensitivity, not a new discovery family.

For every sensitivity sample set:

1. Keep the formal master family, MAF gate and seeded feature definitions frozen.
2. Subset the already trace-normalized per-subgenome GRMs and renormalize each retained kernel
   to trace/sample count.
3. Subset frozen gene features.
4. Refit the null model and reapply INT to the retained/updated phenotype.
5. Score all frozen edges once and ACAT-reduce groups exactly as in the formal engine.
6. Extract only the formal rejected hypothesis IDs.
7. Do not run bootstrap-minP or emit new rejection decisions.

Material folds are deterministic SHA-256 partitions of sorted string sample IDs with seed 2026.
Every analysis sample must occur in exactly one deleted fold; fold sizes differ by at most one.

### Environment deletion

When `--environment-col` is absent, status is `NOT_AVAILABLE_NO_ENVIRONMENT_COLUMN`.

When phenotype rows are unique per sample, deleting an environment removes the samples assigned
to that environment. When samples have repeated environment records, deleting one environment
removes those rows, recomputes each remaining sample's mean phenotype using the same aggregation
rule as `interact`, and retains samples with at least one remaining non-missing record.

An environment level is rejected when deletion leaves fewer than 10 samples. The command reports
that level as unavailable rather than fabricating a score.

## 6. Stability summaries

For each formal hypothesis report:

- formal and replay raw P;
- formal bootstrap-minP adjusted P;
- material median/max P and fraction with nominal `P < 0.05`;
- environment median/max P and fraction with nominal `P < 0.05` when available;
- driving edge/component agreement fractions;
- minimum retained sample count;
- explicit interpretation: internal sensitivity only.

Nominal P is a stability descriptor. It is never treated as a second discovery threshold.

## 7. Independent-region clustering

Coordinates are derived from the exact analysis BIM/gene mapping, not a different genome build.

Two formal units merge when either condition is satisfied:

- physical rule: they use the same copy set/direction, are on the same chromosome in every
  compared copy and lie within 1 Mb in every compared copy;
- LD rule: their frozen PC1 features have `r² >= 0.64` in every compared copy.

At least two copies must be comparable. Edge directions are never merged across different copy
sets solely because they share one gene. Group units compare all declared copies. Connected
components receive deterministic `HOMEO_LOCUS_01`, `HOMEO_LOCUS_02`, ... identifiers.

The output records the actual distances and per-copy PC1 r². A physical-only merge is described
as adjacent signals in one reporting region, not one interchangeable effect.

## 8. Evidence manifest and adapters

Optional evidence is declared in a generated YAML, never embedded in species-specific code:

```yaml
evidence_version: 1
sources:
  - name: ORDER
    kind: expression
    path: expression.tsv
    gene_col: gene_id
    support_col: expressed
    citation: 10.1186/s12870-020-02509-x
  - name: curated_flowering_qtl
    kind: qtl
    path: qtl_overlap.tsv
    gene_col: gene_id
    support_col: overlaps_window
```

Supported `kind` values are `annotation`, `orthology`, `expression`, `qtl`, `functional` and
`literature`. Every source receives a SHA-256, row count, selected columns and citation string in
`evidence_provenance.json`.

The adapter performs left joins from the four/three/two-copy candidate gene set. Missing records
remain explicit. It never treats absence from a source as proof of no function.

## 9. Evidence tiers

Per gene and per formal unit, the deterministic highest tier is:

1. `direct functional evidence`: truthy support from a `functional` source;
2. `QTL or literature support`: truthy support from `qtl` or candidate-specific `literature`;
3. `matched-tissue expression support`: truthy support from `expression`;
4. `orthology/domain annotation only`: annotation/orthology record without stronger support;
5. `no linked external evidence`: no joined record.

Source-specific text is retained so authors can refine wording. The software does not infer a
flowering, fibre or seed mechanism from a generic protein-domain label.

## 10. Outputs

`RESULTS_DIR/followup/` contains:

- `followup_summary.json`;
- `formal_hit_reproduction.tsv`;
- `material_deletion.tsv` and `stability_summary.tsv`;
- `environment_deletion.tsv` when available;
- `locus_links.tsv` and `independent_loci.tsv`;
- `candidate_evidence.tsv`, `evidence_tiers.tsv` and `evidence_provenance.json`;
- `FOLLOWUP_SUMMARY.md` with manuscript-safe wording;
- `independent_audit.json` containing internal consistency checks.

## 11. Claims

Allowed wording:

- edge primary: “encoding-robust omnibus interaction evidence for a homoeolog pair”;
- group primary: “encoding-robust omnibus pairwise interaction evidence within a homoeolog group”;
- deletion: “stable in candidate-only internal deletion sensitivity analyses”.

Forbidden wording:

- direct third-/fourth-order interaction;
- physical protein interaction;
- causal mechanism;
- independent replication when the same analysis cohort is reused.

## 12. Acceptance criteria

1. One implementation handles two-, three- and four-copy canonical groups.
2. All-sample replay matches a frozen formal ranking within the declared tolerance.
3. Material folds cover every sample exactly once and do not alter the formal family.
4. Missing environment metadata produces an explicit unavailable status, not failure.
5. No-hit results stop cleanly without candidate fabrication.
6. Physical and LD merging reasons are separately visible.
7. Evidence joins are generic table adapters with hashes and no species branches.
8. Follow-up never emits a new FWER rejection set or changes formal adjusted P values.

