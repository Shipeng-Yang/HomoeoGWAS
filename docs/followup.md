# Canonical group omniB follow-up

`homoeogwas follow-up` turns formal group omniB discoveries into an audited, candidate-only
stability and functional-evidence dossier. The same implementation is used for two-, three- and
four-copy groups; species names are metadata and do not select different statistics.

## Run

```bash
homoeogwas follow-up results/my_interaction \
  --material-folds 20 \
  --n-jobs 16 \
  --grm-blas-threads 16 \
  --environment-col environment \
  --evidence analyses/my_trait.evidence.yaml
```

The result directory must contain the generated canonical config at
`configs/interact.generated.group.omnib.yaml` and the complete INT ranking. Explicit `--config`
and `--ranking` paths are accepted when a result was archived in another layout.

Before any sensitivity analysis, HomoeoGWAS reloads the exact phenotype, BED-bound SNP-to-gene
mappings and master group family and reproduces every formal interaction p-value and evidence
driver. It refuses legacy direction-wise results, unverified mappings and changed formal inputs.

## Stability analyses

Material deletion assigns every analyzed sample to exactly one deterministic SHA-256 fold. Each
fold is removed once, the null model is refit, and only the already formal units are rescored with
the frozen gene blocks, allele orientation, MAF gates, burden cap and PC features. This does not
redefine the discovery family.

When `--environment-col` is supplied, raw phenotype rows from each environment are deleted and
the remaining rows are reaggregated per material before rescoring. If the column or usable levels
are absent, the summary records an explicit `NOT_AVAILABLE_*` state; it does not manufacture an
environment analysis or invalidate the formal result.

## Independent reporting regions

Formal units are merged only when every comparable subgenome copy satisfies the same physical
rule (within 1 Mb) or the same frozen-feature PC1 linkage rule (r² at least 0.64). The pairwise
link table records the distances, r² values and merge reason. Region IDs are a reporting layer;
they do not change the experiment-wide bootstrap-minP inference.

## Evidence manifest

Evidence is supplied as a small YAML manifest. Every table is bound by SHA-256 and its row count,
and every candidate is retained when a source has no match.

```yaml
evidence_version: 1
sources:
  - name: reference_annotation
    kind: annotation
    path: evidence/annotation.tsv
    gene_col: gene_id
    columns: [description, ortholog]
  - name: matched_tissue_expression
    kind: expression
    path: evidence/expression.tsv
    gene_col: gene_id
    support_col: expressed
    citation: 10.0000/example
    columns: [expressed, tissue, experiment]
  - name: published_qtl
    kind: qtl
    path: evidence/qtl.tsv
    gene_col: gene_id
    support_col: overlaps_qtl
    columns: [overlaps_qtl, qtl_name, distance_bp]
```

Supported kinds are `annotation`, `orthology`, `expression`, `qtl`, `functional` and
`literature`. A support column accepts true/false, 1/0, yes/no, supported/positive. Evidence tiers
are conservative: direct functional evidence; QTL or literature support; matched-tissue
expression; orthology/domain annotation only; or no linked external evidence.

## Outputs and interpretation

- `formal_hit_reproduction.tsv`: exact formal replay and the evidence-driving omniB component.
- `material_deletion.tsv` and `environment_deletion.tsv`: candidate-only deletion results.
- `stability_summary.tsv`: median/worst p-value, nominal support and driver agreement.
- `independent_loci.tsv` and `locus_links.tsv`: reporting regions and merge audit trail.
- `candidate_evidence.tsv`, `evidence_tiers.tsv`, `evidence_provenance.json`: source-grounded
  functional evidence.
- `independent_audit.json`, `followup_summary.json`, `FOLLOWUP_SUMMARY.md`: completion and
  consistency checks.

The publishable wording is “encoding-robust omnibus pairwise interaction evidence for a
homoeolog pair” for edge-primary analyses, or “encoding-robust omnibus pairwise interaction
evidence within a homoeolog group” for group-primary analyses. These outputs do not establish a
third-/fourth-order coefficient, a physical interaction, causality or independent replication.

If the formal family has no FWER discovery, follow-up stops safely with
`NO_FORMAL_DISCOVERY`; it never promotes top-ranked nonsignificant units into candidates.
