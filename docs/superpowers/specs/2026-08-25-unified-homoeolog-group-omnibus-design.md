# Unified homoeolog-group omnibus interaction design

Date: 2026-08-25
Status: user-approved design; implementation awaiting written-spec review
Scope: one interaction engine for pair, triad and 4+ subgenome analyses, plus a matched wheat pairwise rerun

## 1. Purpose

HomoeoGWAS currently exposes pairwise and triad workflows whose production
paths share much of the same mathematics but differ in configuration,
candidate-family construction and multiplicity handling. That difference led
the first wheat pairwise rerun to use a different phenotype, a different GRM
method, 40,678 independently assembled pairs and a post-run factor-three
direction correction, whereas the successful wheat triad analysis used a
frozen family of 2,143 groups, an environment-adjusted phenotype and one
group-level bootstrap family.

This design makes the homoeolog group the common input object and the
encoding-robust pair interaction the common primitive. Ploidy changes only the
number of pair edges inside a group:

- two copies: one edge;
- three copies: three edges;
- four copies: six edges;
- more copies: all predeclared pair subsets, subject to explicit size limits.

The production statistic remains omniB. A group p-value is the ACAT of its
constituent edge-level omniB p-values. This is an omnibus for pairwise,
multidimensional interaction evidence inside a homoeolog group; it is not a
third- or fourth-order product term.

## 2. Goals

1. Use one statistical kernel for pairs, triads and higher-ploidy groups.
2. Make the hypothesis unit and experiment-wide family explicit before a run.
3. Use one kinship-preserving bootstrap stream for every hypothesis in the
   declared family, including hypotheses from different subgenome directions.
4. Derive pair and subset families deterministically from one master homoeolog
   table rather than rebuilding them independently.
5. Re-run the wheat pairwise analysis on exactly the phenotype, samples,
   mappings, GRM semantics and F2143 source family used by Route B.
6. Preserve the rule that HomoeoGWAS never fits or claims a single four-way
   interaction coefficient.
7. Emit enough provenance and audit information for a software paper and a
   biological manuscript to distinguish formal discoveries from localization.

## 3. Non-goals

This change will not:

- guarantee a significant wheat pair;
- choose candidate families or analysis settings after viewing association
  results;
- relabel a group-omnibus hit as an A×B×D or A×B×C×D mechanism;
- union edge, group, component, transform or subset-order discoveries without
  an explicitly calibrated joint family;
- replace the experimental `triad3` conditional A×B×D statistic, which remains
  a separate opt-in estimand;
- make raw phenotype-scale results inferential when INT is predeclared primary;
- change or overwrite completed historical analyses.

## 4. Canonical data model

### 4.1 Master group table

The canonical interaction input is a wide table with one row per homoeolog
group:

```text
group_id  gene_A  gene_B  gene_D
...
```

For four subgenomes the same schema adds `gene_C`; the order of `gene_<S>`
columns follows `interact.subgenomes`. `group_id` must be unique and stable. A
legacy pair or triad table can be adapted to this schema without changing its
row order.

The preparation layer validates and records:

- exact group-row count and ordered group-ID SHA-256;
- exact gene-copy cardinality;
- duplicate genes, duplicate groups and duplicate derived edges;
- per-copy callable SNP counts;
- BED/BIM-bound NPZ fingerprints;
- all dropped groups and their reasons.

### 4.2 Deterministic subset expansion

For a group with `k` copies, `subset_order: 2` enumerates the `C(k,2)` unique
edges in lexicographic subgenome order. Each edge record carries:

- `group_id`;
- `edge_id` made from ordered subgenome labels and gene IDs;
- `sub_x`, `sub_y`, `gene_x`, `gene_y`;
- source row index;
- callability and SNP-count metadata.

Exact duplicate biological edges are tested once and retain all source-group
memberships. The manifest records both the expanded row count and unique
hypothesis count. Multiple appearances of the same edge never spend alpha
multiple times.

For 4+ subgenomes, `subset_order: 3` may additionally enumerate triad subsets.
Only one subset order may define the primary family unless a joint calibration
over the union is explicitly requested.

## 5. Common statistical primitive

Every pair edge uses the same three encoding-robust interaction components:

1. minor-allele-oriented burden-product interaction;
2. PC1×PC1 interaction;
3. low-rank kernel-Hadamard interaction.

The edge-level omniB p-value is their equal-weight ACAT combination. Allele
orientation, MAF gating, SNP cap, minimum SNP count, PC count and estimability
rules are identical at every ploidy.

For a group with `k` copies, the group-level omniB p-value is equal-weight ACAT
over its `C(k,2)` estimable edge-level omniB p-values. A two-copy group reduces
exactly to its single edge p-value. This reduction is a required invariant in
the test suite.

Component and edge p-values localize the evidence. They are not discoveries
when the declared primary unit is `group`.

## 6. Shared null model

All hypotheses derived from the same master group analysis use one null LMM.
The production default is:

- one GRM per declared subgenome;
- `grm.method: grm_from_X`;
- `maf_min: 0.01`;
- trace-normalized kernels;
- all declared subgenome kernels included even when a particular edge uses
  only two gene copies;
- one REML fit and one whitener for the entire family;
- no phenotype-derived covariates;
- fixed, outcome-independent covariates applied identically to the observed
  phenotype and every bootstrap replicate when configured.

Using all declared kernels means wheat AB, AD and BD edges are tested under the
same A/B/D background covariance rather than three direction-specific null
models.

The primary phenotype transform is INT. The raw transform is a sensitivity
analysis and emits no rejection fields unless it was independently
predeclared as primary in a separate run.

## 7. Formal hypothesis units

The config must declare exactly one `hypothesis_unit`:

### 7.1 `edge`

One formal hypothesis per unique derived pair edge. The experiment-wide family
contains all directions generated from the master table. Direction labels are
strata for reporting, not separate alpha families.

This is the required setting for the matched wheat pairwise rerun.

### 7.2 `group`

One formal hypothesis per master homoeolog group. Its observed and bootstrap
p-values are ACAT combinations of the same group's edge-level omniB values.
This is the current Route-B triad interpretation and the production default
for three or more copies.

For two copies, `group` and `edge` are numerically and inferentially identical.

### 7.3 Multiple layers

A run cannot emit independent edge and group discoveries by applying two
separate 0.05 thresholds. If both layers are requested, the user must choose
one of:

- one primary layer plus descriptive localization at the other layer; or
- `family_scope: joint`, in which bootstrap min-P is computed over the union
  of edge and group p-values.

The first option is the production default. Manuscripts must identify which
layer was primary. A significant group may be followed by edge localization,
but unadjusted component p-values cannot be called formal pair discoveries.

## 8. Bootstrap FWER calibration

Production omniB inference uses a kinship-preserving parametric bootstrap with
`B: 2000` and a frozen seed. The null LMM is fitted once. The observed response
and all bootstrap responses are whitened with the same fitted null and scanned
through exactly the same feature, ACAT and estimability paths.

For primary p-value matrix `P` with hypotheses in rows and observed plus
bootstrap responses in columns:

1. compute the minimum primary p-value across all hypotheses for each null
   replicate;
2. compute plus-one single-step adjusted p-values for every observed
   hypothesis;
3. use the same null-minimum distribution for the global test, rejection set
   and 5% threshold;
4. require exact agreement among threshold decisions, adjusted-p decisions
   and the serialized hit set.

AB, AD and BD are therefore calibrated together. No post-run `×3` correction
is applied. Bonferroni values may be retained as descriptive analytic screens
but are not the production discovery layer.

The audit records the number of degenerate bootstrap replicates. A non-finite
primary statistic follows the existing conservative degeneracy policy and can
never silently become a non-rejection.

## 9. Configuration interface

The new canonical form is:

```yaml
interact:
  mode: group
  subgenomes: [A, B, D]
  groups: inputs/f2143.tsv
  statistic: omniB
  hypothesis_unit: edge        # edge | group
  subset_order: 2
  family_scope: primary_only   # primary_only | joint
  primary_transform: INT
  primary_multiplicity: bootstrap_minp
  genotype: {A: <prefixA>, B: <prefixB>, D: <prefixD>}
  snp_to_gene: {A: <npzA>, B: <npzB>, D: <npzD>}
  phenotype: <phenotype.tsv>
  sample_col: sample
  trait: <trait>
  burden: {cap: 150, min_snp: 3, maf_min: 0.01, n_pc: 3}
  grm: {method: grm_from_X, maf_min: 0.01, scope: all_subgenomes}
  calibration: {method: bootstrap, B: 2000, seed: 2026}
outputs: {out_dir: <out_dir>, full_ranking: true}
```

Legacy `mode: pairwise` and `mode: triad` configs remain readable. Validation
normalizes them internally to the group model:

- pairwise table → two-copy master groups, `hypothesis_unit: edge`;
- triad table → three-copy master groups, `hypothesis_unit: group`.

The workflow generator writes only the canonical form for new analyses. It
never asks a biological user to construct this YAML manually.

## 10. Wheat matched pairwise rerun

The rerun is a new analysis and does not overwrite the completed 40,678-pair
run. It uses:

- master groups: the exact frozen F2143 file with 2,143 ordered groups;
- candidate expansion: unique AB, AD and BD edges, expected upper bound 6,429;
- phenotype: the exact released `days_to_emerg_env_adjusted` file used by
  wheat Route B;
- samples: the same 827 string IDs in the same genotype intersection;
- genotype and mapping files: the same fingerprint-bound A/B/D inputs;
- null kernels: A, B and D together via `grm_from_X`;
- primary statistic: edge-level omniB;
- primary transform: INT;
- family: every unique edge across all three directions;
- calibration: one shared bootstrap-minP run, B=2000, seed=2026;
- workers: maximum safe worker count with every BLAS library fixed to one
  thread per worker;
- raw results: sensitivity only.

Required pre-run checks include exact phenotype SHA-256, F2143 ordered hash,
BED/BIM/NPZ fingerprints, sample order, unique edge count, per-direction
counts, zero unresolved subgenome labels and a successful generated-config
validation.

The result summary reports:

- global min-P FWER result;
- every edge-level adjusted p-value;
- significant edge count overall and by direction;
- evidence-driving omniB component;
- lambda-GC overall and by direction as diagnostics;
- tail excess as descriptive only;
- complete ranking and audit paths.

No significance is promised. The purpose is to make the wheat pairwise test a
valid, matched comparison with the triad analysis.

## 11. Four and higher subgenomes

HomoeoGWAS continues to refuse a direct fourth-order product statistic.

For four subgenomes, the default group omnibus combines the six unique pair
edges. This is an aggregation of supported two-subgenome tests, not a four-way
interaction. Optional triad-subset analyses enumerate the four three-copy
subsets and use the same group engine. Pair and triad-subset layers cannot both
claim 0.05 discoveries unless the union is jointly bootstrap-calibrated.

For more than four subgenomes, explicit limits on group size and derived
hypothesis count are validated before loading genotype matrices. The generated
config records the chosen subset order and aggregation rule.

## 12. Implementation architecture

The existing `run_clique_scan_omnib` already computes edge omniB values and a
generic group ACAT for `C(k,2)` edges. Implementation should preserve that
tested numerical core and separate it into bounded layers:

1. master-group validation and deterministic subset expansion;
2. feature preparation and shared all-subgenome null fitting;
3. one edge score matrix for observed and bootstrap responses;
4. optional group aggregation from that same matrix;
5. one generic min-P calibration function over the selected primary matrix;
6. result serialization, interpretation guards and independent audit.

`workflow.py` owns high-level biological inputs and config generation.
`interact.py` owns statistics. The CLI validates and routes but does not
reimplement either. The MCP server calls the workflow layer.

## 13. Failure handling and resumability

Validation stops before an expensive run when:

- sample IDs or sample orders disagree;
- phenotype or trait columns are missing;
- a mapping is legacy, unverified or mismatched to its BIM;
- a group has ambiguous copy cardinality;
- chromosome/subgenome labels are inconsistent;
- the declared primary family is empty;
- unsupported multiple primary layers are requested;
- a four-way product statistic is requested;
- formal bootstrap B is below the production minimum.

The generated config and a frozen run manifest are written under
`<out_dir>/configs/` and `<out_dir>/provenance/`. Long runs checkpoint by
bootstrap block or deterministic hypothesis block without changing random
identities. Resume verifies the manifest before accepting any checkpoint.

## 14. Output and interpretation contract

Every run writes:

- a JSON result with the declared primary unit and family hash;
- a complete primary ranking with raw and adjusted p-values;
- edge/component decomposition;
- family and callability manifests;
- bootstrap diagnostics;
- an audit JSON/TSV/Markdown bundle;
- a biological summary that names the tested estimand.

Permitted claims are:

- edge primary: "encoding-robust omnibus interaction evidence for a
  homoeolog pair";
- group primary: "encoding-robust omnibus pairwise interaction evidence
  within a homoeolog group".

Forbidden claims include burden-product mechanism, physical protein
interaction, conditional A×B×D causality or fourth-order interaction unless a
separate appropriate experiment directly supports them.

## 15. Tests and release gates

Implementation is complete only when all of the following pass:

1. two-copy group p equals edge p exactly;
2. three-copy group p equals ACAT of its three edge omniB p-values;
3. four-copy group p equals ACAT of its six edge omniB p-values;
4. edge expansion is deterministic and duplicate edges spend alpha once;
5. permuting input group rows changes no p-values or decisions after restoring
   canonical IDs;
6. AB/AD/BD edges share the same null fit and bootstrap response hashes;
7. adjusted-p, threshold and serialized rejection sets agree exactly;
8. pairwise and canonical two-copy configs are numerically identical;
9. legacy triad and canonical three-copy configs are numerically identical;
10. direct four-way statistics are rejected with a biological explanation;
11. interrupted and uninterrupted runs are byte-identical after audit;
12. one-worker and multi-worker runs are numerically and decision identical;
13. sample-ID, BIM fingerprint and non-SNP encoding gotchas are covered;
14. the full relevant test suite passes;
15. a small null simulation shows controlled FWER and a planted pair signal
    shows the expected pair/group power relationship.

After tests pass, the wheat matched run still requires its generated config to
validate before execution. Completion requires the independent audit to pass;
a successful process exit alone is insufficient.

## 16. Publication presentation

The software paper should present one diagram and one mathematical definition:
edge omniB is the primitive, while the group statistic is an ACAT over the
complete set of supported pair edges. Ploidy changes the graph size, not the
statistical engine.

The wheat biological manuscript should treat the existing Route-B triad result
as group-level omnibus evidence. The matched pairwise rerun is a distinct
edge-level analysis using the same frozen biological universe and null model.
If both are discussed, the manuscript must state their separate hypothesis
units and must not union their discovery lists without joint calibration.
