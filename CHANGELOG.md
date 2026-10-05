# Changelog

## Unreleased

## v2.1.0 — heteroscedasticity-robust bootstrap null, parallel scans and agent tools (unreleased draft)

### Added

- `interact.calibration.null_variance: smooth_pc4` for canonical group omniB:
  a parametric-bootstrap null whose residual variance follows a smooth
  function of the leading genotype principal components, fitted by IRLS.
  It restored familywise calibration in the tested settings where residual
  variance tracks population structure; `homoscedastic` remains the default
  of the low-level config and
  is reported as a sensitivity analysis. Results record
  `smooth_variance_parametric_bootstrap_minp_plus_one` and the fitted weights.
  Requires bootstrap checkpointing.
- `homoeogwas interact` batched cases share phenotype-independent panel
  preparation (`interact_batch`), and an opt-in hash-keyed GRM cache
  (`grm_cache`) reuses genotype relationship matrices across runs with
  identical inputs.
- `scan.n_jobs` for streaming single-locus scans (plain and LOCO): variant
  chunks are scored by forked worker processes with the same chunk boundaries
  and batches as the serial scan and one native thread each; the
  decompressed summary statistics equal the serial output run with one BLAS
  thread. Each worker holds about `chunk_size × n_samples × 20` bytes; a dead
  worker stops the scan with an error. Gzip output is written as a
  multi-member gzip stream. `scan.n_jobs: 1` (default) keeps the serial code
  path; in-memory and GPU scans ignore it (with a warning in memory mode).
- MCP tools `audit_results` and `summarize_results`; `run_interaction`
  exposes `null_variance` (default `smooth_pc4`) and `run_gwas` exposes
  `scan_jobs`. Workflow-generated group omniB configs enable checkpointing
  when `smooth_pc4` is selected.
- `interact.burden.feature_seed` separates gene-feature randomness from the
  bootstrap seed (defaults to the bootstrap seed, as before).

### Changed

- In group-unit and joint families, a group is eligible only when every
  declared pair edge is estimable; groups with an unestimable edge are marked
  unestimable instead of being scored on a partial edge set, which can reduce
  the family size. Edge-unit families are unchanged.
- Bootstrap workers are reused across checkpoint blocks and edge tasks are
  sized by work; checkpoint manifests hash arrays in streaming fashion.
- Partial omniB response errors fail closed; prepared omniB design identity,
  formal marker QC masks and feature seeds are bound into provenance.
- The application-evidence exporter accepts the smooth_pc4 bootstrap method,
  records it as `fwer_method`, and reads `driving_component` for group-unit
  discoveries.


## v2.0.1 — unified omniB, evidence audit and input hardening (2026-08-29)

### Changed

- Canonical `mode: group`, `statistic: omniB` block scoring now uses observable
  POSIX worker processes with copy-on-write shared numerical state and one
  native-library thread per worker. This removes the previous near-single-core
  Python thread bottleneck without changing bootstrap streams, score arrays,
  checkpoint identities or FWER decisions. Results serialize requested and
  effective jobs, backend and observed worker PIDs; unsupported platforms
  record a serial fallback. The console launcher now fixes native numeric
  thread limits before NumPy/SciPy import, and canonical parallel omniB refuses
  unsafe bypass invocations whose native pools were already oversubscribed.
  Candidate follow-up material/environment deletion now uses the same observable
  fork/shared-memory process runner instead of the former joblib threading path.

### Added

- Experimental `interact.statistic: triad3` for exactly three subgenomes. It
  tests the minor-burden `A:B:D` coefficient conditional on all three main
  effects and all three pairwise interactions, freezes estimability on the raw
  hierarchical design, reports target residual-information diagnostics, and
  supports kinship-preserving parametric-bootstrap min-P calibration.
- `homoeogwas audit <result-json-or-directory>` writes JSON, TSV and Markdown
  evidence audits that separate computational validity, internal familywise
  discovery, component-specific interpretation and replication status.
- Optional `reml.pve_bootstrap` performs a fitted-model parametric bootstrap of
  the subgenome PVE partition and reports percentile intervals, boundary rates,
  refit success and component-rank stability. Bootstrap refits now inherit the
  observed model's multi-start search by default and use independent
  `SeedSequence` streams.
- Biallelic PAV, SV and haplotype pseudo-markers can be bound to exact
  per-subgenome marker manifests. Scan and GRM inputs have separate encoding
  contracts, and `fit`/`predict` apply the same preflight validation.

### Fixed

- The production omniB path now retains the minor-allele burden, PC1 and
  kernel-Hadamard component p-values for top/significant units. With
  `outputs.full_ranking: true`, pairwise and triad/group scans write complete
  component-aware ranking TSVs; pairwise scans also write top-pair minor-burden
  values. Previously only the combined omniB p was available, so a kernel-driven
  omnibus hit could be incorrectly described as a burden-product interaction.
- Interaction provenance now records the actual statistic and calibration
  method and no longer claims that a full ranking was produced when the omniB
  path had ignored the requested output.

## v2.0.0 — corrected multiplicity for s>=3 homoeolog groups (breaking)

**Advisory.** Versions up to and including 1.0.2 applied the wrong significance threshold in
`interact` group scans (`mode: triad|homoeolog|clique`, `statistic: burden`) whenever a group has
three or more subgenomes. Each group contributes `K = C(s,2)` pairwise contrasts, but the engine
emitted `bonferroni_alpha = 0.05/G` — the threshold for the `G` tests inside ONE contrast — for
every contrast, together with an `n_sig`/`sig` list at that level. With `s = 3` the three contrasts
each spent the full alpha, so the run-wide error rate was `1 - 0.95^3 = 14.3%` rather than 5%. A
pairwise p selected across the contrasts of its group belongs to the `G*K` family and must be judged
at `0.05/(G*K)`.

Two-subgenome pair scans (`K = 1`) and every `statistic: omniB` scan are NOT affected: for those the
reported threshold was already the correct one for the family being tested.

Anyone who ran a triad/clique burden scan with 1.0.2 or earlier should re-check hits against
`pairwise.<contrast>.bonferroni_alpha` in a 2.0.0 result, or simply multiply the reported p by
`G*K`. Some hits will no longer be significant.

### Breaking changes

- `pairwise.<contrast>.bonferroni_alpha`, `n_sig` and `sig` now carry the full `G*K` family. The
  previous per-contrast meaning moved to `exploratory_within_contrast`, which reports the threshold
  and a descriptive count but deliberately emits no identifier list.
- Non-estimable tests return `null` instead of `p = 1.0`. A placeholder of 1.0 entered ACAT and
  genomic-control as a real, maximally non-significant observation; a missing test must not.
  Consequently ACAT combinations, `lambda_gc`, rankings and minima can differ from 1.0.2.
- JSON output is strict RFC 8259: non-finite values serialize as `null` and `allow_nan=False` is
  enforced. Parsers that relied on bare `NaN` must be updated.
- Exactly one procedure may spend alpha. New `primary_weighting` (`unweighted`|`weighted`),
  `primary_multiplicity` (`bonferroni`|`permutation_minp`) and `primary_transform` (`INT`|`raw`)
  select it; every non-primary procedure reports descriptive statistics with `null` rejection
  fields. Previously the weighted and unweighted procedures, the Bonferroni and permutation
  procedures, and both transforms each emitted a full-alpha rejection set, so taking discoveries
  from whichever one rejected was an uncontrolled union.
- Pairwise rejections are now gated on the group omnibus: a contrast is reported only for a group
  that already rejected in the primary family. This is hierarchical gatekeeping and preserves
  familywise control while keeping the localisation the pairwise tests are for.
- Invalid prior weights raise instead of being silently rewritten to 1.0, which had faked a uniform
  prior. Weights are normalised as `w * G / sum(w)` after scaling by the largest weight, so the
  procedure is invariant to rescaling and cannot overflow.
- A scan with no estimable unit, and a clique scan retaining no complete group, now raise instead of
  returning a result whose empirical p would be computed against an undefined statistic.
- `weighted.bonferroni_n_sig` is `null` unless the weighted procedure is primary.

### Fixed

- Estimability is decided on the RAW, unwhitened design. The whitener is fitted to the phenotype, so
  deciding it after whitening let the tested family depend on `y`; the same mask is now frozen
  across the observed scan and every permutation replicate.
- The per-coefficient test uses Frisch–Waugh–Lovell residualisation of the tested column against the
  nuisance block, with `df = n - rank(Z) - 1`. The previous gate rejected on the condition number of
  the whole design, discarding tests that were perfectly estimable when only nuisance columns were
  collinear.
- The residual sum of squares is an explicit residual norm. Computing it as `y'y - (U'y)'(U'y)`
  cancels catastrophically on a good fit and can go negative, which a clamp then turned into a NaN
  model.
- A permutation replicate whose design-valid test loses its statistic is counted as maximally
  extreme rather than discarded. Those failures are degenerate fits, i.e. the tail of the null, so
  dropping them shrank the empirical-p numerator and raised the min-p cutoff in the
  anticonservative direction.
- The permutation cutoff is the k-th order statistic with `k = floor(alpha*(B+1))`, rejected on
  strict inequality, so it can no longer disagree with the `(1+#)/(B+1)` empirical p. An
  interpolated 5th percentile is not a valid permutation cutoff and mishandled ties at zero.
- New `contrast_omnibus` family: the K contrast-level ACAT p-values are Bonferroni-adjusted across K.
- Results report `n_planned`, `n_valid`, `n_unestimable`, the estimability policy and its
  tolerances, the excluded unit identifiers with reasons, and the number of degenerate permutation
  replicates.
- Only design-determined reasons can retire a hypothesis. A response-dependent failure is an
  analysis error and raises; a failed decomposition means estimability was not determined, not that
  the target was unestimable.
- Multi-trait scans no longer fold missing components to `p = 1.0`, and share one raw-design mask
  across every trait and every permutation.

### Added

- `homoeogwas design`: parametric sequencing-depth pre-flight calculator.
  Allopolyploid genomes are large, so blind WGS is expensive; this estimates how
  much coverage depth (x) is needed by chaining depth -> per-genotype confident-
  call recovery -> usable in-gene density -> the validated callable-pair density
  curve -> discovery design-band, separately for the marginal subgenome scan and
  the stricter homoeolog-interaction test (which needs more depth to genotype and
  distinguish both homoeolog copies). Reports the inverted "depth to reach the
  discovery-feasible/strong band" with a mappability sensitivity range, and built-
  in species anchors (`--like wheat|cotton|oat|rapeseed`) so a planner needs no
  VCF. Framed as a planning heuristic, not an empirical depth calibration (no raw
  reads are used). New module `design_depth.py`.

### Release hardening

- Workflow orchestration now stops immediately when `validate`, `fit`, or
  `interact` fails and propagates a truthful `ok: false` result.
- Fit and interaction phenotype readers accept TSV/CSV, preserve sample IDs as
  strings, and correctly handle PLINK prefixes containing dots.
- `interact.burden.min_snp` and the burden MAF gate are passed to the production
  omniB scan; the configured hypothesis universe is no longer silently replaced
  by function defaults.
- `prep-snps` records the source BIM SHA-256 and variant count. Interaction
  validation refuses legacy/unverified or mismatched SNP-to-gene mappings.
- The fit CLI now uses the documented homoeolog-kernel auto policy: full
  Hadamard for 2–3 subgenomes and pairwise-mean for 4+.
- `homoeogwas validate` supports both fit and interaction configs.
- Manuscript analysis code, figure workflows, and small audit outputs now live
  in the separate
  [HomoeoGWAS-reproducibility](https://github.com/Shipeng-Yang/HomoeoGWAS-reproducibility)
  repository; installable software releases no longer mix product and paper
  workflows.

## v1.0.2 — homoeolog-interaction dominance adjustment

- `homoeolog interaction`: optional `dominance_adjust` flag (config
  `interact.burden.dominance_adjust`, default `False` → byte-identical to the
  legacy `[C, b_X, b_Y, b_X·b_Y]` design, existing results unchanged). When
  enabled, the per-pair whitened GLS design adds the per-copy quadratic terms
  `b_X², b_Y²`, so the tested product effect is conditional on both the additive
  and per-gene dominance/curvature main effects. This closes a type-I leak under
  homoeolog collinearity (a calibration simulation showed genome-wide FWER
  rising toward ~1.0 as cross-copy correlation increases when the squared
  burdens are unmodelled; conditioning restores ~0.05). Threaded through
  `run_pair_scan`, `run_clique_scan`, and `run_multitrait_pair_scan` (observed +
  permutation paths) and recorded in run provenance.

## v1.0.1 — first PyPI / conda release

First release published to PyPI (and submitted to bioconda) — `pip install
homoeogwas`. Same features as v1.0.0; version bumped for distribution.

## v1.0.0 — first public release

Subgenome-stratified linear-mixed-model GWAS for allopolyploids (and diploids),
validated across ploidies 2n–8n (wheat AABBDD, cotton AADD, rapeseed AACC,
strawberry AABBCCDD, oat, rice).

### Core
- `homoeogwas fit` — end-to-end subgenome-stratified LMM GWAS from one YAML:
  per-subgenome VanRaden GRMs + optional homoeolog Hadamard kernel, multi-kernel
  REML variance components (per-subgenome PVE), fixed-V per-SNP scan with
  in-memory and streaming backends, optional LOCO, per-SNP sumstats + λ_GC.
- `homoeogwas split` — split a panel VCF into per-subgenome BED/pgen from a
  species YAML (bcftools + plink2).
- `homoeogwas interact` — gene-resolution homoeolog-pair / triad burden-product
  interaction test with ACAT and permutation calibration.
- `homoeogwas validate`, `homoeogwas demo` (install self-test).

### Publication-grade visualization
- New `src/homoeogwas/plots.py` and `homoeogwas plot <results_dir>`: four figures
  (per-subgenome variance/PVE, subgenome-faceted Manhattan, stratified QQ with
  per-subgenome λ + a 95% null band, λ_GC QC bar) as PNG + editable PDF/SVG with
  a colourblind-safe publication theme; `fit` emits them automatically.

### Interaction input tooling (crop-agnostic)
- `homoeogwas prep-snps` — build the `snp_to_gene` NPZ + gene table from a GFF +
  per-subgenome BEDs + a subgenome map.
- `homoeogwas prep-homoeologs` — assemble the homoeolog pair/triad table from a
  user orthology table or DIAMOND reciprocal best hits (base-group-restricted).
- `docs/interact_inputs.md` documents every required input and the di-/tetra-/
  hexa-/octoploid workflow, with a worked octoploid example.

### Agent-native interface
- `AGENTS.md` — a single, agent-readable workflow specification any LLM/agent can
  follow to drive the tool from breeder-level inputs.
- A Claude Code skill (`.claude/skills/homoeogwas/`).
- `src/homoeogwas/workflow.py` — engine that turns a few high-level inputs into a
  generated, validated config, runs the CLI, and summarizes results (blocking
  early on common input pitfalls).
- An MCP server (`homoeogwas mcp` / `homoeogwas-mcp`, optional `[mcp]` extra) so
  any MCP client (Claude, Cursor, Cline, …) can run the full pipeline.

### Quality
- 318 tests pass; ruff-clean; CPU/GPU Docker images; reproducible-by-config runs.

[unreleased]: https://github.com/Shipeng-Yang/HomoeoGWAS/compare/v2.1.0...HEAD
[2.1.0]: https://github.com/Shipeng-Yang/HomoeoGWAS/compare/v2.0.1...v2.1.0
[2.0.1]: https://github.com/Shipeng-Yang/HomoeoGWAS/compare/v2.0.0...v2.0.1
[2.0.0]: https://github.com/Shipeng-Yang/HomoeoGWAS/compare/v1.0.2...v2.0.0
