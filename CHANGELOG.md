# Changelog

## Unreleased

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

[unreleased]: https://github.com/Shipeng-Yang/HomoeoGWAS/compare/v1.0.0...HEAD
