---
name: homoeogwas
description: >-
  Run HomoeoGWAS end-to-end from breeder-level inputs — generate the YAML
  configs, validate, run subgenome-stratified GWAS or homoeolog-interaction
  workflows, and summarize results. Use when the user wants to run polyploid /
  subgenome-stratified GWAS, split a VCF by subgenome, build homoeolog
  interaction inputs, run the homoeolog interaction test, plot HomoeoGWAS
  results, validate a config, or troubleshoot HomoeoGWAS inputs (wheat, cotton,
  rapeseed, oat, hexaploid, octoploid, etc.).
---

# Driving HomoeoGWAS for a breeder / bioinformatician

**First read `AGENTS.md` in the repository root — it is the canonical workflow
specification.** Follow it. This skill is only a thin entry point.

## Operating principle

Ask the user for **biological/file inputs only**, then build everything yourself:

- genotype source: a **VCF** (+ species YAML / subgenome map) **or** per-subgenome
  **PLINK BED prefixes**
- **phenotype** file, its **sample column**, and the **trait** name
- the **subgenome labels** (e.g. `A B D`)
- an **output directory**
- (interaction only) a GFF + subgenome map, or proteins + a DIAMOND 2.1.x binary

Never ask the user to hand-write a YAML config — generate it under
`<out_dir>/configs/`, validate it, then run.

## What to do

1. Pick the workflow (AGENTS.md §2 decision tree): VCF→GWAS, BED→GWAS, or
   interaction.
2. Generate the config (AGENTS.md §4), enforcing the gotcha rules (§5):
   - coerce integer-like sample IDs to strings on both genotype and phenotype;
   - confirm GFF and `.bim` chromosome names match before `prep-snps`;
   - interaction mode by ploidy (2→pairwise, 3→triad, 4+→2-/3-subsets, never 4-way);
   - use DIAMOND 2.1.x via `--diamond` (2.2.0 deadlocks).
3. `homoeogwas validate -c …` → `fit`/`interact` → `plot`.
4. Summarize **biologically** (AGENTS.md §7): per-subgenome PVE (which
   subgenomes carry heritability), λ_GC calibration, top hits, figure paths,
   warnings, next step.

## If something fails

Return a concrete repair step, not a stack trace — most failures are: integer
sample IDs (0 sample overlap), GFF/`.bim` chromosome-name mismatch, a missing
`.bed/.bim/.fam`, or DIAMOND 2.2.0. Each has a one-line fix in AGENTS.md §5.

## Power users

The agent generates configs, but everything maps to the plain CLI
(`homoeogwas {fit,split,interact,prep-snps,prep-homoeologs,plot,validate,demo}`)
and to the `homoeogwas-mcp` server (same workflow, any MCP client). See
`docs/interact_inputs.md` for the interaction-input details.
