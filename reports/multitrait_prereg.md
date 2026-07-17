# Pre-registration — multi-trait expansion of the homoeolog-interaction scan

> Locked 2026-06-05, BEFORE running any non-emergence/non-fibre trait. Dual-planned (Claude + Codex),
> user-ratified: **calibration + breadth is the primary methods-paper claim; new hits are a
> pre-registered discovery catalogue, NOT new centerpieces.** The deeply-validated centerpieces stay:
> wheat days_to_emergence chr1 B–D (+ chr5 A–D) and cotton fibre-length-uniformity Hit2 (+ Hit1
> caveated) — these are ANCHORS / positive controls, kept in a separate denominator from new
> discoveries. "Ran all, reported all" — no trait is selected on its result.

## Frozen trait grid (declared before looking)

- **Cotton (local now)**: all **13** BLUE traits in `pheno_m3_4_blue.tsv` — fibre_length,
  fibre_strength, micronaire, elongation, length_uniformity, maturity, spinning_consistency,
  boll_weight, lint_percentage, seed_index, lint_index, fibre_weight_per_boll, fibre_density. n=419.
  Both pair modes (body, flank ±2 kb). Anchors: fibre_length (Hit1), length_uniformity (Hit2).
- **Wheat (after Watkins-portal download)**: all **137** released Watkins traits on the SAME 827
  re-sequenced landraces (2024 Nature Watkins portal). Genome-wide ~12,768 curated 1:1:1 triads ×
  3 pairwise + triad ACAT. Anchor: days_to_emergence (chr1 B–D, chr5 A–D).

## Two analysis modes (different jobs)

1. **Per-trait single-trait scan** (primary for calibration + trait-specific biology): one whitened
   scan per (trait × mode). Wheat primary unit = triad ACAT omnibus; pairwise A-B/A-D/B-D = secondary
   decomposition. Cotton primary = A-D pair.
2. **Pleiotropy ACAT-across-traits** (frozen trait families; multiplicity G not G×T): cotton all-13
   and cotton fibre-quality subset; wheat all-137 and biologically grouped sets (development /
   morphology / yield / disease / stress). Answers "does this pair interact for ANY trait in the
   family"; significant pleiotropy hits are followed by single-trait decomposition. NOT used to claim
   which trait is causal.

## Multiplicity / significance (locked)

- Within each (trait × mode) scan: genome-wide bar = Bonferroni 0.05/G over primary units; also report
  BH-FDR within trait. Every reported hit additionally requires an **empirical** calibrated p
  (y-shuffle / Freedman-Lane) and an acceptable trait-level **λ_GC** (flag/exclude inflated traits
  from discovery claims).
- Across the grid: BH-FDR over all primary-unit top signals (two-stage: within-trait threshold →
  across-trait FDR). **Anchors are scored separately** from new discoveries (not mixed in the
  denominator); anchors serve as reproducibility positive controls.

## Minimum QC before reporting a NEW hit (deep causal ladder reserved for centerpieces)

passes declared threshold / labelled FDR tier · empirical p supports asymptotic p · acceptable λ_GC ·
conditional-sanity (interaction not explained by single-gene marginals; VIF≈1) · DFBETA/leave-one-out
not single-sample-driven · adequate burden support (not ultra-rare) · wheat: pairwise decomposition
consistent with the triad omnibus · cotton: body/flank consistency or mode-specific labelling.

## Cotton trait-correlation handling

The 13 fibre traits are correlated. Primary multiplicity uses all 13 (conservative); report the
**effective number of independent traits** (trait-correlation-matrix eigenvalues) as sensitivity, not
as the significance basis. Interpretation: one A-D pair hitting length+uniformity+strength = **one**
correlated fibre-quality locus, not three independent discoveries. Show a clustered trait×hit matrix.

## Compute tiers

Discovery scan = asymptotic p + λ_GC (fast). Candidates passing a relaxed screen → empirical B=1,000.
Final reported hits → B=10,000+. Parallelise by trait / pair-block. Order: cotton-13 first (local;
freeze templates) → Watkins-137 download + harmonise to the 827 panel → wheat-137 same discipline.

## Reporting + traps

Result table columns: `species | trait | trait_group | n | h2_or_reliability | mode | unit | pair |
p_asym | p_emp | p_ACAT | p_Bonf | FDR_within | FDR_global | lambda_GC | direction | robustness_flag |
known_or_new | notes`. **Report null/calibrated traits too** (they ARE the methods evidence). Traps
held: correlated traits inflating hit count; winner's-curse (shrink betas); environment-specific
signals (use BLUEs, report provenance); low-h² traits underpowered (report h2, don't over-read nulls);
nominal≠discovery; do NOT re-rank the manuscript around a fresh weak hit; do NOT change trait groups
after seeing results.
