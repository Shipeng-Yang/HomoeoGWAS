# 03 — Architecture

> **Status note (2026-06-02).** This document was rewritten to match the *implemented*
> code in `src/homoeogwas/`. The previous version described a Rust + JAX/Triton stack
> (AI-REML on GPU, REGENIE two-step ridge, saddlepoint SPA, graph-node/SV integration,
> hexaploid `{0..6}` dosage, knockoffs, Zarr/GFA, package name `polygwas`) — **none of
> which was built**. Those ideas are preserved, in past tense, in the
> [Considered but not implemented](#considered-but-not-implemented) appendix.

## What this is

`homoeogwas` (v0.1.0.dev0) is a single **Python package** for allopolyploid GWAS. Core
numerics are NumPy/SciPy; GPU paths are optional (Torch for the per-SNP scan, CuPy for the
REML eigensolve, `transformers` only inside the standalone DL-prior scripts). There is no
Rust, JAX, or Triton.

It exposes **three CLI subcommands**:

| Command | Purpose |
|---|---|
| `homoeogwas fit` | end-to-end subgenome-stratified single-locus LMM GWAS from a YAML config |
| `homoeogwas split` | YAML-driven per-subgenome genotype split for a new allopolyploid species |
| `homoeogwas interact` | gene-resolution homoeolog-pair burden-product interaction scan |

### Implementation / claim matrix

| Capability | Implemented | Status |
|---|---|---|
| PLINK1 BED hardcall input | yes | main |
| VCF input | via `plink2` conversion (`io.vcf_to_bed`) | main |
| Subgenome VanRaden GRM (+ LOCO) | yes | main |
| Multi-kernel REML variance components | yes | main |
| Per-SNP EMMAX/P3D LMM scan (CPU + 1-GPU) | yes | main |
| `K_hom` Hadamard homoeolog kernel | yes | **supplementary / defensive only** (variance-null on real traits) |
| Gene-pair burden-product interaction engine | yes | **main methodological contribution** |
| GBLUP genomic prediction | yes | secondary |
| Phenotype-simulation benchmark | yes | secondary |
| Calibration / boundary-LRT / bootstrap diagnostics | yes | supporting |
| Zero-shot DL variant prior (PlantCaduceus / AgroNT) | standalone scripts (`scripts/phase3/`), not in the package | supplementary |
| REGENIE ridge · JAX/Triton · saddlepoint SPA · graph/SV · `{0..6}` dosage · knockoff · Group-BH | **no** | considered only (appendix) |

## Dataflow

```
PLINK1 .bed/.bim/.fam   (or VCF.gz --plink2--> bed/pgen)
        │
        ▼  io.load_bed_hardcall            additive {0,1,2}∪NaN dosage
   per-subgenome dosage matrices  (A / B / D · A / D · A / C · …)
        │
        ▼  grm.compute_grm                 VanRaden K_s; LOCO partials by subtraction
   {K_s}  (+ optional K_hom = ⊙ K_s, kernel.build_homoeolog_kernel)
        │
        ▼  lmm.fit_multi_reml              multi-kernel REML σ² → PVE
   variance components (fixed for the scan, EMMAX/P3D)
        │
        ├─▶ scan.build_scan_context → scan_snps / scan_bed_stream   single-locus χ²₁ scan (+LOCO)
        ├─▶ interact.run_pair_scan / run_triad_scan                 homoeolog-pair interaction
        ├─▶ gp.run_cv_gblup                                         genomic prediction
        └─▶ calibration / diagnostics                              null-sim type-I, boundary LRT, bootstrap
```

All single-locus scanning uses **additive `{0,1,2}` hardcall dosage only**. The interaction
engine builds gene-level burdens from the same hardcalls.

## Module map (`src/homoeogwas/`, ~8.8k LOC)

| Module | Responsibility |
|---|---|
| `io.py` | PLINK1 BED hardcall reader (`{0,1,2,NaN}` float32); `vcf_to_bed` via `plink2` |
| `grm.py` | VanRaden per-subgenome GRM `K_s = ZZ'/Σ2p(1-p)` (trace ≈ n); LOCO via numerator/denominator subtraction (`GRMPart`) |
| `kernel.py` | `sum_kernel`, `hadamard_kernel` (`K_hom`), `pairwise_mean_kernel` (≥4 subgenomes), `normalize_kernel`, `build_homoeolog_kernel` (auto) |
| `lmm.py` | single-kernel REML (`fit_reml`, bounded Brent on δ); multi-kernel REML (`fit_multi_reml`, L-BFGS-B on raw σ²); `fit_with_homoeolog_fallback` (auto-drops `K_hom`) |
| `scan.py` | EMMAX/P3D fixed-V score test (`χ² = U²/I ~ χ²₁`), batched `P@G` GEMM, CPU NumPy + single-GPU Torch, LOCO variants, wheat-scale streaming, `lambda_gc` |
| `calibration.py` | null-simulation χ̄² mixture LRT type-I + Wilson CI |
| `diagnostics.py` | nested REML LRT, Self–Liang boundary LRT, parametric bootstrap LRT, PVE sensitivity grid |
| `khom_tier1.py` | `K_hom` Tier-1 **defensive** analyses (charter §2.1 hard gate) — boundary LRT, recall ablation, spike-in power *(one spike-in path is a placeholder, `paper_value=False`)* |
| `gp.py` | GBLUP genomic prediction with subgenome-stratified GRM tiers, stratified CV, top-k enrichment, paired-bootstrap CI |
| `sim.py` | pure phenotype-simulation benchmark utilities (no external binaries, no Torch) |
| `species_split.py` | generalized per-subgenome split driving `bcftools` + `plink2` from a species YAML |
| `species_config.py` | species configuration schema (replaces v1 hardcoded per-species paths) |
| `interact.py` | Phase-7 gene-resolution homoeolog-pair burden-product interaction engine (pairwise + triad, weighted + multi-trait extensions, conditional-sanity diagnostic) |
| `cli.py` | argument parsing + `fit` / `split` / `interact` dispatch |

The zero-shot DL variant prior is **not** a package module; it lives in
`scripts/phase3/m3_3_score_dl_prior.py` (+ fusion/eval scripts) and uses borrowed
pretrained models (PlantCaduceus, AgroNT). See [`04_method.md`](04_method.md) §7.

## Data formats (actually used)

| Stage | Format |
|---|---|
| Genotype on disk | PLINK1 `.bed/.bim/.fam` and PLINK2 `.pgen/.pvar/.psam` |
| Genotype ingest | VCF.gz → BED/pgen via `plink2` |
| GRM / burden caches | `.npz` / `.npy` |
| Phenotype | `.tsv` (sample × trait) |
| Results | `.json` (run summaries, provenance) and `.tsv` (per-SNP / per-pair tables) |

No Zarr probability cubes, no Parquet result store, no GFA/graph formats.

## Compute model (2× RTX 3080 20 GB, no NVLink)

- **CPU-first.** GRM, REML, calibration, the interaction engine, GBLUP and simulation all
  run on NumPy/SciPy. For the panels analysed (n ≤ ~1050) these are seconds-to-minutes.
- **Single-GPU per-SNP scan.** `scan.py` has an optional Torch backend (`cuda:0`,
  `allow_tf32=False` for exact small p-values) that batches the per-marker test as one
  `P @ G` GEMM; a streaming variant supports bounded-memory scanning for wheat-scale
  (~86 M SNP) panels.
  An optional CuPy path accelerates the REML eigendecomposition.
- **"Dual-GPU" = two independent processes**, not a scheduler. The wheat scan can run two
  separate Python processes pinned to different subgenomes/cards; DL-prior inference is a
  separate script on the other card. There is **no** NCCL, NVLink, FSDP, or unified
  multi-GPU dispatcher.

### Environments

- `polygwas-cpu` (micromamba): Python + `plink2`, `bcftools`, `diamond`, `MCScanX`,
  scikit-learn, joblib — the package core + all Phase-7 ETL.
- `polygwas-gpu` (conda): Torch + `transformers==4.46.3` (locked; 5.x breaks Torch 2.4) +
  PlantCaduceus / AgroNT — DL-prior inference only.

## CI

GitHub Actions: `ruff` lint gate, `pytest` unit tests, and a build job. The package installs
as `pip install -e .` into `polygwas-cpu` (no GPU needed for the core/tests).

## Considered but not implemented

These were in earlier design notes and are **not part of the current system**. They are
recorded here as past design exploration, not as roadmap commitments or implied capability:

- **Rust I/O layer** (`noodles`/`polars`/`arrow2`, pgen/BGEN/GFA readers, Zarr dosage cube).
- **JAX + Triton numerical core** (fused discrete-dosage GRM kernels, matrix-free CG REML,
  GPU AI-REML).
- **REGENIE-style two-step stacked-ridge LMM** and **saddlepoint approximation (SPA)** for
  skewed/rare-variant tests. The implemented scan is a single-step EMMAX/P3D score test.
- **Graph-node / structural-variant integration** (GFA bubble dosage, NodeGWAS/Varigraph
  node tests). De-scoped 2026-05-20 — Tier-1 panels have no reads/graph (see charter §2.2).
- **Polyploid dosage beyond additive hardcalls** — hexaploid `{0..6}` / octoploid `{0..8}`
  expected dosage and the allelic-series basis (`g`, `g(g−1)`, …). The scan uses `{0,1,2}`.
- **Knockoff filter** and **multi-resolution Group-BH** FDR. Implemented multiplicity control
  is Bonferroni (per subgenome / per pair set) and ACAT.
- **BSLMM / `hibayes` Bayesian engine** as an alternative or primary model.
- **Workflow engine** (Snakemake/Nextflow). Runs are driven by YAML configs + shell scripts.
