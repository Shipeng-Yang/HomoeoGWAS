# API Reference

The Python package lives at `src/homoeogwas/`. This page summarizes the module-level surface; for full docstrings, browse the source or build the docs locally with mkdocstrings (planned for v1.0).

## Top-level entry points

```python
from homoeogwas import species_config, species_split, grm, kernel, lmm, gp, scan, cli
```

| Module | What it does |
|---|---|
| `species_config` | Pydantic schema for `configs/species/*.yaml`; tiered validator (file existence + chrom_map coverage + pheno header + dosage_model sanity + optional pysam fasta-chrom resolution) |
| `species_split` | Generic VCF → per-subgenome pgen splitter, bcftools/plink2 backends, driven by `SpeciesConfig` |
| `grm` | Per-subgenome GRM construction; LOCO partial-GRM utilities |
| `kernel` | `hadamard_kernel`, `pairwise_mean_kernel`, `build_homoeolog_kernel` (auto-fallback), `sum_kernel`, `normalize_kernel` |
| `lmm` | REML multi-kernel LMM, score test for per-SNP association |
| `gp` | GBLUP (Phase 5c): `fit_gblup`, `stratified_cv`, `paired_bootstrap_ci` |
| `scan` | Per-SNP scan (CPU + dual-GPU), LOCO context builder, EMMAX backend |
| `diagnostics` | λ_GC, QQ plot, retained-fraction warning, top-hit cluster collapse |
| `calibration` | Null-simulation type-I error |
| `sim` | Power-vs-FDR simulation benchmark (Phase 2 M2.6) |
| `cli` | `homoeogwas fit` / `homoeogwas split` entrypoint |
| `io` | bed-reader + plink2 + bcftools wrappers |

## CLI

```
$ homoeogwas --help
usage: homoeogwas [-h] {fit,split} ...

  fit    Run subgenome-aware LMM scan on a panel × trait
  split  Split a panel VCF into per-subgenome pgen files (driven by configs/species/*.yaml)
```

## Selected example — load a species config

```python
from homoeogwas.species_config import load_species_config

cfg = load_species_config("configs/species/wheat_aestivum.yaml")
print(cfg.id, cfg.ploidy, [sg.id for sg in cfg.subgenomes])
# wheat_aestivum 6 ['A', 'B', 'D']

# A `_validation_report` is attached if `validate=True` (default)
report = cfg.__dict__["_validation_report"]
print(report.n_chroms_total, report.n_known_qtl, report.warnings)
# 21 13 []
```

## Selected example — build K_hom with auto-fallback

```python
import numpy as np
from homoeogwas.kernel import build_homoeolog_kernel

# Wheat: 3 sub → 3-way Hadamard
grms_wheat = {"A": K_A, "B": K_B, "D": K_D}
K_hom, mode = build_homoeolog_kernel(grms_wheat, mode="auto")
assert mode == "hadamard"

# Strawberry: 4 sub → pairwise_mean auto-fallback (avoids 4-way sparse collapse)
grms_straw = {"A": K_A, "B": K_B, "C": K_C, "D": K_D}
K_hom, mode = build_homoeolog_kernel(grms_straw, mode="auto")
assert mode == "pairwise_mean"
```

## Selected example — GBLUP CV with paired bootstrap CI

```python
from homoeogwas.gp import fit_gblup, stratified_cv, paired_bootstrap_ci

# tier0 = K_pool only;  tier1 = K_pool + K_hom;  tier2 = K_A + K_B + K_hom (full)
r2_tier0 = stratified_cv(y, K_list=[K_pool], n_folds=10, n_repeats=20)
r2_tier1 = stratified_cv(y, K_list=[K_pool, K_hom], n_folds=10, n_repeats=20)

delta, ci_lo, ci_hi, sig = paired_bootstrap_ci(r2_tier0, r2_tier1)
```

For full module APIs, please see source docstrings under `src/homoeogwas/`.
