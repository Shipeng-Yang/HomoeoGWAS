# HomoeoGWAS

**Subgenome-aware scalable LMM + DL prior conditional lift for allopolyploid GWAS**

[![CI](https://github.com/Shipeng-Yang/HomoeoGWAS/actions/workflows/ci.yml/badge.svg)](https://github.com/Shipeng-Yang/HomoeoGWAS/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](pyproject.toml)
[![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)](pyproject.toml)
[![Tests](https://img.shields.io/badge/tests-260%20passed-brightgreen.svg)](#testing)
<!-- DOI badge added after the first Zenodo release: -->
<!-- [![DOI](https://zenodo.org/badge/DOI/<10.5281/zenodo.XXXXXXX>.svg)](https://doi.org/<10.5281/zenodo.XXXXXXX>) -->

HomoeoGWAS is a GWAS framework for **allopolyploid crops** (wheat AABBDD, cotton AADD, rapeseed AACC, oat AACCDD, strawberry octoploid). It combines:

1. **Subgenome-partitioned random effects LMM** — `y = Xβ + u_A + u_B [+ u_D ...] + ε`, fit per-subgenome GRM under REML, with LOCO option per logical chromosome.
2. **Optional homoeolog Hadamard kernel** (`K_hom = K_A ⊙ K_B [⊙ K_D]`) as **scope-conditional** epistasis term — currently treated as transparent negative finding pending synteny-aware revival (see [charter §2.1](docs/00_charter.md)).
3. **Zero-shot DL prior re-ranking** — PlantCaduceus + AgroNT log-likelihood fused with GWAS p-value as `z(-log10 p) + β · z(|LLR|)` over suggestive hits + LD blocks (~10⁵ SNPs); per-panel β chosen by LOQO grid.
4. **Dual-GPU native stack** — per-SNP LMM scan on 2× RTX 3080 20 GB (~90 min for wheat 83.3M markers × 827 samples LOCO);ablation evidence + scalable workflow as the engineering contribution.

## Status (2026-05-25)

| Phase | Status |
|---|---|
| Phase 0 (data + env)              | ✅ |
| Phase 1 (3 Tier-1 panel build-out) | ✅ |
| Phase 2 (Core LMM + Horvath + wheat + simulation) | ✅ |
| Phase 3 (M3.1 LOCO + M3.2 known QTL + M3.3 DL prior + M3.4 v2 three-panel cross-trait) | ✅ |
| Phase 5a (Species YAML schema + `homoeogwas split` + K_hom n-sub fallback) | ✅ |
| Phase 5b (strawberry 8n Pincot 2018 validation panel) | ✅ |
| Phase 5c (GBLUP Table 1 — K_hom universal null finding) | ✅ |
| Phase 5d (oat AACCDD 6n blind-run, Rahman 2025 OLD panel) | ✅ — `pytest 260 passed + 1 skipped` |
| Phase 4 (v1.0 release: repo + Zenodo + Snakemake + mkdocs + CI) | ⏳ this artifact set |

Five panels spanning ploidy 2n→8n run through the same framework with zero
core-code change per species (subgenome-aware LMM universality).

**Real-data novel / recovered loci (charter §3.3 hard gate ≥1; we have ≥4 panels)**:
- Wheat *Triticum aestivum* Watkins, days_to_emerg: `chr6D:142021157 p=4.9e-12`, near *TaCAD-D1* (cinnamyl alcohol dehydrogenase, ~90 kb)
- Cotton *Gossypium hirsutum* hebau, fiber_length: `chrD11:21714989 p=5.3e-14` (novel; ortholog-coordinate-corrected anchor set)
- Rapeseed *Brassica napus* Horvath2020, bloom_50pct: `chrA10` near *BnaA10.FT* (14 kb recovery)
- Strawberry Pincot 2018, mean_score: `chr2A:28339895 p=8.5e-153` (Fw1 *Fusarium oxysporum* race 1 resistance, paper replication; 87% subgenome-A marker bias caveat)
- Oat *Avena sativa* Rahman 2025 OLD: D-subgenome-localized association signals for environmental-differentiation traits (31 novel candidates; sanity/robustness panel, not a main figure)

## Quick start

```bash
# 1. Install (CPU)
pip install -e ".[dev]"

# 2. (Optional) GPU extras for per-SNP scan + DL prior
pip install -e ".[gpu]"   # transformers locked at 4.46.3

# 3. Fit a single trait
homoeogwas fit --config configs/runs/fit_horvath2020_plant_height_loco.yaml

# 4. Configure a new allopolyploid species (zero code change in framework)
cp configs/species/wheat_aestivum.yaml configs/species/myspecies.yaml
# edit subgenomes + chrom_map + paths
homoeogwas split --species configs/species/myspecies.yaml --vcf in.vcf.gz --out-dir data/processed/myspecies/
```

The CLI exposes **≤ 5 flags** (`--config`, `--out-dir`, `--backend`, `--dry-run`, `--force`) per charter §3.3 release item.

## Supported species (Phase 5d milestone)

| Species | Genome | Panel | Samples | SNPs | Reference | Status |
|---|---|---|---|---|---|---|
| Wheat *T. aestivum* | AABBDD 6n | Watkins | 827 | 83.3M | IWGSC RefSeq v1.0 | ✅ |
| Cotton *G. hirsutum* | AADD 4n | hebau | 419 | 498k | HBAU NDM8 | ✅ |
| Rapeseed *B. napus* | AACC 4n | Horvath2020 | 297-428 | 51k | Darmor v4.1 | ✅ |
| Strawberry *F. ananassa* | 8n (4 sub) | Pincot 2018 | 564 | 30k | NIHHS Seolhyang (BLAST remap) | ✅ |
| Oat *A. sativa* | AACCDD 6n | Rahman 2025 OLD | 737 | 394k | OT3098 v2 | ✅ |

## Architecture

```
src/homoeogwas/
├── species_config.py    # Pydantic schema for configs/species/*.yaml
├── species_split.py     # Generic VCF → per-subgenome pgen splitter
├── grm.py               # Per-subgenome GRM + LOCO GRM utilities
├── kernel.py            # K_pool (additive sum), K_hom (Hadamard / pairwise_mean auto-fallback)
├── lmm.py               # REML multi-kernel LMM
├── gp.py                # GBLUP prediction + stratified CV + paired bootstrap CI
├── scan.py              # CPU + dual-GPU per-SNP scan (LOCO & EMMAX backends)
├── diagnostics.py       # λ_GC, QQ, retained-fraction warnings
├── calibration.py       # null-simulation type-I error
├── sim.py               # power-vs-FDR simulation benchmark
├── cli.py               # `homoeogwas fit` / `homoeogwas split`
└── io.py                # io helpers
```

See [`docs/00_charter.md`](docs/00_charter.md) for the frozen claim hierarchy and [`docs/03_architecture.md`](docs/03_architecture.md) for tech stack details.

## DL prior conditional lift (Phase 3.3 / 3.4 result)

Across three panels and four traits, fusing the PlantCaduceus + AgroNT ensemble log-likelihood with the GWAS p-value (β ∈ {0.25, 2, 5} chosen per-panel by LOQO) recovered canonical genes that the GWAS alone ranked outside the top-100:

| Panel | Trait | β | recall@100 GWAS → fused | Best single rank flip |
|---|---|---|---|---|
| Horvath2020 | bloom_50pct | 2 | 0.10 → **0.80** | *BnaC03.GA20ox* 221 → 2 |
| Horvath2020 | plant_height | 0.25 | 0.60 → 0.90 | *BnaC09.FLC* 148 → 72 |
| Cotton hebau | fiber_length | 5 | 0.00 → 0.20 | ***GhMYB25* 763 → 44** |
| Cotton hebau | lint_percentage | 5 | 0.00 → 0.20 | ***GhMYB25* 951 → 34** |
| Wheat Watkins (reference) | days_to_emerg | 0.25 | n/a | 0 (power-saturated baseline; see paper Methods) |

**Framing**: DL prior gain is **panel-size-dependent** — lift is largest when GWAS marginal power is limited (small panels). Wheat (n=827, 31k suggestive) shows no recall@100 lift, consistent with marginal-power saturation; presented as supplementary panel-size scaling story.

## Testing

```bash
pytest -m "not gpu and not slow"   # CPU tests (~3-5 min)
pytest -m "not slow"               # + GPU tests (requires polygwas-gpu env)
pytest                             # full suite incl. simulation benchmarks (260 passed + 1 skipped)
```

CI runs ruff + pytest CPU smoke on Python 3.10 / 3.11 / 3.12. See `.github/workflows/ci.yml`.

## Reproducing baselines

The eight benchmark tools (NodeGWAS, GWASpoly, networkGWAS, regenie, SAIGE,
PanGenie, STAAR, deepRVAT) are **not vendored** in this repository (size +
heterogeneous licenses). Clone them on demand:

```bash
bash scripts/reproduce_baselines.sh   # clones into benchmarks/
```

Large inputs (`data/`, ~950 GB) and intermediate outputs (`results/`, ~19 GB)
are git-ignored. Release reproducibility fingerprints (SHA256 of the key paper
artifacts) are committed at `results/phase4/reproducibility_fingerprints.tsv`;
re-generate the DL-prior figure inputs with:

```bash
rm -rf results/phase3/m3_3_dl_prior_v2 && snakemake paper_figures --cores 4
```

## Citation

Manuscript in preparation.

```bibtex
@unpublished{homoeogwas2026,
  title  = {HomoeoGWAS: subgenome-aware scalable LMM with DL prior conditional lift for allopolyploid GWAS},
  author = {Yang, Shipeng},
  year   = {2026},
  note   = {In preparation; target Nature Communications},
  url    = {https://github.com/Shipeng-Yang/HomoeoGWAS},
}
```

See [`CITATION.cff`](CITATION.cff) for machine-readable citation metadata.

## License

MIT — see [LICENSE](LICENSE).
