# HomoeoGWAS

**Subgenome-aware scalable LMM + DL prior conditional lift for allopolyploid GWAS**

[![CI](https://github.com/Shipeng-Yang/HomoeoGWAS/actions/workflows/ci.yml/badge.svg)](https://github.com/Shipeng-Yang/HomoeoGWAS/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](pyproject.toml)
[![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)](pyproject.toml)
[![Tests](https://img.shields.io/badge/tests-287%20passed-brightgreen.svg)](#testing)
<!-- DOI badge added after the first Zenodo release: -->
<!-- [![DOI](https://zenodo.org/badge/DOI/<10.5281/zenodo.XXXXXXX>.svg)](https://doi.org/<10.5281/zenodo.XXXXXXX>) -->

HomoeoGWAS is a GWAS framework for **any allopolyploid crop**. A new species is
added through a single YAML config — no framework code changes (see
[Adding a new species](#adding-a-new-species)). The five panels validated so far
span ploidy 2n–8n and are illustrative, **not** an exhaustive list of supported
species: wheat AABBDD, cotton AADD, rapeseed AACC, oat AACCDD, strawberry
octoploid. The method requires distinguishable subgenomes (a homoeologous
chromosome naming or `chrom_map`) and, for the DL-prior step, a reference FASTA.

It combines:

1. **Subgenome-partitioned random effects LMM** — `y = Xβ + u_A + u_B [+ u_D ...] + ε`, fit per-subgenome GRM under REML, with LOCO option per logical chromosome.
2. **Optional homoeolog Hadamard kernel** (`K_hom = K_A ⊙ K_B [⊙ K_D]`) as **scope-conditional** epistasis term — currently treated as transparent negative finding pending synteny-aware revival (see the project charter §2.1 under `reproducibility/paper/notes/`).
3. **Zero-shot DL prior re-ranking** — PlantCaduceus + AgroNT log-likelihood fused with GWAS p-value as `z(-log10 p) + β · z(|LLR|)` over suggestive hits + LD blocks (~10⁵ SNPs); per-panel β chosen by LOQO grid.
4. **Dual-GPU native stack** — per-SNP LMM scan on 2× RTX 3080 20 GB (~90 min for wheat 83.3M markers × 827 samples LOCO);ablation evidence + scalable workflow as the engineering contribution.

## Release status & known limitations

**v1.0.0 is a software + analysis-code release; the associated manuscript is
unpublished (in preparation, target Nature Communications). A Zenodo DOI is
pending the first GitHub release.** Please treat the loci below as illustrative
of the pipeline, not as validated discoveries.

Known limitations a reviewer should be aware of:
- **Reference-match rate 45–50%** for the Horvath2020 and cotton hebau DL-prior
  subset analyses (REF-allele agreement on the FASTA windows). Current results
  are PASS-REF-subset; a Darmor v4.1 original-FASTA re-run + cotton SNP-call
  audit is a planned follow-up.
- The homoeolog Hadamard kernel `K_hom` is a **scope-conditional, transparent
  negative finding** (variance component at the REML boundary in all four GBLUP
  panels), not a headline result.
- Oat and strawberry are **validation / sanity panels**; their anchors are not
  independent of the source studies.

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
| Phase 5d (oat AACCDD 6n blind-run, Rahman 2025 OLD panel) | ✅ — `pytest 287 passed + 1 skipped` (CPU suite) |
| Phase 4 (v1.0 release: repo + Zenodo + Snakemake + mkdocs + CI) | ⏳ this artifact set |

Five panels spanning ploidy 2n→8n run through the same framework with zero
core-code change per species (subgenome-aware LMM universality).

**Recovered known loci and candidate associations (illustrative, not independently validated):**
- Wheat *Triticum aestivum* Watkins, days_to_emerg: `chr6D:142021157 p=4.9e-12`, candidate locus near *TaCAD-D1* (cinnamyl alcohol dehydrogenase, ~90 kb)
- Cotton *Gossypium hirsutum* hebau, fiber_length: `chrD11:21714989 p=5.3e-14`, candidate locus on an ortholog-coordinate-corrected anchor set (putative; not independently replicated)
- Rapeseed *Brassica napus* Horvath2020, bloom_50pct: `chrA10`, recovery of canonical *BnaA10.FT* (lead 14 kb away)
- Strawberry Pincot 2018, mean_score: `chr2A:28339895 p=8.5e-153`, recovery of the published *Fw1* (*Fusarium oxysporum* race 1) signal (87% subgenome-A marker bias caveat)
- Oat *Avena sativa* Rahman 2025 OLD: D-subgenome-localized association signals for environmental-differentiation traits (candidate loci; sanity/robustness panel, not a main figure, and anchors come from the same study so this is a pipeline sanity check rather than independent replication)

## Quick start

```bash
# 1. Install (CPU)
pip install homoeogwas            # or: pip install -e ".[dev]" from a checkout

# 2. Verify the install end-to-end (~2 s): synthesise a tiny dataset + run a fit
homoeogwas demo --keep            # prints acceptance checks + lists the outputs

# 3. Run on your own data
homoeogwas validate -c my_run.yaml    # check config + input paths first
homoeogwas fit -c my_run.yaml -o results/my_run

# (Optional) GPU extras for the per-SNP scan + DL prior
pip install "homoeogwas[gpu]"     # transformers locked at 4.46.3
```

See [`examples/minimal/`](examples/minimal/) for the demo dataset + an annotated
config, and [the I/O contract](docs/io.md) for input/output formats. CLI
subcommands: `fit`, `validate`, `demo`, `split`, `interact`.

## Docker

```bash
# CPU image (includes plink2 + bcftools, so split/VCF -> fit all work)
docker build -t homoeogwas:cpu .
docker run --rm homoeogwas:cpu demo                       # self-test
docker run --rm -v "$PWD":/work -w /work homoeogwas:cpu fit -c run.yaml

# GPU image (per-SNP scan + DL prior; CUDA 12.1)
docker build -f Dockerfile.gpu -t homoeogwas:gpu .
docker run --rm --gpus all -v "$PWD":/work -w /work homoeogwas:gpu fit -c run.yaml --backend gpu
```

Pass `--build-arg PIP_INDEX_URL=<mirror>` to build through a faster pip mirror.

## Adding a new species

The framework is **not limited to the validated panels below** — any
allopolyploid is supported through configuration alone:

1. Copy an existing `configs/species/*.yaml` and edit `subgenomes`, the
   chromosome-naming prefix / `chrom_map`, the reference assembly path, and
   `ploidy`. The schema (`src/homoeogwas/species_config.py`) validates it.
2. `homoeogwas split --species <yaml> --vcf <in.vcf.gz> --out-dir ...` splits
   markers into per-subgenome genotype sets. `K_hom` auto-selects the right
   form for the subgenome count (2/3-way Hadamard; pairwise-mean for 4 sub to
   stay full-rank).
3. `homoeogwas fit --config <run.yaml>` runs the LMM scan; the optional DL-prior
   step additionally needs the species reference FASTA.

No Python is edited at any step. The oat panel (the 5th species) was added this
way as a zero-code-change test of the schema.

Requirements: distinguishable subgenomes (homoeologous chromosome naming or a
`chrom_map`); a reference FASTA for the DL-prior re-ranking. Diploids can run
the LMM, but the homoeolog Hadamard kernel `K_hom` is not meaningful for them.

## Validated species panels (Phase 5d milestone)

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

See [`reproducibility/paper/notes/00_charter.md`](reproducibility/paper/notes/00_charter.md) for the frozen claim hierarchy and [`reproducibility/paper/notes/03_architecture.md`](reproducibility/paper/notes/03_architecture.md) for tech-stack details.

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
pytest                             # CPU suite: 287 passed + 1 skipped (GPU scan test skips without torch)
```

CI runs ruff + pytest CPU smoke on Python 3.10 / 3.11 / 3.12. See `.github/workflows/ci.yml`.

## Reproducing baselines

The eight benchmark tools (NodeGWAS, GWASpoly, networkGWAS, regenie, SAIGE,
PanGenie, STAAR, deepRVAT) are **not vendored** in this repository (size +
heterogeneous licenses). Clone them on demand:

```bash
bash reproducibility/paper/scripts/reproduce_baselines.sh   # clones into benchmarks/
```

Large inputs (`data/`, ~950 GB) and intermediate outputs (`results/`, ~19 GB)
are git-ignored. Release reproducibility fingerprints (SHA256 of the key paper
artifacts) are committed at `reproducibility/paper/results_phase4/reproducibility_fingerprints.tsv`;
re-generate the DL-prior figure inputs with:

```bash
cd reproducibility && rm -rf ../results/phase3/m3_3_dl_prior_v2 && snakemake paper_figures --cores 4
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
