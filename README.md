# HomoeoGWAS

**Subgenome-aware trait architecture and homoeolog-interaction analysis for allopolyploid crops.**

[![CI](https://github.com/Shipeng-Yang/HomoeoGWAS/actions/workflows/ci.yml/badge.svg)](https://github.com/Shipeng-Yang/HomoeoGWAS/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](pyproject.toml)
[![Version](https://img.shields.io/badge/version-2.1.0-blue.svg)](pyproject.toml)
[![Tests](https://img.shields.io/badge/tests-CI%20passing-brightgreen.svg)](#testing)
<!-- DOI badge added after the first Zenodo release:
[![DOI](https://zenodo.org/badge/DOI/<10.5281/zenodo.XXXXXXX>.svg)](https://doi.org/<10.5281/zenodo.XXXXXXX>) -->

> **Current release: [v2.1.0](https://github.com/Shipeng-Yang/HomoeoGWAS/releases/tag/v2.1.0).**
> This release adds a heteroscedasticity-robust bootstrap null for group omniB
> (`null_variance: smooth_pc4`), chunk-parallel streaming scans and MCP audit
> and summary tools. See the [release notes](docs/releases/v2.1.0.md), the
> [release history](#release-history) below and the [changelog](CHANGELOG.md).

HomoeoGWAS runs GWAS on **allopolyploid crops** (wheat, cotton, rapeseed, oat,
peanut, strawberry, …) by modelling each subgenome explicitly. A new species is added
through a single YAML config — no framework code changes. The only requirement
is that the subgenomes are distinguishable (a homoeologous chromosome naming or a
`chrom_map`).

It combines:

1. **Subgenome-partitioned linear mixed model** — `y = Xβ + u_A + u_B [+ u_D …] + ε`,
   with a per-subgenome GRM fit by REML and an optional leave-one-chromosome-out
   (LOCO) correction.
2. **Gene-resolution homoeolog-group interaction tests** — one engine for 2,
   3 or 4+ copies, with pair omniB as the primitive and kinship-preserving
   bootstrap min-P as the experiment-wide calibration. omniB combines
   minor-burden, PC1 and kernel-Hadamard evidence while retaining all three
   components for interpretation.
3. **Evidence-aware outputs** — the per-subgenome variance fingerprint,
   interaction rankings, prediction comparison and a result audit that
   distinguishes internal discovery from replication.
4. **Scalable per-SNP scanning** — streaming CPU and optional CUDA backends for
   panels with tens of millions of markers.

The global `K_hom` kernel and phenotype-independent external priors remain
optional research extensions. They are not required for the primary variance
partition or interaction workflow and should not be treated as discovery
evidence without a frozen benchmark.

## Quick start

```bash
# 1. Install the current v2.1.0 release (CPU)
python -m pip install \
  "homoeogwas @ https://github.com/Shipeng-Yang/HomoeoGWAS/archive/refs/tags/v2.1.0.tar.gz"
# Contributors may instead use: python -m pip install -e ".[dev]"

# 2. Verify the install end-to-end (~2 s): synthesise a tiny dataset + run a fit
homoeogwas demo --keep            # prints acceptance checks + lists the outputs

# 3. Run on your own data
homoeogwas validate -c my_run.yaml    # check config + input paths first
homoeogwas fit -c my_run.yaml -o results/my_run
homoeogwas audit results/my_run       # validity, uncertainty and evidence limits

# (Optional) GPU extras for the per-SNP scan
python -m pip install \
  "homoeogwas[gpu] @ https://github.com/Shipeng-Yang/HomoeoGWAS/archive/refs/tags/v2.1.0.tar.gz"
```

PyPI currently serves the older v1.0.1 package. Until v2.1.0 is published
there, use the versioned GitHub command above for new analyses; an unpinned
`pip install homoeogwas` will not install the algorithms described here.

See [`examples/minimal/`](examples/minimal/) for the demo dataset + an annotated
config, and [the I/O contract](docs/io.md) for input/output formats. Main CLI
subcommands: `split`, `validate`, `fit`, `predict`, `interact`, `follow-up`,
`registry`, `audit`, `plot`, `locus`, `rplot`, `design`, `prep-snps`, and
`prep-homoeologs`.

### Freeze several species in one production registry

HomoeoGWAS can generate, validate, resume, audit and index multiple species through one
species-independent registry:

```bash
homoeogwas registry validate -c analyses/cross_species_interaction_inventory.yaml
homoeogwas registry run -c analyses/cross_species_interaction_inventory.yaml --dry-run
```

Species are metadata. New two-, three- and four-copy interaction runs use the same pair-edge
group omniB engine; see [the registry guide](docs/run_registry.md).

### Stability and functional evidence for formal discoveries

After a canonical group omniB run, one species-independent command reproduces the formal
discoveries, performs deterministic material deletion, optionally deletes environments, merges
redundant units into reporting regions, and joins provenance-bound annotation, expression, QTL,
literature or functional evidence:

```bash
homoeogwas follow-up results/my_interaction \
  --material-folds 20 --n-jobs 16 \
  --environment-col environment \
  --evidence analyses/my_trait.evidence.yaml
```

The deletion analyses are candidate-only internal sensitivity checks, not a new discovery family
or independent replication. The command requires the full audited family and writes an immutable
identity-bound dossier, so edited rankings and stale output files fail closed. See
[the follow-up guide](docs/followup.md).

## Run it by talking to an AI agent (no YAML, no coding)

You do **not** have to write config files. HomoeoGWAS ships an **agent interface**:
tell an AI coding agent your data files and which trait you care about, and the
agent writes the configs, runs the analysis, and explains the results for you.
This is the fastest way for a breeder or bioinformatician to get going.

You just say, in plain language:

> "Run a subgenome-stratified GWAS on my wheat. Genotypes are in
> `data/A.bed`, `data/B.bed`, `data/D.bed`; phenotype is `pheno.tsv`,
> sample column `id`, trait `heading_date`. Then plot it."

The agent collects only those *biological* inputs and does the rest. Three ways
to connect, pick whichever matches the agent you already use:

```bash
# ── Option A · Claude Code (zero setup) ───────────────────────────────────────
# The skill is bundled at .claude/skills/homoeogwas/. Open this repo in Claude
# Code and just ask in plain language — the `homoeogwas` skill auto-activates.

# ── Option B · Any MCP client (Cursor, Cline, Windsurf, Claude Desktop, …) ─────
# Install the current release with the MCP dependency:
python -m pip install \
  "homoeogwas[mcp] @ https://github.com/Shipeng-Yang/HomoeoGWAS/archive/refs/tags/v2.1.0.tar.gz"
homoeogwas mcp                    # starts the MCP server (stdio)
# Then register this server in your client's MCP config. Minimal entry:
#   {"mcpServers": {"homoeogwas": {"command": "homoeogwas", "args": ["mcp"]}}}
#   ^ replace "command" with an absolute path if `homoeogwas` is not on PATH,
#     e.g. "/home/you/miniconda3/envs/poly/bin/homoeogwas"

# ── Option C · Any other LLM/agent (ChatGPT, Gemini, a custom bot) ────────────
# Point it at AGENTS.md in the repo root — that file IS the full, canonical
# spec the agent follows. No plugin needed; the agent reads it and drives the CLI.
```

`AGENTS.md` is the single source of truth; the Claude skill and the MCP server
both defer to it, so all three routes behave identically. Under the hood the
agent calls the same `src/homoeogwas/workflow.py` engine (high-level inputs →
auto-generated YAML → run → summary), and it **blocks on common mistakes**
(wrong chromosome naming, missing homoeolog map, …) before wasting a run.

**No GPU? You're fine — CPU is the default.** The versioned CPU installation
above runs everything on CPU; the agent uses `--backend auto`, which silently picks
GPU *only if one is present* and otherwise falls back to CPU with identical
results. A GPU is **purely optional acceleration** for the genome-wide per-SNP
scan — never a requirement. So tell the
agent "use CPU" (or just say nothing) and it works on any laptop:

```bash
# Force CPU explicitly if you like (this is also the no-GPU auto behaviour):
homoeogwas fit -c run.yaml --backend cpu     # any machine, no GPU needed
homoeogwas fit -c run.yaml --backend auto    # GPU if available, else CPU (default)
homoeogwas fit -c run.yaml --backend gpu     # opt-in acceleration; needs CUDA
```

## Containers

```bash
# Docker — CPU (bundles plink2 + bcftools, so split/VCF -> fit all work)
docker build -t homoeogwas:cpu .
docker run --rm homoeogwas:cpu demo
docker run --rm -v "$PWD":/work -w /work homoeogwas:cpu fit -c run.yaml

# Docker — GPU (per-SNP scan; CUDA 12.1)
docker build -f Dockerfile.gpu -t homoeogwas:gpu .
docker run --rm --gpus all -v "$PWD":/work -w /work homoeogwas:gpu fit -c run.yaml --backend gpu

# Apptainer / Singularity (HPC, no root) — convert the Docker image
apptainer build homoeogwas.sif docker-daemon://homoeogwas:cpu
apptainer run homoeogwas.sif demo
```

Pass `--build-arg PIP_INDEX_URL=<mirror>` to build through a faster pip mirror.

## How it works

One genotype set goes in and is split by subgenome. It feeds two analyses:

- **Workflow 1 — subgenome-stratified mixed model.** One kinship matrix per
  subgenome, a multi-kernel REML fit that partitions trait variance among the
  subgenomes (PVE), and a leave-one-chromosome-out per-SNP scan.
- **Workflow 2 — homoeolog interaction.** Homoeologous genes are grouped, every
  pair of copies (an *edge*) is tested with omniB, which combines minor-burden,
  PC1 and kernel-Hadamard encodings, and all edges or groups form one
  experiment-wide bootstrap-minP family with a kinship-preserving null.

`homoeogwas audit` then labels every result as computationally valid, an
internal discovery, or in need of replication.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/how_it_works_dark.svg">
  <img alt="HomoeoGWAS workflow: inputs are split by subgenome and feed a subgenome-stratified mixed model and a homoeolog interaction test; both end in one evidence audit" src="docs/img/how_it_works_light.svg" width="100%">
</picture>

### One interaction contract across ploidies

New interaction runs use one master `group_id,gene_<S>...` table and canonical
`mode: group`. A two-copy group has one pair edge, a three-copy group has three,
and a four-copy group has six. Edge omniB is the shared statistical primitive;
group omniB is ACAT over the group's pair edges. All directions share one
all-subgenome null model and one bootstrap-minP family—never separate AB/AD/BD
runs followed by a factor correction.

The generated config is
`<out_dir>/configs/interact.generated.group.omnib.yaml`. It declares one primary
unit: `edge` for formal homoeolog-pair discoveries, or `group` for formal
within-group omnibus evidence. The other layer and the three omniB components
are localization only unless `family_scope: joint` calibrates their union once.
Legacy pairwise/triad omniB configs remain readable. Formal canonical output is
INT-only; raw-scale sensitivity cannot inherit discovery fields.

A four-copy group therefore uses six supported pair interactions. HomoeoGWAS
never fits or claims a direct fourth-order coefficient. See the
[interaction input contract](docs/interact_inputs.md) and the
[strawberry example](docs/examples/strawberry_octoploid.md).

## Adding a new species

Any allopolyploid is supported through configuration alone:

1. Copy an existing `configs/species/*.yaml` and edit `subgenomes`, the
   chromosome naming / `chrom_map`, the reference assembly path, and `ploidy`.
   The schema in `src/homoeogwas/species_config.py` validates it.
2. `homoeogwas split --species-yaml <yaml> --vcf <in.vcf.gz> -o ...` splits the
   markers into per-subgenome genotype sets.
3. `homoeogwas fit --config <run.yaml>` runs the mixed-model scan. Interaction
   analysis additionally needs SNP-to-gene and homoeolog mappings; four copies
   become six pair edges in one group family, never one four-way test.

No Python is edited at any step. Diploids can run the mixed model, but they do
not have a cross-subgenome homoeolog interaction.

## Tested species

The framework has been run end-to-end on six allopolyploid crops, from tetraploid
to octoploid, through the same code path; this list is illustrative, not a limit
on supported species.

| Species | Subgenomes | Reference assembly |
|---|---|---|
| Wheat (*Triticum aestivum*)     | AABBDD (6n) | IWGSC RefSeq v1.0 |
| Cotton (*Gossypium hirsutum*)   | AADD (4n)   | HBAU NDM8 |
| Rapeseed (*Brassica napus*)     | AACC (4n)   | Darmor v4.1 |
| Peanut (*Arachis hypogaea*)     | AABB (4n)   | NDH108 (PeanutPan) |
| Oat (*Avena sativa*)            | AACCDD (6n) | OT3098 v2 |
| Strawberry (*Fragaria × ananassa*) | octoploid (4 subgenomes) | NIHHS Seolhyang |

## Package layout

```
src/homoeogwas/
├── cli.py               # command-line interface
├── species_config.py    # species YAML schema
├── species_split.py     # VCF -> per-subgenome genotype sets
├── grm.py, grm_cache.py # per-subgenome and LOCO kinship matrices
├── lmm.py               # multi-kernel REML mixed model
├── scan.py              # per-SNP scan (streaming CPU, parallel chunks, optional GPU)
├── gp.py                # GBLUP prediction + cross-validation
├── prep.py              # SNP-to-gene maps and homoeolog tables
├── interact.py          # homoeolog interaction entry point
├── omnib_family.py      # edge/group omniB scores and bootstrap scoring
├── group_family.py      # homoeolog groups and their pair edges
├── parallel.py          # forked worker pool for bootstrap blocks
├── followup.py          # stability and evidence follow-up
├── audit.py             # evidence/status audit of finished runs
├── run_registry.py      # multi-species production registry
├── plots.py             # publication figures from finished runs
├── workflow.py          # high-level orchestration used by the agent tools
└── mcp_server.py        # MCP server for AI agents
```

## Testing

```bash
pytest -m "not gpu and not slow"   # CPU suite; external-tool/GPU tests may skip
pytest -m "not slow"               # + GPU tests (needs torch)
pytest                             # full suite incl. simulation benchmarks
```

CI runs ruff + the CPU test suite on Python 3.10 / 3.11 / 3.12.

## Status

HomoeoGWAS is research software under active development. A preprint describing
the methods and their applications is in preparation and will be posted on
bioRxiv; this page will link to it once it is available.

## Citation

Until the preprint is available, please cite the software release you used:

```bibtex
@software{homoeogwas,
  author  = {Yang, Shipeng},
  title   = {HomoeoGWAS: subgenome-aware trait architecture and homoeolog-interaction
             analysis for allopolyploid crops},
  year    = {2026},
  version = {2.1.0},
  url     = {https://github.com/Shipeng-Yang/HomoeoGWAS}
}
```

See [`CITATION.cff`](CITATION.cff) for machine-readable metadata.

## Release history

Full details are in the [changelog](CHANGELOG.md) and the
[release notes](docs/releases/).

| Version | Date | Main changes |
|---|---|---|
| [v2.1.0](https://github.com/Shipeng-Yang/HomoeoGWAS/releases/tag/v2.1.0) | 2026-10-05 | Heteroscedasticity-robust bootstrap null for group omniB (`null_variance: smooth_pc4`, the default for agent-run analyses); `scan.n_jobs` runs streaming scan chunks in parallel with output identical to the serial scan; MCP `audit_results` and `summarize_results`; a group is tested only when all of its pair edges are estimable. |
| [v2.0.1](https://github.com/Shipeng-Yang/HomoeoGWAS/releases/tag/v2.0.1) | 2026-08-29 | One interaction contract across ploidies: pair edges as the shared primitive for 2-, 3- and 4-copy groups with one bootstrap-minP family; component-aware rankings, `homoeogwas audit` and `follow-up` stability checks; process-based parallel workers; BIM-bound marker manifests. |
| [v2.0.0](https://github.com/Shipeng-Yang/HomoeoGWAS/releases/tag/v2.0.0) | 2026-07-25 | Breaking: corrected the multiplicity of triad/clique burden scans (groups with three or more copies), non-estimable tests reported as missing instead of p = 1, strict JSON output, and exactly one alpha-spending procedure per run. |
| [v1.0.2](https://github.com/Shipeng-Yang/HomoeoGWAS/releases/tag/v1.0.2) | 2026-06-21 | Optional dominance adjustment for the homoeolog interaction test. |
| [v1.0.1](https://github.com/Shipeng-Yang/HomoeoGWAS/releases/tag/v1.0.1) | 2026-06-12 | First PyPI release (features as v1.0.0). |
| [v1.0.0](https://github.com/Shipeng-Yang/HomoeoGWAS/releases/tag/v1.0.0) | 2026-06-12 | First public release: subgenome-stratified mixed-model GWAS, VCF splitting, homoeolog interaction test, plotting, input-preparation tools and the agent interface (AGENTS.md, skill, MCP server). |

## License

MIT — see [LICENSE](LICENSE).
