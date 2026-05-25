# 03 — Architecture

## Three-layer stack

```
┌──────────────────────────────────────────────────────────────┐
│  Layer 3: DL Prior  (PyTorch, GPU-1)                         │
│    PlantCaduceus / AgroNT zero-shot inference                │
│    Output: per-variant prior weight π_j → JSON / Parquet     │
├──────────────────────────────────────────────────────────────┤
│  Layer 2: Numerical Core  (JAX + Triton, GPU-0)              │
│    • Subgenome GRM (K_A, K_B, K_D)                          │
│    • Matrix-free CG REML                                    │
│    • REGENIE-style two-step LMM                             │
│    • Homoeolog Hadamard interaction kernel                  │
│    • Score test + SPA                                       │
├──────────────────────────────────────────────────────────────┤
│  Layer 1: I/O & Tensor  (Rust, CPU)                          │
│    • pgen / BGEN / VCF / GFA-bubble readers                 │
│    • Zarr-backed dosage probability tensor (N × M × P+1)    │
│    • Streaming block dispatcher                             │
└──────────────────────────────────────────────────────────────┘
```

## Language / framework choices

| Layer | Choice | Why |
|---|---|---|
| I/O | **Rust** (`noodles`, `polars`, `arrow2`) | safe, fast streaming over 16 Gb wheat genomes; SIMD; ecosystem mature for bio |
| Numerics | **JAX + Triton** | `vmap`/`pmap` for multi-phenotype batching; Triton lets us write fused kernels for discrete-dosage GRM (cuBLAS can't) |
| DL prior | **PyTorch** + HuggingFace `transformers` | PlantCaduceus uses Mamba; HF ecosystem is the path of least resistance |
| Workflow | **Snakemake** (light) | reproducibility; lightweight relative to Nextflow for a single-lab project |
| CLI | **Click** (Python) | wraps Rust core via `pyo3` |

## Module map (under `src/`)

```
src/
├── rust_io/                              # Cargo workspace
│   ├── Cargo.toml
│   └── crates/
│       ├── ploid_io/                     # pgen/VCF/BGEN streaming reader
│       ├── ploid_tensor/                 # Zarr dosage cube writer
│       └── ploid_graph/                  # GFA bubble parser, node-dosage emitter
│
├── polygwas/                             # Python package
│   ├── __init__.py
│   ├── cli.py                            # `polygwas run` entry
│   ├── io.py                             # thin pyo3 wrapper over rust_io
│   ├── grm.py                            # subgenome GRM builder (JAX)
│   ├── lmm/
│   │   ├── reml.py                       # AI-REML on GPU
│   │   ├── step1_ridge.py                # REGENIE-style stacked ridge
│   │   ├── step2_score.py                # score test + SPA
│   │   └── interaction.py                # K_AB Hadamard kernel
│   ├── sv.py                             # graph-node dosage integration
│   ├── dl_prior.py                       # STAAR-style weight from DL embeddings
│   ├── fdr.py                            # Group-BH / knockoff / ACAT
│   └── plot.py                           # Manhattan per subgenome
│
└── kernels/                              # Triton .py kernels
    ├── grm_dosage.py                     # discrete-dosage GRM kernel
    ├── ridge_block.py                    # block-coordinate ridge solver
    └── interaction_2d.py                 # 2D scan kernel
```

## Data formats

| Stage | Format | Reason |
|---|---|---|
| Genotype on disk | `.pgen` (PLINK 2) + `.zarr` (probability cube) | pgen for hardcalls (4× smaller than BGEN), zarr for full dosage distribution |
| Pangenome graph | `.gfa.gz` from minigraph-cactus | de facto standard |
| Bubble dosage | VCF with INFO/PATH + FORMAT/DS (graph-aware) | NodeGWAS / Varigraph compatible |
| Intermediate GRM | `.h5` (Blosc-compressed HDF5) | random access, ~50% disk vs npy |
| Results | `.parquet` (per chromosome) | DuckDB / Polars queryable |
| DL prior | `.parquet` (variant_id → π) | joined at score-test time |

## GPU plan (2× RTX 3080 20 GB, no NVLink)

### Workload allocation

| GPU | Workload | Peak VRAM |
|---|---|---|
| GPU-0 | LMM numerics: GRM + ridge + score test + interaction scan | ~6 GB sustained |
| GPU-1 | DL prior inference: PlantCaduceus / AgroNT batched | ~3 GB |

### Streaming strategy

- Genotype block size: **5,000 SNPs × 10,000 samples × float16 ≈ 100 MB**. Comfortably fits.
- GRM (N=10k, float32): 400 MB. Cache-resident.
- DL prior runs **once** per reference panel (zero-shot), result is reused across all phenotypes.

### What we **don't** try to do

- Train a DNA-LM from scratch (needs A100 80 GB × multiple — not feasible).
- FSDP multi-GPU LMM (no NVLink → too much PCIe traffic).
- Hexaploid full 9-state probability tensor in VRAM (we collapse to expected dosage before step-2).

## CI / reproducibility

- GitHub Actions: lint (ruff, rustfmt), unit tests (pytest + cargo test), smoke test (`polygwas run` on Arabidopsis demo).
- Tagged releases pin: Python version, JAX version, CUDA version, Rust toolchain.
- `envs/polygwas.yaml` pins everything.

## Performance targets (v1)

| Task | Target |
|---|---|
| Wheat Watkins (N=1035, ~10⁷ SNPs), single trait, base scan | < 4 GPU-hr |
| Same + homoeolog 2D interaction scan | < 24 GPU-hr |
| Cotton 3,724 panel, base scan | < 12 GPU-hr |
| DL prior inference, ~10M variants × 200 bp context | < 6 GPU-hr (one-shot per panel) |

If we miss these by >3×, descope path: drop hexaploid full-dosage, use diploidized expected-dosage scalar.
