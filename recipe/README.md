# conda recipe

CPU-core conda package for HomoeoGWAS. GPU extras (torch / transformers /
PlantCaduceus / mamba-ssm) are intentionally **excluded** — install them with
`pip install "homoeogwas[gpu]"` into a CUDA-enabled env (see
`docs/getting_started.md`).

## Local build from a checkout

```bash
conda install -n base conda-build      # if not present
conda build recipe/                     # uses source: path: ..
```

## Submitting to bioconda

1. Cut a GitHub release tag `v<version>` so the tarball URL is stable.
2. In `meta.yaml`, switch the `source:` block from `path: ..` to the
   `url:` + `sha256:` form (sha256 of the release tarball).
3. Open a PR against https://github.com/bioconda/bioconda-recipes adding
   `recipes/homoeogwas/meta.yaml`. CI lints + builds; a maintainer merges.

bioconda is the natural channel (genomics deps `pysam`, `bed-reader` already
live there / on conda-forge). conda-forge would also work for the pure-Python
core but bioconda keeps it discoverable alongside the baseline tools.
