# HomoeoGWAS v2.0.1 benchmark operator guide

This package orchestrates the frozen software benchmark. It can initialize and
run an audited **noninferential pilot**; it intentionally cannot execute the
formal benchmark.

## Prepare biological inputs

Create one JSON input specification (not YAML) with schema
`homoeogwas-v201-benchmark-inputs-v1`. All paths must be normalized absolute
paths to regular, non-symlinked files. The CLI generates every HomoeoGWAS YAML.

```json
{
  "schema": "homoeogwas-v201-benchmark-inputs-v1",
  "application_registry": "/absolute/path/to/cross_species_runs.yaml",
  "fit": {
    "cotton": {
      "subgenomes": ["A", "D"],
      "bed_prefix_template": "/absolute/cotton/{subgenome}/all",
      "phenotype": "/absolute/cotton/anchor_phenotype.tsv",
      "sample_col": "sample",
      "trait": "trait"
    },
    "wheat": {
      "subgenomes": ["A", "B", "D"],
      "bed_prefix_template": "/absolute/wheat/{subgenome}/all",
      "phenotype": "/absolute/wheat/anchor_phenotype.tsv",
      "sample_col": "sample",
      "trait": "trait"
    }
  },
  "omnib": {
    "cotton": {
      "subgenomes": ["A", "D"],
      "bed_prefixes": {"A": "/absolute/cotton/A/all", "D": "/absolute/cotton/D/all"},
      "snp_to_gene": {"A": "/absolute/cotton/A/map.npz", "D": "/absolute/cotton/D/map.npz"},
      "groups": "/absolute/cotton/groups_80.tsv",
      "phenotype": "/absolute/cotton/anchor_phenotype.tsv",
      "sample_col": "sample",
      "trait": "trait"
    }
  }
}
```

The `omnib` mapping must contain exactly these 12 context keys:

```text
cotton, wheat, quartet,
cotton:g80, cotton:g500, cotton:g2000,
wheat:g80, wheat:g500, wheat:g2000,
quartet:g80, quartet:g500, quartet:g2000
```

Each context uses the same fields shown for cotton. NPZ mappings must carry the
source BIM SHA-256 and variant count. BED/FAM sample IDs must align as strings;
groups must contain exactly the declared 80, 500, or 2,000 ordered groups.
External BED/NPZ/groups/phenotype files are hashed in place and are never copied
into the benchmark root.

## Safe command sequence

```bash
ROOT=results/benchmarks/homoeogwas_v201_method_v1
INPUTS=/absolute/path/to/homoeogwas_v201_inputs.json

uv run python -m scripts.benchmarks.v201.cli init \
  --root "$ROOT" --input-spec "$INPUTS"

uv run python -m scripts.benchmarks.v201.cli pilot \
  --root "$ROOT" --all --validate-only

uv run python -m scripts.benchmarks.v201.cli pilot \
  --root "$ROOT" --all --n-jobs 8

uv run python -m scripts.benchmarks.v201.cli aggregate \
  --root "$ROOT" --stage pilot

uv run python -m scripts.benchmarks.v201.cli audit \
  --root "$ROOT" --stage pilot

uv run python -m scripts.benchmarks.v201.cli project-formal \
  --root "$ROOT" --effective-workers 32
```

`init` accepts only an absent or empty, non-symlinked root. It generates both
canonical registries, all scenario-bound pilot/formal configs, panel-specific
comparator preflights, real omniB context artifacts, and one LOCO
phenotype/config/truth artifact per replicate. It runs strict real-context
validation and `homoeogwas validate` on every generated YAML before writing
`design_lock.json`. Any failure removes only the CLI-created partial root and
leaves external inputs untouched.

`pilot --validate-only` repeats all strict validation without creating shards or
running statistics. Ordinary pilot execution resumes immutable matching shards
and never overwrites an existing shard. Every pilot calibration uses `B=199`,
`qa_only=true`, and the label `noninferential_do_not_threshold`.

Pilot QA results are not publication thresholds, formal rejections, or formal
benchmark evidence. `project-formal` uses measured pilot CPU time and output
bytes and exits nonzero if CPU, elapsed-time, or storage caps are exceeded. A
formal execution command can be added only in a new plan after the pilot audit,
resource projection, and explicit user approval; no such command exists here.
