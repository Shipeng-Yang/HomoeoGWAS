# Cross-Species Run Registry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a species-independent, resume-safe registry that dispatches canonical HomoeoGWAS GWAS/interaction workflows and indexes current and historical multi-species runs.

**Architecture:** A focused `run_registry.py` module owns schema parsing, canonical identity, atomic state and dispatch to the existing `workflow.py` entry points. The CLI adds nested `registry validate|run` commands; association mathematics remains unchanged.

**Tech Stack:** Python 3.10+, dataclasses, PyYAML, hashlib, pathlib, pandas, pytest.

**Spec:** `docs/superpowers/specs/2026-08-28-cross-species-run-registry-design.md`

## Global Constraints

- Canonical interaction is `mode: group`, `statistic: omniB`, `primary_transform: INT`, `primary_multiplicity: bootstrap_minp`, `subset_order: 2`.
- Species names and ploidy are metadata; execution code must contain no species-name branches.
- Generate configs below `<out_dir>/configs/` and validate before long runs.
- Four-copy groups aggregate six pair edges and never fit a direct four-way coefficient.
- Existing historical results are read-only fixtures and must not be relabelled as canonical.
- Existing dirty main-worktree changes are not modified; implementation occurs on `codex/rapeseed-generic-provenance` in `/tmp/U7_GWAS-rapeseed-v1`.

---

### Task 1: Registry schema and relative-path resolution

**Files:**
- Create: `src/homoeogwas/run_registry.py`
- Create: `tests/test_run_registry.py`

**Interfaces:**
- Produces: `RegistryRun`, `RunRegistry`, `load_registry(path: str | Path) -> RunRegistry`
- Produces: `validate_registry(registry: RunRegistry) -> None`
- Produces: `RegistryError(ValueError)` with breeder-readable messages

- [ ] **Step 1: Write failing schema tests**

```python
def test_load_registry_resolves_paths_and_keeps_species_as_metadata(tmp_path):
    path = write_registry(tmp_path, runs=[interaction_run("r1", ["A", "C"])])
    registry = load_registry(path)
    assert registry.runs[0].species == "Brassica napus"
    assert registry.runs[0].groups == (tmp_path / "groups.tsv").resolve()
    assert registry.runs[0].subgenomes == ("A", "C")

@pytest.mark.parametrize("subgenomes", [("A",), ("A", "A")])
def test_interaction_requires_two_unique_subgenomes(tmp_path, subgenomes):
    path = write_registry(tmp_path, runs=[interaction_run("bad", subgenomes)])
    with pytest.raises(RegistryError, match="at least two unique"):
        load_registry(path)

def test_registry_rejects_duplicate_ids(tmp_path):
    run = interaction_run("same", ["A", "C"])
    with pytest.raises(RegistryError, match="duplicate run id"):
        load_registry(write_registry(tmp_path, runs=[run, run]))
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `uv run pytest tests/test_run_registry.py -q`  
Expected: collection fails because `homoeogwas.run_registry` does not exist.

- [ ] **Step 3: Implement immutable schema objects and parser**

Implement frozen dataclasses with explicit fields:

```python
@dataclass(frozen=True)
class RegistryRun:
    id: str
    kind: str
    species: str
    panel: str
    subgenomes: tuple[str, ...]
    out_dir: Path | None
    phenotype: Path | None = None
    sample_col: str | None = None
    trait: str | None = None
    bed_prefixes: Mapping[str, Path] = field(default_factory=dict)
    snp_to_gene: Mapping[str, Path] = field(default_factory=dict)
    groups: Path | None = None
    hypothesis_unit: str = "group"
    family_scope: str = "primary_only"
    bootstrap_B: int = 2000
    n_jobs: int = 8
    result_root: Path | None = None
    analysis_shape: str | None = None

@dataclass(frozen=True)
class RunRegistry:
    registry_version: int
    name: str
    index_dir: Path
    source_path: Path
    runs: tuple[RegistryRun, ...]
```

Resolve path fields against `source_path.parent`. Reject unsupported keys rather than silently
ignoring misspellings. Validate the exact conditions in spec sections 3-4.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run: `uv run pytest tests/test_run_registry.py -q`  
Expected: all Task 1 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/homoeogwas/run_registry.py tests/test_run_registry.py
git commit -m "feat: add cross-species registry schema"
```

### Task 2: Canonical identity and atomic run state

**Files:**
- Modify: `src/homoeogwas/run_registry.py`
- Modify: `tests/test_run_registry.py`

**Interfaces:**
- Consumes: `RegistryRun`
- Produces: `canonical_run_identity(run: RegistryRun) -> dict`
- Produces: `run_identity_sha256(run: RegistryRun) -> str`
- Produces: `load_run_state(out_dir: Path) -> dict | None`
- Produces: `write_run_state(out_dir: Path, payload: Mapping) -> Path`
- Produces: `resume_decision(run, existing_state, *, resume: bool) -> str`

- [ ] **Step 1: Write failing identity/state tests**

```python
def test_identity_is_order_stable_and_changes_with_bim(tmp_path):
    run = materialized_interaction_run(tmp_path)
    first = run_identity_sha256(run)
    second = run_identity_sha256(replace(run, bed_prefixes=dict(reversed(run.bed_prefixes.items()))))
    assert first == second
    Path(str(run.bed_prefixes["A"]) + ".bim").write_text("1\trs1\t0\t2\tA\tG\n")
    assert run_identity_sha256(run) != first

def test_resume_skips_only_matching_complete_identity(tmp_path):
    run = materialized_interaction_run(tmp_path)
    digest = run_identity_sha256(run)
    assert resume_decision(run, {"status": "COMPLETE", "identity": digest}, resume=True) == "SKIP"
    with pytest.raises(RegistryError, match="identity mismatch"):
        resume_decision(run, {"status": "COMPLETE", "identity": "0" * 64}, resume=True)
```

- [ ] **Step 2: Verify RED**

Run: `uv run pytest tests/test_run_registry.py -q`  
Expected: identity/state functions are missing.

- [ ] **Step 3: Implement canonical JSON identities and atomic state**

Hash small text files and BIM/FAM content with streaming SHA-256. Record BED path/size but do not
hash BED bytes. Serialize mappings with sorted keys and JSON separators `(',', ':')`. Write state
through a sibling temporary file followed by `Path.replace()`.

- [ ] **Step 4: Verify GREEN**

Run: `uv run pytest tests/test_run_registry.py -q`  
Expected: schema and identity/state tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/homoeogwas/run_registry.py tests/test_run_registry.py
git commit -m "feat: bind registry runs to immutable identities"
```

### Task 3: Workflow dispatch, resume and cross-run indexes

**Files:**
- Modify: `src/homoeogwas/run_registry.py`
- Modify: `tests/test_run_registry.py`

**Interfaces:**
- Produces: `execute_registry(path, *, only=(), resume=True, fail_fast=False, dry_run=False, gwas_runner=workflow.run_gwas, interaction_runner=workflow.run_interaction) -> dict`
- Produces: `write_registry_indexes(registry, records) -> Mapping[str, str]`

- [ ] **Step 1: Write failing dispatch tests with injected runners**

```python
def test_interaction_dispatch_forces_canonical_group_omnib(tmp_path):
    path = write_registry(tmp_path, runs=[interaction_run("r1", ["A", "B", "D"])])
    calls = []
    result = execute_registry(path, dry_run=True,
        interaction_runner=lambda **kwargs: calls.append(kwargs) or {"ok": True, "summary": {}})
    assert result["ok"]
    assert calls[0]["statistic"] == "omniB"
    assert calls[0]["groups"].endswith("groups.tsv")
    assert calls[0]["hypothesis_unit"] == "group"
    assert calls[0]["subset_order"] == 2

def test_four_copy_dispatch_has_no_four_way_option(tmp_path):
    path = write_registry(tmp_path, runs=[interaction_run("q", ["A", "B", "C", "D"])])
    calls = []
    execute_registry(path, dry_run=True,
        interaction_runner=lambda **kwargs: calls.append(kwargs) or {"ok": True, "summary": {}})
    assert set(calls[0]) >= {"groups", "family_scope", "subset_order"}
    assert not any("four" in key or "4way" in key for key in calls[0])

def test_historical_entry_is_indexed_without_dispatch(tmp_path):
    path = write_registry(tmp_path, runs=[historical_run("old")])
    called = False
    result = execute_registry(path,
        interaction_runner=lambda **kwargs: (_ for _ in ()).throw(AssertionError("dispatched")))
    assert result["runs"][0]["status"] == "HISTORICAL"
```

- [ ] **Step 2: Verify RED**

Run: `uv run pytest tests/test_run_registry.py -q`  
Expected: `execute_registry` and index writer are missing.

- [ ] **Step 3: Implement dispatch and state transitions**

Call the existing workflow functions with registry fields only. Write `PLANNED`, then `RUNNING`,
then `COMPLETE` or `FAILED`. A matching complete run returns `SKIPPED_COMPLETE`. Continue after an
independent failure unless `fail_fast=True`.

- [ ] **Step 4: Implement JSON/TSV/Markdown indexes**

Use a fixed column order and `pandas.DataFrame.to_csv(sep='\t')`. Markdown rows must include run ID,
species, trait, subgenomes, status, significant count, audit status and next action. Write resolved
registry YAML into the index directory.

- [ ] **Step 5: Verify GREEN**

Run: `uv run pytest tests/test_run_registry.py -q`  
Expected: all registry tests pass and index files exist in temporary directories.

- [ ] **Step 6: Commit**

```bash
git add src/homoeogwas/run_registry.py tests/test_run_registry.py
git commit -m "feat: execute and resume registered workflows"
```

### Task 4: CLI, documentation and migration inventory

**Files:**
- Modify: `src/homoeogwas/cli.py`
- Modify: `tests/test_run_registry.py`
- Create: `analyses/cross_species_interaction_inventory.yaml`
- Create: `docs/run_registry.md`
- Modify: `README.md`

**Interfaces:**
- Produces CLI: `homoeogwas registry validate -c PATH`
- Produces CLI: `homoeogwas registry run -c PATH [--only ID ...] [--resume|--no-resume] [--fail-fast] [--dry-run]`

- [ ] **Step 1: Write failing parser/CLI tests**

```python
def test_registry_cli_validate_and_dry_run(tmp_path, capsys):
    path = write_registry(tmp_path, runs=[interaction_run("r1", ["A", "C"])])
    assert main(["registry", "validate", "-c", str(path)]) == 0
    assert "registry schema OK" in capsys.readouterr().out
    assert main(["registry", "run", "-c", str(path), "--dry-run"]) == 0
```

- [ ] **Step 2: Verify RED**

Run: `uv run pytest tests/test_run_registry.py::test_registry_cli_validate_and_dry_run -q`  
Expected: parser rejects `registry`.

- [ ] **Step 3: Add nested CLI parsers and handlers**

Add `add_registry_subparser(sub)` and `cmd_registry(args)` in `run_registry.py`; `cli.py` only wires
them into `build_parser()` and `main()`.

- [ ] **Step 4: Add historical migration inventory**

Record rapeseed as canonical and wheat Route-B, cotton FibLen/FibElo and the two peanut traits as
`kind: historical` with their existing result roots and explicit `analysis_shape`. The file is an
inventory; execution tests use temporary fixtures and do not require project data.

- [ ] **Step 5: Document breeder usage and failure repair**

Document generated configs, resume behavior, identity mismatch repair (use a new output root),
historical semantics and the fact that no species-specific statistic exists.

- [ ] **Step 6: Run focused and full verification**

Run: `uv run pytest tests/test_run_registry.py tests/test_workflow.py -q`  
Expected: all focused tests pass.

Run: `uv run ruff check src/homoeogwas/run_registry.py src/homoeogwas/cli.py tests/test_run_registry.py`  
Expected: no lint errors.

Run: `uv run pytest -q`  
Expected: full suite passes with only environment-dependent skips.

- [ ] **Step 7: Commit**

```bash
git add src/homoeogwas/cli.py src/homoeogwas/run_registry.py tests/test_run_registry.py analyses/cross_species_interaction_inventory.yaml docs/run_registry.md README.md
git commit -m "docs: expose the cross-species registry workflow"
```

