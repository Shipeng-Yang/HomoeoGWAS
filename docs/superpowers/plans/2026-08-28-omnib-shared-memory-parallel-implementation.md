# omniB Shared-Memory Process Parallelism Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make canonical group omniB use real memory-safe worker processes, prove result identity and resource scaling, then rerun cotton and peanut through the unified group API.

**Architecture:** A small POSIX fork runner executes module-level block workers against read-only parent state inherited by copy-on-write. Serial and parallel execution call the same worker functions, every child limits native libraries to one thread, and execution metadata is descriptive rather than part of the statistical manifest.

**Tech Stack:** Python 3.10+, NumPy, SciPy, multiprocessing, threadpoolctl, pytest, HomoeoGWAS CLI.

**Spec:** `docs/superpowers/specs/2026-08-28-omnib-shared-memory-parallel-design.md`

## Global Constraints

- Do not change omniB components, INT, RNG streams, ACAT, hypothesis order, bootstrap-minP, FWER decisions, or checkpoint identities.
- Canonical group analysis remains pair-edge based; no direct third- or fourth-order coefficient is introduced.
- `n_jobs=1` is serial; `n_jobs>1` uses POSIX fork workers when available and otherwise records a serial fallback.
- Every worker limits BLAS/OpenMP/native libraries to one thread.
- Formal cotton and peanut runs use bootstrap `B=2000`, seed `2026`, and new output directories.
- Validate every generated config before a long run and audit every completed output.
- Never overwrite the legacy pairwise results used for equivalence comparisons.

---

### Task 1: Fork runner with observable process identity

**Files:**
- Create: `src/homoeogwas/parallel.py`
- Create: `tests/test_parallel.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Produces: `ParallelExecution` dataclass and `run_fork_blocks(blocks, worker, *, n_jobs, state_setter, state_clearer) -> tuple[list, ParallelExecution]`.
- Guarantees: ordered results, worker exception propagation, state cleanup, one native thread per child, no thread-backend fallback.

- [ ] **Step 1: Write failing tests for real process workers**

```python
def test_fork_runner_observes_multiple_worker_processes():
    results, execution = run_fork_blocks(
        list(range(16)), pid_worker, n_jobs=2,
        state_setter=set_test_state, state_clearer=clear_test_state)
    assert execution.backend == "fork_shared_memory"
    assert execution.effective_jobs == 2
    assert len(set(execution.worker_pids)) == 2
    assert [value for _, value in results] == list(range(16))

def test_serial_runner_uses_parent_and_same_worker():
    results, execution = run_fork_blocks(
        list(range(4)), pid_worker, n_jobs=1,
        state_setter=set_test_state, state_clearer=clear_test_state)
    assert execution.backend == "serial"
    assert execution.worker_pids == (os.getpid(),)

def test_worker_exception_is_raised_and_state_is_cleared():
    with pytest.raises(ParallelBlockError, match="block 3"):
        run_fork_blocks(
            list(range(8)), failing_worker, n_jobs=2,
            state_setter=set_test_state, state_clearer=clear_test_state)
    assert get_test_state() is None
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `uv run pytest -q tests/test_parallel.py`

Expected: collection fails because `homoeogwas.parallel` does not exist.

- [ ] **Step 3: Implement the minimal runner**

Implement an immutable execution record with `requested_jobs`,
`effective_jobs`, `backend`, `process_model`, `inner_threads`, `parent_pid`,
`worker_pids`, and `fallback_reason`. Bound effective jobs by CPU count and
block count. Use `multiprocessing.get_context("fork").Pool`, ordered `map`, a
module-level worker wrapper that enters `threadpool_limits(1)`, and `finally`
cleanup in the parent. Add `threadpoolctl>=3.1` as an explicit dependency.

- [ ] **Step 4: Run targeted tests and verify GREEN**

Run: `uv run pytest -q tests/test_parallel.py`

Expected: all runner tests pass and the multi-worker test observes exactly two
child PIDs.

- [ ] **Step 5: Commit the runner**

```bash
git add src/homoeogwas/parallel.py tests/test_parallel.py pyproject.toml
git commit -m "feat: add shared-memory fork block runner"
```

---

### Task 2: Move canonical omniB scoring onto the shared runner

**Files:**
- Modify: `src/homoeogwas/omnib_family.py`
- Modify: `tests/test_omnib_family.py`
- Modify: `tests/test_omnib_family_calibration.py`
- Modify: `tests/test_resampling_checkpoint.py`

**Interfaces:**
- Consumes: `run_fork_blocks` and `ParallelExecution` from Task 1.
- Produces: one shared block implementation for serial/fork scoring and a `parallel_execution` record retained on `OmniBFamilyScores`.

- [ ] **Step 1: Write the exact-equivalence regression tests**

```python
def test_group_omnib_fork_is_bit_exact_to_serial():
    serial, _ = _family_scores(("A", "B", "D"), 16, 19, n_jobs=1)
    parallel, _ = _family_scores(("A", "B", "D"), 16, 19, n_jobs=2)
    np.testing.assert_array_equal(parallel.edge_p, serial.edge_p)
    np.testing.assert_array_equal(parallel.group_p, serial.group_p)
    np.testing.assert_array_equal(
        parallel.edge_components_obs, serial.edge_components_obs)
    assert parallel.parallel_execution["backend"] == "fork_shared_memory"
    assert len(parallel.parallel_execution["worker_pids"]) == 2

def test_checkpoint_manifest_does_not_depend_on_worker_count(tmp_path):
    one = run_fixed_checkpoint_scan(tmp_path / "one", n_jobs=1)
    two = run_fixed_checkpoint_scan(tmp_path / "two", n_jobs=2)
    assert checkpoint_manifest_id(one) == checkpoint_manifest_id(two)
    assert primary_null_hash(one) == primary_null_hash(two)
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `uv run pytest -q tests/test_omnib_family.py tests/test_omnib_family_calibration.py tests/test_resampling_checkpoint.py -k 'fork or worker_count'`

Expected: assertions fail because scoring still uses joblib threads and emits no
process execution record.

- [ ] **Step 3: Extract module-level omniB workers and state**

Replace the nested threaded functions in `score_omnib_family`,
`score_omnib_subset`, and `_score_prepared_responses` with module-level workers.
Assign only immutable/read-only arrays, family objects, prepared projection
caches, response blocks, and valid indices to a module-level state immediately
before `fork`. The worker returns `(block_ordinal, edge_indices, values,
components, pid)`; the parent writes results by edge index.

- [ ] **Step 4: Use the same worker for serial and process execution**

Route every canonical block list through `run_fork_blocks`, capture the latest
execution record on `OmniBFamilyScores`, and clear state after each call. Do not
alter response construction, microblock padding, component calculations, or
ACAT reduction order.

- [ ] **Step 5: Run exact-equivalence and checkpoint tests**

Run: `uv run pytest -q tests/test_omnib_family.py tests/test_omnib_family_calibration.py tests/test_resampling_checkpoint.py`

Expected: all pass; `n_jobs=1` and `2` arrays and checkpoint hashes are exactly
equal.

- [ ] **Step 6: Commit scorer integration**

```bash
git add src/homoeogwas/omnib_family.py tests/test_omnib_family.py tests/test_omnib_family_calibration.py tests/test_resampling_checkpoint.py
git commit -m "fix: run canonical omniB scoring in fork workers"
```

---

### Task 3: Publish execution metadata and accurate CLI documentation

**Files:**
- Modify: `src/homoeogwas/interact.py`
- Modify: `tests/test_interact.py`
- Modify: `tests/test_cli.py`
- Modify: `docs/interact_inputs.md`
- Modify: `AGENTS.md`
- Modify: `CHANGELOG.md`

**Interfaces:**
- Consumes: `scores.parallel_execution` from Task 2.
- Produces: `model_diagnostics.parallel_execution`, provenance summary, and accurate CLI help.

- [ ] **Step 1: Write failing serialization and CLI tests**

```python
def test_group_result_serializes_parallel_execution():
    result = run_small_group(n_jobs=2)
    execution = result.model_diagnostics["parallel_execution"]
    assert execution["backend"] == "fork_shared_memory"
    assert execution["process_model"] == "processes"
    assert execution["inner_threads"] == 1
    assert execution["effective_jobs"] == 2

def test_interact_help_calls_n_jobs_worker_processes(capsys):
    with pytest.raises(SystemExit):
        main(["interact", "--help"])
    assert "worker processes" in capsys.readouterr().out
```

- [ ] **Step 2: Run tests and verify RED**

Run: `uv run pytest -q tests/test_interact.py tests/test_cli.py -k 'parallel_execution or worker_processes'`

Expected: metadata and help assertions fail.

- [ ] **Step 3: Add result and provenance metadata**

Insert the JSON-safe execution record into
`result.model_diagnostics["parallel_execution"]` and the top-level provenance.
Print the requested/effective process count before formal scoring. Keep the
record outside `_checkpoint_manifest`.

- [ ] **Step 4: Correct public wording**

Change canonical `--n-jobs` documentation from ambiguous “workers” to “worker
processes for canonical group omniB; legacy engines may differ”. Document
serial fallback and one native thread per process. Record the behavior-only
change in the changelog.

- [ ] **Step 5: Verify targeted and full suites**

Run: `uv run pytest -q tests/test_interact.py tests/test_cli.py tests/test_audit.py tests/test_workflow.py`

Expected: all pass, with only documented optional-dependency skips.

- [ ] **Step 6: Commit metadata and docs**

```bash
git add src/homoeogwas/interact.py tests/test_interact.py tests/test_cli.py docs/interact_inputs.md AGENTS.md CHANGELOG.md
git commit -m "docs: expose canonical omniB process execution"
```

---

### Task 4: Deterministic resource benchmark and release gate

**Files:**
- Create: `scripts/benchmark_omnib_parallel.py`
- Create: `tests/test_benchmark_omnib_parallel.py`
- Create at runtime: `results/validation/omnib_parallel_20260828/benchmark.json`

**Interfaces:**
- Produces: one JSON report with per-worker wall time, process PIDs, aggregate CPU, peak RSS, result hashes, speedups, and gate decisions.

- [ ] **Step 1: Write failing benchmark-policy tests**

```python
def test_acceptance_requires_identity_cpu_speed_and_memory():
    report = evaluate_runs(fixed_runs())
    assert report["accepted"] is True
    assert report["selected_jobs"] == 4

def test_memory_ratio_above_gate_is_rejected():
    runs = fixed_runs()
    runs[4]["peak_rss_bytes"] = int(runs[1]["peak_rss_bytes"] * 1.36)
    assert evaluate_runs(runs)["accepted"] is False
```

- [ ] **Step 2: Run policy tests and verify RED**

Run: `uv run pytest -q tests/test_benchmark_omnib_parallel.py`

Expected: collection fails because the benchmark module does not exist.

- [ ] **Step 3: Implement fixture, measurement, and policy**

Build one fixed prepared 3-subgenome fixture with enough edges and response
columns to keep 8 workers busy. Run isolated subprocesses for jobs `1`, `4`,
and `8`; collect `/usr/bin/time -v`, `/proc` CPU samples, execution metadata,
and SHA-256 identities for all result arrays. Accept four jobs only when hashes
match serial, observed CPU exceeds 200%, wall time improves, and peak RSS ratio
is at most 1.35. Select the fastest accepted worker count within the same
gates.

- [ ] **Step 4: Verify benchmark-policy tests GREEN**

Run: `uv run pytest -q tests/test_benchmark_omnib_parallel.py`

- [ ] **Step 5: Run the release benchmark**

Run:

```bash
uv run python scripts/benchmark_omnib_parallel.py \
  --jobs 1,4,8 \
  --out /mnt/7302share/fast_ysp/U7_GWAS/results/validation/omnib_parallel_20260828/benchmark.json
```

Expected: `accepted=true`, identical hashes, multiple worker PIDs, four-job
CPU over 200%, speedup over serial, and RSS ratio no more than 1.35. If any gate
fails, stop before formal species runs and return to root-cause diagnosis.

- [ ] **Step 6: Run complete verification and commit**

Run: `uv run pytest -q`

```bash
git add scripts/benchmark_omnib_parallel.py tests/test_benchmark_omnib_parallel.py
git commit -m "test: gate omniB process scaling"
```

---

### Task 5: Generate and validate canonical cotton and peanut inputs

**Files:**
- Create: `results/production/cross_species_group_omnib_v1/inputs/cotton_groups_AD.tsv`
- Create: `results/production/cross_species_group_omnib_v1/inputs/peanut_groups_AB.tsv`
- Create: `results/production/cross_species_group_omnib_v1/cotton/FibLen/configs/interact.generated.group.omnib.yaml`
- Create: `results/production/cross_species_group_omnib_v1/cotton/FibElo/configs/interact.generated.group.omnib.yaml`
- Create: `results/production/cross_species_group_omnib_v1/peanut/hundred_seed_weight/configs/interact.generated.group.omnib.yaml`
- Create: `results/production/cross_species_group_omnib_v1/peanut/seed_length/configs/interact.generated.group.omnib.yaml`

**Interfaces:**
- Consumes: frozen legacy pair tables and verified SNP-to-gene NPZ files.
- Produces: deterministic two-copy group tables and validated canonical configs.

- [ ] **Step 1: Preflight immutable inputs**

Check exact chromosome/BIM fingerprints, NPZ recorded variant counts, sample-ID
strings, phenotype overlap, pair uniqueness, and source SHA-256 values. Required
source pair hashes are:

- cotton: `352ad925002b007854b3c587154192d5415642b46c1afcd93f6379f3ea798050`;
- peanut: `dd74b1551b67ac9a40d0f75817d5c2d28f2d249185d1d9b458b3a8c69a49dd50`.

- [ ] **Step 2: Build canonical group tables**

Create `group_id` as `gene_A|gene_D` for 3,757 cotton rows and
`gene_A|gene_B` for 379 peanut rows. Preserve source row order and verify no
blank or duplicate biological group.

- [ ] **Step 3: Generate four configs**

Every config declares `mode: group`, `statistic: omniB`,
`hypothesis_unit: edge`, `subset_order: 2`, `family_scope: primary_only`,
`primary_transform: INT`, `primary_multiplicity: bootstrap_minp`,
`calibration: {method: bootstrap, B: 2000, seed: 2026}`, and a checkpoint root
inside its new output directory. Cotton traits are `FibLen` and `FibElo` with
subgenomes `[A,D]`; peanut traits are `hundred_seed_weight` and `seed_length`
with `[A,B]`.

- [ ] **Step 4: Validate all four configs**

Run `uv run homoeogwas validate -c` separately on each generated config. Expected:
four successful validations, verified NPZ/BIM binding, and no sample-ID or
chromosome mismatch.

---

### Task 6: Formal species runs, audit, and legacy equivalence

**Files:**
- Create: four new formal result directories under `results/production/cross_species_group_omnib_v1/`.
- Create: per-run audit directories and a consolidated `cross_species_summary.tsv`.

**Interfaces:**
- Consumes: selected worker count from Task 4 and validated configs from Task 5.
- Produces: formal bootstrap-minP rankings, audits, and legacy/canonical crosswalks.

- [ ] **Step 1: Run cotton FibLen and FibElo**

For each validated config run:

```bash
OMNIB_SELECTED_JOBS="$(uv run python -c 'import json; print(json.load(open("/mnt/7302share/fast_ysp/U7_GWAS/results/validation/omnib_parallel_20260828/benchmark.json"))["selected_jobs"])')"
uv run homoeogwas interact -c /mnt/7302share/fast_ysp/U7_GWAS/results/production/cross_species_group_omnib_v1/cotton/FibLen/configs/interact.generated.group.omnib.yaml --n-jobs "$OMNIB_SELECTED_JOBS"
uv run homoeogwas audit /mnt/7302share/fast_ysp/U7_GWAS/results/production/cross_species_group_omnib_v1/cotton/FibLen
uv run homoeogwas interact -c /mnt/7302share/fast_ysp/U7_GWAS/results/production/cross_species_group_omnib_v1/cotton/FibElo/configs/interact.generated.group.omnib.yaml --n-jobs "$OMNIB_SELECTED_JOBS"
uv run homoeogwas audit /mnt/7302share/fast_ysp/U7_GWAS/results/production/cross_species_group_omnib_v1/cotton/FibElo
```

Run one trait at a time so the 1 TiB machine never holds two full genotype/GRM
families concurrently.

- [ ] **Step 2: Run peanut hundred-seed-weight and seed-length**

Use the same accepted software build and worker policy:

```bash
uv run homoeogwas interact -c /mnt/7302share/fast_ysp/U7_GWAS/results/production/cross_species_group_omnib_v1/peanut/hundred_seed_weight/configs/interact.generated.group.omnib.yaml --n-jobs "$OMNIB_SELECTED_JOBS"
uv run homoeogwas audit /mnt/7302share/fast_ysp/U7_GWAS/results/production/cross_species_group_omnib_v1/peanut/hundred_seed_weight
uv run homoeogwas interact -c /mnt/7302share/fast_ysp/U7_GWAS/results/production/cross_species_group_omnib_v1/peanut/seed_length/configs/interact.generated.group.omnib.yaml --n-jobs "$OMNIB_SELECTED_JOBS"
uv run homoeogwas audit /mnt/7302share/fast_ysp/U7_GWAS/results/production/cross_species_group_omnib_v1/peanut/seed_length
```

- [ ] **Step 3: Verify two-copy legacy equivalence**

For each trait, compare the canonical group edge ranking against the legacy
pairwise INT ranking after aligning by the ordered gene pair. Require equal
hypothesis count, identical raw `p_interaction`, identical bootstrap-minP
adjusted p-values, identical significant set, and matching family ordering. A
discrepancy blocks publication reporting.

- [ ] **Step 4: Consolidate biological and audit results**

Write one summary row per trait containing cohort size, groups/edges planned,
valid/unestimable counts, bootstrap B/seed, adjusted threshold, significant
unit count, component drivers, execution backend/effective jobs, family hashes,
audit status, output path, and legacy-equivalence status.

- [ ] **Step 5: Final verification**

Run fresh checks over all four JSON results, rankings, configs, checkpoints,
audits, and the consolidated table. Confirm no output points to an old pairwise
directory and no new run uses `mode: pairwise`.
