# Triad3 D3 Confirmatory Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a phenotype-sealed D3 runner, independent auditor and QA
workflow that evaluate four calibration cells for classical F and HC3 using
2,000 outer replicates and true outer-level parallelism.

**Architecture:** New D3-only modules isolate the immutable statistical
contract, four-cell computation/scheduling, independent audit and compute
benchmark. The existing manifest-bound D1 runner is never edited. Formal
preparation ends at an all-PASS manifest ledger; consuming the formal stream
requires a later explicit authorization.

**Tech Stack:** Python 3.12, NumPy, SciPy, pandas, joblib/concurrent futures,
PyYAML, pytest, ruff, existing HomoeoGWAS LMM and triad3 statistical helpers.

## Global Constraints

- Follow
  `docs/superpowers/specs/2026-07-30-triad3-d3-confirmatory-design.md`
  exactly.
- Do not modify `scripts/triad3_d1_calibration.py` or any frozen D1 artifact.
- Do not read, hash or inspect the real phenotype trait column.
- Do not call Claude.
- Family order is the frozen 24-triad D1 order.
- Cells are `raw_oracle_balanced`, `int_oracle_balanced`,
  `int_reml_balanced`, and `int_reml_low_h2`.
- Methods are `classical_f` and `hc3_t`; method identity never enters an RNG
  namespace.
- Outer B is 2,000; inner B is 999; decision zones are acceptable at
  X <= 120, inconclusive at 121--125, and unacceptable at X >= 126.
- Every worker uses BLAS threads=1.
- Formal execution is forbidden until a new explicit user authorization.
- Preserve the existing dirty worktree. Stage and commit only files named by
  the active task.

---

## File map

- Create `scripts/triad3_d3_contract.py`: strict config, constants, exact
  decisions, canonical serialization, RNG namespaces, checkpoint schema and
  atomic I/O.
- Create `scripts/triad3_d3_confirmation.py`: outcome-free input preparation,
  manifest, four-cell computation, outer-bundle scheduler and CLI.
- Create `scripts/triad3_d3_independent_audit.py`: raw-checkpoint rehash,
  identity/pairing verification and independent statistical decisions.
- Create `scripts/triad3_d3_compute_benchmark.py`: fixed-QA worker scaling and
  deterministic worker-selection rule.
- Create `tests/test_triad3_d3_contract.py`: contract/RNG/checkpoint tests.
- Create `tests/test_triad3_d3_confirmation.py`: cell formula, scheduler,
  sealing and resume tests.
- Create `tests/test_triad3_d3_independent_audit.py`: independent-audit
  adversarial fixtures.
- Create `tests/test_triad3_d3_compute_benchmark.py`: worker selection and
  benchmark identity tests.
- Create `results/experimental/TRIAD3_STAGE1C_D3_IMPLEMENTATION_AUDIT.md` only
  after implementation verification.
- Create a new D3 QA output directory during Task 9.
- Create formal config/manifest/lock artifacts only during Task 10; do not
  create checkpoints or consume the formal stream.

---

### Task 1: Immutable D3 statistical contract

**Files:**
- Create: `scripts/triad3_d3_contract.py`
- Create: `tests/test_triad3_d3_contract.py`

**Interfaces:**
- Produces:
  - `CELLS: tuple[str, ...]`
  - `METHODS: tuple[str, ...]`
  - `D3Decision` dataclass
  - `validate_d3_config(value: dict) -> dict`
  - `decision_zone(rejections: int) -> str`
  - `canonical_json_bytes(value: object) -> bytes`
  - `derive_seed_integer(master_seed_hex: str, domain: str, *labels) -> int`
  - `rng_for(master_seed_hex: str, domain: str, *labels) -> np.random.Generator`
- Consumes: no D3 module.

- [ ] **Step 1: Write failing config and decision tests**

```python
def test_d3_decision_boundaries_are_exact():
    assert D3.decision_zone(120) == "acceptable"
    assert D3.decision_zone(121) == "inconclusive"
    assert D3.decision_zone(125) == "inconclusive"
    assert D3.decision_zone(126) == "unacceptable"


def test_config_requires_exact_protocol():
    config = valid_config()
    config["protocol"]["outer_B"] = 1999
    with pytest.raises(ValueError, match="outer_B must equal 2000"):
        D3.validate_d3_config(config)
```

- [ ] **Step 2: Run the tests and verify RED**

Run:
`uv run pytest -q tests/test_triad3_d3_contract.py`

Expected: FAIL because `triad3_d3_contract.py` does not exist.

- [ ] **Step 3: Implement the exact constants and strict validator**

```python
CELLS = (
    "raw_oracle_balanced",
    "int_oracle_balanced",
    "int_reml_balanced",
    "int_reml_low_h2",
)
METHODS = ("classical_f", "hc3_t")
OUTER_B = 2000
INNER_B = 999
ACCEPTABLE_MAX = 120
UNACCEPTABLE_MIN = 126


def decision_zone(rejections: int) -> str:
    if rejections <= ACCEPTABLE_MAX:
        return "acceptable"
    if rejections >= UNACCEPTABLE_MIN:
        return "unacceptable"
    return "inconclusive"
```

The validator must reject unknown keys, booleans used as integers, a changed
cell/method order, any B/threshold mismatch, a non-INT production transform,
or a QA/formal seed-domain mismatch.

- [ ] **Step 4: Add deterministic seed tests**

```python
def test_methods_do_not_enter_rng_identity():
    a = D3.rng_for(SEED_HEX, "formal", "d3", "outer", 7).standard_normal(8)
    b = D3.rng_for(SEED_HEX, "formal", "d3", "outer", 7).standard_normal(8)
    np.testing.assert_array_equal(a, b)


def test_qa_and_formal_domains_are_disjoint():
    assert D3.derive_seed_integer(SEED_HEX, "qa", "outer", 0) != (
        D3.derive_seed_integer(SEED_HEX, "formal", "outer", 0)
    )
```

- [ ] **Step 5: Run focused tests and lint**

Run:

```bash
uv run pytest -q tests/test_triad3_d3_contract.py
uv run ruff check scripts/triad3_d3_contract.py tests/test_triad3_d3_contract.py
uv run python -m py_compile scripts/triad3_d3_contract.py
```

Expected: all exit 0.

- [ ] **Step 6: Commit only the contract task**

```bash
git add scripts/triad3_d3_contract.py tests/test_triad3_d3_contract.py
git commit -m "feat: add triad3 D3 statistical contract"
```

---

### Task 2: Atomic checkpoint and resume contract

**Files:**
- Modify: `scripts/triad3_d3_contract.py`
- Modify: `tests/test_triad3_d3_contract.py`

**Interfaces:**
- Produces:
  - `checkpoint_path(out_dir: Path, cell: str, outer_index: int) -> Path`
  - `validate_checkpoint(record: dict, *, manifest_id: str, cell: str,
    outer_index: int) -> dict`
  - `atomic_write_json(path: Path, value: object) -> None`
  - `load_checkpoint(path: Path, **identity) -> dict`
- Consumes: Task 1 canonical serialization and exact constants.

- [ ] **Step 1: Write failing checkpoint identity tests**

```python
def test_checkpoint_rejects_wrong_identity(valid_checkpoint):
    changed = copy.deepcopy(valid_checkpoint)
    changed["outer_index"] = 8
    with pytest.raises(ValueError, match="outer_index"):
        D3.validate_checkpoint(
            changed,
            manifest_id=MANIFEST_ID,
            cell="raw_oracle_balanced",
            outer_index=7,
        )


def test_atomic_write_leaves_no_temporary_file(tmp_path):
    path = tmp_path / "checkpoint.json"
    D3.atomic_write_json(path, {"status": "success"})
    assert path.exists()
    assert not path.with_name(path.name + ".tmp").exists()
```

- [ ] **Step 2: Run the two tests and verify RED**

Run:
`uv run pytest -q tests/test_triad3_d3_contract.py -k 'checkpoint or atomic'`

Expected: FAIL because the checkpoint functions are absent.

- [ ] **Step 3: Implement schema validation and atomic persistence**

The validator must require exact top-level keys, all method records, explicit
typed `not_applicable` values for oracle REML fields, finite successful
statistics, plus-one consistency
`p == (1 + comparison_count) / 1000`, and exact rejection
`p <= 0.05`. Failure checkpoints must retain stream hashes and structured
error type/message.

- [ ] **Step 4: Add corrupted/truncated/unexpected-path tests**

```python
def test_load_checkpoint_rejects_truncated_json(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"status":')
    with pytest.raises(ValueError, match="invalid checkpoint JSON"):
        D3.load_checkpoint(
            path,
            manifest_id=MANIFEST_ID,
            cell="raw_oracle_balanced",
            outer_index=0,
        )
```

- [ ] **Step 5: Run contract tests and commit**

```bash
uv run pytest -q tests/test_triad3_d3_contract.py
uv run ruff check scripts/triad3_d3_contract.py tests/test_triad3_d3_contract.py
git add scripts/triad3_d3_contract.py tests/test_triad3_d3_contract.py
git commit -m "feat: add D3 atomic checkpoint contract"
```

---

### Task 3: Four-cell statistical engine

**Files:**
- Create: `scripts/triad3_d3_confirmation.py`
- Create: `tests/test_triad3_d3_confirmation.py`

**Interfaces:**
- Produces:
  - `PreparedD3Inputs` dataclass
  - `prepare_inputs(config: dict) -> PreparedD3Inputs`
  - `run_outer_bundle(*, outer_index: int, requested_cells: tuple[str, ...],
    manifest_id: str, master_seed_hex: str, inputs: PreparedD3Inputs) ->
    dict[str, dict]`
- Consumes:
  - Task 1 RNG/contract functions.
  - `triad3_full_family_minp.prepare_test`
  - `triad3_refit_d0_diagnostic.family_minp_with_masks`
  - `homoeogwas.interact.rank_int`
  - `homoeogwas.lmm.fit_multi_reml`

- [ ] **Step 1: Write a failing raw/INT pairing test**

```python
def test_bundle_pairs_raw_and_int_oracle_responses(prepared_tiny):
    result = D3RUN.run_outer_bundle(
        outer_index=0,
        requested_cells=(
            "raw_oracle_balanced",
            "int_oracle_balanced",
        ),
        manifest_id=MANIFEST_ID,
        master_seed_hex=SEED_HEX,
        inputs=prepared_tiny,
    )
    raw = result["raw_oracle_balanced"]
    transformed = result["int_oracle_balanced"]
    assert raw["shared"]["outer_base_normal_sha256"] == (
        transformed["shared"]["outer_base_normal_sha256"]
    )
    assert raw["shared"]["raw_response_sha256"] == (
        transformed["shared"]["raw_response_sha256"]
    )
    assert raw["shared"]["analyzed_response_sha256"] != (
        transformed["shared"]["analyzed_response_sha256"]
    )
```

- [ ] **Step 2: Run and verify RED**

Run:
`uv run pytest -q tests/test_triad3_d3_confirmation.py`

Expected: FAIL because the runner module is absent.

- [ ] **Step 3: Implement outcome-free preparation**

Follow D1 input preparation without importing the D1 runner:

- read only interact config, geometry, family table, BED/mapping metadata,
  kernels and sample IDs;
- rebuild the exact ordered 24 designs;
- construct balanced and low-h2 covariance/root arrays;
- construct the balanced oracle whitener and 24 prepared tests;
- expose phenotype path/sample column metadata without reading the trait.

- [ ] **Step 4: Implement raw and INT oracle cells**

For one outer index:

```python
outer_z = rng_for(seed, "formal", "d3", "outer", index).standard_normal(n)
raw_balanced = balanced_root @ outer_z
raw_analyzed = raw_balanced
int_analyzed = rank_int(raw_balanced)
```

Generate one balanced inner base-normal matrix, then create cell-specific
intercepts and analyzed-scale responses. Scan observed plus 999 inner columns
with both methods and record comparison counts, empirical P and hashes.

- [ ] **Step 5: Implement balanced and low-h2 REML cells**

Fit the intercept-only multi-kernel null on the rank-INT response using the
profile-specific REML namespace. Reject optimizer non-success. Retain boundary
fits, reconstruct fitted covariance/whitener, prepare the 24 tests, and scan
the profile-specific fitted-null inner draws.

- [ ] **Step 6: Add formula and failure regression tests**

Tests must assert:

- raw oracle observed/inner covariance paths match the known covariance;
- raw and INT oracle share outer/inner base hashes;
- balanced oracle and balanced REML share base streams;
- low-h2 shares outer but not inner namespace;
- method records share identical response/inner hashes;
- optimizer non-success returns a terminal failure;
- degeneracy returns a terminal failure;
- boundary components remain successful descriptive fields.

- [ ] **Step 7: Run focused regression and commit**

```bash
uv run pytest -q tests/test_triad3_d3_confirmation.py
uv run pytest -q tests/test_triad3_d1_calibration.py
uv run ruff check scripts/triad3_d3_confirmation.py tests/test_triad3_d3_confirmation.py
git add scripts/triad3_d3_confirmation.py tests/test_triad3_d3_confirmation.py
git commit -m "feat: add D3 four-cell calibration engine"
```

---

### Task 4: True-concurrency scheduler and resume

**Files:**
- Modify: `scripts/triad3_d3_confirmation.py`
- Modify: `tests/test_triad3_d3_confirmation.py`

**Interfaces:**
- Produces:
  - `SchedulerMetrics` dataclass
  - `execute_pending(*, pending: dict[int, tuple[str, ...]],
    compute_bundle: Callable, outer_workers: int, max_inflight: int,
    persist: Callable) -> SchedulerMetrics`
- Consumes: Task 2 checkpoint functions and Task 3 bundle engine.

- [ ] **Step 1: Write the scheduler regression that fails under D1 batching**

```python
def test_scheduler_reaches_multiple_active_tasks():
    tracker = ActiveTracker()
    pending = {i: ("raw_oracle_balanced",) for i in range(16)}
    metrics = D3RUN.execute_pending(
        pending=pending,
        compute_bundle=tracker.compute,
        outer_workers=8,
        max_inflight=16,
        persist=lambda *_: None,
    )
    assert metrics.max_active >= 4
    assert metrics.submitted == 16
    assert metrics.completed == 16
```

- [ ] **Step 2: Run and verify RED**

Run:
`uv run pytest -q tests/test_triad3_d3_confirmation.py -k scheduler`

Expected: FAIL because `execute_pending` is absent.

- [ ] **Step 3: Implement completion-order scheduling**

Use `ThreadPoolExecutor(max_workers=outer_workers)` and `as_completed`.
Maintain at most `max_inflight` futures, submit a replacement immediately when
one finishes, and call `persist(cell, checkpoint)` for each returned cell.
Never use checkpoint frequency as a scheduling batch size.

- [ ] **Step 4: Add interruption/resume and worker-identity tests**

Create a six-bundle QA fixture. Interrupt after two completed bundles, reload
accepted checkpoints, resume with another worker count, and require the final
checkpoint bytes and TSV bytes to match a clean one-worker run.

- [ ] **Step 5: Run scheduler/resume tests and commit**

```bash
uv run pytest -q tests/test_triad3_d3_confirmation.py -k 'scheduler or resume or worker'
uv run ruff check scripts/triad3_d3_confirmation.py tests/test_triad3_d3_confirmation.py
git add scripts/triad3_d3_confirmation.py tests/test_triad3_d3_confirmation.py
git commit -m "feat: add truly parallel D3 scheduler"
```

---

### Task 5: Immutable manifest and sealed CLI

**Files:**
- Modify: `scripts/triad3_d3_confirmation.py`
- Modify: `tests/test_triad3_d3_confirmation.py`

**Interfaces:**
- Produces:
  - `build_manifest_body(config_path: Path, config: dict,
    inputs: PreparedD3Inputs) -> dict`
  - `write_or_validate_manifest(out_dir: Path, body: dict) -> str`
  - CLI modes `--validate-only`, `--prepare-only`, and formal execution.
- Consumes: Tasks 1--4.

- [ ] **Step 1: Write failing manifest and phenotype-seal tests**

```python
def test_prepare_only_does_not_consume_rng_or_trait(monkeypatch, tmp_path):
    monkeypatch.setattr(D3RUN, "rng_for", forbidden)
    monkeypatch.setattr(pd, "read_csv", guarded_read_csv_without_trait)
    result = D3RUN.prepare_only(CONFIG_PATH, tmp_path)
    assert result["random_stream_consumed"] is False
    assert result["real_trait_values_loaded"] is False
```

- [ ] **Step 2: Run and verify RED**

Run:
`uv run pytest -q tests/test_triad3_d3_confirmation.py -k 'manifest or prepare_only or trait'`

Expected: FAIL because manifest/CLI preparation is absent.

- [ ] **Step 3: Implement non-circular manifest binding**

Bind the approved spec, lock record, D0/D1/D2 decisions, config, D3 code,
transitive statistical code, git state, environment, ordered family/design
hashes, kernels, sample IDs, covariances, RNG domains and phenotype-access
metadata. Compute the manifest ID from a body that does not contain its own
ID.

- [ ] **Step 4: Implement fail-closed CLI modes**

`--validate-only` performs schema/path checks without input preparation or RNG.
`--prepare-only` rebuilds inputs and writes/validates the manifest without
calling RNG. Formal execution requires the exact existing manifest and rejects
unexpected checkpoints.

The completion JSON contains only identity counts, failure counts, scheduling
metrics, resource use and `no_inferential_summary=true`.

- [ ] **Step 5: Run seal/manifest tests and commit**

```bash
uv run pytest -q tests/test_triad3_d3_confirmation.py
uv run ruff check scripts/triad3_d3_confirmation.py tests/test_triad3_d3_confirmation.py
git add scripts/triad3_d3_confirmation.py tests/test_triad3_d3_confirmation.py
git commit -m "feat: seal D3 manifest and CLI"
```

---

### Task 6: Independent raw-checkpoint auditor

**Files:**
- Create: `scripts/triad3_d3_independent_audit.py`
- Create: `tests/test_triad3_d3_independent_audit.py`

**Interfaces:**
- Produces:
  - `audit_d3(out_dir: Path, manifest_path: Path) -> dict`
  - CLI `--out-dir`, `--manifest`, `--output`
- Consumes only checkpoint JSON, replicate TSV, manifest and SciPy; it must not
  import runner aggregation or runner summary functions.

- [ ] **Step 1: Write adversarial audit fixtures**

```python
@pytest.mark.parametrize(
    "mutation, message",
    [
        ("missing_identity", "missing"),
        ("duplicate_identity", "duplicate"),
        ("wrong_pair_hash", "pairing"),
        ("wrong_comparison_count", "plus-one"),
        ("unexpected_checkpoint", "unexpected"),
    ],
)
def test_audit_fails_closed(tmp_path, mutation, message):
    fixture = build_complete_fixture(tmp_path)
    mutate_fixture(fixture, mutation)
    with pytest.raises(ValueError, match=message):
        AUDIT.audit_d3(fixture.out_dir, fixture.manifest)
```

- [ ] **Step 2: Run and verify RED**

Run:
`uv run pytest -q tests/test_triad3_d3_independent_audit.py`

Expected: FAIL because the auditor is absent.

- [ ] **Step 3: Implement independent rehash and identity audit**

Walk all 8,000 expected cell/index keys; reject missing, duplicate, extra,
nonterminal or schema-inconsistent records. Rehash manifest-bound files and
arrays. Verify shared outer hashes across all cells, shared balanced inner
hashes across three cells, separate low-h2 inner hashes and method-independent
data identity.

- [ ] **Step 4: Implement independent decisions**

From raw comparison counts compute `(1 + count) / 1000`, reject at P <= 0.05,
sum eight counts, apply 120/126 zones and the candidate-specific four-gate
rule. Compute exact Clopper--Pearson intervals and endpoint binomial
probabilities directly with SciPy.

- [ ] **Step 5: Add TSV cross-check and no-summary-import guard**

Require exactly 16,000 ordered method rows. Add a source-level test asserting
the auditor does not import `triad3_d3_confirmation` or read a runner
inferential summary.

- [ ] **Step 6: Run tests and commit**

```bash
uv run pytest -q tests/test_triad3_d3_independent_audit.py
uv run ruff check scripts/triad3_d3_independent_audit.py tests/test_triad3_d3_independent_audit.py
git add scripts/triad3_d3_independent_audit.py tests/test_triad3_d3_independent_audit.py
git commit -m "feat: add independent D3 checkpoint audit"
```

---

### Task 7: Deterministic compute benchmark and worker selection

**Files:**
- Create: `scripts/triad3_d3_compute_benchmark.py`
- Create: `tests/test_triad3_d3_compute_benchmark.py`

**Interfaces:**
- Produces:
  - `select_workers(results: pd.DataFrame) -> int`
  - benchmark CLI for worker values 32, 64, 96, two repeats and 192 QA bundles.
- Consumes: Task 3 engine, Task 4 scheduler and QA RNG domain.

- [ ] **Step 1: Write failing worker-selection tests**

```python
def test_selects_smallest_worker_count_within_ten_percent_of_fastest():
    frame = pd.DataFrame(
        {
            "workers": [32, 32, 64, 64, 96, 96],
            "wall_seconds": [120, 118, 70, 72, 66, 68],
            "status": ["PASS"] * 6,
            "bytes_identical": [True] * 6,
        }
    )
    assert BENCH.select_workers(frame) == 64
```

- [ ] **Step 2: Run and verify RED**

Run:
`uv run pytest -q tests/test_triad3_d3_compute_benchmark.py`

Expected: FAIL because the benchmark module is absent.

- [ ] **Step 3: Implement benchmark validation and selection**

Reject a worker count if either repeat fails, checkpoint bytes differ, maximum
active concurrency is <=1, or resource records are missing. Compute median
wall time per worker count and choose the smallest count no slower than
`1.10 * fastest_median`.

- [ ] **Step 4: Implement aggregate byte ledger**

For each repeat hash sorted `(relative_checkpoint_path, checkpoint_bytes)`
pairs. Require the same aggregate digest for 1/32/64/96 byte-identity QA and
for both benchmark repeats.

- [ ] **Step 5: Run benchmark-unit tests and commit**

```bash
uv run pytest -q tests/test_triad3_d3_compute_benchmark.py
uv run ruff check scripts/triad3_d3_compute_benchmark.py tests/test_triad3_d3_compute_benchmark.py
git add scripts/triad3_d3_compute_benchmark.py tests/test_triad3_d3_compute_benchmark.py
git commit -m "feat: benchmark D3 outer concurrency"
```

---

### Task 8: Integrated QA, resume and 16-checkpoint replay

**Files:**
- Modify: D3 scripts/tests only if QA reveals a demonstrated defect.
- Create: new directory under
  `results/experimental/wheat_watkins_triad3_stage1c_d3_qa_*`
- Create: `results/experimental/TRIAD3_STAGE1C_D3_COMPUTE_QA.md`

**Interfaces:**
- Consumes all prior tasks.
- Produces measured scheduling, memory, byte-identity and audit evidence.

- [ ] **Step 1: Run the full D3-focused test suite**

```bash
uv run pytest -q \
  tests/test_triad3_d3_contract.py \
  tests/test_triad3_d3_confirmation.py \
  tests/test_triad3_d3_independent_audit.py \
  tests/test_triad3_d3_compute_benchmark.py
```

Expected: all PASS.

- [ ] **Step 2: Run interrupted/resumed QA**

Use a config that retains the exact B=2,000/999 statistical contract but sets
`execution.qa=true`. Select an explicit QA-only identity subset without
changing protocol B. Stop after two bundles, resume the same subset with a
different pre-benchmarked worker count, run the independent auditor in
QA-subset mode, and require zero missing/failure units within that declared
subset.

- [ ] **Step 3: Run clean one-worker byte comparison**

Recompute the same QA identities with one worker in a clean directory.
Compare every checkpoint and the replicate TSV byte-for-byte with the resumed
run.

- [ ] **Step 4: Replay the frozen 16 identities**

Recompute indices 0, 1, 999 and 1999 across all four cells using QA streams.
Require 16/16 checkpoint byte identity. Do not consume formal streams.

- [ ] **Step 5: Run the 32/64/96 benchmark**

Run 192 QA bundles twice per worker count with BLAS=1. Record wall time, peak
RSS, maximum active concurrency, checkpoint throughput and aggregate bytes.
Apply `select_workers` without manual override.

- [ ] **Step 6: Write the compute QA report**

Record commands, environment, chosen worker count, measured resource
reservation, all hashes and PASS/FAIL gates. State explicitly that results are
engineering-only and formal seed consumption is false.

- [ ] **Step 7: Commit QA evidence without bulk outputs**

```bash
git add results/experimental/TRIAD3_STAGE1C_D3_COMPUTE_QA.md
git commit -m "docs: record triad3 D3 compute QA"
```

Do not commit large checkpoint directories.

---

### Task 9: Full regression, code audit and third-path fixture

**Files:**
- Create: `results/experimental/TRIAD3_STAGE1C_D3_IMPLEMENTATION_AUDIT.md`
- Create: `scripts/triad3_d3_third_path_recompute.py`
- Create: `tests/test_triad3_d3_third_path_recompute.py`

**Interfaces:**
- The third path consumes checkpoint/TSV fixtures and imports neither the
  runner nor primary auditor.

- [ ] **Step 1: Write a failing third-path fixture test**

```python
def test_third_path_reproduces_eight_counts(complete_fixture):
    result = THIRD.recompute(complete_fixture.out_dir)
    assert result["status"] == "PASS"
    assert set(result["evaluations"]) == {
        f"{cell}:{method}"
        for cell in CELLS
        for method in METHODS
    }
```

- [ ] **Step 2: Run and verify RED**

Run:
`uv run pytest -q tests/test_triad3_d3_third_path_recompute.py`

Expected: FAIL because the third-path module is absent.

- [ ] **Step 3: Implement the minimal independent reparser**

Parse canonical JSON directly, recompute hashes, plus-one P, eight counts,
120/126 zones and candidate decisions. Compare with the primary auditor only
after independent computation.

- [ ] **Step 4: Run all triad3 and repository tests**

```bash
uv run ruff check scripts/triad3_d3_*.py tests/test_triad3_d3_*.py
uv run python -m py_compile scripts/triad3_d3_*.py
uv run pytest -q tests/test_triad3_*.py
uv run pytest
```

Expected: zero failures; environment-dependent skips must be enumerated.

- [ ] **Step 5: Write the implementation audit**

Document line-by-line spec coverage, phenotype sealing, RNG namespace
separation, scheduler concurrency, resume/byte evidence, auditor independence,
tests, environment skips and remaining formal-preparation gates.

- [ ] **Step 6: Commit the third path and audit**

```bash
git add \
  scripts/triad3_d3_third_path_recompute.py \
  tests/test_triad3_d3_third_path_recompute.py \
  results/experimental/TRIAD3_STAGE1C_D3_IMPLEMENTATION_AUDIT.md
git commit -m "test: audit triad3 D3 implementation"
```

---

### Task 10: Lock, formal config, prepare-only manifest and pre-run ledger

**Files:**
- Create: `results/experimental/TRIAD3_STAGE1C_D3_LOCK_DECISION.md`
- Create: new formal D3 directory with
  `configs/d3.generated.yaml`, `manifest.frozen.json` and sidecars.
- Create: `results/experimental/TRIAD3_STAGE1C_D3_PRE_RUN_CODEX_CHECK.md`

**Interfaces:**
- Consumes approved spec, completed implementation audit and measured QA.
- Produces technical readiness only; no formal checkpoints.

- [ ] **Step 1: Reconfirm authorization boundary**

Before file generation, record that design/implementation authorization does
not authorize formal simulation. If implementation execution was not
explicitly authorized, stop this task before deriving the seed or config.

- [ ] **Step 2: Lock the approved spec digest**

Hash the committed spec bytes. Derive the 128-bit formal master seed exactly
from the domain and spec digest. Record the derivation and digest in the lock
decision; do not instantiate an RNG.

- [ ] **Step 3: Generate and validate formal config**

Generate the exact four cells, two methods, B=2,000/999, 120/126 thresholds,
benchmark-selected workers, BLAS=1, no-peeking and new output directory.
Run `--validate-only` and require
`formal_random_stream_consumed=false`.

- [ ] **Step 4: Create manifest with `--prepare-only`**

Require:

- canonical manifest sidecar matches;
- every bound file/array independently rehashes;
- trait column was not read or hashed;
- formal RNG was not instantiated;
- no checkpoint, replicate TSV or run-complete file exists.

- [ ] **Step 5: Execute the independent ten-part pre-run check**

Cover protocol/code match, manifest immutability, RNG namespaces, exact
binomial boundaries, ordered identities, checkpoint/resume QA, worker
benchmark, 16 replay, tests/lint and phenotype/no-peeking seal.

- [ ] **Step 6: Sign the all-PASS ledger**

Write SHA-256 sidecars for the lock decision, manifest audit and pre-run
report. If any item is not PASS, formal readiness remains false.

- [ ] **Step 7: Stop before formal execution**

Report the exact manifest ID and technical readiness. Do not call the formal
runner without a new explicit instruction that authorizes the B=2,000 D3 run.

---

## Final verification checklist

- [ ] D1 runner and D1 artifacts rehash unchanged.
- [ ] D3 spec requirements map to Tasks 1--10.
- [ ] All D3-focused tests pass.
- [ ] All triad3 regression tests pass.
- [ ] Full repository tests have zero failures.
- [ ] Relevant ruff and py_compile pass.
- [ ] 1/32/64/96 QA checkpoint bytes match.
- [ ] Scheduler maximum active concurrency is greater than one.
- [ ] Interrupted/resumed and clean QA outputs match byte-for-byte.
- [ ] 16/16 replay checkpoints match.
- [ ] Independent auditor and third path agree.
- [ ] Formal prepare-only directory contains no inferential/run artifacts.
- [ ] Real phenotype trait values remain unread.
- [ ] Claude remains uncalled.
- [ ] Formal D3 execution remains locked pending explicit authorization.
