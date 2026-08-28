# Group omniB Follow-up and Evidence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one candidate-only stability and evidence command for canonical edge- or group-primary omniB results with two or more subgenomes.

**Architecture:** `omnib_family.py` exposes a small subset-scoring API that reuses frozen formal features while refitting the deletion null model. A separate `followup.py` owns result loading, deterministic deletions, locus clustering and outputs; `evidence.py` owns provenance-bound external-table joins and evidence tiers.

**Tech Stack:** Python 3.10+, NumPy, SciPy, pandas, PyYAML, joblib, pytest.

**Spec:** `docs/superpowers/specs/2026-08-28-group-omnib-followup-evidence-design.md`

## Global Constraints

- Follow-up is candidate-only sensitivity and never creates discoveries or recalibrates FWER.
- Formal master family, MAF gate and seeded omniB feature definitions remain frozen in deletions.
- Exact replay must pass before deletion analysis starts.
- Support canonical group omniB for 2+ subgenomes; no direct third-/fourth-order coefficient.
- Missing environment metadata is an explicit unavailable state, not a fabricated analysis.
- Evidence adapters are species-independent and every source is content-hashed.
- Manuscript output may claim omnibus pairwise evidence only, never causal or physical mechanism.

---

### Task 1: Frozen prepared-context subset scorer

**Files:**
- Modify: `src/homoeogwas/omnib_family.py`
- Create: `tests/test_omnib_followup.py`

**Interfaces:**
- Produces dataclass: `OmniBSubsetScores(edge_p, group_p, edge_components, covariance_components)`
- Produces: `score_omnib_subset(scores, family, expanded, keep, y_raw, *, n_jobs=8) -> OmniBSubsetScores`
- Consumes an `OmniBFamilyScores` created by `score_omnib_family(..., bootstrap_B=0)`

- [ ] **Step 1: Write failing two-/three-/four-copy equivalence tests**

```python
@pytest.mark.parametrize("subgenomes", [("A", "C"), ("A", "B", "D"), ("A", "B", "C", "D")])
def test_subset_all_rows_replays_prepared_observed(subgenomes):
    subdata, family, y, sample_idx = toy_group_inputs(subgenomes)
    scores, expanded = score_omnib_family(
        subdata, family, y, sample_idx, bootstrap_B=0, bootstrap_seed=2026, n_jobs=1)
    replay = score_omnib_subset(
        scores, family, expanded, np.arange(len(y)), y, n_jobs=1)
    np.testing.assert_allclose(replay.edge_p, scores.edge_p[:, 0], rtol=1e-6, atol=1e-10)
    np.testing.assert_allclose(replay.group_p, scores.group_p[:, 0], rtol=1e-6, atol=1e-10)

def test_subset_scorer_rejects_covariate_context_until_supported():
    scores = prepared_scores_with_covariates()
    with pytest.raises(ValueError, match="fixed covariates"):
        score_omnib_subset(scores, family, expanded, keep, y)
```

- [ ] **Step 2: Verify RED**

Run: `uv run pytest tests/test_omnib_followup.py -q`  
Expected: subset scorer is missing.

- [ ] **Step 3: Implement subset kernel/null/feature scoring**

Validate sorted unique `keep` indices, finite aligned phenotype and at least ten samples. For a
proper subset, slice each `scores.null_kernels` matrix and divide by `trace(K)/n`. Refit with
`interact.null_lmm_fit(..., seed=42)`, apply `interact.rank_int`, subset each frozen feature block,
score every frozen estimable edge with `omnib_components_over_Y`, then reduce each group using the
existing edge membership. For all rows, retain the original kernel matrices exactly.

- [ ] **Step 4: Verify GREEN and four-copy six-edge behavior**

Run: `uv run pytest tests/test_omnib_followup.py -q`  
Expected: all-row equality passes for 2/3/4 copies and the four-copy fixture has six edges.

- [ ] **Step 5: Commit**

```bash
git add src/homoeogwas/omnib_family.py tests/test_omnib_followup.py
git commit -m "feat: score frozen omniB features after sample deletion"
```

### Task 2: Canonical result loading and formal-hit replay

**Files:**
- Create: `src/homoeogwas/followup.py`
- Modify: `tests/test_omnib_followup.py`

**Interfaces:**
- Produces: `FollowupInputs`
- Produces: `load_followup_inputs(results_dir, config=None, ranking=None) -> FollowupInputs`
- Produces: `prepare_formal_replay(inputs, *, n_jobs=8) -> PreparedFollowup`
- Produces: `extract_primary_scores(subset_scores, prepared) -> pd.DataFrame`

- [ ] **Step 1: Write failing canonical/no-hit/legacy tests**

```python
def test_loader_selects_only_formal_primary_hits(tmp_path):
    paths = write_canonical_result_fixture(tmp_path, hypothesis_unit="edge", sig=[True, False])
    loaded = load_followup_inputs(tmp_path)
    assert loaded.formal_hits["hypothesis_id"].tolist() == ["edge:AC:gA1:gC1"]

def test_loader_returns_no_discovery_without_candidates(tmp_path):
    write_canonical_result_fixture(tmp_path, hypothesis_unit="group", sig=[False, False])
    loaded = load_followup_inputs(tmp_path)
    assert loaded.formal_hits.empty

def test_loader_refuses_legacy_ranking_as_canonical(tmp_path):
    write_legacy_pair_fixture(tmp_path)
    with pytest.raises(FollowupError, match="legacy.*migration"):
        load_followup_inputs(tmp_path)
```

- [ ] **Step 2: Verify RED**

Run: `uv run pytest tests/test_omnib_followup.py -q`  
Expected: follow-up loader is missing.

- [ ] **Step 3: Implement config/ranking discovery and canonical gates**

Normalize the config with `normalize_interact_config`, validate it, load the exact master family,
verified subgenome NPZ/BED inputs, string sample IDs and phenotype aggregation matching
`cmd_interact`. Require all canonical fields from spec section 3.

- [ ] **Step 4: Implement prepared replay and tolerance checks**

Call `score_omnib_family(..., bootstrap_B=0)` using config parameters. Map edge identities as
`edge:<edge_id>` and group identities as `group:<group_id>`. Compare every formal hit's replay P to
the ranking with `rtol=1e-6, atol=1e-10`; compare the evidence-driving edge/component where present.
Raise `FollowupError` before any deletion on mismatch.

- [ ] **Step 5: Verify GREEN**

Run: `uv run pytest tests/test_omnib_followup.py -q`  
Expected: canonical loading, no-hit and exact-replay tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/homoeogwas/followup.py tests/test_omnib_followup.py
git commit -m "feat: replay canonical group omniB discoveries"
```

### Task 3: Deterministic material and environment deletion

**Files:**
- Modify: `src/homoeogwas/followup.py`
- Modify: `tests/test_omnib_followup.py`

**Interfaces:**
- Produces: `deterministic_folds(samples, *, n_folds=20, seed=2026) -> tuple[tuple[str, ...], ...]`
- Produces: `run_material_deletion(prepared, *, n_folds, n_jobs) -> pd.DataFrame`
- Produces: `run_environment_deletion(prepared, environment_col, *, n_jobs) -> tuple[pd.DataFrame, str]`
- Produces: `summarize_stability(prepared, material, environment) -> pd.DataFrame`

- [ ] **Step 1: Write failing fold and environment tests**

```python
def test_material_folds_cover_every_string_sample_once():
    samples = [f"S{i:03d}" for i in range(41)]
    folds = deterministic_folds(samples, n_folds=8, seed=2026)
    flat = [sample for fold in folds for sample in fold]
    assert sorted(flat) == sorted(samples)
    assert max(map(len, folds)) - min(map(len, folds)) <= 1

def test_environment_deletion_reaggregates_repeated_rows(prepared_fixture):
    frame, status = run_environment_deletion(
        prepared_fixture.with_repeated_environments(), "environment", n_jobs=1)
    assert status == "COMPLETED"
    assert set(frame["deleted_environment"]) == {"E1", "E2"}
    assert (frame["n"] >= 10).all()

def test_missing_environment_is_explicit(prepared_fixture):
    frame, status = run_environment_deletion(prepared_fixture, None, n_jobs=1)
    assert frame.empty
    assert status == "NOT_AVAILABLE_NO_ENVIRONMENT_COLUMN"
```

- [ ] **Step 2: Verify RED**

Run: `uv run pytest tests/test_omnib_followup.py -q`  
Expected: deletion functions are missing.

- [ ] **Step 3: Implement deterministic folds and parallel scoring**

Partition sorted string IDs by SHA-256 of `f"{seed}\0{sample}"`. Use joblib threading and force
inner BLAS limits to one. Each fold calls `score_omnib_subset` and extracts only frozen formal IDs.

- [ ] **Step 4: Implement environment record deletion**

For repeated rows, remove one environment level, group remaining non-missing trait values by string
sample ID with `sort=False`, and align the resulting mean to genotype samples. Mark a level
`UNAVAILABLE_LT10_SAMPLES` rather than scoring when fewer than ten remain.

- [ ] **Step 5: Implement stability descriptors**

Compute median/max raw P, nominal-support fraction and driver agreement by formal hypothesis. Copy
formal adjusted P unchanged. Add the fixed interpretation string from the spec.

- [ ] **Step 6: Verify GREEN**

Run: `uv run pytest tests/test_omnib_followup.py -q`  
Expected: folds, environment aggregation and stability summaries pass.

- [ ] **Step 7: Commit**

```bash
git add src/homoeogwas/followup.py tests/test_omnib_followup.py
git commit -m "feat: add omniB deletion sensitivity"
```

### Task 4: Generic homoeolog-region clustering

**Files:**
- Modify: `src/homoeogwas/followup.py`
- Modify: `tests/test_omnib_followup.py`

**Interfaces:**
- Produces: `candidate_coordinates(prepared) -> pd.DataFrame`
- Produces: `cluster_formal_units(hits, feature_pc1, *, max_distance_bp=1_000_000, pc1_r2_threshold=0.64) -> tuple[pd.DataFrame, pd.DataFrame]`

- [ ] **Step 1: Write failing physical/LD/cross-direction tests**

```python
def test_cluster_merges_same_copy_set_by_physical_rule(two_edge_hits):
    loci, links = cluster_formal_units(two_edge_hits, {}, max_distance_bp=1_000_000)
    assert loci["locus_id"].nunique() == 1
    assert links.iloc[0]["merge_reason"].startswith("all_compared_copies_within")

def test_cluster_keeps_different_edge_directions_separate(cross_direction_hits):
    loci, links = cluster_formal_units(cross_direction_hits, {})
    assert loci["locus_id"].nunique() == 2
    assert not links.iloc[0]["merge"]

def test_cluster_reports_physical_merge_even_when_one_copy_ld_is_low(two_edge_hits):
    pc1 = {("A", "g1", "g2"): 0.7, ("C", "h1", "h2"): 0.001}
    loci, links = cluster_formal_units(two_edge_hits, pc1)
    assert loci["locus_id"].nunique() == 1
    assert links.iloc[0]["merge_basis"] == "physical"
```

- [ ] **Step 2: Verify RED**

Run: `uv run pytest tests/test_omnib_followup.py -q`  
Expected: clustering API is missing.

- [ ] **Step 3: Implement exact-build coordinate extraction and union-find clustering**

Read chromosome/position from each `SubgenomeData.chunk` at the frozen gene SNP indices and use the
median SNP position. Require the same compared copy set and at least two comparable copies. Record
every per-copy distance and PC1 r² in `locus_links.tsv`.

- [ ] **Step 4: Verify GREEN**

Run: `uv run pytest tests/test_omnib_followup.py -q`  
Expected: physical, LD and copy-set boundary tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/homoeogwas/followup.py tests/test_omnib_followup.py
git commit -m "feat: cluster homoeolog discoveries into reporting regions"
```

### Task 5: Provenance-bound functional evidence adapters

**Files:**
- Create: `src/homoeogwas/evidence.py`
- Create: `tests/test_evidence.py`

**Interfaces:**
- Produces: `EvidenceSource`, `EvidenceManifest`, `load_evidence_manifest(path)`
- Produces: `join_candidate_evidence(candidate_genes, manifest) -> tuple[pd.DataFrame, dict]`
- Produces: `assign_evidence_tiers(formal_units, evidence) -> pd.DataFrame`

- [ ] **Step 1: Write failing join/tier/provenance tests**

```python
def test_evidence_join_keeps_candidates_missing_from_sources(tmp_path):
    manifest = write_evidence_manifest(tmp_path, annotation={"g1": "transporter"})
    joined, provenance = join_candidate_evidence(pd.DataFrame({"gene_id": ["g1", "g2"]}), manifest)
    assert joined["gene_id"].tolist() == ["g1", "g2"]
    assert provenance["sources"][0]["sha256"]

def test_tier_precedence_is_functional_qtl_expression_annotation():
    evidence = evidence_rows_for_all_tiers()
    tiers = assign_evidence_tiers(formal_units(), evidence)
    assert tiers.set_index("hypothesis_id").loc["h-functional", "evidence_tier"] == "direct functional evidence"
    assert tiers.set_index("hypothesis_id").loc["h-expression", "evidence_tier"] == "matched-tissue expression support"
```

- [ ] **Step 2: Verify RED**

Run: `uv run pytest tests/test_evidence.py -q`  
Expected: evidence module is missing.

- [ ] **Step 3: Implement strict manifest loading and table joins**

Require `evidence_version: 1`, unique source names, supported kinds, existing path, `gene_col` and
optional `support_col`. Preserve candidate order and duplicate source rows as separate evidence
records. Hash source bytes with streaming SHA-256.

- [ ] **Step 4: Implement deterministic evidence tiers**

Treat case-insensitive values in `{1,true,yes,supported,positive}` as truthy. Apply the exact tier
precedence from spec section 9 and retain source/citation lists. Do not infer direct function from
annotation text.

- [ ] **Step 5: Verify GREEN**

Run: `uv run pytest tests/test_evidence.py -q`  
Expected: joins, hashes and tier precedence pass.

- [ ] **Step 6: Commit**

```bash
git add src/homoeogwas/evidence.py tests/test_evidence.py
git commit -m "feat: join candidate functional evidence with provenance"
```

### Task 6: End-to-end follow-up command and audit outputs

**Files:**
- Modify: `src/homoeogwas/followup.py`
- Modify: `src/homoeogwas/cli.py`
- Modify: `tests/test_omnib_followup.py`
- Create: `docs/followup.md`
- Modify: `README.md`

**Interfaces:**
- Produces: `run_followup(results_dir, *, config=None, ranking=None, material_folds=20, environment_col=None, evidence=None, n_jobs=8, grm_blas_threads=8) -> dict`
- Produces CLI: `homoeogwas follow-up RESULTS_DIR [options]`

- [ ] **Step 1: Write failing no-hit and mocked end-to-end tests**

```python
def test_run_followup_no_hit_writes_successful_stop(tmp_path):
    write_canonical_result_fixture(tmp_path, hypothesis_unit="edge", sig=[False])
    result = run_followup(tmp_path, n_jobs=1)
    assert result["status"] == "NO_FORMAL_DISCOVERY"
    assert (tmp_path / "followup" / "followup_summary.json").exists()
    assert not (tmp_path / "followup" / "candidate_evidence.tsv").exists()

def test_followup_cli_parses_generic_options(tmp_path, monkeypatch):
    called = {}
    monkeypatch.setattr("homoeogwas.followup.run_followup", lambda *a, **k: called.update(k) or {"status": "COMPLETED"})
    assert main(["follow-up", str(tmp_path), "--material-folds", "8", "--n-jobs", "4"]) == 0
    assert called["material_folds"] == 8
```

- [ ] **Step 2: Verify RED**

Run: `uv run pytest tests/test_omnib_followup.py -q`  
Expected: orchestrator/CLI is missing.

- [ ] **Step 3: Implement orchestration and atomic outputs**

Run input loading, no-hit stop, prepared replay, locus clustering, material deletion, optional
environment deletion, optional evidence join, then write all outputs from spec section 10. Use
temporary sibling files plus replace for JSON summaries.

- [ ] **Step 4: Implement independent consistency audit**

Assert replay tolerance, fold coverage, unique formal IDs, locus membership, unchanged formal
adjusted P and exact candidate/evidence key coverage. Write `independent_audit.json` with `PASS` or
raise before claiming completion.

- [ ] **Step 5: Add CLI and breeder-facing documentation**

Document allowed claims, no-hit behavior, environment metadata rules, evidence manifest generation
and why deletion nominal P is not another discovery test.

- [ ] **Step 6: Verify focused and full suite**

Run: `uv run pytest tests/test_omnib_followup.py tests/test_evidence.py tests/test_interact.py tests/test_workflow.py -q`  
Expected: all focused tests pass.

Run: `uv run ruff check src/homoeogwas/omnib_family.py src/homoeogwas/followup.py src/homoeogwas/evidence.py src/homoeogwas/cli.py tests/test_omnib_followup.py tests/test_evidence.py`  
Expected: no lint errors.

Run: `uv run pytest -q`  
Expected: full suite passes with only environment-dependent skips.

- [ ] **Step 7: Commit**

```bash
git add src/homoeogwas/followup.py src/homoeogwas/cli.py tests/test_omnib_followup.py docs/followup.md README.md
git commit -m "feat: expose canonical omniB follow-up workflow"
```

### Task 7: Rapeseed regression and multi-species migration checks

**Files:**
- Create: `tests/test_followup_regression_contract.py`
- Modify: `analyses/cross_species_interaction_inventory.yaml`

**Interfaces:**
- Consumes existing frozen result summaries when present; skips data-dependent checks in clean CI
- Proves the product reproduces the current rapeseed audit contract without changing formal results

- [ ] **Step 1: Add local-data guarded regression test**

```python
@pytest.mark.skipif(not RAPESEED_RESULT.exists(), reason="local formal rapeseed result unavailable")
def test_rapeseed_followup_matches_frozen_contract(tmp_path):
    result = run_followup(RAPESEED_RESULT, out_dir=tmp_path, material_folds=20, n_jobs=8)
    audit = json.loads((tmp_path / "independent_audit.json").read_text())
    assert audit["formal_hit_count"] == 2
    assert audit["formal_replay_max_abs_error"] < 1e-10
    assert audit["material_deletion"]["samples_deleted_exactly_once"] == 926
    assert audit["locus_clustering"]["independent_loci"] == 1
```

- [ ] **Step 2: Run the rapeseed contract test**

Run: `uv run pytest tests/test_followup_regression_contract.py -q`  
Expected on the project machine: PASS; expected in clean CI: one explicit skip.

- [ ] **Step 3: Verify historical migration summaries**

Run registry validation and ensure wheat, cotton and peanut fixtures remain `historical`; confirm
the follow-up command does not claim to have rerun them canonically.

- [ ] **Step 4: Run final full verification**

Run: `uv run pytest -q`  
Expected: full suite passes with only declared environment-dependent skips.

Run: `uv run ruff check src tests`  
Expected: no new lint errors; any pre-existing repository-wide lint debt is reported separately.

- [ ] **Step 5: Commit**

```bash
git add tests/test_followup_regression_contract.py analyses/cross_species_interaction_inventory.yaml
git commit -m "test: freeze cross-species follow-up contracts"
```

