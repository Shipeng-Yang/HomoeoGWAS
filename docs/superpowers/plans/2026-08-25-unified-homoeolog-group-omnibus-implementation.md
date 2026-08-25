# Unified Homoeolog-Group Omnibus Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one production interaction engine in which pair omniB is the primitive, group omniB is an ACAT over pair edges, and pair/triad/4+ analyses share one null model and bootstrap-FWER implementation; then run the matched wheat F2143-derived edge analysis.

**Architecture:** A new pure `group_family` module owns canonical homoeolog groups and deterministic edge expansion. The existing numerical helpers in `interact.py` score each unique edge once under one all-subgenome LMM, derive group p-values from the same edge matrix, and calibrate whichever layer is declared primary with the existing plus-one bootstrap min-P function. A small config-normalization module preserves legacy pairwise/triad inputs while new workflow-generated configs use `mode: group`.

**Tech Stack:** Python 3.10+, NumPy, SciPy, pandas, joblib, PyYAML, pytest, PLINK BED/NPZ inputs, HomoeoGWAS CLI and audit framework.

**Spec:** `docs/superpowers/specs/2026-08-25-unified-homoeolog-group-omnibus-design.md`

## Global Constraints

- Preserve all unrelated dirty-worktree changes; stage and commit only files or hunks belonging to the current task.
- The production statistic is `omniB`; legacy burden and experimental `triad3` remain explicit opt-ins.
- The primary transform is `INT`; raw is sensitivity-only unless independently predeclared.
- The production null uses all declared subgenome kernels with `grm_from_X`, `maf_min: 0.01`.
- Formal calibration uses kinship-preserving parametric bootstrap min-P, `B: 2000`, seed `2026` for the wheat run.
- A run declares exactly one primary `hypothesis_unit`, `edge` or `group`; a second layer is localization unless `family_scope: joint` is explicitly selected.
- A direct four-way product statistic is always refused. Four-copy group omniB may aggregate the six supported pair edges.
- Every generated config is validated before an expensive run, and every finished run is audited.
- Sample IDs remain strings, and every SNP-to-gene NPZ must match the exact BIM fingerprint.
- BLAS libraries use one thread per worker during bootstrap scans.
- The existing Route-B and 40,678-pair result directories are immutable historical artifacts.

---

### Task 1: Canonical Master Groups and Deterministic Edge Expansion

**Files:**
- Create: `src/homoeogwas/group_family.py`
- Create: `tests/test_group_family.py`

**Interfaces:**
- Produces: `MasterGroupFamily(subgenomes, group_ids, genes)`.
- Produces: `EdgeRecord(edge_id, sub_x, sub_y, gene_x, gene_y, source_group_ids)`.
- Produces: `ExpandedEdgeFamily(edges, group_edge_indices)`.
- Produces: `load_master_group_family(path, subgenomes, require_group_id=False)`.
- Produces: `expand_pair_edges(family)`.
- Consumes: delimited TSV/CSV files and ordered subgenome labels only; it has no phenotype or genotype dependency.

- [ ] **Step 1: Write failing construction and reduction tests**

```python
from homoeogwas.group_family import (
    MasterGroupFamily, expand_pair_edges, load_master_group_family,
)


def test_expand_three_copy_groups_is_deterministic_and_deduplicates_edges():
    fam = MasterGroupFamily(
        subgenomes=("A", "B", "D"),
        group_ids=("g2", "g1", "g3"),
        genes=(("a2", "b2", "d2"),
               ("a1", "b1", "d1"),
               ("a1", "b1", "d9")),
    )
    out = expand_pair_edges(fam)
    assert [e.direction for e in out.edges] == [
        "AB", "AD", "BD", "AB", "AD", "BD", "AD", "BD"
    ]
    assert out.edges[3].source_group_ids == ("g1", "g3")
    assert out.group_edge_indices[1][0] == out.group_edge_indices[2][0]


def test_two_copy_group_has_exactly_one_edge():
    fam = MasterGroupFamily(
        subgenomes=("A", "D"), group_ids=("g1",),
        genes=(("a1", "d1"),),
    )
    out = expand_pair_edges(fam)
    assert len(out.edges) == 1
    assert out.group_edge_indices == ((0,),)
```

- [ ] **Step 2: Run the focused tests and confirm the module is missing**

Run: `uv run pytest tests/test_group_family.py -v`

Expected: collection fails with `ModuleNotFoundError: homoeogwas.group_family`.

- [ ] **Step 3: Implement immutable family records, validation and expansion**

```python
@dataclass(frozen=True)
class MasterGroupFamily:
    subgenomes: tuple[str, ...]
    group_ids: tuple[str, ...]
    genes: tuple[tuple[str, ...], ...]

    def __post_init__(self) -> None:
        if len(self.subgenomes) < 2 or len(set(self.subgenomes)) != len(self.subgenomes):
            raise ValueError("subgenomes must contain at least two unique labels")
        if len(self.group_ids) != len(self.genes):
            raise ValueError("group_ids and genes must have the same row count")
        if len(set(self.group_ids)) != len(self.group_ids):
            raise ValueError("group_id values must be unique")
        if any(len(row) != len(self.subgenomes) for row in self.genes):
            raise ValueError("every group must contain exactly one gene per subgenome")


@dataclass(frozen=True)
class EdgeRecord:
    edge_id: str
    direction: str
    sub_x: str
    sub_y: str
    gene_x: str
    gene_y: str
    source_group_ids: tuple[str, ...]


@dataclass(frozen=True)
class ExpandedEdgeFamily:
    edges: tuple[EdgeRecord, ...]
    group_edge_indices: tuple[tuple[int, ...], ...]
```

Expansion must iterate master rows in input order and `itertools.combinations(subgenomes, 2)` order. The unique key is `(sub_x, sub_y, gene_x, gene_y)`. A repeated key appends the group ID to the existing edge and reuses its integer index.

- [ ] **Step 4: Add file-loading and malformed-input tests**

```python
def test_load_legacy_table_derives_stable_group_ids(tmp_path):
    path = tmp_path / "triads.tsv"
    path.write_text("gene_A\tgene_B\tgene_D\na1\tb1\td1\n")
    fam = load_master_group_family(path, ["A", "B", "D"])
    assert fam.group_ids == ("a1|b1|d1",)


@pytest.mark.parametrize("body, message", [
    ("group_id\tgene_A\tgene_B\ng1\ta1\n", "missing"),
    ("group_id\tgene_A\tgene_B\ng1\ta1\tb1\ng1\ta2\tb2\n", "unique"),
])
def test_load_group_table_rejects_invalid_schema(tmp_path, body, message):
    path = tmp_path / "groups.tsv"
    path.write_text(body)
    with pytest.raises(ValueError, match=message):
        load_master_group_family(path, ["A", "B"], require_group_id=True)
```

- [ ] **Step 5: Run focused tests**

Run: `uv run pytest tests/test_group_family.py -v`

Expected: all tests pass.

- [ ] **Step 6: Commit the family model**

```bash
git add src/homoeogwas/group_family.py tests/test_group_family.py
git commit -m "feat: add canonical homoeolog group families"
```

---

### Task 2: Canonical Config Normalization and Workflow Generation

**Files:**
- Create: `src/homoeogwas/interaction_config.py`
- Modify: `src/homoeogwas/interact.py`
- Modify: `src/homoeogwas/workflow.py`
- Modify: `tests/test_workflow.py`
- Create: `tests/test_interaction_config.py`

**Interfaces:**
- Consumes: `MasterGroupFamily` file paths and legacy `pairs`/`triads` configs.
- Produces: `normalize_interact_config(cfg: dict) -> dict` without mutating its input.
- Produces: canonical `build_interact_config(..., groups, hypothesis_unit, subset_order, family_scope)` output.
- Preserves: `statistic=burden` and `statistic=triad3` legacy validation semantics.

- [ ] **Step 1: Write failing normalization tests**

```python
def test_normalize_legacy_pairwise_to_two_copy_group():
    cfg = {"interact": {
        "mode": "pairwise", "subgenomes": ["A", "D"],
        "pairs": "pairs.tsv", "statistic": "omniB",
    }}
    got = normalize_interact_config(cfg)
    assert got["interact"]["mode"] == "group"
    assert got["interact"]["groups"] == "pairs.tsv"
    assert got["interact"]["hypothesis_unit"] == "edge"
    assert got["interact"]["subset_order"] == 2
    assert cfg["interact"]["mode"] == "pairwise"


def test_normalize_legacy_triad_to_group_primary():
    cfg = {"interact": {
        "mode": "triad", "subgenomes": ["A", "B", "D"],
        "triads": "triads.tsv", "statistic": "omniB",
    }}
    got = normalize_interact_config(cfg)
    assert got["interact"]["groups"] == "triads.tsv"
    assert got["interact"]["hypothesis_unit"] == "group"
```

- [ ] **Step 2: Run tests and verify failure**

Run: `uv run pytest tests/test_interaction_config.py -v`

Expected: import or assertion failure because canonical normalization is absent.

- [ ] **Step 3: Implement strict normalization and validation**

```python
def normalize_interact_config(cfg: dict) -> dict:
    out = copy.deepcopy(cfg)
    ic = out.setdefault("interact", {})
    mode = str(ic.get("mode", "pairwise")).lower()
    statistic = str(ic.get("statistic", "omniB")).lower()
    if mode == "pairwise":
        ic["mode"] = "group"
        ic["groups"] = ic.pop("pairs")
        ic.setdefault("hypothesis_unit", "edge")
        ic.setdefault("subset_order", 2)
    elif mode == "triad" and statistic == "omnib":
        ic["mode"] = "group"
        ic["groups"] = ic.pop("triads")
        ic.setdefault("hypothesis_unit", "group")
        ic.setdefault("subset_order", 2)
    ic.setdefault("family_scope", "primary_only")
    return out
```

Validation must require `subset_order == 2` for `hypothesis_unit in {edge, group}` in the first release, accept 2+ subgenomes for canonical group omniB, reject `family_scope: joint` until Task 4 implements it, and retain the exact-three-copy requirement for `triad3`.

- [ ] **Step 4: Change workflow tests to expect canonical generated configs**

```python
cfg = workflow.build_interact_config(
    subgenomes=["A", "B", "D"],
    bed_prefixes={"A": "a", "B": "b", "D": "d"},
    snp_to_gene={"A": "na", "B": "nb", "D": "nd"},
    phenotype="p", sample_col="IID", trait="t", out_dir="o",
    groups="groups.tsv", hypothesis_unit="edge",
)
assert cfg["interact"]["mode"] == "group"
assert cfg["interact"]["hypothesis_unit"] == "edge"
assert cfg["interact"]["grm"] == {
    "method": "grm_from_X", "maf_min": 0.01, "scope": "all_subgenomes"
}
assert cfg["interact"]["primary_multiplicity"] == "bootstrap_minp"
```

- [ ] **Step 5: Implement canonical workflow config generation**

Extend `build_interact_config` with keyword-only arguments:

```python
groups: str | None = None
hypothesis_unit: str | None = None
subset_order: int = 2
family_scope: str = "primary_only"
```

New omniB configs must use canonical group mode. Legacy `pairs` and `triads` arguments remain accepted and are converted deterministically. `triad3` continues to emit legacy triad mode because it tests a different coefficient.

- [ ] **Step 6: Route CLI validation through normalization**

At the beginning of `validate_interact_config`, `preflight_interact` and `run_interact`, normalize once and use only the normalized `interact` block. Add a private marker such as `_normalized_version: 1` to prevent repeated conversion.

- [ ] **Step 7: Run focused tests**

Run: `uv run pytest tests/test_interaction_config.py tests/test_workflow.py -v`

Expected: all tests pass, including unchanged legacy cases.

- [ ] **Step 8: Commit config normalization**

```bash
git add src/homoeogwas/interaction_config.py tests/test_interaction_config.py
git add -p src/homoeogwas/interact.py src/homoeogwas/workflow.py \
  tests/test_workflow.py
git commit -m "feat: normalize interaction configs to group mode"
```

---

### Task 3: Shared Edge and Group omniB Score Matrices

**Files:**
- Modify: `src/homoeogwas/interact.py`
- Modify: `tests/test_interact.py`

**Interfaces:**
- Consumes: `MasterGroupFamily` and `ExpandedEdgeFamily` from Task 1.
- Produces: `OmniBFamilyScores(edge_p, group_p, edge_components_obs, edge_estimable, group_estimable)`.
- Produces: `_score_omnib_family(subdata, family, y_raw, sample_idx, *, cap, n_pc, transform, bootstrap_B, bootstrap_seed, n_jobs, grm_method, maf_min, burden_maf, min_snp, covariates=None) -> tuple[OmniBFamilyScores, ExpandedEdgeFamily]`.
- Preserves: `run_pair_scan_omnib` public behavior and `run_clique_scan_omnib` as a compatibility wrapper.

- [ ] **Step 1: Write failing score-matrix invariance tests**

```python
def _synthetic_family_scores(subgenomes, n_groups, B, return_family=False):
    rng = np.random.default_rng(912)
    subdata = {s: _make_sub_maf(rng, g=n_groups) for s in subgenomes}
    family = MasterGroupFamily(
        subgenomes=tuple(subgenomes),
        group_ids=tuple(f"group_{i}" for i in range(n_groups)),
        genes=tuple(
            tuple(f"g{i}" for _ in subgenomes) for i in range(n_groups)
        ),
    )
    scores, expanded = _score_omnib_family(
        subdata, family, rng.standard_normal(N), np.arange(N),
        cap=150, n_pc=3, transform="INT", bootstrap_B=B,
        bootstrap_seed=2026, n_jobs=1, grm_method="grm_from_X",
        maf_min=0.01, burden_maf=0.01, min_snp=3,
    )
    return (scores, expanded) if return_family else scores


def test_omnib_family_two_copy_group_equals_edge_for_all_bootstrap_columns():
    scores = _synthetic_family_scores(subgenomes=("A", "D"), n_groups=8, B=19)
    assert scores.edge_p.shape == (8, 20)
    assert scores.group_p.shape == (8, 20)
    np.testing.assert_array_equal(scores.group_p, scores.edge_p)


def test_omnib_family_three_and_four_copy_group_acat_reduction():
    for subs, expected_edges in [(('A', 'B', 'D'), 3),
                                 (('A', 'B', 'C', 'D'), 6)]:
        scores, expanded = _synthetic_family_scores(
            subgenomes=subs, n_groups=5, B=3, return_family=True)
        assert len(expanded.group_edge_indices[0]) == expected_edges
        for gi, edge_idx in enumerate(expanded.group_edge_indices):
            for col in range(scores.group_p.shape[1]):
                assert scores.group_p[gi, col] == pytest.approx(
                    acat(scores.edge_p[list(edge_idx), col]))
```

- [ ] **Step 2: Run focused tests and verify failure**

Run: `uv run pytest tests/test_interact.py -k 'omnib_family' -v`

Expected: failure because `OmniBFamilyScores` and `_score_omnib_family` do not exist.

- [ ] **Step 3: Add the score container**

```python
@dataclass
class OmniBFamilyScores:
    edge_p: np.ndarray
    group_p: np.ndarray
    edge_components_obs: np.ndarray
    edge_estimable: np.ndarray
    group_estimable: np.ndarray
    W: np.ndarray
    y: np.ndarray
    covariance_components: dict[str, float]
```

- [ ] **Step 4: Refactor feature preparation to cache each gene once**

Build the minor burden, PC1 and full PC feature block by `(subgenome, gene_id)`. Gate SNPs before caching. Reject the complete run if a design-valid observed edge becomes non-finite after whitening; preserve NaN only for predeclared non-estimable edges.

- [ ] **Step 5: Score each unique edge for observed and shared null responses**

Allocate `edge_p = np.full((E, B + 1), np.nan)` and `edge_components_obs = np.full((E, 3), np.nan)`. Each joblib task receives deterministic contiguous edge indices, uses `_omnib_components_over_Y`, ACAT-combines the three components for every response column and returns only its block. All subgenome directions use the same `Yw` and `Cw` arrays.

- [ ] **Step 6: Derive the group matrix from the edge matrix**

```python
group_p = np.full((len(family.group_ids), edge_p.shape[1]), np.nan)
for gi, idx in enumerate(expanded.group_edge_indices):
    for col in range(edge_p.shape[1]):
        group_p[gi, col] = acat(edge_p[np.asarray(idx, int), col])
```

The implementation must mark a group estimable only when its declared edge-combination rule yields a finite p-value and must record partial groups explicitly.

- [ ] **Step 7: Rebuild compatibility wrappers on the shared scorer**

`run_pair_scan_omnib` constructs a two-copy `MasterGroupFamily` and selects the edge matrix. `run_clique_scan_omnib` constructs a k-copy family and selects the group matrix. The wrappers keep their current function names and output file naming so completed external scripts do not break.

- [ ] **Step 8: Run pair, triad and quartet numerical tests**

Run: `uv run pytest tests/test_interact.py -k 'omnib or clique_n4' -v`

Expected: all selected tests pass; the quartet test evaluates six pair edges and no fourth-order product.

- [ ] **Step 9: Commit the shared scorer**

```bash
git add -p src/homoeogwas/interact.py tests/test_interact.py
git commit -m "refactor: score omniB edges and groups in one engine"
```

---

### Task 4: Primary-Family Calibration, Adjusted P Values and Joint-Family Guard

**Files:**
- Modify: `src/homoeogwas/interact.py`
- Modify: `tests/test_interact.py`
- Modify: `tests/test_audit.py`

**Interfaces:**
- Consumes: `OmniBFamilyScores` from Task 3.
- Produces: `run_group_scan_omnib(..., hypothesis_unit, family_scope) -> InteractResult`.
- Produces: one authoritative `model_diagnostics.bootstrap_fwer` object whose `family_id`, decisions and adjusted p-values match the declared primary matrix.

- [ ] **Step 1: Write failing edge-family calibration test**

```python
def test_edge_primary_calibrates_all_directions_in_one_minp_family():
    result = _run_small_group_omnib(
        subgenomes=("A", "B", "D"), hypothesis_unit="edge",
        B=19, full_dump_path=None)
    fwer = result.model_diagnostics["bootstrap_fwer"]
    assert fwer["family_id"] == "edge"
    assert fwer["n_hypotheses"] == result.G
    assert {h["direction"] for h in result.top} <= {"AB", "AD", "BD"}
    assert fwer["n_rejected"] == len(fwer["sig"])
    assert result.n_sig == fwer["n_rejected"]
```

- [ ] **Step 2: Write failing decision-consistency test**

```python
def test_group_primary_adjusted_p_threshold_and_hit_set_agree(tmp_path):
    ranking_path = tmp_path / "ranking.tsv"
    result = _run_small_group_omnib(
        subgenomes=("A", "B", "D"), hypothesis_unit="group",
        B=19, full_dump_path=str(ranking_path))
    ranking = pd.read_csv(ranking_path, sep="\t")
    fwer = result.model_diagnostics["bootstrap_fwer"]
    rejected = ranking.loc[ranking.primary_sig == 1]
    assert set(rejected.group_id) == {
        h["group_id"] for h in fwer["sig"]
    }
    assert (rejected.p_adjusted_bootstrap_minp <= 0.05).all()
    assert (rejected.p_interaction < fwer["threshold"]).all()
```

Add this concrete test helper immediately above the tests:

```python
def _run_small_group_omnib(subgenomes, hypothesis_unit, B, full_dump_path):
    rng = np.random.default_rng(515)
    n_groups = 12
    subdata = {s: _make_sub_maf(rng, g=n_groups) for s in subgenomes}
    family = MasterGroupFamily(
        subgenomes=tuple(subgenomes),
        group_ids=tuple(f"group_{i}" for i in range(n_groups)),
        genes=tuple(tuple(f"g{i}" for _ in subgenomes)
                    for i in range(n_groups)),
    )
    return run_group_scan_omnib(
        subdata, family, rng.standard_normal(N), np.arange(N),
        hypothesis_unit=hypothesis_unit, family_scope="primary_only",
        cap=150, n_pc=3, transform="INT", bootstrap_B=B,
        bootstrap_seed=2026, n_jobs=1, grm_method="grm_from_X",
        maf_min=0.01, burden_maf=0.01, min_snp=3,
        full_dump_path=full_dump_path,
    )
```

- [ ] **Step 3: Run focused tests and verify failure**

Run: `uv run pytest tests/test_interact.py -k 'primary_calibrates or decision_consistency' -v`

Expected: assertion or API failure because group and mixed-direction edge calibration are not yet exposed.

- [ ] **Step 4: Implement primary matrix selection and one calibration call**

```python
primary_p = scores.edge_p if hypothesis_unit == "edge" else scores.group_p
finite = np.isfinite(primary_p[:, 0])
cal = _bootstrap_minp_calibration(
    primary_p[finite, 0], primary_p[finite, 1:], alpha=0.05)
```

Map `cal["rejected_local"]` and `cal["adjusted_p_local"]` back to full family indices. Use those mapped indices for `InteractResult.sig`, `n_sig`, ranking `primary_sig`, JSON `bootstrap_fwer.sig` and top-level `minp_boot_rejected`.

- [ ] **Step 5: Serialize explicit hypothesis identities**

Edge hits contain `edge_id`, `group_ids`, `direction`, `sub_x`, `sub_y`, `gene_x`, `gene_y`, observed p, adjusted p and component localization. Group hits contain `group_id`, ordered genes, group p, adjusted p, driving edge and driving component.

- [ ] **Step 6: Implement the joint-family guard**

For the first production release, accept `family_scope: primary_only`. If `family_scope: joint` is requested, concatenate the finite edge and group matrices and calibrate the union in one call; prefix identities with `edge:` and `group:`. Reject any other value. Never run two independent 0.05 calibrations from one config.

- [ ] **Step 7: Add authoritative audit assertions**

Tests must create a small result JSON and verify the auditor rejects:

```python
payload["results"]["INT"]["n_sig"] += 1
record = audit_result(result_path)
assert "OMNIB_FWER_TOPLEVEL_COUNT_MISMATCH" in {
    f.code for f in record.flags
}
```

Add equivalent corruption tests for wrong `family_id`, missing adjusted p, threshold/hit disagreement and an uncalibrated second primary layer.

- [ ] **Step 8: Run calibration and audit tests**

Run: `uv run pytest tests/test_interact.py tests/test_audit.py -k 'omnib or family' -v`

Expected: all selected tests pass.

- [ ] **Step 9: Commit calibration and audit consistency**

```bash
git add -p src/homoeogwas/interact.py tests/test_interact.py tests/test_audit.py
git commit -m "feat: calibrate unified omniB hypothesis families"
```

---

### Task 5: Canonical CLI Routing, Four-Copy Support and Legacy Equivalence

**Files:**
- Modify: `src/homoeogwas/interact.py`
- Modify: `src/homoeogwas/cli.py`
- Modify: `tests/test_interact.py`
- Modify: `tests/test_workflow.py`

**Interfaces:**
- Consumes: normalized canonical configs from Task 2 and family scanner from Task 4.
- Produces: one `homoeogwas interact` path for 2+ copy group omniB.
- Preserves: legacy pairwise and triad output numerics and `triad3` exact-three-copy behavior.

- [ ] **Step 1: Write failing CLI validation tests**

```python
def test_group_mode_accepts_four_copies_but_refuses_fourway_statistic():
    cfg = {"interact": {
        "mode": "group", "subgenomes": ["A", "B", "C", "D"],
        "groups": "groups.tsv", "statistic": "omniB",
        "hypothesis_unit": "group", "subset_order": 2,
        "family_scope": "primary_only", "primary_transform": "INT",
        "primary_multiplicity": "bootstrap_minp",
        "genotype": {s: s.lower() for s in "ABCD"},
        "snp_to_gene": {s: f"n{s.lower()}" for s in "ABCD"},
        "phenotype": "p.tsv", "sample_col": "sample", "trait": "trait",
        "burden": {"cap": 150, "min_snp": 3, "maf_min": 0.01},
        "grm": {"method": "grm_from_X", "maf_min": 0.01,
                "scope": "all_subgenomes"},
        "calibration": {"method": "bootstrap", "B": 2000, "seed": 2026},
    }}
    validate_interact_config(cfg)
    cfg["interact"]["statistic"] = "fourway"
    with pytest.raises(SystemExit, match="never fits a direct four-way"):
        validate_interact_config(cfg)


def test_legacy_and_canonical_two_copy_configs_normalize_identically():
    base = {
        "subgenomes": ["A", "D"], "statistic": "omniB",
        "genotype": {"A": "a", "D": "d"},
        "snp_to_gene": {"A": "na", "D": "nd"},
        "phenotype": "p.tsv", "sample_col": "sample", "trait": "trait",
    }
    legacy = {"interact": base | {"mode": "pairwise", "pairs": "pairs.tsv"}}
    canonical = {"interact": base | {
        "mode": "group", "groups": "pairs.tsv",
        "hypothesis_unit": "edge", "subset_order": 2,
        "family_scope": "primary_only",
    }}
    assert normalize_interact_config(legacy) == normalize_interact_config(canonical)
```

- [ ] **Step 2: Run the new CLI tests and verify failure**

Run: `uv run pytest tests/test_interact.py tests/test_workflow.py -k 'four_copies or legacy_and_canonical' -v`

Expected: canonical group mode is rejected or routed incorrectly.

- [ ] **Step 3: Replace fixed mode cardinality routing**

Canonical `mode: group` accepts `len(subgenomes) >= 2`, loads `interact.groups` with `load_master_group_family`, expands edges once and calls `run_group_scan_omnib`. `statistic: triad3` bypasses normalization and retains its exact-three-copy path. Error messages describe the biological repair rather than a stack trace.

- [ ] **Step 4: Emit canonical result provenance**

Add these fields:

```python
{
    "mode": "group",
    "hypothesis_unit": hypothesis_unit,
    "subset_order": 2,
    "family_scope": family_scope,
    "group_family_sha256": family_digest,
    "edge_family_sha256": edge_digest,
    "n_groups_raw": len(family.group_ids),
    "n_unique_edges": len(expanded.edges),
    "grm_scope": "all_subgenomes",
}
```

- [ ] **Step 5: Add exact quartet and no-fourth-order tests**

The quartet test must assert six edge columns, one group p, one bootstrap family and absence of `p_fourway`, `A:B:C:D` and any fourth-order coefficient in JSON and ranking headers.

- [ ] **Step 6: Run CLI, workflow and interaction suites**

Run: `uv run pytest tests/test_interact.py tests/test_workflow.py -v`

Expected: all tests pass.

- [ ] **Step 7: Commit routing and compatibility**

```bash
git add -p src/homoeogwas/interact.py src/homoeogwas/cli.py \
  tests/test_interact.py tests/test_workflow.py
git commit -m "feat: route all ploidies through group omniB"
```

---

### Task 6: Workflow, MCP, Audit and Documentation Contract

**Files:**
- Modify: `src/homoeogwas/workflow.py`
- Modify: `src/homoeogwas/mcp_server.py`
- Modify: `src/homoeogwas/audit.py`
- Modify: `AGENTS.md`
- Modify: `README.md`
- Modify: `docs/interact_inputs.md`
- Modify: `docs/examples/strawberry_octoploid.md`
- Modify: `tests/test_workflow.py`
- Modify: `tests/test_audit.py`

**Interfaces:**
- Consumes: canonical config and result schema from Tasks 2–5.
- Produces: breeder-level config generation without requesting YAML fields.
- Produces: audit language that distinguishes edge evidence, group evidence and forbidden higher-order claims.

- [ ] **Step 1: Write failing workflow and audit summary tests**

```python
def test_run_interaction_group_dry_run_generates_and_validates(tmp_path, monkeypatch):
    result = workflow.run_interaction(
        phenotype="p.tsv", sample_col="sample", trait="trait",
        subgenomes=["A", "B", "D"],
        bed_prefixes={"A": "a", "B": "b", "D": "d"},
        snp_to_gene={"A": "na", "B": "nb", "D": "nd"},
        groups="f2143.tsv",
        hypothesis_unit="edge", out_dir=str(tmp_path), dry_run=True,
    )
    assert result["mode"] == "group"
    assert result["hypothesis_unit"] == "edge"
    assert result["steps"][0]["command"][3] == "validate"


def test_audit_group_claim_boundary(tmp_path):
    group_result_path = tmp_path / "interact_trait.json"
    group_result_path.write_text(json.dumps({
        "command": "interact", "mode": "group", "trait": "trait",
        "provenance": {
            "statistic": "omniB", "primary_transform": "INT",
            "hypothesis_unit": "group", "calibration_method": "bootstrap",
        },
        "results": {"INT": {
            "statistic": "omniB", "n": 100, "G": 4,
            "n_planned": 4, "n_valid": 4, "n_unestimable": 0,
            "n_sig": 0, "lambda_gc_obs": 1.0, "bootstrap_B": 2000,
            "model_diagnostics": {"bootstrap_fwer": {
                "method": "parametric_bootstrap_minp_plus_one",
                "family_id": "group", "alpha": 0.05, "B": 2000,
                "empirical_p": 1.0, "inferential": True,
                "rejected": False, "n_rejected": 0, "sig": [],
            }},
        }},
    }))
    record = audit_result(group_result_path)
    assert any("within a homoeolog group" in s for s in record.evidence_boundary)
    assert any("not" in s and "fourth-order" in s for s in record.evidence_boundary)
```

- [ ] **Step 2: Run focused tests and verify failure**

Run: `uv run pytest tests/test_workflow.py tests/test_audit.py -k 'group' -v`

Expected: missing canonical workflow or evidence-boundary fields.

- [ ] **Step 3: Extend workflow and MCP high-level inputs**

Expose biological arguments `groups`, `hypothesis_unit` and optional `subset_order`; default `hypothesis_unit` to `edge` for two-copy inputs and `group` for three-or-more-copy inputs. Generate YAML under `<out_dir>/configs/interact.generated.group.omnib.yaml`, validate, run, audit and summarize.

- [ ] **Step 4: Extend audit checks**

The auditor must require family hashes, expected hypothesis counts, bootstrap method, B, adjusted p on every hit, primary-unit identity and absence of a second uncalibrated rejection layer. Discovery wording follows the spec's permitted claims.

- [ ] **Step 5: Update user documentation and examples**

Document one canonical group YAML, the exact k-to-edge mapping table, legacy compatibility and an octoploid example that aggregates six pair edges without using a four-way coefficient. Update `AGENTS.md` so all agents generate canonical configs and never independently run AB/AD/BD then multiply adjusted p-values.

- [ ] **Step 6: Run workflow, MCP and audit tests**

Run: `uv run pytest tests/test_workflow.py tests/test_audit.py -v`

Expected: all tests pass.

- [ ] **Step 7: Commit the public contract**

```bash
git add -p src/homoeogwas/workflow.py src/homoeogwas/mcp_server.py \
  src/homoeogwas/audit.py AGENTS.md README.md docs/interact_inputs.md \
  docs/examples/strawberry_octoploid.md tests/test_workflow.py tests/test_audit.py
git commit -m "docs: publish unified homoeolog interaction contract"
```

---

### Task 7: Deterministic Bootstrap Checkpoint and Resume

**Files:**
- Create: `src/homoeogwas/resampling_checkpoint.py`
- Create: `tests/test_resampling_checkpoint.py`
- Modify: `src/homoeogwas/interact.py`
- Modify: `tests/test_interact.py`

**Interfaces:**
- Produces: `replicate_seed(base_seed: int, replicate_index: int) -> int` using SHA-256 domain separation, never Python `hash()`.
- Produces: `CheckpointStore(root, manifest_id, B, block_size)` with atomic `write_block`, strict `read_block` and `completed_ranges` methods.
- Consumes: primary null-p blocks shaped `(n_hypotheses, n_replicates_in_block)`.
- Guarantees: interrupted/resumed and uninterrupted runs produce identical bootstrap columns, adjusted p-values and serialized decisions.

- [ ] **Step 1: Write failing RNG and resume tests**

```python
def run_fake_blocks(root, stop_after):
    store = CheckpointStore(root, manifest_id="fixture-v1", B=10, block_size=3)
    written = 0
    for start in range(0, 10, 3):
        stop = min(start + 3, 10)
        if store.has_range(start, stop):
            continue
        cols = []
        for i in range(start, stop):
            rng = np.random.default_rng(replicate_seed(2026, i))
            cols.append(rng.uniform(size=7))
        store.write_block(start, stop, np.column_stack(cols))
        written += 1
        if stop_after is not None and written == stop_after:
            break
    null_p = store.concatenate(require_complete=stop_after is None)
    digest = hashlib.sha256(null_p.tobytes(order="C")).hexdigest()
    return SimpleNamespace(null_p=null_p, null_p_sha256=digest)


def test_replicate_seed_depends_only_on_base_and_index():
    first = [replicate_seed(2026, i) for i in range(8)]
    second = [replicate_seed(2026, i) for i in reversed(range(8))]
    assert first == list(reversed(second))
    assert len(set(first)) == 8


def test_checkpoint_resume_is_byte_identical(tmp_path):
    full = run_fake_blocks(tmp_path / "full", stop_after=None)
    run_fake_blocks(tmp_path / "resume", stop_after=2)
    resumed = run_fake_blocks(tmp_path / "resume", stop_after=None)
    assert full.null_p_sha256 == resumed.null_p_sha256
    np.testing.assert_array_equal(full.null_p, resumed.null_p)
```

`run_fake_blocks` uses `replicate_seed(2026, i)` to generate a fixed 7-by-10 null-p matrix in blocks of three, writes through `CheckpointStore`, deliberately stops after two blocks when requested, then resumes from the same manifest.

- [ ] **Step 2: Run checkpoint tests and verify failure**

Run: `uv run pytest tests/test_resampling_checkpoint.py -v`

Expected: module import failure.

- [ ] **Step 3: Implement domain-separated replicate seeds**

```python
def replicate_seed(base_seed: int, replicate_index: int) -> int:
    if replicate_index < 0:
        raise ValueError("replicate_index must be non-negative")
    body = f"homoeogwas-omnib-bootstrap-v1\0{base_seed}\0{replicate_index}".encode()
    return int.from_bytes(hashlib.sha256(body).digest()[:16], "little")
```

- [ ] **Step 4: Implement strict atomic block storage**

Each block file is `block_<start>_<stop>.npz` and contains schema version,
manifest ID, start/stop indices, replicate-seed IDs, primary null-p matrix and
its SHA-256. Write to a same-directory temporary file, flush and fsync, replace
atomically, then fsync the directory. Reading rejects wrong schema, manifest,
shape, range, hash, duplicate range, overlap, gap or out-of-bounds replicate.

- [ ] **Step 5: Make null-response generation index-addressable**

Add `null_replicates_by_index(kernels, y, C, indices, base_seed, null_fit)` in
`interact.py`. Each requested replicate uses its own `replicate_seed`, so
worker count, block size and completion order cannot change its phenotype
bytes.

- [ ] **Step 6: Integrate checkpoint blocks into group omniB**

Run and persist the observed column first. For each missing bootstrap range,
generate its indexed null phenotypes, score the selected primary family and
write the block. Once all blocks validate, concatenate strictly by replicate
index and call `_bootstrap_minp_calibration`. Record manifest ID, block size,
completed ranges and the concatenated null-p SHA-256 in result provenance.

- [ ] **Step 7: Add worker/block/resume equivalence tests**

Run the same small family with `(n_jobs, block_size)` values `(1, 5)`, `(2, 3)`
and `(4, 7)`, including one interrupted run. Assert identical observed p,
bootstrap matrix hash, threshold, adjusted p-values and hit identities.

- [ ] **Step 8: Run checkpoint and interaction tests**

Run: `uv run pytest tests/test_resampling_checkpoint.py tests/test_interact.py -k 'checkpoint or resume or worker' -v`

Expected: all selected tests pass.

- [ ] **Step 9: Commit resumability**

```bash
git add src/homoeogwas/resampling_checkpoint.py tests/test_resampling_checkpoint.py
git add -p src/homoeogwas/interact.py tests/test_interact.py
git commit -m "feat: checkpoint unified omniB bootstrap scans"
```

---

### Task 8: Matched Wheat F2143 Edge-Analysis Preparation

**Files:**
- Create: `scripts/prepare_wheat_f2143_matched_edges.py`
- Create: `tests/test_prepare_wheat_f2143_matched_edges.py`
- Create at runtime: `results/experimental/wheat_watkins_pairwise_omnib_f2143_matched_v1/configs/interact.generated.group.omnib.yaml`
- Create at runtime: `results/experimental/wheat_watkins_pairwise_omnib_f2143_matched_v1/provenance/pre_run_manifest.json`

**Interfaces:**
- Consumes: frozen F2143 SHA-256 `4a905dab9aba53b639958aa4a540b7e5494ef61cce35c9c2f6409e83394684dc`.
- Consumes: released adjusted phenotype SHA-256 `908ebabbd0a8e669d8ab6bee3758c2f91f2ed17d7525bca5f3f11da66534aebc`.
- Produces: `prepare_wheat_inputs(source_groups, source_phenotype, out_dir, expected_group_sha256, expected_phenotype_sha256, production=False) -> PreparedWheatRun`.
- Produces: a durable copied input bundle, canonical config and pre-run manifest.

- [ ] **Step 1: Write failing deterministic-preparation test**

```python
def test_prepare_binds_inputs_and_expected_family(tmp_path):
    fixture_f2143 = tmp_path / "f2143.tsv"
    fixture_f2143.write_text(
        "group_id\tgene_A\tgene_B\tgene_D\n"
        "g1\ta1\tb1\td1\n"
        "g2\ta2\tb2\td2\n"
    )
    fixture_pheno = tmp_path / "phenotype.tsv"
    fixture_pheno.write_text(
        "sample\tdays_to_emerg_env_adjusted\n"
        "s01\t40.0\n"
        "s02\t41.0\n"
    )
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    result = prepare_wheat_inputs(
        source_groups=fixture_f2143,
        source_phenotype=fixture_pheno,
        out_dir=tmp_path,
        expected_group_sha256=digest(fixture_f2143),
        expected_phenotype_sha256=digest(fixture_pheno),
        production=False,
    )
    manifest = json.loads(Path(result.manifest).read_text())
    assert manifest["n_groups"] == 2
    assert manifest["n_unique_edges"] == 6
    assert manifest["direction_counts"] == {"AB": 2, "AD": 2, "BD": 2}
    assert manifest["trait"] == "days_to_emerg_env_adjusted"
```

- [ ] **Step 2: Run the preparation test and verify failure**

Run: `uv run pytest tests/test_prepare_wheat_f2143_matched_edges.py -v`

Expected: import failure because the preparation script does not exist.

- [ ] **Step 3: Implement hash-locked input preparation**

The script reads source bytes, verifies both expected SHA-256 values before copying, writes with atomic temporary-file replacement, loads all sample IDs as strings, expands the family without phenotype access and records:

```python
{
    "n_groups": 2143,
    "n_unique_edges": edge_count,
    "direction_counts": direction_counts,
    "group_sha256": expected_group_sha256,
    "phenotype_sha256": expected_phenotype_sha256,
    "trait": "days_to_emerg_env_adjusted",
    "n_samples": 827,
    "bootstrap": {"B": 2000, "seed": 2026},
}
```

It refuses any count other than 2,143 groups or 827 non-missing unique samples in production mode.

- [ ] **Step 4: Generate the exact canonical config**

Use A/B/D genotype and verified mapping inputs from the Route-B config, `hypothesis_unit: edge`, `family_scope: primary_only`, `subset_order: 2`, all-subgenome `grm_from_X`, INT and B=2000. Enable deterministic resampling checkpoints with block size 25 under `<out_dir>/checkpoints/`.

- [ ] **Step 5: Run preparation tests**

Run: `uv run pytest tests/test_prepare_wheat_f2143_matched_edges.py -v`

Expected: all tests pass, including wrong-hash, duplicate-sample and wrong-row-count cases.

- [ ] **Step 6: Prepare the real analysis bundle**

Run:

```bash
uv run python scripts/prepare_wheat_f2143_matched_edges.py \
  --source-groups /tmp/U7_GWAS-d3-sdd/results/experimental/wheat_watkins_triad3_stage1c_plan_s_real_days_to_emerg_env_adjusted_classical_f_v2/inputs/f2143.generated.tsv \
  --source-phenotype /tmp/U7_GWAS-d3-sdd/results/experimental/wheat_watkins_triad3_stage1c_plan_s_real_days_to_emerg_env_adjusted_classical_f_v2/inputs/phenotype.release.tsv \
  --out-dir results/experimental/wheat_watkins_pairwise_omnib_f2143_matched_v1
```

Expected: exactly 2,143 groups, at most 6,429 unique edges, 827 samples and both expected SHA-256 values.

- [ ] **Step 7: Validate the generated config**

Run:

```bash
uv run homoeogwas validate -c \
  results/experimental/wheat_watkins_pairwise_omnib_f2143_matched_v1/configs/interact.generated.group.omnib.yaml
```

Expected: validation passes with all BED/NPZ fingerprints and sample joins valid.

- [ ] **Step 8: Commit the preparation code, not generated results**

```bash
git add scripts/prepare_wheat_f2143_matched_edges.py \
  tests/test_prepare_wheat_f2143_matched_edges.py
git commit -m "feat: prepare matched wheat F2143 edge analysis"
```

---

### Task 9: Verification, Performance QA, Formal Wheat Run and Audit

**Files:**
- Modify after a reproduced failure: only files already in Tasks 1–8, with a failing regression test added before every correction.
- Create at runtime: `results/experimental/wheat_watkins_pairwise_omnib_f2143_matched_v1/qa/`
- Create at runtime: `results/experimental/wheat_watkins_pairwise_omnib_f2143_matched_v1/interact_days_to_emerg_env_adjusted.json`
- Create at runtime: `results/experimental/wheat_watkins_pairwise_omnib_f2143_matched_v1/audit/`
- Create at runtime: `results/experimental/wheat_watkins_pairwise_omnib_f2143_matched_v1/biological_summary.zh_en.md`

**Interfaces:**
- Consumes: validated canonical wheat config from Task 8.
- Produces: audited B=2000 wheat edge-family result and manuscript-safe summary.

- [ ] **Step 1: Run static and complete automated verification**

Run:

```bash
uv run python -m compileall -q src scripts
uv run pytest -q
```

Expected: compile succeeds and the full repository test suite passes. Any failure is diagnosed before changing code; no test is weakened to accommodate a regression.

- [ ] **Step 2: Run null and planted-signal calibration tests**

Run:

```bash
uv run pytest tests/test_interact.py -k \
  'null or planted or two_copy or three_and_four or decision_consistency' -v
```

Expected: null FWER test stays within its frozen simulation acceptance interval; planted pair signal is strongest on its injected edge and propagates to the containing group.

- [ ] **Step 3: Benchmark safe worker counts with QA-only B=19**

Generate identical QA configs for `n_jobs` 16, 32 and 64, set all BLAS thread variables to `1`, and record wall time and peak RSS. Select the largest worker count that does not increase peak memory beyond available RAM and is no slower than the next smaller eligible count. Result bytes and decision fields must agree across worker counts.

- [ ] **Step 4: Revalidate immutable formal inputs immediately before launch**

Run `homoeogwas validate`, rehash the copied F2143 and phenotype files, and compare BED/BIM/NPZ fingerprints with the manifest. Abort on any mismatch.

- [ ] **Step 5: Run the formal B=2000 analysis**

Run with the worker count selected in Step 3 and one BLAS thread per worker:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 uv run homoeogwas interact \
  -c results/experimental/wheat_watkins_pairwise_omnib_f2143_matched_v1/configs/interact.generated.group.omnib.yaml \
  --n-jobs 32
```

Replace `32` only with the frozen eligible worker count recorded by Step 3. Resume uses the same manifest, B and seed.

- [ ] **Step 6: Run the official audit**

Run:

```bash
uv run homoeogwas audit \
  results/experimental/wheat_watkins_pairwise_omnib_f2143_matched_v1
```

Expected: computational validity passes; the audit reports the authoritative edge family, adjusted p-values, degenerate-bootstrap count and replication status without upgrading association to mechanism.

- [ ] **Step 7: Perform independent read-only consistency checks**

Independently load JSON and ranking TSV and verify:

- ranking row count equals manifest unique-edge count;
- all and only `primary_sig=1` rows have adjusted p `<= 0.05` and raw p strictly below the bootstrap threshold;
- JSON and TSV hit identities match;
- no raw-transform row has a formal rejection field;
- AB/AD/BD direction counts equal the pre-run manifest;
- bootstrap B is 2,000, seed is 2026 and degenerate count is recorded;
- all three directions share the same null-fit and bootstrap-stream hashes.

- [ ] **Step 8: Write the biological result summary**

The summary reports sample count, group and edge counts, global FWER p, threshold, significant pairs by direction, leading genes, driving components, lambda-GC, stability/replication requirement and exact file paths. If no edge passes FWER, state that the matched analysis found no formal pair discovery while retaining the complete ranking; do not switch phenotype, family or primary layer.

- [ ] **Step 9: Run final verification before claiming completion**

Run:

```bash
git status --short
uv run pytest -q
uv run homoeogwas audit \
  results/experimental/wheat_watkins_pairwise_omnib_f2143_matched_v1
```

Expected: tests pass, audit is computationally valid and unrelated dirty files remain untouched.
