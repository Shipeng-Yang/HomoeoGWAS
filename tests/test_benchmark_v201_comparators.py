from __future__ import annotations

import numpy as np
import pytest

import homoeogwas.interact as I
import homoeogwas.omnib_family as F
from homoeogwas.group_family import (
    EdgeRecord,
    ExpandedEdgeFamily,
    MasterGroupFamily,
    expand_pair_edges,
)
from homoeogwas.interact import SubgenomeData
from scripts.benchmarks.v201 import comparators as comparator_module
from scripts.benchmarks.v201.comparators import (
    GLOBAL_VC_METHOD,
    MethodScoreBank,
    group_component_p,
    score_global_hadamard_vc,
    score_legacy_burden_product,
    score_snpxsnp_family,
)

METHODS = (
    "omnib",
    "minor_burden",
    "pc1",
    "kernel_hadamard",
    "snpxsnp",
)


def test_legacy_burden_product_is_compatibility_only_not_a_method_row():
    assert tuple(comparator_module.METHOD_NAMES) == METHODS
    assert "legacy_burden_product" not in comparator_module.METHOD_NAMES


def test_global_hadamard_vc_is_real_detection_only_and_not_local_method():
    rng = np.random.default_rng(20260830)
    n = 28
    Xa = rng.normal(size=(n, 6))
    Xb = rng.normal(size=(n, 6))
    grms = {"A": Xa @ Xa.T / Xa.shape[1], "B": Xb @ Xb.T / Xb.shape[1]}
    responses = rng.normal(size=(n, 2))
    result = score_global_hadamard_vc(
        responses,
        np.ones((n, 1)),
        grms,
        fit_kwargs={"n_starts": 1, "maxiter": 50},
    )
    assert GLOBAL_VC_METHOD not in METHODS
    assert result["method"] == GLOBAL_VC_METHOD
    assert result["hypothesis_unit"] == "global"
    assert result["detection_only"] is True
    assert len(result["p_values"]) == 2
    assert all(0.0 <= value <= 1.0 for value in result["p_values"])
    assert "group_ids" not in result
    assert "causal_recall" not in result
    assert result["kernel_manifest"]["construction"] == "hadamard_product"


def test_global_hadamard_vc_retains_single_response_failure(monkeypatch):
    calls = 0

    def compare(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("optimizer failed")
        return object()

    test = type("Boundary", (), {
        "null_model": "A+B+e", "alt_model": "A+B+hom+e",
        "ll_null": -10.0, "ll_alt": -9.0, "statistic": 2.0,
        "statistic_raw": 2.0, "df_added": 1, "p_naive": 0.1,
        "p_mixture": 0.05, "mixture_weights": {"point_mass_0": 0.5,
                                                  "chi2_df_1": 0.5},
        "added_components": ["hom"], "null_boundary_components": [],
        "is_nested": True, "clipped": False, "both_converged": True,
        "boundary_method": "self_liang_50_50", "bootstrap_p": None,
    })()
    monkeypatch.setattr(comparator_module, "compare_nested_reml", compare)
    monkeypatch.setattr(comparator_module, "boundary_lrt", lambda *args: test)
    identity = np.eye(8)
    result = score_global_hadamard_vc(
        np.ones((8, 2)), np.ones((8, 1)), {"A": identity, "B": identity}
    )
    assert result["p_values"][0] == 0.05
    assert np.isnan(result["p_values"][1])
    assert result["failed_response_indices"] == [1]
    assert result["lrt_evidence"][0]["status"] == "completed"
    assert result["lrt_evidence"][1]["status"] == "failed"


def _edge(
    edge_id: str = "AB:gA:gB",
    direction: str = "AB",
    sub_x: str = "A",
    sub_y: str = "B",
    gene_x: str = "gA",
    gene_y: str = "gB",
) -> EdgeRecord:
    return EdgeRecord(
        edge_id=edge_id,
        direction=direction,
        sub_x=sub_x,
        sub_y=sub_y,
        gene_x=gene_x,
        gene_y=gene_y,
        source_group_ids=("g0",),
    )


def _valid_bank_values() -> dict[str, np.ndarray]:
    return {name: np.ones((2, 3)) for name in METHODS}


def _valid_family_sizes() -> dict[str, int]:
    return {
        "omnib": 2,
        "minor_burden": 2,
        "pc1": 2,
        "kernel_hadamard": 2,
        "snpxsnp": 12,
    }


def test_two_copy_component_group_equals_edge():
    edge_components = np.array([[[0.01, 0.20], [0.03, 0.40], [0.05, 0.60]]])
    expanded = ExpandedEdgeFamily(edges=(_edge(),), group_edge_indices=((0,),))
    group = group_component_p(edge_components, expanded, component_index=1)
    assert np.array_equal(group, edge_components[:, 1, :])


def test_multi_edge_component_group_uses_production_acat():
    expanded = ExpandedEdgeFamily(
        edges=(
            _edge(),
            _edge("AD:gA:gD", "AD", "A", "D", "gA", "gD"),
            _edge("BD:gB:gD", "BD", "B", "D", "gB", "gD"),
        ),
        group_edge_indices=((0, 1, 2),),
    )
    components = np.array(
        [
            [[0.50, 0.50], [0.01, 0.10], [0.40, 0.40]],
            [[0.50, 0.50], [0.02, 0.20], [0.40, 0.40]],
            [[0.50, 0.50], [0.03, 0.30], [0.40, 0.40]],
        ]
    )
    actual = group_component_p(components, expanded, component_index=1)
    expected = np.array([[I.acat([0.01, 0.02, 0.03]), I.acat([0.10, 0.20, 0.30])]])
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "components,expanded,component_index,match",
    [
        (np.ones((1, 3)), ExpandedEdgeFamily((_edge(),), ((0,),)), 0, "three-dimensional"),
        (np.ones((1, 3, 2)), ExpandedEdgeFamily((), ()), 0, "edge axis"),
        (np.ones((1, 3, 2)), ExpandedEdgeFamily((_edge(),), ((1,),)), 0, "index"),
        (np.ones((1, 3, 2)), ExpandedEdgeFamily((_edge(),), ((),)), 0, "non-empty"),
        (np.ones((1, 3, 2)), ExpandedEdgeFamily((_edge(),), ((0,),)), 3, "component_index"),
    ],
)
def test_group_component_rejects_malformed_family_or_dimensions(
    components, expanded, component_index, match
):
    with pytest.raises(ValueError, match=match):
        group_component_p(components, expanded, component_index)


def test_comparators_keep_one_declared_family():
    bank = MethodScoreBank(
        family_ids=("g0", "g1"),
        p_by_method=_valid_bank_values(),
        tested_family_sizes=_valid_family_sizes(),
    )
    assert bank.family_ids == ("g0", "g1")
    assert set(bank.p_by_method) == set(METHODS)
    assert all(values.shape[0] == 2 for values in bank.p_by_method.values())
    assert np.isnan(
        MethodScoreBank(
            family_ids=("g0", "g1"),
            p_by_method={
                name: np.where(
                    np.indices((2, 3))[0] == 0,
                    np.nan,
                    np.ones((2, 3)),
                )
                for name in METHODS
            },
            tested_family_sizes=_valid_family_sizes(),
        ).p_by_method["omnib"][0]
    ).all()


def test_method_score_bank_rejects_scalar_family_id_container():
    with pytest.raises(ValueError, match="tuple"):
        MethodScoreBank("g0", _valid_bank_values(), _valid_family_sizes())


@pytest.mark.parametrize(
    "family_ids,p_values,family_sizes,match",
    [
        (("g0", "g0"), _valid_bank_values(), _valid_family_sizes(), "unique"),
        (("g0", ""), _valid_bank_values(), _valid_family_sizes(), "non-empty strings"),
        (
            ("g0", "g1"),
            {name: value for name, value in _valid_bank_values().items() if name != "snpxsnp"},
            _valid_family_sizes(),
            "method names",
        ),
        (
            ("g0", "g1"),
            {**_valid_bank_values(), "omnib": np.ones((1, 3))},
            _valid_family_sizes(),
            "row count",
        ),
        (
            ("g0", "g1"),
            {**_valid_bank_values(), "omnib": np.ones((2, 2))},
            _valid_family_sizes(),
            "response count",
        ),
        (
            ("g0", "g1"),
            {**_valid_bank_values(), "omnib": np.array([[0.1, 1.1, 0.2], [0.3, 0.4, 0.5]])},
            _valid_family_sizes(),
            r"\[0, 1\]",
        ),
        (
            ("g0", "g1"),
            {**_valid_bank_values(), "omnib": np.array([[0.1, np.inf, 0.2], [0.3, 0.4, 0.5]])},
            _valid_family_sizes(),
            "NaN",
        ),
        (
            ("g0", "g1"),
            _valid_bank_values(),
            {**_valid_family_sizes(), "snpxsnp": 0},
            "positive integers",
        ),
        (
            ("g0", "g1"),
            _valid_bank_values(),
            {**_valid_family_sizes(), "snpxsnp": True},
            "positive integers",
        ),
    ],
)
def test_method_score_bank_rejects_family_drift_and_invalid_probabilities(
    family_ids, p_values, family_sizes, match
):
    with pytest.raises(ValueError, match=match):
        MethodScoreBank(family_ids, p_values, family_sizes)


def _prepared_two_copy_fixture():
    rng = np.random.default_rng(919)
    n = 48
    subdata = {
        "A": SubgenomeData(
            X=rng.binomial(2, 0.75, size=(n, 4)).astype(float),
            gene_snp={"gA": np.arange(4)},
            samples=[f"s{i}" for i in range(n)],
            chunk=None,
        ),
        "B": SubgenomeData(
            X=rng.binomial(2, 0.25, size=(n, 4)).astype(float),
            gene_snp={"gB": np.arange(4)},
            samples=[f"s{i}" for i in range(n)],
            chunk=None,
        ),
    }
    family = MasterGroupFamily(("A", "B"), ("g0",), (("gA", "gB"),))
    scores, expanded = F._prepare_checkpoint_omnib(
        subdata,
        family,
        rng.normal(size=n),
        np.arange(n),
        cap=150,
        n_pc=3,
        transform="INT",
        bootstrap_seed=2026,
        n_jobs=1,
        grm_method="grm_from_X",
        maf_min=0.01,
        burden_maf=0.01,
        min_snp=3,
        covariates=None,
    )
    responses = np.column_stack((scores.y, rng.normal(size=n), rng.normal(size=n)))
    return scores, family, expanded, responses


def test_legacy_burden_product_matches_frozen_minor_burden_nested_term():
    scores, family, expanded, responses = _prepared_two_copy_fixture()
    _edge_p, _group_p, components = F.score_omnib_responses(
        scores, family, expanded, responses, n_jobs=1
    )
    expected = group_component_p(components, expanded, component_index=0)
    actual, tested_count = score_legacy_burden_product(
        scores, family, expanded, responses, n_jobs=1
    )
    np.testing.assert_array_equal(actual, expected)
    assert tested_count == 1


def test_legacy_burden_product_does_not_acat_drop_failed_component(monkeypatch):
    family = MasterGroupFamily(
        ("A", "B", "D"), ("g0",), (("gA", "gB", "gD"),)
    )
    expanded = expand_pair_edges(family)
    scores = _minimal_scores(12)
    scores.edge_estimable = np.ones(3, dtype=bool)
    components = np.full((3, 3, 2), 0.2, dtype=float)
    components[1, 0, 1] = np.nan
    diagnostics = F.OmniBResponseDiagnostics(
        failed_response_mask=np.array([False, True]),
        failed_response_indices=(1,),
        failed_response_indices_by_component={
            "minor_burden": (1,),
            "pc1": (),
            "kernel_hadamard": (),
        },
        nonfinite_component_counts=(0, 1),
        attempted=2,
        successful=1,
        retried=0,
        terminal_failures=1,
    )

    def scored(*_args, return_diagnostics=False, **_kwargs):
        assert return_diagnostics is True
        return (
            np.full((3, 2), 0.2),
            np.full((1, 2), 0.2),
            components,
            diagnostics,
        )

    monkeypatch.setattr(comparator_module, "score_omnib_responses", scored)
    actual, tested_count = score_legacy_burden_product(
        scores, family, expanded, np.ones((12, 2)), n_jobs=1
    )

    assert tested_count == 1
    assert np.isfinite(actual[:, 0]).all()
    assert np.isnan(actual[:, 1]).all()


def _minimal_scores(n: int) -> F.OmniBFamilyScores:
    return F.OmniBFamilyScores(
        edge_p=np.empty((1, 0)),
        group_p=np.empty((1, 0)),
        edge_components_obs=np.empty((1, 3)),
        edge_estimable=np.ones(1, dtype=bool),
        group_estimable=np.ones(1, dtype=bool),
        W=np.eye(n),
        y=np.zeros(n),
        covariance_components={"e": 1.0},
        null_design=np.ones((n, 1)),
    )


def _raw_snpxsnp_p(
    scores: F.OmniBFamilyScores,
    left: np.ndarray,
    right: np.ndarray,
    responses: np.ndarray,
) -> np.ndarray:
    yw = scores.W @ responses
    cw = scores.W @ scores.null_design
    rows = []
    for left_index in range(left.shape[1]):
        for right_index in range(right.shape[1]):
            x = left[:, left_index]
            y = right[:, right_index]
            reduced = np.column_stack((cw, scores.W @ x, scores.W @ y))
            added = scores.W @ (x * y)[:, None]
            rows.append(
                I._batch_nested_f(
                    yw, reduced, added, response_axis_stable=True
                )
            )
    return np.asarray(rows)


def test_snpxsnp_returns_raw_complete_pair_family_minimum():
    rng = np.random.default_rng(882)
    n = 48
    left = rng.binomial(2, 0.25, size=(n, 2)).astype(float)
    right = rng.binomial(2, 0.35, size=(n, 3)).astype(float)
    responses = rng.normal(size=(n, 2))
    family = MasterGroupFamily(("A", "B"), ("g0",), (("gA", "gB"),))
    expanded = expand_pair_edges(family)
    scores = _minimal_scores(n)
    scores.gated_snp = {
        ("A", "gA"): np.arange(left.shape[1]),
        ("B", "gB"): np.arange(right.shape[1]),
    }

    result = score_snpxsnp_family(
        scores,
        family,
        expanded,
        {("A", "gA"): left, ("B", "gB"): right},
        responses,
        max_offered_pairs=6,
    )

    target_p = _raw_snpxsnp_p(scores, left, right, responses)
    expected = np.min(target_p, axis=0, keepdims=True)

    np.testing.assert_array_equal(result.group_p, expected)
    assert result.tested_pair_count == left.shape[1] * right.shape[1] == 6

    reordered = score_snpxsnp_family(
        scores,
        family,
        expanded,
        {("A", "gA"): left, ("B", "gB"): right},
        responses[:, ::-1],
        max_offered_pairs=6,
    )
    np.testing.assert_array_equal(reordered.group_p, result.group_p[:, ::-1])
    assert reordered.tested_pair_count == result.tested_pair_count


def test_snpxsnp_rejects_missing_blocks_and_invalid_pair_ceiling():
    rng = np.random.default_rng(44)
    n = 32
    family = MasterGroupFamily(("A", "B"), ("g0",), (("gA", "gB"),))
    expanded = expand_pair_edges(family)
    scores = _minimal_scores(n)
    responses = rng.normal(size=(n, 2))
    blocks = {("A", "gA"): rng.binomial(2, 0.2, size=(n, 2)).astype(float)}
    with pytest.raises(ValueError, match="missing genotype block"):
        score_snpxsnp_family(
            scores,
            family,
            expanded,
            blocks,
            responses,
            max_offered_pairs=4,
        )
    with pytest.raises(ValueError, match="positive integer"):
        score_snpxsnp_family(
            scores,
            family,
            expanded,
            {
                **blocks,
                ("B", "gB"): rng.binomial(2, 0.3, size=(n, 2)).astype(float),
            },
            responses,
            max_offered_pairs=0,
        )
