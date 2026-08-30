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
from scripts.benchmarks.v201.comparators import (
    MethodScoreBank,
    group_component_p,
    score_legacy_burden_product,
    score_snpxsnp_family,
)

METHODS = (
    "omnib",
    "minor_burden",
    "pc1",
    "kernel_hadamard",
    "legacy_burden_product",
    "snpxsnp",
)


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
        "legacy_burden_product": 2,
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
            rows.append(I._batch_nested_f(yw, reduced, added))
    return np.asarray(rows)


def test_snpxsnp_calibrates_complete_pair_family_before_group_score():
    rng = np.random.default_rng(882)
    n = 48
    left = rng.binomial(2, 0.25, size=(n, 2)).astype(float)
    right = rng.binomial(2, 0.35, size=(n, 3)).astype(float)
    responses = rng.normal(size=(n, 2))
    calibration_responses = rng.normal(size=(n, 39))
    family = MasterGroupFamily(("A", "B"), ("g0",), (("gA", "gB"),))
    expanded = expand_pair_edges(family)
    scores = _minimal_scores(n)

    actual, tested_count = score_snpxsnp_family(
        scores,
        family,
        expanded,
        {("A", "gA"): left, ("B", "gB"): right},
        responses,
        calibration_responses=calibration_responses,
    )

    target_p = _raw_snpxsnp_p(scores, left, right, responses)
    calibration_p = _raw_snpxsnp_p(scores, left, right, calibration_responses)
    expected = np.array(
        [
            min(
                F.bootstrap_minp_calibration(
                    target_p[:, column], calibration_p
                )["adjusted_p_local"]
            )
            for column in range(responses.shape[1])
        ]
    )[None, :]
    best_pair = int(np.argmin(target_p[:, 0]))
    selected_pair_only = (
        1 + np.sum(calibration_p[best_pair] <= target_p[best_pair, 0])
    ) / (calibration_p.shape[1] + 1)

    np.testing.assert_array_equal(actual, expected)
    assert actual[0, 0] != selected_pair_only
    assert tested_count == left.shape[1] * right.shape[1] == 6

    reordered, reordered_count = score_snpxsnp_family(
        scores,
        family,
        expanded,
        {("A", "gA"): left, ("B", "gB"): right},
        responses[:, ::-1],
        calibration_responses=calibration_responses,
    )
    np.testing.assert_array_equal(reordered, actual[:, ::-1])
    assert reordered_count == tested_count


def test_snpxsnp_rejects_missing_blocks_and_empty_calibration_bank():
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
            calibration_responses=rng.normal(size=(n, 19)),
        )
    with pytest.raises(ValueError, match="at least one"):
        score_snpxsnp_family(
            scores,
            family,
            expanded,
            {
                **blocks,
                ("B", "gB"): rng.binomial(2, 0.3, size=(n, 2)).astype(float),
            },
            responses,
            calibration_responses=np.empty((n, 0)),
        )
