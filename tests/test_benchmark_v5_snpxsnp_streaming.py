from __future__ import annotations

import numpy as np
import pytest

import homoeogwas.interact as interact_module
import homoeogwas.omnib_family as omnib_module
from homoeogwas.group_family import MasterGroupFamily, expand_pair_edges
from scripts.benchmarks.v201 import comparators as comparator_module
from scripts.benchmarks.v201.track_omnib import apply_threshold, empirical_threshold


def _scores(n: int, *, left_ids: tuple[int, ...], right_ids: tuple[int, ...]):
    return omnib_module.OmniBFamilyScores(
        edge_p=np.empty((1, 0)),
        group_p=np.empty((1, 0)),
        edge_components_obs=np.empty((1, 3)),
        edge_estimable=np.ones(1, dtype=bool),
        group_estimable=np.ones(1, dtype=bool),
        W=np.eye(n),
        y=np.zeros(n),
        covariance_components={"e": 1.0},
        null_design=np.ones((n, 1)),
        gated_snp={
            ("A", "gA"): np.asarray(left_ids, dtype=int),
            ("B", "gB"): np.asarray(right_ids, dtype=int),
        },
    )


def _family():
    family = MasterGroupFamily(("A", "B"), ("g0",), (("gA", "gB"),))
    return family, expand_pair_edges(family)


def test_streamed_raw_score_matches_direct_nested_f_and_counts_skips():
    rng = np.random.default_rng(9201)
    n = 40
    left = np.column_stack((rng.binomial(2, 0.3, n), np.zeros(n))).astype(float)
    right = np.column_stack(
        (rng.binomial(2, 0.4, n), rng.binomial(2, 0.2, n))
    ).astype(float)
    responses = rng.normal(size=(n, 3))
    scores = _scores(n, left_ids=(11, 12), right_ids=(21, 22))
    family, expanded = _family()

    result = comparator_module.score_snpxsnp_family(
        scores,
        family,
        expanded,
        {("A", "gA"): left, ("B", "gB"): right},
        responses,
        max_offered_pairs=4,
    )

    direct = []
    for right_index in range(2):
        design = comparator_module._nested_snp_product_design(
            scores.W,
            scores.W @ scores.null_design,
            left[:, 0],
            right[:, right_index],
        )
        assert design is not None
        direct.append(
            interact_module._batch_nested_f(
                scores.W @ responses, design[0], design[1]
            )
        )
    expected = np.min(np.vstack(direct), axis=0, keepdims=True)

    np.testing.assert_allclose(result.group_p, expected)
    assert result.offered_pair_count == 4
    assert result.design_nonestimable_pair_count == 2
    assert result.tested_pair_count == 2
    assert result.offered_pair_count_by_group == (4,)
    assert result.design_nonestimable_pair_count_by_group == (2,)
    assert result.tested_pair_count_by_group == (2,)
    assert result.member_ids == (
        f"{expanded.edges[0].edge_id}|11|21",
        f"{expanded.edges[0].edge_id}|11|22",
    )
    assert len(result.member_family_sha256) == 64
    assert result.failed_response_indices == ()


def test_exact_tie_chooses_lexicographically_smallest_pair_id():
    rng = np.random.default_rng(9202)
    n = 40
    marker = rng.binomial(2, 0.3, n).astype(float)
    left = np.column_stack((marker, marker))
    right = rng.binomial(2, 0.4, size=(n, 1)).astype(float)
    responses = rng.normal(size=(n, 2))
    scores = _scores(n, left_ids=(9, 3), right_ids=(8,))
    family, expanded = _family()

    result = comparator_module.score_snpxsnp_family(
        scores,
        family,
        expanded,
        {("A", "gA"): left, ("B", "gB"): right},
        responses,
        max_offered_pairs=2,
    )

    expected_id = f"{expanded.edges[0].edge_id}|3|8"
    assert result.member_ids == (
        f"{expanded.edges[0].edge_id}|9|8",
        expected_id,
    )
    assert all(
        result.member_ids[index] == expected_id
        for index in result.argmin_member_index[0]
    )


def test_pair_ceiling_aborts_before_any_nested_f_score(monkeypatch):
    rng = np.random.default_rng(9203)
    n = 32
    left = rng.binomial(2, 0.3, size=(n, 2)).astype(float)
    right = rng.binomial(2, 0.4, size=(n, 3)).astype(float)
    scores = _scores(n, left_ids=(1, 2), right_ids=(3, 4, 5))
    family, expanded = _family()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("nested F scoring started before the pair ceiling gate")

    monkeypatch.setattr(interact_module, "_batch_nested_f", forbidden)
    with pytest.raises(ValueError, match="offered pair ceiling"):
        comparator_module.score_snpxsnp_family(
            scores,
            family,
            expanded,
            {("A", "gA"): left, ("B", "gB"): right},
            rng.normal(size=(n, 2)),
            max_offered_pairs=5,
        )


def test_scorer_never_materializes_a_python_list_of_pair_rows(monkeypatch):
    rng = np.random.default_rng(9204)
    n = 36
    left = rng.binomial(2, 0.3, size=(n, 2)).astype(float)
    right = rng.binomial(2, 0.4, size=(n, 2)).astype(float)
    scores = _scores(n, left_ids=(1, 2), right_ids=(3, 4))
    family, expanded = _family()
    original_asarray = comparator_module.np.asarray

    def reject_pair_rows(value, *args, **kwargs):
        if (
            isinstance(value, list)
            and len(value) > 1
            and all(isinstance(row, np.ndarray) and row.ndim == 1 for row in value)
        ):
            raise AssertionError("pair-by-response rows were materialized")
        return original_asarray(value, *args, **kwargs)

    monkeypatch.setattr(comparator_module.np, "asarray", reject_pair_rows)
    result = comparator_module.score_snpxsnp_family(
        scores,
        family,
        expanded,
        {("A", "gA"): left, ("B", "gB"): right},
        rng.normal(size=(n, 2)),
        max_offered_pairs=4,
    )
    assert result.group_p.shape == (1, 2)


def test_nonfinite_pair_score_marks_the_whole_response_failed():
    rng = np.random.default_rng(9205)
    n = 36
    left = rng.binomial(2, 0.3, size=(n, 1)).astype(float)
    right = rng.binomial(2, 0.4, size=(n, 1)).astype(float)
    responses = np.column_stack((rng.normal(size=n), np.zeros(n)))
    scores = _scores(n, left_ids=(1,), right_ids=(2,))
    family, expanded = _family()

    result = comparator_module.score_snpxsnp_family(
        scores,
        family,
        expanded,
        {("A", "gA"): left, ("B", "gB"): right},
        responses,
        max_offered_pairs=1,
    )

    assert np.isfinite(result.group_p[0, 0])
    assert np.isnan(result.group_p[:, 1]).all()
    assert result.failed_response_indices == (1,)
    assert result.nonfinite_pair_score_count == 1


def test_inner_calibration_arguments_are_not_part_of_the_raw_api():
    rng = np.random.default_rng(9206)
    n = 32
    left = rng.binomial(2, 0.3, size=(n, 1)).astype(float)
    right = rng.binomial(2, 0.4, size=(n, 1)).astype(float)
    scores = _scores(n, left_ids=(1,), right_ids=(2,))
    family, expanded = _family()

    with pytest.raises(TypeError, match="calibration_responses"):
        comparator_module.score_snpxsnp_family(
            scores,
            family,
            expanded,
            {("A", "gA"): left, ("B", "gB"): right},
            rng.normal(size=(n, 1)),
            calibration_responses=rng.normal(size=(n, 3)),
        )


def test_raw_outer_decisions_match_legacy_inner_then_outer_fixture():
    """One-member fixture proves the removed nesting preserves decisions."""
    rng = np.random.default_rng(9207)
    n = 40
    left = rng.binomial(2, 0.3, size=(n, 1)).astype(float)
    right = rng.binomial(2, 0.4, size=(n, 1)).astype(float)
    calibration_responses = rng.normal(size=(n, 39))
    target_responses = rng.normal(size=(n, 11))
    scores = _scores(n, left_ids=(1,), right_ids=(2,))
    family, expanded = _family()
    blocks = {("A", "gA"): left, ("B", "gB"): right}

    raw_calibration = comparator_module.score_snpxsnp_family(
        scores,
        family,
        expanded,
        blocks,
        calibration_responses,
        max_offered_pairs=1,
    ).group_p
    raw_target = comparator_module.score_snpxsnp_family(
        scores,
        family,
        expanded,
        blocks,
        target_responses,
        max_offered_pairs=1,
    ).group_p

    def legacy_inner_adjust(raw: np.ndarray) -> np.ndarray:
        adjusted = np.empty_like(raw)
        for response_index in range(raw.shape[1]):
            result = omnib_module.bootstrap_minp_calibration(
                raw[:, response_index], raw_calibration, alpha=0.0
            )
            adjusted[:, response_index] = result["adjusted_p_local"]
        return adjusted

    legacy_calibration = legacy_inner_adjust(raw_calibration)
    legacy_target = legacy_inner_adjust(raw_target)
    raw_threshold = empirical_threshold(raw_calibration, alpha=0.10)
    legacy_threshold = empirical_threshold(legacy_calibration, alpha=0.10)

    np.testing.assert_array_equal(
        apply_threshold(raw_target, raw_threshold),
        apply_threshold(legacy_target, legacy_threshold),
    )
