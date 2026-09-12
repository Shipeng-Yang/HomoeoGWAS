from __future__ import annotations

import numpy as np
import pytest

import homoeogwas.interact as interact_module
import homoeogwas.omnib_family as omnib_module
from homoeogwas.group_family import MasterGroupFamily, expand_pair_edges
from scripts.benchmarks.v201 import comparators as comparator_module
from scripts.benchmarks.v201.contracts import sha256_payload
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


def _three_copy_shared_edge_fixture():
    rng = np.random.default_rng(9210)
    n = 56
    rotation, _ = np.linalg.qr(rng.normal(size=(n, n)))
    W = (rotation * np.linspace(0.7, 1.3, n)) @ rotation.T
    a_marker = rng.binomial(2, 0.31, n).astype(float)
    blocks = {
        ("A", "a0"): np.column_stack((a_marker, a_marker)),
        ("B", "b0"): rng.binomial(2, (0.37, 0.22), size=(n, 2)).astype(float),
        ("D", "d0"): rng.binomial(2, 0.28, size=(n, 1)).astype(float),
        ("D", "d1"): rng.binomial(2, 0.43, size=(n, 1)).astype(float),
    }
    family = MasterGroupFamily(
        ("A", "B", "D"),
        ("g0", "g1"),
        (("a0", "b0", "d0"), ("a0", "b0", "d1")),
    )
    expanded = expand_pair_edges(family)
    scores = omnib_module.OmniBFamilyScores(
        edge_p=np.empty((len(expanded.edges), 0)),
        group_p=np.empty((len(family.group_ids), 0)),
        edge_components_obs=np.empty((len(expanded.edges), 3)),
        edge_estimable=np.ones(len(expanded.edges), dtype=bool),
        group_estimable=np.ones(len(family.group_ids), dtype=bool),
        W=W,
        y=np.zeros(n),
        covariance_components={"e": 1.0},
        null_design=np.ones((n, 1)),
        gated_snp={
            ("A", "a0"): np.asarray((9, 3), dtype=int),
            ("B", "b0"): np.asarray((8, 2), dtype=int),
            ("D", "d0"): np.asarray((7,), dtype=int),
            ("D", "d1"): np.asarray((6,), dtype=int),
        },
    )
    signal_responses = np.column_stack(
        (
            blocks[("A", "a0")][:, 0] * blocks[("B", "b0")][:, 0],
            blocks[("A", "a0")][:, 0] * blocks[("D", "d1")][:, 0],
            blocks[("B", "b0")][:, 1] * blocks[("D", "d0")][:, 0],
        )
    ) + rng.normal(scale=0.01, size=(n, 3))
    responses = np.column_stack((signal_responses, rng.normal(size=(n, 28))))
    return scores, family, expanded, blocks, responses


def test_shared_input_binding_is_readonly_and_preserves_scorer_evidence(monkeypatch):
    scores, family, expanded, blocks, responses = _three_copy_shared_edge_fixture()
    before_blocks = {key: value.copy() for key, value in blocks.items()}
    before_columns = {key: value.copy() for key, value in scores.gated_snp.items()}
    bind_inputs = getattr(comparator_module, "_bind_snpxsnp_inputs", None)
    assert callable(bind_inputs), "producer and scorer need one shared input binding helper"

    def forbidden(*_args, **_kwargs):
        raise AssertionError("input binding must not construct pairs or score responses")

    with monkeypatch.context() as guard:
        guard.setattr(comparator_module, "_nested_snp_product_design", forbidden)
        guard.setattr(comparator_module, "_whiten_columns", forbidden)
        guard.setattr(interact_module, "_batch_nested_f", forbidden)
        inputs = bind_inputs(scores, family, expanded, blocks)
    assert inputs.family_sha256 == (
        "27d541aa9ba8e686b2d67d5a7cbd03a13391b1c42f07e23774548dd15e5dad2a"
    )
    for key, value in before_blocks.items():
        np.testing.assert_array_equal(blocks[key], value)
        np.testing.assert_array_equal(inputs.blocks[key], value)
        np.testing.assert_array_equal(scores.gated_snp[key], before_columns[key])
        np.testing.assert_array_equal(inputs.source_columns[key], before_columns[key])

    result = comparator_module.score_snpxsnp_family(
        scores, family, expanded, blocks, responses, max_offered_pairs=12,
    )
    assert result.input_family_sha256 == inputs.family_sha256
    assert tuple(dict(value) for value in result.input_block_bindings) == inputs.bindings
    # Recorded from the unchanged scorer before factoring its input validation.
    # This binds the full raw evidence, including memberships, argmins and counts.
    assert sha256_payload(result.evidence_payload()) == (
        "f3c04503624c4b97aa1735932fad2e110afa4a095557d23b031c178900cd2c4b"
    )


@pytest.mark.parametrize("mutation", ["missing_block", "extra_block", "nan", "columns"])
def test_shared_input_binding_and_scorer_reject_the_same_invalid_inputs(mutation):
    scores, family, expanded, blocks, responses = _three_copy_shared_edge_fixture()
    if mutation == "missing_block":
        del blocks[("A", "a0")]
    elif mutation == "extra_block":
        blocks[("A", "extra")] = np.zeros((56, 1))
    elif mutation == "nan":
        blocks[("A", "a0")][0, 0] = np.nan
    else:
        blocks[("A", "a0")] = blocks[("A", "a0")][:, :1]
    with pytest.raises(ValueError) as helper_error:
        comparator_module._bind_snpxsnp_inputs(scores, family, expanded, blocks)
    with pytest.raises(ValueError) as scorer_error:
        comparator_module.score_snpxsnp_family(
            scores, family, expanded, blocks, responses, max_offered_pairs=12,
        )
    assert str(helper_error.value) == str(scorer_error.value)


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
                scores.W @ responses,
                design[0],
                design[1],
                response_axis_stable=True,
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


def test_missing_gated_snp_identity_fails_closed():
    rng = np.random.default_rng(9208)
    n = 32
    left = rng.binomial(2, 0.3, size=(n, 1)).astype(float)
    right = rng.binomial(2, 0.4, size=(n, 1)).astype(float)
    scores = _scores(n, left_ids=(1,), right_ids=(2,))
    scores.gated_snp.pop(("B", "gB"))
    family, expanded = _family()

    with pytest.raises(ValueError, match="missing gated SNP indices"):
        comparator_module.score_snpxsnp_family(
            scores,
            family,
            expanded,
            {("A", "gA"): left, ("B", "gB"): right},
            rng.normal(size=(n, 2)),
            max_offered_pairs=1,
        )


def test_dosage_bytes_are_bound_separately_from_member_family_identity():
    scores, family, expanded, blocks, responses = _three_copy_shared_edge_fixture()
    first = comparator_module.score_snpxsnp_family(
        scores, family, expanded, blocks, responses[:, :2], max_offered_pairs=12,
    )
    changed_blocks = {key: value.copy() for key, value in blocks.items()}
    changed_blocks[("D", "d1")][0, 0] = (
        2.0 - changed_blocks[("D", "d1")][0, 0]
    )
    second = comparator_module.score_snpxsnp_family(
        scores,
        family,
        expanded,
        changed_blocks,
        responses[:, :2],
        max_offered_pairs=12,
    )

    assert first.member_family_sha256 == second.member_family_sha256
    assert first.input_family_sha256 != second.input_family_sha256
    first_bindings = {
        (record["subgenome"], record["gene_id"]): record["binding_sha256"]
        for record in first.input_block_bindings
    }
    second_bindings = {
        (record["subgenome"], record["gene_id"]): record["binding_sha256"]
        for record in second.input_block_bindings
    }
    assert first_bindings[("D", "d1")] != second_bindings[("D", "d1")]
    assert {
        key for key in first_bindings if first_bindings[key] != second_bindings[key]
    } == {("D", "d1")}


def test_shared_edge_nonestimable_and_nonfinite_counts_are_membership_aware(
    monkeypatch,
):
    scores, family, expanded, blocks, responses = _three_copy_shared_edge_fixture()
    original_design = comparator_module._nested_snp_product_design
    design_calls = 0

    def first_pair_nonestimable(*args):
        nonlocal design_calls
        design_calls += 1
        if design_calls == 1:
            return None
        return original_design(*args)

    original_score = interact_module._batch_nested_f
    score_calls = 0

    def first_tested_pair_has_one_nonfinite(*args, **kwargs):
        nonlocal score_calls
        score_calls += 1
        values = original_score(*args, **kwargs)
        if score_calls == 1:
            values = values.copy()
            values[1] = np.nan
        return values

    monkeypatch.setattr(
        comparator_module, "_nested_snp_product_design", first_pair_nonestimable,
    )
    monkeypatch.setattr(
        interact_module, "_batch_nested_f", first_tested_pair_has_one_nonfinite,
    )
    result = comparator_module.score_snpxsnp_family(
        scores, family, expanded, blocks, responses[:, :3], max_offered_pairs=12,
    )

    assert result.offered_pair_count == 12
    assert result.design_nonestimable_pair_count == 1
    assert result.tested_pair_count == 11
    assert result.offered_pair_count_by_group == (8, 8)
    assert result.design_nonestimable_pair_count_by_group == (1, 1)
    assert result.tested_pair_count_by_group == (7, 7)
    assert result.nonfinite_pair_score_count == 1
    assert result.nonfinite_pair_score_count_by_group == (1, 1)
    assert result.failed_response_indices == (1,)
    assert np.isnan(result.group_p[:, 1]).all()


def test_three_copy_shared_edge_family_counts_minima_ties_and_response_invariance():
    scores, family, expanded, blocks, responses = _three_copy_shared_edge_fixture()
    assert not np.allclose(scores.W, np.eye(scores.W.shape[0]))
    assert expanded.group_edge_indices == ((0, 1, 2), (0, 3, 4))
    assert expanded.edges[0].source_group_ids == ("g0", "g1")

    result = comparator_module.score_snpxsnp_family(
        scores,
        family,
        expanded,
        blocks,
        responses,
        max_offered_pairs=12,
    )

    assert result.offered_pair_count == 12
    assert result.design_nonestimable_pair_count == 0
    assert result.tested_pair_count == 12
    assert result.offered_pair_count_by_group == (8, 8)
    assert result.design_nonestimable_pair_count_by_group == (0, 0)
    assert result.tested_pair_count_by_group == (8, 8)
    assert result.group_memberships[:4] == ((0, 1),) * 4
    assert result.member_family_sha256 == (
        "663d054a4d04a5252d62dd2cbafc61d7d018f5e8d0bc8007e7faab4d2e72f75d"
    )

    expected_p = np.full_like(result.group_p, np.inf)
    expected_ids = np.full(result.group_p.shape, "", dtype=object)
    for edge_index, edge in enumerate(expanded.edges):
        memberships = tuple(
            group_index
            for group_index, edge_indices in enumerate(expanded.group_edge_indices)
            if edge_index in edge_indices
        )
        left_key = (edge.sub_x, edge.gene_x)
        right_key = (edge.sub_y, edge.gene_y)
        left = blocks[left_key]
        right = blocks[right_key]
        for left_index, left_id in enumerate(scores.gated_snp[left_key]):
            for right_index, right_id in enumerate(scores.gated_snp[right_key]):
                design = comparator_module._nested_snp_product_design(
                    scores.W,
                    scores.W @ scores.null_design,
                    left[:, left_index],
                    right[:, right_index],
                )
                assert design is not None
                pair_p = interact_module._batch_nested_f(
                    scores.W @ responses,
                    design[0],
                    design[1],
                    response_axis_stable=True,
                )
                member_id = f"{edge.edge_id}|{int(left_id)}|{int(right_id)}"
                for group_index in memberships:
                    for response_index, value in enumerate(pair_p):
                        if value < expected_p[group_index, response_index] or (
                            value == expected_p[group_index, response_index]
                            and member_id < expected_ids[group_index, response_index]
                        ):
                            expected_p[group_index, response_index] = value
                            expected_ids[group_index, response_index] = member_id

    observed_ids = np.asarray(result.member_ids, dtype=object)[
        result.argmin_member_index
    ]
    np.testing.assert_allclose(result.group_p, expected_p)
    np.testing.assert_array_equal(observed_ids, expected_ids)
    assert observed_ids[0, 0] == f"{expanded.edges[0].edge_id}|3|8"
    assert observed_ids[1, 1] == f"{expanded.edges[3].edge_id}|3|6"
    assert observed_ids[0, 2] == f"{expanded.edges[2].edge_id}|2|7"

    chunks = [
        comparator_module.score_snpxsnp_family(
            scores,
            family,
            expanded,
            blocks,
            responses[:, start:stop],
            max_offered_pairs=12,
        )
        for start, stop in ((0, 7), (7, 26), (26, 31))
    ]
    for chunk in chunks:
        assert chunk.member_ids == result.member_ids
        assert chunk.group_memberships == result.group_memberships
        assert chunk.member_family_sha256 == result.member_family_sha256
    np.testing.assert_array_equal(
        np.concatenate([chunk.group_p for chunk in chunks], axis=1), result.group_p
    )
    np.testing.assert_array_equal(
        np.concatenate([chunk.argmin_member_index for chunk in chunks], axis=1),
        result.argmin_member_index,
    )

    reversed_result = comparator_module.score_snpxsnp_family(
        scores,
        family,
        expanded,
        blocks,
        responses[:, ::-1],
        max_offered_pairs=12,
    )
    assert reversed_result.member_ids == result.member_ids
    assert reversed_result.group_memberships == result.group_memberships
    assert reversed_result.member_family_sha256 == result.member_family_sha256
    np.testing.assert_array_equal(reversed_result.group_p[:, ::-1], result.group_p)
    np.testing.assert_array_equal(
        reversed_result.argmin_member_index[:, ::-1], result.argmin_member_index
    )


def test_width_one_response_is_bitwise_identical_inside_wide_raw_bank():
    scores, family, expanded, blocks, responses = _three_copy_shared_edge_fixture()

    alone = comparator_module.score_snpxsnp_family(
        scores,
        family,
        expanded,
        blocks,
        np.ascontiguousarray(responses[:, :1]),
        max_offered_pairs=12,
    )
    embedded = comparator_module.score_snpxsnp_family(
        scores,
        family,
        expanded,
        blocks,
        responses,
        max_offered_pairs=12,
    )

    np.testing.assert_array_equal(alone.group_p[:, 0], embedded.group_p[:, 0])
    np.testing.assert_array_equal(
        alone.argmin_member_index[:, 0], embedded.argmin_member_index[:, 0]
    )


def test_snpxsnp_production_explicitly_requests_stable_response_axis(monkeypatch):
    scores, family, expanded, blocks, responses = _three_copy_shared_edge_fixture()
    original = interact_module._batch_nested_f
    stable_flags = []
    response_widths = []

    def recording_score(*args, **kwargs):
        stable_flags.append(kwargs.get("response_axis_stable"))
        response_widths.append(args[0].shape[1])
        return original(*args, **kwargs)

    monkeypatch.setattr(interact_module, "_batch_nested_f", recording_score)
    comparator_module.score_snpxsnp_family(
        scores,
        family,
        expanded,
        blocks,
        responses[:, :2],
        max_offered_pairs=12,
    )

    assert stable_flags
    assert all(flag is True for flag in stable_flags)
    assert set(response_widths) == {omnib_module.INDEXED_SCORE_MICROBLOCK}


def test_one_member_fixture_is_only_a_legacy_compatibility_limit():
    """One tested member is the narrow limit where both calibrations agree."""
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
    assert raw_calibration.shape[0] == 1
    assert raw_target.shape[0] == 1

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
