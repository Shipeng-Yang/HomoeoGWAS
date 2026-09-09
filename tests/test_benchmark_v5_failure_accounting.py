"""Explicit response-failure and conservative decision contracts."""

from copy import deepcopy

import numpy as np
import pytest

import homoeogwas.omnib_family as F
import scripts.benchmarks.v201.track_omnib as T
from homoeogwas.group_family import MasterGroupFamily
from homoeogwas.interact import SubgenomeData
from scripts.benchmarks.v201.audit import (
    BenchmarkAuditError,
    _audit_end2end_decisions,
    _audit_robustness_record,
    core_fwer_gate,
)
from scripts.benchmarks.v201.track_omnib import (
    apply_failure_policy,
    build_synthetic_omnib_context,
    empirical_threshold,
    run_conditional_bank,
    run_end_to_end_null,
)


def _fixture():
    rng = np.random.default_rng(950)
    n = 44
    subdata = {}
    for sub in ("A", "D"):
        subdata[sub] = SubgenomeData(
            X=rng.integers(0, 3, size=(n, 6)).astype(float),
            gene_snp={"g0": np.arange(6)},
            samples=[f"s{i}" for i in range(n)],
            chunk=None,
        )
    family = MasterGroupFamily(("A", "D"), ("g",), (("g0", "g0"),))
    phenotype = rng.normal(size=n)
    scores, expanded = F.prepare_omnib_design(
        subdata,
        family,
        phenotype,
        np.arange(n),
        transform="INT",
        feature_seed=17,
        retained_variant_masks={sub: np.ones(6, bool) for sub in subdata},
        grm_method="grm_from_X",
        maf_min=0.01,
        burden_maf=0.01,
        min_snp=3,
        cap=150,
        n_pc=3,
    )
    return subdata, family, phenotype, scores, expanded


def test_nonfinite_fixed_component_marks_the_whole_response_failed(monkeypatch):
    _subdata, family, _phenotype, scores, expanded = _fixture()
    original = F._prepared_components_over_Y

    def fail_second(Yw, prepared):
        result = original(Yw, prepared)
        result[:, 1] = np.nan
        return result

    monkeypatch.setattr(F, "_prepared_components_over_Y", fail_second)
    edge, group, components, diagnostics = F.score_omnib_responses(
        scores,
        family,
        expanded,
        np.column_stack((scores.y, scores.y[::-1])),
        n_jobs=1,
        return_diagnostics=True,
    )

    assert edge.shape == group.shape == (1, 2)
    assert components.shape == (1, 3, 2)
    assert diagnostics.failed_response_mask.tolist() == [False, True]
    assert diagnostics.failed_response_indices == (1,)
    assert diagnostics.attempted == 2
    assert diagnostics.successful == 1
    assert diagnostics.terminal_failures == 1


def test_failure_policy_is_worst_case_for_null_and_power():
    failed = np.array([False, True])
    np.testing.assert_array_equal(
        apply_failure_policy(
            np.array([False, False]), failed, response_role="heldout"
        ),
        np.array([False, True]),
    )
    np.testing.assert_array_equal(
        apply_failure_policy(
            np.array([False, False]), failed, response_role="calibration"
        ),
        np.array([False, True]),
    )
    np.testing.assert_array_equal(
        apply_failure_policy(
            np.array([True, True]), failed, response_role="power"
        ),
        np.array([True, False]),
    )


def test_native_observed_response_failure_aborts(monkeypatch):
    _subdata, family, _phenotype, scores, expanded = _fixture()

    def nonfinite(Yw, prepared):
        return np.full((len(prepared), Yw.shape[1]), np.nan)

    monkeypatch.setattr(F, "_prepared_components_over_Y", nonfinite)
    with pytest.raises(RuntimeError, match="native observed.*failed"):
        F.score_omnib_observed(scores, family, expanded, n_jobs=1)


def test_score_omnib_family_partial_observed_component_failure_aborts(monkeypatch):
    subdata, family, phenotype, _scores, _expanded = _fixture()
    original = F._prepared_components_over_Y

    def fail_one_observed_component(Yw, prepared):
        result = original(Yw, prepared)
        result[0, 0] = np.nan
        return result

    monkeypatch.setattr(
        F, "_prepared_components_over_Y", fail_one_observed_component)
    with pytest.raises(RuntimeError, match="native observed.*failed"):
        F.score_omnib_family(
            subdata,
            family,
            phenotype,
            np.arange(phenotype.size),
            transform="INT",
            feature_seed=17,
            retained_variant_masks={sub: np.ones(6, bool) for sub in subdata},
            bootstrap_B=0,
            n_jobs=1,
            grm_method="grm_from_X",
            min_snp=3,
        )


def test_native_bootstrap_failure_is_retained_as_degenerate(monkeypatch):
    subdata, family, phenotype, _scores, _expanded = _fixture()
    original = F._prepared_components_over_Y

    def fail_bootstrap(Yw, prepared):
        result = original(Yw, prepared)
        result[:, 1] = np.nan
        return result

    monkeypatch.setattr(F, "_prepared_components_over_Y", fail_bootstrap)
    scores, _ = F.score_omnib_family(
        subdata,
        family,
        phenotype,
        np.arange(phenotype.size),
        transform="INT",
        feature_seed=17,
        retained_variant_masks={sub: np.ones(6, bool) for sub in subdata},
        bootstrap_B=1,
        bootstrap_seed=29,
        n_jobs=1,
        grm_method="grm_from_X",
        min_snp=3,
    )
    assert scores.response_diagnostics.failed_response_indices == (1,)
    calibration = F.bootstrap_minp_calibration(
        scores.group_p[:, 0], scores.group_p[:, 1:]
    )
    assert calibration["n_degenerate_replicates"] == 1
    assert calibration["degenerate_policy"] == (
        "any_nonfinite_statistic_sets_null_min_to_zero"
    )


def test_partial_bootstrap_component_failure_forces_null_minimum_zero(monkeypatch):
    subdata, family, phenotype, _scores, _expanded = _fixture()
    original = F._prepared_components_over_Y

    def fail_one_bootstrap_component(Yw, prepared):
        result = original(Yw, prepared)
        result[0, 1] = np.nan
        return result

    monkeypatch.setattr(
        F, "_prepared_components_over_Y", fail_one_bootstrap_component)
    scores, _ = F.score_omnib_family(
        subdata,
        family,
        phenotype,
        np.arange(phenotype.size),
        transform="INT",
        feature_seed=17,
        retained_variant_masks={sub: np.ones(6, bool) for sub in subdata},
        bootstrap_B=1,
        bootstrap_seed=29,
        n_jobs=1,
        grm_method="grm_from_X",
        min_snp=3,
    )

    assert scores.response_diagnostics.failed_response_indices == (1,)
    assert np.isnan(scores.group_p[:, 1]).all()
    calibration = F.bootstrap_minp_calibration(
        scores.group_p[:, 0], scores.group_p[:, 1:], alpha=0.5
    )
    assert calibration["n_degenerate_replicates"] == 1
    assert calibration["threshold"] == 0.0


def test_checkpoint_partial_component_failures_are_persisted_and_reported(
    monkeypatch, tmp_path,
):
    subdata, family, phenotype, _scores, _expanded = _fixture()
    original = F._prepared_components_over_Y
    calls = {"count": 0}

    def fail_each_checkpoint_bootstrap(Yw, prepared):
        result = original(Yw, prepared)
        calls["count"] += 1
        if calls["count"] > 1:
            result[0, 0] = np.nan
        return result

    monkeypatch.setattr(
        F, "_prepared_components_over_Y", fail_each_checkpoint_bootstrap)
    result = F.run_group_scan_omnib(
        subdata,
        family,
        phenotype,
        np.arange(phenotype.size),
        hypothesis_unit="group",
        bootstrap_B=2,
        bootstrap_seed=29,
        feature_seed=17,
        n_jobs=1,
        grm_method="grm_from_X",
        min_snp=3,
        checkpoint_dir=tmp_path / "checkpoint",
        checkpoint_block_size=1,
    )

    fwer = result.model_diagnostics["bootstrap_fwer"]
    failure = result.model_diagnostics["response_failures"]
    checkpoint = result.model_diagnostics["resampling_checkpoint"]
    assert fwer["n_degenerate_replicates"] == 2
    assert failure == {
        "schema": "homoeogwas-native-response-failure-v1",
        "attempted": 3,
        "successful": 1,
        "retried": 0,
        "terminal_failures": 2,
        "observed_failed": False,
        "bootstrap_failed_response_indices": [0, 1],
        "bootstrap_failure_reason": "fixed_component_nonfinite",
        "worst_case_mapping": {
            "observed": "abort_without_scientific_result",
            "bootstrap": "set_null_family_column_nonfinite_for_null_minimum_zero",
        },
    }
    assert checkpoint["bootstrap_failed_response_indices"] == [0, 1]
    assert checkpoint["bootstrap_failure_reason"] == "fixed_component_nonfinite"


def test_gaussian_binding_failure_ceiling_is_point_zero_zero_two():
    at_boundary = core_fwer_gate(
        "gaussian", 0, 500, 1, stage="formal", evidence_path="fixture",
        failure_rate_max=0.002,
    )
    above_boundary = core_fwer_gate(
        "gaussian", 0, 500, 2, stage="formal", evidence_path="fixture",
        failure_rate_max=0.002,
    )
    assert at_boundary.passed is True
    assert above_boundary.passed is False


def test_failed_calibration_response_maps_to_zero_family_minimum():
    threshold = empirical_threshold(
        np.array([[0.2, 0.3]]),
        alpha=0.5,
        failed_mask=np.array([True, False]),
    )

    assert threshold == 0.0


def test_within_ceiling_response_failure_does_not_discard_outer_shard():
    envelope = T._bank_failure_envelope(
        {
            "failed": True,
            "status": "partial_failure",
            "attempted": 500,
            "successful": 499,
            "terminal_failures": 1,
            "terminal_failure_rate": 0.002,
            "failure_rate_max": 0.002,
            "within_failure_ceiling": True,
            "worst_case_mapping": "failure_counts_as_rejection",
        }
    )

    assert envelope["failed"] is False
    assert envelope["response_failures_present"] is True
    assert envelope["status"] == "completed_with_worst_case_response_failures"


def test_conditional_bank_records_fixed_response_denominator():
    context = build_synthetic_omnib_context(n=36, groups=1, copies=2, seed=951)
    bank = run_conditional_bank(
        context,
        bank="calibration",
        count=2,
        design_hash="a" * 64,
        n_jobs=1,
        qa_only=True,
        scenario_id="B.conditional.synthetic.gaussian.calibration",
    )

    assert bank.failure["attempted"] == 2
    assert bank.failure["successful"] == 2
    assert bank.failure["retried"] == 0
    assert bank.failure["terminal_failures"] == 0
    assert bank.failure["failure_rate_max"] == 0.002
    assert bank.failure["within_failure_ceiling"] is True
    assert bank.failure["worst_case_mapping"] == "failure_counts_as_rejection"


def test_conditional_bank_poison_partial_component_failure_for_each_method(
    monkeypatch,
):
    context = build_synthetic_omnib_context(n=44, groups=1, copies=2, seed=953)
    original = F._prepared_components_over_Y

    def fail_minor_burden_in_second_response(Yw, prepared):
        result = original(Yw, prepared)
        result[0, 1] = np.nan
        return result

    monkeypatch.setattr(
        F, "_prepared_components_over_Y", fail_minor_burden_in_second_response
    )
    bank = run_conditional_bank(
        context,
        bank="calibration",
        count=2,
        design_hash="c" * 64,
        n_jobs=1,
        qa_only=True,
        scenario_id="B.conditional.synthetic.gaussian.calibration",
    )

    assert np.isnan(bank.p_by_method["omnib"][:, 1]).all()
    assert np.isnan(bank.p_by_method["minor_burden"][:, 1]).all()
    assert np.isfinite(bank.p_by_method["pc1"][:, 1]).all()
    assert np.isfinite(bank.p_by_method["kernel_hadamard"][:, 1]).all()
    assert bank.failure["nonfinite_response_indices_by_method"]["omnib"] == [1]
    assert bank.failure["nonfinite_response_indices_by_method"]["minor_burden"] == [1]
    assert bank.failure["nonfinite_response_indices_by_method"]["pc1"] == []
    assert empirical_threshold(
        bank.p_by_method["omnib"],
        alpha=0.5,
        failed_mask=T._method_failure_mask(bank, "omnib"),
    ) == 0.0


def test_family_size_stress_preserves_separate_calibration_and_target_failures(
    monkeypatch,
):
    context = build_synthetic_omnib_context(n=44, groups=1, copies=2, seed=954)
    original = T._method_scores
    calls = 0

    def poison_distinct_banks(prepared, responses, *, n_jobs):
        nonlocal calls
        returned = original(prepared, responses, n_jobs=n_jobs)
        if len(returned) == 3:
            bank, execution, snpxsnp = returned
        else:
            bank, execution, snpxsnp, _original_masks = returned
        values = {
            method: np.array(scores, dtype=float, copy=True)
            for method, scores in bank.p_by_method.items()
        }
        masks = {
            method: np.zeros(responses.shape[1], dtype=bool)
            for method in values
        }
        failed_index = 0 if calls == 0 else 1
        values["omnib"][:, failed_index] = np.nan
        masks["omnib"][failed_index] = True
        calls += 1
        return (
            T.MethodScoreBank(
                bank.family_ids, values, dict(bank.tested_family_sizes)
            ),
            execution,
            snpxsnp,
            masks,
        )

    monkeypatch.setattr(T, "_method_scores", poison_distinct_banks)
    payload = T.run_family_size_stress(
        context,
        family_size=1,
        response_count=2,
        calibration_count=20,
        design_hash="d" * 64,
        qa_only=True,
        n_jobs=1,
        scenario_id="B.family_size.synthetic.g1",
    )

    assert payload["calibration_response_failures"][
        "failed_response_indices_by_method"
    ]["omnib"] == [0]
    assert payload["target_response_failures"][
        "failed_response_indices_by_method"
    ]["omnib"] == [1]
    assert payload["calibration_minima_by_method"]["omnib"][0] == 0.0
    assert payload["qa_rejections_by_method"]["omnib"] == [False, True]
    assert payload["target_response_failures"]["attempted"] == 2
    assert payload["target_response_failures"]["failure_rate_max"] == 0.01
    assert payload["target_response_failures"]["worst_case_mapping"] == (
        "failure_counts_as_rejection"
    )


def test_robustness_preserves_bank_specific_method_failures(monkeypatch):
    context = build_synthetic_omnib_context(n=44, groups=1, copies=2, seed=955)
    calls = 0

    monkeypatch.setattr(
        T,
        "_basic_robustness_contexts",
        lambda base, seed, *, with_metadata: (
            {"missingness_2pct": base}, {"missingness_2pct": None}
        ),
    )

    def poison_robustness_bank(prepared, responses, *, n_jobs):
        nonlocal calls
        scores, original_masks = T._local_component_scores(
            prepared, responses, n_jobs=n_jobs
        )
        values = {
            method: np.array(matrix, dtype=float, copy=True)
            for method, matrix in scores.items()
        }
        masks = {
            method: np.array(mask, dtype=bool, copy=True)
            for method, mask in original_masks.items()
        }
        if calls == 0:
            method, index = "minor_burden", 0
        elif calls == 1:
            method, index = "omnib", 1
        elif calls == 2:
            method, index = "pc1", 2
        else:
            method = None
        if method is not None:
            values[method][:, index] = np.nan
            masks[method][index] = True
        calls += 1
        return values, masks

    monkeypatch.setattr(T, "_robustness_score_bank", poison_robustness_bank)
    payload = T.run_encoding_check(
        context,
        bootstrap_B=199,
        design_hash="e" * 64,
        n_jobs=1,
        parallel_jobs=1,
        include_robustness=True,
        qa_only=True,
    )

    record = payload["robustness_checks"]["missingness_2pct"]
    assert record["calibration"]["response_failures"][
        "failed_response_indices_by_method"
    ]["minor_burden"] == [0]
    assert record["heldout"]["response_failures"][
        "failed_response_indices_by_method"
    ]["omnib"] == [1]
    assert record["heldout"]["qa_rejections_by_method"]["omnib"][1] is True
    first = record["qa_power_by_architecture"]["minor_burden_aligned"]
    assert first["response_failures"]["failed_response_indices_by_method"][
        "pc1"
    ] == [2]
    assert first["response_failures"]["worst_case_mapping"] == (
        "failure_counts_as_non_detection"
    )
    _audit_robustness_record("missingness_2pct", record, payload)
    tampered = deepcopy(record)
    tampered["heldout"]["response_failures"][
        "failed_response_indices_by_method"
    ]["omnib"] = []
    with pytest.raises(BenchmarkAuditError, match="response failure"):
        _audit_robustness_record("missingness_2pct", tampered, payload)


def test_end_to_end_payload_serializes_observed_and_bootstrap_diagnostics():
    context = build_synthetic_omnib_context(n=36, groups=1, copies=2, seed=952)
    result = run_end_to_end_null(
        context,
        replicate=0,
        bootstrap_B=1,
        design_hash="b" * 64,
        qa_only=True,
        n_jobs=1,
        scenario_id="B.end2end.synthetic.gaussian",
    )

    failure = result["failure"]
    assert failure["failed"] is False
    assert failure["observed_failed"] is False
    assert failure["response_diagnostics"]["attempted"] == 2
    assert failure["response_diagnostics"]["successful"] == 2
    assert failure["bootstrap_failed_response_indices"] == []
    assert failure["bootstrap_degenerate_policy"] == (
        "any_nonfinite_statistic_sets_null_min_to_zero"
    )
    assert isinstance(_audit_end2end_decisions(result), bool)

    tampered = deepcopy(result)
    tampered["failure"]["bootstrap_within_failure_ceiling"] = False
    with pytest.raises(BenchmarkAuditError, match="bootstrap failure accounting"):
        _audit_end2end_decisions(tampered)


def test_omitted_kernel_partial_bootstrap_failure_forces_null_minimum_zero(
    monkeypatch,
):
    context = build_synthetic_omnib_context(n=44, groups=1, copies=2, seed=954)
    original = F._prepared_components_over_Y

    def fail_one_bootstrap_component(Yw, prepared):
        result = original(Yw, prepared)
        result[0, 1] = np.nan
        return result

    monkeypatch.setattr(
        F, "_prepared_components_over_Y", fail_one_bootstrap_component
    )
    result = run_end_to_end_null(
        context,
        replicate=0,
        bootstrap_B=1,
        design_hash="d" * 64,
        qa_only=True,
        n_jobs=1,
        null_model="omitted_kernel",
        scenario_id="B.end2end.synthetic.omitted_kernel",
    )

    assert result["failure"]["failed"] is False
    assert result["failure"]["bootstrap_failed_response_indices"] == [0]
    assert result["bootstrap_minp"]["n_degenerate_replicates"] == 1
    assert result["null_minima"] == [0.0]
