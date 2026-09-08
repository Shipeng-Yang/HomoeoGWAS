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
