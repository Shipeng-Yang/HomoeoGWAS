from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import numpy as np
import pytest

from homoeogwas.interact import SubgenomeData
from scripts.benchmarks.v201.comparators import METHOD_NAMES
from scripts.benchmarks.v201.contracts import Scenario, sha256_payload
from scripts.benchmarks.v201.shards import ShardConflict
from scripts.benchmarks.v201.track_omnib import (
    OmniBBenchmarkContext,
    apply_threshold,
    build_synthetic_omnib_context,
    empirical_threshold,
    run_conditional_bank,
    run_encoding_check,
    run_end_to_end_null,
    run_omnib_replicate,
    run_power_replicate,
)


@pytest.fixture(scope="module")
def tiny_context():
    return build_synthetic_omnib_context(n=72, groups=2, copies=3, seed=1)


def test_context_canonicalizes_subgenome_mapping_order(tiny_context):
    reversed_subdata = dict(reversed(tuple(tiny_context.subdata.items())))
    reordered = OmniBBenchmarkContext(
        reversed_subdata,
        tiny_context.family,
        tiny_context.phenotype,
        tiny_context.sample_idx,
    )
    assert tuple(reordered.subdata) == tiny_context.family.subgenomes


def test_calibration_and_evaluation_seed_namespaces_are_disjoint(tiny_context):
    calibration = run_conditional_bank(
        tiny_context,
        bank="calibration",
        count=3,
        design_hash="a" * 64,
        n_jobs=1,
        scenario_id="B.conditional.synthetic.calibration",
    )
    evaluation = run_conditional_bank(
        tiny_context,
        bank="evaluation",
        count=3,
        design_hash="a" * 64,
        n_jobs=1,
        scenario_id="B.conditional.synthetic.heldout",
    )
    power = run_conditional_bank(
        tiny_context,
        bank="power",
        count=3,
        design_hash="a" * 64,
        n_jobs=1,
        scenario_id="B.conditional.synthetic.power",
    )

    assert calibration.requested_role == calibration.canonical_role == "calibration"
    assert evaluation.requested_role == "evaluation"
    assert evaluation.canonical_role == "heldout"
    assert power.canonical_role == "power"
    assert set(calibration.seed_ids).isdisjoint(evaluation.seed_ids)
    assert set(calibration.seed_ids).isdisjoint(power.seed_ids)
    assert set(evaluation.seed_ids).isdisjoint(power.seed_ids)
    assert not np.shares_memory(calibration.responses, evaluation.responses)
    assert not np.shares_memory(calibration.responses, power.responses)
    assert calibration.family_ids == evaluation.family_ids == power.family_ids
    assert calibration.family_hash == evaluation.family_hash == power.family_hash
    assert calibration.response_hash != evaluation.response_hash != power.response_hash
    assert calibration.runtime_seconds >= 0.0
    assert calibration.null_covariance["shape"] == [72, 72]
    assert len(calibration.null_covariance["sha256"]) == 64
    calibration_payload = calibration.to_payload(include_scores=False)
    assert calibration_payload["formal"] is False
    assert calibration_payload["qa_only"] is True
    assert calibration_payload["requested_jobs"] == 1
    assert evaluation.calibration_reference["canonical_role"] == "calibration"
    assert set(evaluation.seed_ids).isdisjoint(
        evaluation.calibration_reference["seed_ids"]
    )
    assert evaluation.response_hash != evaluation.calibration_reference["response_hash"]
    assert evaluation.calibration_reference["shares_memory"] is False
    assert evaluation.calibration_reference["response_hash"] == calibration.response_hash


def test_conditional_bank_rejects_duplicate_seeds_and_method_provenance_tampering(
    tiny_context,
):
    """Duplicated seeds or a deleted score hash must not survive serialization."""

    bank = run_conditional_bank(
        tiny_context,
        bank="calibration",
        count=2,
        design_hash="f" * 64,
        n_jobs=1,
    )
    with pytest.raises(ValueError, match="seeds must be unique"):
        replace(bank, seeds=(bank.seeds[0], bank.seeds[0]))
    with pytest.raises(ValueError, match="method sets must match"):
        replace(bank, score_matrix_hashes={})


def test_end_to_end_rejection_uses_group_minp_only(tiny_context):
    result = run_end_to_end_null(
        tiny_context,
        replicate=0,
        bootstrap_B=19,
        qa_only=True,
        n_jobs=1,
    )
    assert result["hypothesis_unit"] == "group"
    assert result["family_scope"] == "primary_only"
    assert result["qa_only"] is True
    assert result["stage"] == "pilot"
    assert result["formal_rejections"] == []
    assert result["inference_status"] == "noninferential_do_not_threshold"
    assert result["pair_edges_per_group"] == 3
    assert result["direct_higher_order_term"] is False
    assert result["bootstrap_B"] == 19
    assert result["failure"]["failed"] is False

    # Replacing the serialized bootstrap stream by a self-consistent decision
    # triple must be detectable from these actual null-family minima.
    minima = np.asarray(result["null_minima"], dtype=float)
    observed = np.asarray(result["observed_group_p"], dtype=float)
    assert minima.shape == (19,)
    expected_adjusted = (
        1 + (minima[None, :] <= observed[:, None]).sum(axis=1)
    ) / 20
    np.testing.assert_allclose(result["adjusted_p"], expected_adjusted)
    threshold_index = int(np.floor(0.05 * 20)) - 1
    assert result["bootstrap_minp"]["threshold"] == pytest.approx(
        np.sort(minima)[threshold_index]
    )


def test_empirical_threshold_is_learned_once_and_applied_strictly():
    calibration = np.array(
        [
            np.linspace(0.001, 0.019, 19),
            np.linspace(0.20, 0.38, 19),
        ]
    )
    threshold = empirical_threshold(calibration)
    assert threshold == 0.001
    target = np.array([[0.0009, 0.0010, np.nan], [0.9, 0.9, np.nan]])
    np.testing.assert_array_equal(
        apply_threshold(target, threshold), np.array([True, False, False])
    )
    np.testing.assert_array_equal(
        apply_threshold(target, None), np.zeros(3, dtype=bool)
    )
    partial_missing = np.array([[np.nan, np.nan], [0.0005, np.nan]])
    np.testing.assert_array_equal(
        apply_threshold(partial_missing, threshold), np.array([True, False])
    )
    with pytest.raises(ValueError, match="all-NaN"):
        empirical_threshold(np.full((2, 19), np.nan))


def test_power_freezes_calibration_thresholds_before_causal_bank(tiny_context):
    calibration = run_conditional_bank(
        tiny_context, bank="calibration", count=19,
        design_hash="b" * 64, qa_only=True, null_model="gaussian",
        scenario_id="B.conditional.synthetic.gaussian.calibration", n_jobs=1,
    )
    result = run_power_replicate(
        tiny_context,
        calibration_bank=calibration,
        calibration_scenario_id="B.conditional.synthetic.gaussian.calibration",
        replicate=2,
        architecture="minor_burden_aligned",
        interaction_pve=0.05,
        causal_groups=1,
        calibration_count=19,
        response_count=1,
        design_hash="b" * 64,
        qa_only=True,
        n_jobs=1,
    )
    assert result["calibration_role"] == "calibration"
    assert result["target_role"] == "power"
    assert set(result["calibration_seed_ids"]).isdisjoint(result["target_seed_ids"])
    assert result["threshold_source"] == "independent_calibration_bank"
    assert result["family_scope"] == "primary_only"
    assert result["direct_higher_order_term"] is False
    assert result["causal_group_ids"] == ["group_0"]
    assert set(result["thresholds"]) == set(result["rejections_by_method"])

    assert result["calibration_response_ids"] == result["calibration_seed_ids"]
    assert result["target_response_ids"] == result["target_seed_ids"]
    assert result["failed_calibration_response_indices"] == []
    assert result["failed_target_response_indices"] == []
    assert set(result["calibration_minima_by_method"]) == set(
        result["calibration_minima_hashes"]
    )
    assert set(result["target_minima_by_method"]) == set(
        result["target_minima_hashes"]
    )
    assert set(result["causal_minima_by_method"]) == set(
        result["causal_minima_hashes"]
    )
    for method, calibration_minima in result["calibration_minima_by_method"].items():
        target_minima = result["target_minima_by_method"][method]
        causal = np.asarray(result["causal_minima_by_method"][method], dtype=float)
        assert len(calibration_minima) == 19
        assert len(target_minima) == 1
        assert causal.shape == (1, 1)
        threshold = empirical_threshold(np.asarray(calibration_minima)[None, :])
        assert result["thresholds"][method] == pytest.approx(threshold)
        assert result["rejections_by_method"][method] == [
            value < threshold for value in target_minima
        ]
        assert result["causal_detection_by_method"][method] == [
            value < threshold for value in causal[0]
        ]
        assert result["recall_by_method"][method] == [
            float(value < threshold) for value in causal[0]
        ]
        assert result["calibration_minima_hashes"][method] == sha256_payload(
            calibration_minima
        )
        assert result["target_minima_hashes"][method] == sha256_payload(
            target_minima
        )
        assert result["causal_minima_hashes"][method] == sha256_payload(
            result["causal_minima_by_method"][method]
        )


def test_exact_encoding_check_restores_primary_family(tiny_context):
    result = run_encoding_check(
        tiny_context,
        bootstrap_B=3,
        design_hash="c" * 64,
        n_jobs=1,
        parallel_jobs=2,
        include_robustness=False,
    )
    assert isinstance(result["all_required_exact"], bool)
    assert set(result["exact_checks"]) == {
        "allele_flip_25pct",
        "allele_flip_50pct",
        "allele_flip_100pct",
        "within_gene_snp_column_permutation",
        "group_row_permutation_restored_ids",
        "serial_vs_admitted_parallel",
    }
    restored = result["exact_checks"]["group_row_permutation_restored_ids"]
    assert restored["observed_arrays_identical"] is True
    assert restored["adjusted_decisions_identical"] is True
    assert restored["ranking_hash_identical"] is True
    assert restored["rejection_sets_identical"] is True
    required = [check for check in result["exact_checks"].values() if check["required"]]
    assert result["all_required_exact"] is all(
        all(
            check[field]
            for field in (
                "observed_arrays_identical",
                "adjusted_decisions_identical",
                "ranking_hash_identical",
                "rejection_sets_identical",
            )
        )
        for check in required
    )
    assert result["failure"]["failed"] is (not result["all_required_exact"])
    assert result["response_hash"]
    assert result["request_hash"]
    assert result["context_fingerprint"]
    assert result["null_covariance"]["sha256"]


def test_robustness_checks_are_not_mislabeled_as_exact():
    context = build_synthetic_omnib_context(n=48, groups=1, copies=2, seed=44)
    result = run_encoding_check(
        context,
        bootstrap_B=1,
        design_hash="9" * 64,
        n_jobs=1,
        parallel_jobs=1,
        include_robustness=True,
    )
    assert result["robustness_is_exact_invariance"] is False
    assert set(result["robustness_checks"]) == {
        "missingness_2pct",
        "missingness_5pct",
        "miscoding_1pct",
        "markers_per_gene_3",
        "markers_per_gene_10",
        "markers_per_gene_30",
        "maf_0p01_0p05",
        "maf_0p05_0p20",
        "maf_above_0p20",
        "unbalanced_marker_counts",
    }
    for metrics in result["robustness_checks"].values():
        assert metrics["status"] in {"completed", "failed"}
        assert {
            "fwer",
            "power",
            "rank_correlation",
            "top_k_jaccard",
            "non_estimable_rate",
            "absolute_power_regret",
        } <= set(metrics)


def test_replicate_orchestration_writes_one_immutable_canonical_shard(
    tiny_context, tmp_path
):
    scenario = Scenario(
        "B.end2end.synthetic.gaussian",
        "omnib",
        "pilot",
        1,
        3,
        {"experiment": "end2end", "null_model": "gaussian", "qa_only": True},
    )
    path = tmp_path / "replicate.json"
    first = run_omnib_replicate(
        scenario,
        tiny_context,
        replicate=0,
        design_hash="d" * 64,
        n_jobs=1,
        shard_path=path,
    )
    second = run_omnib_replicate(
        scenario,
        tiny_context,
        replicate=0,
        design_hash="d" * 64,
        n_jobs=1,
        shard_path=path,
    )
    assert first == second
    assert json.loads(path.read_text()) == first
    assert len(first["request_hash"]) == 64
    with pytest.raises(ShardConflict):
        run_omnib_replicate(
            scenario,
            tiny_context,
            replicate=0,
            design_hash="e" * 64,
            n_jobs=1,
            shard_path=path,
        )


def test_resume_conflicts_when_context_or_scientific_request_changes(
    tiny_context, tmp_path
):
    path = tmp_path / "bound.json"
    scenario = Scenario(
        "B.end2end.synthetic.gaussian",
        "omnib",
        "pilot",
        1,
        3,
        {"experiment": "end2end", "null_model": "gaussian", "qa_only": True},
    )
    run_omnib_replicate(
        scenario,
        tiny_context,
        replicate=0,
        design_hash="6" * 64,
        n_jobs=1,
        shard_path=path,
    )
    changed_context = build_synthetic_omnib_context(
        n=72, groups=2, copies=3, seed=2
    )
    with pytest.raises(ShardConflict):
        run_omnib_replicate(
            scenario,
            changed_context,
            replicate=0,
            design_hash="6" * 64,
            n_jobs=1,
            shard_path=path,
        )
    changed_b = Scenario(
        scenario.scenario_id,
        "omnib",
        "pilot",
        1,
        4,
        scenario.parameters,
    )
    with pytest.raises(ShardConflict):
        run_omnib_replicate(
            changed_b,
            tiny_context,
            replicate=0,
            design_hash="6" * 64,
            n_jobs=1,
            shard_path=path,
        )
    changed_stage = Scenario(
        scenario.scenario_id,
        "omnib",
        "formal",
        1,
        2_000,
        {**scenario.parameters, "qa_only": False},
    )
    with pytest.raises(ShardConflict):
        run_omnib_replicate(
            changed_stage,
            tiny_context,
            replicate=0,
            design_hash="6" * 64,
            n_jobs=1,
            shard_path=path,
        )


def test_conditional_dispatcher_builds_the_complete_declared_bank(tiny_context):
    scenario = Scenario(
        "B.conditional.synthetic.heldout",
        "omnib",
        "pilot",
        2,
        0,
        {
            "experiment": "conditional",
            "bank": "heldout",
            "null_model": "student_t5",
            "qa_only": True,
        },
    )
    result = run_omnib_replicate(
        scenario,
        tiny_context,
        replicate=0,
        design_hash="5" * 64,
        n_jobs=1,
    )
    assert result["bank"]["response_shape"] == [72, 2]
    assert len(result["bank"]["seed_ids"]) == scenario.replicates
    assert {
        row["canonical_kind"] for row in result["bank"]["response_metadata"]
    } == {"t5"}


def test_encoding_dispatcher_runs_one_declared_batch(tiny_context):
    """Reject a dispatcher that registers encoding but cannot execute it."""

    scenario = Scenario(
        "B.encoding.wheat",
        "omnib",
        "pilot",
        1,
        1,
        {
            "experiment": "encoding",
            "backbone": "wheat",
            "include_robustness": False,
            "qa_only": True,
        },
    )
    result = run_omnib_replicate(
        scenario,
        tiny_context,
        replicate=0,
        design_hash="1" * 64,
        n_jobs=1,
    )
    assert result["experiment"] == "encoding"
    assert result["scenario_id"] == scenario.scenario_id
    assert result["replicate"] == 0
    assert result["failure"]["failed"] is (not result["all_required_exact"])


def test_encoding_payload_has_exact_locked_check_sets(tiny_context):
    result = run_encoding_check(
        tiny_context, bootstrap_B=1, design_hash="1" * 64,
        n_jobs=1, parallel_jobs=2, include_robustness=True,
    )
    assert set(result["exact_checks"]) == {
        "allele_flip_25pct", "allele_flip_50pct", "allele_flip_100pct",
        "within_gene_snp_column_permutation",
        "group_row_permutation_restored_ids", "serial_vs_admitted_parallel",
    }
    assert set(result["robustness_checks"]) == {
        "missingness_2pct", "missingness_5pct", "miscoding_1pct",
        "markers_per_gene_3", "markers_per_gene_10", "markers_per_gene_30",
        "maf_0p01_0p05", "maf_0p05_0p20", "maf_above_0p20",
        "unbalanced_marker_counts",
    }
    expected = all(
        not check["required"] or all(check[field] is True for field in (
            "observed_arrays_identical", "adjusted_decisions_identical",
            "ranking_hash_identical", "rejection_sets_identical",
        ))
        for check in result["exact_checks"].values()
    )
    assert result["all_required_exact"] is expected


def test_power_dispatcher_uses_declared_calibration_count(tiny_context):
    scenario = Scenario(
        "B.power.synthetic.minor",
        "omnib",
        "pilot",
        1,
        0,
        {
            "experiment": "power",
            "architecture": "minor_burden_aligned",
            "causal_groups": 1,
            "interaction_pve": 0.05,
            "calibration_count": 3,
            "response_count": 1,
            "null_model": "gaussian",
            "calibration_scenario_id": (
                "B.conditional.synthetic.gaussian.calibration"
            ),
            "qa_only": True,
        },
    )
    calibration = run_conditional_bank(
        tiny_context, bank="calibration", count=3,
        design_hash="4" * 64, qa_only=True, null_model="gaussian",
        scenario_id="B.conditional.synthetic.gaussian.calibration", n_jobs=1,
    )
    result = run_omnib_replicate(
        scenario,
        tiny_context,
        replicate=0,
        design_hash="4" * 64,
        n_jobs=1,
        power_calibration_bank=calibration,
    )
    assert len(result["calibration_seed_ids"]) == 3


def test_power_cells_share_one_frozen_backbone_calibration_bank(tiny_context):
    design_hash = "5" * 64
    calibration_id = "B.conditional.synthetic.gaussian.calibration"
    calibration = run_conditional_bank(
        tiny_context, bank="calibration", count=3,
        design_hash=design_hash, qa_only=True, null_model="gaussian",
        scenario_id=calibration_id, n_jobs=1,
    )
    hashes = []
    for index, (architecture, pve) in enumerate((
        ("minor_burden_aligned", 0.02), ("mixed_sign", 0.10),
    )):
        payload = run_power_replicate(
            tiny_context, calibration_bank=calibration,
            calibration_scenario_id=calibration_id, replicate=index,
            architecture=architecture, interaction_pve=pve, causal_groups=1,
            calibration_count=3, response_count=1, design_hash=design_hash,
            qa_only=True, n_jobs=1,
            scenario_id=f"B.power.synthetic.{architecture}",
        )
        hashes.append(payload["calibration_bank_manifest_hash"])
    assert hashes[0] == hashes[1]


def test_tested_family_hash_binds_actual_snpxsnp_members(tiny_context):
    original = run_conditional_bank(
        tiny_context,
        bank="calibration",
        count=1,
        design_hash="3" * 64,
        n_jobs=1,
    )
    permuted = OmniBBenchmarkContext(
        {
            label: SubgenomeData(
                X=data.X,
                gene_snp={
                    gene: np.asarray(indices)[::-1]
                    for gene, indices in data.gene_snp.items()
                },
                samples=data.samples,
                chunk=None,
            )
            for label, data in tiny_context.subdata.items()
        },
        tiny_context.family,
        tiny_context.phenotype,
        tiny_context.sample_idx,
    )
    reordered = run_conditional_bank(
        permuted,
        bank="calibration",
        count=1,
        design_hash="3" * 64,
        n_jobs=1,
    )
    assert (
        original.score_bank.tested_family_sizes["snpxsnp"]
        == reordered.score_bank.tested_family_sizes["snpxsnp"]
    )
    assert (
        original.tested_family_hashes["snpxsnp"]
        != reordered.tested_family_hashes["snpxsnp"]
    )
    payload = original.to_payload(include_scores=False)
    assert set(payload["score_matrix_hashes"]) == set(
        payload["tested_family_members"]
    )
    for method, members in payload["tested_family_members"].items():
        assert len(members) == payload["tested_family_sizes"][method]
        assert payload["tested_family_hashes"][method] == sha256_payload(
            {"method": method, "ordered_member_ids": members}
        )
        values = np.ascontiguousarray(original.p_by_method[method])
        digest = hashlib.sha256()
        digest.update(str(values.dtype).encode("ascii"))
        digest.update(repr(values.shape).encode("ascii"))
        digest.update(values.tobytes())
        assert payload["score_matrix_hashes"][method] == digest.hexdigest()

    # The score bank stays group x response while SNPxSNP calibration covers
    # the complete, independently larger raw SNP-pair family.
    assert original.p_by_method["snpxsnp"].shape[0] == len(original.family_ids)
    assert len(original.tested_family_members["snpxsnp"]) > len(original.family_ids)


def test_every_conditional_payload_has_exact_canonical_method_set(tiny_context):
    bank = run_conditional_bank(
        tiny_context, bank="calibration", count=2,
        design_hash="9" * 64, n_jobs=1,
    )
    payload = bank.to_payload()
    assert set(payload["p_by_method"]) == set(METHOD_NAMES)
    assert set(payload["tested_family_members"]) == set(METHOD_NAMES)
    tampered = dict(payload["p_by_method"])
    tampered.pop("pc1")
    with pytest.raises(ValueError, match="method names"):
        replace(bank.score_bank, p_by_method=tampered)


def test_base_manifests_are_the_complete_hash_inputs(tiny_context):
    """Changing a manifest without changing its digest must be auditable."""

    result = run_end_to_end_null(
        tiny_context,
        replicate=0,
        bootstrap_B=1,
        qa_only=True,
        n_jobs=1,
    )
    assert result["family_hash"] == sha256_payload(result["family_manifest"])
    assert result["context_fingerprint"] == sha256_payload(
        result["context_manifest"]
    )
    assert result["context_manifest"]["family"] == result["family_manifest"]
    for subgenome in result["context_manifest"]["subgenomes"]:
        assert set(subgenome["X"]) == {"shape", "dtype", "sha256"}
        assert all(set(row) == {"gene", "indices"} for row in subgenome["gene_snp"])


def test_conditional_execution_is_from_response_scoring(tiny_context):
    result = run_conditional_bank(
        tiny_context,
        bank="calibration",
        count=2,
        design_hash="2" * 64,
        n_jobs=2,
    )
    assert result.execution["requested_jobs"] == 2
    assert result.execution["effective_jobs"] == 2
    assert result.execution["backend"] == "fork_shared_memory"
    assert len(result.execution["worker_pids"]) == 2


def test_direct_end_to_end_resume_does_not_recompute_volatile_runtime(
    tiny_context, tmp_path
):
    path = tmp_path / "direct.json"
    first = run_end_to_end_null(
        tiny_context,
        replicate=0,
        bootstrap_B=3,
        qa_only=True,
        design_hash="8" * 64,
        shard_path=path,
    )
    second = run_end_to_end_null(
        tiny_context,
        replicate=0,
        bootstrap_B=3,
        qa_only=True,
        design_hash="8" * 64,
        shard_path=path,
    )
    assert second == first
    with pytest.raises(ShardConflict):
        run_end_to_end_null(
            tiny_context,
            replicate=0,
            bootstrap_B=4,
            qa_only=True,
            design_hash="8" * 64,
            shard_path=path,
        )
    with pytest.raises(ShardConflict):
        run_end_to_end_null(
            tiny_context,
            replicate=0,
            bootstrap_B=3,
            qa_only=False,
            design_hash="8" * 64,
            shard_path=path,
        )


def test_failed_replicate_is_serialized_instead_of_dropped(tiny_context, tmp_path):
    degenerate = OmniBBenchmarkContext(
        {
            label: SubgenomeData(
                X=np.zeros_like(data.X),
                gene_snp=data.gene_snp,
                samples=data.samples,
                chunk=None,
            )
            for label, data in tiny_context.subdata.items()
        },
        tiny_context.family,
        tiny_context.phenotype,
        tiny_context.sample_idx,
    )
    scenario = Scenario(
        "B.conditional.synthetic.calibration",
        "omnib",
        "pilot",
        1,
        0,
        {"experiment": "conditional", "bank": "calibration", "qa_only": True},
    )
    path = tmp_path / "failed.json"
    result = run_omnib_replicate(
        scenario,
        degenerate,
        replicate=0,
        design_hash="7" * 64,
        n_jobs=1,
        shard_path=path,
    )
    assert result["failure"]["failed"] is True
    assert result["failure"]["error_type"] == "ValueError"
    assert result["runtime_seconds"] >= 0.0
    assert json.loads(path.read_text()) == result
