from __future__ import annotations

import json

import numpy as np
import pytest

from scripts.benchmarks.v201.contracts import Scenario
from scripts.benchmarks.v201.shards import ShardConflict
from scripts.benchmarks.v201.track_fit import (
    build_fit_config,
    build_scan_comparators,
    experiment_wide_scan_fwer,
    fit_pve_replicate,
    parse_fit_metrics,
    read_comparator_preflight,
    run_fit_replicate,
    select_pilot_samples,
    simulate_fit_truth,
    write_comparator_preflight,
)


def _toy_kernels(n=36):
    rng = np.random.default_rng(91)
    kernels = {}
    for name in ("A", "D"):
        z = rng.normal(size=(n, 18))
        z -= z.mean(axis=0)
        kernel = z @ z.T / z.shape[1]
        kernels[name] = kernel / (np.trace(kernel) / n)
    return kernels


def test_pilot_sample_selection_is_pc1_spanning_and_deterministic():
    ids = np.array([f"s{i:03d}" for i in range(250)])
    pc1 = np.linspace(-3, 3, 250)[::-1]

    first = select_pilot_samples(ids, pc1, n=192)
    second = select_pilot_samples(ids, pc1, n=192)

    assert np.array_equal(first, second)
    assert first.size == 192
    assert np.unique(first).size == 192
    assert pc1[first].min() == pc1.min()
    assert pc1[first].max() == pc1.max()


def test_pilot_selection_manifest_preserves_ordered_leading_zero_ids():
    ids = np.array(["0007", "0012", "0100", "1000", "2000"])
    pc1 = np.array([0.0, -2.0, 3.0, 1.0, -1.0])

    selected, manifest = select_pilot_samples(
        ids, pc1, n=4, return_metadata=True
    )

    assert manifest["ordered_sample_ids"] == ids[selected].tolist()
    assert "0012" in manifest["ordered_sample_ids"]
    assert all(isinstance(value, str) for value in manifest["ordered_sample_ids"])
    assert len(manifest["ordered_sample_ids_sha256"]) == 64
    assert manifest["n_selected"] == 4


def test_generated_fit_config_has_locked_reml_and_output(tmp_path):
    fixture = {
        "panel": "cotton_aadd",
        "subgenomes": ["A", "D"],
        "phenotype": tmp_path / "phenotype.tsv",
        "sample_col": "sample",
        "trait": "simulated_trait",
        "bed_prefixes": {"A": tmp_path / "A", "D": tmp_path / "D"},
    }

    cfg = build_fit_config(fixture, tmp_path / "rep-000001")

    assert cfg["reml"]["n_starts"] == 10
    assert cfg["reml"]["seed"] == 2026
    assert cfg["outputs"]["out_dir"].endswith("rep-000001")
    assert cfg["scan"]["loco"]["enabled"] is False
    assert isinstance(cfg["phenotype"]["path"], str)
    assert isinstance(cfg["genotype"]["scan_bed_prefix_template"], str)


def test_coverage_config_locks_released_bootstrap_and_validates(tmp_path):
    from homoeogwas.cli import validate_config

    fixture = {
        "panel": "wheat_aabbdd",
        "subgenomes": ["A", "B", "D"],
        "phenotype": tmp_path / "phenotype.tsv",
        "sample_col": "sample",
        "trait": "simulated_trait",
        "bed_template": tmp_path / "geno" / "{subgenome}" / "all",
    }

    cfg = build_fit_config(
        fixture, tmp_path / "rep-000004", coverage=True, bootstrap_jobs=3
    )

    assert cfg["reml"]["pve_bootstrap"] == {
        "enabled": True,
        "B": 200,
        "level": 0.95,
        "n_jobs": 3,
        "n_starts": 10,
    }
    validate_config(cfg)


def test_each_built_fit_template_is_production_validated(tmp_path):
    fixture = {
        "panel": "cotton_aadd",
        "subgenomes": ["A", "D"],
        "phenotype": tmp_path / "phenotype.tsv",
        "sample_col": "sample",
        "trait": "simulated_trait",
        "bed_template": tmp_path / "geno" / "{subgenome}" / "all",
    }

    with pytest.raises(SystemExit, match="pve_bootstrap.n_jobs"):
        build_fit_config(
            fixture,
            tmp_path / "invalid",
            coverage=True,
            bootstrap_jobs=0,
        )


def test_fit_truth_uses_fingerprinted_kernels_and_exact_component_scaling():
    kernels = _toy_kernels()

    y, truth = simulate_fit_truth(kernels, "single_dominant", seed=812)
    repeated_y, repeated_truth = simulate_fit_truth(
        kernels, "single_dominant", seed=812
    )

    np.testing.assert_array_equal(y, repeated_y)
    assert truth == repeated_truth
    assert np.var(y, ddof=1) == pytest.approx(1.0)
    assert truth["seed"] == 812
    assert truth["target_pve"] == {"A": 0.32, "D": 0.08, "e": 0.6}
    assert truth["target_component_variance"] == truth["target_pve"]
    assert truth["realized_component_variance"]["A"] == pytest.approx(0.32)
    assert truth["realized_component_variance"]["D"] == pytest.approx(0.08)
    assert truth["realized_component_variance"]["e"] == pytest.approx(0.6)
    assert set(truth["kernel_fingerprints"]) == {"A", "D"}
    assert len(truth["context_hash"]) == 64
    assert len(truth["allocation_hash"]) == 64


def test_fit_truth_does_not_reuse_identity_across_allocations():
    kernels = _toy_kernels()

    _, balanced = simulate_fit_truth(kernels, "balanced", seed=19)
    _, dominant = simulate_fit_truth(kernels, "single_dominant", seed=19)

    assert balanced["allocation_hash"] != dominant["allocation_hash"]
    assert balanced["truth_hash"] != dominant["truth_hash"]
    assert balanced["target_pve"] != dominant["target_pve"]


def test_fit_pve_replicate_uses_ten_starts_and_keeps_optimizer_state():
    kernels = _toy_kernels(n=28)
    y, _ = simulate_fit_truth(kernels, "balanced", seed=121)

    result = fit_pve_replicate(y, kernels, seed=121)

    assert set(result["estimated_pve"]) == {"A", "D", "e"}
    assert result["n_starts"] == 10
    assert isinstance(result["optimizer_status"], bool)
    assert isinstance(result["optimizer_message"], str)
    assert isinstance(result["boundary_components"], list)
    assert result["runtime_seconds"] >= 0.0


def test_parse_fit_metrics_reads_real_summary_and_bootstrap_table(tmp_path):
    truth = {
        "target_pve": {"A": 0.2, "D": 0.2, "e": 0.6},
        "realized_pve": {"A": 0.21, "D": 0.19, "e": 0.6},
    }
    summary = {
        "runtime_sec": 12.5,
        "acceptance_all_passed": False,
        "acceptance": [{"check": "reml_converged", "passed": False}],
        "reml": {
            "pve": {"A": 0.25, "D": 0.15, "e": 0.6},
            "sigma2": {"A": 0.25, "D": 0.15, "e": 0.6},
            "optimizer_status": False,
            "boundary_components": ["D"],
            "n_starts": 10,
            "pve_uncertainty": {
                "method": "parametric_bootstrap_fitted_multi_kernel_reml",
                "level": 0.95,
                "B_requested": 200,
                "B_success": 198,
                "B_failed": 2,
                "components": {
                    "A": {"ci_low": 0.10, "ci_high": 0.31},
                    "D": {"ci_low": 0.03, "ci_high": 0.22},
                    "e": {"ci_low": 0.49, "ci_high": 0.72},
                },
            },
        },
    }
    (tmp_path / "summary_simulated_trait.json").write_text(json.dumps(summary))
    (tmp_path / "pve_bootstrap_simulated_trait.tsv").write_text(
        "replicate\tA\tD\te\tboundary_A\tboundary_D\tboundary_e\n"
        "0\t0.24\t0.16\t0.60\tFalse\tFalse\tFalse\n"
        "1\t0.26\t0.14\t0.60\tFalse\tTrue\tFalse\n"
    )
    (tmp_path / "variance_components_simulated_trait.png").write_bytes(b"ignored")

    metrics = parse_fit_metrics(tmp_path, "simulated_trait", truth=truth)

    assert metrics["true_pve"] == truth["realized_pve"]
    assert metrics["estimated_pve"] == summary["reml"]["pve"]
    assert metrics["pve_bias"] == pytest.approx({"A": 0.04, "D": -0.04, "e": 0.0})
    assert metrics["optimizer_status"] is False
    assert metrics["converged"] is False
    assert metrics["boundary_components"] == ["D"]
    assert metrics["runtime_seconds"] == 12.5
    assert metrics["pve_bootstrap"]["rows"] == 2
    assert len(metrics["pve_bootstrap"]["sha256"]) == 64
    assert metrics["pve_bootstrap"]["coverage"] == {"A": True, "D": True, "e": True}


def test_run_fit_replicate_resumes_only_same_sample_and_request(tmp_path):
    kernels = _toy_kernels(n=24)
    scenario = Scenario(
        "A.recovery.cotton.balanced",
        "fit",
        "pilot",
        2,
        0,
        {"experiment": "recovery", "allocation": "balanced", "total_pve": 0.4},
    )
    shard = tmp_path / "replicate-000000.json"
    sample_ids = np.array([f"00{i:02d}" for i in range(24)])

    first = run_fit_replicate(
        scenario,
        kernels,
        replicate=0,
        design_hash="d" * 64,
        sample_ids=sample_ids,
        shard_path=shard,
    )
    second = run_fit_replicate(
        scenario,
        kernels,
        replicate=0,
        design_hash="d" * 64,
        sample_ids=sample_ids,
        shard_path=shard,
    )

    assert first == second == json.loads(shard.read_text())
    assert first["true_pve"] == first["truth"]["realized_pve"]
    assert set(first["pve_bias"]) == {"A", "D", "e"}
    assert first["sample_manifest"]["ordered_sample_ids"][0] == "0000"
    assert len(first["request_hash"]) == len(first["context_fingerprint"]) == 64
    with pytest.raises(ShardConflict):
        run_fit_replicate(
            scenario,
            kernels,
            replicate=0,
            design_hash="d" * 64,
            sample_ids=sample_ids[::-1],
            shard_path=shard,
        )


def test_run_fit_replicate_writes_real_failure_shard(tmp_path):
    scenario = Scenario(
        "A.recovery.cotton.balanced",
        "fit",
        "pilot",
        1,
        0,
        {"experiment": "recovery", "allocation": "balanced", "total_pve": 0.4},
    )
    zero = np.zeros((12, 12))
    shard = tmp_path / "failed.json"

    result = run_fit_replicate(
        scenario,
        {"A": zero, "D": zero},
        replicate=0,
        design_hash="f" * 64,
        sample_ids=np.array([f"s{i:02d}" for i in range(12)]),
        shard_path=shard,
    )

    assert result["failure"]["failed"] is True
    assert result["failure"]["error_type"] == "ValueError"
    assert json.loads(shard.read_text()) == result


def test_fit_runner_preserves_runtime_from_released_summary(tmp_path):
    output = tmp_path / "fit-output"
    output.mkdir()
    (output / "summary_simulated_trait.json").write_text(
        json.dumps(
            {
                "runtime_sec": 9.5,
                "acceptance_all_passed": True,
                "acceptance": [],
                "reml": {
                    "pve": {"A": 0.2, "D": 0.2, "e": 0.6},
                    "sigma2": {"A": 0.2, "D": 0.2, "e": 0.6},
                    "optimizer_status": True,
                    "boundary_components": [],
                    "n_starts": 10,
                    "pve_uncertainty": None,
                },
            }
        )
    )
    scenario = Scenario(
        "A.recovery.cotton.balanced",
        "fit",
        "pilot",
        1,
        0,
        {"experiment": "recovery", "allocation": "balanced", "total_pve": 0.4},
    )

    result = run_fit_replicate(
        scenario,
        _toy_kernels(n=18),
        replicate=0,
        design_hash="b" * 64,
        fit_output_dir=output,
    )

    assert result["runtime_seconds"] == 9.5
    assert result["orchestration_runtime_seconds"] >= 0.0


def test_scan_comparators_share_context_and_pooled_kernel_is_trace_normalized():
    kernels = _toy_kernels(n=20)
    rng = np.random.default_rng(510)
    sample_ids = np.array([f"00{i:02d}" for i in range(20)])
    variants = {"A": ["a1", "a2"], "D": ["d1", "d2", "d3"]}
    standardized = {
        name: (lambda x: (x - x.mean(axis=0)) / x.std(axis=0, ddof=1))(
            rng.normal(size=(20, len(ids)))
        )
        for name, ids in variants.items()
    }
    phenotype = np.linspace(-1.0, 1.0, 20)
    covariates = np.column_stack((np.ones(20), np.linspace(0.0, 2.0, 20)))

    comparators = build_scan_comparators(
        kernels,
        sample_ids=sample_ids,
        variant_ids=variants,
        standardized_variants=standardized,
        phenotype=phenotype,
        covariates=covariates,
    )

    assert set(comparators["canonical_multi_kernel"]) == {"A", "D"}
    assert set(comparators["pooled_trace_sum"]) == {"pooled"}
    assert np.trace(comparators["pooled_trace_sum"]["pooled"]) == pytest.approx(20.0)
    assert set(comparators["independent_subgenome"]) == {"A", "D"}
    assert set(comparators["independent_subgenome"]["A"]) == {"A"}
    assert comparators["standardized_variants"]["A"].flags.writeable is False
    assert len(comparators["context_fingerprint"]) == 64
    assert comparators["loco"]["separate_sensitivity_arm"] is True


def test_independent_scans_use_one_experiment_wide_fwer_family():
    result = experiment_wide_scan_fwer(
        {"A": np.array([0.03]), "D": np.array([0.03])},
        variant_ids={"A": ["a1"], "D": ["d1"]},
        alpha=0.05,
    )

    assert result["family_scope"] == "experiment_wide_subgenome_union"
    assert result["family_size"] == 2
    assert result["adjusted_p"] == {"A": [0.06], "D": [0.06]}
    assert result["rejected"] == {"A": [False], "D": [False]}
    assert len(result["ordered_family_hash"]) == 64


def test_comparator_preflight_is_two_row_frozen_and_hash_checked(tmp_path):
    gcta = tmp_path / "gcta64"
    gcta.write_text("#!/bin/sh\necho 'GCTA 1.94.1'\n")
    gcta.chmod(0o755)
    path = tmp_path / "comparator_preflight.tsv"

    written = write_comparator_preflight(
        path,
        sample_order_hash="1" * 64,
        qc_hash="2" * 64,
        covariates_hash="3" * 64,
        executables={"GCTA": gcta, "GEMMA": tmp_path / "missing-gemma"},
        matched={
            "GCTA": {"sample_order": True, "qc": True, "covariates": True},
            "GEMMA": {"sample_order": True, "qc": True, "covariates": True},
        },
    )
    loaded = read_comparator_preflight(path, expected_hash=written["sha256"])

    assert [row["comparator"] for row in loaded["rows"]] == ["GCTA", "GEMMA"]
    assert loaded["rows"][0]["status"] == "COMPARABLE"
    assert loaded["rows"][1]["status"] == "UNAVAILABLE_OR_NONCOMPARABLE"
    assert loaded["rows"][1]["executable_path"] == ""
    with pytest.raises(FileExistsError, match="frozen"):
        write_comparator_preflight(
            path,
            sample_order_hash="1" * 64,
            qc_hash="2" * 64,
            covariates_hash="3" * 64,
        )

    path.write_text(path.read_text().replace("GCTA 1.94.1", "GCTA 9.99"))
    with pytest.raises(ShardConflict, match="hash"):
        read_comparator_preflight(path, expected_hash=written["sha256"])


def test_comparator_preflight_cannot_be_created_after_outcomes(tmp_path):
    with pytest.raises(RuntimeError, match="before outcomes"):
        write_comparator_preflight(
            tmp_path / "comparator_preflight.tsv",
            sample_order_hash="1" * 64,
            qc_hash="2" * 64,
            covariates_hash="3" * 64,
            outcomes_exist=True,
        )


def test_fit_runner_only_reads_and_binds_frozen_comparator_preflight(tmp_path):
    preflight = write_comparator_preflight(
        tmp_path / "comparator_preflight.tsv",
        sample_order_hash="1" * 64,
        qc_hash="2" * 64,
        covariates_hash="3" * 64,
        executables={
            "GCTA": tmp_path / "missing-gcta",
            "GEMMA": tmp_path / "missing-gemma",
        },
    )
    scenario = Scenario(
        "A.recovery.cotton.balanced",
        "fit",
        "pilot",
        1,
        0,
        {"experiment": "recovery", "allocation": "balanced", "total_pve": 0.4},
    )
    shard = tmp_path / "replicate.json"
    with pytest.raises(ValueError, match="frozen hash"):
        run_fit_replicate(
            scenario,
            _toy_kernels(n=18),
            replicate=0,
            design_hash="a" * 64,
            comparator_preflight_path=preflight["path"],
        )
    result = run_fit_replicate(
        scenario,
        _toy_kernels(n=18),
        replicate=0,
        design_hash="a" * 64,
        shard_path=shard,
        comparator_preflight_path=preflight["path"],
        comparator_preflight_hash=preflight["sha256"],
    )

    assert result["comparator_preflight_hash"] == preflight["sha256"]
    assert {row["status"] for row in result["external_comparators"]} == {
        "UNAVAILABLE_OR_NONCOMPARABLE"
    }
    preflight_path = tmp_path / "comparator_preflight.tsv"
    preflight_path.write_text(preflight_path.read_text() + "\n")
    with pytest.raises(ShardConflict, match="hash"):
        run_fit_replicate(
            scenario,
            _toy_kernels(n=18),
            replicate=0,
            design_hash="a" * 64,
            shard_path=shard,
            comparator_preflight_path=preflight_path,
            comparator_preflight_hash=preflight["sha256"],
        )
