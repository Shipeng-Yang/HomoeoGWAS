from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

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
    run_scan_replicate,
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


def _write_dummy_plink_prefix(prefix):
    prefix.parent.mkdir(parents=True, exist_ok=True)
    (prefix.parent / f"{prefix.name}.bed").write_bytes(b"l\x1b\x01")
    (prefix.parent / f"{prefix.name}.bim").write_text("1\trs1\t0\t1\tA\tC\n")
    (prefix.parent / f"{prefix.name}.fam").write_text("F\t0001\t0\t0\t0\t-9\n")
    return prefix


def _write_released_fit_output(
    output,
    *,
    sample_ids,
    trait="simulated_trait",
    subgenomes=("A", "D"),
    bootstrap=True,
    loco=False,
    acceptance=True,
):
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "configs" / "fit.generated.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_path = output / "resolved_config.yaml"
    samples_path = output / "analysis_samples.tsv"
    sumstats_path = output / f"sumstats_{trait}.tsv"
    lambda_path = output / f"lambda_gc_{trait}.tsv"
    config = {
        "fit_version": 1,
        "panel": {"name": "fixture", "subgenomes": list(subgenomes)},
        "phenotype": {
            "path": str(output / "phenotype.tsv"),
            "sample_col": "sample",
            "trait": trait,
        },
        "genotype": {"scan_bed_prefix_template": "geno/{subgenome}/all"},
        "kernels": {"normalize": "trace", "include_hadamard": False},
        "reml": {"n_starts": 10, "seed": 2026},
        "scan": {
            "mode": "memory",
            "backend": "cpu",
            "maf_min": 0.05,
            "call_rate_min": 0.9,
            "loco": {"enabled": loco},
        },
        "outputs": {"out_dir": str(output.resolve()), "prefix": trait},
    }
    if bootstrap:
        config["reml"]["pve_bootstrap"] = {
            "enabled": True,
            "B": 2,
            "level": 0.95,
            "n_starts": 10,
        }
    encoded_config = yaml.safe_dump(config, sort_keys=False)
    config_path.write_text(encoded_config)
    resolved_path.write_text(encoded_config)
    samples_path.write_text(
        "sample\n" + "".join(f"{sample_id}\n" for sample_id in sample_ids)
    )
    sumstats_path.write_text(
        "snp_id\tsubgenome\tchrom\tpos\tbeta\tse\tchi2\tp\tmaf\n"
        "a1\tA\t1A\t10\t0.1\t0.1\t1\t0.03\t0.2\n"
        "d1\tD\t1D\t20\t0.1\t0.1\t1\t0.04\t0.2\n"
    )
    lambda_path.write_text(
        "scope\tlevel\tn_markers\tlambda_gc\nall\tall\t2\t1.01\n"
    )
    uncertainty = None
    bootstrap_path = output / f"pve_bootstrap_{trait}.tsv"
    if bootstrap:
        bootstrap_path.write_text(
            "replicate\tA\tD\te\tboundary_A\tboundary_D\tboundary_e\n"
            "0\t0.24\t0.16\t0.60\tFalse\tFalse\tFalse\n"
            "1\t0.26\t0.14\t0.60\tFalse\tTrue\tFalse\n"
        )
        uncertainty = {
            "method": "parametric_bootstrap_fitted_multi_kernel_reml",
            "level": 0.95,
            "B_requested": 2,
            "B_success": 2,
            "B_failed": 0,
            "components": {
                "A": {"estimate": 0.25, "ci_low": 0.10, "ci_high": 0.31},
                "D": {"estimate": 0.15, "ci_low": 0.03, "ci_high": 0.22},
                "e": {"estimate": 0.60, "ci_low": 0.49, "ci_high": 0.72},
            },
        }
    summary = {
        "tool": "homoeogwas",
        "command": "fit",
        "trait": trait,
        "subgenomes": list(subgenomes),
        "n_analysis": len(sample_ids),
        "kernel_names": list(subgenomes),
        "runtime_sec": 12.5,
        "config": str(config_path.resolve()),
        "acceptance_all_passed": acceptance,
        "acceptance": [
            {"check": "reml_converged", "passed": acceptance},
            {"check": "scan_has_markers", "passed": acceptance},
        ],
        "reml": {
            "pve": {"A": 0.25, "D": 0.15, "e": 0.6},
            "sigma2": {"A": 0.25, "D": 0.15, "e": 0.6},
            "optimizer_status": acceptance,
            "optimizer_message": "" if acceptance else "did not converge",
            "boundary_components": ["D"],
            "n_starts": 10,
            "pve_uncertainty": uncertainty,
        },
        "scan": {
            "loco_enabled": loco,
            "n_markers_input": 2,
            "n_markers_kept": 2,
        },
        "outputs": {
            "out_dir": str(output.resolve()),
            "sumstats": [str(sumstats_path.resolve())],
            "lambda_gc_tsv": str(lambda_path.resolve()),
            "pve_bootstrap_samples": (
                str(bootstrap_path.resolve()) if bootstrap else None
            ),
            "analysis_samples": str(samples_path.resolve()),
            "resolved_config": str(resolved_path.resolve()),
        },
    }
    summary_path = output / f"summary_{trait}.json"
    summary_path.write_text(json.dumps(summary))
    return summary


def _preflight_inputs(n=4):
    return {
        "sample_ids": [f"00{i}" for i in range(n)],
        "qc_declaration": {"maf_min": 0.05, "call_rate_min": 0.9},
        "covariates": np.column_stack((np.ones(n), np.arange(n, dtype=float))),
        "variant_ids": {"A": ["a1", "a2"], "D": ["d1"]},
    }


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
    prefixes = {
        "A": _write_dummy_plink_prefix(tmp_path / "A"),
        "D": _write_dummy_plink_prefix(tmp_path / "D"),
    }
    fixture = {
        "panel": "cotton_aadd",
        "subgenomes": ["A", "D"],
        "phenotype": tmp_path / "phenotype.tsv",
        "sample_col": "sample",
        "trait": "simulated_trait",
        "bed_prefixes": prefixes,
    }

    cfg = build_fit_config(fixture, tmp_path / "rep-000001")

    assert cfg["reml"]["n_starts"] == 10
    assert cfg["reml"]["seed"] == 2026
    assert cfg["outputs"]["out_dir"].endswith("rep-000001")
    assert cfg["scan"]["loco"]["enabled"] is False
    assert isinstance(cfg["phenotype"]["path"], str)
    assert isinstance(cfg["genotype"]["scan_bed_prefix_template"], str)
    config_path = tmp_path / "rep-000001" / "configs" / "fit.generated.yaml"
    assert config_path.is_file()
    assert config_path.read_text().startswith("fit_version: 1\n")


def test_coverage_config_locks_released_bootstrap_and_validates(tmp_path):
    from homoeogwas.cli import validate_config

    for subgenome in ("A", "B", "D"):
        _write_dummy_plink_prefix(tmp_path / "geno" / subgenome / "all")
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
    for subgenome in ("A", "D"):
        _write_dummy_plink_prefix(tmp_path / "geno" / subgenome / "all")
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


def test_nonshared_bed_prefixes_materialize_managed_template(tmp_path):
    prefixes = {
        "A": _write_dummy_plink_prefix(tmp_path / "source-a" / "alpha"),
        "D": _write_dummy_plink_prefix(tmp_path / "source-d" / "delta"),
    }
    fixture = {
        "panel": "cotton_aadd",
        "subgenomes": ["A", "D"],
        "phenotype": tmp_path / "phenotype.tsv",
        "sample_col": "sample",
        "trait": "simulated_trait",
        "bed_prefixes": prefixes,
    }
    output = tmp_path / "rep-000009"

    cfg = build_fit_config(fixture, output)

    assert cfg["genotype"]["scan_bed_prefix_template"] == str(
        output / "geno" / "{subgenome}" / "all"
    )
    for subgenome, source in prefixes.items():
        for extension in (".bed", ".bim", ".fam"):
            destination = output / "geno" / subgenome / f"all{extension}"
            assert destination.is_symlink()
            assert destination.resolve() == Path(f"{source}{extension}").resolve()


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
    assert sum(truth["target_pve"].values()) == pytest.approx(1.0)
    assert truth["target_allocation"] == {"A": 0.32, "D": 0.08}
    assert truth["observed_total_genetic_pve"] == pytest.approx(0.4)
    assert set(truth["component_effect_sample_variance"]) == {"A", "D", "e"}
    cross = truth["component_cross_covariance"]
    assert set(cross) == {"A", "D", "e"}
    assert all(set(row) == {"A", "D", "e"} for row in cross.values())
    reconstructed = sum(cross[left][right] for left in cross for right in cross)
    assert reconstructed == pytest.approx(1.0)
    assert cross["e"]["A"] + cross["e"]["D"] == pytest.approx(0.0, abs=1e-12)
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
    sample_ids = ["0001", "0002"]
    summary = _write_released_fit_output(
        tmp_path,
        sample_ids=sample_ids,
        bootstrap=True,
        acceptance=False,
    )
    (tmp_path / "variance_components_simulated_trait.png").write_bytes(b"ignored")

    metrics = parse_fit_metrics(
        tmp_path,
        "simulated_trait",
        truth=truth,
        expected_subgenomes=("A", "D"),
        expected_sample_ids=sample_ids,
        experiment="coverage",
    )

    assert metrics["true_pve"] == truth["target_pve"]
    assert metrics["estimated_pve"] == summary["reml"]["pve"]
    assert metrics["pve_bias"] == pytest.approx({"A": 0.05, "D": -0.05, "e": 0.0})
    assert metrics["optimizer_status"] is False
    assert metrics["converged"] is False
    assert metrics["boundary_components"] == ["D"]
    assert metrics["runtime_seconds"] == 12.5
    assert metrics["pve_bootstrap"]["rows"] == 2
    assert len(metrics["pve_bootstrap"]["sha256"]) == 64
    assert metrics["pve_bootstrap"]["coverage"] == {
        "A": True,
        "D": True,
        "e": True,
        "total_genetic": True,
    }
    assert metrics["failure"]["error_type"] == "AcceptanceFailure"


def test_parse_fit_metrics_rejects_stale_tool_or_sample_provenance(tmp_path):
    sample_ids = ["0001", "0002"]
    _write_released_fit_output(tmp_path, sample_ids=sample_ids)
    summary_path = tmp_path / "summary_simulated_trait.json"
    summary = json.loads(summary_path.read_text())
    summary["tool"] = "not-homoeogwas"
    summary_path.write_text(json.dumps(summary))

    with pytest.raises(ValueError, match="tool/command"):
        parse_fit_metrics(
            tmp_path,
            "simulated_trait",
            expected_subgenomes=("A", "D"),
            expected_sample_ids=sample_ids,
            experiment="coverage",
        )

    summary["tool"] = "homoeogwas"
    summary_path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="sample order"):
        parse_fit_metrics(
            tmp_path,
            "simulated_trait",
            expected_subgenomes=("A", "D"),
            expected_sample_ids=sample_ids[::-1],
            experiment="coverage",
        )


def test_parse_fit_metrics_rejects_nonboolean_acceptance_evidence(tmp_path):
    sample_ids = ["0001", "0002"]
    summary = _write_released_fit_output(tmp_path, sample_ids=sample_ids)
    summary["acceptance_all_passed"] = "False"
    (tmp_path / "summary_simulated_trait.json").write_text(json.dumps(summary))

    with pytest.raises(ValueError, match="acceptance evidence"):
        parse_fit_metrics(
            tmp_path,
            "simulated_trait",
            expected_subgenomes=("A", "D"),
            expected_sample_ids=sample_ids,
            experiment="coverage",
        )

@pytest.mark.parametrize("corruption", ["schema", "count", "ci"])
def test_parse_fit_metrics_rejects_invalid_bootstrap_evidence(tmp_path, corruption):
    sample_ids = ["0001", "0002"]
    _write_released_fit_output(tmp_path, sample_ids=sample_ids)
    summary_path = tmp_path / "summary_simulated_trait.json"
    bootstrap_path = tmp_path / "pve_bootstrap_simulated_trait.tsv"
    summary = json.loads(summary_path.read_text())
    if corruption == "schema":
        bootstrap_path.write_text(
            "replicate\tA\tD\te\n0\t0.24\t0.16\t0.60\n1\t0.26\t0.14\t0.60\n"
        )
    elif corruption == "count":
        summary["reml"]["pve_uncertainty"]["B_success"] = 1
        summary["reml"]["pve_uncertainty"]["B_failed"] = 1
        summary_path.write_text(json.dumps(summary))
    else:
        summary["reml"]["pve_uncertainty"]["components"]["A"]["ci_low"] = 0.26
        summary_path.write_text(json.dumps(summary))

    with pytest.raises(ValueError, match="bootstrap"):
        parse_fit_metrics(
            tmp_path,
            "simulated_trait",
            expected_subgenomes=("A", "D"),
            expected_sample_ids=sample_ids,
            experiment="coverage",
        )


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
    assert first["true_pve"] == first["truth"]["target_pve"]
    assert first["true_pve"] != first["truth"]["marginal_component_effect_pve"]
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
    sample_ids = [f"s{i:02d}" for i in range(18)]
    _write_released_fit_output(output, sample_ids=sample_ids)
    scenario = Scenario(
        "A.coverage.cotton.balanced",
        "fit",
        "pilot",
        1,
        2,
        {"experiment": "coverage", "allocation": "balanced", "total_pve": 0.4},
    )

    result = run_fit_replicate(
        scenario,
        _toy_kernels(n=18),
        replicate=0,
        design_hash="b" * 64,
        sample_ids=np.asarray(sample_ids),
        fit_output_dir=output,
    )

    assert result["runtime_seconds"] == 12.5
    assert result["orchestration_runtime_seconds"] >= 0.0


def test_fit_runner_dispatches_coverage_and_binds_released_source_hashes(tmp_path):
    output = tmp_path / "fit-output"
    sample_ids = [f"00{i:02d}" for i in range(18)]
    _write_released_fit_output(output, sample_ids=sample_ids)
    scenario = Scenario(
        "A.coverage.cotton.balanced",
        "fit",
        "pilot",
        1,
        2,
        {"experiment": "coverage", "allocation": "balanced", "total_pve": 0.4},
    )
    shard = tmp_path / "coverage.json"

    result = run_fit_replicate(
        scenario,
        _toy_kernels(n=18),
        replicate=0,
        design_hash="c" * 64,
        sample_ids=np.asarray(sample_ids),
        shard_path=shard,
        fit_output_dir=output,
    )

    assert result["experiment"] == "coverage"
    assert result["pve_bootstrap"]["B_requested"] == 2
    assert set(result["released_source_manifest"]["files"]) >= {
        "summary",
        "bootstrap",
        "resolved_config",
        "analysis_samples",
    }
    summary_path = output / "summary_simulated_trait.json"
    summary = json.loads(summary_path.read_text())
    summary["runtime_sec"] = 13.0
    summary_path.write_text(json.dumps(summary))
    with pytest.raises(ShardConflict, match="existing shard differs"):
        run_fit_replicate(
            scenario,
            _toy_kernels(n=18),
            replicate=0,
            design_hash="c" * 64,
            sample_ids=np.asarray(sample_ids),
            shard_path=shard,
            fit_output_dir=output,
        )


@pytest.mark.parametrize("experiment", ["coverage", "scan", "loco", "unknown"])
def test_fit_runner_never_falls_back_to_recovery_for_missing_arm_inputs(
    tmp_path, experiment
):
    scenario = Scenario(
        f"A.{experiment}.cotton.case",
        "fit",
        "pilot",
        1,
        0,
        {"experiment": experiment},
    )
    shard = tmp_path / f"{experiment}.json"

    result = run_fit_replicate(
        scenario,
        _toy_kernels(n=14),
        replicate=0,
        design_hash="e" * 64,
        sample_ids=np.asarray([f"s{i:02d}" for i in range(14)]),
        shard_path=shard,
    )

    assert result["failure"]["failed"] is True
    assert "truth" not in result
    assert json.loads(shard.read_text()) == result


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
    assert comparators["phenotype"].flags.writeable is False
    assert comparators["covariates"].flags.writeable is False
    assert comparators["sample_ids"].tolist() == sample_ids.tolist()
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


def test_scan_fwer_rejects_empty_family_and_retains_nonfinite_planned_tests():
    with pytest.raises(ValueError, match="empty"):
        experiment_wide_scan_fwer(
            {"A": np.array([])}, variant_ids={"A": []}, alpha=0.05
        )

    result = experiment_wide_scan_fwer(
        {"A": np.array([0.025]), "D": np.array([np.nan])},
        variant_ids={"A": ["a1"], "D": ["d1"]},
        alpha=0.05,
    )

    assert result["planned_count"] == 2
    assert result["finite_count"] == 1
    assert result["nonfinite_ids"] == [{"subgenome": "D", "variant_id": "d1"}]
    assert result["adjusted_p"] == {"A": [0.05], "D": [None]}
    assert result["rejected"] == {"A": [True], "D": [False]}
    assert result["failure"]["error_type"] == "MissingPValues"


def test_run_scan_replicate_executes_three_matched_production_comparators():
    n = 16
    kernels = _toy_kernels(n=n)
    rng = np.random.default_rng(723)
    sample_ids = np.asarray([f"s{i:02d}" for i in range(n)])
    variants = {"A": ["a1", "a2"], "D": ["d1", "d2"]}
    standardized = {
        name: rng.normal(size=(n, len(ids))) for name, ids in variants.items()
    }
    phenotype = rng.normal(size=n)
    covariates = np.column_stack((np.ones(n), np.linspace(-1.0, 1.0, n)))
    comparators = build_scan_comparators(
        kernels,
        sample_ids=sample_ids,
        variant_ids=variants,
        standardized_variants=standardized,
        phenotype=phenotype,
        covariates=covariates,
    )

    result = run_scan_replicate(comparators, seed=812)

    assert result["scan_arm"] == "primary"
    assert set(result["comparators"]) == {
        "canonical_multi_kernel",
        "pooled_trace_sum",
        "independent_subgenome",
    }
    for method in result["comparators"].values():
        assert method["fwer"]["planned_count"] == 4
        assert method["fwer"]["family_scope"] == "experiment_wide_subgenome_union"
    assert result["comparators"]["independent_subgenome"][
        "one_experiment_wide_family"
    ] is True


def test_fit_runner_dispatches_loco_only_to_strict_released_loco_output(tmp_path):
    output = tmp_path / "loco-output"
    sample_ids = [f"s{i:02d}" for i in range(14)]
    _write_released_fit_output(
        output, sample_ids=sample_ids, bootstrap=False, loco=True
    )
    scenario = Scenario(
        "A.loco.cotton.pve_0",
        "fit",
        "pilot",
        1,
        0,
        {"experiment": "loco", "scan_pve": 0.0},
    )

    result = run_fit_replicate(
        scenario,
        _toy_kernels(n=14),
        replicate=0,
        design_hash="7" * 64,
        sample_ids=np.asarray(sample_ids),
        fit_output_dir=output,
    )

    assert result["failure"]["failed"] is False
    assert result["scan_arm"] == "loco_sensitivity"
    assert result["separate_sensitivity_arm"] is True
    assert result["comparators"]["canonical_multi_kernel"]["fwer"][
        "planned_count"
    ] == 2
    assert result["released_source_manifest"]["files"]["sumstats"]["exists"]


def test_released_loco_rejects_stale_sumstats_provenance(tmp_path):
    output = tmp_path / "loco-output"
    sample_ids = [f"s{i:02d}" for i in range(14)]
    summary = _write_released_fit_output(
        output, sample_ids=sample_ids, bootstrap=False, loco=True
    )
    summary["outputs"]["sumstats"] = str((tmp_path / "stale.tsv").resolve())
    (output / "summary_simulated_trait.json").write_text(json.dumps(summary))

    with pytest.raises(ValueError, match="scan output paths"):
        run_scan_replicate(
            released_output_dir=output,
            expected_subgenomes=("A", "D"),
            expected_sample_ids=sample_ids,
            loco=True,
        )


def test_fit_runner_dispatches_scan_to_scan_helper(monkeypatch):
    called = {}

    def fake_scan(comparators, **kwargs):
        called["comparators"] = comparators
        called.update(kwargs)
        return {
            "scan_arm": "primary",
            "comparators": {"canonical": {}, "pooled": {}, "independent": {}},
            "failure": {"failed": False, "error_type": None, "message": None},
        }

    monkeypatch.setattr(
        "scripts.benchmarks.v201.track_fit.run_scan_replicate", fake_scan
    )
    scenario = Scenario(
        "A.scan.cotton.one_subgenome.pve_0p05",
        "fit",
        "pilot",
        1,
        0,
        {"experiment": "scan", "placement": "one_subgenome", "scan_pve": 0.05},
    )
    scan_inputs = {"context_fingerprint": "8" * 64}

    result = run_fit_replicate(
        scenario,
        _toy_kernels(n=14),
        replicate=0,
        design_hash="9" * 64,
        sample_ids=np.asarray([f"s{i:02d}" for i in range(14)]),
        scan_comparators=scan_inputs,
    )

    assert result["failure"]["failed"] is False
    assert called["comparators"] is scan_inputs
    assert called["loco"] is False
    assert "truth" not in result


def test_scan_resume_binds_current_data_not_only_declared_context(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "scripts.benchmarks.v201.track_fit.run_scan_replicate",
        lambda *args, **kwargs: {
            "scan_arm": "primary",
            "failure": {"failed": False, "error_type": None, "message": None},
        },
    )
    n = 12
    kernels = _toy_kernels(n=n)
    rng = np.random.default_rng(44)
    sample_ids = np.asarray([f"s{i:02d}" for i in range(n)])
    comparators = build_scan_comparators(
        kernels,
        sample_ids=sample_ids,
        variant_ids={"A": ["a1"], "D": ["d1"]},
        standardized_variants={
            "A": rng.normal(size=(n, 1)),
            "D": rng.normal(size=(n, 1)),
        },
        phenotype=rng.normal(size=n),
        covariates=np.ones((n, 1)),
    )
    scenario = Scenario(
        "A.scan.cotton.one_subgenome.pve_0",
        "fit",
        "pilot",
        1,
        0,
        {"experiment": "scan", "placement": "one_subgenome", "scan_pve": 0.0},
    )
    shard = tmp_path / "scan.json"
    run_fit_replicate(
        scenario,
        kernels,
        replicate=0,
        design_hash="4" * 64,
        sample_ids=sample_ids,
        shard_path=shard,
        scan_comparators=comparators,
    )
    comparators["phenotype"] = np.asarray(comparators["phenotype"]) + 0.01

    with pytest.raises(ShardConflict, match="existing shard differs"):
        run_fit_replicate(
            scenario,
            kernels,
            replicate=0,
            design_hash="4" * 64,
            sample_ids=sample_ids,
            shard_path=shard,
            scan_comparators=comparators,
        )


def test_comparator_preflight_is_two_row_frozen_and_hash_checked(tmp_path):
    gcta = tmp_path / "gcta64"
    gcta.write_text("#!/bin/sh\necho 'GCTA 1.94.1'\n")
    gcta.chmod(0o755)
    path = tmp_path / "comparator_preflight.tsv"

    written = write_comparator_preflight(
        path,
        **_preflight_inputs(),
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
            **_preflight_inputs(),
        )

    path.write_text(path.read_text().replace("GCTA 1.94.1", "GCTA 9.99"))
    with pytest.raises(ShardConflict, match="hash"):
        read_comparator_preflight(path, expected_hash=written["sha256"])


def test_comparator_preflight_cannot_be_created_after_outcomes(tmp_path):
    with pytest.raises(RuntimeError, match="before outcomes"):
        write_comparator_preflight(
            tmp_path / "comparator_preflight.tsv",
            **_preflight_inputs(),
            outcomes_exist=True,
        )


def test_comparator_preflight_rejects_string_flags_and_false_tool_identity(tmp_path):
    impostor = tmp_path / "not-gcta"
    impostor.write_text("#!/bin/sh\necho 'GCTA 1.94.1'\n")
    impostor.chmod(0o755)
    with pytest.raises(ValueError, match="native bool"):
        write_comparator_preflight(
            tmp_path / "bad-bool.tsv",
            **_preflight_inputs(),
            matched={
                "GCTA": {"sample_order": "true", "qc": True, "covariates": True},
                "GEMMA": {"sample_order": True, "qc": True, "covariates": True},
            },
        )

    result = write_comparator_preflight(
        tmp_path / "bad-identity.tsv",
        **_preflight_inputs(),
        executables={"GCTA": impostor, "GEMMA": tmp_path / "missing-gemma"},
        matched={
            "GCTA": {"sample_order": True, "qc": True, "covariates": True},
            "GEMMA": {"sample_order": True, "qc": True, "covariates": True},
        },
    )
    assert result["rows"][0]["status"] == "UNAVAILABLE_OR_NONCOMPARABLE"
    assert "identity" in result["rows"][0]["reason"]

    failing = tmp_path / "gcta64"
    failing.write_text("#!/bin/sh\necho 'GCTA 1.94.1'\nexit 1\n")
    failing.chmod(0o755)
    result = write_comparator_preflight(
        tmp_path / "bad-returncode.tsv",
        **_preflight_inputs(),
        executables={"GCTA": failing, "GEMMA": tmp_path / "missing-gemma"},
        matched={
            "GCTA": {"sample_order": True, "qc": True, "covariates": True},
            "GEMMA": {"sample_order": True, "qc": True, "covariates": True},
        },
    )
    assert result["rows"][0]["status"] == "UNAVAILABLE_OR_NONCOMPARABLE"
    assert "version" in result["rows"][0]["reason"]


def test_fit_runner_only_reads_and_binds_frozen_comparator_preflight(tmp_path):
    preflight_inputs = _preflight_inputs(n=18)
    preflight = write_comparator_preflight(
        tmp_path / "comparator_preflight.tsv",
        **preflight_inputs,
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
        sample_ids=np.asarray(preflight_inputs["sample_ids"]),
        comparator_preflight_path=preflight["path"],
        )
    result = run_fit_replicate(
        scenario,
        _toy_kernels(n=18),
        replicate=0,
        design_hash="a" * 64,
        sample_ids=np.asarray(preflight_inputs["sample_ids"]),
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
            sample_ids=np.asarray(preflight_inputs["sample_ids"]),
            shard_path=shard,
            comparator_preflight_path=preflight_path,
            comparator_preflight_hash=preflight["sha256"],
        )


def test_fit_runner_rejects_preflight_from_different_sample_order(tmp_path):
    inputs = _preflight_inputs(n=12)
    preflight = write_comparator_preflight(
        tmp_path / "comparator_preflight.tsv",
        **inputs,
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

    result = run_fit_replicate(
        scenario,
        _toy_kernels(n=12),
        replicate=0,
        design_hash="6" * 64,
        sample_ids=np.asarray(inputs["sample_ids"][::-1]),
        comparator_preflight_path=preflight["path"],
        comparator_preflight_hash=preflight["sha256"],
    )

    assert result["failure"]["failed"] is True
    assert "sample-order hash differs" in result["failure"]["message"]


def test_scan_runner_compares_each_preflight_context_hash(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "scripts.benchmarks.v201.track_fit.run_scan_replicate",
        lambda *args, **kwargs: {
            "scan_arm": "primary",
            "failure": {"failed": False, "error_type": None, "message": None},
        },
    )
    n = 12
    kernels = _toy_kernels(n=n)
    rng = np.random.default_rng(140)
    sample_ids = [f"00{i}" for i in range(n)]
    comparators = build_scan_comparators(
        kernels,
        sample_ids=np.asarray(sample_ids),
        variant_ids={"A": ["a1", "a2"], "D": ["d1"]},
        standardized_variants={
            "A": rng.normal(size=(n, 2)),
            "D": rng.normal(size=(n, 1)),
        },
        phenotype=rng.normal(size=n),
        covariates=np.column_stack((np.ones(n), np.arange(n))),
    )
    preflight = write_comparator_preflight(
        tmp_path / "comparator_preflight.tsv",
        sample_ids=sample_ids,
        qc_declaration=comparators["context"]["qc_declaration"],
        covariates=comparators["covariates"],
        variant_ids={"A": ["a1", "a2"], "D": ["wrong-d1"]},
        executables={
            "GCTA": tmp_path / "missing-gcta",
            "GEMMA": tmp_path / "missing-gemma",
        },
    )
    scenario = Scenario(
        "A.scan.cotton.one_subgenome.pve_0",
        "fit",
        "pilot",
        1,
        0,
        {"experiment": "scan", "placement": "one_subgenome", "scan_pve": 0.0},
    )

    result = run_fit_replicate(
        scenario,
        kernels,
        replicate=0,
        design_hash="5" * 64,
        sample_ids=np.asarray(sample_ids),
        scan_comparators=comparators,
        comparator_preflight_path=preflight["path"],
        comparator_preflight_hash=preflight["sha256"],
    )

    assert result["failure"]["failed"] is True
    assert "variant_manifest_hash differs" in result["failure"]["message"]
