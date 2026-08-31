from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from scripts.benchmarks.v201.audit import _audit_fit_scenario
from scripts.benchmarks.v201.contracts import Scenario, derive_seed, sha256_payload
from scripts.benchmarks.v201.shards import ShardConflict
from scripts.benchmarks.v201.track_fit import (
    _released_config_context,
    _released_fit_inputs,
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


def _write_released_plink_prefix(prefix, sample_ids, *, seed):
    from bed_reader import to_bed

    rng = np.random.default_rng(seed)
    n_markers = 12
    dosage = rng.binomial(2, 0.3, size=(len(sample_ids), n_markers)).astype(
        np.float32
    )
    prefix.parent.mkdir(parents=True, exist_ok=True)
    to_bed(
        str(prefix) + ".bed",
        dosage,
        properties={
            "fid": ["F"] * len(sample_ids),
            "iid": list(sample_ids),
            "sid": [f"{prefix.parent.name}_v{index:02d}" for index in range(n_markers)],
            "chromosome": [f"{prefix.parent.name}1"] * n_markers,
            "bp_position": [100 * (index + 1) for index in range(n_markers)],
            "allele_1": ["A"] * n_markers,
            "allele_2": ["C"] * n_markers,
        },
        count_A1=True,
    )
    return prefix


def _write_released_fit_output(
    root,
    *,
    sample_ids,
    trait="simulated_trait",
    subgenomes=("A", "D"),
    bootstrap=True,
    bootstrap_B=2,
    loco=False,
    acceptance=True,
):
    output = root / "results"
    output.mkdir(parents=True, exist_ok=True)
    config_path = root / "configs" / "fit.generated.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_path = output / "resolved_config.yaml"
    samples_path = output / "analysis_samples.tsv"
    sumstats_path = output / f"sumstats_{trait}.tsv"
    lambda_path = output / f"lambda_gc_{trait}.tsv"
    phenotype_path = root / "phenotype.tsv"
    phenotype_path.write_text(
        "sample\t" + trait + "\n"
        + "".join(f"{sample_id}\t{index / 10:.1f}\n" for index, sample_id in enumerate(sample_ids))
    )
    bed_template = root / "input-geno" / "{subgenome}" / "all"
    for index, subgenome in enumerate(subgenomes):
        _write_released_plink_prefix(
            root / "input-geno" / subgenome / "all",
            sample_ids,
            seed=701 + index,
        )
    config = {
        "fit_version": 1,
        "panel": {"name": "fixture", "subgenomes": list(subgenomes)},
        "phenotype": {
            "path": str(phenotype_path.resolve()),
            "sample_col": "sample",
            "trait": trait,
        },
        "genotype": {
            "scan_bed_prefix_template": str(bed_template),
            "grm": {
                "source": "bed",
                "bed_prefix_template": str(bed_template),
                "maf_min": 0.05,
            },
        },
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
            "B": bootstrap_B,
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
            + "".join(
                (
                    f"{index}\t0.24\t0.16\t0.60\tFalse\tFalse\tFalse\n"
                    if index % 2 == 0
                    else f"{index}\t0.26\t0.14\t0.60\tFalse\tTrue\tFalse\n"
                )
                for index in range(bootstrap_B)
            )
        )
        uncertainty = {
            "method": "parametric_bootstrap_fitted_multi_kernel_reml",
            "level": 0.95,
            "B_requested": bootstrap_B,
            "B_success": bootstrap_B,
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
        "grm_info": {
            "source": "bed",
            "normalize": "trace",
            "kernel_names": list(subgenomes),
            "marker_input": {
                source_name: {
                    "subgenomes": {
                        subgenome: {
                            "bed_prefix": str(
                                root / "input-geno" / subgenome / "all"
                            )
                        }
                        for subgenome in subgenomes
                    }
                }
                for source_name in ("scan", "grm")
            },
        },
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


def _production_released_inputs(root):
    from homoeogwas.cli import build_kernels, join_samples

    config = yaml.safe_load(
        (root / "results" / "resolved_config.yaml").read_text()
    )
    samples, phenotype, _ = join_samples(config)
    kernels, _ = build_kernels(config, samples)
    return samples.tolist(), phenotype, kernels


def _write_phenotype_values(root, sample_ids, values, trait="simulated_trait"):
    (root / "phenotype.tsv").write_text(
        "sample\t" + trait + "\n"
        + "".join(
            f"{sample_id}\t{float(value)!r}\n"
            for sample_id, value in zip(sample_ids, values, strict=True)
        )
    )


def _write_bound_coverage_output(
    root,
    *,
    scenario,
    design_hash,
    sample_ids,
    replicate=0,
):
    _write_released_fit_output(
        root,
        sample_ids=sample_ids,
        bootstrap_B=scenario.bootstrap_B,
    )
    joined_ids, _, kernels = _production_released_inputs(root)
    assert joined_ids == list(sample_ids)
    seed = derive_seed(
        design_hash, "fit", scenario.scenario_id, replicate, scenario.stage
    )
    phenotype, truth = simulate_fit_truth(
        kernels,
        scenario.parameters["allocation"],
        seed=seed,
        total_pve=scenario.parameters.get("total_pve"),
        dominant_subgenomes=scenario.parameters.get("dominant_subgenomes"),
        null_subgenome=scenario.parameters.get("null_subgenome"),
    )
    _write_phenotype_values(root, sample_ids, phenotype)
    return kernels, phenotype, truth


def _preflight_inputs(n=4):
    genotypes = {
        "A": np.arange(n * 2, dtype=np.float64).reshape(n, 2),
        "D": np.arange(n, dtype=np.float64).reshape(n, 1) + 0.5,
    }
    return {
        "sample_ids": [f"00{i}" for i in range(n)],
        "qc_declaration": {"maf_min": 0.05, "call_rate_min": 0.9},
        "covariates": np.column_stack((np.ones(n), np.arange(n, dtype=float))),
        "variant_ids": {"A": ["a1", "a2"], "D": ["d1"]},
        "genotypes": genotypes,
    }


def _scan_truth_metadata(*, scan_pve=0.05, placement="one_subgenome"):
    causal = [] if scan_pve == 0.0 else [{"variant_id": "a1", "subgenome": "A"}]
    return {
        "placement": placement,
        "scan_pve": scan_pve,
        "causal_variants": causal,
        "realized_signal_pve": scan_pve,
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

    root = tmp_path / "rep-000001"
    cfg = build_fit_config(fixture, root)

    assert cfg["reml"]["n_starts"] == 10
    assert cfg["reml"]["seed"] == 2026
    assert cfg["outputs"]["out_dir"].endswith("rep-000001/results")
    assert cfg["scan"]["loco"]["enabled"] is False
    assert isinstance(cfg["phenotype"]["path"], str)
    assert isinstance(cfg["genotype"]["scan_bed_prefix_template"], str)
    config_path = root / "configs" / "fit.generated.yaml"
    assert config_path.is_file()
    assert config_path.read_text().startswith("fit_version: 1\n")
    assert not (root / "results").exists()


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
        fixture,
        tmp_path / "rep-000004",
        coverage=True,
        bootstrap_B=199,
        bootstrap_jobs=3,
    )

    assert cfg["reml"]["pve_bootstrap"] == {
        "enabled": True,
        "B": 199,
        "level": 0.95,
        "n_jobs": 3,
        "n_starts": 10,
    }
    validate_config(cfg)


def test_formal_coverage_config_uses_explicit_200_bootstraps(tmp_path):
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

    cfg = build_fit_config(
        fixture,
        tmp_path / "formal-replicate",
        coverage=True,
        bootstrap_B=200,
    )

    assert cfg["reml"]["pve_bootstrap"]["B"] == 200
    assert cfg["outputs"]["out_dir"] == str(
        tmp_path / "formal-replicate" / "results"
    )


def test_fit_config_refuses_nonempty_sibling_results_without_force(tmp_path):
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
    root = tmp_path / "occupied"
    (root / "results").mkdir(parents=True)
    (root / "results" / "partial.txt").write_text("partial")

    with pytest.raises(FileExistsError, match="results directory is not empty"):
        build_fit_config(fixture, root)


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
    sample_ids = [f"{index:04d}" for index in range(24)]
    summary = _write_released_fit_output(
        tmp_path,
        sample_ids=sample_ids,
        bootstrap=True,
        acceptance=False,
    )
    (tmp_path / "results" / "variance_components_simulated_trait.png").write_bytes(
        b"ignored"
    )

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


@pytest.mark.parametrize(("stage", "bootstrap_B"), [("pilot", 199), ("formal", 200)])
def test_coverage_parser_binds_stage_specific_bootstrap_budget(
    tmp_path, stage, bootstrap_B
):
    sample_ids = [f"{index:04d}" for index in range(24)]
    _write_released_fit_output(
        tmp_path, sample_ids=sample_ids, bootstrap_B=bootstrap_B
    )
    scenario = Scenario(
        f"A.coverage.cotton.balanced.{stage}",
        "fit",
        stage,
        1,
        bootstrap_B,
        {"experiment": "coverage", "allocation": "balanced", "total_pve": 0.4},
    )

    metrics = parse_fit_metrics(
        tmp_path,
        "simulated_trait",
        truth={"target_pve": {"A": 0.2, "D": 0.2, "e": 0.6}},
        expected_subgenomes=("A", "D"),
        expected_sample_ids=sample_ids,
        expected_scenario=scenario,
        experiment="coverage",
    )

    assert metrics["pve_bootstrap"]["B_requested"] == bootstrap_B


def test_parse_fit_metrics_rejects_stale_tool_or_sample_provenance(tmp_path):
    sample_ids = [f"{index:04d}" for index in range(24)]
    _write_released_fit_output(tmp_path, sample_ids=sample_ids)
    summary_path = tmp_path / "results" / "summary_simulated_trait.json"
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
    sample_ids = [f"{index:04d}" for index in range(24)]
    summary = _write_released_fit_output(tmp_path, sample_ids=sample_ids)
    summary["acceptance_all_passed"] = "False"
    (tmp_path / "results" / "summary_simulated_trait.json").write_text(
        json.dumps(summary)
    )

    with pytest.raises(ValueError, match="acceptance evidence"):
        parse_fit_metrics(
            tmp_path,
            "simulated_trait",
            expected_subgenomes=("A", "D"),
            expected_sample_ids=sample_ids,
            experiment="coverage",
        )

@pytest.mark.parametrize("corruption", ["schema", "count"])
def test_parse_fit_metrics_rejects_invalid_bootstrap_evidence(tmp_path, corruption):
    sample_ids = [f"{index:04d}" for index in range(24)]
    _write_released_fit_output(tmp_path, sample_ids=sample_ids)
    summary_path = tmp_path / "results" / "summary_simulated_trait.json"
    bootstrap_path = tmp_path / "results" / "pve_bootstrap_simulated_trait.tsv"
    summary = json.loads(summary_path.read_text())
    if corruption == "schema":
        bootstrap_path.write_text(
            "replicate\tA\tD\te\n0\t0.24\t0.16\t0.60\n1\t0.26\t0.14\t0.60\n"
        )
    elif corruption == "count":
        summary["reml"]["pve_uncertainty"]["B_success"] = 1
        summary["reml"]["pve_uncertainty"]["B_failed"] = 1
        summary_path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="bootstrap"):
        parse_fit_metrics(
            tmp_path,
            "simulated_trait",
            expected_subgenomes=("A", "D"),
            expected_sample_ids=sample_ids,
            experiment="coverage",
        )


def test_bootstrap_percentile_interval_may_exclude_point_estimate(tmp_path):
    sample_ids = [f"{index:04d}" for index in range(24)]
    summary = _write_released_fit_output(tmp_path, sample_ids=sample_ids)
    summary["reml"]["pve_uncertainty"]["components"]["A"].update(
        {"ci_low": 0.26, "ci_high": 0.31}
    )
    summary_path = tmp_path / "results" / "summary_simulated_trait.json"
    summary_path.write_text(json.dumps(summary))

    metrics = parse_fit_metrics(
        tmp_path,
        "simulated_trait",
        truth={"target_pve": {"A": 0.2, "D": 0.2, "e": 0.6}},
        expected_subgenomes=("A", "D"),
        expected_sample_ids=sample_ids,
        experiment="coverage",
    )

    assert metrics["pve_bootstrap"]["intervals"]["A"] == {
        "estimate": 0.25,
        "ci_low": 0.26,
        "ci_high": 0.31,
    }
    assert metrics["pve_bootstrap"]["coverage"]["A"] is False


def test_released_fit_rejects_wrong_phenotype_and_grm_provenance(tmp_path):
    sample_ids = [f"{index:04d}" for index in range(24)]
    summary = _write_released_fit_output(tmp_path, sample_ids=sample_ids)
    phenotype_path = tmp_path / "phenotype.tsv"
    phenotype_path.write_text(
        "sample\tsimulated_trait\n"
        + "".join(
            f"{sample_id}\t{index / 10:.1f}\n"
            for index, sample_id in enumerate(sample_ids[1:], start=1)
        )
    )

    with pytest.raises(ValueError, match="duplicate-averaging/inner sample join"):
        parse_fit_metrics(
            tmp_path,
            "simulated_trait",
            expected_subgenomes=("A", "D"),
            expected_sample_ids=sample_ids,
            experiment="coverage",
        )

    _write_phenotype_values(
        tmp_path, sample_ids, np.arange(24, dtype=np.float64) / 10.0
    )
    summary["grm_info"]["source"] = "npz"
    (tmp_path / "results" / "summary_simulated_trait.json").write_text(
        json.dumps(summary)
    )
    with pytest.raises(ValueError, match="GRM provenance"):
        parse_fit_metrics(
            tmp_path,
            "simulated_trait",
            expected_subgenomes=("A", "D"),
            expected_sample_ids=sample_ids,
            experiment="coverage",
        )


def test_released_fit_rejects_summary_bed_source_different_from_config(tmp_path):
    sample_ids = [f"{index:04d}" for index in range(24)]
    summary = _write_released_fit_output(tmp_path, sample_ids=sample_ids)
    summary["grm_info"]["marker_input"]["grm"]["subgenomes"]["A"][
        "bed_prefix"
    ] = str(tmp_path / "wrong" / "A")
    (tmp_path / "results" / "summary_simulated_trait.json").write_text(
        json.dumps(summary)
    )

    with pytest.raises(ValueError, match="marker source differs"):
        parse_fit_metrics(
            tmp_path,
            "simulated_trait",
            expected_subgenomes=("A", "D"),
            expected_sample_ids=sample_ids,
            experiment="coverage",
        )


@pytest.mark.parametrize("fam_change", ["mismatch", "extra", "missing"])
def test_released_fit_rejects_fam_iids_inconsistent_with_production_join(
    tmp_path, fam_change
):
    sample_ids = [f"00{index:02d}" for index in range(24)]
    _write_released_fit_output(tmp_path, sample_ids=sample_ids)
    fam_path = tmp_path / "input-geno" / "A" / "all.fam"
    rows = fam_path.read_text().splitlines()
    if fam_change == "mismatch":
        fields = rows[0].split()
        fields[1] = "wrong-iid"
        rows[0] = "\t".join(fields)
    elif fam_change == "extra":
        rows.append("F\textra-iid\t0\t0\t0\t-9")
    else:
        rows.pop()
    fam_path.write_text("\n".join(rows) + "\n")

    with pytest.raises(ValueError, match="FAM|sample join"):
        parse_fit_metrics(
            tmp_path,
            "simulated_trait",
            expected_subgenomes=("A", "D"),
            expected_sample_ids=sample_ids,
            experiment="coverage",
        )


def test_released_fit_uses_production_duplicate_phenotype_average(tmp_path):
    sample_ids = [f"00{index:02d}" for index in range(24)]
    _write_released_fit_output(tmp_path, sample_ids=sample_ids)
    expected = np.arange(24, dtype=np.float64) / 10.0
    phenotype_path = tmp_path / "phenotype.tsv"
    phenotype_path.write_text(
        "sample\tsimulated_trait\n"
        f"{sample_ids[0]}\t-1.0\n"
        f"{sample_ids[0]}\t1.0\n"
        + "".join(
            f"{sample_id}\t{float(expected[index])!r}\n"
            for index, sample_id in enumerate(sample_ids[1:], start=1)
        )
    )

    metrics = parse_fit_metrics(
        tmp_path,
        "simulated_trait",
        expected_subgenomes=("A", "D"),
        expected_sample_ids=sample_ids,
        expected_phenotype=expected,
        experiment="coverage",
    )
    assert metrics["released_input_binding"]["analysis_sample_ids"] == sample_ids

    phenotype_path.write_text(
        phenotype_path.read_text().replace(f"{sample_ids[0]}\t1.0", f"{sample_ids[0]}\t3.0")
    )
    with pytest.raises(ValueError, match="joined phenotype differs"):
        parse_fit_metrics(
            tmp_path,
            "simulated_trait",
            expected_subgenomes=("A", "D"),
            expected_sample_ids=sample_ids,
            expected_phenotype=expected,
            experiment="coverage",
        )


def test_released_fit_rejects_trait_content_different_from_request(tmp_path):
    sample_ids = [f"00{index:02d}" for index in range(24)]
    _write_released_fit_output(tmp_path, sample_ids=sample_ids)
    expected = np.arange(24, dtype=np.float64) / 10.0
    phenotype_path = tmp_path / "phenotype.tsv"
    phenotype_path.write_text(
        phenotype_path.read_text().replace(f"{sample_ids[7]}\t0.7", f"{sample_ids[7]}\t9.7")
    )

    with pytest.raises(ValueError, match="joined phenotype differs"):
        parse_fit_metrics(
            tmp_path,
            "simulated_trait",
            expected_subgenomes=("A", "D"),
            expected_sample_ids=sample_ids,
            expected_phenotype=expected,
            experiment="coverage",
        )


def test_released_fit_rejects_bed_kernels_detached_from_request(tmp_path):
    sample_ids = [f"00{index:02d}" for index in range(24)]
    _write_released_fit_output(tmp_path, sample_ids=sample_ids)
    _, _, released_kernels = _production_released_inputs(tmp_path)
    detached = {name: value.copy() for name, value in released_kernels.items()}
    detached["A"][0, 0] += 0.125

    with pytest.raises(ValueError, match="BED-derived kernel differs"):
        parse_fit_metrics(
            tmp_path,
            "simulated_trait",
            expected_subgenomes=("A", "D"),
            expected_sample_ids=sample_ids,
            expected_kernels=detached,
            experiment="coverage",
        )


def test_coverage_rejects_arbitrary_released_phenotype_as_immutable_failure(tmp_path):
    root = tmp_path / "fit-output"
    sample_ids = [f"00{index:02d}" for index in range(24)]
    _write_released_fit_output(root, sample_ids=sample_ids)
    _, _, kernels = _production_released_inputs(root)
    scenario = Scenario(
        "A.coverage.cotton.balanced",
        "fit",
        "pilot",
        1,
        2,
        {"experiment": "coverage", "allocation": "balanced", "total_pve": 0.4},
    )
    shard = tmp_path / "arbitrary-phenotype.json"

    result = run_fit_replicate(
        scenario,
        kernels,
        replicate=0,
        design_hash="a" * 64,
        sample_ids=np.asarray(sample_ids),
        shard_path=shard,
        fit_output_dir=root,
    )

    assert result["failure"]["failed"] is True
    assert "joined phenotype differs" in result["failure"]["message"]
    assert json.loads(shard.read_text()) == result


def test_coverage_binds_simulated_y_truth_and_bed_derived_kernels(tmp_path):
    root = tmp_path / "fit-output"
    sample_ids = [f"00{index:02d}" for index in range(24)]
    scenario = Scenario(
        "A.coverage.cotton.balanced",
        "fit",
        "pilot",
        1,
        2,
        {"experiment": "coverage", "allocation": "balanced", "total_pve": 0.4},
    )
    kernels, phenotype, truth = _write_bound_coverage_output(
        root,
        scenario=scenario,
        design_hash="b" * 64,
        sample_ids=sample_ids,
    )

    result = run_fit_replicate(
        scenario,
        kernels,
        replicate=0,
        design_hash="b" * 64,
        sample_ids=np.asarray(sample_ids),
        fit_output_dir=root,
    )

    assert result["failure"]["failed"] is False, result["failure"]
    assert result["coverage_request_binding"]["truth_hash"] == truth["truth_hash"]
    assert result["coverage_request_binding"]["phenotype"]["shape"] == [24]
    assert result["released_input_binding"]["joined_phenotype"]["shape"] == [24]
    assert result["released_input_binding"]["kernel_order"] == ["A", "D"]
    assert result["released_input_binding"]["requested_phenotype"]["sha256"] == result[
        "coverage_request_binding"
    ]["phenotype"]["sha256"]
    assert result["released_request_binding"]["joined_phenotype"] == result[
        "released_input_binding"
    ]["joined_phenotype"]
    assert result["released_request_binding"]["truth_hash"] == truth["truth_hash"]
    assert result["released_input_binding"]["truth_hash"] == truth["truth_hash"]
    assert result["released_input_binding"]["phenotype_max_abs_difference"] <= (
        8.0 * np.finfo(np.float64).eps
    )
    assert result["context_fingerprint"] == sha256_payload(
        result["fit_context_manifest"]
    )
    assert result["request_hash"] == sha256_payload(result["request_manifest"])
    assert result["truth_hash"] == sha256_payload(result["truth_manifest"])
    assert result["request_manifest"]["seed"] == result["seed"]
    assert result["request_manifest"]["truth_hash"] == truth["truth_hash"]
    assert result["request_manifest"]["released_source_manifest"] == result[
        "released_source_manifest"
    ]
    assert result["request_manifest"]["released_request_binding"] == result[
        "released_request_binding"
    ]
    assert phenotype.shape == (24,)

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
    assert first["context_fingerprint"] == sha256_payload(
        first["fit_context_manifest"]
    )
    assert first["request_hash"] == sha256_payload(first["request_manifest"])
    assert first["truth_hash"] == sha256_payload(first["truth_manifest"])
    assert first["truth_hash"] == first["truth"]["truth_hash"]
    assert first["request_manifest"]["seed"] == first["seed"]
    assert first["request_manifest"]["truth_hash"] == first["truth_hash"]
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
    sample_ids = [f"s{i:02d}" for i in range(24)]
    scenario = Scenario(
        "A.coverage.cotton.balanced",
        "fit",
        "pilot",
        1,
        2,
        {"experiment": "coverage", "allocation": "balanced", "total_pve": 0.4},
    )
    kernels, _, _ = _write_bound_coverage_output(
        output,
        scenario=scenario,
        design_hash="b" * 64,
        sample_ids=sample_ids,
    )

    result = run_fit_replicate(
        scenario,
        kernels,
        replicate=0,
        design_hash="b" * 64,
        sample_ids=np.asarray(sample_ids),
        fit_output_dir=output,
    )

    assert result["runtime_seconds"] == 12.5
    assert result["orchestration_runtime_seconds"] >= 0.0


def test_fit_runner_dispatches_coverage_and_binds_released_source_hashes(tmp_path):
    output = tmp_path / "fit-output"
    sample_ids = [f"00{i:02d}" for i in range(24)]
    scenario = Scenario(
        "A.coverage.cotton.balanced",
        "fit",
        "pilot",
        1,
        2,
        {"experiment": "coverage", "allocation": "balanced", "total_pve": 0.4},
    )
    shard = tmp_path / "coverage.json"
    kernels, _, _ = _write_bound_coverage_output(
        output,
        scenario=scenario,
        design_hash="c" * 64,
        sample_ids=sample_ids,
    )

    result = run_fit_replicate(
        scenario,
        kernels,
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
        "phenotype",
        "scan_A_bed",
        "scan_A_bim",
        "scan_A_fam",
        "grm_D_bed",
    }
    assert result["released_source_manifest"]["kernel_provenance"] == {
        "matrix_hash_available": False,
        "evidence_boundary": (
            "ordered samples plus resolved QC/config and all scan/GRM "
            "BED/BIM/FAM source bytes"
        ),
    }
    summary_path = output / "results" / "summary_simulated_trait.json"
    summary = json.loads(summary_path.read_text())
    summary["runtime_sec"] = 13.0
    summary_path.write_text(json.dumps(summary))
    with pytest.raises(ShardConflict, match="existing shard differs"):
        run_fit_replicate(
            scenario,
            kernels,
            replicate=0,
            design_hash="c" * 64,
            sample_ids=np.asarray(sample_ids),
            shard_path=shard,
            fit_output_dir=output,
        )


@pytest.mark.parametrize("source_kind", ["phenotype", "bed"])
def test_coverage_resume_conflicts_when_input_source_mutates(tmp_path, source_kind):
    root = tmp_path / "fit-root"
    sample_ids = [f"00{i:02d}" for i in range(24)]
    scenario = Scenario(
        "A.coverage.cotton.balanced",
        "fit",
        "pilot",
        1,
        2,
        {"experiment": "coverage", "allocation": "balanced", "total_pve": 0.4},
    )
    shard = tmp_path / f"{source_kind}.json"
    kernels, _, _ = _write_bound_coverage_output(
        root,
        scenario=scenario,
        design_hash="3" * 64,
        sample_ids=sample_ids,
    )
    first = run_fit_replicate(
        scenario,
        kernels,
        replicate=0,
        design_hash="3" * 64,
        sample_ids=np.asarray(sample_ids),
        shard_path=shard,
        fit_output_dir=root,
    )
    assert first["failure"]["failed"] is False
    if source_kind == "phenotype":
        (root / "phenotype.tsv").write_text(
            "sample\tsimulated_trait\n"
            + "".join(f"{sample_id}\t9.0\n" for sample_id in sample_ids)
        )
    else:
        bed_path = root / "input-geno" / "A" / "all.bed"
        bed_path.write_bytes(bed_path.read_bytes() + b"mutation")

    with pytest.raises(ShardConflict, match="existing shard differs"):
        run_fit_replicate(
            scenario,
            kernels,
            replicate=0,
            design_hash="3" * 64,
            sample_ids=np.asarray(sample_ids),
            shard_path=shard,
            fit_output_dir=root,
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
        truth_metadata=_scan_truth_metadata(),
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
    assert comparators["truth"]["placement"] == "one_subgenome"
    assert comparators["truth"]["scan_pve"] == 0.05
    assert len(comparators["truth"]["truth_hash"]) == 64


def test_independent_scans_use_one_experiment_wide_fwer_family():
    result = experiment_wide_scan_fwer(
        {"A": np.array([0.03]), "D": np.array([0.03])},
        variant_ids={"A": ["a1"], "D": ["d1"]},
        variant_positions_bp={"A": [1100], "D": [2200]},
        alpha=0.05,
    )

    assert result["family_scope"] == "experiment_wide_subgenome_union"
    assert result["family_size"] == 2
    assert result["adjusted_p"] == {"A": [0.06], "D": [0.06]}
    assert result["rejected"] == {"A": [False], "D": [False]}
    assert result["distance_unit"] == "bp"
    assert result["ordered_family"] == [
        {"subgenome": "A", "variant_id": "a1", "position_bp": 1100},
        {"subgenome": "D", "variant_id": "d1", "position_bp": 2200},
    ]
    assert len(result["ordered_family_hash"]) == 64


def test_scan_fwer_rejects_empty_family_and_retains_nonfinite_planned_tests():
    with pytest.raises(ValueError, match="empty"):
        experiment_wide_scan_fwer(
            {"A": np.array([])}, variant_ids={"A": []},
            variant_positions_bp={"A": []}, alpha=0.05
        )

    result = experiment_wide_scan_fwer(
        {"A": np.array([0.025]), "D": np.array([np.nan])},
        variant_ids={"A": ["a1"], "D": ["d1"]},
        variant_positions_bp={"A": [1100], "D": [2200]},
        alpha=0.05,
    )

    assert result["planned_count"] == 2
    assert result["finite_count"] == 1
    assert result["nonfinite_ids"] == [
        {"subgenome": "D", "variant_id": "d1", "position_bp": 2200}
    ]
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
        truth_metadata=_scan_truth_metadata(),
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
    sample_ids = [f"s{i:02d}" for i in range(24)]
    _write_released_fit_output(
        output, sample_ids=sample_ids, bootstrap=False, loco=True
    )
    _, _, kernels = _production_released_inputs(output)
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
        kernels,
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


@pytest.mark.parametrize(
    ("scan_pve", "causal_variants"),
    [
        (0.0, []),
        (0.05, [{"variant_id": "a1", "subgenome": "A", "position_bp": 10}]),
    ],
)
def test_formal_released_loco_binds_explicit_truth_and_bp_family(
    tmp_path, scan_pve, causal_variants,
):
    output = tmp_path / f"formal-loco-{scan_pve}"
    sample_ids = [f"s{i:02d}" for i in range(24)]
    _write_released_fit_output(
        output, sample_ids=sample_ids, bootstrap=False, loco=True
    )
    _, _, kernels = _production_released_inputs(output)
    binding = _released_fit_inputs(
        _released_config_context(output), sample_ids,
        expected_phenotype=None, expected_kernels=kernels,
        expected_truth_hash=None,
    )
    scenario = Scenario(
        f"A.loco.cotton.pve_{scan_pve}", "fit", "formal", 1, 0,
        {"experiment": "loco", "scan_pve": scan_pve, "panel": "cotton"},
    )
    seed_design = {"fixture": f"formal-{scan_pve}"}
    seed_design_hash = sha256_payload(seed_design)
    truth = {
        "scan_pve": scan_pve, "distance_unit": "bp",
        "causal_variants": causal_variants,
        "source": "deterministic_pre_fit_simulation",
        "seed": derive_seed(
            seed_design_hash, "fit", scenario.scenario_id, 0,
            "formal:loco_truth",
        ),
        "analysis_context": {
            "analysis_sample_ids_sha256": binding["analysis_sample_ids_sha256"],
            "joined_phenotype": binding["joined_phenotype"],
            "kernel_order": binding["kernel_order"],
            "kernel_fingerprints": binding["kernel_fingerprints"],
            "generated_config_sha256": hashlib.sha256(
                (output / "configs" / "fit.generated.yaml").read_bytes()
            ).hexdigest(),
        },
    }
    truth["truth_hash"] = sha256_payload(truth)
    design_root = tmp_path / f"design-{scan_pve}"
    (design_root / "inputs").mkdir(parents=True)
    truth_path = design_root / "inputs" / "loco-truth.json"
    truth_path.write_text(json.dumps(truth, sort_keys=True) + "\n")
    truth_sha = hashlib.sha256(truth_path.read_bytes()).hexdigest()
    (design_root / "design_lock.json").write_text(json.dumps({
        "design_hash": "9" * 64,
        "seed_design": seed_design, "seed_design_hash": seed_design_hash,
        "loco_truth_artifacts": {"formal": {scenario.scenario_id: {"0": {
            "path": "inputs/loco-truth.json", "sha256": truth_sha,
            "truth_hash": truth["truth_hash"], "source": truth["source"],
            "seed": truth["seed"],
            "generated_config_sha256": truth["analysis_context"]["generated_config_sha256"],
            "phenotype_sha256": truth["analysis_context"]["joined_phenotype"]["sha256"],
        }}}},
    }, sort_keys=True) + "\n")
    result = run_fit_replicate(
        scenario, kernels, replicate=0, design_hash="9" * 64,
        sample_ids=np.asarray(sample_ids), fit_output_dir=output,
        loco_design_root=design_root,
    )
    assert result["failure"]["failed"] is False
    assert result["released_scan_truth_binding"]["distance_unit"] == "bp"
    assert result["released_request_binding"]["truth_hash"] == truth["truth_hash"]
    assert result["scan_request_manifest"]["bp_positions_explicit"] is True
    assert _audit_fit_scenario(result, scenario) is True


def test_released_loco_uses_distinct_presealed_truth_for_each_replicate(tmp_path):
    sample_ids = [f"s{i:02d}" for i in range(24)]
    scenario = Scenario(
        "A.loco.cotton.pve_0", "fit", "pilot", 2, 0,
        {"experiment": "loco", "scan_pve": 0.0, "panel": "cotton"},
    )
    design_root = tmp_path / "design-two-replicates"
    (design_root / "inputs").mkdir(parents=True)
    records = {}
    runs = []
    seed_design = {"fixture": "pilot-two-replicates"}
    seed_design_hash = sha256_payload(seed_design)
    for replicate in range(2):
        output = tmp_path / f"loco-output-{replicate}"
        _write_released_fit_output(
            output, sample_ids=sample_ids, bootstrap=False, loco=True,
        )
        phenotype = output / "phenotype.tsv"
        phenotype.write_text(
            "sample\tsimulated_trait\n" + "".join(
                f"{sample_id}\t{replicate + index / 100:.12g}\n"
                for index, sample_id in enumerate(sample_ids)
            )
        )
        _, _, kernels = _production_released_inputs(output)
        binding = _released_fit_inputs(
            _released_config_context(output), sample_ids,
            expected_phenotype=None, expected_kernels=kernels,
            expected_truth_hash=None,
        )
        truth = {
            "scan_pve": 0.0, "distance_unit": "bp", "causal_variants": [],
            "source": "deterministic_pre_fit_simulation",
            "seed": derive_seed(
                seed_design_hash, "fit", scenario.scenario_id, replicate,
                "pilot:loco_truth",
            ),
            "analysis_context": {
                "analysis_sample_ids_sha256": binding["analysis_sample_ids_sha256"],
                "joined_phenotype": binding["joined_phenotype"],
                "kernel_order": binding["kernel_order"],
                "kernel_fingerprints": binding["kernel_fingerprints"],
                "generated_config_sha256": hashlib.sha256(
                    (output / "configs" / "fit.generated.yaml").read_bytes()
                ).hexdigest(),
            },
        }
        truth["truth_hash"] = sha256_payload(truth)
        truth_path = design_root / "inputs" / f"loco-truth-{replicate}.json"
        truth_path.write_text(json.dumps(truth, sort_keys=True) + "\n")
        records[str(replicate)] = {
            "path": truth_path.relative_to(design_root).as_posix(),
            "sha256": hashlib.sha256(truth_path.read_bytes()).hexdigest(),
            "truth_hash": truth["truth_hash"], "source": truth["source"],
            "seed": truth["seed"],
            "generated_config_sha256": truth["analysis_context"]["generated_config_sha256"],
            "phenotype_sha256": truth["analysis_context"]["joined_phenotype"]["sha256"],
        }
        runs.append((output, kernels))
    (design_root / "design_lock.json").write_text(json.dumps({
        "design_hash": "4" * 64,
        "seed_design": seed_design, "seed_design_hash": seed_design_hash,
        "loco_truth_artifacts": {"pilot": {scenario.scenario_id: records}},
    }, sort_keys=True) + "\n")

    results = [
        run_fit_replicate(
            scenario, kernels, replicate=replicate, design_hash="4" * 64,
            sample_ids=np.asarray(sample_ids), fit_output_dir=output,
            loco_design_root=design_root,
        )
        for replicate, (output, kernels) in enumerate(runs)
    ]

    assert all(result["failure"]["failed"] is False for result in results)
    assert [result["released_scan_truth_binding"]["seed"] for result in results] == [
        derive_seed(seed_design_hash, "fit", scenario.scenario_id, replicate,
                    "pilot:loco_truth")
        for replicate in range(2)
    ]
    assert len({result["truth_hash"] for result in results}) == 2


def test_loco_forbids_comparator_injection_even_with_released_truth(tmp_path):
    output = tmp_path / "loco-no-comparator"
    sample_ids = [f"s{i:02d}" for i in range(24)]
    _write_released_fit_output(
        output, sample_ids=sample_ids, bootstrap=False, loco=True
    )
    _, _, kernels = _production_released_inputs(output)
    scenario = Scenario(
        "A.loco.cotton.pve_0", "fit", "formal", 1, 0,
        {"experiment": "loco", "scan_pve": 0.0, "panel": "cotton"},
    )
    seed_design = {"fixture": "formal-no-comparator"}
    seed_design_hash = sha256_payload(seed_design)
    binding = _released_fit_inputs(
        _released_config_context(output), sample_ids,
        expected_phenotype=None, expected_kernels=kernels,
        expected_truth_hash=None,
    )
    truth = {
        "scan_pve": 0.0, "distance_unit": "bp", "causal_variants": [],
        "source": "deterministic_pre_fit_simulation",
        "seed": derive_seed(
            seed_design_hash, "fit", scenario.scenario_id, 0,
            "formal:loco_truth",
        ),
        "analysis_context": {
            "analysis_sample_ids_sha256": binding["analysis_sample_ids_sha256"],
            "joined_phenotype": binding["joined_phenotype"],
            "kernel_order": binding["kernel_order"],
            "kernel_fingerprints": binding["kernel_fingerprints"],
            "generated_config_sha256": hashlib.sha256(
                (output / "configs" / "fit.generated.yaml").read_bytes()
            ).hexdigest(),
        },
    }
    truth["truth_hash"] = sha256_payload(truth)
    design_root = tmp_path / "design-no-comparator"
    (design_root / "inputs").mkdir(parents=True)
    truth_path = design_root / "inputs" / "loco-truth.json"
    truth_path.write_text(json.dumps(truth, sort_keys=True) + "\n")
    truth_sha = hashlib.sha256(truth_path.read_bytes()).hexdigest()
    (design_root / "design_lock.json").write_text(json.dumps({
        "design_hash": "8" * 64,
        "seed_design": seed_design, "seed_design_hash": seed_design_hash,
        "loco_truth_artifacts": {"formal": {scenario.scenario_id: {"0": {
            "path": "inputs/loco-truth.json", "sha256": truth_sha,
            "truth_hash": truth["truth_hash"], "source": truth["source"],
            "seed": truth["seed"],
            "generated_config_sha256": truth["analysis_context"]["generated_config_sha256"],
            "phenotype_sha256": truth["analysis_context"]["joined_phenotype"]["sha256"],
        }}}},
    }, sort_keys=True) + "\n")
    result = run_fit_replicate(
        scenario, kernels, replicate=0, design_hash="8" * 64,
        sample_ids=np.asarray(sample_ids), fit_output_dir=output,
        loco_design_root=design_root, scan_comparators={},
    )
    assert result["failure"]["failed"] is True


def test_released_loco_rejects_stale_sumstats_provenance(tmp_path):
    output = tmp_path / "loco-output"
    sample_ids = [f"s{i:02d}" for i in range(24)]
    summary = _write_released_fit_output(
        output, sample_ids=sample_ids, bootstrap=False, loco=True
    )
    summary["outputs"]["sumstats"] = str((tmp_path / "stale.tsv").resolve())
    (output / "results" / "summary_simulated_trait.json").write_text(
        json.dumps(summary)
    )

    with pytest.raises(ValueError, match="scan output paths"):
        run_scan_replicate(
            released_output_dir=output,
            expected_subgenomes=("A", "D"),
            expected_sample_ids=sample_ids,
            loco=True,
        )


def test_fit_runner_dispatches_scan_to_scan_helper():
    n = 12
    kernels = _toy_kernels(n=n)
    rng = np.random.default_rng(810)
    sample_ids = np.asarray([f"s{i:02d}" for i in range(n)])
    scan_inputs = build_scan_comparators(
        kernels,
        sample_ids=sample_ids,
        variant_ids={"A": ["a1"], "D": ["d1"]},
        standardized_variants={
            "A": rng.normal(size=(n, 1)),
            "D": rng.normal(size=(n, 1)),
        },
        phenotype=rng.normal(size=n),
        covariates=np.ones((n, 1)),
        truth_metadata=_scan_truth_metadata(),
    )
    scenario = Scenario(
        "A.scan.cotton.one_subgenome.pve_0p05",
        "fit",
        "pilot",
        1,
        0,
        {"experiment": "scan", "placement": "one_subgenome", "scan_pve": 0.05},
    )
    result = run_fit_replicate(
        scenario,
        kernels,
        replicate=0,
        design_hash="9" * 64,
        sample_ids=sample_ids,
        scan_comparators=scan_inputs,
    )

    assert result["failure"]["failed"] is False
    assert set(result["comparators"]) == {
        "canonical_multi_kernel",
        "pooled_trace_sum",
        "independent_subgenome",
    }
    assert "truth" not in result


@pytest.mark.parametrize(
    ("scenario_parameters", "message"),
    [
        ({"placement": "two_subgenomes", "scan_pve": 0.05}, "placement"),
        ({"placement": "one_subgenome", "scan_pve": 0.10}, "scan_pve"),
    ],
)
def test_scan_runner_rejects_truth_attached_to_wrong_arm(
    tmp_path, scenario_parameters, message
):
    n = 12
    kernels = _toy_kernels(n=n)
    rng = np.random.default_rng(901)
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
        truth_metadata=_scan_truth_metadata(),
    )
    scenario = Scenario(
        "A.scan.cotton.mismatched",
        "fit",
        "pilot",
        1,
        0,
        {"experiment": "scan", **scenario_parameters},
    )

    result = run_fit_replicate(
        scenario,
        kernels,
        replicate=0,
        design_hash="2" * 64,
        sample_ids=sample_ids,
        shard_path=tmp_path / "mismatch.json",
        scan_comparators=comparators,
    )

    assert result["failure"]["failed"] is True
    assert message in result["failure"]["message"]


def test_pilot_scan_rejects_realized_signal_pve_drift_over_one_point(tmp_path):
    n = 12
    kernels = _toy_kernels(n=n)
    rng = np.random.default_rng(902)
    sample_ids = np.asarray([f"s{i:02d}" for i in range(n)])
    truth = _scan_truth_metadata()
    truth["realized_signal_pve"] = 0.061
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
        truth_metadata=truth,
    )
    scenario = Scenario(
        "A.scan.cotton.one_subgenome.pve_0p05",
        "fit",
        "pilot",
        1,
        0,
        {"experiment": "scan", "placement": "one_subgenome", "scan_pve": 0.05},
    )

    result = run_fit_replicate(
        scenario,
        kernels,
        replicate=0,
        design_hash="1" * 64,
        sample_ids=sample_ids,
        shard_path=tmp_path / "drift.json",
        scan_comparators=comparators,
    )

    assert result["failure"]["failed"] is True
    assert "realized signal PVE" in result["failure"]["message"]


@pytest.mark.parametrize("mutation", ["phenotype", "pooled", "independent"])
def test_scan_resume_binds_every_current_mapping(monkeypatch, tmp_path, mutation):
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
        truth_metadata=_scan_truth_metadata(scan_pve=0.0),
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
    if mutation == "phenotype":
        comparators["phenotype"] = np.asarray(comparators["phenotype"]) + 0.01
    elif mutation == "pooled":
        comparators["pooled_trace_sum"] = {
            "pooled": np.asarray(comparators["pooled_trace_sum"]["pooled"]) + 0.01
        }
    else:
        comparators["independent_subgenome"] = {
            **comparators["independent_subgenome"],
            "A": {
                "A": np.asarray(
                    comparators["independent_subgenome"]["A"]["A"]
                )
                + 0.01
            },
        }

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


def test_scan_bp_positions_are_serialized_and_bound_into_request_hash(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        "scripts.benchmarks.v201.track_fit.run_scan_replicate",
        lambda *args, **kwargs: {
            "scan_arm": "primary",
            "failure": {"failed": False, "error_type": None, "message": None},
        },
    )
    n = 12
    kernels = _toy_kernels(n=n)
    rng = np.random.default_rng(45)
    sample_ids = np.asarray([f"s{i:02d}" for i in range(n)])
    variant_ids = {"A": ["a1"], "D": ["d1"]}
    positions = {"A": [10100], "D": [20200]}

    def build(current_positions):
        return build_scan_comparators(
            kernels,
            sample_ids=sample_ids,
            variant_ids=variant_ids,
            variant_positions_bp=current_positions,
            standardized_variants={
                "A": rng.normal(size=(n, 1)),
                "D": rng.normal(size=(n, 1)),
            },
            phenotype=rng.normal(size=n),
            covariates=np.ones((n, 1)),
            truth_metadata=_scan_truth_metadata(scan_pve=0.0),
        )

    comparators = build(positions)
    scenario = Scenario(
        "A.scan.cotton.one_subgenome.pve_0",
        "fit",
        "pilot",
        1,
        0,
        {"experiment": "scan", "placement": "one_subgenome", "scan_pve": 0.0},
    )
    first = run_fit_replicate(
        scenario,
        kernels,
        replicate=0,
        design_hash="6" * 64,
        sample_ids=sample_ids,
        scan_comparators=comparators,
    )

    assert first["failure"]["failed"] is False
    assert first["scan_request_manifest"]["variant_positions_bp"] == positions
    assert first["scan_request_manifest"]["distance_unit"] == "bp"
    assert first["request_hash"] == sha256_payload(first["request_manifest"])
    original_hash = first["request_hash"]
    comparators["context"]["variant_positions_bp"]["A"][0] += 1
    second = run_fit_replicate(
        scenario,
        kernels,
        replicate=0,
        design_hash="6" * 64,
        sample_ids=sample_ids,
        scan_comparators=comparators,
    )
    assert second["request_hash"] != original_hash


def test_formal_scan_without_explicit_bp_positions_fails_closed(monkeypatch):
    monkeypatch.setattr(
        "scripts.benchmarks.v201.track_fit.run_scan_replicate",
        lambda *args, **kwargs: {
            "scan_arm": "primary",
            "failure": {"failed": False, "error_type": None, "message": None},
        },
    )
    n = 12
    kernels = _toy_kernels(n=n)
    rng = np.random.default_rng(46)
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
        truth_metadata=_scan_truth_metadata(scan_pve=0.0),
    )
    scenario = Scenario(
        "A.scan.cotton.one_subgenome.pve_0",
        "fit",
        "formal",
        1,
        0,
        {"experiment": "scan", "placement": "one_subgenome", "scan_pve": 0.0},
    )

    result = run_fit_replicate(
        scenario,
        kernels,
        replicate=0,
        design_hash="7" * 64,
        sample_ids=sample_ids,
        scan_comparators=comparators,
    )

    assert result["failure"]["failed"] is True
    assert "explicit bp positions" in result["failure"]["message"]


def test_comparator_preflight_is_two_row_frozen_and_hash_checked(tmp_path):
    gcta = tmp_path / "gcta64"
    gcta.write_text("#!/bin/sh\necho 'GCTA 1.94.1'\n")
    gcta.chmod(0o755)
    path = tmp_path / "comparator_preflight.tsv"

    written = write_comparator_preflight(
        path,
        **_preflight_inputs(),
        executables={"GCTA": gcta, "GEMMA": tmp_path / "missing-gemma"},
    )
    loaded = read_comparator_preflight(path, expected_hash=written["sha256"])

    assert [row["comparator"] for row in loaded["rows"]] == ["GCTA", "GEMMA"]
    assert loaded["rows"][0]["status"] == "UNAVAILABLE_OR_NONCOMPARABLE"
    assert loaded["rows"][0]["executable_basename"] == "gcta64"
    assert loaded["rows"][0]["parsed_identity"] == "GCTA"
    assert loaded["rows"][0]["parsed_version"] == "1.94.1"
    assert loaded["rows"][0]["version_returncode"] == 0
    assert loaded["rows"][0]["smoke_status"] == "NO_ADAPTER"
    assert len(loaded["rows"][0]["binary_sha256"]) == 64
    assert len(loaded["rows"][0]["genotype_hash"]) == 64
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


def test_comparator_preflight_records_false_tool_identity_and_returncode(tmp_path):
    impostor = tmp_path / "not-gcta"
    impostor.write_text("#!/bin/sh\necho 'GCTA 1.94.1'\n")
    impostor.chmod(0o755)
    result = write_comparator_preflight(
        tmp_path / "bad-identity.tsv",
        **_preflight_inputs(),
        executables={"GCTA": impostor, "GEMMA": tmp_path / "missing-gemma"},
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
    )
    assert result["rows"][0]["status"] == "UNAVAILABLE_OR_NONCOMPARABLE"
    assert result["rows"][0]["version_returncode"] == 1


def test_preflight_reader_never_reexecutes_binary(monkeypatch, tmp_path):
    gcta = tmp_path / "gcta64"
    gcta.write_text("#!/bin/sh\necho 'GCTA 1.94.1'\n")
    gcta.chmod(0o755)
    written = write_comparator_preflight(
        tmp_path / "comparator_preflight.tsv",
        **_preflight_inputs(),
        executables={"GCTA": gcta, "GEMMA": tmp_path / "missing-gemma"},
    )
    gcta.write_text("#!/bin/sh\necho mutated\n")
    monkeypatch.setattr(
        "scripts.benchmarks.v201.track_fit.subprocess.run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("reader re-executed comparator")
        ),
    )

    loaded = read_comparator_preflight(
        written["path"], expected_hash=written["sha256"]
    )

    assert loaded["rows"][0]["binary_sha256"] == written["rows"][0][
        "binary_sha256"
    ]


def test_preflight_read_error_becomes_immutable_failure_shard(tmp_path):
    scenario = Scenario(
        "A.recovery.cotton.balanced",
        "fit",
        "pilot",
        1,
        0,
        {"experiment": "recovery", "allocation": "balanced", "total_pve": 0.4},
    )
    shard = tmp_path / "replicate.json"
    result = run_fit_replicate(
        scenario,
        _toy_kernels(n=12),
        replicate=0,
        design_hash="a" * 64,
        sample_ids=np.asarray([f"s{i:02d}" for i in range(12)]),
        shard_path=shard,
        comparator_preflight_path=tmp_path / "missing.tsv",
        comparator_preflight_hash="1" * 64,
    )

    assert result["failure"]["failed"] is True
    assert result["failure"]["error_type"] == "FileNotFoundError"
    assert json.loads(shard.read_text()) == result


@pytest.mark.parametrize("mutation", ["binary", "genotype"])
def test_scan_runner_rejects_mutated_preflight_evidence(tmp_path, mutation):
    n = 12
    kernels = _toy_kernels(n=n)
    inputs = _preflight_inputs(n=n)
    sample_ids = inputs["sample_ids"]
    comparators = build_scan_comparators(
        kernels,
        sample_ids=np.asarray(sample_ids),
        variant_ids=inputs["variant_ids"],
        standardized_variants=inputs["genotypes"],
        phenotype=np.linspace(-1.0, 1.0, n),
        covariates=inputs["covariates"],
        qc_declaration=inputs["qc_declaration"],
        truth_metadata=_scan_truth_metadata(scan_pve=0.0),
    )
    gcta = tmp_path / "gcta64"
    gcta.write_text("#!/bin/sh\necho 'GCTA 1.94.1'\n")
    gcta.chmod(0o755)
    preflight_inputs = dict(inputs)
    if mutation == "genotype":
        preflight_inputs["genotypes"] = {
            **inputs["genotypes"],
            "A": inputs["genotypes"]["A"] + 1.0,
        }
    preflight = write_comparator_preflight(
        tmp_path / "comparator_preflight.tsv",
        **preflight_inputs,
        executables={
            "GCTA": gcta,
            "GEMMA": tmp_path / "missing-gemma",
        },
    )
    if mutation == "binary":
        gcta.write_text("#!/bin/sh\necho 'GCTA 9.9.9'\n")
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
    assert f"{mutation}_hash differs" in result["failure"]["message"]
