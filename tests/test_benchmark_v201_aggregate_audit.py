import csv
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from scripts.benchmarks.v201 import aggregate as aggregate_module
from scripts.benchmarks.v201.aggregate import (
    ACCEPTANCE_RULES,
    TABLE_SCHEMAS,
    aggregate_benchmark,
)
from scripts.benchmarks.v201.audit import (
    BenchmarkAuditError,
    _audit_application_scenario,
    _audit_fit_scenario,
    _audit_scaling_scenario,
    _validate_bank_envelope,
    audit_benchmark,
    summarize_binomial,
)
from scripts.benchmarks.v201.contracts import (
    Scenario,
    canonical_json,
    derive_seed,
    sha256_payload,
)
from scripts.benchmarks.v201.shards import write_shard_exclusive
from scripts.benchmarks.v201.track_fit import run_fit_replicate


def _write_registry(root: Path, *, stage: str, total: int) -> Path:
    path = root / "scenario_registry.tsv"
    path.parent.mkdir(parents=True, exist_ok=True)
    parameters = {
        "experiment": "end2end",
        "null_model": "gaussian",
        "qa_only": stage == "pilot",
    }
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow((
            "scenario_id", "track", "stage", "replicates", "bootstrap_B",
            "parameters",
        ))
        writer.writerow(("core", "omnib", stage, total, 199 if stage == "pilot" else 2000,
                         canonical_json(parameters)))
    return path


def _write_null_shards(
    root: Path, monkeypatch, *, rejections: int, total: int,
    stage: str = "formal",
) -> Path:
    registry = _write_registry(root, stage=stage, total=total)
    parameters = {
        "experiment": "end2end", "null_model": "gaussian",
        "qa_only": stage == "pilot",
    }
    canonical_scenario = Scenario(
        "core", "omnib", stage, total, 199 if stage == "pilot" else 2000,
        parameters,
    )
    monkeypatch.setattr(
        aggregate_module, "build_scenarios", lambda requested: (
            [canonical_scenario] if requested == stage else []
        ),
    )
    inputs = root / "inputs"
    configs = root / "configs"
    inputs.mkdir(exist_ok=True)
    configs.mkdir(exist_ok=True)
    (inputs / "manifest.tsv").write_text("input\tsha256\nfixture\t" + "1" * 64 + "\n")
    (inputs / "comparator_preflight.tsv").write_text(
        "comparator\tstatus\nfixture\tUNAVAILABLE_OR_NONCOMPARABLE\n"
    )
    config = configs / "fixture.yaml"
    config.write_text("fixture: true\n", encoding="utf-8")
    config_hash = hashlib.sha256(config.read_bytes()).hexdigest()
    (configs / "manifest.tsv").write_text(
        f"path\tsha256\nconfigs/fixture.yaml\t{config_hash}\n",
        encoding="utf-8",
    )
    design_payload = {
        "schema": "homoeogwas-v201-benchmark-lock-v1",
        "stage": stage,
        "ordered_scenario_ids": ["core"],
        "scenario_registry_sha256": hashlib.sha256(registry.read_bytes()).hexdigest(),
        "master_seed": 20260830,
        "software": {
            "version": "2.0.1",
            "git_commit": subprocess.run(
                ["git", "rev-parse", "HEAD"], check=True, capture_output=True,
                text=True,
            ).stdout.strip(),
        },
        "input_manifest_sha256": hashlib.sha256(
            (inputs / "manifest.tsv").read_bytes()
        ).hexdigest(),
        "comparator_preflight_sha256": hashlib.sha256(
            (inputs / "comparator_preflight.tsv").read_bytes()
        ).hexdigest(),
        "config_manifest_sha256": hashlib.sha256(
            (configs / "manifest.tsv").read_bytes()
        ).hexdigest(),
        "config_hashes": {"configs/fixture.yaml": config_hash},
        "acceptance_rules": ACCEPTANCE_RULES,
    }
    design_hash = sha256_payload(design_payload)
    (root / "design_lock.json").write_text(
        canonical_json({**design_payload, "design_hash": design_hash}) + "\n",
        encoding="utf-8",
    )
    family_ids = ["group-1"]
    family_manifest = {
        "subgenomes": ["A", "B"], "group_ids": family_ids,
        "genes": [["gene-A", "gene-B"]],
    }
    family_hash = sha256_payload(family_manifest)
    context_manifest = {
        "subgenomes": [], "family": family_manifest,
        "sample_idx": {"values": [0, 1], "sha256": "2" * 64},
        "phenotype": {"shape": [2], "dtype": "float64", "sha256": "3" * 64},
    }
    context_fingerprint = sha256_payload(context_manifest)
    scenario_record = {
        "scenario_id": "core", "track": "omnib", "stage": stage,
        "replicates": total, "bootstrap_B": 199 if stage == "pilot" else 2000,
        "parameters": {
            "experiment": "end2end", "null_model": "gaussian",
            "qa_only": stage == "pilot",
        },
    }
    for index in range(total):
        rejected = index < rejections
        bootstrap_b = 199 if stage == "pilot" else 2000
        k = int(0.05 * (bootstrap_b + 1))
        null_minima = [0.005] * (k - 1) + [0.05] + [0.5] * (bootstrap_b - k)
        observed = 0.01 if rejected else 0.25
        adjusted = (1 + sum(value <= observed for value in null_minima)) / (
            bootstrap_b + 1
        )
        response_seed = derive_seed(
            design_hash, "omnib", "core", index, f"{stage}:heldout",
        )
        calibration_seed = derive_seed(
            design_hash, "omnib", "core", index, f"{stage}:calibration",
        )
        request_hash = sha256_payload({
            "design_hash": design_hash,
            "context_fingerprint": context_fingerprint,
            "request": {
                "entrypoint": "run_omnib_replicate", "scenario": scenario_record,
                "replicate": index, "experiment": "end2end",
                "canonical_bank": None, "n_jobs": 1,
            },
        })
        payload = {
            "track": "omnib",
            "scenario_id": "core",
            "replicate": index,
            "design_hash": design_hash,
            "stage": stage,
            "formal": stage == "formal",
            "qa_only": stage == "pilot",
            "inference_status": (
                "formal" if stage == "formal"
                else "noninferential_do_not_threshold"
            ),
            "experiment": "end2end",
            "mode": "group",
            "statistic": "omniB",
            "hypothesis_unit": "group",
            "family_scope": "primary_only",
            "subset_order": 2,
            "pair_edges_per_group": 1,
            "direct_higher_order_term": False,
            "bootstrap_B": bootstrap_b,
            "request_hash": request_hash,
            "context_fingerprint": context_fingerprint,
            "requested_jobs": 1,
            "effective_jobs": 1,
            "parallel_backend": "serial",
            "worker_pids": [1000 + index],
            "family_ids": family_ids,
            "family_manifest": family_manifest,
            "family_hash": family_hash,
            "family_order_hash": sha256_payload(family_ids),
            "context_manifest": context_manifest,
            "response_seed": response_seed,
            "response_seed_id": (
                f"{stage}:heldout:core:{index}:{response_seed:016x}"
            ),
            "calibration_seed": calibration_seed,
            "calibration_seed_id": (
                f"{stage}:calibration:core:{index}:{calibration_seed:016x}"
            ),
            "observed_group_p": [observed],
            "adjusted_p": [adjusted],
            "adjusted_decisions": [rejected],
            "bootstrap_minp": {
                "alpha": 0.05,
                "method": "parametric_bootstrap_minp_plus_one",
                "B": bootstrap_b,
                "empirical_p": adjusted,
                "adjusted_p_local": [adjusted],
                "rejected_local": [0] if rejected else [],
                "rejected": rejected,
                "threshold": 0.05,
                "threshold_comparator": "strict_less_than",
            },
            "null_minima": null_minima,
            "formal_rejections": family_ids if rejected and stage == "formal" else [],
            "qa_diagnostic_rejections": family_ids if rejected else [],
            "runtime_seconds": 0.1,
            "failure": {"failed": False, "error_type": None, "message": None},
        }
        write_shard_exclusive(
            root / stage / "omnib" / "core" / f"replicate-{index:06d}.json",
            payload,
        )
    return root


def test_standard_wilson_fixture_and_core_fwer_gate(tmp_path, monkeypatch):
    summary = summarize_binomial(50, 1000)
    assert summary["ci_low"] == pytest.approx(0.0381302624)
    assert summary["ci_high"] == pytest.approx(0.0653138202)

    root = _write_null_shards(tmp_path, monkeypatch, rejections=50, total=1000)
    report = audit_benchmark(root)
    gate = report.gates["B.core_fwer.core"]
    assert gate.passed is True
    assert gate.upper_ci < 0.075
    assert gate.failures == 0


def test_audit_seals_then_detects_changed_shard(tmp_path, monkeypatch):
    root = _write_null_shards(tmp_path, monkeypatch, rejections=1, total=20)
    first = audit_benchmark(root)
    assert len(first.shard_manifest) == 20

    shard = next((root / "formal" / "omnib").rglob("replicate-*.json"))
    payload = json.loads(shard.read_text(encoding="utf-8"))
    payload["adjusted_decisions"] = [not payload["adjusted_decisions"][0]]
    shard.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    with pytest.raises(BenchmarkAuditError, match="shard hash mismatch"):
        audit_benchmark(root)


def test_decision_disagreement_and_seed_overlap_fail_closed(tmp_path, monkeypatch):
    root = _write_null_shards(tmp_path, monkeypatch, rejections=1, total=20)
    shard = next((root / "formal" / "omnib").rglob("replicate-*.json"))
    payload = json.loads(shard.read_text(encoding="utf-8"))
    payload["adjusted_decisions"] = [False]
    payload["calibration_seed_id"] = payload["response_seed_id"]
    shard.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    with pytest.raises(
        BenchmarkAuditError, match="decision disagreement|seed roles overlap",
    ):
        audit_benchmark(root)


def test_aggregation_writes_only_exact_tables_with_strict_headers(tmp_path, monkeypatch):
    root = _write_null_shards(tmp_path, monkeypatch, rejections=1, total=20)
    result = aggregate_benchmark(root)
    assert set(path.name for path in result.table_paths) == set(TABLE_SCHEMAS)
    assert set(TABLE_SCHEMAS) == {
        "fit_pve_recovery.tsv", "fit_pve_coverage.tsv", "fit_scan_metrics.tsv",
        "omnib_null_replicates.tsv", "omnib_power_replicates.tsv",
        "omnib_encoding_robustness.tsv", "omnib_family_manifest.tsv",
        "scaling_runs.tsv", "cross_species_application.tsv",
        "benchmark_acceptance.tsv",
    }
    for name, fields in TABLE_SCHEMAS.items():
        with (root / "tables" / name).open(encoding="utf-8", newline="") as handle:
            assert next(csv.reader(handle, delimiter="\t")) == list(fields)
    with (root / "tables" / "omnib_null_replicates.tsv").open(
        encoding="utf-8", newline="",
    ) as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert [int(row["replicate"]) for row in rows] == list(range(20))


def test_pilot_is_noninferential_and_has_no_formal_decision(tmp_path, monkeypatch):
    root = _write_null_shards(
        tmp_path, monkeypatch, rejections=1, total=20, stage="pilot",
    )
    report = audit_benchmark(root)
    assert report.stage == "pilot"
    assert report.inference_status == "noninferential_do_not_threshold"
    assert report.formal_overall_passed is None
    assert all(gate.passed is None for gate in report.gates.values())
    audit_json = json.loads(
        (root / "audit" / "benchmark_audit.json").read_text(encoding="utf-8")
    )
    assert "formal_overall_passed" not in audit_json
    assert all("passed" not in gate for gate in audit_json["gates"].values())
    acceptance = (root / "tables" / "benchmark_acceptance.tsv").read_text(
        encoding="utf-8",
    )
    assert "noninferential_do_not_threshold" in acceptance
    assert "formal_pass" not in acceptance


def test_explicit_failed_shard_is_retained_in_failure_denominator(tmp_path, monkeypatch):
    root = _write_null_shards(tmp_path, monkeypatch, rejections=0, total=20)
    shard = next((root / "formal" / "omnib").rglob("replicate-*.json"))
    payload = json.loads(shard.read_text(encoding="utf-8"))
    payload["failure"] = {
        "failed": True, "error_type": "SyntheticFailure", "message": "retained",
    }
    for field in (
        "observed_group_p", "adjusted_p", "adjusted_decisions", "bootstrap_minp",
        "formal_rejections", "qa_diagnostic_rejections",
    ):
        payload.pop(field)
    shard.write_text(canonical_json(payload) + "\n", encoding="utf-8")

    report = audit_benchmark(root)
    gate = report.gates["B.core_fwer.core"]
    assert gate.failures == 1
    assert gate.failure_rate == pytest.approx(0.05)
    assert gate.passed is False
    null_table = (root / "tables" / "omnib_null_replicates.tsv").read_text(
        encoding="utf-8",
    )
    assert "SyntheticFailure" in null_table


def test_extra_opposite_stage_shard_is_rejected(tmp_path, monkeypatch):
    root = _write_null_shards(tmp_path, monkeypatch, rejections=0, total=20)
    source = next((root / "formal" / "omnib").rglob("replicate-*.json"))
    extra = root / "pilot" / "omnib" / "core" / "replicate-000000.json"
    extra.parent.mkdir(parents=True)
    extra.write_bytes(source.read_bytes())
    with pytest.raises(BenchmarkAuditError, match="unexpected shard|duplicate shard"):
        audit_benchmark(root)


@pytest.mark.parametrize("fake_track", ["fit", "application", "scaling", "conditional"])
def test_self_selected_formal_registry_cannot_pass_as_canonical(
    tmp_path, monkeypatch, fake_track,
):
    root = _write_null_shards(tmp_path, monkeypatch, rejections=0, total=20)
    monkeypatch.undo()
    with pytest.raises(BenchmarkAuditError, match="canonical design"):
        audit_benchmark(root)


def test_fake_fit_summary_without_truth_is_rejected():
    scenario = Scenario(
        "A.recovery.cotton.balanced", "fit", "formal", 300, 0,
        {"qa_only": False, "panel": "cotton", "experiment": "recovery",
         "allocation": "balanced", "total_pve": 0.4},
    )
    context = {"kernel_order": ["A", "D"], "kernel_fingerprints": {},
               "sample_manifest": {}}
    request = {
        "design_hash": "d" * 64, "context_fingerprint": sha256_payload(context),
        "scenario": scenario.to_dict(), "replicate": 0, "seed": 1,
        "source": "fit_multi_reml", "released_source_manifest": None,
        "released_request_binding": None, "coverage_request_binding": None,
        "truth_hash": None, "truth_manifest": None,
        "scan_context_fingerprint": None, "scan_request_manifest": None,
        "comparator_preflight": None,
    }
    payload = {
        "design_hash": "d" * 64, "replicate": 0, "seed": 1,
        "result_source": "fit_multi_reml", "fit_context_manifest": context,
        "context_fingerprint": sha256_payload(context),
        "request_manifest": request, "request_hash": sha256_payload(request),
        "truth_hash": None, "truth_manifest": None,
        "released_source_manifest": None, "released_request_binding": None,
        "coverage_request_binding": None, "scan_context_fingerprint": None,
        "scan_request_manifest": None, "comparator_preflight_evidence": None,
        "failure": {"failed": False}, "true_pve": {"A": 0.2, "D": 0.2, "e": 0.6},
        "estimated_pve": {"A": 0.2, "D": 0.2, "e": 0.6},
        "pve_bias": {"A": 0.0, "D": 0.0, "e": 0.0},
    }
    with pytest.raises(BenchmarkAuditError, match="frozen truth"):
        _audit_fit_scenario(payload, scenario)


def test_real_track_a_recovery_payload_is_independently_recomputed():
    rng = np.random.default_rng(91)
    kernels = {}
    for name in ("A", "D"):
        values = rng.normal(size=(36, 18))
        values -= values.mean(axis=0)
        kernel = values @ values.T / values.shape[1]
        kernels[name] = kernel / (np.trace(kernel) / 36)
    scenario = Scenario(
        "A.recovery.cotton.balanced", "fit", "pilot", 1, 199,
        {"qa_only": True, "panel": "cotton", "experiment": "recovery",
         "allocation": "balanced", "total_pve": 0.4},
    )
    payload = run_fit_replicate(
        scenario, kernels, replicate=0, design_hash="d" * 64,
        sample_ids=np.asarray([f"sample-{index}" for index in range(36)]),
    )
    assert payload["failure"]["failed"] is False
    assert _audit_fit_scenario(payload, scenario) is True
    assert payload["audit_derived"]["dominant_correct"] is None


def test_fake_scaling_boolean_summary_without_repeats_is_rejected():
    anchor = {
        "anchor_id": "small_qa", "n": 192, "groups": 192, "copies": 3,
        "edges": 3, "bootstrap_B": 199, "jobs": [1, 4, 8], "repeats": 3,
    }
    scenario = Scenario(
        "C.small_qa", "scaling", "formal", 3, 199,
        {"qa_only": False, "anchor": anchor},
    )
    payload = {
        "failure": {"failed": False},
        "summary": {"exact_result_hash_identity": True,
                    "release_policy": {"accepted": True}},
    }
    with pytest.raises(BenchmarkAuditError, match="jobs matrix"):
        _audit_scaling_scenario(payload, scenario)


def test_garbage_application_status_is_rejected():
    scenario = Scenario(
        "D.wheat", "application", "formal", 1, 0,
        {"qa_only": False, "species": "wheat", "read_only": True,
         "rescan": False},
    )
    payload = {
        "failure": {"failed": False},
        "rows": [{"analysis_id": "fake", "species": "wheat",
                  "repair_required": False, "status": "GARBAGE",
                  "audit_status": "GARBAGE"}],
    }
    with pytest.raises(BenchmarkAuditError, match="schema|unauthoritative"):
        _audit_application_scenario(payload, scenario)


def test_underfilled_conditional_bank_is_rejected():
    bank = {
        "canonical_role": "heldout", "seeds": list(range(100)),
        "seed_ids": [f"seed-{index}" for index in range(100)],
        "response_metadata": [
            {"seed": index, "seed_id": f"seed-{index}"} for index in range(100)
        ],
        "response_shape": [10, 100],
        "failure": {"failed": False, "failed_response_indices": []},
    }
    with pytest.raises(BenchmarkAuditError, match="count/identity"):
        _validate_bank_envelope(bank, 2_000, "heldout")
