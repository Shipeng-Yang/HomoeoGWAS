import csv
import hashlib
import json
from pathlib import Path

import pytest

from scripts.benchmarks.v201.aggregate import TABLE_SCHEMAS, aggregate_benchmark
from scripts.benchmarks.v201.audit import (
    BenchmarkAuditError,
    audit_benchmark,
    summarize_binomial,
)
from scripts.benchmarks.v201.contracts import canonical_json, derive_seed, sha256_payload
from scripts.benchmarks.v201.shards import write_shard_exclusive


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
    root: Path, *, rejections: int, total: int, stage: str = "formal",
) -> Path:
    registry = _write_registry(root, stage=stage, total=total)
    design_payload = {
        "schema": "homoeogwas-v201-benchmark-lock-v1",
        "stage": stage,
        "ordered_scenario_ids": ["core"],
        "scenario_registry_sha256": hashlib.sha256(registry.read_bytes()).hexdigest(),
    }
    design_hash = sha256_payload(design_payload)
    (root / "design_lock.json").write_text(
        canonical_json({**design_payload, "design_hash": design_hash}) + "\n",
        encoding="utf-8",
    )
    family_ids = ["group-1"]
    family_hash = sha256_payload(family_ids)
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
        adjusted = 0.04 if rejected else 0.50
        response_seed = derive_seed(
            design_hash, "omnib", "core", index, f"{stage}:heldout",
        )
        calibration_seed = derive_seed(
            design_hash, "omnib", "core", index, f"{stage}:calibration",
        )
        request_hash = sha256_payload({
            "design_hash": design_hash,
            "context_fingerprint": "c" * 64,
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
            "bootstrap_B": 199 if stage == "pilot" else 2000,
            "request_hash": request_hash,
            "context_fingerprint": "c" * 64,
            "requested_jobs": 1,
            "effective_jobs": 1,
            "parallel_backend": "serial",
            "worker_pids": [1000 + index],
            "family_ids": family_ids,
            "family_hash": family_hash,
            "family_order_hash": family_hash,
            "response_seed": response_seed,
            "response_seed_id": (
                f"{stage}:heldout:core:{index}:{response_seed:016x}"
            ),
            "calibration_seed": calibration_seed,
            "calibration_seed_id": (
                f"{stage}:calibration:core:{index}:{calibration_seed:016x}"
            ),
            "observed_group_p": [0.01 if rejected else 0.25],
            "adjusted_p": [adjusted],
            "adjusted_decisions": [rejected],
            "bootstrap_minp": {
                "adjusted_p_local": [adjusted],
                "rejected_local": [0] if rejected else [],
                "threshold": 0.05,
            },
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


def test_standard_wilson_fixture_and_core_fwer_gate(tmp_path):
    summary = summarize_binomial(50, 1000)
    assert summary["ci_low"] == pytest.approx(0.0381302624)
    assert summary["ci_high"] == pytest.approx(0.0653138202)

    root = _write_null_shards(tmp_path, rejections=50, total=1000)
    report = audit_benchmark(root)
    gate = report.gates["B.core_fwer.core"]
    assert gate.passed is True
    assert gate.upper_ci < 0.075
    assert gate.failures == 0


def test_audit_seals_then_detects_changed_shard(tmp_path):
    root = _write_null_shards(tmp_path, rejections=1, total=20)
    first = audit_benchmark(root)
    assert len(first.shard_manifest) == 20

    shard = next((root / "formal" / "omnib").rglob("replicate-*.json"))
    payload = json.loads(shard.read_text(encoding="utf-8"))
    payload["adjusted_decisions"] = [not payload["adjusted_decisions"][0]]
    shard.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    with pytest.raises(BenchmarkAuditError, match="shard hash mismatch"):
        audit_benchmark(root)


def test_decision_disagreement_and_seed_overlap_fail_closed(tmp_path):
    root = _write_null_shards(tmp_path, rejections=1, total=20)
    shard = next((root / "formal" / "omnib").rglob("replicate-*.json"))
    payload = json.loads(shard.read_text(encoding="utf-8"))
    payload["adjusted_decisions"] = [False]
    payload["calibration_seed_id"] = payload["response_seed_id"]
    shard.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    with pytest.raises(
        BenchmarkAuditError, match="decision disagreement|seed roles overlap",
    ):
        audit_benchmark(root)


def test_aggregation_writes_only_exact_tables_with_strict_headers(tmp_path):
    root = _write_null_shards(tmp_path, rejections=1, total=20)
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


def test_pilot_is_noninferential_and_has_no_formal_decision(tmp_path):
    root = _write_null_shards(
        tmp_path, rejections=1, total=20, stage="pilot",
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


def test_explicit_failed_shard_is_retained_in_failure_denominator(tmp_path):
    root = _write_null_shards(tmp_path, rejections=0, total=20)
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


def test_extra_opposite_stage_shard_is_rejected(tmp_path):
    root = _write_null_shards(tmp_path, rejections=0, total=20)
    source = next((root / "formal" / "omnib").rglob("replicate-*.json"))
    extra = root / "pilot" / "omnib" / "core" / "replicate-000000.json"
    extra.parent.mkdir(parents=True)
    extra.write_bytes(source.read_bytes())
    with pytest.raises(BenchmarkAuditError, match="unexpected shard|duplicate shard"):
        audit_benchmark(root)
