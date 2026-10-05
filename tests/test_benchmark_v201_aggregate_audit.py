import copy
import csv
import hashlib
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from scripts.benchmarks.v201 import aggregate as aggregate_module
from scripts.benchmarks.v201.aggregate import (
    ACCEPTANCE_RULES,
    TABLE_SCHEMAS,
    BenchmarkAggregateError,
    _validate_comparator_preflights,
    _validate_config_manifest,
    _validate_context_artifact,
    _validate_input_manifest,
    _validate_loco_truth_artifacts,
    aggregate_benchmark,
    seed_design_payload,
    table_schemas,
)
from scripts.benchmarks.v201.audit import (
    AuditGate,
    BenchmarkAuditError,
    _audit_application_scenario,
    _audit_comparator_probe_series,
    _audit_families_and_parallel,
    _audit_fit_scenario,
    _audit_power_evidence,
    _audit_robustness_record,
    _audit_scaling_scenario,
    _conditional_score_matrices,
    _fit_scan_gates,
    _omnib_power_gates,
    _pilot_and_engineering_gates,
    _validate_bank_envelope,
    audit_benchmark,
    summarize_binomial,
)
from scripts.benchmarks.v201.comparators import METHOD_NAMES
from scripts.benchmarks.v201.contracts import (
    ScalingAnchor,
    Scenario,
    canonical_json,
    derive_seed,
    sha256_payload,
)
from scripts.benchmarks.v201.shards import write_shard_exclusive
from scripts.benchmarks.v201.track_fit import run_fit_replicate
from scripts.benchmarks.v201.track_omnib import (
    _ranking_metrics,
    build_synthetic_omnib_context,
    run_conditional_bank,
    run_global_vc_bank,
    run_omnib_replicate,
    run_power_replicate,
)
from scripts.benchmarks.v201.track_scaling import (
    NUMERIC_THREAD_ENV,
    ScalingAnchorRun,
    summarize_anchor,
)



@pytest.fixture(autouse=True)
def _v201_release_target(monkeypatch):
    import homoeogwas
    monkeypatch.setattr(homoeogwas, "__version__", "2.0.1")


def _comparator_probe(width: int) -> dict[str, object]:
    return {
        "schema": "homoeogwas-snpxsnp-resource-probe-v1",
        "panel_id": "REALG.CGVD1245",
        "sample_context": "full",
        "family_size": 80,
        "copies": 2,
        "response_width": width,
        "design_hash": "a" * 64,
        "context_fingerprint": "b" * 64,
        "prepared_design_sha256": "c" * 64,
        "member_family_sha256": "d" * 64,
        "scorer_wall_seconds": float(width),
        "scorer_cpu_seconds": float(width * 2),
        "peak_parent_rss_bytes": 4 * 1024 ** 3,
        "peak_aggregate_pss_bytes": 8 * 1024 ** 3,
        "output_bytes": width * 100,
        "offered_pair_count": 7_405,
        "design_nonestimable_pair_count": 5,
        "tested_pair_count": 7_400,
        "nonfinite_pair_score_count": 0,
        "failed_response_indices": [],
        "gated_marker_count_by_gene": {"A|g1": 11, "D|g1": 9},
        "requested_jobs": 1,
        "effective_jobs": 1,
        "parallel_backend": "serial",
        "worker_pids": [12345],
        "inference_status": "noninferential_resource_probe",
        "execution_authorized": False,
    }


def test_audit_wraps_comparator_resource_probe_validation_fail_closed():
    projection = _audit_comparator_probe_series(
        [_comparator_probe(width) for width in (1, 5, 20)]
    )
    assert projection["accepted"] is True

    malformed = [_comparator_probe(width) for width in (1, 5, 20)]
    malformed[-1].pop("peak_aggregate_pss_bytes")
    with pytest.raises(BenchmarkAuditError, match="resource probe"):
        _audit_comparator_probe_series(malformed)


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
    fixture_input = inputs / "fixture.txt"
    fixture_input.write_text("fixture\n", encoding="utf-8")
    (inputs / "comparator_preflight.tsv").write_text(
        "comparator\tstatus\nfixture\tUNAVAILABLE_OR_NONCOMPARABLE\n"
    )
    input_records = {}
    manifest_lines = ["path\tsize\tsha256\ttype"]
    for path, input_type in (
        (fixture_input, "fixture"),
        (inputs / "comparator_preflight.tsv", "comparator_preflight"),
    ):
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest_lines.append(f"{relative}\t{size}\t{digest}\t{input_type}")
        input_records[relative] = {
            "size": size, "sha256": digest, "type": input_type,
        }
    (inputs / "manifest.tsv").write_text(
        "\n".join(manifest_lines) + "\n", encoding="utf-8",
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
        "target_release": {
            "version": "2.0.1", "tag": "v2.0.1",
            "git_commit": "015e023439addf2cca658cd01720ff6f856023be",
        },
        "harness": {
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
        "input_records": input_records,
        "scenario_config_bindings": {"core": "configs/fixture.yaml"},
        "benchmark_contexts": {},
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
            "null_model": "gaussian",
            "null_generation": {
                "kind": "gaussian", "canonical_kind": "gaussian",
            },
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
            "qa_diagnostic_rejections": family_ids if rejected else [],
            "runtime_seconds": 0.1,
            "failure": {
                "failed": False,
                "status": "completed",
                "error_type": None,
                "message": None,
                "observed_failed": False,
                "response_diagnostics": {
                    "failed_response_indices": [],
                    "failed_response_indices_by_component": {
                        "minor_burden": [], "pc1": [], "kernel_hadamard": [],
                    },
                    "nonfinite_component_counts": [],
                    "attempted": bootstrap_b + 1,
                    "successful": bootstrap_b + 1,
                    "retried": 0,
                    "terminal_failures": 0,
                },
                "bootstrap_attempted": bootstrap_b,
                "bootstrap_successful": bootstrap_b,
                "bootstrap_retried": 0,
                "bootstrap_terminal_failures": 0,
                "bootstrap_terminal_failure_rate": 0.0,
                "bootstrap_failure_rate_max": 0.002,
                "bootstrap_within_failure_ceiling": True,
                "bootstrap_failed_response_indices": [],
                "bootstrap_degenerate_policy": (
                    "any_nonfinite_statistic_sets_null_min_to_zero"
                ),
            },
            **(
                {
                    "adjusted_decisions": [rejected],
                    "formal_rejections": family_ids if rejected else [],
                }
                if stage == "formal" else {
                    "qa_adjusted_diagnostic_decisions": [rejected],
                }
            ),
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


def test_input_manifest_accepts_only_normalized_external_regular_files(tmp_path):
    root = tmp_path / "benchmark"
    inputs = root / "inputs"
    inputs.mkdir(parents=True)
    external = tmp_path / "real-input.bed"
    external.write_bytes(b"real-plink-bed")
    digest = hashlib.sha256(external.read_bytes()).hexdigest()
    manifest = inputs / "manifest.tsv"

    def write(path: str, *, sha256: str = digest) -> None:
        manifest.write_text(
            "path\tsize\tsha256\ttype\n"
            f"{path}\t{external.stat().st_size}\t{sha256}\tbed\n",
            encoding="utf-8",
        )

    write(str(external.resolve()))
    records = _validate_input_manifest(root, manifest)
    assert records == {
        str(external.resolve()): {
            "size": external.stat().st_size,
            "sha256": digest,
            "type": "bed",
        }
    }

    family_ids = [f"group-{index:03d}" for index in range(80)]
    family = {
        "subgenomes": ["A", "B"], "group_ids": family_ids,
        "genes": [[f"a-{index}", f"b-{index}"] for index in range(80)],
    }
    context = {
        "family": family,
        "subgenomes": [
            {"label": label, "X": {}, "gene_snp": [], "samples": ["sample-1"]}
            for label in ("A", "B")
        ],
        "sample_idx": {"values": [0]},
        "phenotype": {"shape": [1]},
    }
    artifact = {
        "schema": "homoeogwas-v201-omnib-context-v1",
        "backbone": "cotton", "context_manifest": context,
        "context_fingerprint": sha256_payload(context),
        "family_manifest": family, "family_hash": sha256_payload(family),
        "group_count": 80, "ordered_family_ids": family_ids,
        "ordered_family_ids_hash": sha256_payload(family_ids),
        "source_inputs": [{
            "path": str(external.resolve()), "sha256": digest, "type": "bed",
        }],
    }
    artifact_path = inputs / "cotton.context.json"
    artifact_path.write_text(canonical_json(artifact) + "\n", encoding="utf-8")
    artifact_digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    manifest.write_text(
        "path\tsize\tsha256\ttype\n"
        f"{external.resolve()}\t{external.stat().st_size}\t{digest}\tbed\n"
        f"inputs/cotton.context.json\t{artifact_path.stat().st_size}\t"
        f"{artifact_digest}\tomnib_context\n",
        encoding="utf-8",
    )
    context_records = _validate_input_manifest(root, manifest)
    _validate_context_artifact(root, "cotton", {
        "artifact_path": "inputs/cotton.context.json",
        "artifact_sha256": artifact_digest,
        "group_count": 80, "ordered_family_ids": family_ids,
        "ordered_family_ids_hash": sha256_payload(family_ids),
        "context_fingerprint": sha256_payload(context),
        "family_hash": sha256_payload(family),
    }, context_records)

    real_context_dir = inputs / "real-contexts"
    real_context_dir.mkdir()
    linked_artifact = real_context_dir / "cotton.context.json"
    linked_artifact.write_bytes(artifact_path.read_bytes())
    linked_context_dir = inputs / "linked-contexts"
    linked_context_dir.symlink_to(real_context_dir, target_is_directory=True)
    linked_key = "inputs/linked-contexts/cotton.context.json"
    linked_records = {
        str(external.resolve()): context_records[str(external.resolve())],
        linked_key: {
            "size": linked_artifact.stat().st_size,
            "sha256": artifact_digest,
            "type": "omnib_context",
        },
    }
    with pytest.raises(BenchmarkAggregateError, match="internal regular file"):
        _validate_context_artifact(root, "cotton", {
            "artifact_path": linked_key,
            "artifact_sha256": artifact_digest,
            "group_count": 80, "ordered_family_ids": family_ids,
            "ordered_family_ids_hash": sha256_payload(family_ids),
            "context_fingerprint": sha256_payload(context),
            "family_hash": sha256_payload(family),
        }, linked_records)

    write(str(external.parent / "missing" / ".." / external.name))
    with pytest.raises(BenchmarkAggregateError, match="normalized absolute"):
        _validate_input_manifest(root, manifest)

    link = tmp_path / "external-link.bed"
    link.symlink_to(external)
    write(str(link))
    with pytest.raises(BenchmarkAggregateError, match="regular file"):
        _validate_input_manifest(root, manifest)

    write(str(external.resolve()), sha256="0" * 64)
    with pytest.raises(BenchmarkAggregateError, match="size/hash mismatch"):
        _validate_input_manifest(root, manifest)

    row = (
        f"{external.resolve()}\t{external.stat().st_size}\t{digest}\tbed\n"
    )
    manifest.write_text(
        "path\tsize\tsha256\ttype\n" + row + row,
        encoding="utf-8",
    )
    with pytest.raises(BenchmarkAggregateError, match="duplicate"):
        _validate_input_manifest(root, manifest)


@pytest.mark.parametrize("separator", ["/./", "//"])
def test_input_manifest_rejects_noncanonical_absolute_spelling(
    tmp_path, separator,
):
    root = tmp_path / "benchmark"
    inputs = root / "inputs"
    inputs.mkdir(parents=True)
    external = tmp_path / "real-input.bed"
    external.write_bytes(b"real-plink-bed")
    digest = hashlib.sha256(external.read_bytes()).hexdigest()
    declared = f"{external.parent}{separator}{external.name}"
    (inputs / "manifest.tsv").write_text(
        "path\tsize\tsha256\ttype\n"
        f"{declared}\t{external.stat().st_size}\t{digest}\tbed\n",
        encoding="utf-8",
    )

    with pytest.raises(BenchmarkAggregateError, match="normalized absolute"):
        _validate_input_manifest(root, inputs / "manifest.tsv")


def test_audit_document_self_hash_is_verified_on_replay(tmp_path, monkeypatch):
    root = _write_null_shards(tmp_path, monkeypatch, rejections=0, total=20)
    audit_benchmark(root)
    path = root / "audit" / "benchmark_audit.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["audit_sha256"] == sha256_payload({
        key: value for key, value in document.items() if key != "audit_sha256"
    })
    first_gate = next(iter(document["gates"].values()))
    first_gate["status"] = "tampered"
    path.write_text(canonical_json(document) + "\n", encoding="utf-8")
    with pytest.raises(BenchmarkAuditError, match="self-hash"):
        audit_benchmark(root)


def test_audit_rejects_symlink_lock_and_stages_before_publish(tmp_path, monkeypatch):
    symlink_root = _write_null_shards(
        tmp_path / "symlink", monkeypatch, rejections=0, total=20
    )
    target = symlink_root / "lock-target"
    target.write_text("target\n", encoding="utf-8")
    (symlink_root / ".benchmark-audit.lock").symlink_to(target)
    with pytest.raises(BenchmarkAuditError, match="lock"):
        audit_benchmark(symlink_root)

    root = _write_null_shards(
        tmp_path / "staging", monkeypatch, rejections=0, total=20
    )
    audit_benchmark(root)
    before = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (root / "tables").glob("*.tsv")
    }
    original = aggregate_module._atomic_tsv
    calls = 0

    def fail_in_staging(path, fields, rows):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("staging failure")
        return original(path, fields, rows)

    monkeypatch.setattr(aggregate_module, "_atomic_tsv", fail_in_staging)
    with pytest.raises(RuntimeError, match="staging failure"):
        audit_benchmark(root)
    after = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (root / "tables").glob("*.tsv")
    }
    assert after == before


def test_publish_replace_failure_rolls_back_tables_and_audit_bytes(tmp_path, monkeypatch):
    root = _write_null_shards(tmp_path, monkeypatch, rejections=0, total=20)
    audit_benchmark(root)
    live = [*(sorted((root / "tables").glob("*.tsv"))),
            root / "audit" / "benchmark_audit.json"]
    before = {path: path.read_bytes() for path in live}
    original_replace = os.replace
    live_replaces = 0

    def fail_third_live_replace(source, destination):
        nonlocal live_replaces
        destination = Path(destination)
        if destination.parent in {root / "tables", root / "audit"}:
            live_replaces += 1
            if live_replaces == 3:
                raise OSError("injected publication failure")
        return original_replace(source, destination)

    monkeypatch.setattr(os, "replace", fail_third_live_replace)
    with pytest.raises(OSError, match="injected publication failure"):
        audit_benchmark(root)
    assert {path: path.read_bytes() for path in live} == before


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
    for name, fields in table_schemas(result.evidence.stage).items():
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
    with (root / "tables" / "benchmark_acceptance.tsv").open(
        encoding="utf-8", newline="",
    ) as handle:
        assert "passed" not in next(csv.reader(handle, delimiter="\t"))
    with (root / "tables" / "omnib_null_replicates.tsv").open(
        encoding="utf-8", newline="",
    ) as handle:
        pilot_header = next(csv.reader(handle, delimiter="\t"))
    assert "threshold" not in pilot_header
    assert "rejected" not in pilot_header


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


def test_input_manifest_rehashes_files_and_rejects_directory_drift(
    tmp_path, monkeypatch,
):
    root = _write_null_shards(tmp_path, monkeypatch, rejections=0, total=20)
    (root / "inputs" / "fixture.txt").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(BenchmarkAuditError, match="input size/hash"):
        audit_benchmark(root)

    root = _write_null_shards(
        tmp_path / "extra", monkeypatch, rejections=0, total=20,
    )
    (root / "inputs" / "undeclared.txt").write_text("extra\n", encoding="utf-8")
    with pytest.raises(BenchmarkAuditError, match="input directory"):
        audit_benchmark(root)


def test_configs_are_manifest_complete_and_parse_as_yaml(tmp_path):
    configs = tmp_path / "configs"
    configs.mkdir()
    config = configs / "scientific.yaml"
    config.write_text(
        "interact:\n  mode: group\n  statistic: omniB\n  hypothesis_unit: group\n"
        "  subset_order: 2\n  family_scope: primary_only\n"
        "  primary_transform: INT\n  primary_multiplicity: bootstrap_minp\n"
        "  subgenomes: [A, B]\n  groups: inputs/groups.tsv\n"
        "  genotype: {A: inputs/a, B: inputs/b}\n"
        "  snp_to_gene: {A: inputs/a.npz, B: inputs/b.npz}\n"
        "  phenotype: inputs/p.tsv\n  sample_col: sample\n  trait: trait\n"
        "  grm: {method: grm_from_X, maf_min: 0.01, scope: all_subgenomes}\n"
        "  calibration: {method: bootstrap, B: 199, seed: 2026}\n",
        encoding="utf-8",
    )
    digest = hashlib.sha256(config.read_bytes()).hexdigest()
    manifest = configs / "manifest.tsv"
    manifest.write_text(
        f"path\tsha256\nconfigs/scientific.yaml\t{digest}\n", encoding="utf-8"
    )
    with pytest.raises(BenchmarkAggregateError, match="canonical group-omniB"):
        _validate_config_manifest(tmp_path, manifest)
    config.write_text("- not\n- a\n- mapping\n", encoding="utf-8")
    digest = hashlib.sha256(config.read_bytes()).hexdigest()
    manifest.write_text(
        f"path\tsha256\nconfigs/scientific.yaml\t{digest}\n", encoding="utf-8"
    )
    with pytest.raises(BenchmarkAggregateError, match="YAML"):
        _validate_config_manifest(tmp_path, manifest)


def test_context_artifact_rejects_self_hashed_empty_context(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    source = inputs / "groups.tsv"
    source.write_text("group_id\n", encoding="utf-8")
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    family_ids = [f"group-{index:03d}" for index in range(80)]
    family = {
        "subgenomes": ["A", "B"], "group_ids": family_ids,
        "genes": [[f"a-{index}", f"b-{index}"] for index in range(80)],
    }
    context = {"family": family, "subgenomes": [], "sample_idx": {}}
    artifact = {
        "schema": "homoeogwas-v201-omnib-context-v1",
        "backbone": "cotton", "context_manifest": context,
        "context_fingerprint": sha256_payload(context),
        "family_manifest": family, "family_hash": sha256_payload(family),
        "group_count": 80, "ordered_family_ids": family_ids,
        "ordered_family_ids_hash": sha256_payload(family_ids),
        "source_inputs": [{
            "path": "inputs/groups.tsv", "sha256": source_sha, "type": "groups",
        }],
    }
    artifact_path = inputs / "cotton.context.json"
    artifact_path.write_text(canonical_json(artifact) + "\n", encoding="utf-8")
    artifact_sha = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    records = {
        "inputs/groups.tsv": {"size": source.stat().st_size, "sha256": source_sha,
                              "type": "groups"},
        "inputs/cotton.context.json": {"size": artifact_path.stat().st_size,
                                       "sha256": artifact_sha,
                                       "type": "omnib_context"},
    }
    record = {
        "artifact_path": "inputs/cotton.context.json",
        "artifact_sha256": artifact_sha,
        "group_count": 80, "ordered_family_ids": family_ids,
        "ordered_family_ids_hash": sha256_payload(family_ids),
        "context_fingerprint": sha256_payload(context),
        "family_hash": sha256_payload(family),
    }
    with pytest.raises(BenchmarkAggregateError, match="context artifact is invalid"):
        _validate_context_artifact(tmp_path, "cotton", record, records)

def test_design_lock_distinguishes_target_release_from_harness(
    tmp_path, monkeypatch,
):
    root = _write_null_shards(tmp_path, monkeypatch, rejections=0, total=20)
    lock_path = root / "design_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["target_release"]["git_commit"] = lock["harness"]["git_commit"]
    payload = {key: value for key, value in lock.items() if key != "design_hash"}
    lock["design_hash"] = sha256_payload(payload)
    lock_path.write_text(canonical_json(lock) + "\n", encoding="utf-8")
    with pytest.raises(BenchmarkAuditError, match="release target/harness"):
        audit_benchmark(root)


def test_loco_lock_requires_complete_per_replicate_artifact_mapping(tmp_path):
    seed_design_hash = "d" * 64
    scenarios = [Scenario(
        "A.loco.cotton.pve_0", "fit", "pilot", 2, 199,
        {"panel": "cotton", "experiment": "loco", "scan_pve": 0.0},
    )]
    records = {}
    artifacts = {scenarios[0].scenario_id: {}}
    for replicate in range(2):
        truth = tmp_path / f"truth-{replicate}.json"
        truth.write_text(f"{{\"replicate\":{replicate}}}\n")
        digest = hashlib.sha256(truth.read_bytes()).hexdigest()
        key = f"inputs/truth-{replicate}.json"
        records[key] = {
            "size": truth.stat().st_size, "sha256": digest, "type": "loco_truth",
        }
        artifacts[scenarios[0].scenario_id][str(replicate)] = {
            "path": key, "sha256": digest, "truth_hash": f"{replicate + 1}" * 64,
            "source": "deterministic_pre_fit_simulation",
            "seed": derive_seed(
                seed_design_hash, "fit", scenarios[0].scenario_id, replicate,
                "pilot:loco_truth",
            ),
            "generated_config_sha256": f"{replicate + 10:x}" * 64,
            "phenotype_sha256": f"{replicate + 2}" * 64,
        }

    _validate_loco_truth_artifacts(
        scenarios, artifacts, records, seed_design_hash=seed_design_hash,
    )
    missing = json.loads(json.dumps(artifacts))
    del missing[scenarios[0].scenario_id]["1"]
    with pytest.raises(BenchmarkAggregateError, match="replicate coverage"):
        _validate_loco_truth_artifacts(
            scenarios, missing, records, seed_design_hash=seed_design_hash,
        )
    legacy = {scenarios[0].scenario_id: artifacts[scenarios[0].scenario_id]["0"]}
    with pytest.raises(BenchmarkAggregateError, match="legacy single-artifact"):
        _validate_loco_truth_artifacts(
            scenarios, legacy, records, seed_design_hash=seed_design_hash,
        )
    wrong_seed = json.loads(json.dumps(artifacts))
    wrong_seed[scenarios[0].scenario_id]["1"]["seed"] += 1
    with pytest.raises(BenchmarkAggregateError, match="derived seed"):
        _validate_loco_truth_artifacts(
            scenarios, wrong_seed, records, seed_design_hash=seed_design_hash,
        )


def test_comparator_preflights_are_bound_separately_for_both_fit_panels():
    records = {
        "inputs/comparator_preflight.cotton.tsv": {
            "size": 10, "sha256": "a" * 64, "type": "comparator_preflight",
        },
        "inputs/comparator_preflight.wheat.tsv": {
            "size": 11, "sha256": "b" * 64, "type": "comparator_preflight",
        },
    }
    mapping = {
        "cotton": {"path": "inputs/comparator_preflight.cotton.tsv", "sha256": "a" * 64},
        "wheat": {"path": "inputs/comparator_preflight.wheat.tsv", "sha256": "b" * 64},
    }
    assert _validate_comparator_preflights({"cotton", "wheat"}, mapping, records) == mapping
    with pytest.raises(BenchmarkAggregateError, match="panel coverage"):
        _validate_comparator_preflights({"cotton", "wheat"}, {"cotton": mapping["cotton"]}, records)
    with pytest.raises(BenchmarkAggregateError, match="legacy global"):
        _validate_comparator_preflights({"cotton", "wheat"}, "a" * 64, records)


def test_seed_design_excludes_dynamic_paths_but_detects_scientific_config_tamper(tmp_path):
    config = tmp_path / "configs" / "fit.yaml"
    config.parent.mkdir()
    base = {
        "fit_version": 1,
        "panel": {"name": "cotton", "subgenomes": ["A", "D"]},
        "phenotype": {"path": "/generated/p0.tsv", "sample_col": "sample", "trait": "trait"},
        "scan": {"maf_min": 0.05, "loco": {"enabled": True}},
        "outputs": {"out_dir": "/generated/run0", "prefix": "trait"},
    }
    config.write_text(yaml.safe_dump(base, sort_keys=False))
    kwargs = {
        "input_records": {
            "/external/a.bed": {"sha256": "a" * 64, "type": "bed"},
        },
        "config_hashes": {"configs/fit.yaml": hashlib.sha256(config.read_bytes()).hexdigest()},
        "contexts": {}, "target_release": {"version": "2.0.1"},
        "harness": {"git_commit": "b" * 40},
    }
    first = seed_design_payload(tmp_path, **kwargs)
    base["phenotype"]["path"] = "/generated/p1.tsv"
    base["outputs"]["out_dir"] = "/generated/run1"
    config.write_text(yaml.safe_dump(base, sort_keys=False))
    assert seed_design_payload(tmp_path, **kwargs) == first
    base["scan"]["maf_min"] = 0.01
    config.write_text(yaml.safe_dump(base, sort_keys=False))
    assert seed_design_payload(tmp_path, **kwargs) != first


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


def test_real_conditional_snpxsnp_group_rows_and_raw_family_are_distinct():
    context = build_synthetic_omnib_context(n=72, groups=2, copies=3, seed=1)
    bank = run_conditional_bank(
        context, bank="calibration", count=2,
        design_hash="a" * 64, n_jobs=1,
    ).to_payload()
    failed, _ids = _validate_bank_envelope(bank, 2, "calibration")
    checked = _conditional_score_matrices(bank, 2, failed)
    assert len(checked["snpxsnp"]) == len(bank["family_ids"])
    assert bank["tested_family_sizes"]["snpxsnp"] > len(checked["snpxsnp"])
    evidence = bank["snpxsnp_evidence"]
    assert evidence["schema"] == "snpxsnp_raw_stream_v2"
    assert len(evidence["input_block_bindings"]) > 0
    assert len(evidence["input_family_sha256"]) == 64
    assert (
        evidence["offered_pair_count"]
        - evidence["design_nonestimable_pair_count"]
        == evidence["tested_pair_count"]
    )
    assert "calibration_p" not in evidence


def test_conditional_snpxsnp_per_group_pair_counts_are_audited():
    context = build_synthetic_omnib_context(n=72, groups=2, copies=3, seed=2)
    bank = run_conditional_bank(
        context, bank="calibration", count=2,
        design_hash="b" * 64, n_jobs=1,
    ).to_payload()
    bank["snpxsnp_evidence"]["tested_pair_count_by_group"][0] -= 1
    failed, _ids = _validate_bank_envelope(bank, 2, "calibration")
    with pytest.raises(BenchmarkAuditError, match="SNPxSNP streaming evidence"):
        _conditional_score_matrices(bank, 2, failed)


def test_conditional_snpxsnp_input_block_binding_is_audited():
    context = build_synthetic_omnib_context(n=72, groups=2, copies=3, seed=3)
    bank = run_conditional_bank(
        context, bank="calibration", count=2,
        design_hash="d" * 64, n_jobs=1,
    ).to_payload()
    bank["snpxsnp_evidence"]["input_block_bindings"][0][
        "dosage_sha256"
    ] = "0" * 64
    failed, _ids = _validate_bank_envelope(bank, 2, "calibration")
    with pytest.raises(BenchmarkAuditError, match="SNPxSNP streaming evidence"):
        _conditional_score_matrices(bank, 2, failed)


def test_conditional_snpxsnp_malformed_evidence_fails_as_audit_error():
    context = build_synthetic_omnib_context(n=48, groups=1, copies=2, seed=5)
    bank = run_conditional_bank(
        context, bank="calibration", count=1,
        design_hash="c" * 64, n_jobs=1,
    ).to_payload()
    bank["snpxsnp_evidence"] = []
    failed, _ids = _validate_bank_envelope(bank, 1, "calibration")
    with pytest.raises(BenchmarkAuditError, match="SNPxSNP streaming evidence"):
        _conditional_score_matrices(bank, 1, failed)


def test_conditional_family_manifest_exports_streamed_pair_counts():
    context = build_synthetic_omnib_context(n=48, groups=2, copies=2, seed=19)
    scenario = Scenario(
        "B.conditional.synthetic.gaussian.calibration",
        "omnib",
        "pilot",
        2,
        0,
        {
            "experiment": "conditional",
            "bank": "calibration",
            "null_model": "gaussian",
        },
    )
    payload = run_omnib_replicate(
        scenario, context, replicate=0, design_hash="d" * 64, n_jobs=1,
    )
    rows = aggregate_module._omnib_rows(payload, scenario)[
        "omnib_family_manifest.tsv"
    ]
    row = next(item for item in rows if item["method"] == "snpxsnp")
    evidence = payload["bank"]["snpxsnp_evidence"]
    assert row["offered_pair_count"] == evidence["offered_pair_count"]
    assert row["design_nonestimable_pair_count"] == evidence[
        "design_nonestimable_pair_count"
    ]
    assert row["tested_pair_count"] == evidence["tested_pair_count"]
    assert row["member_family_sha256"] == evidence["member_family_sha256"]


def test_global_vc_heldout_is_bound_to_registered_calibration_bank():
    context = build_synthetic_omnib_context(n=32, groups=2, copies=2, seed=9)
    calibration_id = "B.global_vc.synthetic.gaussian.calibration"
    heldout_id = "B.global_vc.synthetic.gaussian.heldout"
    calibration = {
        "track": "omnib", "replicate": 0,
        **run_global_vc_bank(
            context, bank="calibration", count=1, design_hash="1" * 64,
            qa_only=True, scenario_id=calibration_id,
        ),
    }
    heldout = {
        "track": "omnib", "replicate": 0,
        **run_global_vc_bank(
            context, bank="heldout", count=1, design_hash="1" * 64,
            qa_only=True, scenario_id=heldout_id,
        ),
    }
    evidence = SimpleNamespace(
        shards=((Path("cal.json"), calibration), (Path("held.json"), heldout)),
        registry=(),
    )
    _audit_families_and_parallel(evidence)
    heldout["calibration_p_values"] = [
        (heldout["calibration_p_values"][0] + 0.25) % 1.0
    ]
    heldout["calibration_p_hash"] = sha256_payload(heldout["calibration_p_values"])
    with pytest.raises(BenchmarkAuditError, match="global VC (calibration|LRT)"):
        _audit_families_and_parallel(evidence)


def test_global_vc_positive_power_is_detection_only_and_recomputable():
    context = build_synthetic_omnib_context(n=32, groups=2, copies=2, seed=10)
    calibration_id = "B.global_vc.synthetic.gaussian.calibration"
    power_id = "B.global_vc.synthetic.gaussian.power"
    calibration = run_global_vc_bank(
        context, bank="calibration", count=1, design_hash="2" * 64,
        qa_only=True, scenario_id=calibration_id,
    )
    power = run_global_vc_bank(
        context, bank="power", count=2, calibration_count=1,
        design_hash="2" * 64, qa_only=True, scenario_id=power_id,
    )
    evidence = SimpleNamespace(
        shards=((Path("cal.json"), calibration), (Path("power.json"), power)),
        registry=(), design_lock={},
    )
    _audit_families_and_parallel(evidence)
    assert power["detection_only"] is True
    assert power["detection_power"] == np.mean(power["qa_detection_flags"])
    assert not ({"causal_group_ids", "recall_by_method"} & set(power))


def test_negative_power_gate_uses_specificity_without_causal_detection():
    context = build_synthetic_omnib_context(n=48, groups=2, copies=2, seed=111)
    scenario = Scenario(
        "B.power.cotton.mispaired.g1.pve_0p05", "omnib", "pilot", 1, 0,
        {"experiment": "power", "backbone": "cotton", "control_type": "negative",
         "architecture": "mispaired", "causal_groups": 1, "interaction_pve": 0.05},
    )
    calibration = run_conditional_bank(
        context, bank="calibration", count=19, design_hash="3" * 64,
        qa_only=True, scenario_id="B.conditional.cotton.gaussian.calibration",
        n_jobs=1,
    )
    payload = run_power_replicate(
        context, calibration_bank=calibration,
        calibration_scenario_id="B.conditional.cotton.gaussian.calibration",
        replicate=0, architecture="mispaired", interaction_pve=0.05,
        causal_groups=1, calibration_count=19, response_count=1,
        design_hash="3" * 64, qa_only=True, n_jobs=1,
        scenario_id=scenario.scenario_id,
    )
    rows = aggregate_module._omnib_rows(payload, scenario)
    null_gate = AuditGate(
        "null", "omnib", "null", "core_fwer", 0, 1, 0.0, 0.0, 1.0,
        0, 0.0, None, True, "QA_PASS", "fixture", "fixture",
    )
    evidence = SimpleNamespace(stage="pilot", registry=(scenario,))
    gates = _omnib_power_gates(
        evidence, rows,
        {"B.core_fwer.B.conditional.cotton.gaussian.heldout." + method: null_gate
         for method in METHOD_NAMES},
    )
    for method in METHOD_NAMES:
        gate = gates[f"B.specificity.{scenario.scenario_id}.{method}"]
        assert gate.gate_kind == "negative_control_specificity"
        assert gate.qa_passed is True


def test_encoding_exact_failure_creates_a_failing_pilot_gate():
    scenario = Scenario(
        "B.encoding.cotton", "omnib", "pilot", 1, 199,
        {"experiment": "encoding", "backbone": "cotton", "edges_per_group": 1},
    )
    evidence = SimpleNamespace(
        stage="pilot", registry=(scenario,),
        shards=((Path("encoding.json"), {
            "track": "omnib", "scenario_id": scenario.scenario_id,
            "pair_edges_per_group": 1, "all_required_exact": False,
        }),),
    )
    gates = _pilot_and_engineering_gates(evidence)
    gate = gates[f"B.encoding_exact.{scenario.scenario_id}"]
    assert gate.qa_passed is False
    assert gate.status == "QA_FAIL"


def test_robustness_table_is_architecture_method_stratified_from_raw_responses():
    scenario = Scenario(
        "B.encoding.cotton", "omnib", "pilot", 1, 199,
        {"experiment": "encoding", "backbone": "cotton"},
    )
    methods = ("omnib", "minor_burden", "pc1", "kernel_hadamard")
    heldout_scores = {
        method: [[0.2, 0.01], [0.8, 0.9]] for method in methods
    }
    power_scores = {
        "omnib": [[0.01, 0.2], [0.8, 0.9]],
        "minor_burden": [[0.8, 0.9], [0.01, 0.2]],
        "pc1": [[0.2, 0.01], [0.9, 0.8]],
        "kernel_hadamard": [[0.9, 0.8], [0.2, 0.01]],
    }
    correlations = {
        "omnib": [1.0, 0.5], "minor_burden": [0.5, 0.0],
        "pc1": [0.0, -0.5], "kernel_hadamard": [-0.5, -1.0],
    }
    overlaps = {
        "omnib": [1.0, 0.5], "minor_burden": [0.5, 0.5],
        "pc1": [0.5, 0.0], "kernel_hadamard": [0.0, 0.0],
    }
    record = {
        "status": "completed", "required": None,
        "calibration": {
            "qa_cutoffs_by_method": {method: 0.05 for method in methods}
        },
        "heldout": {
            "p_by_method": heldout_scores,
            "qa_rejections_by_method": {
                method: [False, True] for method in methods
            },
        },
        "qa_power_by_architecture": {
            "minor_burden_aligned": {
                "p_by_method": power_scores,
                "qa_detection_by_method": {
                    method: [True, False] for method in methods
                },
                "rank_correlation_by_method": correlations,
                "top_k_jaccard_by_method": overlaps,
                "qa_absolute_power_regret_by_method": {
                    "omnib": 0.25, "minor_burden": None,
                    "pc1": None, "kernel_hadamard": None,
                },
            }
        },
    }
    payload = {
        "stage": "pilot", "track": "omnib", "experiment": "encoding",
        "scenario_id": scenario.scenario_id,
        "replicate": 0, "design_hash": "4" * 64,
        "request_hash": "5" * 64, "context_fingerprint": "6" * 64,
        "inference_status": "noninferential_do_not_threshold",
        "failure": {"failed": False}, "exact_checks": {},
        "family_ids": ["group-0", "group-1"],
        "family_hash": "7" * 64,
        "family_order_hash": sha256_payload(["group-0", "group-1"]),
        "robustness_checks": {"missingness_2pct": record},
    }
    rows = aggregate_module._omnib_rows(
        payload, scenario
    )["omnib_encoding_robustness.tsv"]
    assert {(row["architecture"], row["method"]) for row in rows} == {
        ("minor_burden_aligned", method) for method in methods
    }
    for row in rows:
        assert row["qa_fwer_successes"] == 1
        assert row["qa_fwer_total"] == 2
        assert row["qa_power_successes"] == 1
        assert row["qa_power_total"] == 2
        assert row["fwer_successes"] is None
        assert row["power_successes"] is None
    by_method = {row["method"]: row for row in rows}
    assert {method: row["rank_correlation"] for method, row in by_method.items()} == {
        method: np.mean(values) for method, values in correlations.items()
    }
    assert {method: row["top_k_jaccard"] for method, row in by_method.items()} == {
        method: np.mean(values) for method, values in overlaps.items()
    }
    assert by_method["omnib"]["absolute_power_regret"] == 0.25
    assert all(
        by_method[method]["absolute_power_regret"] is None
        for method in ("minor_burden", "pc1", "kernel_hadamard")
    )


def test_robustness_audit_recomputes_each_method_and_rejects_formal_pilot_keys():
    methods = ("omnib", "minor_burden", "pc1", "kernel_hadamard")
    family_ids = [f"group-{index:02d}" for index in range(12)]
    calibration_count, response_count = 199, 20
    calibration_ids = [f"cal-{index}" for index in range(calibration_count)]
    heldout_ids = [f"held-{index}" for index in range(response_count)]

    def no_failures(count: int, role: str) -> dict:
        return {
            "attempted": count,
            "failed_response_indices": [],
            "failed_response_indices_by_method": {
                method: [] for method in methods
            },
            "terminal_failures": 0,
            "terminal_failures_by_method": {method: 0 for method in methods},
            "terminal_failure_rate": 0.0,
            "terminal_failure_rate_by_method": {
                method: 0.0 for method in methods
            },
            "failure_rate_max": 0.01,
            "within_failure_ceiling": True,
            "within_failure_ceiling_by_method": {
                method: True for method in methods
            },
            "worst_case_mapping": (
                "failure_counts_as_rejection"
                if role in {"calibration", "heldout"}
                else "failure_counts_as_non_detection"
            ),
        }

    def bank_scores(count: int, offset: float = 0.0) -> dict[str, list[list[float]]]:
        return {
            method: [
                [0.40 + offset + row / 1_000 + column / 100_000
                 for column in range(count)]
                for row in range(len(family_ids))
            ]
            for method in methods
        }

    calibration_scores = bank_scores(calibration_count)
    heldout_scores = bank_scores(response_count, 0.05)
    thresholds = {
        method: sorted(
            min(row[column] for row in values)
            for column in range(calibration_count)
        )[9]
        for method, values in calibration_scores.items()
    }
    heldout_decisions = {
        method: [False] * response_count for method in methods
    }
    calibration = {
        "response_ids": calibration_ids,
        "seeds": list(range(calibration_count)),
        "response_hash": "1" * 64,
        "p_by_method": calibration_scores,
        "p_hashes": {
            method: sha256_payload(values)
            for method, values in calibration_scores.items()
        },
        "response_failures": no_failures(calibration_count, "calibration"),
        "qa_cutoffs_by_method": thresholds,
    }
    heldout = {
        "response_ids": heldout_ids,
        "seeds": list(range(10_000, 10_000 + response_count)),
        "response_hash": "2" * 64,
        "p_by_method": heldout_scores,
        "p_hashes": {
            method: sha256_payload(values)
            for method, values in heldout_scores.items()
        },
        "response_failures": no_failures(response_count, "heldout"),
        "qa_rejections_by_method": heldout_decisions,
    }
    architectures = [
        "minor_burden_aligned", "pc1_distributed", "kernel_multidimensional",
        "single_snp_pair", "mixed_sign",
    ]
    strata = {}
    permutations = {
        "omnib": np.arange(12),
        "minor_burden": np.arange(12)[::-1],
        "pc1": np.roll(np.arange(12), 3),
        "kernel_hadamard": np.roll(np.arange(12), 7),
    }
    for architecture_index, architecture in enumerate(architectures):
        baseline = bank_scores(response_count, 0.10)
        candidate = {
            method: np.asarray(values, dtype=float)[permutations[method]].tolist()
            for method, values in baseline.items()
        }
        correlations = {}
        overlaps = {}
        for method in methods:
            correlations[method], overlaps[method] = _ranking_metrics(
                np.asarray(baseline[method]), np.asarray(candidate[method]), family_ids,
            )
        detection = {
            method: [False] * response_count for method in methods
        }
        powers = {method: 0.0 for method in methods}
        strata[architecture] = {
            "response_ids": [
                f"power-{architecture_index}-{index}"
                for index in range(response_count)
            ],
            "seeds": [
                20_000 + architecture_index * 100 + index
                for index in range(response_count)
            ],
            "response_hash": f"{architecture_index + 3:x}" * 64,
            "causal_group_ids": [family_ids[0]],
            "response_failures": no_failures(response_count, "power"),
            "baseline_response_failures": no_failures(response_count, "power"),
            "p_by_method": candidate,
            "p_hashes": {
                method: sha256_payload(values) for method, values in candidate.items()
            },
            "baseline_p_by_method": baseline,
            "baseline_p_hashes": {
                method: sha256_payload(values) for method, values in baseline.items()
            },
            "rank_correlation_by_method": correlations,
            "top_k_jaccard_by_method": overlaps,
            "qa_detection_by_method": detection,
            "qa_power_by_method": powers,
            "qa_absolute_power_regret_by_method": {
                "omnib": 0.0, "minor_burden": None,
                "pc1": None, "kernel_hadamard": None,
            },
        }
    record = {
        "status": "completed", "error_type": None, "message": None,
        "rank_correlation": float(np.mean([
            value for arm in strata.values()
            for value in arm["rank_correlation_by_method"]["omnib"]
            if value is not None
        ])),
        "top_k": 10,
        "top_k_jaccard": float(np.mean([
            value for arm in strata.values()
            for value in arm["top_k_jaccard_by_method"]["omnib"]
        ])),
        "non_estimable_rate": 0.0,
        "realized_marker_design": None,
        "note": "fixture",
        "design_ruling": {
            "interaction_pve": 0.05, "causal_groups": 1,
            "architectures": architectures, "calibration_count": 199,
            "heldout_count": 20, "power_count_per_architecture": 20,
            "stratify_by_architecture": True,
            "component_regret_reference": [
                "minor_burden", "pc1", "kernel_hadamard",
            ],
        },
        "calibration": calibration, "heldout": heldout,
        "qa_power_by_architecture": strata,
        "qa_fwer": 0.0, "qa_power": None,
        "qa_absolute_power_regret": None,
    }
    payload = {
        "stage": "pilot", "bootstrap_B": 199,
        "pair_edges_per_group": 1, "family_ids": family_ids,
    }
    _audit_robustness_record("missingness_2pct", record, payload)

    tampered = copy.deepcopy(record)
    tampered["qa_power_by_architecture"][architectures[0]][
        "rank_correlation_by_method"
    ]["pc1"][0] = 0.123
    with pytest.raises(BenchmarkAuditError, match="ranking metrics"):
        _audit_robustness_record("missingness_2pct", tampered, payload)

    cross_stage = copy.deepcopy(record)
    cross_stage["calibration"]["thresholds"] = thresholds
    with pytest.raises(BenchmarkAuditError, match="raw stage schema"):
        _audit_robustness_record("missingness_2pct", cross_stage, payload)


def test_family_size_audit_recomputes_response_level_fwer():
    context = build_synthetic_omnib_context(n=72, groups=80, copies=2, seed=3)
    scenario = Scenario(
        "B.family_size.synthetic.g80", "omnib", "pilot", 20, 199,
        {"experiment": "family_size", "family_size": 80,
         "null_model": "gaussian", "snpxsnp_status": "applicable"},
    )
    payload = run_omnib_replicate(
        scenario, context, replicate=0, design_hash="b" * 64, n_jobs=1,
    )
    evidence = SimpleNamespace(shards=((Path("family.json"), payload),), registry=(scenario,))
    _audit_families_and_parallel(evidence)
    rows = aggregate_module._omnib_rows(payload, scenario)["omnib_null_replicates.tsv"]
    snpxsnp = [row for row in rows if row["method"] == "snpxsnp"]
    assert len(snpxsnp) == 20
    assert snpxsnp[0]["family_size"] == 80
    assert snpxsnp[0]["score_family_size"] == 80
    assert snpxsnp[0]["tested_family_size"] > 80
    assert snpxsnp[0]["tested_family_hash"] == payload["tested_family_hashes"]["snpxsnp"]
    detached = copy.deepcopy(payload)
    detached["snpxsnp_evidence"]["tested_pair_count_by_group"][0] -= 1
    detached_evidence = SimpleNamespace(
        shards=((Path("family-detached.json"), detached),), registry=(scenario,),
    )
    with pytest.raises(BenchmarkAuditError, match="SNPxSNP streaming evidence"):
        _audit_families_and_parallel(detached_evidence)
    detached_calibration = copy.deepcopy(payload)
    detached_calibration["snpxsnp_calibration_evidence"][
        "input_family_sha256"
    ] = "0" * 64
    detached_calibration_evidence = SimpleNamespace(
        shards=((Path("family-calibration-detached.json"), detached_calibration),),
        registry=(scenario,),
    )
    with pytest.raises(BenchmarkAuditError, match="SNPxSNP streaming evidence"):
        _audit_families_and_parallel(detached_calibration_evidence)
    for evidence_field in (
        "snpxsnp_calibration_evidence",
        "snpxsnp_evidence",
    ):
        detached_raw_failure = copy.deepcopy(payload)
        raw_evidence = detached_raw_failure[evidence_field]
        raw_evidence["failed_response_indices"] = [0]
        for row in raw_evidence["argmin_member_index"]:
            row[0] = -1
        raw_evidence["nonfinite_pair_score_count"] = 1
        raw_evidence["nonfinite_pair_score_count_by_group"][0] = 1
        detached_raw_failure_evidence = SimpleNamespace(
            shards=((Path("family-raw-failure-detached.json"), detached_raw_failure),),
            registry=(scenario,),
        )
        with pytest.raises(BenchmarkAuditError, match="SNPxSNP failure mask differs"):
            _audit_families_and_parallel(detached_raw_failure_evidence)
    failure_detached = copy.deepcopy(payload)
    failure_detached["target_response_failures"][
        "failed_response_indices_by_method"
    ]["omnib"] = [0]
    failure_evidence = SimpleNamespace(
        shards=((Path("family-failure-detached.json"), failure_detached),),
        registry=(scenario,),
    )
    with pytest.raises(BenchmarkAuditError, match="response failure"):
        _audit_families_and_parallel(failure_evidence)
    payload["fwer"]["omnib"] = 0.123
    with pytest.raises(BenchmarkAuditError, match="family-size"):
        _audit_families_and_parallel(evidence)


def _real_scaling_payload(*, native_valid=True):
    anchor = ScalingAnchor("small_qa", 192, 192, 3, 3, 199, (1, 4), 3)
    runs = []
    for jobs in anchor.jobs:
        for repeat in range(3):
            runs.append(ScalingAnchorRun(
                anchor_id="small_qa", jobs=jobs, repeat=repeat,
                effective_jobs=jobs, backend=("serial" if jobs == 1 else "fork_shared_memory"),
                worker_pids=tuple(range(100 * jobs, 100 * jobs + jobs)),
                wall_seconds=10.0 / jobs, cpu_seconds=9.0,
                aggregate_cpu_percent=90.0 * jobs,
                peak_aggregate_pss_bytes=1_000, peak_aggregate_rss_bytes=1_200,
                max_process_threads=1,
                max_process_threads_by_pid=tuple(
                    (pid, 1) for pid in range(100 * jobs, 100 * jobs + jobs)
                ),
                numeric_threadpool_max_threads=1,
                numeric_threadpool_info_json='[{"num_threads":1}]',
                runtime_oversubscription_guard_passed=native_valid,
                result_sha256="1" * 64, family_sha256="2" * 64,
                ranking_sha256="3" * 64,
                numeric_thread_env=tuple(
                    (name, "1") for name in NUMERIC_THREAD_ENV
                ),
                numeric_thread_limit_ok=True,
                command=("fixture",),
            ))
    summary = summarize_anchor(anchor, runs)
    scenario = Scenario(
        "C.small_qa", "scaling", "formal", 3, 199,
        {"qa_only": False, "anchor": anchor.to_dict()},
    )
    return {"failure": {"failed": False}, "summary": summary}, scenario


def test_real_scaling_producer_fields_are_used_for_contract_gate():
    payload, scenario = _real_scaling_payload(native_valid=True)
    recomputed = _audit_scaling_scenario(payload, scenario)
    assert recomputed["native_thread_contract_valid"] is True
    assert recomputed["parallel_execution_contract_valid"] is True

    bad_payload, bad_scenario = _real_scaling_payload(native_valid=False)
    recomputed_bad = _audit_scaling_scenario(bad_payload, bad_scenario)
    assert recomputed_bad["native_thread_contract_valid"] is False


def test_track_a_true_pve_cannot_diverge_from_frozen_truth():
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
    payload["true_pve"]["A"] += 0.1
    payload["pve_bias"]["A"] = payload["estimated_pve"]["A"] - payload["true_pve"]["A"]
    with pytest.raises(BenchmarkAuditError, match="truth"):
        _audit_fit_scenario(payload, scenario)


def test_track_a_boundary_labels_are_recomputed_from_estimated_pve():
    kernels = {}
    rng = np.random.default_rng(12)
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
        scenario, kernels, replicate=0, design_hash="e" * 64,
        sample_ids=np.asarray([f"sample-{index}" for index in range(36)]),
    )
    payload["boundary_components"] = ["fabricated"]
    with pytest.raises(BenchmarkAuditError, match="boundary"):
        _audit_fit_scenario(payload, scenario)


def test_power_registry_metadata_and_single_response_are_fail_closed():
    context = build_synthetic_omnib_context(n=72, groups=2, copies=3, seed=7)
    design_hash = "c" * 64
    calibration_id = "B.conditional.synthetic.gaussian.calibration"
    calibration = run_conditional_bank(
        context, bank="calibration", count=3, design_hash=design_hash,
        qa_only=True, null_model="gaussian", scenario_id=calibration_id,
        n_jobs=1,
    )
    payload = run_power_replicate(
        context, calibration_bank=calibration,
        calibration_scenario_id=calibration_id, replicate=0,
        architecture="minor_burden_aligned", interaction_pve=0.05,
        causal_groups=1, calibration_count=3, response_count=1,
        design_hash=design_hash, qa_only=True, n_jobs=1,
    )
    scenario = Scenario(
        "B.power.synthetic", "omnib", "pilot", 1, 0,
        {"experiment": "power", "architecture": "minor_burden_aligned",
         "interaction_pve": 0.05, "causal_groups": 1,
         "calibration_count": 3, "response_count": 1,
         "null_model": "gaussian", "calibration_scenario_id": calibration_id},
    )
    _audit_power_evidence(payload, scenario)
    detached = copy.deepcopy(payload)
    detached["target_bank"]["snpxsnp_evidence"][
        "member_family_sha256"
    ] = "0" * 64
    with pytest.raises(BenchmarkAuditError, match="SNPxSNP"):
        _audit_power_evidence(detached, scenario)
    detached = copy.deepcopy(payload)
    detached["target_bank"]["prepared_design_sha256"] = "0" * 64
    with pytest.raises(BenchmarkAuditError, match="prepared identities"):
        _audit_power_evidence(detached, scenario)
    detached = copy.deepcopy(payload)
    detached["target_minima_by_method"]["omnib"] = [0.0]
    detached["target_minima_hashes"]["omnib"] = sha256_payload([0.0])
    detached["qa_rejections_by_method"]["omnib"] = [False]
    with pytest.raises(BenchmarkAuditError, match="score matrix"):
        _audit_power_evidence(detached, scenario)
    payload["architecture"] = "mixed_sign"
    with pytest.raises(BenchmarkAuditError, match="registry"):
        _audit_power_evidence(payload, scenario)
    with pytest.raises(ValueError, match="exactly one"):
        run_power_replicate(
            context, calibration_bank=calibration,
            calibration_scenario_id=calibration_id, replicate=0,
            architecture="minor_burden_aligned", interaction_pve=0.05,
            causal_groups=1, calibration_count=3, response_count=2,
            design_hash=design_hash, qa_only=True, n_jobs=1,
        )


def test_track_a_null_scan_gate_uses_any_rejection_and_fixed_denominator():
    scenario = Scenario(
        "A.loco.cotton.pve_0", "fit", "formal", 1_000, 0,
        {"panel": "cotton", "experiment": "loco", "scan_pve": 0.0},
    )
    rows = {name: [] for name in TABLE_SCHEMAS}
    rows["fit_scan_metrics.tsv"] = [
        {"scenario_id": scenario.scenario_id, "method": "canonical_multi_kernel",
         "scan_arm": "loco_sensitivity", "failed": False,
         "rejected": index < 50,
         "causal_detected": False}
        for index in range(1_000)
    ]
    evidence = SimpleNamespace(stage="formal", registry=(scenario,))
    gate = next(iter(_fit_scan_gates(evidence, rows).values()))
    assert gate.successes == 50
    assert gate.total == 1_000
    assert gate.upper_ci < 0.075
    assert gate.passed is True
    for index in range(11):
        rows["fit_scan_metrics.tsv"][-index - 1]["failed"] = True
    gate = next(iter(_fit_scan_gates(evidence, rows).values()))
    assert gate.failures == 11
    assert gate.passed is False
