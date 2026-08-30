from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest
import yaml

from scripts.benchmarks.v201.track_application import export_application_rows


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _result_payload(*, discoveries: int = 2) -> dict:
    significant = [
        {
            "hypothesis_id": "edge:AC:gA:gC",
            "p_adjusted_bootstrap_minp": 0.01,
            "smallest_component": "minor_burden",
        },
        {
            "hypothesis_id": "edge:AC:hA:hC",
            "p_adjusted_bootstrap_minp": 0.04,
            "smallest_component": "kernel_hadamard",
        },
    ][:discoveries]
    return {
        "tool": "homoeogwas",
        "command": "interact",
        "mode": "group",
        "subgenomes": ["A", "C"],
        "trait": "flowering_time",
        "provenance": {
            "n_samples": 926,
            "hypothesis_unit": "edge",
            "calibration_method": "bootstrap",
            "parallel_execution": {
                "requested_jobs": 8,
                "effective_jobs": 8,
                "backend": "fork_shared_memory",
                "worker_pids": [101, 102],
            },
            "group_family_sha256": "1" * 64,
            "edge_family_sha256": "2" * 64,
            "n_groups_raw": 17_404,
            "n_unique_edges": 17_404,
        },
        "results": {
            "INT": {
                "n": 926,
                "G": 17_404,
                "n_sig": discoveries,
                "sig": significant,
                "n_planned": 17_404,
                "n_valid": 17_404,
                "calibration_method": "bootstrap",
                "bootstrap_B": 2000,
                "component_diagnostics": {
                    "smallest_component_counts": {
                        "minor_burden": 6000,
                        "pc1": 5000,
                        "kernel_hadamard": 6404,
                    },
                },
                "model_diagnostics": {
                    "bootstrap_fwer": {
                        "family_id": "edge",
                        "declared_hypothesis_unit": "edge",
                        "n_hypotheses": 17_404,
                        "B": 2000,
                    },
                    "grm_provenance": {
                        "method": "grm_from_X",
                        "subgenomes": {
                            "A": {"n_variants_input": 735_832},
                            "C": {"n_variants_input": 1_080_939},
                        },
                    },
                },
            },
        },
    }


def _audit_payload(result: Path, *, discoveries: int = 2, status: str = "AUDIT_COMPLETE") -> dict:
    return {
        "overall_status": status,
        "n_results": 1,
        "records": [{
            "source": str(result.resolve()),
            "command": "interact",
            "trait": "flowering_time",
            "mode": "group",
            "status": status,
            "discovery_count": discoveries,
            "n": 926,
            "n_planned": 17_404,
            "n_valid": 17_404,
            "calibration": "omniB+bootstrap(B=2000)",
            "replication_status": "NOT_ASSESSED",
            "evidence_boundary": ["Independent validation remains separate."],
        }],
    }


def _historical_run(root: Path, *, run_id: str = "rapeseed.model-a", species: str = "Brassica napus") -> dict:
    result = root / "interact_flowering_time.json"
    audit = root / "audit" / "homoeogwas_audit.json"
    return {
        "id": run_id,
        "kind": "historical",
        "species": species,
        "panel": "fixture panel",
        "subgenomes": ["A", "C"],
        "result_root": str(root),
        "analysis_shape": "canonical_group_omnib_edge_primary",
        "artifact_inventory": [
            {"role": "result", "path": result.relative_to(root).as_posix(),
             "size": result.stat().st_size, "sha256": _sha256(result)},
            {"role": "audit", "path": audit.relative_to(root).as_posix(),
             "size": audit.stat().st_size, "sha256": _sha256(audit)},
        ],
        "expected": {
            "n_significant": 2,
            "primary_unit": "edge",
            "n_planned": 17_404,
            "limitations": ["Marker density differs among subgenomes."],
        },
    }


def _write_registry(tmp_path: Path, runs: list[dict]) -> Path:
    path = tmp_path / "registry.yaml"
    path.write_text(yaml.safe_dump({
        "registry_version": 1,
        "name": "application-fixture",
        "index_dir": "index",
        "runs": runs,
    }, sort_keys=False))
    return path


def _materialize_run(tmp_path: Path, *, name: str = "run", discoveries: int = 2) -> tuple[Path, dict]:
    root = tmp_path / name
    result = root / "interact_flowering_time.json"
    audit = root / "audit" / "homoeogwas_audit.json"
    _write_json(result, _result_payload(discoveries=discoveries))
    _write_json(audit, _audit_payload(result, discoveries=discoveries))
    run = _historical_run(root, run_id=name)
    run["expected"]["n_significant"] = discoveries
    return root, run


def test_exports_complete_read_only_application_evidence(tmp_path, monkeypatch):
    root, run = _materialize_run(tmp_path)
    before = {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("no subprocess"))

    rows = export_application_rows(_write_registry(tmp_path, [run]))

    assert len(rows) == 1
    row = rows[0]
    assert row["analysis_id"] == "run"
    assert row["species"] == "Brassica napus"
    assert row["ploidy"] == 4
    assert row["subgenomes"] == ["A", "C"]
    assert row["sample_count"] == 926
    assert row["marker_count"] == 1_816_771
    assert row["group_family_count"] == 17_404
    assert row["edge_family_count"] == 17_404
    assert row["requested_jobs"] == row["effective_jobs"] == 8
    assert row["backend"] == "fork_shared_memory"
    assert row["worker_pids"] == [101, 102]
    assert row["primary_unit"] == "edge"
    assert row["calibration_method"] == "bootstrap"
    assert row["calibration_B"] == 2000
    assert row["adjusted_discovery_count"] == 2
    assert row["negative_result"] is False
    assert row["adjusted_discoveries"] == [
        {"hypothesis_id": "edge:AC:gA:gC", "adjusted_p": 0.01},
        {"hypothesis_id": "edge:AC:hA:hC", "adjusted_p": 0.04},
    ]
    assert row["component_driver_distribution"] == {
        "minor_burden": 1, "kernel_hadamard": 1,
    }
    assert row["audit_status"] == row["status"] == "AUDIT_COMPLETE"
    assert row["family_hash"] == "2" * 64
    assert row["repair_required"] is False
    assert "Independent validation remains separate." in row["limitations"]
    assert "Marker density differs among subgenomes." in row["limitations"]
    after = {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}
    assert after == before


def test_missing_application_artifact_is_reported_not_run(tmp_path, monkeypatch):
    run = {
        "id": "cotton_missing",
        "kind": "historical",
        "species": "Gossypium hirsutum",
        "panel": "fixture",
        "subgenomes": ["A", "D"],
        "result_root": "missing",
        "analysis_shape": "group_omnib",
        "artifact_inventory": [
            {"role": "result", "path": "result.json", "size": 1, "sha256": "a" * 64},
            {"role": "audit", "path": "audit/homoeogwas_audit.json", "size": 1,
             "sha256": "b" * 64},
        ],
    }
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("no subprocess"))

    rows = export_application_rows(_write_registry(tmp_path, [run]))

    assert rows[0]["status"] == "MISSING_AUTHORITATIVE_OUTPUT"
    assert rows[0]["repair_required"] is True
    assert "missing" in rows[0]["repair_reason"].lower()


def test_inventory_hash_mismatch_fails_closed(tmp_path):
    root, run = _materialize_run(tmp_path)
    run["artifact_inventory"][0]["sha256"] = "f" * 64

    row = export_application_rows(_write_registry(tmp_path, [run]))[0]

    assert row["status"] == "AUTHORITATIVE_ARTIFACT_MISMATCH"
    assert row["repair_required"] is True
    assert row["audit_status"] is None
    assert str(root) in row["repair_reason"]


def test_malformed_json_and_failed_audit_are_explicit(tmp_path):
    malformed_root, malformed = _materialize_run(tmp_path, name="malformed")
    result = malformed_root / "interact_flowering_time.json"
    result.write_text("{not-json")
    malformed["artifact_inventory"][0].update(
        size=result.stat().st_size, sha256=_sha256(result))

    failed_root, failed = _materialize_run(tmp_path, name="failed")
    failed_result = failed_root / "interact_flowering_time.json"
    audit = failed_root / "audit" / "homoeogwas_audit.json"
    _write_json(audit, _audit_payload(
        failed_result, status="ANALYSIS_INVALID"))
    failed["artifact_inventory"][1].update(
        size=audit.stat().st_size, sha256=_sha256(audit))

    rows = export_application_rows(_write_registry(tmp_path, [malformed, failed]))

    assert [row["status"] for row in rows] == [
        "UNREADABLE_AUTHORITATIVE_OUTPUT", "AUDIT_FAILED"]
    assert all(row["repair_required"] for row in rows)


def test_result_audit_semantic_mismatch_is_not_exported_as_pass(tmp_path):
    root, run = _materialize_run(tmp_path)
    audit = root / "audit" / "homoeogwas_audit.json"
    payload = _audit_payload(root / "interact_flowering_time.json", discoveries=1)
    _write_json(audit, payload)
    run["artifact_inventory"][1].update(size=audit.stat().st_size, sha256=_sha256(audit))

    row = export_application_rows(_write_registry(tmp_path, [run]))[0]

    assert row["status"] == "SEMANTIC_MISMATCH"
    assert row["repair_required"] is True
    assert "discovery" in row["repair_reason"].lower()


def test_missing_critical_family_or_execution_fields_is_schema_incomplete(tmp_path):
    root, run = _materialize_run(tmp_path)
    result = root / "interact_flowering_time.json"
    payload = _result_payload()
    payload["provenance"].pop("edge_family_sha256")
    payload["provenance"].pop("parallel_execution")
    _write_json(result, payload)
    run["artifact_inventory"][0].update(size=result.stat().st_size, sha256=_sha256(result))

    row = export_application_rows(_write_registry(tmp_path, [run]))[0]

    assert row["status"] == "SCHEMA_INCOMPLETE"
    assert row["audit_status"] == "AUDIT_COMPLETE"
    assert row["family_hash"] is None
    assert row["repair_required"] is True
    assert "family_hash" in row["repair_reason"]
    assert "requested_jobs" in row["repair_reason"]


def test_custom_release_audit_and_pairwise_aliases_are_strictly_supported(tmp_path):
    root, run = _materialize_run(tmp_path)
    result_path = root / "interact_flowering_time.json"
    result = _result_payload()
    result["mode"] = "pairwise"
    result["hypothesis_unit"] = "edge"
    result["provenance"]["mode"] = "pairwise"
    result["provenance"].pop("hypothesis_unit")
    result["family_hash"] = result["provenance"].pop(
        "edge_family_sha256")
    result["parallel_execution"] = result["provenance"].pop("parallel_execution")
    result["provenance"]["n_units_raw"] = result["provenance"].pop(
        "n_unique_edges")
    _write_json(result_path, result)
    audit_path = root / "audit" / "homoeogwas_audit.json"
    _write_json(audit_path, {
        "status": "PASS",
        "output_sha256": {"interact_json": _sha256(result_path)},
        "component_driver_counts_among_formal_hits": {
            "minor_burden": 1, "kernel_hadamard": 1,
        },
        "limitations": ["Legacy pairwise release schema."],
    })
    run["analysis_shape"] = "legacy_pairwise_omnib"
    run["expected"]["primary_unit"] = "edge"
    run["artifact_inventory"] = [
        {"role": "result", "path": result_path.relative_to(root).as_posix(),
         "size": result_path.stat().st_size, "sha256": _sha256(result_path)},
        {"role": "audit", "path": audit_path.relative_to(root).as_posix(),
         "size": audit_path.stat().st_size, "sha256": _sha256(audit_path)},
    ]

    row = export_application_rows(_write_registry(tmp_path, [run]))[0]

    assert row["status"] == "PASS"
    assert row["primary_unit"] == "edge"
    assert row["edge_family_count"] == 17_404
    assert row["family_hash"] == "2" * 64
    assert row["component_driver_distribution"] == {
        "minor_burden": 1, "kernel_hadamard": 1,
    }
    assert row["repair_required"] is False


def test_legacy_triad_alias_is_group_primary_without_merging_models(tmp_path):
    root, run = _materialize_run(tmp_path, name="wheat.model-a")
    result_path = root / "interact_flowering_time.json"
    result = _result_payload()
    result["mode"] = "triad"
    result["subgenomes"] = ["A", "B", "D"]
    result["provenance"]["mode"] = "triad"
    result["provenance"].pop("hypothesis_unit")
    result["provenance"].pop("n_unique_edges")
    fwer = result["results"]["INT"]["model_diagnostics"]["bootstrap_fwer"]
    fwer["family_id"] = fwer["declared_hypothesis_unit"] = "group"
    _write_json(result_path, result)
    audit_path = root / "audit" / "homoeogwas_audit.json"
    audit = _audit_payload(result_path)
    audit["records"][0]["mode"] = "triad"
    _write_json(audit_path, audit)
    run["species"] = "Triticum aestivum"
    run["subgenomes"] = ["A", "B", "D"]
    run["analysis_shape"] = "legacy_route_b_group_omnib"
    run["expected"]["primary_unit"] = "group"
    run["artifact_inventory"] = [
        {"role": "result", "path": result_path.relative_to(root).as_posix(),
         "size": result_path.stat().st_size, "sha256": _sha256(result_path)},
        {"role": "audit", "path": audit_path.relative_to(root).as_posix(),
         "size": audit_path.stat().st_size, "sha256": _sha256(audit_path)},
    ]

    row = export_application_rows(_write_registry(tmp_path, [run]))[0]

    assert row["status"] == "AUDIT_COMPLETE"
    assert row["ploidy"] == 6
    assert row["primary_unit"] == "group"
    assert row["group_family_count"] == 17_404
    assert row["edge_family_count"] == 52_212
    assert row["family_hash"] == "1" * 64


@pytest.mark.parametrize("mutation", ["fwer_unit", "audit_B", "family_hash"])
def test_family_and_calibration_semantics_fail_closed(tmp_path, mutation):
    root, run = _materialize_run(tmp_path)
    result_path = root / "interact_flowering_time.json"
    audit_path = root / "audit" / "homoeogwas_audit.json"
    if mutation == "fwer_unit":
        result = _result_payload()
        fwer = result["results"]["INT"]["model_diagnostics"]["bootstrap_fwer"]
        fwer["family_id"] = fwer["declared_hypothesis_unit"] = "group"
        _write_json(result_path, result)
        run["artifact_inventory"][0].update(
            size=result_path.stat().st_size, sha256=_sha256(result_path))
    elif mutation == "audit_B":
        audit = _audit_payload(result_path)
        audit["records"][0]["calibration"] = "omniB+bootstrap(B=999)"
        _write_json(audit_path, audit)
        run["artifact_inventory"][1].update(
            size=audit_path.stat().st_size, sha256=_sha256(audit_path))
    else:
        result = _result_payload()
        result["provenance"]["edge_family_sha256"] = "not-a-sha256"
        _write_json(result_path, result)
        run["artifact_inventory"][0].update(
            size=result_path.stat().st_size, sha256=_sha256(result_path))

    row = export_application_rows(_write_registry(tmp_path, [run]))[0]

    assert row["repair_required"] is True
    assert row["status"] in {"SEMANTIC_MISMATCH", "SCHEMA_INCOMPLETE"}


def test_same_species_models_remain_independent_rows(tmp_path):
    _, model_a = _materialize_run(tmp_path, name="wheat.model-a", discoveries=2)
    _, model_b = _materialize_run(tmp_path, name="wheat.model-b", discoveries=0)
    for run in (model_a, model_b):
        run["species"] = "Triticum aestivum"

    rows = export_application_rows(_write_registry(tmp_path, [model_a, model_b]))

    assert [row["analysis_id"] for row in rows] == ["wheat.model-a", "wheat.model-b"]
    assert [row["adjusted_discovery_count"] for row in rows] == [2, 0]
    assert rows[1]["negative_result"] is True


def test_inventory_with_multiple_results_is_rejected_without_guessing(tmp_path):
    root, run = _materialize_run(tmp_path)
    extra = root / "interact_second.json"
    _write_json(extra, _result_payload())
    run["artifact_inventory"].insert(1, {
        "role": "result", "path": "interact_second.json",
        "size": extra.stat().st_size, "sha256": _sha256(extra),
    })

    row = export_application_rows(_write_registry(tmp_path, [run]))[0]

    assert row["status"] == "AMBIGUOUS_AUTHORITATIVE_INVENTORY"
    assert row["repair_required"] is True
