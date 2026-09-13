from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest
import yaml
from bm_native_qa_harness import authority


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_fixture(tmp_path: Path) -> tuple[Path, Path]:
    task_root = tmp_path / "task"
    runner = task_root / "bm_native_qa_harness" / "runner.py"
    test = task_root / "tests" / "test_runner.py"
    runner.parent.mkdir(parents=True)
    test.parent.mkdir(parents=True)
    runner.write_text("VALUE = 1\n", encoding="utf-8")
    test.write_text("def test_value(): pass\n", encoding="utf-8")

    bound_files = {}
    for name, text in (
        ("successor.yaml", "successor\n"),
        ("decision.md", "decision\n"),
        ("review.md", "FINAL_CODE_REVIEW_VERDICT: ACCEPT\n"),
        ("input.bin", "input\n"),
        ("context.json", '{"phenotype_values_read": false}\n'),
        ("amendment.yaml", "status: accepted\n"),
        ("fixture.json", "{}\n"),
    ):
        path = tmp_path / name
        path.write_text(text, encoding="utf-8")
        bound_files[name] = path

    reverify = tmp_path / "reverify.yaml"
    reverify.write_text(
        yaml.safe_dump(
            {
                "schema": "homoeogwas-bm-native-qa-source-input-reverification-v2",
                "status": "VERIFIED",
                "files": [
                    {
                        "path": str(bound_files["input.bin"]),
                        "sha256": _sha256(bound_files["input.bin"]),
                    }
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    runner_hashes = {
        "bm_native_qa_harness/runner.py": _sha256(runner),
        "tests/test_runner.py": _sha256(test),
    }
    design_payload = {
        "future_artifacts": {"root": str(tmp_path / "materialized")}
    }
    qa_design_hash = authority._mapping_sha256(design_payload)
    inventory = tmp_path / "inventory.json"
    inventory.write_text(
        json.dumps(
            {
                "schema": "homoeogwas-bm-native-qa-prospective-inventory-v2",
                "response_materialization_authorized": False,
                "execution_authorized": False,
                "qa_design_hash": qa_design_hash,
                "runner_test_sha256s": runner_hashes,
                "design_payload": design_payload,
                "source_bindings": {
                    "successor_design_sha256": _sha256(
                        bound_files["successor.yaml"]
                    ),
                    "worker_decision_sha256": _sha256(bound_files["decision.md"]),
                    "context_evidence_sha256": _sha256(
                        bound_files["context.json"]
                    ),
                    "amendment_sha256": _sha256(bound_files["amendment.yaml"]),
                    "fixture_manifest_sha256": _sha256(bound_files["fixture.json"]),
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    authority_path = tmp_path / "authority.yaml"
    authority_path.write_text(
        yaml.safe_dump(
            {
                "schema": "homoeogwas-bm-native-qa-materialization-authority-v1",
                "response_materialization_authorized": True,
                "execution_authorized": False,
                "run_namespace": "qa_real80_njobs128_v1",
                "successor_design_sha256": _sha256(bound_files["successor.yaml"]),
                "decision_sha256": _sha256(bound_files["decision.md"]),
                "qa_design_hash": qa_design_hash,
                "prospective_inventory_sha256": _sha256(inventory),
                "runner_test_sha256s": runner_hashes,
                "runner_review_sha256": _sha256(bound_files["review.md"]),
                "runner_review_verdict": "ACCEPT",
                "source_input_reverification_sha256": _sha256(reverify),
                "paths": {
                    "successor_design": str(bound_files["successor.yaml"]),
                    "decision": str(bound_files["decision.md"]),
                    "prospective_inventory": str(inventory),
                    "runner_review": str(bound_files["review.md"]),
                    "source_input_reverification": str(reverify),
                    "artifact_root": str(tmp_path / "materialized"),
                    "context_evidence": str(bound_files["context.json"]),
                    "amendment": str(bound_files["amendment.yaml"]),
                    "fixture_manifest": str(bound_files["fixture.json"]),
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return authority_path, task_root


def test_authority_verifies_all_bindings_before_returning(tmp_path: Path) -> None:
    authority_path, task_root = _write_fixture(tmp_path)

    verified = authority.verify_materialization_authority(
        authority_path,
        expected_authority_path=authority_path,
        task_root=task_root,
        expected_successor_design_sha256=_sha256(tmp_path / "successor.yaml"),
        expected_worker_decision_sha256=_sha256(tmp_path / "decision.md"),
    )

    assert verified.qa_design_hash == authority._mapping_sha256(
        verified.inventory["design_payload"]
    )
    assert verified.inventory["schema"].endswith("v2")
    assert verified.artifact_root == tmp_path / "materialized"
    assert verified.binding_hashes["materialization_authority_sha256"] == _sha256(
        authority_path
    )


def test_authority_rehashes_reverified_files_and_fails_closed(tmp_path: Path) -> None:
    authority_path, task_root = _write_fixture(tmp_path)
    (tmp_path / "input.bin").write_text("changed\n", encoding="utf-8")

    with pytest.raises(authority.AuthorityBlocked, match="reverified file hash"):
        authority.verify_materialization_authority(
            authority_path,
            expected_authority_path=authority_path,
            task_root=task_root,
            expected_successor_design_sha256=_sha256(tmp_path / "successor.yaml"),
            expected_worker_decision_sha256=_sha256(tmp_path / "decision.md"),
        )


def test_authority_rejects_inventory_payload_hash_mismatch(tmp_path: Path) -> None:
    authority_path, task_root = _write_fixture(tmp_path)
    record = yaml.safe_load(authority_path.read_text(encoding="utf-8"))
    inventory_path = Path(record["paths"]["prospective_inventory"])
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    inventory["design_payload"]["future_artifacts"]["extra"] = "tampered"
    inventory_path.write_text(json.dumps(inventory), encoding="utf-8")
    record["prospective_inventory_sha256"] = _sha256(inventory_path)
    authority_path.write_text(yaml.safe_dump(record), encoding="utf-8")

    with pytest.raises(authority.AuthorityBlocked, match="payload hash"):
        authority.verify_materialization_authority(
            authority_path,
            expected_authority_path=authority_path,
            task_root=task_root,
            expected_successor_design_sha256=_sha256(tmp_path / "successor.yaml"),
            expected_worker_decision_sha256=_sha256(tmp_path / "decision.md"),
        )


def test_authority_rejects_changed_or_nonblind_context_evidence(tmp_path: Path) -> None:
    authority_path, task_root = _write_fixture(tmp_path)
    record = yaml.safe_load(authority_path.read_text(encoding="utf-8"))
    context_path = Path(record["paths"]["context_evidence"])
    context_path.write_text('{"phenotype_values_read": true}\n', encoding="utf-8")
    inventory_path = Path(record["paths"]["prospective_inventory"])
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    inventory["source_bindings"]["context_evidence_sha256"] = _sha256(context_path)
    inventory_path.write_text(json.dumps(inventory), encoding="utf-8")
    record["prospective_inventory_sha256"] = _sha256(inventory_path)
    authority_path.write_text(yaml.safe_dump(record), encoding="utf-8")

    with pytest.raises(authority.AuthorityBlocked, match="phenotype-blind"):
        authority.verify_materialization_authority(
            authority_path,
            expected_authority_path=authority_path,
            task_root=task_root,
            expected_successor_design_sha256=_sha256(tmp_path / "successor.yaml"),
            expected_worker_decision_sha256=_sha256(tmp_path / "decision.md"),
        )


def test_authority_rejects_unlisted_runner_source(tmp_path: Path) -> None:
    authority_path, task_root = _write_fixture(tmp_path)
    (task_root / "bm_native_qa_harness" / "late.py").write_text(
        "VALUE = 2\n", encoding="utf-8"
    )

    with pytest.raises(authority.AuthorityBlocked, match="path set"):
        authority.verify_materialization_authority(
            authority_path,
            expected_authority_path=authority_path,
            task_root=task_root,
            expected_successor_design_sha256=_sha256(tmp_path / "successor.yaml"),
            expected_worker_decision_sha256=_sha256(tmp_path / "decision.md"),
        )


def test_runtime_isolation_requires_fully_bound_runtime_record(tmp_path: Path) -> None:
    authority_path, task_root = _write_fixture(tmp_path)
    verified = authority.verify_materialization_authority(
        authority_path,
        expected_authority_path=authority_path,
        task_root=task_root,
        expected_successor_design_sha256=_sha256(tmp_path / "successor.yaml"),
        expected_worker_decision_sha256=_sha256(tmp_path / "decision.md"),
    )

    with pytest.raises(authority.AuthorityBlocked, match="runtime isolation"):
        authority.verify_runtime_import_isolation(verified, task_root=task_root)


def test_python_source_tree_hash_binds_paths_and_bytes(tmp_path: Path) -> None:
    root = tmp_path / "package"
    root.mkdir()
    (root / "a.py").write_text("A = 1\n", encoding="utf-8")
    first = authority.python_source_tree_sha256(root, domain="test-domain-v1")
    (root / "a.py").write_text("A = 2\n", encoding="utf-8")
    second = authority.python_source_tree_sha256(root, domain="test-domain-v1")

    assert len(first) == 64
    assert first != second


def test_authority_exact_path_is_mandatory(tmp_path: Path) -> None:
    authority_path, task_root = _write_fixture(tmp_path)

    with pytest.raises(authority.AuthorityBlocked, match="exact reviewed path"):
        authority.verify_materialization_authority(
            authority_path,
            expected_authority_path=tmp_path / "different.yaml",
            task_root=task_root,
            expected_successor_design_sha256=_sha256(tmp_path / "successor.yaml"),
            expected_worker_decision_sha256=_sha256(tmp_path / "decision.md"),
        )


def test_authority_rejects_any_v1_design_identity(tmp_path: Path) -> None:
    authority_path, task_root = _write_fixture(tmp_path)
    payload = yaml.safe_load(authority_path.read_text(encoding="utf-8"))
    payload["qa_design_hash"] = (
        "65ba6556d86db4d27b03169ff70d5b39796c5968ce8ece9f782e866995cf5d5e"
    )
    authority_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(authority.AuthorityBlocked, match="v1 identity"):
        authority.verify_materialization_authority(
            authority_path,
            expected_authority_path=authority_path,
            task_root=task_root,
            expected_successor_design_sha256=_sha256(tmp_path / "successor.yaml"),
            expected_worker_decision_sha256=_sha256(tmp_path / "decision.md"),
        )


def test_authority_rejects_v1_runner_mapping_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority_path, task_root = _write_fixture(tmp_path)
    record = yaml.safe_load(authority_path.read_text(encoding="utf-8"))
    monkeypatch.setattr(
        authority,
        "V1_RUNNER_TEST_MAPPING_SHA256",
        authority._mapping_sha256(record["runner_test_sha256s"]),
    )

    with pytest.raises(authority.AuthorityBlocked, match="v1 runner/test"):
        authority.verify_materialization_authority(
            authority_path,
            expected_authority_path=authority_path,
            task_root=task_root,
            expected_successor_design_sha256=_sha256(tmp_path / "successor.yaml"),
            expected_worker_decision_sha256=_sha256(tmp_path / "decision.md"),
        )


def test_v1_runner_mapping_literal_matches_accepted_commit() -> None:
    raw = subprocess.check_output(
        [
            "git",
            "show",
            "3aa5e07f0c87575fae90463943b17ae141662cc1:"
            "tasks/BM-NATIVE-QA/prospective-inventory.v1.json",
        ],
        cwd="/tmp/U7_GWAS-bm-native-qa-materialize-v1",
        text=True,
    )
    mapping = json.loads(raw)["runner_test_sha256s"]

    assert authority._mapping_sha256(mapping) == (
        authority.V1_RUNNER_TEST_MAPPING_SHA256
    )
