from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest
import yaml
from bm_native_qa_harness import authority, cli


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
                "schema": "homoeogwas-bm-native-qa-source-input-reverification-v3",
                "status": "VERIFIED",
                "runtime_dependencies": {
                    "include_system_site_packages": False,
                    "python_no_user_site": True,
                    "required": [],
                    "thread_environment": {},
                    "threadpool_info": [],
                },
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
    bound_files["review.md"].write_text(
        "FINAL_CODE_REVIEW_VERDICT: ACCEPT\n"
        "REVIEWED_RUNNER_TEST_MAPPING_SHA256: "
        f"{authority._mapping_sha256(runner_hashes)}\n",
        encoding="utf-8",
    )
    design_payload = {
        "future_artifacts": {"root": str(tmp_path / "materialized")}
    }
    qa_design_hash = authority._mapping_sha256(design_payload)
    inventory = tmp_path / "inventory.json"
    inventory.write_text(
        json.dumps(
            {
                "schema": "homoeogwas-bm-native-qa-prospective-inventory-v5",
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
                "schema": "homoeogwas-bm-native-qa-materialization-authority-v2",
                "response_materialization_authorized": True,
                "execution_authorized": False,
                "run_namespace": "qa_real80_njobs128_v5",
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
    assert verified.run_namespace == "qa_real80_njobs128_v5"
    assert verified.inventory["schema"].endswith("v5")
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


def test_authority_distinguishes_reverification_schema_from_status(
    tmp_path: Path,
) -> None:
    authority_path, task_root = _write_fixture(tmp_path)
    record = yaml.safe_load(authority_path.read_text(encoding="utf-8"))
    reverify_path = Path(record["paths"]["source_input_reverification"])
    reverify = yaml.safe_load(reverify_path.read_text(encoding="utf-8"))
    reverify["schema"] = "homoeogwas-bm-native-qa-source-input-reverification-v2"
    reverify_path.write_text(yaml.safe_dump(reverify), encoding="utf-8")
    record["source_input_reverification_sha256"] = _sha256(reverify_path)
    authority_path.write_text(yaml.safe_dump(record), encoding="utf-8")

    with pytest.raises(authority.AuthorityBlocked, match="reverification schema"):
        authority.verify_materialization_authority(
            authority_path,
            expected_authority_path=authority_path,
            task_root=task_root,
            expected_successor_design_sha256=_sha256(tmp_path / "successor.yaml"),
            expected_worker_decision_sha256=_sha256(tmp_path / "decision.md"),
        )


def test_authority_requires_review_to_bind_runner_mapping(tmp_path: Path) -> None:
    authority_path, task_root = _write_fixture(tmp_path)
    record = yaml.safe_load(authority_path.read_text(encoding="utf-8"))
    review_path = Path(record["paths"]["runner_review"])
    review_path.write_text("FINAL_CODE_REVIEW_VERDICT: ACCEPT\n", encoding="utf-8")
    record["runner_review_sha256"] = _sha256(review_path)
    authority_path.write_text(yaml.safe_dump(record), encoding="utf-8")

    with pytest.raises(authority.AuthorityBlocked, match="runner/test mapping"):
        authority.verify_materialization_authority(
            authority_path,
            expected_authority_path=authority_path,
            task_root=task_root,
            expected_successor_design_sha256=_sha256(tmp_path / "successor.yaml"),
            expected_worker_decision_sha256=_sha256(tmp_path / "decision.md"),
        )


def test_materialization_launch_contract_is_exact(tmp_path: Path) -> None:
    task_root = tmp_path / "runtime-task-v5"
    helper_root = tmp_path / "r3-source"
    authority_path = tmp_path / "authority.yaml"
    artifact_root = tmp_path / "materialized" / "njobs128-v5"
    runtime = {
        "harness_task_root": str(task_root),
        "accepted_helper_source_root": str(helper_root),
        "materialization_launch": {
            "cwd": str(task_root),
            "pythonpath": str(helper_root),
            "python_no_user_site": True,
            "argv": [
                "materialize",
                "--authority",
                str(authority_path),
                "--out",
                str(artifact_root),
            ],
        },
    }
    verified = authority.VerifiedMaterializationAuthority(
        run_namespace="qa_real80_njobs128_v5",
        qa_design_hash="1" * 64,
        inventory={},
        artifact_root=artifact_root,
        paths={},
        binding_hashes={},
        source_reverification={"runtime": runtime},
    )

    authority.verify_materialization_launch_contract(
        verified,
        task_root=task_root,
        authority_path=authority_path,
        out_path=artifact_root,
        observed_cwd=task_root,
        observed_pythonpath=str(helper_root),
        observed_python_no_user_site="1",
    )

    runtime["materialization_launch"]["pythonpath"] = str(tmp_path / "wrong")
    with pytest.raises(authority.AuthorityBlocked, match="launch contract"):
        authority.verify_materialization_launch_contract(
            verified,
            task_root=task_root,
            authority_path=authority_path,
            out_path=artifact_root,
            observed_cwd=task_root,
            observed_pythonpath=str(helper_root),
            observed_python_no_user_site="1",
        )

    runtime["materialization_launch"]["pythonpath"] = str(helper_root)
    effective_mismatches = (
        {
            "observed_cwd": tmp_path / "wrong-cwd",
            "observed_pythonpath": str(helper_root),
            "observed_python_no_user_site": "1",
        },
        {
            "observed_cwd": task_root,
            "observed_pythonpath": str(helper_root) + ":" + str(tmp_path / "extra"),
            "observed_python_no_user_site": "1",
        },
        {
            "observed_cwd": task_root,
            "observed_pythonpath": str(helper_root),
            "observed_python_no_user_site": "0",
        },
    )
    for mismatch in effective_mismatches:
        with pytest.raises(authority.AuthorityBlocked, match="effective.*contract"):
            authority.verify_materialization_launch_contract(
                verified,
                task_root=task_root,
                authority_path=authority_path,
                out_path=artifact_root,
                **mismatch,
            )


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_authority_rejects_inexact_runtime_dependency_key_set(
    tmp_path: Path,
    mutation: str,
) -> None:
    authority_path, task_root = _write_fixture(tmp_path)
    record = yaml.safe_load(authority_path.read_text(encoding="utf-8"))
    reverify_path = Path(record["paths"]["source_input_reverification"])
    reverify = yaml.safe_load(reverify_path.read_text(encoding="utf-8"))
    dependencies = reverify["runtime_dependencies"]
    if mutation == "missing":
        dependencies.pop("threadpool_info")
    else:
        dependencies["unexpected"] = True
    reverify_path.write_text(yaml.safe_dump(reverify), encoding="utf-8")
    record["source_input_reverification_sha256"] = _sha256(reverify_path)
    authority_path.write_text(yaml.safe_dump(record), encoding="utf-8")

    with pytest.raises(authority.AuthorityBlocked, match="record is incomplete"):
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


@pytest.mark.parametrize(
    ("field", "forbidden"),
    [
        ("qa_design_hash", authority.FAILED_NJOBS128_V1_QA_DESIGN_HASH),
        (
            "prospective_inventory_sha256",
            authority.FAILED_NJOBS128_V1_INVENTORY_SHA256,
        ),
    ],
)
def test_authority_rejects_failed_njobs128_v1_flat_identity(
    tmp_path: Path,
    field: str,
    forbidden: str,
) -> None:
    authority_path, task_root = _write_fixture(tmp_path)
    payload = yaml.safe_load(authority_path.read_text(encoding="utf-8"))
    payload[field] = forbidden
    authority_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(authority.AuthorityBlocked, match="v1 identity"):
        authority.verify_materialization_authority(
            authority_path,
            expected_authority_path=authority_path,
            task_root=task_root,
            expected_successor_design_sha256=_sha256(tmp_path / "successor.yaml"),
            expected_worker_decision_sha256=_sha256(tmp_path / "decision.md"),
        )


@pytest.mark.parametrize(
    ("field", "forbidden"),
    [
        (
            "qa_design_hash",
            "523194f359e8fc0f07c30d345d5cef9eb3968456c8890277fd6c2bbc3ea3eca2",
        ),
        (
            "prospective_inventory_sha256",
            "584358ded7e8c7f3e028af5afac5834ee06fc4c85a1284bc6f966b039629dbb3",
        ),
        (
            "qa_design_hash",
            "383e6322c3cd6e7deccc8542f22f8a31d3e6638d92a0e03a62f1017542ff0c09",
        ),
        (
            "prospective_inventory_sha256",
            "16b67d9030ce32cc8a67f3551640a7d7470063494b3155eb7ac1b8ef2c204d96",
        ),
        (
            "qa_design_hash",
            "8a527004528387f4872f31030a33d6b0129b81f2d891565e88974713fc41562e",
        ),
        (
            "prospective_inventory_sha256",
            "c0361c4baab8b7b5ffe9781abb14c9305ed357c66625bdf595313e0c0ffa8e85",
        ),
        (
            "qa_design_hash",
            "62c8cd9a1ca3ccbfae8664813cb34c3b0166ba74efd00b738a4645b565172b75",
        ),
        (
            "prospective_inventory_sha256",
            "177a7db1104ba907eb78a9eca4be5bb2713046ce2fcc37061299871f21a79bc6",
        ),
    ],
)
def test_authority_rejects_rejected_v2_flat_identity(
    tmp_path: Path,
    field: str,
    forbidden: str,
) -> None:
    authority_path, task_root = _write_fixture(tmp_path)
    payload = yaml.safe_load(authority_path.read_text(encoding="utf-8"))
    payload[field] = forbidden
    authority_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(authority.AuthorityBlocked, match="rejected v2 identity"):
        authority.verify_materialization_authority(
            authority_path,
            expected_authority_path=authority_path,
            task_root=task_root,
            expected_successor_design_sha256=_sha256(tmp_path / "successor.yaml"),
            expected_worker_decision_sha256=_sha256(tmp_path / "decision.md"),
        )


@pytest.mark.parametrize(
    "filename",
    [
        "prospective-inventory.njobs128-v2.pre-code-review-changes-required-20260913.json",
        "prospective-inventory.njobs128-v2.design-delta-changes-required-20260913.json",
        "prospective-inventory.njobs128-v2.second-design-delta-changes-required-20260913.json",
        "prospective-inventory.njobs128-v2.json",
    ],
)
def test_authority_rejects_rejected_v2_runner_mapping(
    tmp_path: Path,
    filename: str,
) -> None:
    authority_path, task_root = _write_fixture(tmp_path)
    record = yaml.safe_load(authority_path.read_text(encoding="utf-8"))
    inventory_path = Path(record["paths"]["prospective_inventory"])
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    rejected = json.loads(
        (
            Path("/mnt/7302share/fast_ysp/U7_GWAS/tasks/BM-NATIVE-QA")
            / filename
        ).read_text(encoding="utf-8")
    )
    rejected_mapping = rejected["runner_test_sha256s"]
    inventory["runner_test_sha256s"] = rejected_mapping
    inventory_path.write_text(json.dumps(inventory), encoding="utf-8")
    record["runner_test_sha256s"] = rejected_mapping
    record["prospective_inventory_sha256"] = _sha256(inventory_path)
    authority_path.write_text(yaml.safe_dump(record), encoding="utf-8")

    with pytest.raises(authority.AuthorityBlocked, match="rejected v2 runner/test"):
        authority.verify_materialization_authority(
            authority_path,
            expected_authority_path=authority_path,
            task_root=task_root,
            expected_successor_design_sha256=_sha256(tmp_path / "successor.yaml"),
            expected_worker_decision_sha256=_sha256(tmp_path / "decision.md"),
        )


@pytest.mark.parametrize(
    ("field", "forbidden"),
    [
        (
            "qa_design_hash",
            "a78e2eca7ee9beab05d83167ac2b051b8db02d10b428cf08251d7852e56f1968",
        ),
        (
            "prospective_inventory_sha256",
            "9470941e68661f32a583fb04c72dd6aacdb5c6f35dc5f0aa6f1a5ca677150662",
        ),
        (
            "qa_design_hash",
            "ff8e917324aa5227b6fd4d1501ed3c08966b9f45cb5d10ec45f369d66c9d9215",
        ),
        (
            "prospective_inventory_sha256",
            "e6d8a3ed2f2ab75bddafdcd4628526f15fe70d68d9e17a1f6f53e833c78b81ca",
        ),
    ],
)
def test_authority_rejects_rejected_v3_flat_identity(
    tmp_path: Path,
    field: str,
    forbidden: str,
) -> None:
    authority_path, task_root = _write_fixture(tmp_path)
    payload = yaml.safe_load(authority_path.read_text(encoding="utf-8"))
    payload[field] = forbidden
    authority_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(authority.AuthorityBlocked, match="rejected v3 identity"):
        authority.verify_materialization_authority(
            authority_path,
            expected_authority_path=authority_path,
            task_root=task_root,
            expected_successor_design_sha256=_sha256(tmp_path / "successor.yaml"),
            expected_worker_decision_sha256=_sha256(tmp_path / "decision.md"),
        )


@pytest.mark.parametrize(
    "filename",
    [
        "prospective-inventory.njobs128-v3.pre-code-review-changes-required-20260913.json",
        "prospective-inventory.njobs128-v3.pre-real-context-preflight-failure-20260913.json",
        "prospective-inventory.njobs128-v3.pre-rejected-v3-identity-gate-20260913.json",
        "prospective-inventory.njobs128-v3.json",
    ],
)
def test_authority_rejects_rejected_v3_runner_mapping(
    tmp_path: Path,
    filename: str,
) -> None:
    authority_path, task_root = _write_fixture(tmp_path)
    record = yaml.safe_load(authority_path.read_text(encoding="utf-8"))
    inventory_path = Path(record["paths"]["prospective_inventory"])
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    rejected = json.loads(
        (
            Path("/mnt/7302share/fast_ysp/U7_GWAS/tasks/BM-NATIVE-QA")
            / filename
        ).read_text(encoding="utf-8")
    )
    rejected_mapping = rejected["runner_test_sha256s"]
    inventory["runner_test_sha256s"] = rejected_mapping
    inventory_path.write_text(json.dumps(inventory), encoding="utf-8")
    record["runner_test_sha256s"] = rejected_mapping
    record["prospective_inventory_sha256"] = _sha256(inventory_path)
    authority_path.write_text(yaml.safe_dump(record), encoding="utf-8")

    with pytest.raises(authority.AuthorityBlocked, match="rejected v3 runner/test"):
        authority.verify_materialization_authority(
            authority_path,
            expected_authority_path=authority_path,
            task_root=task_root,
            expected_successor_design_sha256=_sha256(tmp_path / "successor.yaml"),
            expected_worker_decision_sha256=_sha256(tmp_path / "decision.md"),
        )


def test_authority_rejects_failed_njobs128_v1_runner_mapping(tmp_path: Path) -> None:
    authority_path, task_root = _write_fixture(tmp_path)
    record = yaml.safe_load(authority_path.read_text(encoding="utf-8"))
    inventory_path = Path(record["paths"]["prospective_inventory"])
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    failed_inventory = json.loads(
        (
            Path("/mnt/7302share/fast_ysp/U7_GWAS")
            / "tasks/BM-NATIVE-QA/prospective-inventory.njobs128-v1.json"
        ).read_text(encoding="utf-8")
    )
    failed_mapping = failed_inventory["runner_test_sha256s"]
    inventory["runner_test_sha256s"] = failed_mapping
    inventory_path.write_text(json.dumps(inventory), encoding="utf-8")
    record["runner_test_sha256s"] = failed_mapping
    record["prospective_inventory_sha256"] = _sha256(inventory_path)
    authority_path.write_text(yaml.safe_dump(record), encoding="utf-8")

    with pytest.raises(authority.AuthorityBlocked, match="v1 runner/test"):
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
        ["git", "show", "3aa5e07f0c87575fae90463943b17ae141662cc1:"
         "tasks/BM-NATIVE-QA/prospective-inventory.v1.json"],
        cwd=Path(__file__).resolve().parents[3],
        text=True,
    )
    mapping = json.loads(raw)["runner_test_sha256s"]

    assert authority._mapping_sha256(mapping) == (
        authority.V1_RUNNER_TEST_MAPPING_SHA256
    )


def test_failed_njobs128_v1_literals_match_preserved_inventory() -> None:
    path = (
        Path("/mnt/7302share/fast_ysp/U7_GWAS")
        / "tasks/BM-NATIVE-QA/prospective-inventory.njobs128-v1.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert _sha256(path) == authority.FAILED_NJOBS128_V1_INVENTORY_SHA256
    assert payload["qa_design_hash"] == authority.FAILED_NJOBS128_V1_QA_DESIGN_HASH
    assert authority._mapping_sha256(payload["runner_test_sha256s"]) == (
        authority.FAILED_NJOBS128_V1_RUNNER_TEST_MAPPING_SHA256
    )


@pytest.mark.parametrize(
    ("filename", "inventory_sha256", "qa_design_hash", "mapping_sha256"),
    [
        (
            "prospective-inventory.njobs128-v2.pre-code-review-changes-required-20260913.json",
            "584358ded7e8c7f3e028af5afac5834ee06fc4c85a1284bc6f966b039629dbb3",
            "523194f359e8fc0f07c30d345d5cef9eb3968456c8890277fd6c2bbc3ea3eca2",
            "41c01cce14bfb214cfb654af1e9ec14f48bfd9e7dd2317984a8373ec42e0efd3",
        ),
        (
            "prospective-inventory.njobs128-v2.design-delta-changes-required-20260913.json",
            "16b67d9030ce32cc8a67f3551640a7d7470063494b3155eb7ac1b8ef2c204d96",
            "383e6322c3cd6e7deccc8542f22f8a31d3e6638d92a0e03a62f1017542ff0c09",
            "6a59398b21e32228fb9456e11a8a71a9ea1bf41e2493b18a5ec201d260b94931",
        ),
        (
            "prospective-inventory.njobs128-v2.second-design-delta-changes-required-20260913.json",
            "c0361c4baab8b7b5ffe9781abb14c9305ed357c66625bdf595313e0c0ffa8e85",
            "8a527004528387f4872f31030a33d6b0129b81f2d891565e88974713fc41562e",
            "931c4b5cd88b960e971f7ebf37d3fe3231c73db68c28f9d4531acf3d5ba1fbca",
        ),
        (
            "prospective-inventory.njobs128-v2.json",
            "177a7db1104ba907eb78a9eca4be5bb2713046ce2fcc37061299871f21a79bc6",
            "62c8cd9a1ca3ccbfae8664813cb34c3b0166ba74efd00b738a4645b565172b75",
            "13d8ef76a5e2c3bb9031c825ed485f761987189499393f875a80bfdcbfef9dbe",
        ),
    ],
)
def test_rejected_v2_literals_match_preserved_inventories(
    filename: str,
    inventory_sha256: str,
    qa_design_hash: str,
    mapping_sha256: str,
) -> None:
    path = Path("/mnt/7302share/fast_ysp/U7_GWAS/tasks/BM-NATIVE-QA") / filename
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert _sha256(path) == inventory_sha256
    assert payload["qa_design_hash"] == qa_design_hash
    assert authority._mapping_sha256(payload["runner_test_sha256s"]) == mapping_sha256
    assert qa_design_hash in authority.REJECTED_V2_QA_DESIGN_HASHES
    assert inventory_sha256 in authority.REJECTED_V2_INVENTORY_SHA256S
    assert mapping_sha256 in authority.REJECTED_V2_RUNNER_TEST_MAPPING_SHA256S


@pytest.mark.parametrize(
    ("filename", "inventory_sha256", "qa_design_hash", "mapping_sha256"),
    [
        (
            "prospective-inventory.njobs128-v3.pre-code-review-changes-required-20260913.json",
            "9470941e68661f32a583fb04c72dd6aacdb5c6f35dc5f0aa6f1a5ca677150662",
            "a78e2eca7ee9beab05d83167ac2b051b8db02d10b428cf08251d7852e56f1968",
            "af2328831e45d5903700381fd5960e883074d1f6da4e67f7ea0e1c1299e27800",
        ),
        (
            "prospective-inventory.njobs128-v3.pre-real-context-preflight-failure-20260913.json",
            "ed97d365a0b5f0cde7d4997ced0753ab44a14bb1cda4da758316569cc0c976db",
            "f741abbd968bb76bd56340cf715a0cc7031119388cb0de6d95d7b44cae5eaf7e",
            "d0bd1cc0a9c12dfd0b4b8aa0acadc0470a18b304a0fe891baa748de1d4e661fd",
        ),
        (
            "prospective-inventory.njobs128-v3.pre-rejected-v3-identity-gate-20260913.json",
            "1408fe7e9d5846cc7d785bda6d66be981b4f48346ca7de8903bb4fe3415a675b",
            "0b668c77c6792d566a18bc068564ec7736b177ad50105199e3f9490aed5b77fa",
            "13261cc62cdd3e9396f9a6cf236a3291fcec0f4735b82ab6f96401fb9c588482",
        ),
        (
            "prospective-inventory.njobs128-v3.json",
            "e6d8a3ed2f2ab75bddafdcd4628526f15fe70d68d9e17a1f6f53e833c78b81ca",
            "ff8e917324aa5227b6fd4d1501ed3c08966b9f45cb5d10ec45f369d66c9d9215",
            "6253c131a42ca6402b17736ac0076a367a41232a5e96b3b954f30fcaa87de6f6",
        ),
    ],
)
def test_rejected_v3_literals_match_preserved_inventories(
    filename: str,
    inventory_sha256: str,
    qa_design_hash: str,
    mapping_sha256: str,
) -> None:
    path = Path("/mnt/7302share/fast_ysp/U7_GWAS/tasks/BM-NATIVE-QA") / filename
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert _sha256(path) == inventory_sha256
    assert payload["qa_design_hash"] == qa_design_hash
    assert authority._mapping_sha256(payload["runner_test_sha256s"]) == mapping_sha256
    assert qa_design_hash in authority.REJECTED_V3_QA_DESIGN_HASHES
    assert inventory_sha256 in authority.REJECTED_V3_INVENTORY_SHA256S
    assert mapping_sha256 in authority.REJECTED_V3_RUNNER_TEST_MAPPING_SHA256S


@pytest.mark.parametrize(
    ("field", "forbidden"),
    [
        (
            "qa_design_hash",
            "947aab2d9c1a1682e42f03e76ea71a4447df43e6451094d93a42ac04aea13d15",
        ),
        (
            "prospective_inventory_sha256",
            "899ad4efd07383553f87560bf981d847ca4bb9d67bb9e2aa8e035497788df3f4",
        ),
    ],
)
def test_authority_rejects_v4_flat_identity(
    tmp_path: Path, field: str, forbidden: str,
) -> None:
    authority_path, task_root = _write_fixture(tmp_path)
    record = yaml.safe_load(authority_path.read_text(encoding="utf-8"))
    record[field] = forbidden
    authority_path.write_text(yaml.safe_dump(record), encoding="utf-8")

    with pytest.raises(authority.AuthorityBlocked, match="rejected v4 identity"):
        authority.verify_materialization_authority(
            authority_path,
            expected_authority_path=authority_path,
            task_root=task_root,
            expected_successor_design_sha256=_sha256(tmp_path / "successor.yaml"),
            expected_worker_decision_sha256=_sha256(tmp_path / "decision.md"),
        )


def test_authority_rejects_v4_runner_mapping(tmp_path: Path) -> None:
    authority_path, task_root = _write_fixture(tmp_path)
    record = yaml.safe_load(authority_path.read_text(encoding="utf-8"))
    inventory_path = Path(record["paths"]["prospective_inventory"])
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    preserved = json.loads(
        Path(
            "/mnt/7302share/fast_ysp/U7_GWAS/tasks/BM-NATIVE-QA/"
            "prospective-inventory.njobs128-v4.json"
        ).read_text(encoding="utf-8")
    )
    mapping = preserved["runner_test_sha256s"]
    inventory["runner_test_sha256s"] = mapping
    inventory_path.write_text(json.dumps(inventory), encoding="utf-8")
    record["runner_test_sha256s"] = mapping
    record["prospective_inventory_sha256"] = _sha256(inventory_path)
    authority_path.write_text(yaml.safe_dump(record), encoding="utf-8")

    with pytest.raises(authority.AuthorityBlocked, match="rejected v4 runner/test"):
        authority.verify_materialization_authority(
            authority_path,
            expected_authority_path=authority_path,
            task_root=task_root,
            expected_successor_design_sha256=_sha256(tmp_path / "successor.yaml"),
            expected_worker_decision_sha256=_sha256(tmp_path / "decision.md"),
        )


def test_v4_literals_and_authority_match_preserved_bytes() -> None:
    task_root = Path("/mnt/7302share/fast_ysp/U7_GWAS/tasks/BM-NATIVE-QA")
    inventory_path = task_root / "prospective-inventory.njobs128-v4.json"
    authority_path = task_root / (
        "QA-NJOBS128-RESPONSE-MATERIALIZATION-AUTHORIZATION-v4-20260913.yaml"
    )
    payload = json.loads(inventory_path.read_text(encoding="utf-8"))
    record = yaml.safe_load(authority_path.read_text(encoding="utf-8"))
    mapping_hash = hashlib.sha256(
        json.dumps(
            payload["runner_test_sha256s"], sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    assert payload["qa_design_hash"] == record["qa_design_hash"] == (
        "947aab2d9c1a1682e42f03e76ea71a4447df43e6451094d93a42ac04aea13d15"
    )
    assert _sha256(inventory_path) == record["prospective_inventory_sha256"] == (
        "899ad4efd07383553f87560bf981d847ca4bb9d67bb9e2aa8e035497788df3f4"
    )
    assert payload["runner_test_sha256s"] == record["runner_test_sha256s"]
    assert mapping_hash == (
        "c169100739ddcc36d3343da0e34b4e39116e8f3b19f8581e490fd44223335fe1"
    )
    assert _sha256(authority_path) == (
        "5e5204ca2243fc9bafd9f6aa258d81aecf90702301dccd203564d6a40c0855a4"
    )


def test_active_authority_path_rejects_preserved_v4_authority() -> None:
    task_root = Path("/mnt/7302share/fast_ysp/U7_GWAS/tasks/BM-NATIVE-QA")
    authority_path = task_root / (
        "QA-NJOBS128-RESPONSE-MATERIALIZATION-AUTHORIZATION-v4-20260913.yaml"
    )

    with pytest.raises(authority.AuthorityBlocked, match="exact reviewed path"):
        authority.verify_materialization_authority(
            authority_path,
            expected_authority_path=cli.MATERIALIZATION_AUTHORITY_PATH,
            task_root=task_root,
            expected_successor_design_sha256=cli.SUCCESSOR_DESIGN_SHA256,
            expected_worker_decision_sha256=cli.WORKER_DECISION_SHA256,
        )
