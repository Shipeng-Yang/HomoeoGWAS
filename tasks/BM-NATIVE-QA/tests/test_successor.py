from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from bm_native_qa_harness import cli, identity, plan

MANAGEMENT_ROOT = Path("/mnt/7302share/fast_ysp/U7_GWAS")
SUCCESSOR_PATH = (
    MANAGEMENT_ROOT
    / "tasks/BM-NATIVE-QA/QA-NJOBS128-SUCCESSOR-DESIGN-v2-20260913.yaml"
)
AMENDMENT_PATH = (
    MANAGEMENT_ROOT
    / "tasks/BM-NATIVE-QA/QA-EXECUTION-AMENDMENT-v1-20260910.yaml"
)
CONTEXT_EVIDENCE_PATH = (
    MANAGEMENT_ROOT
    / "tasks/BM-INPUTS/staging/real-core-v4-primary80-estimability/manifest.json"
)
FIXTURE_MANIFEST_PATH = (
    MANAGEMENT_ROOT / "tasks/BM-FIXTURE-V3/manifest.20260910-v5-r3.json"
)
FROZEN_INVENTORY_PATH = (
    MANAGEMENT_ROOT / "tasks/BM-NATIVE-QA/prospective-inventory.njobs128-v2.json"
)


def _successor() -> dict:
    return yaml.safe_load(SUCCESSOR_PATH.read_text(encoding="utf-8"))


def test_successor_inventory_has_six_twelve_sixteen_and_only_128_workers(
    amendment: dict,
) -> None:
    inventory = plan.build_successor_inventory(amendment, _successor())

    assert len(inventory.contexts) == 6
    assert len(inventory.responses) == 12
    assert len(inventory.invocations) == 16
    assert {row.jobs for row in inventory.invocations} == {128}
    assert len({row.invocation_id for row in inventory.invocations}) == 16
    assert all(
        row.response_id.startswith("qa_real80_njobs128_v2.")
        for row in inventory.responses
    )


def test_pc1_successor_replicas_share_response_but_have_exclusive_ids(
    amendment: dict,
) -> None:
    rows = [
        row
        for row in plan.build_successor_inventory(amendment, _successor()).invocations
        if row.panel_id == "REALG.CGVD1245"
        and row.sample_context == "pc1_spread_192"
        and row.truth_id == "gaussian_null"
    ]

    assert [(row.jobs, row.invocation_id.rsplit(".", 1)[-1]) for row in rows] == [
        (128, "replica_a"),
        (128, "replica_b"),
    ]
    assert len({row.response_id for row in rows}) == 1


def test_v1_inventory_remains_the_original_jobs_matrix(amendment: dict) -> None:
    inventory = plan.build_inventory(amendment)

    assert [row.jobs for row in inventory.invocations[:4]] == [1, 4, 4, 4]
    assert inventory.invocations[0].invocation_id.endswith(".jobs1")


def test_successor_identity_binds_design_decision_and_absolute_artifact_root(
    amendment: dict,
) -> None:
    artifact_root = Path(
        "/mnt/7302share/fast_ysp/U7_GWAS/tasks/BM-NATIVE-QA/materialized/"
        "njobs128-v2"
    )
    frozen = identity.freeze_successor_identity(
        plan.build_successor_inventory(amendment, _successor()),
        fixture_manifest_sha256="7" * 64,
        amendment_sha256="a" * 64,
        successor_design_sha256="b" * 64,
        worker_decision_sha256="c" * 64,
        runner_test_sha256s={"runner.py": "d" * 64},
        artifact_root=artifact_root,
        run_namespace="qa_real80_njobs128_v2",
    )

    assert frozen.design_payload["successor_design_sha256"] == "b" * 64
    assert frozen.design_payload["worker_decision_sha256"] == "c" * 64
    assert frozen.design_payload["future_artifacts"]["root"] == str(artifact_root)
    assert all(
        row["jobs"] == 128 for row in frozen.design_payload["invocations"]
    )
    assert frozen.qa_design_hash != (
        "65ba6556d86db4d27b03169ff70d5b39796c5968ce8ece9f782e866995cf5d5e"
    )


def test_successor_plan_cli_freezes_v2_namespace_and_workers128(tmp_path: Path) -> None:
    out = tmp_path / "successor.json"

    assert cli.main(
        [
            "plan",
            "--amendment",
            str(AMENDMENT_PATH),
            "--context-evidence",
            str(CONTEXT_EVIDENCE_PATH),
            "--fixture-manifest",
            str(FIXTURE_MANIFEST_PATH),
            "--successor-design",
            str(SUCCESSOR_PATH),
            "--out",
            str(out),
        ]
    ) == 0

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["schema"] == "homoeogwas-bm-native-qa-prospective-inventory-v3"
    assert payload["source_bindings"]["successor_design_sha256"] == (
        "bf859b72f93818fda323239f0aef6b17e959e1c37760429b09fc9dfa65055e30"
    )
    assert payload["source_bindings"]["worker_decision_sha256"] == (
        "ade451d2eaef49530011d1558d042bbce3fc2c2d3f6e9cea78c2bcb244c4e0e0"
    )
    assert {row["jobs"] for row in payload["semantic_invocations"]} == {128}
    assert all(
        row["scenario_id"].startswith("qa_real80_njobs128_v2.")
        for row in payload["seed_derivations"]
    )
    assert payload["design_payload"]["future_artifacts"]["root"] == (
        "/mnt/7302share/fast_ysp/U7_GWAS/tasks/BM-NATIVE-QA/materialized/"
        "njobs128-v2"
    )
    assert out.read_bytes() == FROZEN_INVENTORY_PATH.read_bytes()


def test_successor_plan_cli_rejects_changed_design_bytes(tmp_path: Path) -> None:
    changed = yaml.safe_load(SUCCESSOR_PATH.read_text(encoding="utf-8"))
    changed["invocation_override"]["requested_jobs"] = 64
    changed_path = tmp_path / "changed.yaml"
    changed_path.write_text(yaml.safe_dump(changed), encoding="utf-8")
    out = tmp_path / "must-not-exist.json"

    assert cli.main(
        [
            "plan",
            "--amendment",
            str(AMENDMENT_PATH),
            "--context-evidence",
            str(CONTEXT_EVIDENCE_PATH),
            "--fixture-manifest",
            str(FIXTURE_MANIFEST_PATH),
            "--successor-design",
            str(changed_path),
            "--out",
            str(out),
        ]
    ) != 0
    assert not out.exists()


def test_rehydrated_identity_compares_canonical_json_not_container_types() -> None:
    payload = {"nested": {"values": ("A", "B")}}
    frozen = identity.FrozenIdentity(
        qa_design_hash=identity.sha256_payload(payload),
        design_payload=payload,
    )
    decoded = {"nested": {"values": ["A", "B"]}}

    cli._require_rehydrated_identity(
        frozen,
        qa_design_hash=frozen.qa_design_hash,
        inventory_payload=decoded,
    )

    decoded["nested"]["values"][1] = "changed"
    with pytest.raises(cli.CLIError, match="rehydrated successor identity"):
        cli._require_rehydrated_identity(
            frozen,
            qa_design_hash=frozen.qa_design_hash,
            inventory_payload=decoded,
        )
