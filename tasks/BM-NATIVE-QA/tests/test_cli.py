from __future__ import annotations

import builtins
import json
from pathlib import Path

import yaml
from bm_native_qa_harness import cli

MANAGEMENT_ROOT = Path("/mnt/7302share/fast_ysp/U7_GWAS")
AMENDMENT = (
    MANAGEMENT_ROOT
    / "tasks/BM-NATIVE-QA/QA-EXECUTION-AMENDMENT-v1-20260910.yaml"
)
CONTEXT_EVIDENCE = (
    MANAGEMENT_ROOT
    / "tasks/BM-INPUTS/staging/real-core-v4-primary80-estimability/manifest.json"
)
FIXTURE_MANIFEST = (
    MANAGEMENT_ROOT / "tasks/BM-FIXTURE-V3/manifest.20260910-v5-r3.json"
)


def _plan_args(out: Path) -> list[str]:
    return [
        "plan",
        "--amendment",
        str(AMENDMENT),
        "--context-evidence",
        str(CONTEXT_EVIDENCE),
        "--fixture-manifest",
        str(FIXTURE_MANIFEST),
        "--out",
        str(out),
    ]


def test_plan_cli_writes_stable_prospective_inventory_only(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"

    assert cli.main(_plan_args(first)) == 0
    assert cli.main(_plan_args(second)) == 0

    first_payload = json.loads(first.read_text(encoding="utf-8"))
    second_payload = json.loads(second.read_text(encoding="utf-8"))
    assert first_payload == second_payload
    assert first_payload["counts"] == {
        "anchors": 6,
        "contexts": 6,
        "responses": 12,
        "invocations": 16,
    }
    assert first_payload["response_materialization_authorized"] is False
    assert first_payload["execution_authorized"] is False
    assert len(first_payload["qa_design_hash"]) == 64
    assert len(first_payload["seed_derivations"]) == 30
    contexts = first_payload["design_payload"]["contexts"]
    cotton_hashes = dict(contexts[0]["input_file_sha256s"])
    wheat_hashes = dict(contexts[3]["input_file_sha256s"])
    assert len(cotton_hashes) == 10
    assert len(wheat_hashes) == 14
    assert cotton_hashes[
        "/mnt/7302share/fast_ysp/U7_GWAS/results/viz_preview/"
        "fig3_cotton_locus/rescue_cgvd1245/A_cgvd.bed"
    ] == "01aa0d289c8641e4abda53319396fcd0035c6e98fcbb837e0a5ecc52293a1294"
    encoded = first.read_text(encoding="utf-8")
    assert "seed_value" not in encoded
    assert "generated_array" not in encoded
    assert "npy_sha256" not in encoded
    assert not list(tmp_path.rglob("*.npy"))
    assert not list(tmp_path.rglob("*.tsv"))
    assert not list(tmp_path.rglob("checkpoint*"))
    assert not list(tmp_path.rglob("result*"))


def test_plan_cli_refuses_to_overwrite_existing_inventory(tmp_path: Path) -> None:
    out = tmp_path / "inventory.json"
    out.write_text("preserve\n", encoding="utf-8")

    assert cli.main(_plan_args(out)) != 0
    assert out.read_text(encoding="utf-8") == "preserve\n"


def test_plan_cli_rejects_unaccepted_amendment_status(tmp_path: Path) -> None:
    amendment = yaml.safe_load(AMENDMENT.read_text(encoding="utf-8"))
    amendment["status"] = "pending_review"
    candidate = tmp_path / "candidate.yaml"
    candidate.write_text(yaml.safe_dump(amendment), encoding="utf-8")
    out = tmp_path / "inventory.json"
    args = _plan_args(out)
    args[args.index(str(AMENDMENT))] = str(candidate)

    assert cli.main(args) != 0
    assert not out.exists()


def test_closed_materialize_and_run_stop_before_numerical_or_subprocess_imports(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    imported = []
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "subprocess" or name == "numpy" or name.startswith(
            "scripts.benchmarks"
        ):
            imported.append(name)
            raise AssertionError(f"blocked command imported {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    for command, flag in (
        ("materialize", "response_materialization_authorized=false"),
        ("run", "execution_authorized=false"),
    ):
        out = tmp_path / command
        status = cli.main(
            [command, "--amendment", str(AMENDMENT), "--out", str(out)]
        )
        assert status != 0
        assert flag in capsys.readouterr().err
        assert not out.exists()

    assert imported == []


def test_open_flag_still_requires_a_separately_reviewed_activation_adapter(
    tmp_path: Path,
    capsys,
) -> None:
    amendment = yaml.safe_load(AMENDMENT.read_text(encoding="utf-8"))
    for command, flag in (
        ("materialize", "response_materialization_authorized"),
        ("run", "execution_authorized"),
    ):
        amendment[flag] = True
        candidate = tmp_path / f"{command}.yaml"
        candidate.write_text(yaml.safe_dump(amendment), encoding="utf-8")
        out = tmp_path / command

        assert cli.main(
            [command, "--amendment", str(candidate), "--out", str(out)]
        ) != 0
        assert "separately reviewed activation adapter" in capsys.readouterr().err
        assert not out.exists()
        amendment[flag] = False


def test_runner_hash_inventory_includes_nested_python_sources(tmp_path: Path) -> None:
    task_root = tmp_path / "task"
    package = task_root / "bm_native_qa_harness" / "nested"
    tests = task_root / "tests" / "nested"
    package.mkdir(parents=True)
    tests.mkdir(parents=True)
    (package / "worker.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tests / "test_worker.py").write_text("def test_value(): pass\n", encoding="utf-8")

    hashes = cli._runner_test_sha256s(task_root)

    assert set(hashes) == {
        "bm_native_qa_harness/nested/worker.py",
        "tests/nested/test_worker.py",
    }
