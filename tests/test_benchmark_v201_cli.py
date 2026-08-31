from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.benchmarks.v201 import cli
from scripts.benchmarks.v201.cli import main
from scripts.benchmarks.v201.contracts import canonical_json
from scripts.benchmarks.v201.shards import BudgetExceeded


def test_parser_exposes_only_safe_five_commands():
    parser = cli.build_parser()
    choices = parser._subparsers._group_actions[0].choices
    assert set(choices) == {"init", "pilot", "aggregate", "audit", "project-formal"}
    with pytest.raises(SystemExit) as error:
        parser.parse_args(["formal", "--root", "/tmp/forbidden"])
    assert error.value.code == 2


def test_init_refuses_nonempty_root_without_touching_foreign_file(tmp_path):
    root = tmp_path / "benchmark"
    root.mkdir()
    foreign = root / "foreign.txt"
    foreign.write_text("keep", encoding="utf-8")
    assert main(["init", "--root", str(root), "--input-spec", str(tmp_path / "x.json")]) == 2
    assert foreign.read_text(encoding="utf-8") == "keep"
    assert not (root / "design_lock.json").exists()


@pytest.mark.parametrize("symlink_at_root", [False, True])
def test_init_refuses_a_symlinked_root_or_parent(tmp_path, symlink_at_root):
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real_parent, target_is_directory=True)
    root = link if symlink_at_root else link / "benchmark"

    assert main([
        "init", "--root", str(root),
        "--input-spec", str(tmp_path / "missing.json"),
    ]) == 2
    assert list(real_parent.iterdir()) == []


def test_init_failure_never_publishes_half_design_lock(tmp_path, monkeypatch):
    root = tmp_path / "benchmark"
    spec = tmp_path / "inputs.json"
    spec.write_text("{}\n", encoding="utf-8")

    def fail(*_args, **_kwargs):
        raise ValueError("missing real cotton inputs")

    monkeypatch.setattr(cli, "_prepare_design", fail)
    assert main(["init", "--root", str(root), "--input-spec", str(spec)]) == 2
    assert not root.exists()


def _sealed_minimal_root(root: Path) -> None:
    root.mkdir()
    (root / "design_lock.json").write_text(
        canonical_json({"schema": "fixture", "design_hash": "d" * 64}) + "\n",
        encoding="utf-8",
    )


def test_pilot_validate_only_revalidates_without_writing_shards(tmp_path, monkeypatch):
    root = tmp_path / "benchmark"
    _sealed_minimal_root(root)
    calls: list[str] = []
    monkeypatch.setattr(cli, "_validate_sealed_design", lambda *_a, **_k: calls.append("strict"))
    monkeypatch.setattr(cli, "_dispatch_pilot", lambda *_a, **_k: pytest.fail("statistics ran"))

    assert main(["pilot", "--root", str(root), "--all", "--validate-only"]) == 0
    assert calls == ["strict"]
    assert not (root / "pilot").exists()


def test_pilot_dispatch_receives_only_qa_b199_scenarios(tmp_path, monkeypatch):
    root = tmp_path / "benchmark"
    _sealed_minimal_root(root)
    observed = []
    monkeypatch.setattr(cli, "_validate_sealed_design", lambda *_a, **_k: None)
    monkeypatch.setattr(cli, "_dispatch_pilot", lambda _root, scenarios, **_k: observed.extend(scenarios))

    assert main([
        "pilot", "--root", str(root),
        "--scenario", "B.end2end.cotton.gaussian",
    ]) == 0
    assert len(observed) == 1
    assert observed[0].bootstrap_B == 199
    assert observed[0].parameters["qa_only"] is True


def test_generated_pilot_interaction_config_is_noninferential_b199(tmp_path):
    groups = tmp_path / "groups.tsv"
    phenotype = tmp_path / "phenotype.tsv"
    mapping_a = tmp_path / "A.npz"
    mapping_d = tmp_path / "D.npz"
    for path in (groups, phenotype, mapping_a, mapping_d):
        path.write_bytes(b"fixture")
    config = cli._interact_config(
        {
            "subgenomes": ["A", "D"],
            "bed_prefixes": {"A": str(tmp_path / "A/all"), "D": str(tmp_path / "D/all")},
            "snp_to_gene": {"A": str(mapping_a), "D": str(mapping_d)},
            "groups": str(groups), "phenotype": str(phenotype),
            "sample_col": "sample", "trait": "trait",
        },
        "pilot", tmp_path / "results",
    )

    assert config["interact"]["calibration"] == {
        "method": "bootstrap", "B": 199, "seed": 2026, "qa_only": True,
    }


def test_aggregate_and_audit_delegate_to_approved_task9_functions(tmp_path, monkeypatch):
    root = tmp_path / "benchmark"
    _sealed_minimal_root(root)
    calls = []
    monkeypatch.setattr(cli, "_validate_sealed_design", lambda *_a, **_k: calls.append("validate"))
    monkeypatch.setattr(cli, "aggregate_benchmark", lambda path: calls.append(("aggregate", Path(path))))
    monkeypatch.setattr(cli, "audit_benchmark", lambda path: calls.append(("audit", Path(path))))

    assert main(["aggregate", "--root", str(root), "--stage", "pilot"]) == 0
    assert main(["audit", "--root", str(root), "--stage", "pilot"]) == 0
    assert calls == [
        "validate", ("aggregate", root.resolve()),
        "validate", ("audit", root.resolve()),
    ]


def test_project_formal_returns_nonzero_when_any_cap_is_exceeded(tmp_path, monkeypatch):
    root = tmp_path / "benchmark"
    _sealed_minimal_root(root)
    monkeypatch.setattr(cli, "_validate_sealed_design", lambda *_a, **_k: None)
    monkeypatch.setattr(cli, "_pilot_measurements", lambda *_a, **_k: [{
        "scenario_id": "A.recovery.cotton.balanced",
        "cpu_seconds": 1.0,
        "output_bytes": 1.0,
        "scenario_multiplier": 1.0,
    }])

    def over(*_args, **_kwargs):
        raise BudgetExceeded("cpu_hours exceeds formal budget", {"stage": "formal"})

    monkeypatch.setattr(cli, "project_budget", over)
    assert main(["project-formal", "--root", str(root), "--effective-workers", "1"]) == 2


def test_formal_projection_multiplier_uses_registry_work_not_shard_count():
    pilot = {
        row.scenario_id: row for row in cli.build_scenarios("pilot")
    }
    formal = {
        row.scenario_id: row for row in cli.build_scenarios("formal")
    }

    conditional = "B.conditional.cotton.gaussian.calibration"
    end_to_end = "B.end2end.cotton.gaussian"
    assert cli._formal_multiplier(pilot[conditional], formal[conditional]) == 100.0
    assert cli._formal_multiplier(pilot[end_to_end], formal[end_to_end]) == pytest.approx(
        (500 * 2_000) / (20 * 199)
    )


def test_init_validates_before_sealing_and_does_not_copy_external_inputs(
    tmp_path, monkeypatch,
):
    root = tmp_path / "benchmark"
    spec = tmp_path / "input-spec.json"
    external = tmp_path / "external.bed"
    external.write_bytes(b"external bytes")
    spec.write_text(json.dumps({"external": str(external.resolve())}) + "\n")
    order = []

    def prepare(staging, _spec):
        order.append("generate")
        (staging / "inputs").mkdir(parents=True)
        (staging / "configs").mkdir()
        return {"external": external}

    monkeypatch.setattr(cli, "_prepare_design", prepare)
    monkeypatch.setattr(cli, "_validate_presealed_design", lambda *_a, **_k: order.append("strict"))
    monkeypatch.setattr(cli, "_seal_design", lambda staging, *_a, **_k: (
        order.append("seal"),
        (staging / "design_lock.json").write_text("{}\n", encoding="utf-8"),
    ))

    assert main(["init", "--root", str(root), "--input-spec", str(spec)]) == 0
    assert order == ["generate", "strict", "seal"]
    assert external.read_bytes() == b"external bytes"
    assert not any(path.name == external.name for path in root.rglob("*"))


def test_application_dispatch_accepts_the_auditors_frozen_species_aliases(
    tmp_path, monkeypatch,
):
    root = tmp_path / "benchmark"
    root.mkdir()
    registry = tmp_path / "application-registry.json"
    registry.write_text("{}\n", encoding="utf-8")
    (root / "design_lock.json").write_text(
        canonical_json({"application_registry": str(registry)}) + "\n",
        encoding="utf-8",
    )
    scenario = next(
        item for item in cli.build_scenarios(stage="pilot")
        if item.scenario_id == "D.rapeseed"
    )
    row = {"species": "Brassica napus", "analysis_id": "rapeseed-1"}
    monkeypatch.setattr(cli, "export_application_rows", lambda _path: [row])

    cli._run_application_scenario(root, scenario, "d" * 64)

    shard = root / "pilot/application/D.rapeseed/replicate-000000.json"
    assert json.loads(shard.read_text(encoding="utf-8"))["rows"] == [row]
