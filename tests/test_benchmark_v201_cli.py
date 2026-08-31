from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.benchmarks.v201 import cli
from scripts.benchmarks.v201.cli import main
from scripts.benchmarks.v201.contracts import Scenario, canonical_json
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


@pytest.mark.parametrize("cap", ["cpu_hours", "elapsed_hours", "output_gb"])
def test_project_formal_returns_nonzero_when_any_cap_is_exceeded(
    tmp_path, monkeypatch, cap,
):
    root = tmp_path / "benchmark"
    _sealed_minimal_root(root)
    monkeypatch.setattr(cli, "_validate_sealed_design", lambda *_a, **_k: None)
    monkeypatch.setattr(cli, "_pilot_measurements", lambda *_a, **_k: [{
        "scenario_id": "A.recovery.cotton.balanced",
        "cpu_seconds": 1.0, "wall_seconds": 3.0,
        "peak_rss_bytes": 1_000_000,
        "output_bytes": 1.0,
        "scenario_multiplier": 1.0,
    }])

    def over(*_args, **_kwargs):
        raise BudgetExceeded(f"{cap} exceeds formal budget", {
            "stage": "formal", "totals": {}, "limits": {}, "breakdown": {},
        })

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
    payload = json.loads(shard.read_text(encoding="utf-8"))
    assert payload["rows"] == [row]
    assert payload["cpu_seconds"] >= 0
    assert payload["wall_seconds"] >= 0
    assert payload["peak_rss_bytes"] > 0
    assert payload["replicate_output_bytes"] == 0
    assert payload["replicate_output_manifest"] == []


def _mini_init_scenarios(stage: str) -> list[Scenario]:
    qa = stage == "pilot"
    rows = [
        Scenario(
            "A.loco.cotton.pve_0", "fit", stage, 2,
            199 if qa else 0,
            {"panel": "cotton", "experiment": "loco", "scan_pve": 0.0,
             "qa_only": qa},
        ),
        Scenario(
            "A.recovery.wheat.balanced", "fit", stage, 1,
            199 if qa else 0,
            {"panel": "wheat", "experiment": "recovery",
             "allocation": "balanced", "qa_only": qa},
        ),
    ]
    for backbone in ("cotton", "wheat", "quartet"):
        rows.append(Scenario(
            f"B.end2end.{backbone}.gaussian", "omnib", stage, 1,
            199 if qa else 2_000,
            {"backbone": backbone, "experiment": "end2end",
             "null_model": "gaussian", "qa_only": qa},
        ))
        for family_size in (80, 500, 2_000):
            rows.append(Scenario(
                f"B.family_size.{backbone}.g{family_size}",
                "omnib", stage, 1, 199 if qa else 2_000,
                {"backbone": backbone, "experiment": "family_size",
                 "family_size": family_size, "qa_only": qa},
            ))
    return rows


def _write_real_init_spec(tmp_path: Path) -> Path:
    from bed_reader import to_bed

    from homoeogwas.io import plink_bim_sha256

    samples = [f"sample-{index:03d}" for index in range(72)]
    rng = np.random.default_rng(1902)
    geno_root = tmp_path / "geno"
    mappings: dict[str, str] = {}
    genes = np.asarray([f"g{index}" for index in range(2_000)], dtype=object)
    snp_idx = np.asarray([
        np.asarray([(index + offset) % 12 for offset in range(3)], dtype=int)
        for index in range(2_000)
    ], dtype=object)
    for label in ("A", "B", "C", "D"):
        prefix = geno_root / label / "all"
        prefix.parent.mkdir(parents=True)
        values = rng.integers(0, 3, size=(72, 12)).astype(np.float32)
        to_bed(str(prefix) + ".bed", values, properties={
            "fid": ["0"] * 72, "iid": samples,
            "sid": [f"{label}-{index}" for index in range(12)],
            "chromosome": [label] * 12, "bp_position": list(range(1, 13)),
            "allele_1": ["A"] * 12, "allele_2": ["C"] * 12,
        }, count_A1=True)
        mapping = tmp_path / f"mapping-{label}.npz"
        np.savez(
            mapping, gene_ids=genes, snp_idx=snp_idx,
            bim_sha256=np.asarray(plink_bim_sha256(prefix)),
            n_variants=np.asarray(12), subgenome=np.asarray(label),
        )
        mappings[label] = str(mapping.resolve())
    phenotype = tmp_path / "phenotype.tsv"
    phenotype.write_text(
        "sample\ttrait\n" + "".join(
            f"{sample}\t{rng.normal():.12g}\n" for sample in samples
        ),
        encoding="utf-8",
    )
    groups: dict[tuple[str, int], Path] = {}
    labels_by_backbone = {
        "cotton": ("A", "D"),
        "wheat": ("A", "B", "D"),
        "quartet": ("A", "B", "C", "D"),
    }
    for backbone, labels in labels_by_backbone.items():
        for family_size in (80, 500, 2_000):
            path = tmp_path / f"groups-{backbone}-{family_size}.tsv"
            header = "group_id\t" + "\t".join(f"gene_{label}" for label in labels)
            path.write_text(
                header + "\n" + "".join(
                    f"group_{index}\t" + "\t".join([f"g{index}"] * len(labels)) + "\n"
                    for index in range(family_size)
                ),
                encoding="utf-8",
            )
            groups[(backbone, family_size)] = path
    omnib = {}
    for backbone, labels in labels_by_backbone.items():
        for family_size in (None, 80, 500, 2_000):
            size = 80 if family_size is None else family_size
            key = backbone if family_size is None else f"{backbone}:g{family_size}"
            omnib[key] = {
                "subgenomes": list(labels),
                "bed_prefixes": {
                    label: str((geno_root / label / "all").resolve())
                    for label in labels
                },
                "snp_to_gene": {label: mappings[label] for label in labels},
                "groups": str(groups[(backbone, size)].resolve()),
                "phenotype": str(phenotype.resolve()),
                "sample_col": "sample", "trait": "trait",
            }
    registry = tmp_path / "application-registry.yaml"
    registry.write_text("runs: []\n", encoding="utf-8")
    spec = tmp_path / "input-spec.json"
    spec.write_text(json.dumps({
        "schema": "homoeogwas-v201-benchmark-inputs-v1",
        "application_registry": str(registry.resolve()),
        "fit": {
            "cotton": {
                "subgenomes": ["A", "D"],
                "bed_prefix_template": str((geno_root / "{subgenome}" / "all").absolute()),
                "phenotype": str(phenotype.resolve()),
                "sample_col": "sample", "trait": "trait",
            },
            "wheat": {
                "subgenomes": ["A", "B", "D"],
                "bed_prefix_template": str((geno_root / "{subgenome}" / "all").absolute()),
                "phenotype": str(phenotype.resolve()),
                "sample_col": "sample", "trait": "trait",
            },
        },
        "omnib": omnib,
    }, sort_keys=True) + "\n", encoding="utf-8")
    return spec


def test_real_twelve_context_main_init_preserves_and_validates_both_loco_stages(
    tmp_path, monkeypatch,
):
    from scripts.benchmarks.v201 import aggregate

    spec = _write_real_init_spec(tmp_path)
    root = tmp_path / "benchmark"
    monkeypatch.setattr(cli, "build_scenarios", _mini_init_scenarios)
    monkeypatch.setattr(aggregate, "build_scenarios", _mini_init_scenarios)
    strict_calls: list[tuple[str, str]] = []
    validate_calls: list[str] = []
    original_strict = cli.validate_real_omnib_context
    original_run = cli.run_command

    def strict_spy(path, **kwargs):
        strict_calls.append((str(Path(path).relative_to(root)), str(kwargs["stage"])))
        return original_strict(path, **kwargs)

    def command_spy(argv):
        if tuple(argv[:2]) == ("homoeogwas", "validate"):
            validate_calls.append(str(Path(argv[-1]).relative_to(root)))
        return original_run(argv)

    monkeypatch.setattr(cli, "validate_real_omnib_context", strict_spy)
    monkeypatch.setattr(cli, "run_command", command_spy)

    assert main(["init", "--root", str(root), "--input-spec", str(spec)]) == 0
    lock = json.loads((root / "design_lock.json").read_text(encoding="utf-8"))
    artifacts = lock["loco_truth_artifacts"]
    assert set(artifacts) == {"pilot", "formal"}
    pilot = artifacts["pilot"]["A.loco.cotton.pve_0"]
    formal = artifacts["formal"]["A.loco.cotton.pve_0"]
    assert set(pilot) == set(formal) == {"0", "1"}
    for replicate in ("0", "1"):
        assert pilot[replicate]["path"] != formal[replicate]["path"]
        assert pilot[replicate]["seed"] != formal[replicate]["seed"]
        assert pilot[replicate]["sha256"] != formal[replicate]["sha256"]
    assert len(strict_calls) == 24
    assert len(strict_calls) == len(set(strict_calls))
    assert {stage for _path, stage in strict_calls} == {"pilot", "formal"}
    assert len(validate_calls) == len(lock["config_hashes"])
    assert len(validate_calls) == len(set(validate_calls))
    assert set(validate_calls) == set(lock["config_hashes"])


@pytest.mark.parametrize("corruption", ["missing", "extra", "cross_stage"])
def test_stage_binding_validation_rejects_incomplete_or_cross_stage_maps(corruption):
    scenarios = _mini_init_scenarios("pilot")
    bindings = {
        row.scenario_id: f"configs/interact/pilot/{index}.yaml"
        for index, row in enumerate(scenarios)
    }
    hashes = {path: "a" * 64 for path in bindings.values()}
    if corruption == "missing":
        bindings.pop(next(iter(bindings)))
    elif corruption == "extra":
        bindings["foreign"] = "configs/interact/pilot/foreign.yaml"
        hashes[bindings["foreign"]] = "b" * 64
    else:
        scenario_id = next(iter(bindings))
        bindings[scenario_id] = "configs/interact/formal/cross.yaml"
        hashes[bindings[scenario_id]] = "c" * 64

    with pytest.raises(cli.CLIError, match="coverage|cross-stage"):
        cli._validate_stage_bindings("pilot", scenarios, bindings, hashes)


def _write_measured_shard(root: Path, *, cpu: float = 2.0, wall: float = 9.0) -> Path:
    output_root = root / "work/fit/pilot/A.recovery.cotton.balanced/replicate-000000"
    output_root.mkdir(parents=True)
    side = output_root / "results/large.bin"
    side.parent.mkdir()
    side.write_bytes(b"x" * 10_000)
    relative = side.relative_to(root).as_posix()
    shard = root / "pilot/fit/A.recovery.cotton.balanced/replicate-000000.json"
    shard.parent.mkdir(parents=True)
    shard.write_text(canonical_json({
        "track": "fit", "scenario_id": "A.recovery.cotton.balanced",
        "replicate": 0, "design_hash": "d" * 64, "stage": "pilot",
        "cpu_seconds": cpu, "wall_seconds": wall, "peak_rss_bytes": 123_000_000,
        "replicate_output_root": output_root.relative_to(root).as_posix(),
        "replicate_output_bytes": side.stat().st_size,
        "replicate_output_manifest": [{
            "path": relative, "size": side.stat().st_size,
            "sha256": hashlib.sha256(side.read_bytes()).hexdigest(),
        }],
    }) + "\n", encoding="utf-8")
    return shard


def test_project_measurements_require_cpu_and_count_recursive_side_outputs(
    tmp_path, monkeypatch,
):
    root = tmp_path / "benchmark"
    shard = _write_measured_shard(root, cpu=2.0, wall=9.0)
    pilot = Scenario(
        "A.recovery.cotton.balanced", "fit", "pilot", 1, 199,
        {"panel": "cotton", "experiment": "recovery"},
    )
    formal = Scenario(
        pilot.scenario_id, "fit", "formal", 5, 0,
        {"panel": "cotton", "experiment": "recovery"},
    )
    monkeypatch.setattr(
        cli, "build_scenarios",
        lambda stage: [pilot] if stage == "pilot" else [formal],
    )

    measurement = cli._pilot_measurements(root)[0]
    assert measurement["cpu_seconds"] == 2.0
    assert measurement["wall_seconds"] == 9.0
    assert measurement["peak_rss_bytes"] == 123_000_000
    assert measurement["output_bytes"] == shard.stat().st_size + 10_000

    payload = json.loads(shard.read_text(encoding="utf-8"))
    payload.pop("cpu_seconds")
    shard.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    with pytest.raises(cli.CLIError, match="cpu_seconds"):
        cli._pilot_measurements(root)


def test_project_formal_reports_observed_and_projected_peak_memory(
    tmp_path, monkeypatch, capsys,
):
    root = tmp_path / "benchmark"
    _sealed_minimal_root(root)
    measurements = [{
        "scenario_id": "s", "cpu_seconds": 4.0, "wall_seconds": 8.0,
        "peak_rss_bytes": 2_000_000_000, "output_bytes": 100.0,
        "scenario_multiplier": 3.0,
    }]
    monkeypatch.setattr(cli, "_validate_sealed_design", lambda *_a, **_k: None)
    monkeypatch.setattr(cli, "_pilot_measurements", lambda *_a, **_k: measurements)
    monkeypatch.setattr(cli, "project_budget", lambda *_a, **_k: {
        "stage": "formal", "totals": {}, "limits": {}, "breakdown": {},
    })

    assert main([
        "project-formal", "--root", str(root), "--effective-workers", "3",
    ]) == 0
    projected = json.loads(capsys.readouterr().out)
    assert projected["peak_memory_projection"] == {
        "observed_max_peak_rss_bytes": 2_000_000_000,
        "observed_max_peak_memory_gb": 2.0,
        "projected_peak_memory_bytes": 6_000_000_000,
        "projected_peak_memory_gb": 6.0,
        "cap_gb": None,
        "threshold_status": "not_evaluated_no_formal_memory_cap",
    }


def test_execution_measurement_keeps_process_cpu_distinct_from_wall(monkeypatch):
    snapshots = iter([(10.0, 100), (12.0, 150)])
    clocks = iter([20.0, 29.0])
    monkeypatch.setattr(cli, "_resource_snapshot", lambda: next(snapshots))
    monkeypatch.setattr(cli.time, "perf_counter", lambda: next(clocks))

    result, measurement = cli._measure_operation(lambda: "done")

    assert result == "done"
    assert measurement == {
        "cpu_seconds": 2.0, "wall_seconds": 9.0, "peak_rss_bytes": 150,
    }


def test_project_formal_fails_when_wall_based_elapsed_projection_exceeds_cap(
    tmp_path, monkeypatch, capsys,
):
    root = tmp_path / "benchmark"
    _sealed_minimal_root(root)
    measurements = [{
        "scenario_id": "s", "cpu_seconds": 1.0, "wall_seconds": 7_200.0,
        "peak_rss_bytes": 1_000_000, "output_bytes": 1.0,
        "scenario_multiplier": 2.0,
    }]
    monkeypatch.setattr(cli, "_validate_sealed_design", lambda *_a, **_k: None)
    monkeypatch.setattr(cli, "_pilot_measurements", lambda *_a, **_k: measurements)
    monkeypatch.setattr(cli, "project_budget", lambda *_a, **_k: {
        "stage": "formal", "totals": {"cpu_hours": 1.0},
        "limits": {"elapsed_hours": 1.0}, "breakdown": {},
    })

    assert main([
        "project-formal", "--root", str(root), "--effective-workers", "2",
    ]) == 2
    projected = json.loads(capsys.readouterr().out)
    assert projected["totals"]["elapsed_hours"] == 2.0
