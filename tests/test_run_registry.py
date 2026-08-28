from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from homoeogwas.cli import main
from homoeogwas.run_registry import (
    RegistryError,
    execute_registry,
    load_registry,
    load_run_state,
    resume_decision,
    run_identity_sha256,
    write_run_state,
)


def _interaction(run_id: str, subgenomes=("A", "C")) -> dict:
    labels = list(subgenomes)
    return {
        "id": run_id,
        "kind": "interaction",
        "species": "Brassica napus",
        "panel": "panel-1",
        "subgenomes": labels,
        "phenotype": "phenotype.tsv",
        "sample_col": "sample",
        "trait": "flowering_time",
        "out_dir": f"results/{run_id}",
        "bed_prefixes": {label: f"geno/{label}/all" for label in labels},
        "snp_to_gene": {label: f"maps/{label}.npz" for label in labels},
        "groups": "groups.tsv",
        "hypothesis_unit": "edge" if len(labels) == 2 else "group",
        "family_scope": "primary_only",
        "bootstrap_B": 2000,
        "n_jobs": 4,
    }


def _write_registry(tmp_path: Path, runs: list[dict]) -> Path:
    path = tmp_path / "registry.yaml"
    path.write_text(yaml.safe_dump({
        "registry_version": 1,
        "name": "test-registry",
        "index_dir": "registry-index",
        "runs": runs,
    }, sort_keys=False))
    return path


def test_load_registry_resolves_paths_and_keeps_species_as_metadata(tmp_path):
    registry = load_registry(_write_registry(tmp_path, [_interaction("r1")]))

    run = registry.runs[0]
    assert run.species == "Brassica napus"
    assert run.groups == (tmp_path / "groups.tsv").resolve()
    assert run.subgenomes == ("A", "C")
    assert run.bed_prefixes["A"] == (tmp_path / "geno/A/all").resolve()
    assert registry.index_dir == (tmp_path / "registry-index").resolve()


@pytest.mark.parametrize("subgenomes", [("A",), ("A", "A")])
def test_interaction_requires_two_unique_subgenomes(tmp_path, subgenomes):
    path = _write_registry(tmp_path, [_interaction("bad", subgenomes)])

    with pytest.raises(RegistryError, match="at least two unique"):
        load_registry(path)


def test_registry_rejects_duplicate_ids(tmp_path):
    run = _interaction("same")

    with pytest.raises(RegistryError, match="duplicate run id"):
        load_registry(_write_registry(tmp_path, [run, run]))


def test_registry_rejects_unknown_run_fields(tmp_path):
    run = _interaction("bad") | {"statitic": "omniB"}

    with pytest.raises(RegistryError, match="unsupported fields.*statitic"):
        load_registry(_write_registry(tmp_path, [run]))


def test_historical_run_requires_analysis_shape_and_result_root(tmp_path):
    run = {
        "id": "old",
        "kind": "historical",
        "species": "Triticum aestivum",
        "panel": "Watkins",
        "subgenomes": ["A", "B", "D"],
    }

    with pytest.raises(RegistryError, match="result_root.*analysis_shape"):
        load_registry(_write_registry(tmp_path, [run]))


def _materialized_run(tmp_path: Path):
    for label in ("A", "C"):
        prefix = tmp_path / f"geno/{label}/all"
        prefix.parent.mkdir(parents=True, exist_ok=True)
        Path(str(prefix) + ".bed").write_bytes(b"BED")
        Path(str(prefix) + ".bim").write_text(f"1\trs{label}\t0\t1\tA\tG\n")
        Path(str(prefix) + ".fam").write_text("F S1 0 0 0 -9\n")
        mapping = tmp_path / f"maps/{label}.npz"
        mapping.parent.mkdir(parents=True, exist_ok=True)
        mapping.write_bytes(f"map-{label}".encode())
    (tmp_path / "phenotype.tsv").write_text("sample\tflowering_time\nS1\t10\n")
    (tmp_path / "groups.tsv").write_text("group_id\tgene_A\tgene_C\ng1\ta\tc\n")
    return load_registry(_write_registry(tmp_path, [_interaction("r1")])).runs[0]


def test_identity_is_mapping_order_stable_and_changes_with_bim(tmp_path):
    run = _materialized_run(tmp_path)
    first = run_identity_sha256(run)
    reversed_mapping = dict(reversed(list(run.bed_prefixes.items())))

    assert run_identity_sha256(replace(run, bed_prefixes=reversed_mapping)) == first
    Path(str(run.bed_prefixes["A"]) + ".bim").write_text(
        "1\trsA\t0\t2\tA\tG\n")
    assert run_identity_sha256(run) != first


def test_resume_skips_only_matching_complete_identity(tmp_path):
    run = _materialized_run(tmp_path)
    digest = run_identity_sha256(run)

    assert resume_decision(
        run, {"status": "COMPLETE", "identity": digest}, resume=True) == "SKIP"
    with pytest.raises(RegistryError, match="identity mismatch"):
        resume_decision(
            run, {"status": "COMPLETE", "identity": "0" * 64}, resume=True)


def test_no_resume_rejects_existing_state(tmp_path):
    run = _materialized_run(tmp_path)

    with pytest.raises(RegistryError, match="resume is disabled"):
        resume_decision(
            run, {"status": "FAILED", "identity": run_identity_sha256(run)},
            resume=False)


def test_run_state_roundtrip(tmp_path):
    out = tmp_path / "out"
    written = write_run_state(out, {"status": "RUNNING", "identity": "abc"})

    assert written == out / "registry_run.json"
    assert load_run_state(out) == {"status": "RUNNING", "identity": "abc"}


def _materialize_registry_inputs(tmp_path: Path, subgenomes=("A", "C")) -> Path:
    for label in subgenomes:
        prefix = tmp_path / f"geno/{label}/all"
        prefix.parent.mkdir(parents=True, exist_ok=True)
        Path(str(prefix) + ".bed").write_bytes(b"BED")
        Path(str(prefix) + ".bim").write_text(f"1\trs{label}\t0\t1\tA\tG\n")
        Path(str(prefix) + ".fam").write_text("F S1 0 0 0 -9\n")
        mapping = tmp_path / f"maps/{label}.npz"
        mapping.parent.mkdir(parents=True, exist_ok=True)
        mapping.write_bytes(f"map-{label}".encode())
    (tmp_path / "phenotype.tsv").write_text("sample\tflowering_time\nS1\t10\n")
    header = "group_id\t" + "\t".join(f"gene_{s}" for s in subgenomes) + "\n"
    row = "g1\t" + "\t".join(f"g{s}" for s in subgenomes) + "\n"
    (tmp_path / "groups.tsv").write_text(header + row)
    return _write_registry(tmp_path, [_interaction("r1", subgenomes)])


def test_interaction_dispatch_forces_canonical_group_omnib(tmp_path):
    path = _materialize_registry_inputs(tmp_path, ("A", "B", "D"))
    calls = []

    result = execute_registry(
        path, dry_run=True,
        interaction_runner=lambda **kwargs: calls.append(kwargs) or {
            "ok": True, "summary": {"n_significant": 0}})

    assert result["ok"]
    assert calls[0]["statistic"] == "omniB"
    assert calls[0]["groups"].endswith("groups.tsv")
    assert calls[0]["hypothesis_unit"] == "group"
    assert calls[0]["subset_order"] == 2
    assert calls[0]["dry_run"] is True


def test_four_copy_dispatch_has_no_four_way_option(tmp_path):
    path = _materialize_registry_inputs(tmp_path, ("A", "B", "C", "D"))
    calls = []

    execute_registry(
        path, dry_run=True,
        interaction_runner=lambda **kwargs: calls.append(kwargs) or {
            "ok": True, "summary": {}})

    assert set(calls[0]) >= {"groups", "family_scope", "subset_order"}
    assert not any("four" in key or "4way" in key for key in calls[0])


def test_historical_entry_is_indexed_without_dispatch(tmp_path):
    root = tmp_path / "historical-result"
    root.mkdir()
    run = {
        "id": "old",
        "kind": "historical",
        "species": "Triticum aestivum",
        "panel": "Watkins",
        "subgenomes": ["A", "B", "D"],
        "result_root": "historical-result",
        "analysis_shape": "legacy_route_b_group",
    }
    path = _write_registry(tmp_path, [run])

    result = execute_registry(
        path,
        interaction_runner=lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("historical entry was dispatched")))

    assert result["runs"][0]["status"] == "HISTORICAL"
    assert Path(result["indexes"]["json"]).exists()
    assert Path(result["indexes"]["tsv"]).exists()
    assert Path(result["indexes"]["markdown"]).exists()


def test_matching_complete_run_is_skipped_on_resume(tmp_path):
    path = _materialize_registry_inputs(tmp_path)
    run = load_registry(path).runs[0]
    write_run_state(run.out_dir, {
        "run_id": run.id,
        "status": "COMPLETE",
        "identity": run_identity_sha256(run),
        "summary": {"n_significant": 2},
    })

    result = execute_registry(
        path, resume=True,
        interaction_runner=lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("complete run was dispatched")))

    assert result["runs"][0]["status"] == "SKIPPED_COMPLETE"
    assert result["runs"][0]["n_significant"] == 2


def test_registry_continues_after_independent_failure(tmp_path):
    first = _interaction("first")
    second = _interaction("second")
    _materialized_run(tmp_path)
    path = _write_registry(tmp_path, [first, second])
    calls = []

    def runner(**kwargs):
        calls.append(kwargs["out_dir"])
        return {"ok": len(calls) == 2, "reason": "first failed", "summary": {}}

    result = execute_registry(path, interaction_runner=runner)

    assert [run["status"] for run in result["runs"]] == ["FAILED", "COMPLETE"]
    assert result["ok"] is False


def test_registry_cli_validate_and_dry_run(tmp_path, capsys):
    path = _materialize_registry_inputs(tmp_path)

    assert main(["registry", "validate", "-c", str(path)]) == 0
    assert "registry schema OK" in capsys.readouterr().out
    assert main(["registry", "run", "-c", str(path), "--dry-run"]) == 0
    assert "DRY_RUN" in capsys.readouterr().out
