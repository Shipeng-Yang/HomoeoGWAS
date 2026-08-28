from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from homoeogwas.run_registry import (
    RegistryError,
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
