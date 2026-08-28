from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from homoeogwas.run_registry import RegistryError, load_registry


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
