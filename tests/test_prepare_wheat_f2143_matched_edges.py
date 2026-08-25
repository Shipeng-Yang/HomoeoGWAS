"""Behavior tests for the frozen matched-wheat F2143 preparation bundle."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from homoeogwas.interact import validate_interact_config
from homoeogwas.io import plink_bim_sha256

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "prepare_wheat_f2143_matched_edges.py"
)
SPEC = importlib.util.spec_from_file_location(
    "prepare_wheat_f2143_matched_edges", SCRIPT)
PREP = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = PREP
SPEC.loader.exec_module(PREP)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_sources(
    root: Path,
    *,
    rows: list[tuple[str, str, str, str]] | None = None,
    samples: list[str] | None = None,
    id_column: str = "group_id",
) -> tuple[Path, Path]:
    rows = rows or [
        ("g1", "a1", "b1", "d1"),
        ("g2", "a2", "b2", "d2"),
    ]
    samples = samples or ["s01", "s02"]
    groups = root / "source.f2143.tsv"
    groups.write_text(
        f"{id_column}\tgene_A\tgene_B\tgene_D\n"
        + "".join("\t".join(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    phenotype = root / "source.phenotype.tsv"
    phenotype.write_text(
        "sample\tdays_to_emerg_env_adjusted\n"
        + "".join(f"{sample}\t40.0\n" for sample in samples),
        encoding="utf-8",
    )
    return groups, phenotype


def _write_resources(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    samples: list[str],
) -> tuple[dict[str, str], dict[str, str]]:
    from bed_reader import to_bed

    prefixes: dict[str, str] = {}
    mappings: dict[str, str] = {}
    dosage = np.resize(np.asarray([[0, 1, 2], [2, 1, 0]], dtype=np.float32),
                       (len(samples), 3))
    for subgenome in ("A", "B", "D"):
        prefix = root / "resources" / f"gw_{subgenome}_flank"
        prefix.parent.mkdir(parents=True, exist_ok=True)
        to_bed(
            str(prefix) + ".bed",
            dosage,
            properties={
                "fid": ["0"] * len(samples),
                "iid": samples,
                "sid": [f"{subgenome}_v{i}" for i in range(3)],
                "chromosome": [subgenome] * 3,
                "bp_position": [1, 2, 3],
                "allele_1": ["A"] * 3,
                "allele_2": ["B"] * 3,
            },
            count_A1=True,
        )
        snp_idx = np.empty(1, dtype=object)
        snp_idx[0] = np.asarray([0, 1, 2], dtype=np.int64)
        mapping = root / "resources" / f"snp_to_gene_{subgenome}.npz"
        np.savez(
            mapping,
            gene_ids=np.asarray([f"g{subgenome}1"], dtype=object),
            snp_idx=snp_idx,
            bim_sha256=np.asarray(plink_bim_sha256(prefix)),
            n_variants=np.asarray(3, dtype=np.int64),
            subgenome=np.asarray(subgenome),
        )
        prefixes[subgenome] = str(prefix)
        mappings[subgenome] = str(mapping)
    monkeypatch.setattr(PREP, "GENOTYPE_PREFIXES", prefixes)
    monkeypatch.setattr(PREP, "SNP_TO_GENE", mappings)
    return prefixes, mappings


def _prepare(
    groups: Path,
    phenotype: Path,
    out_dir: Path,
    *,
    production: bool = False,
):
    return PREP.prepare_wheat_inputs(
        source_groups=groups,
        source_phenotype=phenotype,
        out_dir=out_dir,
        expected_group_sha256=_digest(groups),
        expected_phenotype_sha256=_digest(phenotype),
        production=production,
    )


def test_prepare_binds_inputs_and_expected_family(tmp_path, monkeypatch):
    groups, phenotype = _write_sources(tmp_path)
    _write_resources(tmp_path, monkeypatch, ["s01", "s02"])

    result = _prepare(groups, phenotype, tmp_path / "run")
    manifest = json.loads(Path(result.manifest).read_text())

    assert manifest["n_groups"] == 2
    assert manifest["expanded_edge_rows"] == 6
    assert manifest["n_unique_edges"] == 6
    assert manifest["direction_counts"] == {"AB": 2, "AD": 2, "BD": 2}
    assert manifest["trait"] == "days_to_emerg_env_adjusted"
    assert manifest["n_samples"] == 2
    assert manifest["group_family_sha256"] == (
        "10847b5037e0fca718c8c457474dd7bb0e3f13e5467dfb31a7e194fa7402e1ff"
    )
    assert manifest["edge_family_sha256"] == (
        "6c4128d3a3b4691aa65842c20394b01daed7f4417b05051a5337be9f43b331ad"
    )
    assert Path(result.groups).read_bytes() == groups.read_bytes()
    assert Path(result.phenotype).read_bytes() == phenotype.read_bytes()


def test_prepare_emits_canonical_edge_config_with_task7_checkpoint_policy(
    tmp_path, monkeypatch,
):
    groups, phenotype = _write_sources(tmp_path)
    prefixes, mappings = _write_resources(tmp_path, monkeypatch, ["s01", "s02"])
    result = _prepare(groups, phenotype, tmp_path / "run")

    config_path = Path(result.config)
    assert config_path.name == "interact.generated.group.omnib.yaml"
    cfg = yaml.safe_load(config_path.read_text())
    validate_interact_config(cfg)
    ic = cfg["interact"]
    assert ic == {
        "mode": "group",
        "subgenomes": ["A", "B", "D"],
        "groups": str(Path(result.groups).resolve()),
        "statistic": "omniB",
        "hypothesis_unit": "edge",
        "subset_order": 2,
        "family_scope": "primary_only",
        "primary_transform": "INT",
        "primary_multiplicity": "bootstrap_minp",
        "genotype": prefixes,
        "snp_to_gene": mappings,
        "phenotype": str(Path(result.phenotype).resolve()),
        "sample_col": "sample",
        "trait": "days_to_emerg_env_adjusted",
        "burden": {"cap": 150, "min_snp": 3, "maf_min": 0.01, "n_pc": 3},
        "grm": {"method": "grm_from_X", "maf_min": 0.01,
                "scope": "all_subgenomes"},
        "calibration": {
            "method": "bootstrap",
            "B": 2000,
            "seed": 2026,
            "checkpoint": {
                "enabled": True,
                "root": str((tmp_path / "run" / "checkpoints").resolve()),
                "block_size": 25,
            },
        },
    }
    assert cfg["outputs"] == {
        "out_dir": str((tmp_path / "run").resolve()),
        "full_ranking": True,
    }
    assert cfg["provenance"]["pre_run_manifest"] == str(
        (tmp_path / "run" / "provenance" / "pre_run_manifest.json").resolve())
    assert cfg["provenance"]["blas_thread_policy"] == {
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }


def test_prepare_rejects_wrong_source_hash_before_copying(tmp_path, monkeypatch):
    groups, phenotype = _write_sources(tmp_path)
    _write_resources(tmp_path, monkeypatch, ["s01", "s02"])

    with pytest.raises(ValueError, match="groups source SHA-256 mismatch"):
        PREP.prepare_wheat_inputs(
            groups, phenotype, tmp_path / "run",
            expected_group_sha256="0" * 64,
            expected_phenotype_sha256=_digest(phenotype),
        )
    assert not (tmp_path / "run" / "inputs" / "f2143.generated.tsv").exists()


def test_prepare_rejects_post_copy_identity_failure(tmp_path, monkeypatch):
    groups, phenotype = _write_sources(tmp_path)
    _write_resources(tmp_path, monkeypatch, ["s01", "s02"])
    real_atomic_write = PREP._atomic_write_bytes

    def corrupt_groups_copy(path, body):
        if Path(path).name == "f2143.generated.tsv":
            body = b"corrupt\n"
        return real_atomic_write(path, body)

    monkeypatch.setattr(PREP, "_atomic_write_bytes", corrupt_groups_copy)
    with pytest.raises(RuntimeError, match="durable groups copy SHA-256 mismatch"):
        _prepare(groups, phenotype, tmp_path / "run")


@pytest.mark.parametrize(
    "body, message",
    [
        ("sample\tdays_to_emerg_env_adjusted\ns01\t40\ns01\t41\n",
         "duplicate sample"),
        ("sample\tdays_to_emerg_env_adjusted\n\t40\ns02\t41\n",
         "blank or missing sample"),
    ],
)
def test_prepare_rejects_duplicate_or_blank_sample_ids(
    tmp_path, monkeypatch, body, message,
):
    groups, phenotype = _write_sources(tmp_path)
    phenotype.write_text(body, encoding="utf-8")
    _write_resources(tmp_path, monkeypatch, ["s01", "s02"])
    with pytest.raises(ValueError, match=message):
        _prepare(groups, phenotype, tmp_path / "run")


@pytest.mark.parametrize(
    "body, message",
    [
        ("group_id\tgene_A\tgene_B\ng1\ta1\tb1\n", "missing required group"),
        ("group_id\tgene_A\tgene_B\tgene_D\ng1\ta1\t\td1\n",
         "blank group identifier or gene"),
        ("group_id\tgene_A\tgene_B\tgene_D\ng1\ta1\tb1\td1\n"
         "g1\ta2\tb2\td2\n", "duplicate group ID"),
        ("group_id\tgene_A\tgene_B\tgene_D\ng1\ta1\tb1\td1\n"
         "g2\ta1\tb1\td1\n", "duplicate biological group"),
    ],
)
def test_prepare_rejects_malformed_groups(
    tmp_path, monkeypatch, body, message,
):
    groups, phenotype = _write_sources(tmp_path)
    groups.write_text(body, encoding="utf-8")
    _write_resources(tmp_path, monkeypatch, ["s01", "s02"])
    with pytest.raises(ValueError, match=message):
        _prepare(groups, phenotype, tmp_path / "run")


def test_prepare_counts_unique_edges_after_global_deduplication(
    tmp_path, monkeypatch,
):
    groups, phenotype = _write_sources(
        tmp_path,
        rows=[("g1", "a1", "b1", "d1"), ("g2", "a1", "b1", "d2")],
    )
    _write_resources(tmp_path, monkeypatch, ["s01", "s02"])
    result = _prepare(groups, phenotype, tmp_path / "run")
    manifest = json.loads(Path(result.manifest).read_text())
    assert manifest["expanded_edge_rows"] == 6
    assert manifest["n_unique_edges"] == 5
    assert manifest["direction_counts"] == {"AB": 1, "AD": 2, "BD": 2}


def test_prepare_preserves_triad_id_family_identity_without_rewriting_source(
    tmp_path, monkeypatch,
):
    rows = [
        ("a1|b1|d1", "a1", "b1", "d1"),
        ("a2|b2|d2", "a2", "b2", "d2"),
    ]
    groups, phenotype = _write_sources(tmp_path, rows=rows, id_column="triad_id")
    _write_resources(tmp_path, monkeypatch, ["s01", "s02"])
    result = _prepare(groups, phenotype, tmp_path / "run")
    manifest = json.loads(Path(result.manifest).read_text())
    assert manifest["source_group_id_column"] == "triad_id"
    assert manifest["ordered_group_ids_sha256"] == (
        "268839984b6ae5916d01927ef90bdf1acf78bc0c83f3e0f044a953aae7ff6f36"
    )
    assert Path(result.groups).read_bytes() == groups.read_bytes()


def test_prepare_is_byte_deterministic_on_repeat(tmp_path, monkeypatch):
    groups, phenotype = _write_sources(tmp_path)
    _write_resources(tmp_path, monkeypatch, ["s01", "s02"])
    first = _prepare(groups, phenotype, tmp_path / "run")
    first_config = Path(first.config).read_bytes()
    first_manifest = Path(first.manifest).read_bytes()
    second = _prepare(groups, phenotype, tmp_path / "run")
    assert Path(second.config).read_bytes() == first_config
    assert Path(second.manifest).read_bytes() == first_manifest


@pytest.mark.parametrize(
    "n_groups,n_samples,message",
    [
        (2, 827, "production requires exactly 2143 groups"),
        (2143, 2, "production requires exactly 827 unique non-missing samples"),
    ],
)
def test_prepare_enforces_frozen_production_counts(
    tmp_path, monkeypatch, n_groups, n_samples, message,
):
    rows = [
        (f"a{i}|b{i}|d{i}", f"a{i}", f"b{i}", f"d{i}")
        for i in range(n_groups)
    ]
    samples = [f"s{i:04d}" for i in range(n_samples)]
    groups, phenotype = _write_sources(
        tmp_path, rows=rows, samples=samples, id_column="triad_id")
    _write_resources(tmp_path, monkeypatch, samples[: max(2, min(n_samples, 12))])
    monkeypatch.setattr(PREP, "FROZEN_GROUP_SHA256", _digest(groups))
    monkeypatch.setattr(PREP, "FROZEN_PHENOTYPE_SHA256", _digest(phenotype))
    with pytest.raises(ValueError, match=message):
        _prepare(groups, phenotype, tmp_path / "run", production=True)


def test_prepare_production_validates_real_config_and_mapping_fingerprints(
    tmp_path, monkeypatch,
):
    rows = [
        (f"a{i}|b{i}|d{i}", f"a{i}", f"b{i}", f"d{i}")
        for i in range(2143)
    ]
    samples = [f"s{i:04d}" for i in range(827)]
    groups, phenotype = _write_sources(
        tmp_path, rows=rows, samples=samples, id_column="triad_id")
    _write_resources(tmp_path, monkeypatch, samples)
    monkeypatch.setattr(PREP, "FROZEN_GROUP_SHA256", _digest(groups))
    monkeypatch.setattr(PREP, "FROZEN_PHENOTYPE_SHA256", _digest(phenotype))
    monkeypatch.setattr(PREP, "capture_source_identity", lambda: {
        "git_commit": "1" * 40,
        "git_tree": "2" * 40,
        "package_source_sha256": "3" * 64,
        "source_clean": True,
    })

    result = _prepare(groups, phenotype, tmp_path / "run", production=True)
    manifest = json.loads(Path(result.manifest).read_text())
    assert manifest["validation"] == {
        "schema": "passed",
        "mapping_fingerprints": "passed",
        "input_preflight": "passed",
    }
    assert manifest["n_groups"] == 2143
    assert manifest["n_samples"] == 827
    assert manifest["config_sha256"] == _digest(Path(result.config))
    assert manifest["config"]["sha256"] == _digest(Path(result.config))
    assert set(manifest["source_identity"]) == {
        "git_commit", "git_tree", "package_source_sha256", "source_clean"}
    assert len(manifest["source_identity"]["package_source_sha256"]) == 64
    assert manifest["runtime_fingerprint"]["blas_thread_policy"] == {
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }


def test_prepare_rejects_bim_npz_fingerprint_mismatch(tmp_path, monkeypatch):
    groups, phenotype = _write_sources(tmp_path)
    _, mappings = _write_resources(tmp_path, monkeypatch, ["s01", "s02"])
    mapping = Path(mappings["A"])
    with np.load(mapping, allow_pickle=True) as loaded:
        payload = {name: loaded[name] for name in loaded.files}
    payload["bim_sha256"] = np.asarray("f" * 64)
    np.savez(mapping, **payload)

    with pytest.raises(ValueError, match="fingerprint mismatch"):
        _prepare(groups, phenotype, tmp_path / "run")
