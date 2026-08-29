"""Non-SNP biallelic marker contracts."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from bed_reader import to_bed

from homoeogwas import cli
from homoeogwas.cli import preflight, validate_config
from homoeogwas.markers import validate_marker_contract
from homoeogwas.workflow import build_fit_config


def _write_bed(prefix: Path, dosage: np.ndarray, ids: list[str]) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    to_bed(
        str(prefix) + ".bed",
        dosage.astype(np.float32),
        properties={
            "fid": ["0"] * dosage.shape[0],
            "iid": [f"S{i}" for i in range(dosage.shape[0])],
            "sid": ids,
            "chromosome": ["1"] * dosage.shape[1],
            "bp_position": list(range(1, dosage.shape[1] + 1)),
            "allele_1": ["P"] * dosage.shape[1],
            "allele_2": ["A"] * dosage.shape[1],
        },
        count_A1=True,
    )


def _manifest(path: Path, ids: list[str], marker_type: str = "PAV") -> None:
    pd.DataFrame(
        {"variant_id": ids, "marker_type": [marker_type] * len(ids)}
    ).to_csv(path, sep="\t", index=False)


def test_binary_presence_manifest_and_values_pass(tmp_path):
    prefix = tmp_path / "A" / "all"
    ids = ["pav1", "pav2"]
    _write_bed(prefix, np.array([[0, 2], [2, 2], [0, np.nan]]), ids)
    manifest = tmp_path / "manifest_A.tsv"
    _manifest(manifest, ids)
    problems, provenance = validate_marker_contract(
        subgenomes=["A"],
        bed_prefix=lambda _: prefix,
        encoding="binary_presence_0_2",
        manifest_template=str(tmp_path / "manifest_{subgenome}.tsv"),
    )
    assert problems == []
    item = provenance["subgenomes"]["A"]
    assert item["binary_value_check"] == "PASS"
    assert item["marker_type_counts"] == {"PAV": 2}


def test_binary_presence_rejects_zero_one_encoding(tmp_path):
    prefix = tmp_path / "A" / "all"
    ids = ["pav1", "pav2"]
    _write_bed(prefix, np.array([[0, 1], [1, 0], [0, 1]]), ids)
    _manifest(tmp_path / "manifest_A.tsv", ids)
    problems, _ = validate_marker_contract(
        subgenomes=["A"],
        bed_prefix=lambda _: prefix,
        encoding="binary_presence_0_2",
        manifest_template=str(tmp_path / "manifest_{subgenome}.tsv"),
    )
    assert any("outside {0,2}" in problem for problem in problems)


def test_manifest_must_match_exact_bim_ids(tmp_path):
    prefix = tmp_path / "A" / "all"
    _write_bed(prefix, np.array([[0, 2], [2, 0]]), ["pav1", "pav2"])
    _manifest(tmp_path / "manifest_A.tsv", ["pav1", "not_in_bim"])
    problems, _ = validate_marker_contract(
        subgenomes=["A"],
        bed_prefix=lambda _: prefix,
        encoding="binary_presence_0_2",
        manifest_template=str(tmp_path / "manifest_{subgenome}.tsv"),
        check_values=False,
    )
    assert any("not an exact BIM-ID manifest" in problem for problem in problems)


def test_manifest_must_follow_bim_order(tmp_path):
    prefix = tmp_path / "A" / "all"
    _write_bed(prefix, np.array([[0, 2], [2, 0]]), ["pav1", "pav2"])
    _manifest(tmp_path / "manifest_A.tsv", ["pav2", "pav1"])
    problems, _ = validate_marker_contract(
        subgenomes=["A"],
        bed_prefix=lambda _: prefix,
        encoding="binary_presence_0_2",
        manifest_template=str(tmp_path / "manifest_{subgenome}.tsv"),
        check_values=False,
    )
    assert any("different order" in problem for problem in problems)


def test_distinct_standard_grm_does_not_inherit_scan_manifest(tmp_path):
    scan = tmp_path / "scan" / "A" / "all"
    grm = tmp_path / "grm" / "A" / "all"
    _write_bed(scan, np.array([[0, 2], [2, 0], [0, 2]]), ["pav1", "pav2"])
    _write_bed(grm, np.array([[0, 1], [1, 2], [2, 0]]), ["snp1", "snp2"])
    _manifest(tmp_path / "manifest_A.tsv", ["pav1", "pav2"])
    phenotype = tmp_path / "phenotype.tsv"
    pd.DataFrame({"IID": ["S0", "S1", "S2"], "trait": [1, 2, 3]}).to_csv(
        phenotype, sep="\t", index=False
    )
    cfg = {
        "panel": {"subgenomes": ["A"]},
        "phenotype": {
            "path": str(phenotype),
            "sample_col": "IID",
            "trait": "trait",
        },
        "genotype": {
            "scan_bed_prefix_template": str(
                tmp_path / "scan" / "{subgenome}" / "all"
            ),
            "marker_encoding": "binary_presence_0_2",
            "marker_manifest_template": str(
                tmp_path / "manifest_{subgenome}.tsv"
            ),
            "grm": {
                "source": "bed",
                "bed_prefix_template": str(
                    tmp_path / "grm" / "{subgenome}" / "all"
                ),
            },
        },
    }
    validate_config(cfg)
    assert preflight(cfg) == []


def test_distinct_binary_grm_is_value_checked(tmp_path):
    scan = tmp_path / "scan" / "A" / "all"
    grm = tmp_path / "grm" / "A" / "all"
    _write_bed(scan, np.array([[0, 1], [1, 2], [2, 0]]), ["snp1", "snp2"])
    _write_bed(grm, np.array([[0, 1], [1, 0], [0, 1]]), ["pav1", "pav2"])
    _manifest(tmp_path / "grm_manifest_A.tsv", ["pav1", "pav2"])
    phenotype = tmp_path / "phenotype.tsv"
    pd.DataFrame({"IID": ["S0", "S1", "S2"], "trait": [1, 2, 3]}).to_csv(
        phenotype, sep="\t", index=False
    )
    cfg = {
        "panel": {"subgenomes": ["A"]},
        "phenotype": {
            "path": str(phenotype),
            "sample_col": "IID",
            "trait": "trait",
        },
        "genotype": {
            "scan_bed_prefix_template": str(
                tmp_path / "scan" / "{subgenome}" / "all"
            ),
            "grm": {
                "source": "bed",
                "bed_prefix_template": str(
                    tmp_path / "grm" / "{subgenome}" / "all"
                ),
                "marker_encoding": "binary_presence_0_2",
                "marker_manifest_template": str(
                    tmp_path / "grm_manifest_{subgenome}.tsv"
                ),
            },
        },
    }
    validate_config(cfg)
    assert any("outside {0,2}" in problem for problem in preflight(cfg))


def test_predict_refuses_preflight_failure(monkeypatch):
    cfg = {
        "panel": {"subgenomes": ["A"]},
        "phenotype": {"trait": "trait"},
    }
    monkeypatch.setattr(cli, "load_config", lambda _: cfg)
    monkeypatch.setattr(cli, "validate_config", lambda _: None)
    monkeypatch.setattr(cli, "preflight", lambda _: ["invalid GRM encoding"])
    monkeypatch.setattr(
        cli,
        "join_samples",
        lambda _: pytest.fail("join_samples must not run after preflight failure"),
    )
    args = SimpleNamespace(config="unused.yaml", out_dir=None)
    with pytest.raises(SystemExit, match="preflight failed"):
        cli.cmd_predict(args)


def test_workflow_emits_non_snp_marker_contract():
    cfg = build_fit_config(
        subgenomes=["A", "B"],
        phenotype="p.tsv",
        sample_col="IID",
        trait="yield",
        bed_template="g/{subgenome}/all",
        out_dir="out",
        marker_encoding="haplotype_dosage_0_1_2",
        marker_manifest_template="manifest_{subgenome}.tsv",
    )
    assert cfg["genotype"]["marker_encoding"] == "haplotype_dosage_0_1_2"
    assert cfg["genotype"]["marker_manifest_template"].endswith(
        "manifest_{subgenome}.tsv"
    )
