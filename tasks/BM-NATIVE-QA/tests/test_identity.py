from __future__ import annotations

import json
from pathlib import Path

import pytest
from bm_native_qa_harness import identity
from bm_native_qa_harness.identity import freeze_identity
from bm_native_qa_harness.plan import build_inventory


def test_design_payload_excludes_generated_values(amendment: dict) -> None:
    frozen = freeze_identity(
        build_inventory(amendment),
        fixture_manifest_sha256=(
            "7e77399ab34cdabf91f3ca92641eac37f3f0d9f4556b98131c97b45fcdadb08a"
        ),
        amendment_sha256="a" * 64,
        runner_test_sha256s={
            "runner.py": "b" * 64,
            "test_runner.py": "c" * 64,
        },
    )

    encoded = json.dumps(frozen.design_payload, sort_keys=True)
    assert "generated_array" not in encoded
    assert "seed_value" not in encoded
    assert len(frozen.qa_design_hash) == 64


def test_design_payload_binds_canonical_science_and_future_relative_paths(
    amendment: dict,
) -> None:
    frozen = freeze_identity(
        build_inventory(amendment),
        fixture_manifest_sha256="7" * 64,
        amendment_sha256="a" * 64,
        runner_test_sha256s={"runner.py": "b" * 64},
    )

    assert frozen.design_payload["canonical_interact"] == {
        "mode": "group",
        "statistic": "omniB",
        "hypothesis_unit": "group",
        "subset_order": 2,
        "family_scope": "primary_only",
        "primary_transform": "INT",
        "primary_multiplicity": "bootstrap_minp",
        "burden": {"cap": 150, "min_snp": 3, "maf_min": 0.01, "n_pc": 3},
        "grm": {
            "method": "grm_from_X",
            "maf_min": 0.01,
            "scope": "all_subgenomes",
        },
        "calibration": {
            "method": "bootstrap",
            "B": 199,
            "qa_only": True,
            "checkpoint_enabled": True,
            "checkpoint_block_size": 25,
        },
        "sample_col": "sample",
        "trait": "qa_trait",
        "full_ranking": True,
    }
    layout = frozen.design_payload["future_artifacts"]
    assert layout["root"] == "tasks/BM-NATIVE-QA/materialized/v1"
    assert len(layout["anchors"]) == 6
    assert len(layout["responses"]) == 12
    assert len(layout["invocations"]) == 16
    assert layout["anchors"][0]["anchor_npy"].endswith("/anchor.npy")
    assert layout["responses"][0]["phenotype_tsv"].endswith("/phenotype.tsv")
    assert layout["invocations"][0]["config_yaml"].endswith(".yaml")
    assert all(
        not Path(path).is_absolute()
        for section in ("anchors", "responses", "invocations")
        for row in layout[section]
        for key, path in row.items()
        if key.endswith(("_npy", "_tsv", "_yaml", "_root", "_dir"))
    )


def test_seed_ledger_matches_independently_derived_literals(amendment: dict) -> None:
    records = identity.build_seed_ledger("0" * 64, build_inventory(amendment))
    cotton_pc1 = {
        (record.purpose, record.truth_id): record.value
        for record in records
        if record.context_key == "cotton.pc1_spread_192"
    }

    assert cotton_pc1[("anchor", None)] == 6233245192901957
    assert cotton_pc1[("observed", "gaussian_null")] == 13941148495138826656
    assert (
        cotton_pc1[("observed", "mixed_sign_diagnostic_pve0p03")]
        == 6702508585982483686
    )
    assert cotton_pc1[("bootstrap", "gaussian_null")] == 18101087001120419280
    assert len(records) == 30


def test_seed_ledger_rejects_role_collision(amendment: dict, monkeypatch) -> None:
    monkeypatch.setattr(identity, "derive_seed", lambda *args: 7)

    with pytest.raises(RuntimeError, match="seed collision"):
        identity.build_seed_ledger("0" * 64, build_inventory(amendment))


def test_sha256_file_hashes_exact_bytes(tmp_path: Path) -> None:
    path = tmp_path / "member.bin"
    path.write_bytes(b"abc")

    assert identity.sha256_file(path) == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


def test_freeze_identity_rejects_non_sha256_binding(amendment: dict) -> None:
    with pytest.raises(RuntimeError, match="invalid SHA-256"):
        freeze_identity(
            build_inventory(amendment),
            fixture_manifest_sha256="7e77399a",
            amendment_sha256="a" * 64,
            runner_test_sha256s={"runner.py": "b" * 64},
        )
