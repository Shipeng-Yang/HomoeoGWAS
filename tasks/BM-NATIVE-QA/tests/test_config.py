from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest
from bm_native_qa_harness import config as config_module
from bm_native_qa_harness import plan as plan_module


def test_build_interact_config_matches_canonical_cotton_pc1_contract(
    amendment: dict,
    context_evidence: dict,
    tmp_path: Path,
) -> None:
    inventory = plan_module.bind_context_inputs(
        plan_module.build_inventory(amendment),
        context_evidence,
    )
    context = next(row for row in inventory.contexts if row.key == "cotton.pc1_spread_192")
    response = next(
        row
        for row in inventory.responses
        if row.context_key == context.key and row.truth_id == "gaussian_null"
    )
    invocation = next(
        row
        for row in inventory.invocations
        if row.response_id == response.response_id and row.jobs == 1
    )
    phenotype = tmp_path / "future-response.tsv"
    checkpoint = tmp_path / "future-checkpoint"
    output = tmp_path / "future-output"

    cfg = config_module.build_interact_config(
        context,
        response,
        invocation,
        bootstrap_seed=1234,
        phenotype_path=phenotype,
        checkpoint_root=checkpoint,
        output_root=output,
    )

    assert cfg["interact"] == {
        "mode": "group",
        "subgenomes": ["A", "D"],
        "groups": str(context.groups_path),
        "statistic": "omniB",
        "hypothesis_unit": "group",
        "subset_order": 2,
        "family_scope": "primary_only",
        "primary_transform": "INT",
        "primary_multiplicity": "bootstrap_minp",
        "benchmark_identity": {
            "panel_id": "REALG.CGVD1245",
            "sample_context": "pc1_spread_192",
            "feature_seed": 5177468918036905819,
        },
        "genotype": dict(context.bed_prefixes),
        "snp_to_gene": dict(context.snp_to_gene),
        "phenotype": str(phenotype),
        "sample_col": "sample",
        "trait": "qa_trait",
        "burden": {
            "cap": 150,
            "min_snp": 3,
            "maf_min": 0.01,
            "n_pc": 3,
            "feature_seed": 5177468918036905819,
        },
        "grm": {
            "method": "grm_from_X",
            "maf_min": 0.01,
            "scope": "all_subgenomes",
        },
        "calibration": {
            "method": "bootstrap",
            "B": 199,
            "seed": 1234,
            "qa_only": True,
            "checkpoint": {
                "enabled": True,
                "root": str(checkpoint),
                "block_size": 25,
            },
        },
    }
    assert cfg["outputs"] == {"out_dir": str(output), "full_ranking": True}


def test_build_interact_config_rejects_response_invocation_mismatch(
    amendment: dict,
    context_evidence: dict,
    tmp_path: Path,
) -> None:
    inventory = plan_module.bind_context_inputs(
        plan_module.build_inventory(amendment),
        context_evidence,
    )
    context = inventory.contexts[0]
    response = next(row for row in inventory.responses if row.context_key == context.key)
    invocation = next(
        row for row in inventory.invocations if row.response_id == response.response_id
    )

    with pytest.raises(RuntimeError, match="response/invocation identity mismatch"):
        config_module.build_interact_config(
            context,
            response,
            replace(invocation, response_id="wrong-response"),
            bootstrap_seed=1,
            phenotype_path=tmp_path / "response.tsv",
            checkpoint_root=tmp_path / "checkpoint",
            output_root=tmp_path / "output",
        )


def test_build_interact_config_rejects_existing_output_target(
    amendment: dict,
    context_evidence: dict,
    tmp_path: Path,
) -> None:
    inventory = plan_module.bind_context_inputs(
        plan_module.build_inventory(amendment),
        context_evidence,
    )
    context = inventory.contexts[0]
    response = next(row for row in inventory.responses if row.context_key == context.key)
    invocation = next(
        row for row in inventory.invocations if row.response_id == response.response_id
    )
    output = tmp_path / "existing-output"
    output.mkdir()

    with pytest.raises(RuntimeError, match="output target already exists"):
        config_module.build_interact_config(
            context,
            response,
            invocation,
            bootstrap_seed=1,
            phenotype_path=tmp_path / "response.tsv",
            checkpoint_root=tmp_path / "checkpoint",
            output_root=output,
        )


def test_pc1_jobs_one_and_four_share_scientific_config_identity(
    amendment: dict,
    context_evidence: dict,
    tmp_path: Path,
) -> None:
    inventory = plan_module.bind_context_inputs(
        plan_module.build_inventory(amendment),
        context_evidence,
    )
    context = inventory.contexts[0]
    response = next(row for row in inventory.responses if row.context_key == context.key)
    invocations = [
        row
        for row in inventory.invocations
        if row.response_id == response.response_id and row.sample_context == "pc1_spread_192"
    ]
    configs = [
        config_module.build_interact_config(
            context,
            response,
            invocation,
            bootstrap_seed=99,
            phenotype_path=tmp_path / "shared-response.tsv",
            checkpoint_root=tmp_path / f"checkpoint-jobs{invocation.jobs}",
            output_root=tmp_path / f"output-jobs{invocation.jobs}",
        )
        for invocation in invocations
    ]

    assert [row.jobs for row in invocations] == [1, 4]
    assert config_module.scientific_config_sha256(configs[0]) == (
        config_module.scientific_config_sha256(configs[1])
    )


def test_dump_config_writes_deterministic_yaml_bytes(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"

    digest = config_module.dump_config({"z": 2, "a": {"b": 1}}, path)

    expected = b"a:\n  b: 1\nz: 2\n"
    assert path.read_bytes() == expected
    assert digest == hashlib.sha256(expected).hexdigest()


def test_dump_config_rejects_existing_path_without_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("preserve: true\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="config target already exists"):
        config_module.dump_config({"replacement": True}, path)

    assert path.read_text(encoding="utf-8") == "preserve: true\n"
