from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import yaml
from bm_native_qa_harness import authority, bundle, identity, plan

from scripts.benchmarks.v201.track_omnib import build_synthetic_omnib_context


def _inventory(tmp_path: Path) -> plan.ProspectiveInventory:
    panels = (("cotton", "PANEL.COTTON"), ("wheat", "PANEL.WHEAT"))
    contexts = tuple(
        plan.ContextSpec(
            key=f"{species}.{sample_context}",
            panel_id=panel,
            sample_context=sample_context,
            feature_seed=7,
            groups_path=tmp_path / "groups.tsv",
            groups_sha256="a" * 64,
            samples_path=tmp_path / f"{panel}.{sample_context}.samples.tsv",
            samples_sha256="b" * 64,
            expected_calibrated_groups=2,
            subgenomes=("A", "B") if panel.endswith("COTTON") else ("A", "B", "C"),
            bed_prefixes=(
                (("A", str(tmp_path / "A")), ("B", str(tmp_path / "B")))
                if panel.endswith("COTTON")
                else (
                    ("A", str(tmp_path / "A")),
                    ("B", str(tmp_path / "B")),
                    ("C", str(tmp_path / "C")),
                )
            ),
            snp_to_gene=(
                (("A", str(tmp_path / "A.npz")), ("B", str(tmp_path / "B.npz")))
                if panel.endswith("COTTON")
                else (
                    ("A", str(tmp_path / "A.npz")),
                    ("B", str(tmp_path / "B.npz")),
                    ("C", str(tmp_path / "C.npz")),
                )
            ),
        )
        for species, panel in panels
        for sample_context in (
            "pc1_spread_192",
            "seeded_random_192",
            "holdout_192",
        )
    )
    responses = tuple(
        plan.ResponseSpec(
            response_id=(
                f"qa_real80_njobs128_v1.{context.panel_id}."
                f"{context.sample_context}.{truth}"
            ),
            context_key=context.key,
            truth_id=truth,
            generator_scale_interaction_pve=(
                0.0 if truth == "gaussian_null" else 0.03
            ),
        )
        for context in contexts
        for truth in ("gaussian_null", "mixed_sign_diagnostic_pve0p03")
    )
    response_lookup = {
        (response.context_key, response.truth_id): response for response in responses
    }
    invocations = tuple(
        plan.InvocationSpec(
            invocation_id=(
                f"{response_lookup[(context.key, truth)].response_id}."
                f"workers128.{replica}"
            ),
            response_id=response_lookup[(context.key, truth)].response_id,
            panel_id=context.panel_id,
            sample_context=context.sample_context,
            truth_id=truth,
            jobs=128,
        )
        for context in contexts
        for truth in ("gaussian_null", "mixed_sign_diagnostic_pve0p03")
        for replica in (
            ("replica_a", "replica_b")
            if context.sample_context == "pc1_spread_192"
            else ("primary",)
        )
    )
    return plan.ProspectiveInventory(contexts, responses, invocations)


def _verified(
    tmp_path: Path,
    inventory: plan.ProspectiveInventory,
) -> authority.VerifiedMaterializationAuthority:
    artifact_root = tmp_path / "materialized" / "njobs128-v1"
    layout = identity._future_artifact_layout(
        inventory,
        root=str(artifact_root),
        run_namespace="qa_real80_njobs128_v1",
    )
    return authority.VerifiedMaterializationAuthority(
        qa_design_hash="1" * 64,
        inventory={"design_payload": {"future_artifacts": layout}},
        artifact_root=artifact_root,
        paths={},
        binding_hashes={"materialization_authority_sha256": "2" * 64},
        source_reverification={},
    )


def _synthetic_loader(context: plan.ContextSpec):
    built = build_synthetic_omnib_context(
        n=24,
        groups=2,
        copies=len(context.subgenomes),
        seed=7,
    )
    built = replace(
        built,
        panel_id=context.panel_id,
        sample_context=context.sample_context,
        feature_seed=context.feature_seed,
        retained_variant_masks=None,
        marker_mask_identity=None,
    )
    return built, tuple(f"sample_{index}" for index in range(24))


def test_bundle_exclusively_writes_six_twelve_sixteen_and_no_native_dirs(
    tmp_path: Path,
) -> None:
    inventory = _inventory(tmp_path)
    verified = _verified(tmp_path, inventory)
    seeds = identity.build_seed_ledger(
        verified.qa_design_hash,
        inventory,
        run_namespace="qa_real80_njobs128_v1",
    )

    manifest = bundle.materialize_bundle(
        verified,
        inventory=inventory,
        seeds=seeds,
        context_loader=_synthetic_loader,
    )

    root = verified.artifact_root
    assert manifest["counts"] == {"anchors": 6, "responses": 12, "configs": 16}
    assert len(list(root.glob("anchors/*/manifest.json"))) == 6
    assert len(list(root.glob("responses/*/manifest.json"))) == 12
    assert len(list(root.glob("configs/*.yaml"))) == 16
    assert not (root / "checkpoints").exists()
    assert not (root / "results").exists()
    assert not (root / "audits").exists()
    physical = json.loads((root / "materialization-manifest.json").read_text())
    assert physical == manifest
    assert physical["execution_authorized"] is False
    assert physical["authority_bindings"]["materialization_authority_sha256"] == (
        "2" * 64
    )
    assert {row["jobs"] for row in physical["configs"]} == {128}
    assert physical["physical_counts"] == physical["counts"]
    assert Path(physical["materialization_lock"]).exists()

    replicas = [
        row
        for row in physical["configs"]
        if row["invocation_id"].endswith(("replica_a", "replica_b"))
        and ".gaussian_null." in row["invocation_id"]
        and "PANEL.COTTON" in row["invocation_id"]
    ]
    assert len(replicas) == 2
    assert len({row["phenotype_tsv"] for row in replicas}) == 1
    assert len({row["bootstrap_seed"] for row in replicas}) == 1
    assert len({row["scientific_config_sha256"] for row in replicas}) == 1
    for row in replicas:
        config = yaml.safe_load(Path(row["config_path"]).read_text())
        assert config["interact"]["calibration"]["B"] == 199


def test_replica_identity_guard_rejects_any_scientific_difference() -> None:
    rows = [
        {
            "invocation_id": "x.workers128.replica_a",
            "response_id": "response",
            "jobs": 128,
            "bootstrap_seed": 7,
            "phenotype_tsv": "/x/phenotype.tsv",
            "scientific_config_sha256": "a" * 64,
        },
        {
            "invocation_id": "x.workers128.replica_b",
            "response_id": "response",
            "jobs": 128,
            "bootstrap_seed": 8,
            "phenotype_tsv": "/x/phenotype.tsv",
            "scientific_config_sha256": "a" * 64,
        },
    ]

    with pytest.raises(ValueError, match="replica identity mismatch"):
        bundle.require_replica_identity(rows)


def test_bundle_preserves_abort_diagnostics_and_forbids_retry(tmp_path: Path) -> None:
    inventory = _inventory(tmp_path)
    verified = _verified(tmp_path, inventory)
    seeds = identity.build_seed_ledger(
        verified.qa_design_hash,
        inventory,
        run_namespace="qa_real80_njobs128_v1",
    )
    calls = 0

    def failing_loader(context: plan.ContextSpec):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("fixture failure")
        return _synthetic_loader(context)

    with pytest.raises(ValueError, match="fixture failure"):
        bundle.materialize_bundle(
            verified,
            inventory=inventory,
            seeds=seeds,
            context_loader=failing_loader,
        )

    attempt = verified.artifact_root.parent / ".njobs128-v1.materialization-attempt"
    assert not verified.artifact_root.exists()
    abort = json.loads((attempt / "materialization-abort.json").read_text())
    assert abort["qa_design_hash"] == verified.qa_design_hash
    assert abort["authority_bindings"] == verified.binding_hashes
    assert abort["written_files"]
    started = json.loads((attempt / "materialization-attempt.json").read_text())
    assert started["qa_design_hash"] == verified.qa_design_hash
    with pytest.raises(FileExistsError, match="attempt already exists"):
        bundle.materialize_bundle(
            verified,
            inventory=inventory,
            seeds=seeds,
            context_loader=_synthetic_loader,
        )


def test_bundle_refuses_preexisting_exclusive_lock(tmp_path: Path) -> None:
    inventory = _inventory(tmp_path)
    verified = _verified(tmp_path, inventory)
    seeds = identity.build_seed_ledger(
        verified.qa_design_hash,
        inventory,
        run_namespace="qa_real80_njobs128_v1",
    )
    lock = verified.artifact_root.parent / ".njobs128-v1.materialization-lock"
    lock.parent.mkdir(parents=True)
    lock.write_text("occupied\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="lock already exists"):
        bundle.materialize_bundle(
            verified,
            inventory=inventory,
            seeds=seeds,
            context_loader=_synthetic_loader,
        )
