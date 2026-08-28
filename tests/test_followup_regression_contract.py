from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from homoeogwas.followup import load_followup_inputs, run_followup
from homoeogwas.run_registry import load_registry

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAPESEED_RESULT = Path(
    "/mnt/7302share/fast_ysp/U7_GWAS/results/experimental/"
    "rapeseed_flowering_group_omnib_f17404_v1"
)
RAPESEED_EVIDENCE = (
    PROJECT_ROOT / "analyses/evidence/rapeseed_flowering.evidence.yaml"
)
RUN_LOCAL_REGRESSION = os.environ.get(
    "HOMOEOGWAS_RUN_LOCAL_REGRESSION", ""
).strip() == "1"


@pytest.mark.skipif(
    not RAPESEED_RESULT.exists(), reason="local formal rapeseed result unavailable")
def test_rapeseed_full_family_and_discoveries_are_provenance_bound():
    inputs = load_followup_inputs(RAPESEED_RESULT)

    assert inputs.formal_contract["n_planned"] == 17_404
    assert inputs.formal_contract["n_valid"] == 17_404
    assert inputs.formal_contract["n_significant"] == 2
    assert inputs.formal_contract["primary_family"] == "edge"
    assert inputs.formal_contract["edge_family_sha256"] == (
        "d45a69f2656cecf2d28698810d8b46ec70591ee913fc1cd790785fce72dc7e25")
    assert inputs.formal_contract["group_family_sha256"] == (
        "ca6e29d59e48422f586dd5a786dd0da65bb16f566aee2520689176c1b716869f")
    assert inputs.formal_hits["hypothesis_id"].tolist() == [
        "edge:AC:BnaA02g19610D:BnaC02g22960D",
        "edge:AC:BnaA02g19700D:BnaC02g23040D",
    ]
    assert inputs.formal_hits["p_adjusted_bootstrap_minp"].tolist() == pytest.approx([
        0.00849575212393803, 0.020989505247376312])


def test_cross_species_inventory_keeps_unmigrated_runs_historical():
    registry = load_registry(
        PROJECT_ROOT / "analyses/cross_species_interaction_inventory.yaml")
    by_id = {run.id: run for run in registry.runs}

    assert by_id["rapeseed.flowering.formal-20260828"].kind == "historical"
    assert by_id["wheat.days-to-emerg.route-b"].kind == "historical"
    assert by_id["cotton.FibLen.pairwise-omnib"].kind == "historical"
    assert by_id["peanut.hundred-seed-weight.pairwise-omnib"].kind == "historical"
    assert by_id["rapeseed.flowering.canonical-group-omnib.v1"].kind == "interaction"


@pytest.mark.slow
@pytest.mark.skipif(
    not RUN_LOCAL_REGRESSION or not RAPESEED_RESULT.exists(),
    reason=(
        "set HOMOEOGWAS_RUN_LOCAL_REGRESSION=1 on the project machine "
        "to replay the frozen rapeseed result"),
)
def test_rapeseed_followup_matches_frozen_contract(tmp_path):
    result = run_followup(
        RAPESEED_RESULT,
        out_dir=tmp_path,
        material_folds=20,
        evidence=RAPESEED_EVIDENCE,
        n_jobs=20,
        grm_blas_threads=1,
    )
    audit = json.loads((tmp_path / "independent_audit.json").read_text())
    tiers = (tmp_path / "evidence_tiers.tsv").read_text()

    assert result["status"] == "COMPLETED"
    assert audit["status"] == "PASS"
    assert audit["formal_hit_count"] == 2
    assert audit["formal_contract"]["n_planned"] == 17_404
    assert audit["formal_contract"]["n_valid"] == 17_404
    assert audit["formal_contract"]["primary_family"] == "edge"
    assert len(audit["formal_contract"]["edge_family_sha256"]) == 64
    assert len(audit["formal_contract"]["group_family_sha256"]) == 64
    assert audit["formal_replay_max_abs_error"] <= audit[
        "formal_replay_allowed_abs_error"]
    assert audit["material_deletion"]["samples_deleted_exactly_once"] == 926
    assert audit["locus_clustering"]["independent_loci"] == 1
    assert "matched-tissue expression support" in tiers
