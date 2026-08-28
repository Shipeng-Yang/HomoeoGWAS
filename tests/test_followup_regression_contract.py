from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from homoeogwas.followup import run_followup
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
    assert audit["formal_replay_max_abs_error"] <= audit[
        "formal_replay_allowed_abs_error"]
    assert audit["material_deletion"]["samples_deleted_exactly_once"] == 926
    assert audit["locus_clustering"]["independent_loci"] == 1
    assert "matched-tissue expression support" in tiers
