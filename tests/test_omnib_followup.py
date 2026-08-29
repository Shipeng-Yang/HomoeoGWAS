from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from homoeogwas.cli import main
from homoeogwas.followup import (
    FollowupError,
    candidate_coordinates,
    cluster_formal_units,
    deterministic_folds,
    load_followup_inputs,
    prepare_formal_replay,
    run_environment_deletion,
    run_followup,
    run_material_deletion,
    summarize_stability,
)
from homoeogwas.group_family import MasterGroupFamily, expand_pair_edges
from homoeogwas.interact import SubgenomeData
from homoeogwas.omnib_family import (
    _family_provenance,
    score_omnib_family,
    score_omnib_subset,
)
from homoeogwas.workflow import build_interact_config


def _toy_group_inputs(subgenomes, *, n=48, n_groups=3, snps_per_gene=6):
    rng = np.random.default_rng(90210 + len(subgenomes))
    subdata = {}
    for sub in subgenomes:
        matrix = rng.integers(
            0, 3, size=(n, n_groups * snps_per_gene)).astype(float)
        subdata[sub] = SubgenomeData(
            X=matrix,
            gene_snp={
                f"g{i}": np.arange(i * snps_per_gene, (i + 1) * snps_per_gene)
                for i in range(n_groups)
            },
            samples=[f"S{i:03d}" for i in range(n)],
            chunk=None,
        )
    family = MasterGroupFamily(
        subgenomes=tuple(subgenomes),
        group_ids=tuple(f"group_{i}" for i in range(n_groups)),
        genes=tuple(
            tuple(f"g{i}" for _ in subgenomes) for i in range(n_groups)),
    )
    y = rng.normal(size=n)
    return subdata, family, y, np.arange(n)


@pytest.mark.parametrize(
    "subgenomes", [("A", "C"), ("A", "B", "D"), ("A", "B", "C", "D")])
def test_subset_all_rows_replays_prepared_observed(subgenomes):
    subdata, family, y, sample_idx = _toy_group_inputs(subgenomes)
    scores, expanded = score_omnib_family(
        subdata, family, y, sample_idx, bootstrap_B=0,
        bootstrap_seed=2026, n_jobs=1, grm_method="grm_from_X")

    replay = score_omnib_subset(
        scores, family, expanded, np.arange(len(y)), y, n_jobs=1)

    np.testing.assert_allclose(
        replay.edge_p, scores.edge_p[:, 0], rtol=1e-6, atol=1e-10)
    np.testing.assert_allclose(
        replay.group_p, scores.group_p[:, 0], rtol=1e-6, atol=1e-10)
    assert np.nanmax(np.abs(replay.edge_p - scores.edge_p[:, 0])) <= 1e-10
    assert np.nanmax(np.abs(replay.group_p - scores.group_p[:, 0])) <= 1e-10
    if len(subgenomes) == 4:
        assert len(expand_pair_edges(family).edges) == 6 * len(family.group_ids)


def test_subset_scorer_rejects_fixed_covariate_context():
    subdata, family, y, sample_idx = _toy_group_inputs(("A", "C"))
    scores, expanded = score_omnib_family(
        subdata, family, y, sample_idx, bootstrap_B=0,
        bootstrap_seed=2026, n_jobs=1, grm_method="grm_from_X")
    scores.covariate_block = np.ones((len(y), 1))

    with pytest.raises(ValueError, match="fixed covariates"):
        score_omnib_subset(
            scores, family, expanded, np.arange(len(y)), y, n_jobs=1)


def test_subset_scorer_validates_keep_and_minimum_sample_count():
    subdata, family, y, sample_idx = _toy_group_inputs(("A", "C"))
    scores, expanded = score_omnib_family(
        subdata, family, y, sample_idx, bootstrap_B=0,
        bootstrap_seed=2026, n_jobs=1, grm_method="grm_from_X")

    with pytest.raises(ValueError, match="unique sorted"):
        score_omnib_subset(
            scores, family, expanded, [1, 0, 2, 3, 4, 5, 6, 7, 8, 9, 9],
            y[:11], n_jobs=1)
    with pytest.raises(ValueError, match="at least 10"):
        score_omnib_subset(scores, family, expanded, np.arange(9), y[:9], n_jobs=1)


def _write_formal_artifacts(tmp_path, cfg, family, ranking):
    expanded = expand_pair_edges(family)
    provenance = _family_provenance(family, expanded)
    ranked = ranking.sort_values("rank").reset_index(drop=True)
    significant = ranked.loc[ranked["primary_sig"].astype(int) == 1]
    threshold = (
        float(significant["p_interaction"].max()) * 1.01
        if not significant.empty else
        float(ranked["p_interaction"].min()) * 0.5
    )
    result_path = tmp_path / "interact_flowering_time.json"
    result_path.write_text(json.dumps({
        "tool": "homoeogwas", "command": "interact", "mode": "group",
        "trait": "flowering_time", "subgenomes": list(family.subgenomes),
        "provenance": {
            "mode": "group", "statistic": "omniB",
            "primary_transform": "INT",
            "primary_multiplicity": "bootstrap_minp",
            "hypothesis_unit": cfg["interact"]["hypothesis_unit"],
            "family_scope": cfg["interact"]["family_scope"],
            "subset_order": 2, "covariates_detail": {"policy": "none"},
            "config_sha256": hashlib.sha256(
                (tmp_path / "configs/interact.generated.group.omnib.yaml").read_bytes()
            ).hexdigest(),
            **provenance,
        },
        "results": {"INT": {
            "n": 48, "G": len(ranked), "n_planned": len(ranked),
            "n_valid": int((ranked["p_unestimable"].astype(int) == 0).sum()),
            "n_sig": len(significant), "sig": significant.to_dict("records"),
            "bootstrap_B": 2000, "bootstrap_seed": 2026,
            "model_diagnostics": {
                "family_provenance": provenance,
                "bootstrap_fwer": {
                    "family_id": cfg["interact"]["hypothesis_unit"],
                    "declared_hypothesis_unit": cfg["interact"]["hypothesis_unit"],
                    "family_scope": cfg["interact"]["family_scope"],
                    "B": 2000, "n_rejected": len(significant),
                    "threshold": threshold, "sig": significant.to_dict("records"),
                    "inferential": True,
                },
            },
        }},
    }))
    audit = tmp_path / "audit"
    audit.mkdir()
    (audit / "homoeogwas_audit.json").write_text(json.dumps({
        "overall_status": "INTERNAL_DISCOVERY_REPLICATION_REQUIRED",
        "n_results": 1,
        "records": [{
            "source": str(result_path.resolve()), "command": "interact",
            "mode": "group", "statistic": "omniB",
            "trait": "flowering_time", "discovery_count": len(significant),
            "n": 48, "n_planned": len(ranked),
            "n_valid": int((ranked["p_unestimable"].astype(int) == 0).sum()),
        }],
    }))


def _write_canonical_result_fixture(tmp_path, *, significant=(1, 0)):
    cfg = build_interact_config(
        subgenomes=("A", "C"),
        bed_prefixes={"A": "geno/A", "C": "geno/C"},
        snp_to_gene={"A": "maps/A.npz", "C": "maps/C.npz"},
        phenotype="phenotype.tsv", sample_col="sample", trait="flowering_time",
        out_dir=str(tmp_path), groups="groups.tsv", hypothesis_unit="edge",
        subset_order=2, family_scope="primary_only", perm_b=2000,
        statistic="omniB")
    config_dir = tmp_path / "configs"
    config_dir.mkdir(parents=True)
    config_path = config_dir / "interact.generated.group.omnib.yaml"
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    (tmp_path / "groups.tsv").write_text(
        "group_id\tgene_A\tgene_C\n"
        "group1\tgA1\tgC1\n"
        "group2\tgA2\tgC2\n")
    p_values = [1e-6 if significant[0] else 0.2,
                2e-6 if significant[1] else 0.3]
    ranking = pd.DataFrame({
        "rank": [0, 1],
        "hypothesis_id": ["edge:AC:gA1:gC1", "edge:AC:gA2:gC2"],
        "family_id": ["edge", "edge"],
        "hypothesis_unit": ["edge", "edge"],
        "group_id": ["group1", "group2"],
        "edge_id": ["AC:gA1:gC1", "AC:gA2:gC2"],
        "group_ids": ["group1", "group2"],
        "direction": ["AC", "AC"],
        "sub_x": ["A", "A"], "sub_y": ["C", "C"],
        "gene_A": ["gA1", "gA2"], "gene_C": ["gC1", "gC2"],
        "p_interaction": p_values,
        "p_adjusted_bootstrap_minp": [
            0.01 if value else 0.8 for value in significant],
        "primary_sig": list(significant), "p_unestimable": [0, 0],
        "driving_edge": ["AC:gA1:gC1", "AC:gA2:gC2"],
        "driving_component": ["pc1", "minor_burden"],
        "driving_component_p": [3e-7, 0.1],
    })
    ranking_path = tmp_path / "interact_flowering_time_ranking_group_INT.tsv"
    ranking.to_csv(ranking_path, sep="\t", index=False)
    family = MasterGroupFamily(
        subgenomes=("A", "C"), group_ids=("group1", "group2"),
        genes=(("gA1", "gC1"), ("gA2", "gC2")))
    _write_formal_artifacts(tmp_path, cfg, family, ranking)
    return config_path, ranking_path


def test_loader_selects_only_formal_primary_hits(tmp_path):
    _write_canonical_result_fixture(tmp_path, significant=(1, 0))

    loaded = load_followup_inputs(tmp_path)

    assert loaded.formal_hits["hypothesis_id"].tolist() == ["edge:AC:gA1:gC1"]
    assert loaded.hypothesis_unit == "edge"
    assert loaded.subgenomes == ("A", "C")


def test_loader_returns_no_discovery_without_candidates(tmp_path):
    _write_canonical_result_fixture(tmp_path, significant=(0, 0))

    loaded = load_followup_inputs(tmp_path)

    assert loaded.formal_hits.empty


def test_loader_rejects_edited_discovery_labels(tmp_path):
    _, ranking_path = _write_canonical_result_fixture(
        tmp_path, significant=(1, 0))
    ranking = pd.read_csv(ranking_path, sep="\t")
    ranking.loc[1, "primary_sig"] = 1
    ranking.loc[1, "p_adjusted_bootstrap_minp"] = 0.01
    ranking.to_csv(ranking_path, sep="\t", index=False)

    with pytest.raises(FollowupError, match="discovery labels"):
        load_followup_inputs(tmp_path)


def test_loader_rejects_truncated_formal_family(tmp_path):
    _, ranking_path = _write_canonical_result_fixture(
        tmp_path, significant=(1, 0))
    ranking = pd.read_csv(ranking_path, sep="\t").iloc[:1]
    ranking.to_csv(ranking_path, sep="\t", index=False)

    with pytest.raises(FollowupError, match="full hypothesis inventory"):
        load_followup_inputs(tmp_path)


def test_loader_rejects_covariate_formal_context(tmp_path):
    config_path, _ = _write_canonical_result_fixture(
        tmp_path, significant=(1, 0))
    cfg = yaml.safe_load(config_path.read_text())
    cfg["interact"]["covariates"] = {"n_pcs": 2}
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    with pytest.raises(FollowupError, match="covariate-bearing"):
        load_followup_inputs(tmp_path)


def test_loader_refuses_legacy_ranking_as_canonical(tmp_path):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    cfg = {
        "interact": {
            "mode": "pairwise", "subgenomes": ["A", "C"],
            "pairs": "pairs.tsv", "statistic": "omniB",
        },
        "outputs": {"out_dir": str(tmp_path)},
    }
    path = config_dir / "interact.legacy.yaml"
    path.write_text(yaml.safe_dump(cfg))
    pd.DataFrame({"primary_sig": [1]}).to_csv(
        tmp_path / "interact_trait_ranking_pairwise_INT.tsv", sep="\t", index=False)

    with pytest.raises(FollowupError, match="legacy.*migration"):
        load_followup_inputs(tmp_path, config=path)


def _write_replay_fixture(tmp_path, subdata, family, y, *, corrupt_p=False):
    groups = tmp_path / "groups.tsv"
    pd.DataFrame({
        "group_id": family.group_ids,
        **{
            f"gene_{sub}": [genes[index] for genes in family.genes]
            for index, sub in enumerate(family.subgenomes)
        },
    }).to_csv(groups, sep="\t", index=False)
    phenotype = tmp_path / "phenotype.tsv"
    pd.DataFrame({
        "sample": subdata[family.subgenomes[0]].samples,
        "flowering_time": y,
    }).to_csv(phenotype, sep="\t", index=False)
    cfg = build_interact_config(
        subgenomes=family.subgenomes,
        bed_prefixes={sub: f"geno/{sub}" for sub in family.subgenomes},
        snp_to_gene={sub: f"maps/{sub}.npz" for sub in family.subgenomes},
        phenotype=str(phenotype), sample_col="sample", trait="flowering_time",
        out_dir=str(tmp_path), groups=str(groups), hypothesis_unit="edge",
        subset_order=2, family_scope="primary_only", perm_b=2000,
        statistic="omniB")
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_path = config_dir / "interact.generated.group.omnib.yaml"
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    scores, expanded = score_omnib_family(
        subdata, family, y, np.arange(len(y)), bootstrap_B=0,
        bootstrap_seed=2026, n_jobs=1, grm_method="grm_from_X")
    records = []
    for edge_index, edge in enumerate(expanded.edges):
        component = int(np.nanargmin(scores.edge_components_obs[edge_index]))
        p_value = float(scores.edge_p[edge_index, 0])
        records.append({
            "hypothesis_id": f"edge:{edge.edge_id}", "family_id": "edge",
            "hypothesis_unit": "edge", "group_id": edge.source_group_ids[0],
            "edge_id": edge.edge_id, "group_ids": edge.source_group_ids[0],
            "direction": edge.direction, "sub_x": edge.sub_x, "sub_y": edge.sub_y,
            **{
                f"gene_{sub}": edge.gene_x if sub == edge.sub_x else edge.gene_y
                for sub in family.subgenomes
            },
            "p_interaction": p_value,
            "p_adjusted_bootstrap_minp": 0.8,
            "primary_sig": 0, "p_unestimable": 0,
            "driving_edge": edge.edge_id,
            "driving_component": ["minor_burden", "pc1", "kernel_hadamard"][component],
            "driving_component_p": scores.edge_components_obs[edge_index, component],
        })
    ranking = pd.DataFrame.from_records(records).sort_values(
        "p_interaction").reset_index(drop=True)
    ranking.insert(0, "rank", np.arange(len(ranking)))
    ranking.loc[0, "primary_sig"] = 1
    ranking.loc[0, "p_adjusted_bootstrap_minp"] = 0.01
    if corrupt_p:
        ranking.loc[0, "p_interaction"] *= 1.01
    ranking.to_csv(
        tmp_path / "interact_flowering_time_ranking_group_INT.tsv",
        sep="\t", index=False)
    _write_formal_artifacts(tmp_path, cfg, family, ranking)
    return config_path


def test_prepare_formal_replay_matches_frozen_edge(monkeypatch, tmp_path):
    subdata, family, y, _ = _toy_group_inputs(("A", "C"))
    _write_replay_fixture(tmp_path, subdata, family, y)
    loaded = load_followup_inputs(tmp_path)
    monkeypatch.setattr(
        "homoeogwas.followup._load_verified_subgenomes",
        lambda interact, subgenomes, config_dir: subdata)

    prepared = prepare_formal_replay(loaded, n_jobs=1)

    assert prepared.replay["hypothesis_id"].tolist() == [
        loaded.formal_hits.iloc[0]["hypothesis_id"]]
    assert prepared.formal_replay_max_abs_error < 1e-10
    assert prepared.analyzed_samples == tuple(subdata["A"].samples)


def test_prepare_formal_replay_stops_on_p_mismatch(monkeypatch, tmp_path):
    subdata, family, y, _ = _toy_group_inputs(("A", "C"))
    _write_replay_fixture(tmp_path, subdata, family, y, corrupt_p=True)
    loaded = load_followup_inputs(tmp_path)
    monkeypatch.setattr(
        "homoeogwas.followup._load_verified_subgenomes",
        lambda interact, subgenomes, config_dir: subdata)

    with pytest.raises(FollowupError, match="formal p reproduction failed"):
        prepare_formal_replay(loaded, n_jobs=1)


def _prepared_toy(monkeypatch, tmp_path):
    subdata, family, y, _ = _toy_group_inputs(("A", "C"))
    _write_replay_fixture(tmp_path, subdata, family, y)
    loaded = load_followup_inputs(tmp_path)
    monkeypatch.setattr(
        "homoeogwas.followup._load_verified_subgenomes",
        lambda interact, subgenomes, config_dir: subdata)
    return prepare_formal_replay(loaded, n_jobs=1)


def test_material_folds_cover_every_string_sample_once():
    samples = [f"S{i:03d}" for i in range(41)]

    folds = deterministic_folds(samples, n_folds=8, seed=2026)
    flat = [sample for fold in folds for sample in fold]

    assert sorted(flat) == sorted(samples)
    assert max(map(len, folds)) - min(map(len, folds)) <= 1
    assert folds == deterministic_folds(samples, n_folds=8, seed=2026)


def test_material_deletion_scores_only_formal_hits(monkeypatch, tmp_path):
    prepared = _prepared_toy(monkeypatch, tmp_path)

    deletion = run_material_deletion(
        prepared, n_folds=4, n_jobs=1)

    assert len(deletion) == 4 * len(prepared.inputs.formal_hits)
    assert deletion["deleted_fold"].nunique() == 4
    assert set(deletion["hypothesis_id"]) == set(
        prepared.inputs.formal_hits["hypothesis_id"])


def test_material_deletion_uses_observable_fork_workers(monkeypatch, tmp_path):
    prepared = _prepared_toy(monkeypatch, tmp_path)

    deletion = run_material_deletion(prepared, n_folds=4, n_jobs=2)

    execution = deletion.attrs["parallel_execution"]
    assert execution["backend"] == "fork_shared_memory"
    assert execution["effective_jobs"] == 2
    assert execution["process_model"] == "processes"
    assert len(execution["worker_pids"]) == 2
    assert execution["inner_threads"] == 1


def test_environment_deletion_reaggregates_repeated_rows(monkeypatch, tmp_path):
    prepared = _prepared_toy(monkeypatch, tmp_path)
    base = prepared.phenotype_rows[["sample", prepared.inputs.trait]].copy()
    first = base.assign(environment="E1")
    second = base.assign(
        **{prepared.inputs.trait: base[prepared.inputs.trait] + 0.2},
        environment="E2")
    repeated = replace(
        prepared, phenotype_rows=pd.concat([first, second], ignore_index=True))

    frame, status = run_environment_deletion(
        repeated, "environment", n_jobs=1)

    assert status == "COMPLETED"
    assert set(frame["deleted_environment"]) == {"E1", "E2"}
    assert (frame["n"] == len(prepared.analyzed_samples)).all()


def test_environment_deletion_uses_observable_fork_workers(monkeypatch, tmp_path):
    prepared = _prepared_toy(monkeypatch, tmp_path)
    base = prepared.phenotype_rows[["sample", prepared.inputs.trait]].copy()
    repeated = replace(prepared, phenotype_rows=pd.concat([
        base.assign(environment="E1"),
        base.assign(environment="E2"),
    ], ignore_index=True))

    frame, status = run_environment_deletion(
        repeated, "environment", n_jobs=2)

    assert status == "COMPLETED"
    execution = frame.attrs["parallel_execution"]
    assert execution["backend"] == "fork_shared_memory"
    assert execution["effective_jobs"] == 2
    assert len(execution["worker_pids"]) == 2


def test_missing_environment_is_explicit(monkeypatch, tmp_path):
    prepared = _prepared_toy(monkeypatch, tmp_path)

    frame, status = run_environment_deletion(prepared, None, n_jobs=1)

    assert frame.empty
    assert status == "NOT_AVAILABLE_NO_ENVIRONMENT_COLUMN"


def test_environment_levels_below_ten_are_reported(monkeypatch, tmp_path):
    prepared = _prepared_toy(monkeypatch, tmp_path)
    rows = prepared.phenotype_rows.copy()
    rows["environment"] = "only_environment"
    prepared = replace(prepared, phenotype_rows=rows)

    frame, status = run_environment_deletion(
        prepared, "environment", n_jobs=1)

    assert status == "NOT_AVAILABLE_LT10_SAMPLES"
    assert set(frame["deleted_environment"]) == {"only_environment"}
    assert (frame["available"] == 0).all()
    assert set(frame["unavailable_reason"]) == {"LT10_RETAINED_SAMPLES"}


def test_stability_summary_keeps_formal_adjusted_p(monkeypatch, tmp_path):
    prepared = _prepared_toy(monkeypatch, tmp_path)
    material = run_material_deletion(prepared, n_folds=4, n_jobs=1)

    stability = summarize_stability(prepared, material, pd.DataFrame())

    assert stability.iloc[0]["formal_p_fwer"] == pytest.approx(
        prepared.inputs.formal_hits.iloc[0]["p_adjusted_bootstrap_minp"])
    assert 0 <= stability.iloc[0]["material_nominal_support_fraction"] <= 1
    assert 0 <= stability.iloc[0][
        "material_driving_edge_agreement_fraction"] <= 1
    assert stability.iloc[0]["interpretation"].startswith("candidate-only internal")


def _two_edge_hits(*, second_copy_set=("A", "C")):
    return pd.DataFrame({
        "rank": [0, 1],
        "hypothesis_id": ["edge:AC:a1:c1", "edge:AC:a2:c2"],
        "copy_set": [("A", "C"), second_copy_set],
        "gene_A": ["a1", "a2"], "gene_C": ["c1", "c2"],
        "chrom_A": ["A02", "A02"], "pos_A": [100_000, 140_000],
        "chrom_C": ["C02", "C02"], "pos_C": [200_000, 315_000],
    })


def test_cluster_merges_same_copy_set_by_physical_rule():
    loci, links = cluster_formal_units(
        _two_edge_hits(), {}, max_distance_bp=1_000_000)

    assert loci["locus_id"].nunique() == 1
    assert links.iloc[0]["merge_reason"].startswith(
        "all_compared_copies_within")
    assert links.iloc[0]["merge_basis"] == "physical"


def test_cluster_keeps_different_edge_directions_separate():
    hits = _two_edge_hits(second_copy_set=("A", "D"))

    loci, links = cluster_formal_units(hits, {})

    assert loci["locus_id"].nunique() == 2
    assert not bool(links.iloc[0]["merge"])
    assert links.iloc[0]["merge_reason"] == "different_copy_set"


def test_cluster_reports_physical_merge_even_when_one_copy_ld_is_low():
    pc1 = {("A", "a1", "a2"): 0.7, ("C", "c1", "c2"): 0.001}

    loci, links = cluster_formal_units(_two_edge_hits(), pc1)

    assert loci["locus_id"].nunique() == 1
    assert links.iloc[0]["merge_basis"] == "physical"
    assert links.iloc[0]["pc1_r2_C"] == pytest.approx(0.001)


def test_candidate_coordinates_use_exact_loaded_analysis_chunk(monkeypatch, tmp_path):
    prepared = _prepared_toy(monkeypatch, tmp_path)
    updated = {}
    for sub, data in prepared.subdata.items():
        updated[sub] = replace(data, chunk=SimpleNamespace(
            chrom=np.asarray([f"{sub}01"] * data.X.shape[1], dtype=object),
            pos=np.arange(data.X.shape[1]) * 100 + 1))
    prepared = replace(prepared, subdata=updated)

    coordinates = candidate_coordinates(prepared)

    assert coordinates.iloc[0]["copy_set"] == ("A", "C")
    assert coordinates.iloc[0]["chrom_A"] == "A01"
    assert coordinates.iloc[0]["chrom_C"] == "C01"


def test_run_followup_no_hit_writes_successful_stop(tmp_path):
    _write_canonical_result_fixture(tmp_path, significant=(0, 0))

    result = run_followup(tmp_path, n_jobs=1)

    assert result["status"] == "NO_FORMAL_DISCOVERY"
    assert (tmp_path / "followup" / "followup_summary.json").exists()
    assert (tmp_path / "followup" / "followup_identity.json").exists()
    assert not (tmp_path / "followup" / "candidate_evidence.tsv").exists()
    assert run_followup(tmp_path, n_jobs=1) == result


def test_followup_rejects_changed_identity_in_completed_output(tmp_path):
    _write_canonical_result_fixture(tmp_path, significant=(0, 0))
    run_followup(tmp_path, n_jobs=1)

    with pytest.raises(FollowupError, match="identity mismatch"):
        run_followup(tmp_path, n_jobs=1, environment_col="environment")


def test_followup_rejects_invalid_formal_audit_status(tmp_path):
    _write_canonical_result_fixture(tmp_path, significant=(0, 0))
    audit_path = tmp_path / "audit" / "homoeogwas_audit.json"
    audit = json.loads(audit_path.read_text())
    audit["overall_status"] = "ANALYSIS_INVALID"
    audit_path.write_text(json.dumps(audit))

    with pytest.raises(FollowupError, match="audit status.*ANALYSIS_INVALID"):
        run_followup(tmp_path, n_jobs=1)


def test_followup_rejects_unknown_formal_audit_status(tmp_path):
    _write_canonical_result_fixture(tmp_path, significant=(0, 0))
    audit_path = tmp_path / "audit" / "homoeogwas_audit.json"
    audit = json.loads(audit_path.read_text())
    audit["overall_status"] = "LOOKS_FINE_TO_ME"
    audit_path.write_text(json.dumps(audit))

    with pytest.raises(FollowupError, match="audit status.*LOOKS_FINE_TO_ME"):
        run_followup(tmp_path, n_jobs=1)


def test_followup_rejects_tampered_completed_output(tmp_path):
    _write_canonical_result_fixture(tmp_path, significant=(0, 0))
    run_followup(tmp_path, n_jobs=1)
    (tmp_path / "followup" / "followup_summary.json").write_text("{}\n")

    with pytest.raises(FollowupError, match="output hash mismatch"):
        run_followup(tmp_path, n_jobs=1)


@pytest.mark.parametrize("column", ["p_unestimable", "primary_sig"])
def test_followup_rejects_fractional_binary_formal_fields(tmp_path, column):
    _, ranking_path = _write_canonical_result_fixture(tmp_path, significant=(0, 0))
    ranking = pd.read_csv(ranking_path, sep="\t")
    ranking[column] = ranking[column].astype(float)
    ranking.loc[0, column] = 1.5
    ranking.to_csv(ranking_path, sep="\t", index=False)

    with pytest.raises(FollowupError, match=f"{column}.*binary"):
        run_followup(tmp_path, n_jobs=1)


def test_followup_rejects_nonempty_unbound_output_directory(tmp_path):
    _write_canonical_result_fixture(tmp_path, significant=(0, 0))
    out = tmp_path / "followup"
    out.mkdir()
    (out / "candidate_evidence.tsv").write_text("stale\n")

    with pytest.raises(FollowupError, match="non-empty.*identity"):
        run_followup(tmp_path, n_jobs=1)


def test_run_followup_toy_writes_complete_audited_package(monkeypatch, tmp_path):
    subdata, family, y, _ = _toy_group_inputs(("A", "C"))
    for sub, data in subdata.items():
        data.chunk = SimpleNamespace(
            chrom=np.asarray([f"{sub}01"] * data.X.shape[1], dtype=object),
            pos=np.arange(data.X.shape[1]) * 100 + 1)
    _write_replay_fixture(tmp_path, subdata, family, y)
    monkeypatch.setattr(
        "homoeogwas.followup._load_verified_subgenomes",
        lambda interact, subgenomes, config_dir: subdata)

    result = run_followup(
        tmp_path, material_folds=4, n_jobs=1, grm_blas_threads=1)

    out = tmp_path / "followup"
    assert result["status"] == "COMPLETED"
    assert result["audit_status"] == "PASS"
    assert (out / "formal_hit_reproduction.tsv").exists()
    assert (out / "material_deletion.tsv").exists()
    assert (out / "independent_loci.tsv").exists()
    assert (out / "evidence_tiers.tsv").exists()
    assert (out / "independent_audit.json").exists()
    assert (out / "FOLLOWUP_SUMMARY.md").exists()
    audit = json.loads((out / "independent_audit.json").read_text())
    assert audit["formal_contract"]["n_planned"] == len(family.group_ids)
    assert len(audit["formal_contract"]["edge_family_sha256"]) == 64


def test_followup_cli_parses_generic_options(tmp_path, monkeypatch, capsys):
    called = {}

    def fake_run(*args, **kwargs):
        called.update(kwargs)
        return {"status": "COMPLETED", "audit_status": "PASS"}

    monkeypatch.setattr("homoeogwas.followup.run_followup", fake_run)

    assert main([
        "follow-up", str(tmp_path), "--material-folds", "8",
        "--n-jobs", "4", "--grm-blas-threads", "2"]) == 0
    assert called["material_folds"] == 8
    assert called["n_jobs"] == 4
    assert called["grm_blas_threads"] == 2
    assert "COMPLETED" in capsys.readouterr().out
