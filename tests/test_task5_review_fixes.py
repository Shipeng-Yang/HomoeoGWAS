"""Regressions for Task 5 review findings at the public routing boundary."""

from __future__ import annotations

import json
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

import homoeogwas.cli as CLI
import homoeogwas.interact as I
import homoeogwas.io as IO
import homoeogwas.omnib_family as F
from homoeogwas.group_family import MasterGroupFamily, expand_pair_edges
from homoeogwas.interaction_config import normalize_interact_config


def _canonical_config(*, qa_only=False, B=2000):
    return {
        "interact": {
            "mode": "group",
            "subgenomes": ["A", "B"],
            "statistic": "omniB",
            "hypothesis_unit": "group",
            "subset_order": 2,
            "family_scope": "primary_only",
            "genotype": {"A": "geno_A", "B": "geno_B"},
            "snp_to_gene": {"A": "map_A.npz", "B": "map_B.npz"},
            "groups": "groups.tsv",
            "phenotype": "phenotype.tsv",
            "sample_col": "sample",
            "trait": "trait",
            "calibration": {
                "method": "bootstrap", "B": B, "seed": 2026,
                "qa_only": qa_only,
            },
        },
        "outputs": {"out_dir": "out", "full_ranking": True, "plots": False},
    }


def _fixed_scores(B=3):
    family = MasterGroupFamily(
        ("A", "B"), ("one", "two"), (("a1", "b1"), ("a2", "b2")))
    expanded = expand_pair_edges(family)
    base_p = np.array([
        [0.010, 0.20, 0.30, 0.40],
        [0.060, 0.26, 0.36, 0.46],
    ])
    edge_p = np.column_stack([
        base_p[:, 0],
        np.tile(base_p[:, 1:], (1, (B + 2) // 3))[:, :B],
    ])
    group_p = edge_p.copy()
    scores = F.OmniBFamilyScores(
        edge_p=edge_p,
        group_p=group_p,
        edge_components_obs=np.tile(np.array([[0.1, 0.2, 0.3]]), (2, 1)),
        edge_estimable=np.ones(2, dtype=bool),
        group_estimable=np.ones(2, dtype=bool),
        W=np.eye(4),
        y=np.arange(4.0),
        covariance_components={"A": 0.2, "B": 0.2, "e": 0.6},
    )
    return family, scores, expanded


def _forced_calibration(p_obs, p_null, *, alpha):
    assert p_obs.shape == (2,)
    return {
        "alpha": alpha,
        "method": "parametric_bootstrap_minp_plus_one",
        "B": p_null.shape[1],
        "empirical_p": 0.01,
        "threshold": 0.05,
        "threshold_comparator": "strict_less_than",
        "rejected": True,
        "rejected_local": [0],
        "adjusted_p_local": np.array([0.01, 0.50]),
        "n_degenerate_replicates": 0,
        "degenerate_policy": "conservative",
    }


def test_group_qa_bootstrap_has_no_discovery_authority(monkeypatch, tmp_path):
    family, scores, expanded = _fixed_scores()
    monkeypatch.setattr(
        F, "score_omnib_family", lambda *args, **kwargs: (scores, expanded))
    monkeypatch.setattr(F, "bootstrap_minp_calibration", _forced_calibration)
    ranking_path = tmp_path / "ranking.tsv"

    result = F.run_group_scan_omnib(
        {}, family, np.arange(4.0), np.arange(4),
        hypothesis_unit="group", family_scope="primary_only",
        bootstrap_B=3, inferential=False,
        full_dump_path=str(ranking_path),
    )

    fwer = result.model_diagnostics["bootstrap_fwer"]
    assert result.n_sig is None
    assert result.sig is None
    assert result.minp_boot_rejected is None
    assert result.minp_boot_emp is None
    assert result.minp_boot_threshold is None
    assert fwer["inferential"] is False
    assert fwer["formal_discovery_layer"] is False
    assert fwer["rejected"] is None
    assert fwer["empirical_p"] is None
    assert fwer["threshold"] is None
    assert fwer["adjusted_p"] == [None, None]
    assert fwer["n_rejected"] is None
    assert fwer["sig"] is None
    assert fwer["rejected_indices"] is None
    assert fwer["rejected_hypothesis_ids"] is None
    assert all(record["p_adjusted_bootstrap_minp"] is None
               for record in result.top)
    assert all(record["p_adjusted_bootstrap_minp"] is None
               for record in result.analytic_screen_sig)
    assert fwer["qa_diagnostics"]["role"] == (
        "noninferential_do_not_threshold")
    assert fwer["qa_diagnostics"]["empirical_p"] == 0.01
    assert fwer["qa_diagnostics"]["threshold"] == 0.05
    assert fwer["qa_diagnostics"]["adjusted_p"] == [0.01, 0.5]
    ranking = pd.read_csv(ranking_path, sep="\t")
    assert set(ranking["primary_sig"]) == {0}
    assert ranking["p_adjusted_bootstrap_minp"].isna().all()
    assert F.omnib_fwer_consistency_flags(asdict(result)) == ()


def test_group_formal_bootstrap_retains_adjusted_and_cutoff_fields(monkeypatch):
    family, scores, expanded = _fixed_scores()
    monkeypatch.setattr(
        F, "score_omnib_family", lambda *args, **kwargs: (scores, expanded))
    monkeypatch.setattr(F, "bootstrap_minp_calibration", _forced_calibration)

    result = F.run_group_scan_omnib(
        {}, family, np.arange(4.0), np.arange(4),
        hypothesis_unit="group", bootstrap_B=3)

    fwer = result.model_diagnostics["bootstrap_fwer"]
    assert result.minp_boot_threshold == 0.05
    assert fwer["threshold"] == 0.05
    assert fwer["adjusted_p"] == [0.01, 0.5]
    assert result.top[0]["p_adjusted_bootstrap_minp"] == 0.01
    assert "qa_diagnostics" not in fwer
    assert F.omnib_fwer_consistency_flags(asdict(result)) == ()


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload.__setitem__("minp_boot_threshold", 0.05),
        lambda payload: payload["model_diagnostics"]["bootstrap_fwer"].__setitem__(
            "threshold", 0.05),
        lambda payload: payload["model_diagnostics"]["bootstrap_fwer"].__setitem__(
            "adjusted_p", [0.01, None]),
        lambda payload: payload["top"][0].__setitem__(
            "p_adjusted_bootstrap_minp", 0.01),
        lambda payload: payload["analytic_screen_sig"][0].__setitem__(
            "p_adjusted_bootstrap_minp", 0.01),
    ],
)
def test_qa_consistency_rejects_formal_adjustment_leaks(monkeypatch, mutate):
    family, scores, expanded = _fixed_scores()
    monkeypatch.setattr(
        F, "score_omnib_family", lambda *args, **kwargs: (scores, expanded))
    monkeypatch.setattr(F, "bootstrap_minp_calibration", _forced_calibration)
    payload = asdict(F.run_group_scan_omnib(
        {}, family, np.arange(4.0), np.arange(4),
        hypothesis_unit="group", bootstrap_B=3, inferential=False))
    mutate(payload)
    assert "OMNIB_FWER_QA_FORMAL_ADJUSTMENT_PRESENT" in set(
        F.omnib_fwer_consistency_flags(payload))


def test_qa_consistency_rejects_reintroduced_authority(monkeypatch):
    family, scores, expanded = _fixed_scores()
    monkeypatch.setattr(
        F, "score_omnib_family", lambda *args, **kwargs: (scores, expanded))
    monkeypatch.setattr(F, "bootstrap_minp_calibration", _forced_calibration)
    payload = asdict(F.run_group_scan_omnib(
        {}, family, np.arange(4.0), np.arange(4),
        hypothesis_unit="group", bootstrap_B=3, inferential=False))
    payload["model_diagnostics"]["bootstrap_fwer"]["rejected"] = True
    assert "OMNIB_FWER_QA_AUTHORITY_PRESENT" in set(
        F.omnib_fwer_consistency_flags(payload))


def test_normalized_legacy_omnib_receives_canonical_grm_and_maf_defaults():
    cfg = _canonical_config()
    cfg["interact"].update(mode="pairwise", pairs="pairs.tsv")
    cfg["interact"].pop("groups")
    cfg["interact"].pop("hypothesis_unit")
    cfg["interact"].pop("subset_order")

    normalized = normalize_interact_config(cfg)

    assert normalized["interact"]["mode"] == "group"
    assert normalized["interact"]["grm"] == {
        "method": "grm_from_X", "maf_min": 0.01, "scope": "all_subgenomes"}
    assert normalized["interact"]["burden"]["maf_min"] == 0.01
    I.validate_interact_config(normalized)


@pytest.mark.parametrize(
    ("section", "updates", "message"),
    [
        ("grm", {"method": "compute_grm_maf"}, "grm.method=grm_from_X"),
        ("grm", {"maf_min": 0.02}, "grm.maf_min=0.01"),
        ("grm", {"scope": "per_subgenome"}, "grm.scope=all_subgenomes"),
        ("burden", {"maf_min": 0.02}, "burden.maf_min=0.01"),
    ],
)
def test_canonical_group_rejects_nonconforming_null_and_maf_settings(
        section, updates, message):
    cfg = _canonical_config()
    cfg["interact"][section] = updates
    with pytest.raises(SystemExit, match=message):
        I.validate_interact_config(cfg)


def test_canonical_group_ranking_skips_incompatible_legacy_plotter(
        monkeypatch, tmp_path, capsys):
    (tmp_path / "interact_trait_ranking_group_INT.tsv").write_text(
        "hypothesis_id\tp_interaction\n")
    monkeypatch.setattr(
        CLI, "_find_rscript",
        lambda *_: pytest.fail("canonical group ranking reached legacy R plotter"))

    CLI._autoplot_interact_figures(tmp_path)

    output = capsys.readouterr().out
    assert "canonical group plotting is skipped" in output
    assert "pending a canonical plotting adapter" in output
    assert "figures written" not in output


def test_cmd_group_qa_b19_loads_family_once_and_serializes_no_authority(
        monkeypatch, tmp_path):
    config = _canonical_config(qa_only=True, B=19)
    config["interact"]["groups"] = str(tmp_path / "groups.tsv")
    config["interact"]["phenotype"] = str(tmp_path / "phenotype.tsv")
    config["outputs"]["out_dir"] = str(tmp_path / "out")
    config_path = tmp_path / "interact.yaml"
    config_path.write_text(yaml.safe_dump(config))

    family, scores, expanded = _fixed_scores(B=19)
    load_calls = []

    def load_once(path, subs):
        load_calls.append((path, tuple(subs)))
        return family

    def fake_preflight(cfg, *, master_family=None):
        assert master_family is family
        return []

    samples = [f"s{i}" for i in range(12)]
    subdata = I.SubgenomeData(
        X=np.ones((12, 2)), gene_snp={"a1": np.array([0, 1]),
                                     "b1": np.array([0, 1])},
        samples=samples, chunk=None)
    monkeypatch.setattr(I, "load_master_group_family", load_once)
    monkeypatch.setattr(I, "preflight_interact", fake_preflight)
    monkeypatch.setattr(I, "validate_interact_config", lambda cfg: None)
    monkeypatch.setattr(I, "_load_subgenome", lambda *args, **kwargs: subdata)
    monkeypatch.setattr(
        IO, "read_delimited",
        lambda *args, **kwargs: pd.DataFrame({
            "sample": samples, "trait": np.arange(12, dtype=float)}))
    monkeypatch.setattr(
        F, "score_omnib_family", lambda *args, **kwargs: (scores, expanded))
    monkeypatch.setattr(F, "bootstrap_minp_calibration", _forced_calibration)

    actual_scan = I.run_group_scan_omnib

    def checked_scan(subdata_arg, family_arg, y, sample_idx, **kwargs):
        assert family_arg is family
        assert kwargs["inferential"] is False
        assert kwargs["grm_method"] == "grm_from_X"
        assert kwargs["maf_min"] == 0.01
        assert kwargs["burden_maf"] == 0.01
        return actual_scan(
            subdata_arg, family_arg, y, sample_idx, **kwargs)

    monkeypatch.setattr(I, "run_group_scan_omnib", checked_scan)
    rc = I.cmd_interact(SimpleNamespace(config=str(config_path), out_dir=None, n_jobs=1))

    assert rc == 0
    assert len(load_calls) == 1
    payload = json.loads((tmp_path / "out" / "interact_trait.json").read_text())
    result = payload["results"]["INT"]
    assert result["n_sig"] is None
    assert result["sig"] is None
    assert result["minp_boot_rejected"] is None
    assert result["minp_boot_emp"] is None
    assert result["minp_boot_threshold"] is None
    fwer = result["model_diagnostics"]["bootstrap_fwer"]
    assert fwer["inferential"] is False
    assert fwer["empirical_p"] is None
    assert fwer["threshold"] is None
    assert fwer["adjusted_p"] == [None, None]
    assert fwer["qa_diagnostics"]["role"] == (
        "noninferential_do_not_threshold")
    assert all(record["p_adjusted_bootstrap_minp"] is None
               for record in result["top"])
    result["minp_boot_threshold"] = 0.05
    assert "OMNIB_FWER_QA_FORMAL_ADJUSTMENT_PRESENT" in set(
        F.omnib_fwer_consistency_flags(result))
