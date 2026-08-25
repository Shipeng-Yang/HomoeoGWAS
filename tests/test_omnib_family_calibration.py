"""Formal bootstrap-FWER calibration for canonical omniB families."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest

import homoeogwas.omnib_family as F
from homoeogwas.group_family import MasterGroupFamily, expand_pair_edges
from homoeogwas.interact import SubgenomeData, run_group_scan_omnib


def _make_sub(rng, n=72, g=12, spg=5):
    X = rng.integers(0, 3, size=(n, g * spg)).astype(float)
    gene_snp = {
        f"g{i}": np.arange(i * spg, (i + 1) * spg)
        for i in range(g)
    }
    return SubgenomeData(
        X=X, gene_snp=gene_snp,
        samples=[f"sample_{i}" for i in range(n)], chunk=None)


def _run_small_group_omnib(subgenomes, hypothesis_unit, B, full_dump_path):
    rng = np.random.default_rng(515)
    n_groups = 12
    subdata = {s: _make_sub(rng, g=n_groups) for s in subgenomes}
    family = MasterGroupFamily(
        subgenomes=tuple(subgenomes),
        group_ids=tuple(f"group_{i}" for i in range(n_groups)),
        genes=tuple(
            tuple(f"g{i}" for _ in subgenomes)
            for i in range(n_groups)
        ),
    )
    return run_group_scan_omnib(
        subdata, family, rng.standard_normal(72), np.arange(72),
        hypothesis_unit=hypothesis_unit, family_scope="primary_only",
        cap=150, n_pc=3, transform="INT", bootstrap_B=B,
        bootstrap_seed=2026, n_jobs=1, grm_method="grm_from_X",
        maf_min=0.01, burden_maf=0.01, min_snp=3,
        full_dump_path=full_dump_path,
    )


def test_edge_primary_calibrates_all_directions_in_one_minp_family():
    result = _run_small_group_omnib(
        subgenomes=("A", "B", "D"), hypothesis_unit="edge",
        B=19, full_dump_path=None)
    fwer = result.model_diagnostics["bootstrap_fwer"]
    assert fwer["family_id"] == "edge"
    assert fwer["n_hypotheses"] == result.G
    assert {h["direction"] for h in result.top} <= {"AB", "AD", "BD"}
    assert fwer["n_rejected"] == len(fwer["sig"])
    assert result.n_sig == fwer["n_rejected"]
    assert fwer["calibrated_layers"] == ["edge"]


def test_group_primary_adjusted_p_threshold_and_hit_set_agree(tmp_path):
    ranking_path = tmp_path / "ranking.tsv"
    result = _run_small_group_omnib(
        subgenomes=("A", "B", "D"), hypothesis_unit="group",
        B=19, full_dump_path=str(ranking_path))
    ranking = pd.read_csv(ranking_path, sep="\t")
    fwer = result.model_diagnostics["bootstrap_fwer"]
    rejected = ranking.loc[ranking.primary_sig == 1]
    assert set(rejected.group_id) == {
        h["group_id"] for h in fwer["sig"]
    }
    assert (rejected.p_adjusted_bootstrap_minp <= 0.05).all()
    if fwer["threshold"] is not None:
        assert (rejected.p_interaction < fwer["threshold"]).all()
    assert result.sig == fwer["sig"]


def _fixed_scores():
    family = MasterGroupFamily(
        ("A", "B", "D"), ("one", "two"),
        (("a1", "b1", "d1"), ("a2", "b2", "d2")))
    expanded = expand_pair_edges(family)
    edge_p = np.array([
        [0.010, 0.20, 0.30, 0.40],
        [0.020, 0.25, 0.35, 0.45],
        [0.030, 0.22, 0.32, 0.42],
        [0.040, 0.24, 0.34, 0.44],
        [np.nan, np.nan, np.nan, np.nan],
        [0.060, 0.26, 0.36, 0.46],
    ])
    group_p = np.array([
        [0.015, 0.21, 0.31, 0.41],
        [0.050, 0.25, 0.35, 0.45],
    ])
    scores = F.OmniBFamilyScores(
        edge_p=edge_p,
        group_p=group_p,
        edge_components_obs=np.tile(
            np.array([[0.1, 0.2, 0.3]]), (len(expanded.edges), 1)),
        edge_estimable=np.isfinite(edge_p[:, 0]),
        group_estimable=np.isfinite(group_p[:, 0]),
        W=np.eye(4), y=np.arange(4.0),
        covariance_components={"A": 0.2, "B": 0.2, "D": 0.2, "e": 0.4},
    )
    return family, scores, expanded


def test_joint_family_concatenates_once_and_preserves_full_index_mapping(
        monkeypatch, tmp_path):
    family, scores, expanded = _fixed_scores()
    monkeypatch.setattr(
        F, "score_omnib_family", lambda *args, **kwargs: (scores, expanded))
    calls = []

    def fake_calibration(p_obs, p_null, *, alpha):
        calls.append((p_obs.copy(), p_null.copy(), alpha))
        return {
            "alpha": alpha,
            "method": "parametric_bootstrap_minp_plus_one",
            "B": 3,
            "empirical_p": 0.01,
            "threshold": 1.0,
            "threshold_comparator": "strict_less_than",
            "rejected": True,
            "rejected_local": [0, 1, 2, 3, 4, 5, 6],
            "adjusted_p_local": np.array(
                [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]),
            "n_degenerate_replicates": 0,
            "degenerate_policy": "conservative",
        }

    monkeypatch.setattr(F, "bootstrap_minp_calibration", fake_calibration)
    ranking_path = tmp_path / "joint.tsv"
    result = run_group_scan_omnib(
        {}, family, np.arange(4.0), np.arange(4),
        hypothesis_unit="edge", family_scope="joint", transform="INT",
        bootstrap_B=3, full_dump_path=str(ranking_path))

    assert len(calls) == 1
    assert calls[0][0].shape == (7,)  # five finite edges + two groups
    assert result.G == 8
    fwer = result.model_diagnostics["bootstrap_fwer"]
    assert fwer["family_id"] == "joint"
    assert fwer["calibrated_layers"] == ["edge", "group"]
    assert fwer["hypothesis_ids"][0].startswith("edge:")
    assert fwer["hypothesis_ids"][-1].startswith("group:")
    assert fwer["adjusted_p"][4] is None
    ranking = pd.read_csv(ranking_path, sep="\t", keep_default_na=False)
    assert len(ranking) == 8
    nonestimable = ranking.loc[
        ranking.hypothesis_id == fwer["hypothesis_ids"][4]
    ].iloc[0]
    assert nonestimable.p_interaction == "NA"
    assert nonestimable.p_adjusted_bootstrap_minp == "NA"
    assert int(nonestimable.primary_sig) == 0


@pytest.mark.parametrize("bootstrap_B", [0, -1])
def test_formal_group_api_requires_at_least_one_bootstrap(bootstrap_B):
    family, scores, expanded = _fixed_scores()
    with pytest.raises(ValueError, match="at least one"):
        run_group_scan_omnib(
            {}, family, np.arange(4.0), np.arange(4),
            hypothesis_unit="edge", family_scope="primary_only",
            transform="INT", bootstrap_B=bootstrap_B)


def test_formal_group_api_rejects_raw_transform():
    family, _scores, _expanded = _fixed_scores()
    with pytest.raises(ValueError, match="INT"):
        run_group_scan_omnib(
            {}, family, np.arange(4.0), np.arange(4),
            hypothesis_unit="edge", family_scope="primary_only",
            transform="RAW", bootstrap_B=3)


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (lambda payload: payload.__setitem__("n_sig", payload["n_sig"] + 1),
         "OMNIB_FWER_TOPLEVEL_COUNT_MISMATCH"),
        (lambda payload: payload["model_diagnostics"]["bootstrap_fwer"].__setitem__(
            "family_id", "group"), "OMNIB_FWER_FAMILY_ID_MISMATCH"),
        (lambda payload: payload["model_diagnostics"]["bootstrap_fwer"][
            "adjusted_p"].__setitem__(0, None),
         "OMNIB_FWER_ADJUSTED_P_MISSING"),
        (lambda payload: payload["model_diagnostics"]["bootstrap_fwer"].__setitem__(
            "rejected_hypothesis_ids", ["edge:corrupt"]),
         "OMNIB_FWER_THRESHOLD_HIT_MISMATCH"),
        (lambda payload: payload["model_diagnostics"]["bootstrap_fwer"].__setitem__(
            "calibrated_layers", ["edge", "group"]),
         "OMNIB_FWER_UNCALIBRATED_SECOND_PRIMARY_LAYER"),
    ],
)
def test_pure_fwer_consistency_validator_rejects_corruption(monkeypatch, mutate, expected):
    family, scores, expanded = _fixed_scores()
    monkeypatch.setattr(
        F, "score_omnib_family", lambda *args, **kwargs: (scores, expanded))
    result = run_group_scan_omnib(
        {}, family, np.arange(4.0), np.arange(4),
        hypothesis_unit="edge", family_scope="primary_only",
        transform="INT", bootstrap_B=3)
    payload = asdict(result)
    mutate(payload)
    assert expected in set(F.omnib_fwer_consistency_flags(payload))


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (lambda payload: payload.__setitem__(
            "sig", [{"hypothesis_id": "edge:corrupt"}]),
         "OMNIB_FWER_TOPLEVEL_SIG_MISMATCH"),
        (lambda payload: payload.__setitem__(
            "minp_boot_rejected", not payload["minp_boot_rejected"]),
         "OMNIB_FWER_TOPLEVEL_DECISION_MISMATCH"),
        (lambda payload: payload.__setitem__(
            "minp_boot_emp", payload["minp_boot_emp"] + 0.1),
         "OMNIB_FWER_TOPLEVEL_EMPIRICAL_P_MISMATCH"),
        (lambda payload: payload.__setitem__("minp_boot_threshold", 0.123),
         "OMNIB_FWER_TOPLEVEL_THRESHOLD_MISMATCH"),
        (lambda payload: payload.__setitem__(
            "bootstrap_B", payload["bootstrap_B"] + 1),
         "OMNIB_FWER_TOPLEVEL_BOOTSTRAP_B_MISMATCH"),
        (lambda payload: payload["model_diagnostics"]["bootstrap_fwer"].__setitem__(
            "rejected_indices", [0]),
         "OMNIB_FWER_REJECTED_INDEX_MISMATCH"),
        (lambda payload: payload["model_diagnostics"]["bootstrap_fwer"].__setitem__(
            "family_order_sha256", "0" * 64),
         "OMNIB_FWER_FAMILY_HASH_MISMATCH"),
    ],
)
def test_zero_hit_authority_fields_are_independently_audited(
        monkeypatch, mutate, expected):
    family, scores, expanded = _fixed_scores()
    monkeypatch.setattr(
        F, "score_omnib_family", lambda *args, **kwargs: (scores, expanded))
    result = run_group_scan_omnib(
        {}, family, np.arange(4.0), np.arange(4),
        hypothesis_unit="edge", family_scope="primary_only",
        transform="INT", bootstrap_B=3)
    payload = asdict(result)
    assert payload["sig"] == []
    assert F.omnib_fwer_consistency_flags(payload) == ()
    mutate(payload)
    assert expected in set(F.omnib_fwer_consistency_flags(payload))


def _forced_hit_payload(monkeypatch):
    family, scores, expanded = _fixed_scores()
    monkeypatch.setattr(
        F, "score_omnib_family", lambda *args, **kwargs: (scores, expanded))

    def forced_calibration(p_obs, p_null, *, alpha):
        adjusted = np.full(p_obs.size, 0.20)
        adjusted[:3] = 0.01
        return {
            "alpha": alpha,
            "method": "parametric_bootstrap_minp_plus_one",
            "B": 3,
            "empirical_p": 0.01,
            "threshold": 0.035,
            "threshold_comparator": "strict_less_than",
            "rejected": True,
            "rejected_local": [0, 1, 2],
            "adjusted_p_local": adjusted,
            "n_degenerate_replicates": 0,
            "degenerate_policy": "conservative",
        }

    monkeypatch.setattr(F, "bootstrap_minp_calibration", forced_calibration)
    return asdict(run_group_scan_omnib(
        {}, family, np.arange(4.0), np.arange(4),
        hypothesis_unit="edge", family_scope="primary_only",
        transform="INT", bootstrap_B=3))


@pytest.mark.parametrize("authority", ["top_level_sig", "fwer_sig", "rejected_ids"])
def test_rejected_identity_order_must_match_every_authority(monkeypatch, authority):
    payload = _forced_hit_payload(monkeypatch)
    assert F.omnib_fwer_consistency_flags(payload) == ()
    fwer = payload["model_diagnostics"]["bootstrap_fwer"]
    if authority == "top_level_sig":
        payload["sig"] = list(reversed(payload["sig"]))
    elif authority == "fwer_sig":
        fwer["sig"] = list(reversed(fwer["sig"]))
    else:
        fwer["rejected_hypothesis_ids"] = list(
            reversed(fwer["rejected_hypothesis_ids"]))
    assert "OMNIB_FWER_TOPLEVEL_SIG_MISMATCH" in set(
        F.omnib_fwer_consistency_flags(payload))


def test_rejected_full_indices_must_map_in_order_to_rejected_ids(monkeypatch):
    payload = _forced_hit_payload(monkeypatch)
    fwer = payload["model_diagnostics"]["bootstrap_fwer"]
    fwer["rejected_indices"] = list(reversed(fwer["rejected_indices"]))
    assert "OMNIB_FWER_REJECTED_INDEX_MISMATCH" in set(
        F.omnib_fwer_consistency_flags(payload))
