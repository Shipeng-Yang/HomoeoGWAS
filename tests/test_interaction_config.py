"""Canonical interaction-config normalization."""
from __future__ import annotations

import pytest

from homoeogwas.interact import validate_interact_config
from homoeogwas.interaction_config import normalize_interact_config


def _canonical_config(**interact_overrides):
    interact = {
        "mode": "group",
        "subgenomes": ["A", "B", "D"],
        "groups": "groups.tsv",
        "hypothesis_unit": "group",
        "subset_order": 2,
        "statistic": "omniB",
        "genotype": {"A": "a", "B": "b", "D": "d"},
        "snp_to_gene": {"A": "na", "B": "nb", "D": "nd"},
        "phenotype": "p.tsv",
        "sample_col": "IID",
        "trait": "trait",
        "burden": {"cap": 150, "min_snp": 2},
        "calibration": {"method": "bootstrap", "B": 2000},
    }
    interact.update(interact_overrides)
    return {"interact": interact}


def test_normalize_legacy_pairwise_to_two_copy_group():
    cfg = {"interact": {
        "mode": "pairwise", "subgenomes": ["A", "D"],
        "pairs": "pairs.tsv", "statistic": "omniB",
    }}
    got = normalize_interact_config(cfg)
    assert got["interact"]["mode"] == "group"
    assert got["interact"]["groups"] == "pairs.tsv"
    assert got["interact"]["hypothesis_unit"] == "edge"
    assert got["interact"]["subset_order"] == 2
    assert cfg["interact"]["mode"] == "pairwise"


def test_normalize_legacy_triad_to_group_primary():
    cfg = {"interact": {
        "mode": "triad", "subgenomes": ["A", "B", "D"],
        "triads": "triads.tsv", "statistic": "omniB",
    }}
    got = normalize_interact_config(cfg)
    assert got["interact"]["mode"] == "group"
    assert got["interact"]["groups"] == "triads.tsv"
    assert got["interact"]["hypothesis_unit"] == "group"


def test_normalization_is_idempotent_and_preserves_legacy_statistics():
    legacy_burden = {"interact": {
        "mode": "pairwise", "subgenomes": ["A", "D"],
        "pairs": "pairs.tsv", "statistic": "burden",
    }}
    triad3 = {"interact": {
        "mode": "triad", "subgenomes": ["A", "B", "D"],
        "triads": "triads.tsv", "statistic": "triad3",
    }}
    assert normalize_interact_config(legacy_burden)["interact"]["mode"] == "pairwise"
    assert normalize_interact_config(triad3)["interact"]["mode"] == "triad"
    normalized = normalize_interact_config(triad3)
    assert normalize_interact_config(normalized) == normalized


def test_missing_legacy_table_becomes_concrete_group_validation_error():
    cfg = {"interact": {
        "mode": "pairwise", "subgenomes": ["A", "D"], "statistic": "omniB",
        "genotype": {"A": "a", "D": "d"},
        "snp_to_gene": {"A": "na", "D": "nd"},
        "phenotype": "p.tsv", "sample_col": "IID", "trait": "trait",
    }}
    with pytest.raises(SystemExit, match=r"interact\.groups"):
        validate_interact_config(cfg)


@pytest.mark.parametrize("hypothesis_unit", ["edge", "group"])
def test_group_omnib_requires_pairwise_subset_order(hypothesis_unit):
    cfg = _canonical_config(hypothesis_unit=hypothesis_unit, subset_order=3)
    with pytest.raises(SystemExit, match="subset_order=2"):
        validate_interact_config(cfg)


def test_group_validation_accepts_three_subgenomes_and_joint_scope():
    validate_interact_config(_canonical_config())
    validate_interact_config(_canonical_config(family_scope="joint"))


def test_joint_scope_refuses_two_uncalibrated_primary_layers():
    with pytest.raises(SystemExit, match="mode=group.*bootstrap_minp"):
        validate_interact_config(_canonical_config(
            family_scope="joint", primary_multiplicity="bonferroni"))


@pytest.mark.parametrize("family_scope", ["primary_only", "joint"])
def test_every_canonical_group_run_requires_bootstrap_minp(family_scope):
    with pytest.raises(SystemExit, match="mode=group.*bootstrap_minp"):
        validate_interact_config(_canonical_config(
            family_scope=family_scope, primary_multiplicity="bonferroni"))


def test_group_omnib_rejects_raw_as_sensitivity_only():
    with pytest.raises(SystemExit, match="raw is sensitivity-only"):
        validate_interact_config(_canonical_config(primary_transform="RAW"))


def test_formal_triad3_rejects_raw_as_sensitivity_only():
    cfg = _canonical_config(
        mode="triad",
        statistic="triad3",
        triads="triads.tsv",
        primary_transform="raw",
        primary_multiplicity="bootstrap_minp",
    )
    cfg["interact"].pop("groups")
    cfg["interact"].pop("hypothesis_unit")
    cfg["interact"].pop("subset_order")
    with pytest.raises(SystemExit, match="raw is sensitivity-only"):
        validate_interact_config(cfg)


def test_legacy_pairwise_burden_still_accepts_raw_primary():
    cfg = {
        "interact": {
            "mode": "pairwise",
            "subgenomes": ["A", "D"],
            "pairs": "pairs.tsv",
            "statistic": "burden",
            "primary_transform": "RAW",
            "genotype": {"A": "a", "D": "d"},
            "snp_to_gene": {"A": "na", "D": "nd"},
            "phenotype": "p.tsv",
            "sample_col": "IID",
            "trait": "trait",
            "calibration": {"method": "permutation", "perm_B": 2000},
        }
    }
    validate_interact_config(cfg)
