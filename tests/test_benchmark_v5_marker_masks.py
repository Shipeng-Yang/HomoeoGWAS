"""Executable marker-QC and binding contracts for the v5 benchmark."""

import hashlib

import numpy as np
import pytest

import homoeogwas.interact as I
import homoeogwas.omnib_family as F
from homoeogwas.group_family import MasterGroupFamily
from homoeogwas.interact import SubgenomeData


def _mask_fixture():
    rng = np.random.default_rng(930)
    n = 40
    subdata = {}
    for sub in ("A", "D"):
        X = rng.integers(0, 3, size=(n, 6)).astype(float)
        subdata[sub] = SubgenomeData(
            X=X,
            gene_snp={"g0": np.arange(6)},
            samples=[f"s{i}" for i in range(n)],
            chunk=None,
        )
    family = MasterGroupFamily(("A", "D"), ("g",), (("g0", "g0"),))
    phenotype = rng.normal(size=n)
    masks = {
        "A": np.array([True, True, True, False, False, False]),
        "D": np.array([True, True, True, False, False, False]),
    }
    return subdata, family, phenotype, np.arange(n), masks


def _mask_sha256(mask):
    return hashlib.sha256(
        np.ascontiguousarray(mask, dtype=np.uint8).tobytes()
    ).hexdigest()


def test_retained_variant_mask_uses_inclusive_call_maf_and_mac_boundaries():
    n = 250
    X = np.zeros((n, 5), float)
    X[:5, 0] = 1.0
    X[224:, 1] = np.nan
    X[:5, 1] = 1.0
    X[225:, 2] = np.nan
    X[:5, 2] = 1.0
    X[:4, 3] = 1.0
    X[:5, 4] = 1.0

    mask, provenance = I.build_retained_variant_mask(
        X, call_rate_min=0.90, maf_min=0.01, mac_min=5,
    )

    np.testing.assert_array_equal(mask, [True, False, True, False, True])
    assert provenance["call_rate_boundary"] == "inclusive_greater_than_or_equal"
    assert provenance["maf_boundary"] == "inclusive_greater_than_or_equal"
    assert provenance["mac_boundary"] == "inclusive_greater_than_or_equal"
    assert provenance["n_variants_input"] == 5
    assert provenance["n_variants_retained"] == 3
    assert provenance["retained_variant_mask_sha256"] == _mask_sha256(mask)


def test_retained_variant_mask_rejects_non_hard_call_dosages():
    with pytest.raises(ValueError, match="hard-call A1 dosages"):
        I.build_retained_variant_mask(
            np.array([[0.0, 0.5], [1.0, 2.0]]),
            call_rate_min=0.90,
            maf_min=0.01,
            mac_min=1,
        )


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda masks: masks | {"A": np.ones(5, bool)}, "length"),
        (lambda masks: masks | {"A": np.ones(6, np.uint8)}, "boolean"),
        (lambda masks: {"A": masks["A"]}, "missing subgenomes"),
        (
            lambda masks: masks | {
                "A": {"mask": masks["A"], "sha256": "0" * 64}
            },
            "hash mismatch",
        ),
    ],
)
def test_formal_retained_variant_masks_fail_closed(mutator, message):
    subdata, family, phenotype, sample_idx, masks = _mask_fixture()
    with pytest.raises(ValueError, match=message):
        F.score_omnib_family(
            subdata,
            family,
            phenotype,
            sample_idx,
            feature_seed=17,
            retained_variant_masks=mutator(masks),
            bootstrap_B=0,
            n_jobs=1,
            grm_method="grm_from_X",
            min_snp=3,
        )


def test_supplied_mask_governs_grm_and_gene_features():
    subdata, family, phenotype, sample_idx, masks = _mask_fixture()
    scores, _ = F.score_omnib_family(
        subdata,
        family,
        phenotype,
        sample_idx,
        feature_seed=17,
        retained_variant_masks=masks,
        bootstrap_B=0,
        n_jobs=1,
        grm_method="grm_from_X",
        min_snp=3,
    )

    for sub in family.subgenomes:
        provenance = scores.grm_provenance[sub]
        assert provenance["filter_policy"] == "explicit_retained_variant_mask"
        assert provenance["n_variants_used"] == 3
        assert provenance["retained_variant_mask_sha256"] == _mask_sha256(
            masks[sub]
        )
        assert scores.feature_identity[(sub, "g0")][
            "retained_global_variant_indices"
        ] == [0, 1, 2]


def test_legacy_omitted_mask_is_labelled_maf_only():
    subdata, family, phenotype, sample_idx, _masks = _mask_fixture()
    scores, _ = F.score_omnib_family(
        subdata,
        family,
        phenotype,
        sample_idx,
        feature_seed=17,
        bootstrap_B=0,
        n_jobs=1,
        grm_method="grm_from_X",
        min_snp=3,
    )
    assert {
        provenance["filter_policy"]
        for provenance in scores.grm_provenance.values()
    } == {"legacy_maf_only"}
