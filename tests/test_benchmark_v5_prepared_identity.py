"""Fixed-family and shared-preparation contracts for v5 omniB."""

import numpy as np
import pytest

import homoeogwas.omnib_family as F
from homoeogwas.group_family import MasterGroupFamily
from homoeogwas.interact import SubgenomeData


def _prepared_fixture():
    rng = np.random.default_rng(940)
    n = 52
    subdata = {}
    masks = {}
    for sub in ("A", "B", "D"):
        X = rng.integers(0, 3, size=(n, 6)).astype(float)
        subdata[sub] = SubgenomeData(
            X=X,
            gene_snp={"g0": np.arange(6)},
            samples=[f"s{i}" for i in range(n)],
            chunk=None,
        )
        masks[sub] = np.ones(6, bool)
    family = MasterGroupFamily(
        ("A", "B", "D"), ("triad",), (("g0", "g0", "g0"),)
    )
    return subdata, family, rng.normal(size=n), np.arange(n), masks


def _prepare(subdata, family, phenotype, sample_idx, masks):
    return F.prepare_omnib_design(
        subdata,
        family,
        phenotype,
        sample_idx,
        transform="INT",
        feature_seed=17,
        retained_variant_masks=masks,
        grm_method="grm_from_X",
        maf_min=0.01,
        burden_maf=0.01,
        min_snp=3,
        cap=150,
        n_pc=3,
    )


def test_normalized_svd_rank_uses_fixed_absolute_threshold():
    matrix = np.array([[1.0, 1.0], [0.0, 2.0e-11]])
    assert F._normalized_svd_rank(matrix) == 1
    assert F._normalized_svd_rank(matrix, atol=1.0e-12) == 2


def test_prepared_response_scoring_cannot_mutate_fixed_component_family():
    subdata, family, phenotype, sample_idx, masks = _prepared_fixture()
    scores, expanded = _prepare(
        subdata, family, phenotype, sample_idx, masks)
    frozen_components = scores.component_estimable.copy()
    frozen_edges = scores.edge_estimable.copy()
    responses = np.column_stack((scores.y, scores.y[::-1]))

    one = F.score_omnib_responses(
        scores, family, expanded, responses, n_jobs=1)
    four = F.score_omnib_responses(
        scores, family, expanded, responses, n_jobs=4)

    np.testing.assert_array_equal(scores.component_estimable, frozen_components)
    np.testing.assert_array_equal(scores.edge_estimable, frozen_edges)
    np.testing.assert_array_equal(
        scores.edge_estimable, scores.component_estimable.any(axis=1))
    for left, right in zip(one, four, strict=True):
        np.testing.assert_array_equal(left, right)
    assert scores.fixed_mask_sha256 == F._fixed_component_mask_sha256(scores)
    assert len(scores.fixed_mask_sha256) == 64
    assert len(scores.null_fit_sha256) == 64
    assert len(scores.prepared_design_sha256) == 64


def test_whitened_projection_rank_mismatch_aborts_without_changing_raw_mask(
    monkeypatch,
):
    subdata, family, phenotype, sample_idx, masks = _prepared_fixture()
    scores, expanded = _prepare(
        subdata, family, phenotype, sample_idx, masks)
    frozen_components = scores.component_estimable.copy()
    frozen_edges = scores.edge_estimable.copy()
    original = F._prepare_omnib_nested_designs

    def disagree(*args, **kwargs):
        prepared = list(original(*args, **kwargs))
        Qr, Qa, rank_reduced, dfn, dfd = prepared[0]
        prepared[0] = (Qr, Qa, rank_reduced, dfn + 1, dfd - 1)
        return tuple(prepared)

    scores.projection_cache.clear()
    monkeypatch.setattr(F, "_prepare_omnib_nested_designs", disagree)
    with pytest.raises(RuntimeError, match="raw/whitened component rank mismatch"):
        F._prepare_projection_cache(scores, expanded)

    np.testing.assert_array_equal(scores.component_estimable, frozen_components)
    np.testing.assert_array_equal(scores.edge_estimable, frozen_edges)


def test_native_and_conditional_paths_share_one_prepared_identity():
    subdata, family, phenotype, sample_idx, masks = _prepared_fixture()
    native, native_expanded = F.score_omnib_family(
        subdata,
        family,
        phenotype,
        sample_idx,
        transform="INT",
        feature_seed=17,
        retained_variant_masks=masks,
        bootstrap_B=0,
        n_jobs=1,
        grm_method="grm_from_X",
        maf_min=0.01,
        burden_maf=0.01,
        min_snp=3,
        cap=150,
        n_pc=3,
    )
    prepared, prepared_expanded = _prepare(
        subdata, family, phenotype, sample_idx, masks)
    conditional = F.score_omnib_responses(
        prepared,
        family,
        prepared_expanded,
        prepared.y,
        n_jobs=1,
    )
    checkpoint, checkpoint_expanded = F._prepare_checkpoint_omnib(
        subdata,
        family,
        phenotype,
        sample_idx,
        cap=150,
        n_pc=3,
        transform="INT",
        bootstrap_seed=99,
        feature_seed=17,
        n_jobs=1,
        grm_method="grm_from_X",
        maf_min=0.01,
        burden_maf=0.01,
        min_snp=3,
        covariates=None,
        retained_variant_masks=masks,
    )

    assert native_expanded == prepared_expanded == checkpoint_expanded
    assert native.prepared_design_sha256 == prepared.prepared_design_sha256
    assert native.prepared_design_sha256 == checkpoint.prepared_design_sha256
    assert native.feature_cache_sha256 == prepared.feature_cache_sha256
    assert native.fixed_mask_sha256 == prepared.fixed_mask_sha256
    np.testing.assert_array_equal(native.edge_p, conditional[0])
    np.testing.assert_array_equal(native.group_p, conditional[1])
    np.testing.assert_array_equal(
        native.edge_components_obs, conditional[2][:, :, 0])
