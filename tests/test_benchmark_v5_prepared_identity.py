"""Fixed-family and shared-preparation contracts for v5 omniB."""

import numpy as np
import pytest

import homoeogwas.omnib_family as F
from homoeogwas.group_family import MasterGroupFamily
from homoeogwas.interact import SubgenomeData
from homoeogwas.io import GenoChunk


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


def _legacy_chunk_fixture():
    rng = np.random.default_rng(941)
    n = 48
    subdata = {}
    for sub in ("A", "D"):
        X = rng.integers(0, 3, size=(n, 8)).astype(float)
        samples = np.asarray([f"s{i}" for i in range(n)], dtype=object)
        chunk = GenoChunk(
            samples=samples,
            variant_ids=np.asarray([f"{sub}{j}" for j in range(8)], dtype=object),
            chrom=np.asarray([sub] * 8, dtype=object),
            pos=np.arange(8, dtype=np.int64),
            dosage=X,
        )
        subdata[sub] = SubgenomeData(
            X=X,
            gene_snp={"g0": np.arange(8)},
            samples=samples.tolist(),
            chunk=chunk,
        )
    family = MasterGroupFamily(("A", "D"), ("pair",), (("g0", "g0"),))
    return subdata, family, rng.normal(size=n), np.arange(n)


def test_normalized_svd_rank_uses_fixed_absolute_threshold():
    matrix = np.array([[1.0, 1.0], [0.0, 2.0e-11]])
    assert F._normalized_svd_rank(matrix) == 1
    assert F._normalized_svd_rank(matrix, atol=1.0e-12) == 2


def test_near_dependent_added_columns_do_not_gain_rank_after_residualization():
    n = 48
    x = np.linspace(-1.0, 1.0, n)[:, None]
    y = np.ones((n, 1)) + 1.0e-12 * np.sin(np.arange(n))[:, None]
    features_x = (x, x, x)
    features_y = (y, y, y)
    covariates = np.ones((n, 1))

    raw_rank, raw_dfn, raw_dfd = F._component_design_signature(
        features_x, features_y, covariates
    )
    prepared = F._prepare_omnib_nested_designs(
        np.eye(n), covariates, features_x, features_y
    )
    whitened = np.asarray(
        [(rank, dfn, dfd) for _Qr, _Qa, rank, dfn, dfd in prepared]
    )

    np.testing.assert_array_equal(raw_dfn, np.zeros(3, dtype=int))
    np.testing.assert_array_equal(
        whitened,
        np.column_stack((raw_rank, raw_dfn, raw_dfd)),
    )


def test_default_compute_grm_maf_prepared_path_retains_filter_identity():
    subdata, family, phenotype, sample_idx = _legacy_chunk_fixture()

    scores, expanded = F.score_omnib_family(
        subdata,
        family,
        phenotype,
        sample_idx,
        feature_seed=17,
        bootstrap_B=0,
        n_jobs=1,
        grm_method="compute_grm_maf",
        min_snp=3,
    )

    assert len(expanded.edges) == 1
    for provenance in scores.grm_provenance.values():
        assert provenance["filter_policy"] == "legacy_full_chunk_maf_only"
        assert len(provenance["retained_variant_mask_sha256"]) == 64
    assert scores.prepared_design_identity["retained_variant_mask_sha256"] == {
        sub: scores.grm_provenance[sub]["retained_variant_mask_sha256"]
        for sub in family.subgenomes
    }


def test_prepared_design_hash_binds_score_implementation(monkeypatch):
    subdata, family, phenotype, sample_idx, masks = _prepared_fixture()
    first, _ = _prepare(subdata, family, phenotype, sample_idx, masks)

    assert first.prepared_design_identity["implementation"] == {
        "score_algorithm": F.PREPARED_SCORE_ALGORITHM,
        "score_microblock_size": F.INDEXED_SCORE_MICROBLOCK,
        "component_rank_algorithm": F.COMPONENT_RANK_ALGORITHM,
        "component_rank_atol": F.COMPONENT_RANK_ATOL,
        "feature_seed_scheme": F.FEATURE_SEED_SCHEME,
    }
    monkeypatch.setattr(
        F, "PREPARED_SCORE_ALGORITHM", "homoeogwas-test-different-scorer"
    )
    second, _ = _prepare(subdata, family, phenotype, sample_idx, masks)
    assert first.prepared_design_sha256 != second.prepared_design_sha256


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


def test_response_banks_wider_than_microblock_are_partition_and_worker_invariant():
    subdata, family, phenotype, sample_idx, masks = _prepared_fixture()
    scores, expanded = _prepare(subdata, family, phenotype, sample_idx, masks)
    responses = np.random.default_rng(942).normal(size=(phenotype.size, 31))

    serial = F.score_omnib_responses(
        scores, family, expanded, responses, n_jobs=1
    )
    parallel = F.score_omnib_responses(
        scores, family, expanded, responses, n_jobs=4
    )
    partitioned_parts = [
        F.score_omnib_responses(
            scores, family, expanded, responses[:, start:stop], n_jobs=2
        )
        for start, stop in ((0, 7), (7, 26), (26, 31))
    ]
    partitioned = tuple(
        np.concatenate([part[index] for part in partitioned_parts], axis=-1)
        for index in range(3)
    )
    order = np.arange(responses.shape[1])[::-1]
    reversed_scores = F.score_omnib_responses(
        scores, family, expanded, responses[:, order], n_jobs=3
    )

    for serial_values, parallel_values, partitioned_values, reversed_values in zip(
        serial, parallel, partitioned, reversed_scores, strict=True
    ):
        np.testing.assert_array_equal(serial_values, parallel_values)
        np.testing.assert_array_equal(serial_values, partitioned_values)
        np.testing.assert_array_equal(serial_values, reversed_values[..., order])


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
