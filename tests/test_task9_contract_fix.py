"""Regression tests for the Task 9 formal inference contract corrections."""

from __future__ import annotations

from pathlib import Path
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

import homoeogwas.omnib_family as F
from homoeogwas.group_family import MasterGroupFamily
from homoeogwas.interact import SubgenomeData, _build_grm


def _family_fixture(
    *, rank_stressed: bool = False, include_unestimable: bool = False,
):
    rng = np.random.default_rng(2901 if rank_stressed else 2900)
    n = 36
    subdata = {}
    for subgenome in ("A", "D"):
        block = rng.integers(0, 3, size=(n, 5)).astype(float)
        if rank_stressed:
            block[:, 1] = block[:, 0]
            block[:, 3] = block[:, 2]
            block[:, 4] = block[:, 0] + block[:, 2]
        subdata[subgenome] = SubgenomeData(
            X=block,
            gene_snp={"g0": np.arange(block.shape[1])},
            samples=[f"s{i}" for i in range(n)],
            chunk=None,
        )
    family = MasterGroupFamily(
        subgenomes=("A", "D"),
        group_ids=("g", "missing") if include_unestimable else ("g",),
        genes=(
            (("g0", "g0"), ("not_mapped", "not_mapped"))
            if include_unestimable else (("g0", "g0"),)
        ),
    )
    return subdata, family, rng.normal(size=n), np.arange(n)


@pytest.mark.parametrize("rank_stressed", [False, True])
def test_prepared_response_scorer_is_bit_exact_across_width_and_workers(
    rank_stressed,
):
    """A changed microblock/worker partition must not alter any score bit."""
    subdata, family, y, sample_idx = _family_fixture(
        rank_stressed=rank_stressed)
    scores, expanded = F.score_omnib_family(
        subdata,
        family,
        y,
        sample_idx,
        bootstrap_B=0,
        n_jobs=1,
        grm_method="grm_from_X",
        maf_min=0.01,
        min_snp=3,
    )
    responses = np.column_stack([scores.y, scores.y, scores.y])
    one = F._score_prepared_responses(
        scores, family, expanded, responses[:, :1], n_jobs=1)
    wide = F._score_prepared_responses(
        scores, family, expanded, responses, n_jobs=2)
    for one_matrix, wide_matrix in zip(one, wide, strict=True):
        assert np.array_equal(
            one_matrix,
            wide_matrix[..., :1],
            equal_nan=True,
        )
        assert np.array_equal(
            wide_matrix[..., 0],
            wide_matrix[..., 1],
            equal_nan=True,
        )


@pytest.mark.parametrize("rank_stressed", [False, True])
def test_same_response_is_bit_exact_through_observed_and_indexed_entrypoints(
    rank_stressed,
):
    """Splitting observed/indexed component or NaN handling must fail this."""
    import homoeogwas.interact as I

    subdata, family, y, sample_idx = _family_fixture(
        rank_stressed=rank_stressed, include_unestimable=True)
    scores, expanded = F.score_omnib_family(
        subdata,
        family,
        y,
        sample_idx,
        bootstrap_B=0,
        n_jobs=1,
        grm_method="grm_from_X",
        maf_min=0.01,
        min_snp=3,
    )
    null_fit = (
        scores.W,
        scores.null_covariance,
        scores.null_beta,
        scores.covariance_components,
    )
    response_list, _, _ = I.null_replicates_by_index(
        scores.null_kernels,
        scores.y,
        scores.null_design,
        indices=[7],
        base_seed=2026,
        null_fit=null_fit,
    )
    # The observed entry point owns scores.y. The indexed entry point
    # regenerates this exact response from the same frozen fit and index.
    scores.y = np.asarray(response_list[0], float)
    observed = F.score_omnib_observed(
        scores, family, expanded, n_jobs=1)
    indexed = F.score_omnib_null_indices(
        scores,
        family,
        expanded,
        [7],
        base_seed=2026,
        n_jobs=2,
        return_components=True,
    )

    for observed_matrix, indexed_matrix in zip(
        observed, indexed, strict=True,
    ):
        assert np.array_equal(
            np.isnan(observed_matrix), np.isnan(indexed_matrix))
        assert np.array_equal(
            observed_matrix, indexed_matrix, equal_nan=True)
    assert any(np.isnan(matrix).any() for matrix in observed)


def test_checkpoint_setup_never_executes_legacy_observed_scorer(
    monkeypatch, tmp_path,
):
    """A legacy numerical failure must not precede formal prepared scoring."""
    import homoeogwas.interact as I

    subdata, family, y, sample_idx = _family_fixture()

    def forbidden_legacy_score(*_args, **_kwargs):
        raise AssertionError("checkpoint executed legacy observed scorer")

    monkeypatch.setattr(I, "_omnib_components_over_Y", forbidden_legacy_score)
    result = F.run_group_scan_omnib(
        subdata,
        family,
        y,
        sample_idx,
        hypothesis_unit="edge",
        bootstrap_B=1,
        n_jobs=1,
        grm_method="grm_from_X",
        maf_min=0.01,
        min_snp=3,
        checkpoint_dir=tmp_path / "checkpoint",
        checkpoint_block_size=1,
    )
    assert result.G == 1


def test_checkpoint_estimability_is_derived_from_prepared_projections(
    monkeypatch, tmp_path,
):
    """The formal gate must not inherit the raw legacy rank decision."""
    subdata, family, y, sample_idx = _family_fixture(rank_stressed=True)

    def forbidden_legacy_gate(*_args, **_kwargs):
        raise AssertionError("checkpoint executed legacy estimability gate")

    monkeypatch.setattr(F, "_edge_design_estimable", forbidden_legacy_gate)
    result = F.run_group_scan_omnib(
        subdata,
        family,
        y,
        sample_idx,
        hypothesis_unit="edge",
        bootstrap_B=1,
        n_jobs=1,
        grm_method="grm_from_X",
        maf_min=0.01,
        min_snp=3,
        checkpoint_dir=tmp_path / "checkpoint",
        checkpoint_block_size=1,
    )
    assert result.G == 1


def test_checkpoint_observed_artifact_is_written_from_prepared_scorer(
    monkeypatch, tmp_path,
):
    """Bypassing the prepared scorer for observed data must break this test."""
    subdata, family, y, sample_idx = _family_fixture()
    original = F._score_prepared_responses
    calls = []

    def recording(scores, prepared_family, expanded, responses, *, n_jobs):
        result = original(
            scores, prepared_family, expanded, responses, n_jobs=n_jobs)
        calls.append((np.asarray(responses).copy(), result))
        return result

    monkeypatch.setattr(F, "_score_prepared_responses", recording)
    result = F.run_group_scan_omnib(
        subdata,
        family,
        y,
        sample_idx,
        hypothesis_unit="edge",
        bootstrap_B=1,
        n_jobs=1,
        grm_method="grm_from_X",
        maf_min=0.01,
        min_snp=3,
        checkpoint_dir=tmp_path / "checkpoint",
        checkpoint_block_size=1,
    )

    assert calls[0][0].shape == (sample_idx.size, 1)
    prepared_edge_p = calls[0][1][0]
    with np.load(tmp_path / "checkpoint" / "observed.npz") as observed:
        assert np.array_equal(
            observed["observed_p"], prepared_edge_p[:, 0], equal_nan=True)
    manifest = json.loads(
        (tmp_path / "checkpoint" / "manifest.json").read_text())[
            "manifest"]
    assert manifest["score_algorithm"] == F.PREPARED_SCORE_ALGORITHM
    assert result.model_diagnostics["resampling_checkpoint"][
        "score_algorithm"] == F.PREPARED_SCORE_ALGORITHM
    for provenance in result.model_diagnostics["grm_provenance"][
        "subgenomes"].values():
        assert provenance["n_variants_input"] == 5
        assert provenance["n_variants_used"] == 5
        assert len(provenance["retained_variant_mask_sha256"]) == 64


def _subgenome_for_grm(X: np.ndarray) -> SubgenomeData:
    return SubgenomeData(
        X=np.asarray(X, float),
        gene_snp={},
        samples=[f"s{i}" for i in range(len(X))],
        chunk=None,
    )


def test_build_grm_filters_maf_on_analysis_samples_with_inclusive_boundary():
    """Moving the inclusive MAF threshold must change the retained GRM set."""
    X = np.asarray([
        [0.0, 0.0, 0.0, np.nan],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 2.0],
        [2.0, 0.0, 2.0, np.nan],
        [2.0, 2.0, 2.0, 2.0],
    ])
    sd = _subgenome_for_grm(X)
    sample_idx = np.arange(4)
    at_boundary, boundary_info = _build_grm(
        sd, sample_idx, "grm_from_X", 0.25, return_provenance=True)
    above_boundary, above_info = _build_grm(
        sd, sample_idx, "grm_from_X", 0.26, return_provenance=True)

    assert boundary_info["n_variants_input"] == 4
    assert boundary_info["n_variants_used"] == 3
    assert above_info["n_variants_used"] == 2
    assert boundary_info["retained_variant_mask"] == [1, 0, 1, 1]
    assert not np.array_equal(at_boundary, above_boundary)


def test_build_grm_ignores_held_out_rows_and_imputes_analysis_mean():
    """Held-out FAM rows and their values must not enter filtering or imputation."""
    analysis = np.asarray([
        [0.0, 0.0, np.nan],
        [0.0, 1.0, 0.0],
        [2.0, 1.0, 2.0],
        [2.0, 2.0, np.nan],
    ])
    first = np.vstack([analysis, [0.0, 0.0, 0.0]])
    second = np.vstack([analysis, [2.0, np.nan, np.inf]])
    sample_idx = np.arange(4)

    first_K, first_info = _build_grm(
        _subgenome_for_grm(first), sample_idx, "grm_from_X", 0.20,
        return_provenance=True)
    second_K, second_info = _build_grm(
        _subgenome_for_grm(second), sample_idx, "grm_from_X", 0.20,
        return_provenance=True)
    complete = analysis.copy()
    complete[:, 2] = np.asarray([1.0, 0.0, 2.0, 1.0])
    complete_K, _ = _build_grm(
        _subgenome_for_grm(complete), sample_idx, "grm_from_X", 0.20,
        return_provenance=True)

    np.testing.assert_array_equal(first_K, second_K)
    np.testing.assert_array_equal(first_K, complete_K)
    assert first_info == second_info
    assert first_info["retained_variant_mask_sha256"] == (
        second_info["retained_variant_mask_sha256"])


def test_build_grm_refuses_zero_surviving_analysis_variants():
    """A declared GRM MAF filter must fail instead of dividing by zero."""
    sd = _subgenome_for_grm([
        [0.0, np.nan, np.inf],
        [0.0, np.nan, np.nan],
        [0.0, np.nan, -np.inf],
        [0.0, np.nan, np.nan],
    ])
    with pytest.raises(ValueError, match="zero variants survive.*maf_min=0.1"):
        _build_grm(
            sd,
            np.arange(4),
            "grm_from_X",
            0.1,
            return_provenance=True,
        )


def _write_formal_identity_fixture(
    tmp_path, *, source=None, runtime=None, manifest_schema=None,
):
    from homoeogwas.formal_provenance import PRE_RUN_MANIFEST_SCHEMA

    source = source or {
        "git_commit": "1" * 40,
        "git_tree": "2" * 40,
        "package_source_sha256": "3" * 64,
        "source_clean": True,
    }
    runtime = runtime or {
        "homoeogwas": "0.test",
        "python": "3.test",
        "numpy": "2.test",
        "scipy": "1.test",
        "blas_thread_policy": {
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        },
    }
    out_dir = tmp_path / "out"
    manifest_path = out_dir / "provenance" / "pre_run_manifest.json"
    config_path = out_dir / "configs" / "interact.generated.group.omnib.yaml"
    config_path.parent.mkdir(parents=True)
    manifest_path.parent.mkdir(parents=True)
    cfg = {
        "interact": {"mode": "group", "statistic": "omniB"},
        "provenance": {
            "pre_run_manifest": str(manifest_path),
            "blas_thread_policy": runtime["blas_thread_policy"],
        },
        "outputs": {"out_dir": str(out_dir)},
    }
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    config_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
    manifest = {
        "schema": manifest_schema or PRE_RUN_MANIFEST_SCHEMA,
        "config": {"sha256": config_sha},
        "source_identity": source,
        "runtime_fingerprint": runtime,
    }
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
    return config_path, cfg, manifest_path, source, runtime


@pytest.mark.parametrize(
    "manifest_schema",
    [
        "homoeogwas-group-omnib-pre-run-manifest-v1",
        "homoeogwas-wheat-f2143-edge-pre-run-manifest-v2",
    ],
)
def test_formal_launch_accepts_generic_and_legacy_manifest_schemas(
    tmp_path, monkeypatch, manifest_schema,
):
    """Formal provenance must be species-neutral without orphaning wheat."""
    from homoeogwas import formal_provenance as P

    config_path, cfg, _manifest, source, runtime = (
        _write_formal_identity_fixture(
            tmp_path, manifest_schema=manifest_schema))
    monkeypatch.setattr(P, "capture_source_identity", lambda: source)
    monkeypatch.setattr(P, "runtime_fingerprint", lambda _policy: runtime)
    for name in runtime["blas_thread_policy"]:
        monkeypatch.setenv(name, "1")

    verified = P.verify_formal_launch(config_path, cfg)
    assert verified.checkpoint_context["pre_run_manifest_schema"] == (
        manifest_schema)


def test_formal_checkpoint_without_pre_run_manifest_fails_closed(tmp_path):
    """Deleting provenance must not downgrade a checkpoint run to legacy."""
    from homoeogwas import formal_provenance as P

    config_path = tmp_path / "formal-checkpoint.yaml"
    cfg = {
        "interact": {
            "mode": "group",
            "statistic": "omniB",
            "calibration": {
                "method": "bootstrap",
                "B": 2000,
                "checkpoint": {
                    "enabled": True,
                    "root": str(tmp_path / "checkpoint"),
                },
            },
        },
    }
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    with pytest.raises(
        P.FormalLaunchError,
        match="checkpoint.*provenance.pre_run_manifest",
    ):
        P.verify_formal_launch(config_path, cfg)


def test_explicit_noncheckpoint_config_retains_legacy_provenance_opt_out(
    tmp_path,
):
    """Only an explicitly non-checkpoint path may omit formal provenance."""
    from homoeogwas import formal_provenance as P

    config_path = tmp_path / "legacy-noncheckpoint.yaml"
    cfg = {
        "interact": {
            "mode": "group",
            "statistic": "omniB",
            "calibration": {"method": "bootstrap", "B": 19},
        },
    }
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    assert P.verify_formal_launch(config_path, cfg) is None


def test_formal_launch_rejects_changed_raw_config_sha(tmp_path, monkeypatch):
    """Changing even non-semantic YAML bytes must invalidate the preparation."""
    from homoeogwas import formal_provenance as P

    config_path, cfg, _manifest, source, runtime = (
        _write_formal_identity_fixture(tmp_path))
    monkeypatch.setattr(P, "capture_source_identity", lambda: source)
    monkeypatch.setattr(P, "runtime_fingerprint", lambda _policy: runtime)
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "1")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("MKL_NUM_THREADS", "1")
    monkeypatch.setenv("NUMEXPR_NUM_THREADS", "1")
    config_path.write_bytes(config_path.read_bytes() + b"\n")

    with pytest.raises(P.FormalLaunchError, match="raw config SHA-256 mismatch"):
        P.verify_formal_launch(config_path, cfg)


@pytest.mark.parametrize(
    "field,changed",
    [
        ("git_commit", "a" * 40),
        ("git_tree", "b" * 40),
        ("package_source_sha256", "c" * 64),
    ],
)
def test_formal_launch_rejects_changed_source_identity(
    tmp_path, monkeypatch, field, changed,
):
    """A changed executable commit, tree, or package hash is a hard failure."""
    from homoeogwas import formal_provenance as P

    config_path, cfg, _manifest, source, runtime = (
        _write_formal_identity_fixture(tmp_path))
    actual = dict(source)
    actual[field] = changed
    monkeypatch.setattr(P, "capture_source_identity", lambda: actual)
    monkeypatch.setattr(P, "runtime_fingerprint", lambda _policy: runtime)
    for name in runtime["blas_thread_policy"]:
        monkeypatch.setenv(name, "1")

    with pytest.raises(P.FormalLaunchError, match=field):
        P.verify_formal_launch(config_path, cfg)


def test_formal_launch_rejects_dirty_source_before_genotype_loading(
    tmp_path, monkeypatch,
):
    """A dirty formal source tree must abort before the first BED is loaded."""
    from homoeogwas import formal_provenance as P
    import homoeogwas.interact as I

    config_path, cfg, _manifest, source, runtime = (
        _write_formal_identity_fixture(tmp_path))
    dirty = dict(source, source_clean=False)
    monkeypatch.setattr(P, "capture_source_identity", lambda: dirty)
    monkeypatch.setattr(P, "runtime_fingerprint", lambda _policy: runtime)
    for name in runtime["blas_thread_policy"]:
        monkeypatch.setenv(name, "1")
    loaded = {"called": False}

    def forbidden_load(*_args, **_kwargs):
        loaded["called"] = True
        raise AssertionError("genotype load must be unreachable")

    monkeypatch.setattr(I, "_load_subgenome", forbidden_load)
    rc = I.cmd_interact(SimpleNamespace(
        config=str(config_path), out_dir=None, n_jobs=1))
    assert rc == 1
    assert loaded["called"] is False


def test_formal_launch_rejects_pre_run_config_mismatch(tmp_path, monkeypatch):
    """The manifest cannot silently bind a different generated config."""
    from homoeogwas import formal_provenance as P

    config_path, cfg, manifest_path, source, runtime = (
        _write_formal_identity_fixture(tmp_path))
    manifest = json.loads(manifest_path.read_text())
    manifest["config"]["sha256"] = "f" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(P, "capture_source_identity", lambda: source)
    monkeypatch.setattr(P, "runtime_fingerprint", lambda _policy: runtime)
    for name in runtime["blas_thread_policy"]:
        monkeypatch.setenv(name, "1")

    with pytest.raises(P.FormalLaunchError, match="pre-run/config mismatch"):
        P.verify_formal_launch(config_path, cfg)


def test_passed_launch_writes_attestation_and_returns_stable_context(
    tmp_path, monkeypatch,
):
    """Only a fully matched launch may emit passed attestation/context data."""
    from homoeogwas import formal_provenance as P

    config_path, cfg, manifest_path, source, runtime = (
        _write_formal_identity_fixture(tmp_path))
    monkeypatch.setattr(P, "capture_source_identity", lambda: source)
    monkeypatch.setattr(P, "runtime_fingerprint", lambda _policy: runtime)
    for name in runtime["blas_thread_policy"]:
        monkeypatch.setenv(name, "1")

    verified = P.verify_formal_launch(config_path, cfg)
    attestation = json.loads(Path(verified.attestation_path).read_text())
    assert attestation["status"] == "passed"
    assert all(check["passed"] for check in attestation["checks"].values())
    assert verified.checkpoint_context == {
        "raw_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "pre_run_manifest_sha256": hashlib.sha256(
            manifest_path.read_bytes()).hexdigest(),
        "pre_run_manifest_schema": P.PRE_RUN_MANIFEST_SCHEMA,
        "source_identity": source,
        "runtime_fingerprint": runtime,
    }


def test_canonical_cli_forwards_verified_context_to_checkpoint_manifest(
    tmp_path, monkeypatch,
):
    """Dropping verified executable identity before checkpointing is a bug."""
    import homoeogwas.interact as I

    groups = tmp_path / "groups.tsv"
    groups.write_text(
        "group_id\tgene_A\tgene_D\nfamily\ta\td\n", encoding="utf-8")
    phenotype = tmp_path / "phenotype.tsv"
    phenotype.write_text(
        "sample\ttrait\n" + "".join(f"s{i}\t{i}\n" for i in range(12)),
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"
    cfg = {
        "interact": {
            "mode": "group",
            "subgenomes": ["A", "D"],
            "groups": str(groups),
            "statistic": "omniB",
            "hypothesis_unit": "edge",
            "subset_order": 2,
            "family_scope": "primary_only",
            "primary_transform": "INT",
            "primary_multiplicity": "bootstrap_minp",
            "genotype": {"A": "a", "D": "d"},
            "snp_to_gene": {"A": "na", "D": "nd"},
            "phenotype": str(phenotype),
            "sample_col": "sample",
            "trait": "trait",
            "burden": {"cap": 150, "min_snp": 3, "maf_min": 0.01},
            "grm": {"method": "grm_from_X", "maf_min": 0.01,
                    "scope": "all_subgenomes"},
            "calibration": {
                "method": "bootstrap", "B": 19, "seed": 2026,
                "qa_only": True,
                "checkpoint": {"enabled": True, "root": str(out_dir / "cp"),
                               "block_size": 5},
            },
        },
        "outputs": {"out_dir": str(out_dir), "full_ranking": True,
                    "plots": False},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    context = {"raw_config_sha256": "a" * 64}
    monkeypatch.setattr(
        I,
        "verify_formal_launch",
        lambda *_args, **_kwargs: SimpleNamespace(checkpoint_context=context),
        raising=False,
    )
    monkeypatch.setattr(I, "preflight_interact", lambda *_args, **_kwargs: [])
    samples = [f"s{i}" for i in range(12)]
    monkeypatch.setattr(
        I, "_load_subgenome",
        lambda *_args, **_kwargs: SimpleNamespace(samples=samples))
    captured = {}

    def fake_scan(*_args, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            trait="", G=1, min_p=0.2, lambda_gc_obs=1.0,
            bonferroni_alpha=0.05, n_sig=0, sig=[], top=[], covariates=None,
            model_diagnostics={
                "bootstrap_fwer": {"family_id": "edge"},
                "family_provenance": {
                    "group_family_sha256": "g" * 64,
                    "edge_family_sha256": "e" * 64,
                    "n_groups_raw": 1,
                    "n_unique_edges": 1,
                },
            },
        )

    monkeypatch.setattr(I, "run_group_scan_omnib", fake_scan)
    assert I.cmd_interact(SimpleNamespace(
        config=str(config_path), out_dir=None, n_jobs=1)) == 0
    assert captured["checkpoint_manifest_context"] == context
