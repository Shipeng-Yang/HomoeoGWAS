"""Opt-in smooth PC null-variance bootstrap (HETERO-FIX); homoscedastic default unchanged."""

from __future__ import annotations

import json

import numpy as np
import pytest

import homoeogwas.interact as I
import homoeogwas.omnib_family as F
from tests.test_interact_batch import formal_runtime  # noqa: F401 (autouse fixture)
from tests.test_omnib_panel_context import SETTINGS, panel_fixture

B = 19


def _null_fit(n, rng):
    A = rng.normal(size=(n, n))
    V = A @ A.T / n + np.eye(n)
    C = np.ones((n, 1))
    return C, (np.eye(n), V, np.array([0.3]), {})


def test_weighted_draws_scale_the_same_stream():
    rng = np.random.default_rng(3)
    n = 30
    C, fit = _null_fit(n, rng)
    w = rng.uniform(0.5, 2.0, n)
    plain, _, _ = I.null_replicates_by_index(
        {"A": np.eye(n)}, np.zeros(n), C, indices=range(5), base_seed=11, null_fit=fit)
    ones, _, _ = I.null_replicates_by_index(
        {"A": np.eye(n)}, np.zeros(n), C, indices=range(5), base_seed=11, null_fit=fit,
        variance_weights=np.ones(n))
    weighted, _, _ = I.null_replicates_by_index(
        {"A": np.eye(n)}, np.zeros(n), C, indices=range(5), base_seed=11, null_fit=fit,
        variance_weights=w)
    mean = C @ fit[2]
    for a, b, c in zip(plain, ones, weighted):
        np.testing.assert_array_equal(a, b)
        np.testing.assert_allclose(c - mean, np.sqrt(w) * (a - mean), rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("bad", [np.zeros(30), -np.ones(30), np.ones(29), np.full(30, np.nan)])
def test_invalid_weights_are_rejected(bad):
    rng = np.random.default_rng(3)
    C, fit = _null_fit(30, rng)
    with pytest.raises(ValueError, match="variance_weights"):
        I.null_replicates_by_index(
            {"A": np.eye(30)}, np.zeros(30), C, indices=[0], base_seed=1, null_fit=fit,
            variance_weights=bad)


def test_gamma_log_variance_recovers_a_known_variance_function():
    rng = np.random.default_rng(5)
    n = 40000
    X = np.column_stack([np.ones(n), rng.normal(size=(n, 2))])
    truth = np.exp(X @ np.array([0.2, 0.5, -0.3]))
    r2 = truth * rng.standard_normal(n) ** 2
    weights, coef, iterations, converged = F._gamma_log_variance(r2, X)
    assert converged and iterations < F.SMOOTH_VARIANCE_MAX_ITERATIONS
    np.testing.assert_allclose(coef[1:], [0.5, -0.3], atol=0.03)
    assert abs(weights.mean() - 1.0) < 1e-12
    flat, _, _, _ = F._gamma_log_variance(rng.standard_normal(n) ** 2, X)
    assert flat.max() / flat.min() < 1.1


def _run(root, variance, seed=2026):
    subdata, family, responses, sample_idx = panel_fixture()
    return F.run_group_scan_omnib(
        subdata, family, responses[0], sample_idx,
        hypothesis_unit="group", family_scope="primary_only",
        transform="INT", bootstrap_B=B, bootstrap_seed=seed, n_jobs=2,
        checkpoint_dir=root, checkpoint_block_size=5, alpha=0.05,
        inferential=True, null_variance=variance, **SETTINGS,
    )


def test_smooth_variance_is_recorded_bound_and_resumable(tmp_path):
    plain = _run(tmp_path / "plain", "homoscedastic")
    smooth = _run(tmp_path / "smooth", "smooth_pc4")
    again = _run(tmp_path / "smooth", "smooth_pc4")
    pd, sd, ad = (r.model_diagnostics for r in (plain, smooth, again))
    assert "null_variance_model" not in pd
    assert pd["bootstrap_fwer"]["method"] == "parametric_bootstrap_minp_plus_one"
    model = sd["null_variance_model"]
    assert model["model"] == "smooth_pc4" and len(model["coefficients"]) == 5
    assert sd["bootstrap_fwer"]["method"] == (
        "smooth_variance_parametric_bootstrap_minp_plus_one")
    assert (sd["resampling_checkpoint"]["manifest_id"]
            != pd["resampling_checkpoint"]["manifest_id"])
    manifest = json.loads((tmp_path / "smooth" / "manifest.json").read_text())
    assert manifest["manifest"]["null_variance"] == model
    assert sd["bootstrap_fwer"]["observed_p"] == pd["bootstrap_fwer"]["observed_p"]
    assert sd["resampling_checkpoint"]["primary_null_p_sha256"] != (
        pd["resampling_checkpoint"]["primary_null_p_sha256"])
    assert json.dumps(ad["bootstrap_fwer"], sort_keys=True, default=str) == json.dumps(
        sd["bootstrap_fwer"], sort_keys=True, default=str)
    assert ad["resampling_checkpoint"] == sd["resampling_checkpoint"]


def test_smooth_variance_requires_checkpoint_and_known_name(tmp_path):
    with pytest.raises(ValueError, match="checkpointed"):
        _run(None, "smooth_pc4")
    with pytest.raises(ValueError, match="null_variance must be"):
        _run(tmp_path / "x", "wild")


def _config(**calibration):
    return {
        "interact": {
            "mode": "group", "statistic": "omniB", "subgenomes": ["A", "D"],
            "hypothesis_unit": "group", "subset_order": 2,
            "family_scope": "primary_only", "primary_transform": "INT",
            "primary_multiplicity": "bootstrap_minp",
            "groups": "g.tsv", "genotype": {"A": "a", "D": "d"},
            "snp_to_gene": {"A": "a.npz", "D": "d.npz"}, "phenotype": "p.tsv",
            "trait": "t", "sample_col": "sample",
            "calibration": {"method": "bootstrap", "B": 19, "qa_only": True, **calibration},
        },
        "outputs": {"out_dir": "out"},
    }


def test_config_validation_of_null_variance():
    with pytest.raises(SystemExit, match="null_variance must be"):
        I.validate_interact_config(_config(null_variance="wild"))
    with pytest.raises(SystemExit, match="requires checkpoint"):
        I.validate_interact_config(_config(null_variance="smooth_pc4"))
    I.validate_interact_config(_config(
        null_variance="smooth_pc4",
        checkpoint={"enabled": True, "block_size": 5, "root": "/tmp/ckpt"}))


def _harness():
    import importlib.util
    from pathlib import Path

    path = Path("/mnt/7302share/fast_ysp/U7_GWAS/tasks/HETERO-FIX/pilot-v2/pilot_v2.py")
    if not path.exists():
        pytest.skip("HETERO-FIX harness not available on this machine")
    spec = importlib.util.spec_from_file_location("hetero_pilot_v2", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_engine_variance_fit_is_bit_identical_to_validated_harness():
    harness = _harness()
    rng = np.random.default_rng(17)
    n = 120
    kernels = {}
    for sub in ("A", "B", "D"):
        G = rng.normal(size=(n, 300))
        kernels[sub] = G @ G.T / 300
    X_engine = F._null_variance_design(kernels, F.SMOOTH_VARIANCE_PCS)
    X_harness = harness.variance_pcs(kernels, harness.N_VARIANCE_PCS)
    np.testing.assert_array_equal(X_engine, X_harness)
    r2 = np.exp(X_engine @ np.array([0.0, 0.4, -0.2, 0.1, 0.0])) * rng.standard_normal(n) ** 2
    w_engine, coef_engine, _, _ = F._gamma_log_variance(r2, X_engine)
    w_harness, coef_harness = harness.gamma_log_fit(r2, X_harness)
    np.testing.assert_array_equal(w_engine, w_harness)
    np.testing.assert_array_equal(coef_engine, coef_harness)


def test_nonconverged_fit_is_recorded_and_matches_harness(monkeypatch):
    harness = _harness()
    rng = np.random.default_rng(5)
    X = np.column_stack([np.ones(36), rng.normal(size=(36, 4))])
    r2 = np.exp(X @ np.array([0.0, 0.3, -0.3, 0.2, 0.1])) * rng.standard_normal(36) ** 2
    monkeypatch.setattr(F, "SMOOTH_VARIANCE_MAX_ITERATIONS", 3)
    weights, coef, iterations, converged = F._gamma_log_variance(r2, X)
    assert iterations == 3 and converged is False
    expected, _ = harness.gamma_log_fit(r2, X, iterations=3)
    np.testing.assert_array_equal(weights, expected)


def test_nonfinite_weights_fail_closed():
    X = np.column_stack([np.ones(4), [0.0, 1.0, 2.0, 3.0]])
    r2 = np.array([1.0, 1.0, 1.0, np.inf])
    with np.errstate(all="ignore"), pytest.raises(RuntimeError, match="non-finite"):
        F._gamma_log_variance(r2, X)


@pytest.mark.parametrize(("method", "model", "ok"), [
    ("parametric_bootstrap_minp_plus_one", None, True),
    ("smooth_variance_parametric_bootstrap_minp_plus_one", {"model": "smooth_pc4"}, True),
    ("smooth_variance_parametric_bootstrap_minp_plus_one", None, False),
    ("parametric_bootstrap_minp_plus_one", {"model": "smooth_pc4"}, False),
])
def test_audit_pairs_method_with_variance_model(method, model, ok):
    from homoeogwas import audit

    diagnostics = {"bootstrap_fwer": {"method": method}}
    if model is not None:
        diagnostics["null_variance_model"] = model
    assert audit._omnib_fwer_method_ok(diagnostics) is ok


def test_smooth_cli_result_passes_audit(tmp_path):
    from types import SimpleNamespace

    from homoeogwas import audit
    from tests.test_grm_cache import _write_panel
    from tests.test_interact_batch import _config, _phenotypes, _write_config

    panel = _write_panel(tmp_path / "panel")
    (pheno,) = _phenotypes(panel, tmp_path, 1)
    cfg = _config(panel, pheno, tmp_path / "out", 2026, benchmark=False)
    cfg["interact"]["calibration"]["null_variance"] = "smooth_pc4"
    config = _write_config(tmp_path / "cfg.yaml", cfg)
    assert I.cmd_interact(SimpleNamespace(config=str(config), out_dir=None, n_jobs=1)) == 0
    record = audit.audit_result(tmp_path / "out" / "interact_trait.json")
    codes = [flag.code if hasattr(flag, "code") else flag["code"] for flag in record.flags]
    assert "OMNIB_FWER_METHOD_MISMATCH" not in codes
    payload = __import__("json").loads((tmp_path / "out" / "interact_trait.json").read_text())
    model = payload["results"]["INT"]["model_diagnostics"]["null_variance_model"]
    assert isinstance(model["converged"], bool) and model["iterations"] >= 1
    severities = [flag.severity if hasattr(flag, "severity") else flag["severity"] for flag in record.flags]
    assert not any(level in ("error", "fatal") for level in severities)
