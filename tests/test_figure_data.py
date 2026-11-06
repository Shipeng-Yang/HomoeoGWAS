"""Result fields that the output figures read: bootstrap null minima, per-edge encoding
P values of reported groups and the smooth_pc4 variance profile."""

from __future__ import annotations

import numpy as np
import pytest

import homoeogwas.interact as I
import homoeogwas.omnib_family as F
from tests.test_interact_batch import formal_runtime  # noqa: F401 (autouse fixture)
from tests.test_omnib_panel_context import SETTINGS, panel_fixture

B = 19


def _run(root, variance):
    subdata, family, responses, sample_idx = panel_fixture()
    return F.run_group_scan_omnib(
        subdata, family, responses[0], sample_idx,
        hypothesis_unit="group", family_scope="primary_only",
        transform="INT", bootstrap_B=B, bootstrap_seed=2026, n_jobs=2,
        checkpoint_dir=root, checkpoint_block_size=5, alpha=0.05,
        inferential=True, null_variance=variance, **SETTINGS,
    )


def test_calibration_reports_the_null_minima_it_thresholds():
    rng = np.random.default_rng(0)
    p_obs = rng.uniform(1e-4, 1, size=6)
    p_null = rng.uniform(size=(6, 40))
    p_null[2, 7] = np.nan
    cal = F.bootstrap_minp_calibration(p_obs, p_null, alpha=0.25)
    minima = np.asarray(cal["null_minima"])
    expected = np.where(np.isfinite(p_null), p_null, np.inf).min(axis=0)
    expected[7] = 0.0
    np.testing.assert_array_equal(minima, expected)
    k = int(np.floor(0.25 * 41))
    assert cal["threshold"] == np.sort(minima)[k - 1]


@pytest.mark.parametrize("variance", ["homoscedastic", "smooth_pc4"])
def test_group_scan_records_null_minima_and_edge_components(tmp_path, variance):
    result = _run(tmp_path, variance)
    fwer = result.model_diagnostics["bootstrap_fwer"]
    minima = fwer["null_minima"]
    assert len(minima) == B and all(0.0 <= value <= 1.0 for value in minima)
    k = int(np.floor(0.05 * (B + 1)))
    if fwer["threshold"] is not None:
        assert fwer["threshold"] == sorted(minima)[k - 1]
    reported = list(result.top) + list(result.sig or [])
    assert reported
    for record in reported:
        detail = record["edge_components"]
        assert [edge["edge_id"] for edge in detail] == [
            edge["edge_id"] for edge in record["edge_localization"]]
        for edge in detail:
            assert set(edge["component_p"]) == set(F.OMNIB_COMPONENT_NAMES)


def test_smooth_profile_is_reported_outside_the_manifest_bound_model(tmp_path):
    result = _run(tmp_path, "smooth_pc4")
    diagnostics = result.model_diagnostics
    profile = diagnostics["null_variance_profile"]
    n = len(profile["weights"])
    assert n == len(profile["pc1"]) == len(profile["pc2"]) == len(profile["squared_residuals"])
    assert abs(np.mean(profile["weights"]) - 1.0) < 1e-9
    assert set(diagnostics["null_variance_model"]) == {
        "model", "family", "design", "coefficients", "iterations", "converged",
        "weights", "weight_range"}
    plain = _run(tmp_path / "plain", "homoscedastic").model_diagnostics
    assert "null_variance_profile" not in plain


def test_figure_fields_stay_out_of_the_checkpoint_manifest(tmp_path):
    """The figure fields are reported only; the checkpoint manifest and the
    threshold of the panel fixture are unchanged."""
    expected = {"homoscedastic": 0.0216370776442113, "smooth_pc4": 0.022746404360284744}
    for variance, threshold in expected.items():
        root = tmp_path / variance
        diagnostics = I._json_safe(_run(root, variance).model_diagnostics)
        # last-bit differences between numpy builds are not a change in the threshold
        assert diagnostics["bootstrap_fwer"]["threshold"] == pytest.approx(threshold, rel=1e-12)
        manifest = (root / "manifest.json").read_text()
        for field in ("null_minima", "edge_components", "null_variance_profile",
                      "squared_residuals"):
            assert field not in manifest
