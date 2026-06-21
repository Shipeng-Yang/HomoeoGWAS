"""Tests for the parametric sequencing-depth design heuristic."""
from __future__ import annotations

import json

import numpy as np
import pytest

from homoeogwas import design_depth as dd


# ---- baked validated density curve -----------------------------------------
def test_pgene_clamps_and_monotone():
    assert dd.pgene_at_lambda(0.0) == 0.0
    assert dd.pgene_at_lambda(-5.0) == 0.0                     # negative -> 0
    # saturates above the grid at the full-WGS value
    assert dd.pgene_at_lambda(1e6) == pytest.approx(dd._PGENE_GRID[-1])
    assert dd.pgene_at_lambda(dd.WHEAT_LAMBDA_FULL) == pytest.approx(0.8885, abs=1e-3)
    xs = np.linspace(0, 60, 50)
    ys = [dd.pgene_at_lambda(x) for x in xs]
    assert all(b >= a - 1e-12 for a, b in zip(ys, ys[1:], strict=False))     # non-decreasing


# ---- per-sample coverage callability ---------------------------------------
def test_site_callable_prob_limits_and_monotone():
    assert dd.site_callable_prob(0.0, 0.85, 4) == 0.0
    assert dd.site_callable_prob(1000.0, 0.85, 4) == pytest.approx(1.0, abs=1e-6)
    ds = np.linspace(0, 40, 60)
    ps = [dd.site_callable_prob(d, 0.85, 4) for d in ds]
    assert all(b >= a - 1e-12 for a, b in zip(ps, ps[1:], strict=False))
    # higher min_dp -> lower callability at the same depth
    assert dd.site_callable_prob(6, 0.85, 8) < dd.site_callable_prob(6, 0.85, 4)


# ---- retention fraction -----------------------------------------------------
def test_retention_fraction_limits_and_monotone():
    assert dd.retention_fraction(0.0, dd.MARGINAL) == 0.0
    assert dd.retention_fraction(1000.0, dd.MARGINAL) == pytest.approx(1.0, abs=1e-6)
    ds = np.linspace(0, 50, 80)
    fs = [dd.retention_fraction(d, dd.MARGINAL) for d in ds]
    assert all(b >= a - 1e-12 for a, b in zip(fs, fs[1:], strict=False))


def test_interaction_needs_more_depth_than_marginal():
    # stricter min_dp + lower mappability -> interaction retains less at any depth
    for d in (3, 5, 8, 12, 20):
        assert (dd.retention_fraction(d, dd.INTERACTION)
                <= dd.retention_fraction(d, dd.MARGINAL) + 1e-12)


# ---- G prediction -----------------------------------------------------------
def test_predict_G_capped_and_saturates():
    g_full = 1759.0
    r0 = dd.predict_G(0.5, g_full=g_full, lambda_full=dd.WHEAT_LAMBDA_FULL,
                      mode=dd.MARGINAL)
    rhi = dd.predict_G(200.0, g_full=g_full, lambda_full=dd.WHEAT_LAMBDA_FULL,
                       mode=dd.MARGINAL)
    assert 0.0 <= r0["G_pred"] <= g_full
    assert rhi["G_pred"] == pytest.approx(g_full, rel=1e-3)    # full depth -> g_full
    assert rhi["rel_to_full"] == pytest.approx(1.0, abs=1e-3)


def test_predict_G_monotone_in_depth():
    g = [dd.predict_G(d, g_full=372.0, lambda_full=dd.WHEAT_LAMBDA_FULL,
                      mode=dd.MARGINAL)["G_pred"] for d in np.linspace(0.5, 40, 60)]
    assert all(b >= a - 1e-9 for a, b in zip(g, g[1:], strict=False))


def test_classify_band_boundaries():
    assert dd.classify_band(10) == "underpowered/null-likely"
    assert dd.classify_band(100) == "borderline/risky"
    assert dd.classify_band(500) == "discovery-feasible"
    assert dd.classify_band(5000) == "strong-design"


# ---- band inversion ---------------------------------------------------------
def test_depth_for_band_unreachable_when_gfull_below_target():
    # rapeseed-like: full-depth G already below feasible -> no depth helps
    assert dd.depth_for_band(dd.FEASIBLE_G, g_full=14.0,
                             lambda_full=dd.WHEAT_LAMBDA_FULL,
                             mode=dd.MARGINAL) is None


def test_depth_for_band_reachable_and_interaction_costlier():
    dm = dd.depth_for_band(dd.FEASIBLE_G, g_full=1759.0,
                           lambda_full=dd.WHEAT_LAMBDA_FULL, mode=dd.MARGINAL)
    di = dd.depth_for_band(dd.FEASIBLE_G, g_full=1759.0,
                           lambda_full=dd.WHEAT_LAMBDA_FULL, mode=dd.INTERACTION)
    assert dm is not None and di is not None
    assert di > dm                                             # interaction needs more depth


# ---- summary + CLI ----------------------------------------------------------
def test_design_summary_keys():
    s = dd.design_summary(g_full=1759.0, lambda_full=dd.WHEAT_LAMBDA_FULL)
    assert set(s) == {"marginal", "interaction"}
    assert "depth_for_feasible" in s["marginal"]
    assert s["marginal"]["assumptions"]["min_dp"] == dd.MARGINAL.min_dp


def test_cli_design_writes_json(tmp_path):
    from homoeogwas.cli import main
    out = tmp_path / "design.json"
    rc = main(["design", "--like", "wheat", "--depth-grid", "2,8,30",
               "-o", str(out)])
    assert rc == 0
    payload = json.loads(out.read_text())
    assert payload["analysis"] == "design_depth"
    assert payload["species"].startswith("wheat")
    # 3 depths x 2 modes
    assert len(payload["table"]) == 6
    assert payload["summary"]["interaction"]["depth_for_feasible"] is not None


def test_cli_design_custom_anchor_density_limited(tmp_path, capsys):
    from homoeogwas.cli import main
    rc = main(["design", "--g-full", "20", "--species", "demo4n",
               "--depth-grid", "5,30"])
    assert rc == 0
    cap = capsys.readouterr().out
    assert "marker density / capture design" in cap            # density-limited note


def test_cli_design_requires_anchor():
    from homoeogwas.cli import main
    assert main(["design", "--depth-grid", "5"]) == 2          # no --like / --g-full


# ---- input validation -------------------------------------------------------
def test_modeparams_rejects_bad_assumptions():
    with pytest.raises(ValueError):
        dd.ModeParams(name="x", min_dp=0, mappability=0.8)     # min_dp < 1
    with pytest.raises(ValueError):
        dd.ModeParams(name="x", min_dp=4, mappability=1.5)     # mappability > 1
    with pytest.raises(ValueError):
        dd.ModeParams(name="x", min_dp=4, mappability=0.0)     # mappability <= 0


def test_predict_G_rejects_bad_inputs():
    kw = dict(g_full=100.0, lambda_full=dd.WHEAT_LAMBDA_FULL, mode=dd.MARGINAL)
    with pytest.raises(ValueError):
        dd.predict_G(-1.0, **kw)                               # negative depth
    with pytest.raises(ValueError):
        dd.predict_G(5.0, g_full=-1.0, lambda_full=10.0, mode=dd.MARGINAL)
    with pytest.raises(ValueError):
        dd.predict_G(5.0, g_full=100.0, lambda_full=0.0, mode=dd.MARGINAL)


def test_cli_design_rejects_bad_overrides():
    from homoeogwas.cli import main
    assert main(["design", "--like", "wheat",
                 "--marginal-mappability", "1.5"]) == 2
    assert main(["design", "--like", "wheat", "--g-full", "-3"]) == 2
    assert main(["design", "--like", "wheat", "--lambda-full", "0"]) == 2


def test_depth_band_range_ordered():
    # optimistic (better mappability) <= point <= pessimistic depth
    r = dd.depth_band_range(dd.FEASIBLE_G, g_full=1759.0,
                            lambda_full=dd.WHEAT_LAMBDA_FULL, mode=dd.INTERACTION)
    vals = [x for x in r if x is not None]
    assert vals == sorted(vals)                                # opt <= point <= pess


def test_cli_design_warns_on_lambda_clamp(capsys):
    from homoeogwas.cli import main
    rc = main(["design", "--like", "wheat", "--lambda-full", "200",
               "--depth-grid", "10"])
    assert rc == 0
    assert "clamped" in capsys.readouterr().err
