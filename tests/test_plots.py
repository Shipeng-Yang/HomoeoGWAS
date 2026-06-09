"""Smoke tests for the publication plotting module."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from homoeogwas import plots


def _toy_frame(subg=("A", "B", "D"), n_per=60, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for sg in subg:
        for c in (1, 2):
            chrom = f"chr{c}{sg}"
            p = rng.uniform(0, 1, n_per)
            p[0] = 1e-9   # guarantee a labelled top hit per chrom
            for i in range(n_per):
                rows.append({"subgenome": sg, "chrom": chrom,
                             "pos": (i + 1) * 1000, "p": float(p[i]),
                             "chi2": float(-np.log(p[i]))})
    return pd.DataFrame(rows)


def _toy_lambda(subg=("A", "B", "D")):
    rows = [{"scope": "all", "level": "all", "n_markers": 360,
             "lambda_gc": 1.05}]
    for sg in subg:
        rows.append({"scope": "subgenome", "level": sg, "n_markers": 120,
                     "lambda_gc": 0.98})
    return pd.DataFrame(rows)


def test_make_all_plots_writes_every_figure(tmp_path):
    subg = ["A", "B", "D"]
    df = _toy_frame(subg)
    sigma2 = {"A": 1.2, "B": 0.8, "D": 0.0, "e": 0.5}
    pve = {"A": 0.5, "B": 0.3, "D": 0.0, "e": 0.2}
    paths = plots.make_all_plots(
        df, subg, sigma2=sigma2, pve=pve, boundary_components=["D"],
        lambda_df=_toy_lambda(subg), trait="grain_width", out_dir=tmp_path,
        prefix="grain_width")
    assert paths
    # legacy filenames preserved
    assert (tmp_path / "manhattan_grain_width.png").exists()
    assert (tmp_path / "qq_grain_width.png").exists()
    # signature + QC figures
    assert (tmp_path / "variance_components_grain_width.png").exists()
    assert (tmp_path / "lambda_gc_grain_width.png").exists()
    # all four figures emit png + pdf + svg
    for stem in ("manhattan", "qq", "variance_components", "lambda_gc"):
        for fmt in ("png", "pdf", "svg"):
            f = tmp_path / f"{stem}_grain_width.{fmt}"
            assert f.exists() and f.stat().st_size > 0


def test_variance_components_handles_homoeolog_kernel(tmp_path):
    subg = ["A", "D"]
    sigma2 = {"A": 1.0, "D": 0.7, "hom": 0.4, "e": 0.6}
    pve = {"A": 0.37, "D": 0.26, "hom": 0.15, "e": 0.22}
    p = plots.plot_variance_components(
        sigma2, pve, subg, out_dir=tmp_path, prefix="t", trait="fiber")
    assert any(s.endswith("variance_components_t.png") for s in p)


def test_single_subgenome_diploid(tmp_path):
    df = _toy_frame(subg=("A",))
    plots.make_all_plots(
        df, ["A"], sigma2={"A": 1.0, "e": 0.5}, pve={"A": 0.67, "e": 0.33},
        lambda_df=_toy_lambda(("A",)), trait="d", out_dir=tmp_path, prefix="d")
    assert (tmp_path / "manhattan_d.png").exists()
    assert (tmp_path / "qq_d.png").exists()


def test_empty_frame_returns_no_manhattan(tmp_path):
    empty = pd.DataFrame(columns=["subgenome", "chrom", "pos", "p", "chi2"])
    assert plots.plot_manhattan(empty, ["A"], out_dir=tmp_path,
                                prefix="e") == []
    assert plots.plot_qq(empty, ["A"], {}, {}, out_dir=tmp_path,
                         prefix="e") == []


def test_formats_subset(tmp_path):
    df = _toy_frame(subg=("A", "B"))
    paths = plots.make_all_plots(
        df, ["A", "B"], sigma2={"A": 1.0, "B": 1.0, "e": 0.5},
        pve={"A": 0.4, "B": 0.4, "e": 0.2}, lambda_df=_toy_lambda(("A", "B")),
        out_dir=tmp_path, prefix="p", formats=["png"])
    assert all(p.endswith(".png") for p in paths)
    assert not list(tmp_path.glob("*.svg"))


def test_plot_from_results_roundtrip(tmp_path):
    subg = ["A", "D"]
    df = _toy_frame(subg)
    # write a sumstats tsv + lambda tsv + summary json like a finished run
    df.assign(snp_id=[f"s{i}" for i in range(len(df))], beta=0.1, se=0.1,
              n_obs=100, maf=0.3, call_rate=0.99).to_csv(
        tmp_path / "sumstats_yield.tsv", sep="\t", index=False)
    _toy_lambda(subg).to_csv(tmp_path / "lambda_gc_yield.tsv", sep="\t",
                             index=False)
    summary = {
        "trait": "yield", "subgenomes": subg, "include_hadamard": False,
        "reml": {"sigma2": {"A": 1.0, "D": 0.8, "e": 0.5},
                 "pve": {"A": 0.43, "D": 0.35, "e": 0.22},
                 "boundary_components": []},
        "lambda_gc": {"all": 1.05, "A": 0.98, "D": 0.98},
        "outputs": {"sumstats": [str(tmp_path / "sumstats_yield.tsv")]},
    }
    (tmp_path / "summary_yield.json").write_text(json.dumps(summary))

    paths = plots.plot_from_results(tmp_path, max_points=100)
    assert (tmp_path / "manhattan_yield.png").exists()
    assert (tmp_path / "variance_components_yield.png").exists()
    # summary.json was updated with the new plot paths
    updated = json.loads((tmp_path / "summary_yield.json").read_text())
    assert updated["outputs"]["plots"] == paths


def test_qq_ranks_use_true_n_in_tail(tmp_path):
    # thinned frame: 5 tail markers (p<1e-3) complete + sparse bulk; the tail
    # ranks must be exact and expected quantiles must use the TRUE n, not the
    # thinned length, so a small p is not artificially inflated.
    logp_desc = np.array([9.0, 8.0, 7.0, 5.0, 4.0, 2.0, 1.0, 0.5])
    n_true = 1_000_000
    ranks = plots._qq_ranks(logp_desc, n_true)
    assert list(ranks[:5]) == [1, 2, 3, 4, 5]          # exact tail ranks
    assert ranks[-1] <= n_true and ranks[5] > 5         # bulk spread to true n
    exp = -np.log10((ranks - 0.5) / n_true)
    # top marker expected ~ -log10(0.5/1e6) ~ 6.3, NOT -log10(0.5/8)
    assert exp[0] > 6.0


def test_qq_ranks_untouched_when_not_thinned():
    logp_desc = np.array([4.0, 2.0, 1.0, 0.3])
    ranks = plots._qq_ranks(logp_desc, n_true=4)
    assert list(ranks) == [1, 2, 3, 4]


def test_load_plot_df_bounded_keeps_strong(tmp_path):
    rng = np.random.default_rng(1)
    n = 20_000
    df = pd.DataFrame({
        "subgenome": "A", "chrom": "chr1A",
        "pos": np.arange(n), "p": rng.uniform(1e-2, 1, n),
        "chi2": 1.0})
    df.loc[0, "p"] = 1e-12        # one strong hit must survive thinning
    df.loc[1:5, "p"] = np.nan     # non-finite p must be dropped, not sampled
    df.to_csv(tmp_path / "s.tsv", sep="\t", index=False)
    out = plots._load_plot_df([str(tmp_path / "s.tsv")], max_points=500,
                              chunksize=4096)
    assert (out["p"] < 1e-3).sum() == 1                 # strong hit retained
    assert out["p"].notna().all()                        # no NaN rows leaked
    assert len(out) <= 500 + 50                          # bounded near cap


def test_plot_from_results_ambiguous_prefix(tmp_path):
    (tmp_path / "summary_a.json").write_text("{}")
    (tmp_path / "summary_b.json").write_text("{}")
    with pytest.raises(ValueError):
        plots.plot_from_results(tmp_path)
