"""Tests for the optional R plotting channel (`homoeogwas rplot`).

R / CMplot are optional, so the rendering smoke is skipped when they are
absent (like the diamond/torch skips elsewhere); the wrapper logic itself is
tested without R via monkeypatching.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from homoeogwas import cli  # noqa: E402


def _r_available() -> bool:
    rs = cli._find_rscript()
    if rs is None:
        return False
    try:
        out = subprocess.run([rs, cli._r_script_asset(), "--check-deps"],
                             capture_output=True, text=True, timeout=120)
        return "CMplot=TRUE" in out.stdout
    except Exception:
        return False


def test_r_script_asset_exists():
    p = Path(cli._r_script_asset())
    assert p.exists() and p.name == "gwas_plots.R"


def test_rplot_parser_accepts_subcommand():
    args = cli.build_parser().parse_args(
        ["rplot", "somedir", "--kind", "manhattan,qq", "--format", "png"])
    assert args.subcommand == "rplot"
    assert args.kind == "manhattan,qq"


def test_rplot_missing_rscript_errors(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_find_rscript", lambda explicit=None: None)
    rc = cli.main(["rplot", ".", "--check-deps"])
    assert rc == 1
    assert "Rscript not found" in capsys.readouterr().out


def _toy_run(tmp_path):
    rng = np.random.default_rng(0)
    rows = []
    for sg in ("A", "D"):
        for c in (1, 2):
            chrom = f"{sg}0{c}"
            p = rng.uniform(1e-2, 1, 200)
            p[0] = 1e-9
            for i in range(200):
                rows.append({"snp_id": f"{chrom}:{i}", "subgenome": sg,
                             "chrom": chrom, "pos": (i + 1) * 1000,
                             "p": float(p[i])})
    df = pd.DataFrame(rows)
    df.to_csv(tmp_path / "sumstats_t.tsv", sep="\t", index=False)
    (tmp_path / "summary_t.json").write_text(json.dumps({
        "trait": "t", "subgenomes": ["A", "D"],
        "outputs": {"sumstats": [str(tmp_path / "sumstats_t.tsv")]}}))


def test_write_variance_tsv(tmp_path):
    summary = {"trait": "t", "subgenomes": ["A", "D"],
               "reml": {"pve": {"A": 0.3, "D": 0.5, "hom": 0.0, "e": 0.2},
                        "sigma2": {"A": 1.0, "D": 1.6, "hom": 0.0, "e": 0.7},
                        "boundary_components": ["hom"]}}
    out = tmp_path / "v.tsv"
    assert cli._write_variance_tsv(summary, out)
    rows = out.read_text().splitlines()
    assert rows[0].split("\t") == ["component", "pve", "sigma2", "kind",
                                   "is_boundary"]
    body = [r.split("\t") for r in rows[1:]]
    kinds = {r[0]: r[3] for r in body}
    assert kinds["A"] == "subgenome" and kinds["residual"] == "residual"
    assert kinds["homoeolog"] == "homoeolog"
    # boundary flag set for the homoeolog kernel
    hom = [r for r in body if r[0] == "homoeolog"][0]
    assert hom[4] == "1"


def test_write_variance_tsv_empty():
    assert not cli._write_variance_tsv({"reml": {}}, "/tmp/_never.tsv")


def _toy_ranking(tmp_path):
    rng = np.random.default_rng(2)
    n = 60
    p = rng.uniform(1e-2, 1, n)
    p[0] = 4.7e-5
    pmx = rng.uniform(0.1, 1, n)
    pmy = rng.uniform(0.1, 1, n)
    df = pd.DataFrame({
        "rank": range(n),
        "gene_A": [f"A{i}" for i in range(n)],
        "gene_D": [f"D{i}" for i in range(n)],
        "sub_x": "A", "sub_y": "D", "p_interaction": p,
        "neglog10p": -np.log10(p),
        "chrom_x": "A11", "pos_x": np.arange(n) * 1000 + 1,
        "chrom_y": "D11", "pos_y": np.arange(n) * 1000 + 1,
        "p_marginal_x": pmx, "p_marginal_y": pmy,
        "neglog10p_marginal_x": -np.log10(pmx),
        "neglog10p_marginal_y": -np.log10(pmy)})
    f = tmp_path / "interact_t_ranking_pairwise_INT.tsv"
    df.to_csv(f, sep="\t", index=False)
    return f


@pytest.mark.skipif(not _r_available(),
                    reason="Rscript/CMplot not installed")
def test_rplot_distinctive_smoke(tmp_path):
    _toy_ranking(tmp_path)
    out = tmp_path / "fig"
    rc = cli.main(["rplot", str(tmp_path), "--kind", "interaction,marginal",
                   "--format", "png", "--out-dir", str(out)])
    assert rc == 0
    names = [p.name for p in out.glob("*.png")]
    assert any("interaction_manhattan" in n for n in names)
    assert any("interaction_vs_marginal" in n for n in names)


@pytest.mark.skipif(not _r_available(),
                    reason="Rscript/CMplot not installed")
def test_rplot_cmplot_smoke(tmp_path):
    _toy_run(tmp_path)
    out = tmp_path / "fig"
    rc = cli.main(["rplot", str(tmp_path), "--kind", "manhattan,qq",
                   "--format", "png", "--out-dir", str(out)])
    assert rc == 0
    pngs = list(out.glob("*.png"))
    assert any("Manhtn" in p.name for p in pngs)
    assert any("QQ" in p.name for p in pngs)
