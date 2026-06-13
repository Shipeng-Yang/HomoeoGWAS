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
