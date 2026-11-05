"""Figures written by `homoeogwas fit` and redrawn by `homoeogwas plot`."""

from __future__ import annotations

import yaml

from homoeogwas import cli
from homoeogwas.demo_data import make_demo
from homoeogwas.plots import subgenome_palette


def test_subgenome_colours_follow_the_letter():
    assert subgenome_palette(["A", "C"]) == {"A": "#2F6DB5", "C": "#8B4FB0"}
    assert subgenome_palette(["A", "B", "D"])["D"] == "#2E9466"
    other = subgenome_palette(["A", "X"])
    assert other["X"] not in {"#2F6DB5"}


def test_pve_uncertainty_figure_follows_the_bootstrap(tmp_path):
    config_path = make_demo(tmp_path / "demo")
    config = yaml.safe_load(config_path.read_text())
    config["reml"] = {**config.get("reml", {}), "pve_bootstrap": {"B": 12, "seed": 3}}
    config_path.write_text(yaml.safe_dump(config))
    out = tmp_path / "fit"
    assert cli.main(["fit", "-c", str(config_path), "-o", str(out), "--backend", "cpu"]) == 0
    prefix = config["outputs"]["prefix"]
    figure = out / f"pve_uncertainty_{prefix}.png"
    assert figure.stat().st_size > 0
    figure.unlink()
    assert cli.main(["plot", str(out), "--formats", "png", "--figures", "pve"]) == 0
    assert figure.exists()
