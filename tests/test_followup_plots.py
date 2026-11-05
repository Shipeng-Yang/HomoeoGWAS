"""Follow-up figures: PC1-class allele combinations and deletion robustness."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from homoeogwas import followup_plots as P

pytest.importorskip("matplotlib")


def test_two_means_split_separates_two_clusters():
    x = np.r_[np.full(30, -1.0), np.full(10, 2.0)] + np.linspace(0, 0.1, 40)
    assert P.two_means_split(x).sum() == 10
    assert not P.two_means_split(np.ones(5)).any()


def test_combination_table_reports_class_means_and_counts():
    pc_x = np.r_[np.zeros(20), np.ones(20)]
    pc_y = np.tile(np.r_[np.zeros(10), np.ones(10)], 2)
    y = 1.0 * pc_x + 2.0 * pc_y + 5.0 * pc_x * pc_y
    table = P.combination_table(pc_x, pc_y, y)
    assert list(table["n"]) == [10, 10, 10, 10]
    np.testing.assert_allclose(table["mean_shift"], [0, 1, 2, 8])
    assert table["combination"].tolist() == list(P.COMBINATION_LABELS)


def test_robustness_figure_is_written(tmp_path):
    stability = pd.DataFrame({"rank": [0, 1], "gene_A": ["a1", "a2"], "gene_B": ["b1", "b2"],
                              "formal_p": [1e-7, 3e-6]})
    rng = np.random.default_rng(0)
    material = pd.DataFrame({"rank": np.repeat([0, 1], 20),
                             "p_interaction": rng.uniform(1e-8, 1e-3, 40)})
    environment = pd.DataFrame({"rank": [0, 0, 1, 1], "p_interaction": [1e-5, 0.2, 1e-4, 1e-3]})
    written = P.plot_robustness(stability, material, environment, tmp_path, formats=("png",))
    assert [p.rsplit("/", 1)[1] for p in written] == ["followup_robustness.png"]
