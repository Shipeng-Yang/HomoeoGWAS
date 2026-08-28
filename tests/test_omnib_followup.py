from __future__ import annotations

import numpy as np
import pytest

from homoeogwas.group_family import MasterGroupFamily, expand_pair_edges
from homoeogwas.interact import SubgenomeData
from homoeogwas.omnib_family import score_omnib_family, score_omnib_subset


def _toy_group_inputs(subgenomes, *, n=48, n_groups=3, snps_per_gene=6):
    rng = np.random.default_rng(90210 + len(subgenomes))
    subdata = {}
    for sub in subgenomes:
        matrix = rng.integers(
            0, 3, size=(n, n_groups * snps_per_gene)).astype(float)
        subdata[sub] = SubgenomeData(
            X=matrix,
            gene_snp={
                f"g{i}": np.arange(i * snps_per_gene, (i + 1) * snps_per_gene)
                for i in range(n_groups)
            },
            samples=[f"S{i:03d}" for i in range(n)],
            chunk=None,
        )
    family = MasterGroupFamily(
        subgenomes=tuple(subgenomes),
        group_ids=tuple(f"group_{i}" for i in range(n_groups)),
        genes=tuple(
            tuple(f"g{i}" for _ in subgenomes) for i in range(n_groups)),
    )
    y = rng.normal(size=n)
    return subdata, family, y, np.arange(n)


@pytest.mark.parametrize(
    "subgenomes", [("A", "C"), ("A", "B", "D"), ("A", "B", "C", "D")])
def test_subset_all_rows_replays_prepared_observed(subgenomes):
    subdata, family, y, sample_idx = _toy_group_inputs(subgenomes)
    scores, expanded = score_omnib_family(
        subdata, family, y, sample_idx, bootstrap_B=0,
        bootstrap_seed=2026, n_jobs=1, grm_method="grm_from_X")

    replay = score_omnib_subset(
        scores, family, expanded, np.arange(len(y)), y, n_jobs=1)

    np.testing.assert_allclose(
        replay.edge_p, scores.edge_p[:, 0], rtol=1e-6, atol=1e-10)
    np.testing.assert_allclose(
        replay.group_p, scores.group_p[:, 0], rtol=1e-6, atol=1e-10)
    if len(subgenomes) == 4:
        assert len(expand_pair_edges(family).edges) == 6 * len(family.group_ids)


def test_subset_scorer_rejects_fixed_covariate_context():
    subdata, family, y, sample_idx = _toy_group_inputs(("A", "C"))
    scores, expanded = score_omnib_family(
        subdata, family, y, sample_idx, bootstrap_B=0,
        bootstrap_seed=2026, n_jobs=1, grm_method="grm_from_X")
    scores.covariate_block = np.ones((len(y), 1))

    with pytest.raises(ValueError, match="fixed covariates"):
        score_omnib_subset(
            scores, family, expanded, np.arange(len(y)), y, n_jobs=1)


def test_subset_scorer_validates_keep_and_minimum_sample_count():
    subdata, family, y, sample_idx = _toy_group_inputs(("A", "C"))
    scores, expanded = score_omnib_family(
        subdata, family, y, sample_idx, bootstrap_B=0,
        bootstrap_seed=2026, n_jobs=1, grm_method="grm_from_X")

    with pytest.raises(ValueError, match="unique sorted"):
        score_omnib_subset(
            scores, family, expanded, [1, 0, 2, 3, 4, 5, 6, 7, 8, 9, 9],
            y[:11], n_jobs=1)
    with pytest.raises(ValueError, match="at least 10"):
        score_omnib_subset(scores, family, expanded, np.arange(9), y[:9], n_jobs=1)
