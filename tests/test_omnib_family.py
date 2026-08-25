"""Invariants for the shared homoeolog edge/group omniB scorer."""

import numpy as np
import pytest

import homoeogwas.interact as I
import homoeogwas.omnib_family as O
from homoeogwas.group_family import MasterGroupFamily
from homoeogwas.interact import SubgenomeData, _score_omnib_family, acat
from homoeogwas.omnib_family import (
    run_clique_scan_omnib as run_shared_clique_scan_omnib,
)
from homoeogwas.omnib_family import (
    run_pair_scan_omnib as run_shared_pair_scan_omnib,
)


def _subgenome(rng, n, group_count, snps_per_gene=5):
    X = rng.integers(0, 3, size=(n, group_count * snps_per_gene)).astype(float)
    mapping = {
        f"g{i}": np.arange(i * snps_per_gene, (i + 1) * snps_per_gene)
        for i in range(group_count)
    }
    return SubgenomeData(
        X=X, gene_snp=mapping, samples=[f"s{i}" for i in range(n)], chunk=None)


def _family_scores(subgenomes, group_count, B, *, n_jobs=1):
    rng = np.random.default_rng(912)
    n = 72
    subdata = {
        sub: _subgenome(rng, n, group_count)
        for sub in subgenomes
    }
    family = MasterGroupFamily(
        tuple(subgenomes),
        tuple(f"group_{i}" for i in range(group_count)),
        tuple(tuple(f"g{i}" for _ in subgenomes) for i in range(group_count)),
    )
    return _score_omnib_family(
        subdata, family, rng.normal(size=n), np.arange(n),
        bootstrap_B=B, bootstrap_seed=2026, n_jobs=n_jobs,
        grm_method="grm_from_X", min_snp=3)


def test_two_copy_group_is_bit_exact_edge_for_every_response_column():
    scores, _expanded = _family_scores(("A", "D"), 8, 19)
    assert scores.edge_p.shape == (8, 20)
    assert scores.group_p.shape == (8, 20)
    np.testing.assert_array_equal(scores.group_p, scores.edge_p)


@pytest.mark.parametrize(
    ("subgenomes", "edge_count"),
    [(("A", "B", "D"), 3), (("A", "B", "C", "D"), 6)],
)
def test_three_and_four_copy_groups_acat_reduce_unique_edges(subgenomes, edge_count):
    scores, expanded = _family_scores(subgenomes, 5, 3)
    assert len(expanded.group_edge_indices[0]) == edge_count
    for group_index, edge_indices in enumerate(expanded.group_edge_indices):
        for column in range(scores.group_p.shape[1]):
            assert scores.group_p[group_index, column] == pytest.approx(
                acat(scores.edge_p[list(edge_indices), column]))


def test_partial_groups_and_nonestimable_edges_remain_explicit():
    rng = np.random.default_rng(913)
    n = 64
    subdata = {
        sub: _subgenome(rng, n, 2)
        for sub in ("A", "B", "D")
    }
    family = MasterGroupFamily(
        ("A", "B", "D"), ("complete", "partial"),
        (("g0", "g0", "g0"), ("g1", "g1", "missing")))
    scores, expanded = _score_omnib_family(
        subdata, family, rng.normal(size=n), np.arange(n), bootstrap_B=2,
        n_jobs=1, grm_method="grm_from_X", min_snp=3)
    indices = np.asarray(expanded.group_edge_indices[1], int)
    assert scores.group_estimable.tolist() == [True, True]
    assert scores.group_partial.tolist() == [False, True]
    assert scores.edge_estimable[indices].tolist() == [True, False, False]
    assert np.isnan(scores.edge_p[indices[1:]]).all()


def test_callable_edge_nonfinite_after_whitening_aborts(monkeypatch):
    rng = np.random.default_rng(914)
    n = 48
    subdata = {sub: _subgenome(rng, n, 1) for sub in ("A", "D")}
    family = MasterGroupFamily(("A", "D"), ("g",), (("g0", "g0"),))

    monkeypatch.setattr(
        I, "_omnib_components_over_Y",
        lambda *args, **kwargs: np.full((3, 2), np.nan))
    with pytest.raises(RuntimeError, match="post-whitening.*non-finite"):
        _score_omnib_family(
            subdata, family, rng.normal(size=n), np.arange(n), bootstrap_B=1,
            n_jobs=1, grm_method="grm_from_X", min_snp=3)


def test_gene_features_are_cached_once_and_parallel_blocks_are_deterministic(monkeypatch):
    rng = np.random.default_rng(915)
    n = 48
    subdata = {sub: _subgenome(rng, n, 1) for sub in ("A", "D")}
    family = MasterGroupFamily(
        ("A", "D"), ("one", "two"), (("g0", "g0"), ("g0", "g0")))
    calls = {"count": 0}
    original = I.gene_pc_scores

    def counting(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(I, "gene_pc_scores", counting)
    one, expanded = _score_omnib_family(
        subdata, family, rng.normal(size=n), np.arange(n), bootstrap_B=2,
        n_jobs=1, grm_method="grm_from_X", min_snp=3)
    assert len(expanded.edges) == 1
    assert calls["count"] == 2

    # Reconstruct identical inputs and verify block partitioning cannot change values.
    rng = np.random.default_rng(915)
    subdata = {sub: _subgenome(rng, n, 1) for sub in ("A", "D")}
    parallel, _ = _score_omnib_family(
        subdata, family, rng.normal(size=n), np.arange(n), bootstrap_B=2,
        n_jobs=2, grm_method="grm_from_X", min_snp=3)
    np.testing.assert_array_equal(one.edge_p, parallel.edge_p)


def test_pair_wrapper_preserves_bootstrap_primary_family_semantics(monkeypatch):
    rng = np.random.default_rng(916)
    n, group_count = 64, 3
    subdata = {
        sub: _subgenome(rng, n, group_count)
        for sub in ("A", "D")
    }
    pairs = [(f"g{i}", f"g{i}") for i in range(group_count)]

    def forced_calibration(_interact_module, p_obs, p_null, *, alpha=0.05):
        assert p_obs.shape == (group_count,)
        assert p_null.shape == (group_count, 3)
        return {
            "alpha": alpha,
            "method": "parametric_bootstrap_minp_plus_one",
            "B": 3,
            "empirical_p": 0.25,
            "threshold": 1.0,
            "threshold_comparator": "strict_less_than",
            "rejected": True,
            "rejected_local": [1],
            "adjusted_p_local": np.array([0.75, 0.25, 1.0]),
            "n_degenerate_replicates": 0,
            "degenerate_policy": "any_nonfinite_statistic_sets_null_min_to_zero",
        }

    # Patch the shared module's calibration hook rather than an optional
    # compatibility helper on interact.py.  This keeps the wrapper test valid
    # against both the frozen baseline and the extended engine schema.
    monkeypatch.setattr(O, "_bootstrap_minp", forced_calibration)
    result = run_shared_pair_scan_omnib(
        subdata, pairs, rng.normal(size=n), np.arange(n), bootstrap_B=3,
        n_jobs=1, pair_subs=("A", "D"), grm_method="grm_from_X",
        primary_multiplicity="bootstrap_minp")

    assert [tuple(record["pair"]) for record in result.sig] == [pairs[1]]
    assert result.sig[0]["p_adjusted_bootstrap_minp"] == pytest.approx(0.25)
    if hasattr(result, "model_diagnostics"):
        assert result.model_diagnostics["bootstrap_fwer"]["sig"] == result.sig
        assert result.analytic_screen_sig is not result.sig


def test_clique_wrapper_full_ranking_keeps_edge_and_component_localization(tmp_path):
    rng = np.random.default_rng(917)
    n, group_count = 64, 4
    subdata = {
        sub: _subgenome(rng, n, group_count)
        for sub in ("A", "B", "D")
    }
    groups = [tuple(f"g{i}" for _ in subdata) for i in range(group_count)]
    ranking = tmp_path / "groups.tsv"
    result = run_shared_clique_scan_omnib(
        subdata, groups, rng.normal(size=n), np.arange(n), bootstrap_B=0,
        n_jobs=1, grm_method="grm_from_X", full_dump_path=str(ranking))

    header = ranking.read_text().splitlines()[0].split("\t")
    assert len(ranking.read_text().splitlines()) == result.G + 1
    assert {
        "p_omnib_AB", "p_minor_burden_AD", "p_pc1_BD",
        "p_kernel_hadamard_BD", "smallest_pair", "smallest_component",
    } <= set(header)
    assert set(result.top[0]["pairwise"]) == {"AB", "AD", "BD"}
    if hasattr(result, "component_diagnostics"):
        assert result.component_diagnostics["pair_labels"] == ["AB", "AD", "BD"]
