"""Invariants for the shared homoeolog edge/group omniB scorer."""

import numpy as np
import pytest

import homoeogwas.interact as I
import homoeogwas.omnib_family as F
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


def test_gene_feature_seed_is_keyed_not_call_order():
    burden = F._gene_feature_seed(17, "A", "g1", "minor_burden")
    pc = F._gene_feature_seed(17, "A", "g1", "gene_pc")

    assert burden == F._gene_feature_seed(17, "A", "g1", "minor_burden")
    assert burden != pc
    assert burden != F._gene_feature_seed(17, "A", "g2", "minor_burden")
    with pytest.raises(ValueError, match="unknown feature_type"):
        F._gene_feature_seed(17, "A", "g1", "unsupported")


def test_capped_gene_features_survive_family_reordering_and_truncation():
    rng = np.random.default_rng(921)
    n = 56
    subdata = {
        sub: _subgenome(rng, n, 2, snps_per_gene=160)
        for sub in ("A", "D")
    }
    forward_family = MasterGroupFamily(
        ("A", "D"), ("over_cap", "other"),
        (("g0", "g0"), ("g1", "g1")),
    )
    reverse_family = MasterGroupFamily(
        ("A", "D"), ("other", "over_cap"),
        (("g1", "g1"), ("g0", "g0")),
    )
    truncated_family = MasterGroupFamily(
        ("A", "D"), ("over_cap",), (("g0", "g0"),),
    )
    phenotype = rng.normal(size=n)
    sample_idx = np.arange(n)

    forward, _ = F.score_omnib_family(
        subdata, forward_family, phenotype, sample_idx,
        bootstrap_B=0, bootstrap_seed=99, feature_seed=17,
        n_jobs=1, grm_method="grm_from_X", cap=150, n_pc=3,
    )
    reverse, _ = F.score_omnib_family(
        subdata, reverse_family, phenotype, sample_idx,
        bootstrap_B=0, bootstrap_seed=101, feature_seed=17,
        n_jobs=1, grm_method="grm_from_X", cap=150, n_pc=3,
    )
    truncated, _ = F.score_omnib_family(
        subdata, truncated_family, phenotype, sample_idx,
        bootstrap_B=0, bootstrap_seed=103, feature_seed=17,
        n_jobs=1, grm_method="grm_from_X", cap=150, n_pc=3,
    )

    for sub in ("A", "D"):
        key = (sub, "g0")
        for candidate in (reverse, truncated):
            for left, right in zip(
                forward.feature_cache[key], candidate.feature_cache[key], strict=True,
            ):
                np.testing.assert_array_equal(left, right)
            assert forward.feature_identity[key] == candidate.feature_identity[key]
        assert forward.feature_identity[key]["retained_global_variant_indices"] == list(
            range(160)
        )
        assert len(
            forward.feature_identity[key]["selected_global_variant_indices"][
                "minor_burden"
            ]
        ) == 150
        assert len(
            forward.feature_identity[key]["selected_global_variant_indices"]["gene_pc"]
        ) == 150

    assert forward.feature_cache_sha256 == reverse.feature_cache_sha256
    assert (
        forward.feature_identity[("A", "g0")]["child_seeds"]["minor_burden"]
        != forward.feature_identity[("A", "g0")]["child_seeds"]["gene_pc"]
    )


def test_fixed_feature_seed_is_independent_of_bootstrap_seed():
    rng = np.random.default_rng(922)
    n = 48
    subdata = {
        sub: _subgenome(rng, n, 1, snps_per_gene=12)
        for sub in ("A", "D")
    }
    family = MasterGroupFamily(("A", "D"), ("one",), (("g0", "g0"),))
    phenotype = rng.normal(size=n)

    first, _ = F.score_omnib_family(
        subdata, family, phenotype, np.arange(n), cap=4,
        feature_seed=4103, bootstrap_B=1, bootstrap_seed=11,
        n_jobs=1, grm_method="grm_from_X", min_snp=3,
    )
    second, _ = F.score_omnib_family(
        subdata, family, phenotype, np.arange(n), cap=4,
        feature_seed=4103, bootstrap_B=1, bootstrap_seed=29,
        n_jobs=1, grm_method="grm_from_X", min_snp=3,
    )

    assert first.feature_cache_sha256 == second.feature_cache_sha256
    assert first.feature_identity == second.feature_identity
    for key in first.feature_cache:
        for left, right in zip(
            first.feature_cache[key], second.feature_cache[key], strict=True,
        ):
            np.testing.assert_array_equal(left, right)


def test_group_omnib_fork_is_bit_exact_to_serial():
    one, _ = _family_scores(("A", "D"), 8, 3, n_jobs=1)
    parallel, _ = _family_scores(("A", "D"), 8, 3, n_jobs=2)

    np.testing.assert_array_equal(one.edge_p, parallel.edge_p)
    np.testing.assert_array_equal(one.group_p, parallel.group_p)
    np.testing.assert_array_equal(
        one.edge_components_obs, parallel.edge_components_obs)
    np.testing.assert_array_equal(one.edge_estimable, parallel.edge_estimable)
    np.testing.assert_array_equal(one.group_estimable, parallel.group_estimable)
    assert one.parallel_execution["backend"] == "serial"
    assert parallel.parallel_execution["backend"] == "fork_shared_memory"
    assert parallel.parallel_execution["effective_jobs"] == 2
    assert len(parallel.parallel_execution["worker_pids"]) == 2


def test_pair_wrapper_rejects_task4_bootstrap_primary_authority():
    rng = np.random.default_rng(916)
    n, group_count = 64, 3
    subdata = {
        sub: _subgenome(rng, n, group_count)
        for sub in ("A", "D")
    }
    pairs = [(f"g{i}", f"g{i}") for i in range(group_count)]

    with pytest.raises(ValueError, match="only.*bonferroni"):
        run_shared_pair_scan_omnib(
            subdata, pairs, rng.normal(size=n), np.arange(n), bootstrap_B=3,
            n_jobs=1, pair_subs=("A", "D"), grm_method="grm_from_X",
            primary_multiplicity="bootstrap_minp")


def _subgenome_with_nonestimable_gene(rng, n):
    random_block = rng.integers(0, 3, size=(n, 5)).astype(float)
    constant_block = np.ones((n, 5), float)
    return SubgenomeData(
        X=np.column_stack([random_block, constant_block]),
        gene_snp={"g0": np.arange(5), "g1": np.arange(5, 10)},
        samples=[f"s{i}" for i in range(n)], chunk=None)


def test_pair_wrapper_keeps_callable_duplicates_and_nonestimable_rows(tmp_path):
    rng = np.random.default_rng(918)
    n = 64
    subdata = {
        sub: _subgenome_with_nonestimable_gene(rng, n)
        for sub in ("A", "D")
    }
    pairs = [("g0", "g0"), ("g1", "g1"), ("g0", "g0"), ("missing", "g0")]
    ranking = tmp_path / "pairs.tsv"
    result = run_shared_pair_scan_omnib(
        subdata, pairs, rng.normal(size=n), np.arange(n), bootstrap_B=3,
        n_jobs=1, pair_subs=("A", "D"), grm_method="grm_from_X",
        full_dump_path=str(ranking))

    header, *rows = [line.split("\t") for line in ranking.read_text().splitlines()]
    assert result.G == result.n_planned == 3
    assert result.n_valid == 2
    assert result.n_unestimable == 1
    assert result.bonferroni_alpha == pytest.approx(0.05 / 3)
    assert len(rows) == 3
    assert [(row[1], row[2]) for row in rows].count(("g0", "g0")) == 2
    invalid = [row for row in rows if row[1:3] == ["g1", "g1"]]
    assert len(invalid) == 1
    assert invalid[0][header.index("p_interaction")] == "NA"
    assert invalid[0][header.index("p_unestimable")] == "1"


def test_clique_wrapper_keeps_callable_duplicates_and_nonestimable_rows(tmp_path):
    rng = np.random.default_rng(919)
    n = 64
    subdata = {
        sub: _subgenome_with_nonestimable_gene(rng, n)
        for sub in ("A", "B", "D")
    }
    groups = [
        ("g0", "g0", "g0"), ("g1", "g1", "g1"),
        ("g0", "g0", "g0"), ("missing", "g0", "g0"),
    ]
    ranking = tmp_path / "groups_accounting.tsv"
    result = run_shared_clique_scan_omnib(
        subdata, groups, rng.normal(size=n), np.arange(n), bootstrap_B=3,
        n_jobs=1, grm_method="grm_from_X", full_dump_path=str(ranking))

    header, *rows = [line.split("\t") for line in ranking.read_text().splitlines()]
    assert result.G == result.n_planned == 3
    assert result.n_valid == 2
    assert result.n_unestimable == 1
    assert result.bonferroni_alpha == pytest.approx(0.05 / 3)
    assert len(rows) == 3
    assert [tuple(row[1:4]) for row in rows].count(("g0", "g0", "g0")) == 2
    invalid = [row for row in rows if row[1:4] == ["g1", "g1", "g1"]]
    assert len(invalid) == 1
    assert invalid[0][header.index("p_interaction")] == "NA"
    assert invalid[0][header.index("p_unestimable")] == "1"


def _legacy_pair_matrix(subdata, pairs, y_raw, sample_idx, *, B, seed):
    """Frozen pre-refactor pair matrix, including its loop-ordered RNG stream."""
    subs = list(subdata)
    sx, sy = subs
    kernels = {
        sub: I._build_grm(subdata[sub], sample_idx, "grm_from_X", 0.01)
        for sub in subs
    }
    y = I.rank_int(y_raw)
    W, _V, _beta, _cv = I.null_lmm_fit(kernels, y, None, seed=42)
    Cw = (W @ np.ones(sample_idx.size)).reshape(-1, 1)
    rng = np.random.default_rng(seed)
    features = {}
    for gx, gy in pairs:
        for sub, gene in ((sx, gx), (sy, gy)):
            key = (sub, gene)
            if key not in features:
                indices = np.asarray(subdata[sub].gene_snp[gene], int)
                Xg = subdata[sub].X[np.ix_(sample_idx, indices)]
                local = np.arange(indices.size)
                burden = I.block_burden_capped(
                    Xg, local, 150, rng, minor=True).reshape(-1, 1)
                pcs = I.gene_pc_scores(Xg, local, 150, rng, 3)
                features[key] = burden, pcs[:, :1], pcs

    responses = np.empty((sample_idx.size, B + 1), float)
    responses[:, 0] = y
    if B:
        replicates, _W2, _cv2 = I.null_replicates(
            kernels, y, C=None, B=B, method="bootstrap", seed=seed)
        responses[:, 1:] = np.column_stack(replicates)
    Yw = W @ responses
    return np.vstack([
        I._omnib_pair_over_Y(
            W, Yw, Cw, features[(sx, gx)], features[(sy, gy)])
        for gx, gy in pairs
    ])


def test_shared_pair_wrapper_is_frozen_legacy_equivalent(tmp_path):
    rng = np.random.default_rng(920)
    n, group_count, B, seed = 64, 4, 19, 2026
    subdata = {
        sub: _subgenome(rng, n, group_count)
        for sub in ("A", "D")
    }
    pairs = [(f"g{i}", f"g{i}") for i in range(group_count)]
    y = rng.normal(size=n)
    sample_idx = np.arange(n)
    old_P = _legacy_pair_matrix(
        subdata, pairs, y, sample_idx, B=B, seed=seed)

    family = MasterGroupFamily(
        ("A", "D"), tuple(f"pair_{i}" for i in range(group_count)),
        tuple(pairs))
    shared, _expanded = _score_omnib_family(
        subdata, family, y, sample_idx, bootstrap_B=B,
        bootstrap_seed=seed, n_jobs=1, grm_method="grm_from_X", min_snp=3)
    np.testing.assert_array_equal(shared.edge_p[:, 0], old_P[:, 0])
    np.testing.assert_allclose(
        shared.edge_p[:, 1:], old_P[:, 1:], rtol=1e-10, atol=1e-14)

    ranking = tmp_path / "frozen.tsv"
    result = run_shared_pair_scan_omnib(
        subdata, pairs, y, sample_idx, bootstrap_B=B,
        bootstrap_seed=seed, n_jobs=1, pair_subs=("A", "D"),
        grm_method="grm_from_X", full_dump_path=str(ranking))
    null_min = np.nanmin(old_P[:, 1:], axis=0)
    expected_threshold = float(np.quantile(null_min, 0.05))
    expected_global = float(
        (1 + (null_min <= old_P[:, 0].min()).sum()) / (B + 1))
    cutoff = 0.01
    counts = (old_P[:, 1:] < cutoff).sum(axis=0)
    observed_count = int((old_P[:, 0] < cutoff).sum())

    assert result.G == result.n_planned == result.n_valid == group_count
    assert result.n_unestimable == 0
    assert result.min_p == old_P[:, 0].min()
    assert result.minp_boot_threshold == pytest.approx(
        expected_threshold, rel=1e-10, abs=1e-14)
    assert result.minp_boot_emp == expected_global
    assert result.tail_excess["n_below_0.01"] == {
        "observed": observed_count,
        "null_mean": float(counts.mean()),
        "null_q95": float(np.quantile(counts, 0.95)),
        "empirical_p": float(
            (1 + (counts >= observed_count).sum()) / (B + 1)),
    }
    expected_sig = [
        pairs[index]
        for index in np.argsort(old_P[:, 0])
        if old_P[index, 0] < 0.05 / group_count
    ]
    assert [tuple(record["pair"]) for record in result.sig] == expected_sig
    header, *rows = [line.split("\t") for line in ranking.read_text().splitlines()]
    dumped = [float(row[header.index("p_interaction")]) for row in rows]
    assert dumped == sorted(old_P[:, 0].tolist())


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
