import numpy as np
import pytest

from scripts.benchmarks.v201.simulation import (
    compose_exact_pve,
    draw_null,
    interaction_signal,
    standardize,
    wilson_interval,
)


def test_compose_exact_pve_orthogonalizes_residual():
    rng = np.random.default_rng(11)
    signal = np.linspace(-1, 1, 200)
    y, metadata = compose_exact_pve(signal, rng.normal(size=200), 0.10)
    assert np.isclose(np.var(y, ddof=1), 1.0, atol=1e-12)
    assert np.isclose(metadata["realized_pve"], 0.10, atol=1e-12)


def test_wilson_interval_known_fixture():
    lo, hi = wilson_interval(50, 1000, level=0.95)
    assert 0.038 < lo < 0.039
    assert 0.065 < hi < 0.066


def test_standardize_has_exact_sample_moments_and_rejects_bad_vectors():
    result = standardize([1.0, 2.0, 4.0, 8.0])
    assert np.isclose(result.mean(), 0.0, atol=1e-15)
    assert np.isclose(result.var(ddof=1), 1.0, atol=1e-15)
    for bad in ([1.0], [1.0, 1.0], [1.0, np.nan], [[1.0, 2.0]]):
        with pytest.raises(ValueError):
            standardize(bad)


@pytest.mark.parametrize("pve", [0.0, 1.0, -0.1, 1.1, np.nan, True, "0.1"])
def test_compose_exact_pve_rejects_invalid_pve(pve):
    with pytest.raises(ValueError, match="pve"):
        compose_exact_pve([0.0, 1.0, 2.0], [2.0, 0.0, 1.0], pve)


def test_compose_exact_pve_rejects_mismatched_nonfinite_and_collinear_residuals():
    with pytest.raises(ValueError, match="equal length"):
        compose_exact_pve([0.0, 1.0, 2.0], [1.0, 2.0], 0.2)
    with pytest.raises(ValueError, match="finite"):
        compose_exact_pve([0.0, 1.0, 2.0], [1.0, np.inf, 2.0], 0.2)
    with pytest.raises(ValueError, match="degenerate"):
        compose_exact_pve([0.0, 1.0, 2.0], [0.0, 1.0, 2.0], 0.2)


@pytest.mark.parametrize(
    "successes,total,level",
    [
        (-1, 10, 0.95),
        (11, 10, 0.95),
        (1, 0, 0.95),
        (1.5, 10, 0.95),
        (True, 10, 0.95),
        (1, 10, 0.0),
        (1, 10, 1.0),
        (1, 10, np.nan),
        (1, 10, "0.95"),
    ],
)
def test_wilson_interval_rejects_invalid_counts_and_level(successes, total, level):
    with pytest.raises(ValueError):
        wilson_interval(successes, total, level)


def _null_inputs(n=200):
    pc1 = np.linspace(-2.0, 3.0, n)
    root_v = np.eye(n)
    return root_v, pc1


@pytest.mark.parametrize(
    "kind",
    [
        "gaussian",
        "t5",
        "heteroscedastic_pc1",
        "contamination_1pct_6sd",
        "additive_only",
        "omitted_kernel",
    ],
)
def test_draw_null_is_deterministic_standardized_and_auditable(kind):
    root_v, pc1 = _null_inputs()
    first, metadata = draw_null(kind, root_v, np.random.default_rng(123), pc1)
    second, _ = draw_null(kind, root_v, np.random.default_rng(123), pc1)
    np.testing.assert_array_equal(first, second)
    assert np.isclose(first.mean(), 0.0, atol=1e-14)
    assert np.isclose(first.var(ddof=1), 1.0, atol=1e-14)
    assert metadata["kind"] == kind
    assert metadata["n"] == 200
    assert metadata["root_shape"] == [200, 200]
    assert metadata["standardized"] is True
    assert isinstance(metadata["response_generation"], str)


def test_draw_null_uses_root_factor_and_locks_stress_distributions():
    root_v, pc1 = _null_inputs(n=200)
    actual, gaussian_meta = draw_null("gaussian", root_v, np.random.default_rng(91), pc1)
    expected = standardize(np.random.default_rng(91).standard_normal(200))
    np.testing.assert_allclose(actual, expected, atol=1e-14)
    assert gaussian_meta["innovation_distribution"] == "standard_normal"

    _, t_meta = draw_null("t5", root_v, np.random.default_rng(1), pc1)
    assert t_meta["innovation_distribution"] == "standardized_student_t"
    assert t_meta["degrees_of_freedom"] == 5
    assert t_meta["innovation_variance"] == 1.0

    _, hetero_meta = draw_null("heteroscedastic_pc1", root_v, np.random.default_rng(2), pc1)
    assert hetero_meta["variance_ratio_max_min"] == 4.0
    assert hetero_meta["variance_driver"] == "pc1_minmax"

    _, contamination_meta = draw_null(
        "contamination_1pct_6sd", root_v, np.random.default_rng(3), pc1
    )
    assert contamination_meta["contamination_count"] == 2
    assert contamination_meta["contamination_fraction"] == 0.01
    assert contamination_meta["contamination_shift_sd"] == 6.0

    _, additive_meta = draw_null("additive_only", root_v, np.random.default_rng(4), pc1)
    assert additive_meta["interaction_present"] is False
    assert additive_meta["main_effect"] == "pc1"

    _, omitted_meta = draw_null("omitted_kernel", root_v, np.random.default_rng(5), pc1)
    assert omitted_meta["misspecified"] is True
    assert omitted_meta["omitted_component"] == "pc1_rank_one_kernel"


def test_draw_null_rejects_unknown_malformed_and_degenerate_inputs():
    root_v, pc1 = _null_inputs(n=10)
    with pytest.raises(ValueError, match="unknown null"):
        draw_null("other", root_v, np.random.default_rng(1), pc1)
    with pytest.raises(ValueError, match="root_V"):
        draw_null("gaussian", root_v[:, :-1].T, np.random.default_rng(1), pc1)
    with pytest.raises(ValueError, match="finite"):
        bad = root_v.copy()
        bad[0, 0] = np.nan
        draw_null("gaussian", bad, np.random.default_rng(1), pc1)
    with pytest.raises(ValueError, match="pc1"):
        draw_null("gaussian", root_v, np.random.default_rng(1), pc1[:-1])
    with pytest.raises(ValueError, match="degenerate"):
        draw_null("heteroscedastic_pc1", root_v, np.random.default_rng(1), np.ones(10))


def _gene_blocks(copies=("A", "B", "D"), n=180, p=5):
    rng = np.random.default_rng(812)
    return {
        copy: rng.binomial(2, 0.15 + 0.05 * index, size=(n, p)).astype(float)
        for index, copy in enumerate(copies)
    }


@pytest.mark.parametrize(
    "architecture,component",
    [
        ("minor_burden_aligned", "minor_burden"),
        ("pc1_distributed", "pc1"),
        ("kernel_multidimensional", "kernel_hadamard"),
        ("single_snp_pair", "single_snp_pair"),
        ("mixed_sign", "mixed_sign"),
        ("multi_edge_group", "minor_burden"),
        ("additive_only", "additive_main_effect"),
    ],
)
def test_interaction_signal_covers_locked_architectures(architecture, component):
    signal, metadata = interaction_signal(architecture, _gene_blocks())
    assert signal.shape == (180,)
    assert np.isclose(signal.mean(), 0.0, atol=1e-14)
    assert np.isclose(signal.var(ddof=1), 1.0, atol=1e-14)
    assert metadata["architecture"] == architecture
    assert metadata["evidence_component"] == component
    assert metadata["available_pair_edge_count"] == 3
    assert metadata["direct_higher_order_term"] is False


def test_single_snp_pair_uses_one_true_column_per_gene():
    blocks = _gene_blocks(copies=("A", "B"))
    signal, metadata = interaction_signal("single_snp_pair", blocks, snp_indices={"A": 2, "B": 4})
    expected = standardize(standardize(blocks["A"][:, 2]) * standardize(blocks["B"][:, 4]))
    np.testing.assert_allclose(signal, expected, atol=1e-14)
    assert metadata["snp_indices"] == {"A": 2, "B": 4}
    assert metadata["snp_count_per_copy"] == {"A": 1, "B": 1}


def test_multi_edge_group_aggregates_two_of_six_pair_edges_without_four_way_term():
    signal, metadata = interaction_signal(
        "multi_edge_group", _gene_blocks(copies=("A", "B", "C", "D"))
    )
    assert np.isfinite(signal).all()
    assert metadata["available_pair_edge_count"] == 6
    assert metadata["causal_pair_edge_count"] == 2
    assert len(metadata["causal_pair_edges"]) == 2
    assert metadata["direct_higher_order_term"] is False


def test_mispaired_uses_predeclared_wrong_homoeolog_block():
    blocks = _gene_blocks(copies=("A", "B"))
    wrong = np.roll(blocks["B"], 17, axis=0)
    signal, metadata = interaction_signal("mispaired", blocks, wrong_partner=wrong)
    expected = standardize(standardize(blocks["A"].sum(axis=1)) * standardize(wrong.sum(axis=1)))
    np.testing.assert_allclose(signal, expected, atol=1e-14)
    assert metadata["negative_control"] is True
    assert metadata["partner_source"] == "predeclared_wrong_homoeolog"


def test_interaction_signal_rejects_invalid_blocks_edges_and_architectures():
    blocks = _gene_blocks(copies=("A", "B"))
    with pytest.raises(ValueError, match="unknown interaction"):
        interaction_signal("unknown", blocks)
    with pytest.raises(ValueError, match="multi_edge_group"):
        interaction_signal("multi_edge_group", blocks)
    with pytest.raises(ValueError, match="wrong_partner"):
        interaction_signal("mispaired", blocks)
    with pytest.raises(ValueError, match="finite"):
        bad = {name: value.copy() for name, value in blocks.items()}
        bad["A"][0, 0] = np.nan
        interaction_signal("minor_burden_aligned", bad)
    with pytest.raises(ValueError, match="pair edge"):
        interaction_signal("minor_burden_aligned", blocks, pair_edges=[("A", "A")])
    with pytest.raises(ValueError, match="copy labels"):
        interaction_signal("minor_burden_aligned", {"A": blocks["A"], 2: blocks["B"]})
