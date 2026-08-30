"""Shared finite-sample simulation primitives for the v2.0.1 benchmark."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import combinations
from numbers import Real
from typing import Any

import numpy as np
from scipy.stats import norm

_NULL_ALIASES = {
    "gaussian": "gaussian",
    "t5": "t5",
    "student_t5": "t5",
    "heteroscedastic_pc1": "heteroscedastic_pc1",
    "pc1_heteroscedastic": "heteroscedastic_pc1",
    "contamination_1pct_6sd": "contamination_1pct_6sd",
    "contamination": "contamination_1pct_6sd",
    "additive_only": "additive_only",
    "structure_aligned": "additive_only",
    "omitted_kernel": "omitted_kernel",
    "omitted_kernel_sensitivity": "omitted_kernel",
}

_ARCHITECTURES = {
    "minor_burden_aligned",
    "pc1_distributed",
    "kernel_multidimensional",
    "single_snp_pair",
    "mixed_sign",
    "multi_edge_group",
    "additive_only",
    "mispaired",
}


def standardize(values: np.ndarray) -> np.ndarray:
    """Center a finite one-dimensional vector and give it sample variance one."""

    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or x.size < 2 or not np.all(np.isfinite(x)):
        raise ValueError("values must be a finite one-dimensional vector")
    x = x - x.mean()
    scale = float(x.std(ddof=1))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("cannot standardize a degenerate vector")
    return x / scale


def compose_exact_pve(
    signal: np.ndarray,
    residual: np.ndarray,
    pve: float,
) -> tuple[np.ndarray, dict[str, float]]:
    """Compose a unit-variance phenotype with exact finite-sample signal PVE."""

    if isinstance(pve, bool) or not isinstance(pve, Real) or not np.isfinite(pve):
        raise ValueError("pve must be strictly between zero and one")
    pve = float(pve)
    if not 0.0 < pve < 1.0:
        raise ValueError("pve must be strictly between zero and one")
    s = standardize(signal)
    e = np.asarray(residual, dtype=np.float64)
    if e.ndim != 1 or e.shape != s.shape or not np.all(np.isfinite(e)):
        raise ValueError("signal and residual must be finite vectors of equal length")
    e = e - e.mean()
    e = e - s * float(s @ e) / float(s @ s)
    e = standardize(e)
    signal_component = np.sqrt(pve) * s
    residual_component = np.sqrt(1.0 - pve) * e
    y = signal_component + residual_component
    realized_pve = float(np.var(signal_component, ddof=1) / np.var(y, ddof=1))
    return y, {"target_pve": pve, "realized_pve": realized_pve}


def wilson_interval(
    successes: int,
    total: int,
    level: float = 0.95,
) -> tuple[float, float]:
    """Return a two-sided Wilson score interval for a binomial rate."""

    if (
        isinstance(successes, bool)
        or isinstance(total, bool)
        or not isinstance(successes, (int, np.integer))
        or not isinstance(total, (int, np.integer))
        or total < 1
        or not 0 <= successes <= total
    ):
        raise ValueError("invalid binomial counts")
    if isinstance(level, bool) or not isinstance(level, Real) or not np.isfinite(level):
        raise ValueError("level must be strictly between zero and one")
    level = float(level)
    if not 0.0 < level < 1.0:
        raise ValueError("level must be strictly between zero and one")
    z = float(norm.ppf(0.5 + level / 2.0))
    p = successes / total
    den = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / den
    half = z * np.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total**2)) / den
    return float(center - half), float(center + half)


def _validate_null_inputs(
    root_V: np.ndarray,
    rng: np.random.Generator,
    pc1: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be a numpy.random.Generator")
    root = np.asarray(root_V, dtype=np.float64)
    pc = np.asarray(pc1, dtype=np.float64)
    if root.ndim != 2 or root.shape[0] < 2 or root.shape[1] < 1:
        raise ValueError("root_V must be a two-dimensional factor")
    if not np.all(np.isfinite(root)):
        raise ValueError("root_V must contain only finite values")
    if pc.ndim != 1 or pc.shape[0] != root.shape[0]:
        raise ValueError("pc1 must be a vector matching root_V rows")
    if not np.all(np.isfinite(pc)):
        raise ValueError("pc1 must contain only finite values")
    return root, pc


def draw_null(
    kind: str,
    root_V: np.ndarray,
    rng: np.random.Generator,
    pc1: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Draw one standardized Track B null response with audit metadata.

    ``root_V`` is an ``n x k`` covariance factor and always acts on the
    innovation before any declared sample-level stress transformation. ``pc1``
    is the precomputed first genotype-PC score on those same ``n`` samples.
    """

    if not isinstance(kind, str) or kind not in _NULL_ALIASES:
        raise ValueError(f"unknown null kind: {kind!r}")
    root, pc = _validate_null_inputs(root_V, rng, pc1)
    canonical_kind = _NULL_ALIASES[kind]
    n, rank = root.shape
    metadata: dict[str, Any] = {
        "kind": kind,
        "canonical_kind": canonical_kind,
        "n": n,
        "root_shape": [n, rank],
        "standardized": True,
        "interaction_present": False,
    }
    generation = {
        "gaussian": "standardize(root_V @ standard_normal)",
        "t5": "standardize(root_V @ (student_t5 * sqrt(3/5)))",
        "heteroscedastic_pc1": (
            "standardize(sqrt(1 + 3 * minmax(pc1)) * (root_V @ standard_normal))"
        ),
        "contamination_1pct_6sd": (
            "standardize(standardize(root_V @ standard_normal) + signed_six_sd_outliers)"
        ),
        "additive_only": ("standardize(standardize(root_V @ standard_normal) + standardize(pc1))"),
        "omitted_kernel": (
            "standardize(standardize(root_V @ standard_normal) + signed_standardize(pc1))"
        ),
    }
    metadata["response_generation"] = generation[canonical_kind]

    if canonical_kind == "t5":
        innovations = rng.standard_t(df=5, size=rank) * np.sqrt(3.0 / 5.0)
        response = root @ innovations
        metadata.update(
            {
                "innovation_distribution": "standardized_student_t",
                "degrees_of_freedom": 5,
                "innovation_variance": 1.0,
            }
        )
    else:
        innovations = rng.standard_normal(rank)
        response = root @ innovations
        metadata["innovation_distribution"] = "standard_normal"

    if canonical_kind == "heteroscedastic_pc1":
        pc_range = float(np.ptp(pc))
        if not np.isfinite(pc_range) or pc_range <= 0:
            raise ValueError("pc1 is degenerate; heteroscedastic null is undefined")
        position = (pc - float(pc.min())) / pc_range
        variance_scale = 1.0 + 3.0 * position
        response = response * np.sqrt(variance_scale)
        metadata.update(
            {
                "variance_driver": "pc1_minmax",
                "variance_ratio_max_min": 4.0,
                "variance_scale_min": 1.0,
                "variance_scale_max": 4.0,
            }
        )
    elif canonical_kind == "contamination_1pct_6sd":
        response = standardize(response)
        contamination_count = max(1, int(np.rint(0.01 * n)))
        indices = np.sort(rng.choice(n, size=contamination_count, replace=False))
        signs = rng.choice(np.array([-1.0, 1.0]), size=contamination_count)
        response[indices] += 6.0 * signs
        metadata.update(
            {
                "contamination_count": contamination_count,
                "contamination_fraction": contamination_count / n,
                "contamination_indices": indices.tolist(),
                "contamination_shift_sd": 6.0,
            }
        )
    elif canonical_kind == "additive_only":
        main_effect = standardize(pc)
        response = standardize(response) + main_effect
        metadata.update(
            {
                "main_effect": "pc1",
                "main_effect_scale": 1.0,
            }
        )
    elif canonical_kind == "omitted_kernel":
        omitted_score = standardize(pc)
        coefficient = float(rng.choice(np.array([-1.0, 1.0])))
        response = standardize(response) + coefficient * omitted_score
        metadata.update(
            {
                "misspecified": True,
                "omitted_component": "pc1_rank_one_kernel",
                "omitted_component_coefficient": coefficient,
            }
        )

    return standardize(response), metadata


def _validate_gene_blocks(
    gene_blocks: Mapping[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], list[str], int]:
    if not isinstance(gene_blocks, Mapping) or not 2 <= len(gene_blocks) <= 4:
        raise ValueError("gene_blocks must contain two, three, or four copies")
    raw_labels = list(gene_blocks)
    if any(not isinstance(label, str) or not label for label in raw_labels):
        raise ValueError("copy labels must be non-empty strings")
    labels = sorted(raw_labels)
    checked: dict[str, np.ndarray] = {}
    n: int | None = None
    for label in labels:
        block = np.asarray(gene_blocks[label], dtype=np.float64)
        if block.ndim != 2 or block.shape[0] < 2 or block.shape[1] < 1:
            raise ValueError(f"gene block {label!r} must be a non-empty matrix")
        if not np.all(np.isfinite(block)):
            raise ValueError(f"gene block {label!r} must contain only finite values")
        if n is None:
            n = block.shape[0]
        elif block.shape[0] != n:
            raise ValueError("gene blocks must have matching sample counts")
        checked[label] = block
    assert n is not None
    return checked, labels, n


def _validate_edges(
    edges: Sequence[tuple[str, str]] | None,
    labels: list[str],
) -> list[tuple[str, str]]:
    complete = list(combinations(labels, 2))
    if edges is None:
        return complete
    normalized: list[tuple[str, str]] = []
    for edge in edges:
        if not isinstance(edge, (tuple, list)) or len(edge) != 2:
            raise ValueError("each pair edge must contain exactly two copy labels")
        left, right = edge
        if left not in labels or right not in labels or left == right:
            raise ValueError("pair edge must join two distinct available copies")
        normalized.append(tuple(sorted((left, right))))
    if len(set(normalized)) != len(normalized) or set(normalized) != set(complete):
        raise ValueError("pair edges must be the complete unique copy-pair family")
    return complete


def _select_causal_edges(
    requested: Sequence[tuple[str, str]] | None,
    available: list[tuple[str, str]],
    count: int,
) -> list[tuple[str, str]]:
    if requested is None:
        return available[:count]
    selected: list[tuple[str, str]] = []
    for edge in requested:
        if not isinstance(edge, (tuple, list)) or len(edge) != 2:
            raise ValueError("each causal pair edge must contain two labels")
        normalized = tuple(sorted(edge))
        if normalized not in available or normalized in selected:
            raise ValueError("causal pair edge must be a unique available edge")
        selected.append(normalized)
    if len(selected) != count:
        raise ValueError(f"architecture requires exactly {count} causal pair edges")
    return selected


def _minor_burden(block: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    if np.any((block < 0.0) | (block > 2.0)):
        raise ValueError("minor burden requires finite 0/1/2 dosage values in [0, 2]")
    empirical_af = block.mean(axis=0) / 2.0
    flip = empirical_af > 0.5
    oriented = np.where(flip[None, :], 2.0 - block, block)
    return standardize(oriented.sum(axis=1)), {
        "empirical_af": empirical_af.tolist(),
        "flipped_columns": np.flatnonzero(flip).tolist(),
    }


def _mixed_score(block: np.ndarray) -> np.ndarray:
    signs = np.where(np.arange(block.shape[1]) % 2 == 0, 1.0, -1.0)
    return standardize(block @ signs)


def _pc_scores(block: np.ndarray, dimensions: int) -> np.ndarray:
    centered = block - block.mean(axis=0)
    scale = centered.std(axis=0, ddof=1)
    usable = np.isfinite(scale) & (scale > 0)
    if not np.any(usable):
        raise ValueError("gene block is degenerate after genotype standardization")
    standardized = centered[:, usable] / scale[usable]
    left, singular, _ = np.linalg.svd(standardized, full_matrices=False)
    available = min(dimensions, singular.size)
    if available < 1:
        raise ValueError("gene block has no estimable genotype score")
    return left[:, :available] * singular[:available]


def _edge_signal(
    architecture: str,
    left: np.ndarray,
    right: np.ndarray,
    *,
    left_index: int = 0,
    right_index: int = 0,
) -> tuple[np.ndarray, int, dict[str, Any] | None]:
    if architecture in {"minor_burden_aligned", "multi_edge_group", "mispaired"}:
        left_burden, left_orientation = _minor_burden(left)
        right_burden, right_orientation = _minor_burden(right)
        return (
            standardize(left_burden * right_burden),
            1,
            {
                "left": left_orientation,
                "right": right_orientation,
            },
        )
    if architecture == "pc1_distributed":
        left_pc = standardize(_pc_scores(left, 1)[:, 0])
        right_pc = standardize(_pc_scores(right, 1)[:, 0])
        return standardize(left_pc * right_pc), 1, None
    if architecture == "kernel_multidimensional":
        dimensions = min(3, left.shape[1], right.shape[1])
        left_scores = _pc_scores(left, dimensions)
        right_scores = _pc_scores(right, dimensions)
        dimensions = min(left_scores.shape[1], right_scores.shape[1])
        cross_products = [
            standardize(left_scores[:, index] * right_scores[:, index])
            for index in range(dimensions)
        ]
        return standardize(np.sum(cross_products, axis=0)), dimensions, None
    if architecture == "single_snp_pair":
        if not 0 <= left_index < left.shape[1] or not 0 <= right_index < right.shape[1]:
            raise ValueError("snp index is outside its gene block")
        return (
            standardize(standardize(left[:, left_index]) * standardize(right[:, right_index])),
            1,
            None,
        )
    if architecture == "mixed_sign":
        return standardize(_mixed_score(left) * _mixed_score(right)), 1, None
    raise AssertionError(f"unhandled edge architecture: {architecture}")


def interaction_signal(
    architecture: str,
    gene_blocks: Mapping[str, np.ndarray],
    pair_edges: Sequence[tuple[str, str]] | None = None,
    *,
    causal_edges: Sequence[tuple[str, str]] | None = None,
    snp_indices: Mapping[str, int] | None = None,
    wrong_partner: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build one locked Track B truth signal from pair-edge primitives.

    Three- and four-copy groups expose all three or six pair edges. An
    architecture may select one or two of those edges, but no direct third- or
    fourth-order genotype product is ever formed.
    """

    if architecture not in _ARCHITECTURES:
        raise ValueError(f"unknown interaction architecture: {architecture!r}")
    if snp_indices is not None and architecture != "single_snp_pair":
        raise ValueError("snp_indices is only valid for single_snp_pair")
    if wrong_partner is not None and architecture != "mispaired":
        raise ValueError("wrong_partner is only valid for mispaired")
    blocks, labels, n = _validate_gene_blocks(gene_blocks)
    available = _validate_edges(pair_edges, labels)
    components = {
        "minor_burden_aligned": "minor_burden",
        "pc1_distributed": "pc1",
        "kernel_multidimensional": "kernel_hadamard",
        "single_snp_pair": "single_snp_pair",
        "mixed_sign": "mixed_sign",
        "multi_edge_group": "minor_burden",
        "additive_only": "additive_main_effect",
        "mispaired": "minor_burden",
    }
    metadata: dict[str, Any] = {
        "architecture": architecture,
        "copy_labels": labels,
        "available_pair_edges": [list(edge) for edge in available],
        "available_pair_edge_count": len(available),
        "evidence_component": components[architecture],
        "direct_higher_order_term": False,
        "standardized": True,
        "negative_control": architecture in {"additive_only", "mispaired"},
        "interaction_present": architecture not in {"additive_only", "mispaired"},
    }

    if architecture == "additive_only":
        if causal_edges is not None:
            raise ValueError("additive_only has no causal pair edges")
        main_effects = [standardize(_pc_scores(blocks[label], 1)[:, 0]) for label in labels]
        signal = standardize(np.sum(main_effects, axis=0))
        metadata.update(
            {
                "causal_pair_edges": [],
                "causal_pair_edge_count": 0,
                "main_effect_copies": labels,
            }
        )
        return signal, metadata

    if architecture == "mispaired":
        if causal_edges is not None:
            raise ValueError("mispaired has no causal pair edges within the tested group")
        if wrong_partner is None:
            raise ValueError("mispaired architecture requires a predeclared wrong_partner")
        wrong = np.asarray(wrong_partner, dtype=np.float64)
        if wrong.ndim != 2 or wrong.shape[0] != n or wrong.shape[1] < 1:
            raise ValueError("wrong_partner must be a non-empty matrix with matching samples")
        if not np.all(np.isfinite(wrong)):
            raise ValueError("wrong_partner must contain only finite values")
        reference_copy = labels[0]
        signal, dimensions, orientation = _edge_signal(architecture, blocks[reference_copy], wrong)
        assert orientation is not None
        metadata.update(
            {
                "causal_pair_edges": [],
                "causal_pair_edge_count": 0,
                "reference_copy": reference_copy,
                "partner_source": "predeclared_wrong_homoeolog",
                "out_of_group_interaction_present": True,
                "score_dimensions": dimensions,
                "minor_allele_orientation": {
                    "rule": "flip_to_2_minus_dosage_when_empirical_af_gt_0.5",
                    "by_edge": [
                        {
                            "edge": None,
                            "reference_copy": reference_copy,
                            "partner_source": "predeclared_wrong_homoeolog",
                            "empirical_af": {
                                reference_copy: orientation["left"]["empirical_af"],
                                "wrong_partner": orientation["right"]["empirical_af"],
                            },
                            "flipped_columns": {
                                reference_copy: orientation["left"]["flipped_columns"],
                                "wrong_partner": orientation["right"]["flipped_columns"],
                            },
                        }
                    ],
                },
            }
        )
        return signal, metadata

    if architecture == "multi_edge_group":
        if len(available) < 2:
            raise ValueError("multi_edge_group requires a three- or four-copy group")
        selected = _select_causal_edges(causal_edges, available, 2)
    else:
        selected = _select_causal_edges(causal_edges, available, 1)

    index_map: dict[str, int] = {}
    if snp_indices is not None:
        if not isinstance(snp_indices, Mapping):
            raise ValueError("snp_indices must map copy labels to column indices")
        for label, index in snp_indices.items():
            if (
                label not in labels
                or isinstance(index, bool)
                or not isinstance(index, (int, np.integer))
            ):
                raise ValueError("snp_indices contains an invalid copy or index")
            index_map[label] = int(index)

    edge_signals: list[np.ndarray] = []
    dimensions_by_edge: list[int] = []
    orientation_by_edge: list[dict[str, Any]] = []
    for left_label, right_label in selected:
        left_index = index_map.get(left_label, 0)
        right_index = index_map.get(right_label, 0)
        edge_signal, dimensions, orientation = _edge_signal(
            architecture,
            blocks[left_label],
            blocks[right_label],
            left_index=left_index,
            right_index=right_index,
        )
        edge_signals.append(edge_signal)
        dimensions_by_edge.append(dimensions)
        if orientation is not None:
            orientation_by_edge.append(
                {
                    "edge": [left_label, right_label],
                    "empirical_af": {
                        left_label: orientation["left"]["empirical_af"],
                        right_label: orientation["right"]["empirical_af"],
                    },
                    "flipped_columns": {
                        left_label: orientation["left"]["flipped_columns"],
                        right_label: orientation["right"]["flipped_columns"],
                    },
                }
            )
    signal = standardize(np.sum(edge_signals, axis=0))
    metadata.update(
        {
            "causal_pair_edges": [list(edge) for edge in selected],
            "causal_pair_edge_count": len(selected),
            "score_dimensions_by_edge": dimensions_by_edge,
        }
    )
    if orientation_by_edge:
        metadata["minor_allele_orientation"] = {
            "rule": "flip_to_2_minus_dosage_when_empirical_af_gt_0.5",
            "by_edge": orientation_by_edge,
        }
    if architecture == "single_snp_pair":
        used_labels = {label for edge in selected for label in edge}
        used_indices = {label: index_map.get(label, 0) for label in sorted(used_labels)}
        metadata.update(
            {
                "snp_indices": used_indices,
                "snp_count_per_copy": {label: 1 for label in sorted(used_labels)},
            }
        )
    return signal, metadata
