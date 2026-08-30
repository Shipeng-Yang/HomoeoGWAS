"""Matched localizing-method score banks for the v2.0.1 benchmark."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral
from types import MappingProxyType

import numpy as np
from scipy.linalg import orth

from homoeogwas.group_family import (
    EdgeRecord,
    ExpandedEdgeFamily,
    MasterGroupFamily,
    expand_pair_edges,
)
from homoeogwas.omnib_family import (
    OmniBFamilyScores,
    bootstrap_minp_calibration,
    score_omnib_responses,
)

METHOD_NAMES = (
    "omnib",
    "minor_burden",
    "pc1",
    "kernel_hadamard",
    "legacy_burden_product",
    "snpxsnp",
)


@dataclass(frozen=True)
class MethodScoreBank:
    """One frozen group family scored over one shared response bank.

    Matrices are ``group x response``. NaN is reserved for an unestimable
    statistic; infinities and finite values outside the probability range are
    rejected. Arrays and mappings are copied and made read-only so later
    benchmark stages cannot silently change the audited bank.
    """

    family_ids: tuple[str, ...]
    p_by_method: Mapping[str, np.ndarray]
    tested_family_sizes: Mapping[str, int]

    def __post_init__(self) -> None:
        if not isinstance(self.family_ids, tuple):
            raise ValueError("family_ids must be a tuple")
        family_ids = tuple(self.family_ids)
        if not family_ids or any(
            not isinstance(family_id, str) or not family_id
            for family_id in family_ids
        ):
            raise ValueError("family_ids must be non-empty strings")
        if len(set(family_ids)) != len(family_ids):
            raise ValueError("family_ids must be unique")
        if not isinstance(self.p_by_method, Mapping):
            raise ValueError("p_by_method must be a method-to-array mapping")
        if set(self.p_by_method) != set(METHOD_NAMES):
            raise ValueError(
                "p_by_method method names must match the locked comparator set"
            )
        if not isinstance(self.tested_family_sizes, Mapping):
            raise ValueError(
                "tested_family_sizes must be a method-to-count mapping"
            )
        if set(self.tested_family_sizes) != set(METHOD_NAMES):
            raise ValueError(
                "tested_family_sizes method names must match p_by_method"
            )

        checked_p: dict[str, np.ndarray] = {}
        response_count: int | None = None
        for method in METHOD_NAMES:
            raw = self.p_by_method[method]
            if not isinstance(raw, np.ndarray) or not np.issubdtype(
                raw.dtype, np.number
            ) or np.issubdtype(raw.dtype, np.bool_) or np.iscomplexobj(raw):
                raise ValueError(f"{method} scores must be a real numeric ndarray")
            values = np.array(raw, dtype=float, copy=True)
            if values.ndim != 2:
                raise ValueError(f"{method} scores must be a two-dimensional array")
            if values.shape[0] != len(family_ids):
                raise ValueError(
                    f"{method} score row count must match family_ids"
                )
            if values.shape[1] < 1:
                raise ValueError("method score banks require at least one response")
            if response_count is None:
                response_count = values.shape[1]
            elif values.shape[1] != response_count:
                raise ValueError(
                    "all method score arrays must have the same response count"
                )
            if np.isinf(values).any():
                raise ValueError(
                    f"{method} scores may use NaN for unestimable values, not infinity"
                )
            finite = values[np.isfinite(values)]
            if np.any((finite < 0.0) | (finite > 1.0)):
                raise ValueError(f"{method} finite p-values must lie in [0, 1]")
            values.setflags(write=False)
            checked_p[method] = values

        checked_sizes: dict[str, int] = {}
        for method in METHOD_NAMES:
            size = self.tested_family_sizes[method]
            if isinstance(size, bool) or not isinstance(size, Integral) or int(size) < 1:
                raise ValueError("tested family sizes must be positive integers")
            checked_sizes[method] = int(size)

        object.__setattr__(self, "family_ids", family_ids)
        object.__setattr__(self, "p_by_method", MappingProxyType(checked_p))
        object.__setattr__(
            self, "tested_family_sizes", MappingProxyType(checked_sizes)
        )


def _validate_expanded_structure(
    expanded: ExpandedEdgeFamily,
    edge_count: int,
) -> None:
    if len(expanded.edges) != edge_count:
        raise ValueError("expanded edge axis does not match the score matrix")
    if any(not isinstance(edge, EdgeRecord) for edge in expanded.edges):
        raise ValueError("expanded edges must contain EdgeRecord values")
    if not expanded.group_edge_indices:
        raise ValueError("expanded family must contain at least one group")
    for edge_indices in expanded.group_edge_indices:
        if not edge_indices:
            raise ValueError("every expanded group must have non-empty edge indices")
        if any(
            isinstance(index, bool) or not isinstance(index, Integral)
            for index in edge_indices
        ):
            raise ValueError("expanded group edge indices must be integers")
        selected = tuple(int(index) for index in edge_indices)
        if len(set(selected)) != len(selected):
            raise ValueError("expanded group edge indices must be unique within a group")
        if any(index < 0 or index >= edge_count for index in selected):
            raise ValueError("expanded group edge index is outside the edge axis")


def _validate_frozen_family(
    family: MasterGroupFamily,
    expanded: ExpandedEdgeFamily,
) -> None:
    if expanded != expand_pair_edges(family):
        raise ValueError("expanded edges do not match the declared master group family")
    _validate_expanded_structure(expanded, len(expanded.edges))


def _response_matrix(
    responses: np.ndarray,
    sample_count: int,
    *,
    name: str,
    allow_empty: bool = False,
) -> np.ndarray:
    values = np.asarray(responses, dtype=float)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if values.ndim != 2 or values.shape[0] != sample_count:
        raise ValueError(f"{name} must be sample-by-response and aligned to scores.W")
    if not allow_empty and values.shape[1] < 1:
        raise ValueError(f"{name} must contain at least one response")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values")
    return values


def _validate_score_context(scores: OmniBFamilyScores) -> int:
    W = np.asarray(scores.W, dtype=float)
    if (
        W.ndim != 2
        or W.shape[0] < 2
        or W.shape[0] != W.shape[1]
        or not np.all(np.isfinite(W))
    ):
        raise ValueError("scores.W must be a finite square whitening matrix")
    design = scores.null_design
    if design is None:
        raise ValueError("scores must retain the frozen null design")
    design = np.asarray(design, dtype=float)
    if (
        design.ndim != 2
        or design.shape[0] != W.shape[0]
        or design.shape[1] < 1
        or not np.all(np.isfinite(design))
    ):
        raise ValueError("scores.null_design must be a finite aligned design matrix")
    return int(W.shape[0])


def group_component_p(
    edge_components: np.ndarray,
    expanded: ExpandedEdgeFamily,
    component_index: int,
) -> np.ndarray:
    """Aggregate one production omniB component over frozen group edges."""
    from homoeogwas.interact import acat

    values = np.asarray(edge_components, dtype=float)
    if values.ndim != 3:
        raise ValueError(
            "edge_components must be a three-dimensional edge/component/response array"
        )
    _validate_expanded_structure(expanded, values.shape[0])
    if (
        isinstance(component_index, bool)
        or not isinstance(component_index, Integral)
        or not 0 <= int(component_index) < values.shape[1]
    ):
        raise ValueError("component_index is outside the component axis")
    component_index = int(component_index)
    out = np.full((len(expanded.group_edge_indices), values.shape[2]), np.nan)
    for group_index, edge_indices in enumerate(expanded.group_edge_indices):
        selected = np.asarray(edge_indices, dtype=int)
        if selected.size == 1:
            out[group_index] = values[selected[0], component_index]
        else:
            out[group_index] = [
                acat(values[selected, component_index, column])
                for column in range(values.shape[2])
            ]
    return out


def score_legacy_burden_product(
    scores: OmniBFamilyScores,
    family: MasterGroupFamily,
    expanded: ExpandedEdgeFamily,
    responses: np.ndarray,
    *,
    n_jobs: int = 8,
) -> tuple[np.ndarray, int]:
    """Score the sign-coherent minor-burden product on the frozen null fit."""
    _validate_frozen_family(family, expanded)
    _edge_p, _group_p, components = score_omnib_responses(
        scores, family, expanded, responses, n_jobs=n_jobs
    )
    return (
        group_component_p(components, expanded, component_index=0),
        len(family.group_ids),
    )


def _whiten_columns(W: np.ndarray, responses: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [W @ responses[:, column] for column in range(responses.shape[1])]
    )


def _nested_snp_product_design(
    W: np.ndarray,
    Cw: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    reduced = np.column_stack((Cw, W @ left, W @ right))
    added = W @ (left * right)[:, None]
    Qr = orth(reduced)
    residual = added - Qr @ (Qr.T @ added) if Qr.size else added
    Qa = orth(residual)
    dfd = reduced.shape[0] - Qr.shape[1] - Qa.shape[1]
    if Qa.shape[1] < 1 or dfd < 1:
        return None
    return reduced, added


def score_snpxsnp_family(
    scores: OmniBFamilyScores,
    family: MasterGroupFamily,
    expanded: ExpandedEdgeFamily,
    gene_blocks: Mapping[tuple[str, str], np.ndarray],
    responses: np.ndarray,
    *,
    calibration_responses: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Empirically correct every tested SNP product before forming group p.

    ``gene_blocks`` contains sample-aligned, already gated dosage matrices for
    exactly the genes in ``expanded``. The returned integer is the number of
    genotype-estimable SNP-pair products in the complete tested family.
    """
    from homoeogwas.interact import _batch_nested_f

    _validate_frozen_family(family, expanded)
    sample_count = _validate_score_context(scores)
    target = _response_matrix(responses, sample_count, name="responses")
    calibration = _response_matrix(
        calibration_responses,
        sample_count,
        name="calibration_responses",
    )
    if not isinstance(gene_blocks, Mapping):
        raise ValueError("gene_blocks must map (subgenome, gene) to ndarrays")
    required = {
        (edge.sub_x, edge.gene_x) for edge in expanded.edges
    } | {
        (edge.sub_y, edge.gene_y) for edge in expanded.edges
    }
    missing = sorted(required - set(gene_blocks))
    if missing:
        raise ValueError(f"missing genotype block for {missing[0]!r}")
    extra = sorted(set(gene_blocks) - required)
    if extra:
        raise ValueError(f"unexpected genotype block for {extra[0]!r}")

    checked_blocks: dict[tuple[str, str], np.ndarray] = {}
    for key in sorted(required):
        block = gene_blocks[key]
        if not isinstance(block, np.ndarray) or not np.issubdtype(
            block.dtype, np.number
        ) or np.issubdtype(block.dtype, np.bool_) or np.iscomplexobj(block):
            raise ValueError(f"genotype block {key!r} must be a real numeric ndarray")
        values = np.asarray(block, dtype=float)
        if values.ndim != 2 or values.shape[0] != sample_count or values.shape[1] < 1:
            raise ValueError(
                f"genotype block {key!r} must be a non-empty sample-by-SNP matrix"
            )
        if not np.all(np.isfinite(values)) or np.any((values < 0.0) | (values > 2.0)):
            raise ValueError(
                f"genotype block {key!r} must contain finite dosages in [0, 2]"
            )
        checked_blocks[key] = values

    W = np.asarray(scores.W, dtype=float)
    Cw = W @ np.asarray(scores.null_design, dtype=float)
    target_w = _whiten_columns(W, target)
    calibration_w = _whiten_columns(W, calibration)
    groups_by_edge: list[tuple[int, ...]] = []
    for edge_index in range(len(expanded.edges)):
        groups_by_edge.append(tuple(
            group_index
            for group_index, edge_indices in enumerate(expanded.group_edge_indices)
            if edge_index in edge_indices
        ))

    target_rows: list[np.ndarray] = []
    calibration_rows: list[np.ndarray] = []
    raw_group_membership: list[tuple[int, ...]] = []
    for edge_index, edge in enumerate(expanded.edges):
        left = checked_blocks[(edge.sub_x, edge.gene_x)]
        right = checked_blocks[(edge.sub_y, edge.gene_y)]
        for left_index in range(left.shape[1]):
            for right_index in range(right.shape[1]):
                design = _nested_snp_product_design(
                    W, Cw, left[:, left_index], right[:, right_index]
                )
                if design is None:
                    continue
                reduced, added = design
                target_rows.append(_batch_nested_f(target_w, reduced, added))
                calibration_rows.append(
                    _batch_nested_f(calibration_w, reduced, added)
                )
                raw_group_membership.append(groups_by_edge[edge_index])

    tested_family_count = len(target_rows)
    if tested_family_count < 1:
        raise ValueError("SNPxSNP family has no genotype-estimable tested products")
    target_p = np.asarray(target_rows, dtype=float)
    calibration_p = np.asarray(calibration_rows, dtype=float)
    group_p = np.full((len(family.group_ids), target.shape[1]), np.nan)
    pair_indices_by_group = [
        np.asarray(
            [
                pair_index
                for pair_index, memberships in enumerate(raw_group_membership)
                if group_index in memberships
            ],
            dtype=int,
        )
        for group_index in range(len(family.group_ids))
    ]
    for response_index in range(target.shape[1]):
        observed = target_p[:, response_index]
        if not np.all(np.isfinite(observed)):
            continue
        calibration_result = bootstrap_minp_calibration(
            observed, calibration_p, alpha=0.0
        )
        adjusted = np.asarray(calibration_result["adjusted_p_local"], dtype=float)
        for group_index, pair_indices in enumerate(pair_indices_by_group):
            if pair_indices.size:
                group_p[group_index, response_index] = float(
                    np.min(adjusted[pair_indices])
                )
    return group_p, tested_family_count


__all__ = [
    "METHOD_NAMES",
    "MethodScoreBank",
    "group_component_p",
    "score_legacy_burden_product",
    "score_snpxsnp_family",
]
