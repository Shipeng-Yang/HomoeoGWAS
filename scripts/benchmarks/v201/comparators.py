"""Matched localizing-method score banks for the v2.0.1 benchmark."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral
from types import MappingProxyType
from typing import Any

import numpy as np
from scipy.linalg import orth

from homoeogwas.diagnostics import boundary_lrt, compare_nested_reml
from homoeogwas.group_family import (
    EdgeRecord,
    ExpandedEdgeFamily,
    MasterGroupFamily,
    expand_pair_edges,
)
from homoeogwas.kernel import hadamard_kernel, normalize_kernel
from homoeogwas.omnib_family import (
    OmniBFamilyScores,
    score_omnib_responses,
)

METHOD_NAMES = (
    "omnib",
    "minor_burden",
    "pc1",
    "kernel_hadamard",
    "snpxsnp",
)
GLOBAL_VC_METHOD = "global_hadamard_variance_component"


def _numeric_array_hash(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(repr(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def score_global_hadamard_vc(
    responses: np.ndarray,
    fixed_effects: np.ndarray,
    subgenome_grms: Mapping[str, np.ndarray],
    *,
    fit_kwargs: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Fit one genuine global K_hom variance-component test per response.

    This is deliberately separate from :data:`METHOD_NAMES`: it tests a
    single global variance component and cannot localize a homoeolog group.
    The null contains every frozen additive subgenome GRM; the alternative
    adds the trace-normalized Hadamard product and uses the published nested
    REML + Self--Liang boundary LRT implementation.
    """

    if not isinstance(subgenome_grms, Mapping) or len(subgenome_grms) < 2:
        raise ValueError("global Hadamard VC requires at least two subgenome GRMs")
    labels = tuple(subgenome_grms)
    if len(set(labels)) != len(labels) or any(
        not isinstance(label, str) or not label for label in labels
    ):
        raise ValueError("subgenome GRM labels must be unique non-empty strings")
    checked = {
        label: normalize_kernel(np.asarray(subgenome_grms[label], dtype=float), mode="trace")
        for label in labels
    }
    n = next(iter(checked.values())).shape[0]
    if any(value.shape != (n, n) for value in checked.values()):
        raise ValueError("subgenome GRMs must be aligned square matrices")
    Y = np.asarray(responses, dtype=float)
    if Y.ndim == 1:
        Y = Y[:, None]
    X = np.asarray(fixed_effects, dtype=float)
    if Y.ndim != 2 or Y.shape[0] != n or X.ndim != 2 or X.shape[0] != n:
        raise ValueError("responses/fixed_effects must align to subgenome GRMs")
    if not np.all(np.isfinite(Y)) or not np.all(np.isfinite(X)):
        raise ValueError("global Hadamard VC inputs must be finite")
    khom = normalize_kernel(hadamard_kernel(checked), mode="trace")
    null_name = "+".join(labels) + "+e"
    alt_name = "+".join(labels) + "+hom+e"
    model_specs = {
        null_name: list(labels),
        alt_name: [*labels, "hom"],
    }
    kernels = {**checked, "hom": khom}
    p_values: list[float] = []
    statistics: list[float] = []
    convergence: list[bool] = []
    lrt_evidence: list[dict[str, object]] = []
    failed_response_indices: list[int] = []
    for column in range(Y.shape[1]):
        try:
            comparison = compare_nested_reml(
                Y[:, column], X, kernels, model_specs=model_specs,
                fit_kwargs=dict(fit_kwargs or {}),
            )
            test = boundary_lrt(comparison, null_name, alt_name)
            record: dict[str, object] = {
                "status": "completed",
                "error_type": None,
                "message": None,
                "null_model": test.null_model,
                "alt_model": test.alt_model,
                "ll_null": float(test.ll_null),
                "ll_alt": float(test.ll_alt),
                "statistic": float(test.statistic),
                "statistic_raw": float(test.statistic_raw),
                "df_added": int(test.df_added),
                "p_naive": float(test.p_naive),
                "p_mixture": float(test.p_mixture),
                "mixture_weights": dict(test.mixture_weights),
                "added_components": list(test.added_components),
                "null_boundary_components": list(test.null_boundary_components),
                "is_nested": bool(test.is_nested),
                "clipped": bool(test.clipped),
                "both_converged": bool(test.both_converged),
                "boundary_method": test.boundary_method,
                "bootstrap_p": test.bootstrap_p,
            }
            if not test.both_converged or test.clipped:
                record.update(
                    status="failed",
                    error_type=("ClippedBoundaryLRT" if test.clipped
                                else "NonConvergedNestedREML"),
                    message="global Hadamard VC nested REML was not cleanly estimable",
                )
                raise RuntimeError(str(record["message"]))
            p_values.append(float(test.p_mixture))
            statistics.append(float(test.statistic))
            convergence.append(True)
            lrt_evidence.append(record)
        except Exception as error:
            failed_response_indices.append(column)
            p_values.append(float("nan"))
            statistics.append(float("nan"))
            convergence.append(False)
            if "record" not in locals() or record.get("status") != "failed":
                record = {
                    "status": "failed", "error_type": type(error).__name__,
                    "message": str(error), "null_model": null_name,
                    "alt_model": alt_name, "both_converged": False,
                    "clipped": False,
                }
            lrt_evidence.append(record)
        finally:
            if "record" in locals():
                del record
    return {
        "method": GLOBAL_VC_METHOD,
        "hypothesis_unit": "global",
        "detection_only": True,
        "p_values": p_values,
        "lrt_statistics": statistics,
        "both_converged": convergence,
        "failed_response_indices": failed_response_indices,
        "lrt_evidence": lrt_evidence,
        "kernel_manifest": {
            "construction": "hadamard_product",
            "normalization": "trace",
            "subgenomes": list(labels),
            "additive_kernel_sha256": {
                label: _numeric_array_hash(value) for label, value in checked.items()
            },
            "global_hadamard_sha256": _numeric_array_hash(khom),
        },
    }


@dataclass(frozen=True)
class MethodScoreBank:
    """One frozen group family scored over one shared response bank.

    Matrices are ``group x response``. NaN is reserved for an unestimable
    statistic; infinities and finite values outside the probability range are
    rejected. Arrays and mappings are copied and made read-only so later
    benchmark stages cannot silently change the audited bank.

    ``tested_family_sizes`` is provenance, not a multiplicity denominator:
    group-scoring methods record the number of groups, while ``snpxsnp``
    records the number of genotype-estimable raw SNP pairs whose minimum is
    calibrated by the independent outer response bank.
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


def _payload_hash(value: object) -> str:
    from .contracts import sha256_payload

    return sha256_payload(value)


def _snpxsnp_member_digest_update(
    digest: Any, member_id: str, group_membership: tuple[int, ...],
) -> None:
    encoded = member_id.encode("utf-8")
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)
    digest.update(len(group_membership).to_bytes(8, "big"))
    for group_index in group_membership:
        digest.update(int(group_index).to_bytes(8, "big", signed=False))


def _snpxsnp_member_family_hash(
    member_ids: tuple[str, ...],
    group_memberships: tuple[tuple[int, ...], ...],
) -> str:
    digest = hashlib.sha256(b"homoeogwas-snpxsnp-member-family-v1\0")
    for member_id, membership in zip(member_ids, group_memberships, strict=True):
        _snpxsnp_member_digest_update(digest, member_id, membership)
    return digest.hexdigest()


def _is_lower_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


_SNPXSNP_INPUT_BLOCK_FIELDS = {
    "subgenome",
    "gene_id",
    "sample_count",
    "variant_count",
    "source_column_indices_encoding",
    "source_column_indices_sha256",
    "dosage_encoding",
    "dosage_sha256",
    "binding_sha256",
}


def _snpxsnp_input_block_binding(
    key: tuple[str, str],
    source_columns: np.ndarray,
    dosage: np.ndarray,
) -> dict[str, object]:
    columns = np.ascontiguousarray(source_columns, dtype="<i8")
    values = np.ascontiguousarray(dosage, dtype="<f8")
    identity: dict[str, object] = {
        "subgenome": str(key[0]),
        "gene_id": str(key[1]),
        "sample_count": int(values.shape[0]),
        "variant_count": int(values.shape[1]),
        "source_column_indices_encoding": "little_endian_int64_c_order",
        "source_column_indices_sha256": hashlib.sha256(
            columns.tobytes(order="C")
        ).hexdigest(),
        "dosage_encoding": "little_endian_float64_c_order",
        "dosage_sha256": hashlib.sha256(
            values.tobytes(order="C")
        ).hexdigest(),
    }
    identity["binding_sha256"] = _payload_hash(identity)
    return identity


def _snpxsnp_input_family_hash(
    bindings: tuple[Mapping[str, object], ...],
) -> str:
    return _payload_hash({
        "schema": "homoeogwas-snpxsnp-input-family-v1",
        "blocks": [dict(record) for record in bindings],
    })


@dataclass(frozen=True)
class SNPxSNPScoreResult:
    """Streaming raw SNP-product minima and their complete family provenance."""

    group_p: np.ndarray
    argmin_member_index: np.ndarray
    member_ids: tuple[str, ...]
    group_memberships: tuple[tuple[int, ...], ...]
    member_family_sha256: str
    input_block_bindings: tuple[Mapping[str, object], ...]
    input_family_sha256: str
    offered_pair_count: int
    design_nonestimable_pair_count: int
    tested_pair_count: int
    offered_pair_count_by_group: tuple[int, ...]
    design_nonestimable_pair_count_by_group: tuple[int, ...]
    tested_pair_count_by_group: tuple[int, ...]
    nonfinite_pair_score_count: int
    nonfinite_pair_score_count_by_group: tuple[int, ...]
    failed_response_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        group_p = np.array(self.group_p, dtype=float, copy=True)
        argmin = np.array(self.argmin_member_index, dtype=np.int64, copy=True)
        member_ids = tuple(self.member_ids)
        memberships: list[tuple[int, ...]] = []
        for membership in self.group_memberships:
            if (
                not membership
                or any(
                    isinstance(index, bool) or not isinstance(index, Integral)
                    for index in membership
                )
            ):
                raise ValueError("invalid SNPxSNP member group membership")
            memberships.append(tuple(int(index) for index in membership))
        group_memberships = tuple(memberships)
        try:
            input_bindings = tuple(dict(record) for record in self.input_block_bindings)
        except (TypeError, ValueError) as error:
            raise ValueError("invalid SNPxSNP input-block bindings") from error
        block_keys = []
        for record in input_bindings:
            identity = {
                key: value for key, value in record.items()
                if key != "binding_sha256"
            }
            if (
                set(record) != _SNPXSNP_INPUT_BLOCK_FIELDS
                or not isinstance(record["subgenome"], str)
                or not record["subgenome"]
                or not isinstance(record["gene_id"], str)
                or not record["gene_id"]
                or isinstance(record["sample_count"], bool)
                or not isinstance(record["sample_count"], Integral)
                or int(record["sample_count"]) < 1
                or isinstance(record["variant_count"], bool)
                or not isinstance(record["variant_count"], Integral)
                or int(record["variant_count"]) < 1
                or record["source_column_indices_encoding"]
                != "little_endian_int64_c_order"
                or record["dosage_encoding"]
                != "little_endian_float64_c_order"
                or not _is_lower_sha256(record["source_column_indices_sha256"])
                or not _is_lower_sha256(record["dosage_sha256"])
                or record["binding_sha256"] != _payload_hash(identity)
            ):
                raise ValueError("invalid SNPxSNP input-block bindings")
            block_keys.append((record["subgenome"], record["gene_id"]))
        if (
            not input_bindings
            or tuple(block_keys) != tuple(sorted(block_keys))
            or len(block_keys) != len(set(block_keys))
            or not _is_lower_sha256(self.input_family_sha256)
            or self.input_family_sha256
            != _snpxsnp_input_family_hash(input_bindings)
        ):
            raise ValueError("invalid SNPxSNP input-family binding")
        group_count = len(self.offered_pair_count_by_group)
        counts = (
            self.offered_pair_count,
            self.design_nonestimable_pair_count,
            self.tested_pair_count,
            self.nonfinite_pair_score_count,
            *self.offered_pair_count_by_group,
            *self.design_nonestimable_pair_count_by_group,
            *self.tested_pair_count_by_group,
            *self.nonfinite_pair_score_count_by_group,
        )
        if (
            group_p.ndim != 2
            or group_p.shape[0] != group_count
            or group_p.shape[1] < 1
            or argmin.shape != group_p.shape
            or len(self.design_nonestimable_pair_count_by_group) != group_count
            or len(self.tested_pair_count_by_group) != group_count
            or len(self.nonfinite_pair_score_count_by_group) != group_count
            or any(isinstance(value, bool) or not isinstance(value, Integral) or value < 0
                   for value in counts)
            or self.offered_pair_count - self.design_nonestimable_pair_count
            != self.tested_pair_count
            or len(member_ids) != self.tested_pair_count
            or len(group_memberships) != self.tested_pair_count
            or len(set(member_ids)) != len(member_ids)
            or any(not isinstance(member_id, str) or not member_id
                   for member_id in member_ids)
        ):
            raise ValueError("invalid streaming SNPxSNP result dimensions or counts")
        tested_by_group = [0] * group_count
        for membership in group_memberships:
            if (
                len(set(membership)) != len(membership)
                or any(index < 0 or index >= group_count for index in membership)
            ):
                raise ValueError("invalid SNPxSNP member group membership")
            for group_index in membership:
                tested_by_group[group_index] += 1
        if (
            tuple(tested_by_group) != tuple(self.tested_pair_count_by_group)
            or any(
                offered - nonestimable != tested
                for offered, nonestimable, tested in zip(
                    self.offered_pair_count_by_group,
                    self.design_nonestimable_pair_count_by_group,
                    self.tested_pair_count_by_group,
                    strict=True,
                )
            )
            or self.nonfinite_pair_score_count
            > self.tested_pair_count * group_p.shape[1]
            or any(
                nonfinite > tested * group_p.shape[1]
                for nonfinite, tested in zip(
                    self.nonfinite_pair_score_count_by_group,
                    self.tested_pair_count_by_group,
                    strict=True,
                )
            )
        ):
            raise ValueError("invalid streaming SNPxSNP per-group counts")
        finite = np.isfinite(group_p)
        if (
            np.isinf(group_p).any()
            or np.any((group_p[finite] < 0.0) | (group_p[finite] > 1.0))
            or np.any((argmin[finite] < 0) | (argmin[finite] >= len(self.member_ids)))
            or np.any(argmin[~finite] != -1)
        ):
            raise ValueError("invalid streaming SNPxSNP scores or argmins")
        for group_index, response_index in np.argwhere(finite):
            if group_index not in group_memberships[argmin[group_index, response_index]]:
                raise ValueError("SNPxSNP argmin does not belong to its group")
        if any(
            isinstance(index, bool) or not isinstance(index, Integral)
            for index in self.failed_response_indices
        ):
            raise ValueError("invalid streaming SNPxSNP failure or member identity")
        failed = tuple(int(index) for index in self.failed_response_indices)
        if (
            len(failed) != len(set(failed))
            or tuple(sorted(failed)) != failed
            or any(index < 0 or index >= group_p.shape[1] for index in failed)
            or failed != tuple(np.flatnonzero(~finite.any(axis=0)).astype(int))
            or bool(failed) != bool(self.nonfinite_pair_score_count)
            or self.member_family_sha256
            != _snpxsnp_member_family_hash(member_ids, group_memberships)
        ):
            raise ValueError("invalid streaming SNPxSNP failure or member identity")
        group_p.setflags(write=False)
        argmin.setflags(write=False)
        object.__setattr__(self, "group_p", group_p)
        object.__setattr__(self, "argmin_member_index", argmin)
        object.__setattr__(self, "member_ids", member_ids)
        object.__setattr__(self, "group_memberships", group_memberships)
        object.__setattr__(
            self,
            "input_block_bindings",
            tuple(MappingProxyType(record) for record in input_bindings),
        )
        object.__setattr__(self, "failed_response_indices", failed)

    def evidence_payload(self) -> dict[str, object]:
        return {
            "schema": "snpxsnp_raw_stream_v2",
            "hypothesis_unit": "snp_pair_within_group",
            "member_ids": list(self.member_ids),
            "group_memberships": [list(value) for value in self.group_memberships],
            "member_family_sha256": self.member_family_sha256,
            "input_block_bindings": [
                dict(record) for record in self.input_block_bindings
            ],
            "input_family_sha256": self.input_family_sha256,
            "argmin_member_index": self.argmin_member_index.tolist(),
            "offered_pair_count": self.offered_pair_count,
            "design_nonestimable_pair_count": self.design_nonestimable_pair_count,
            "tested_pair_count": self.tested_pair_count,
            "offered_pair_count_by_group": list(self.offered_pair_count_by_group),
            "design_nonestimable_pair_count_by_group": list(
                self.design_nonestimable_pair_count_by_group
            ),
            "tested_pair_count_by_group": list(self.tested_pair_count_by_group),
            "nonfinite_pair_score_count": self.nonfinite_pair_score_count,
            "nonfinite_pair_score_count_by_group": list(
                self.nonfinite_pair_score_count_by_group
            ),
            "failed_response_indices": list(self.failed_response_indices),
        }


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
    _edge_p, _group_p, components, diagnostics = score_omnib_responses(
        scores,
        family,
        expanded,
        responses,
        n_jobs=n_jobs,
        return_diagnostics=True,
    )
    values = group_component_p(components, expanded, component_index=0)
    failed = np.asarray(
        diagnostics.failed_response_indices_by_component["minor_burden"],
        dtype=int,
    )
    if failed.size:
        values[:, failed] = np.nan
    return values, len(family.group_ids)


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
    max_offered_pairs: int,
) -> SNPxSNPScoreResult:
    """Stream raw nested-F pair scores into group minima.

    ``gene_blocks`` contains sample-aligned, already gated dosage matrices for
    exactly the genes in ``expanded``. Pair designs and scores are enumerated
    once; no pair-by-response score matrix is retained.
    """
    from homoeogwas.interact import _batch_nested_f

    _validate_frozen_family(family, expanded)
    sample_count = _validate_score_context(scores)
    target = _response_matrix(responses, sample_count, name="responses")
    if (
        isinstance(max_offered_pairs, bool)
        or not isinstance(max_offered_pairs, Integral)
        or int(max_offered_pairs) < 1
    ):
        raise ValueError("max_offered_pairs must be a positive integer")
    max_offered_pairs = int(max_offered_pairs)
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
    if not isinstance(scores.gated_snp, Mapping):
        raise ValueError("scores must retain gated SNP indices for every genotype block")
    missing_gated = sorted(required - set(scores.gated_snp))
    if missing_gated:
        raise ValueError(f"missing gated SNP indices for {missing_gated[0]!r}")

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

    checked_columns: dict[tuple[str, str], np.ndarray] = {}
    for key in sorted(required):
        columns = np.asarray(scores.gated_snp[key], dtype=int)
        block = checked_blocks[key]
        if (
            columns.shape != (block.shape[1],)
            or np.any(columns < 0)
            or np.unique(columns).size != columns.size
        ):
            raise ValueError("gated SNP indices do not match genotype blocks")
        checked_columns[key] = columns
    input_bindings = tuple(
        _snpxsnp_input_block_binding(
            key, checked_columns[key], checked_blocks[key]
        )
        for key in sorted(required)
    )
    input_family_sha256 = _snpxsnp_input_family_hash(input_bindings)

    groups_by_edge = [
        tuple(
            group_index
            for group_index, edge_indices in enumerate(expanded.group_edge_indices)
            if edge_index in edge_indices
        )
        for edge_index in range(len(expanded.edges))
    ]
    edge_specs: list[
        tuple[EdgeRecord, np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[int, ...]]
    ] = []
    offered_pair_count = 0
    offered_by_group = [0] * len(family.group_ids)
    for edge_index, edge in enumerate(expanded.edges):
        left_key = (edge.sub_x, edge.gene_x)
        right_key = (edge.sub_y, edge.gene_y)
        left = checked_blocks[left_key]
        right = checked_blocks[right_key]
        left_columns = checked_columns[left_key]
        right_columns = checked_columns[right_key]
        memberships = groups_by_edge[edge_index]
        pair_count = int(left.shape[1] * right.shape[1])
        offered_pair_count += pair_count
        for group_index in memberships:
            offered_by_group[group_index] += pair_count
        edge_specs.append(
            (edge, left, right, left_columns, right_columns, memberships)
        )
    if offered_pair_count > max_offered_pairs:
        raise ValueError(
            f"SNPxSNP offered pair ceiling exceeded: "
            f"{offered_pair_count} > {max_offered_pairs}"
        )

    W = np.asarray(scores.W, dtype=float)
    Cw = W @ np.asarray(scores.null_design, dtype=float)
    target_w = _whiten_columns(W, target)
    group_p = np.full((len(family.group_ids), target.shape[1]), np.inf)
    argmin = np.full(group_p.shape, -1, dtype=np.int64)
    member_ids: list[str] = []
    group_memberships: list[tuple[int, ...]] = []
    member_digest = hashlib.sha256(b"homoeogwas-snpxsnp-member-family-v1\0")
    design_nonestimable_pair_count = 0
    design_nonestimable_by_group = [0] * len(family.group_ids)
    tested_by_group = [0] * len(family.group_ids)
    nonfinite_pair_score_count = 0
    nonfinite_by_group = [0] * len(family.group_ids)
    failed_responses: set[int] = set()
    for edge, left, right, left_columns, right_columns, memberships in edge_specs:
        for left_index in range(left.shape[1]):
            for right_index in range(right.shape[1]):
                design = _nested_snp_product_design(
                    W, Cw, left[:, left_index], right[:, right_index]
                )
                if design is None:
                    design_nonestimable_pair_count += 1
                    for group_index in memberships:
                        design_nonestimable_by_group[group_index] += 1
                    continue
                reduced, added = design
                member_id = (
                    f"{edge.edge_id}|{int(left_columns[left_index])}|"
                    f"{int(right_columns[right_index])}"
                )
                member_index = len(member_ids)
                member_ids.append(member_id)
                group_memberships.append(memberships)
                _snpxsnp_member_digest_update(member_digest, member_id, memberships)
                for group_index in memberships:
                    tested_by_group[group_index] += 1
                pair_p = np.asarray(
                    _batch_nested_f(
                        target_w,
                        reduced,
                        added,
                        response_axis_stable=True,
                    ),
                    dtype=float,
                )
                if pair_p.shape != (target.shape[1],):
                    raise RuntimeError("nested SNPxSNP scorer returned an invalid shape")
                finite = np.isfinite(pair_p)
                nonfinite = np.flatnonzero(~finite)
                nonfinite_pair_score_count += int(nonfinite.size)
                failed_responses.update(int(index) for index in nonfinite)
                for group_index in memberships:
                    nonfinite_by_group[group_index] += int(nonfinite.size)
                    current = group_p[group_index]
                    better = finite & (pair_p < current)
                    current[better] = pair_p[better]
                    argmin[group_index, better] = member_index
                    for response_index in np.flatnonzero(finite & (pair_p == current)):
                        old_index = int(argmin[group_index, response_index])
                        if old_index < 0 or member_id < member_ids[old_index]:
                            argmin[group_index, response_index] = member_index

    tested_pair_count = len(member_ids)
    if offered_pair_count - design_nonestimable_pair_count != tested_pair_count:
        raise RuntimeError("SNPxSNP offered/nonestimable/tested count invariant failed")
    if tested_pair_count < 1:
        raise ValueError("SNPxSNP family has no genotype-estimable tested products")
    missing = ~np.isfinite(group_p)
    group_p[missing] = np.nan
    argmin[missing] = -1
    failed_indices = tuple(sorted(failed_responses))
    if failed_indices:
        group_p[:, failed_indices] = np.nan
        argmin[:, failed_indices] = -1
    return SNPxSNPScoreResult(
        group_p=group_p,
        argmin_member_index=argmin,
        member_ids=tuple(member_ids),
        group_memberships=tuple(group_memberships),
        member_family_sha256=member_digest.hexdigest(),
        input_block_bindings=input_bindings,
        input_family_sha256=input_family_sha256,
        offered_pair_count=offered_pair_count,
        design_nonestimable_pair_count=design_nonestimable_pair_count,
        tested_pair_count=tested_pair_count,
        offered_pair_count_by_group=tuple(offered_by_group),
        design_nonestimable_pair_count_by_group=tuple(design_nonestimable_by_group),
        tested_pair_count_by_group=tuple(tested_by_group),
        nonfinite_pair_score_count=nonfinite_pair_score_count,
        nonfinite_pair_score_count_by_group=tuple(nonfinite_by_group),
        failed_response_indices=failed_indices,
    )


__all__ = [
    "GLOBAL_VC_METHOD",
    "METHOD_NAMES",
    "MethodScoreBank",
    "SNPxSNPScoreResult",
    "group_component_p",
    "score_legacy_burden_product",
    "score_snpxsnp_family",
    "score_global_hadamard_vc",
]
