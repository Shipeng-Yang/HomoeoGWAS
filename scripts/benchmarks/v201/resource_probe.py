"""Execution-disabled producer for raw SNPxSNP resource-probe artifacts."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping, Sequence
from numbers import Integral
from typing import Any

import numpy as np

from .comparators import (
    _SNPXSNP_INPUT_BLOCK_FIELDS,
    SNPxSNPScoreResult,
    score_snpxsnp_family,
)
from .contracts import (
    COMPARATOR_PROBE_WIDTHS,
    ComparatorProbeRecordV2,
    _validate_comparator_probe_series_structure,
    canonical_json,
    comparator_resource_limit,
    sha256_payload,
)
from .track_omnib import _context_fingerprint, _snpxsnp_pair_ceiling

_AUTHORIZATION_FIELDS = {
    "schema",
    "authorization_id",
    "implementation_commit",
    "matched_comparator_contract_sha256",
    "panel_context",
    "response_widths",
    "resource_probe_authorized",
    "formal_execution_authorized",
}
_SCORE_EVIDENCE_FIELDS = {
    "schema",
    "panel_id",
    "response_width",
    "response_ids",
    "response_bank_identity",
    "response_prefix_identity",
    "raw_score_evidence",
    "group_p",
}
_RESPONSE_BANK_IDENTITY_FIELDS = {
    "schema", "dtype", "shape", "ordered_response_ids", "array_sha256",
}
_RESPONSE_PREFIX_IDENTITY_FIELDS = {
    "schema", "parent_response_bank_sha256", "response_width", "dtype",
    "shape", "ordered_response_ids", "array_sha256",
}


def _lower_hex(value: Any, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _numeric_array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(repr(array.shape).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _validate_authorization(
    payload: Mapping[str, Any],
    *,
    expected_sha256: str,
    panel_id: str,
    response_width: int,
    implementation_commit: str,
    matched_comparator_contract_sha256: str,
    design_hash: str,
    context_fingerprint: str,
    prepared_design_sha256: str,
    response_bank_sha256: str,
    response_ids_sha256: str,
) -> None:
    if not isinstance(payload, Mapping) or set(payload) != _AUTHORIZATION_FIELDS:
        raise ValueError("resource probe authorization fields differ")
    panel_context = payload["panel_context"]
    widths = payload["response_widths"]
    expected_panel_context = {
        "panel_id": panel_id,
        "design_hash": design_hash,
        "context_fingerprint": context_fingerprint,
        "prepared_design_sha256": prepared_design_sha256,
        "response_bank_sha256": response_bank_sha256,
        "response_ids_sha256": response_ids_sha256,
    }
    if (
        not _lower_hex(expected_sha256, 64)
        or sha256_payload(payload) != expected_sha256
        or payload["schema"]
        != "homoeogwas-snpxsnp-resource-probe-authorization-v1"
        or not isinstance(payload["authorization_id"], str)
        or not payload["authorization_id"]
        or not _lower_hex(implementation_commit, 40)
        or payload["implementation_commit"] != implementation_commit
        or not _lower_hex(matched_comparator_contract_sha256, 64)
        or payload["matched_comparator_contract_sha256"]
        != matched_comparator_contract_sha256
        or panel_context != expected_panel_context
        or not isinstance(widths, list)
        or tuple(widths) != COMPARATOR_PROBE_WIDTHS
        or response_width not in widths
        or payload["formal_execution_authorized"] is not False
    ):
        raise ValueError("resource probe authorization identity or scope is invalid")
    if payload["resource_probe_authorized"] is not True:
        raise ValueError("resource probe is not authorized")


def _response_id_tuple(response_ids: Sequence[str]) -> tuple[str, ...]:
    if isinstance(response_ids, (str, bytes)):
        raise ValueError("response IDs must be an ordered sequence")
    values = tuple(response_ids)
    if (
        len(values) != max(COMPARATOR_PROBE_WIDTHS)
        or any(not isinstance(value, str) or not value for value in values)
        or len(set(values)) != len(values)
    ):
        raise ValueError("resource probe requires 20 unique ordered response IDs")
    return values


def _response_identities(
    response_bank: np.ndarray,
    response_ids: tuple[str, ...],
    response_width: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if (
        not isinstance(response_bank, np.ndarray)
        or response_bank.dtype != np.dtype("float64")
        or response_bank.ndim != 2
        or response_bank.shape[1] != max(COMPARATOR_PROBE_WIDTHS)
        or not response_bank.flags.c_contiguous
        or not np.all(np.isfinite(response_bank))
    ):
        raise ValueError(
            "response bank must be finite C-contiguous float64 with 20 columns"
        )
    prefix = response_bank[:, :response_width]
    if not prefix.flags.c_contiguous:
        prefix = np.ascontiguousarray(prefix)
    bank_identity = {
        "schema": "homoeogwas-snpxsnp-response-bank-v1",
        "dtype": str(response_bank.dtype),
        "shape": list(response_bank.shape),
        "ordered_response_ids": list(response_ids),
        "array_sha256": _numeric_array_sha256(response_bank),
    }
    bank_sha256 = sha256_payload(bank_identity)
    prefix_identity = {
        "schema": "homoeogwas-snpxsnp-response-prefix-v1",
        "parent_response_bank_sha256": bank_sha256,
        "response_width": response_width,
        "dtype": str(prefix.dtype),
        "shape": list(prefix.shape),
        "ordered_response_ids": list(response_ids[:response_width]),
        "array_sha256": _numeric_array_sha256(prefix),
    }
    return bank_identity, prefix_identity


def _gated_marker_counts_from_bindings(bindings: Any) -> dict[str, int]:
    if not isinstance(bindings, (list, tuple)) or not bindings:
        raise ValueError("resource probe input-block bindings are invalid")
    counts = {}
    for binding in bindings:
        if (
            not isinstance(binding, Mapping)
            or set(binding) != _SNPXSNP_INPUT_BLOCK_FIELDS
            or any(
                not isinstance(binding[field], str) or not binding[field]
                for field in ("subgenome", "gene_id")
            )
            or any(
                isinstance(binding[field], bool)
                or not isinstance(binding[field], Integral)
                or binding[field] < 1
                for field in ("sample_count", "variant_count")
            )
            or binding["source_column_indices_encoding"]
            != "little_endian_int64_c_order"
            or binding["dosage_encoding"] != "little_endian_float64_c_order"
            or any(
                not _lower_hex(binding[field], 64)
                for field in (
                    "source_column_indices_sha256", "dosage_sha256", "binding_sha256",
                )
            )
            or binding["binding_sha256"] != sha256_payload({
                key: value for key, value in binding.items() if key != "binding_sha256"
            })
        ):
            raise ValueError("resource probe input-block bindings are invalid")
        key = f"{binding['subgenome']}|{binding['gene_id']}"
        if key in counts:
            raise ValueError("resource probe input-block marker count key is duplicated")
        counts[key] = int(binding["variant_count"])
    return counts


def produce_snpxsnp_resource_probe(
    prepared: Any,
    response_bank: np.ndarray,
    response_ids: Sequence[str],
    *,
    response_width: int,
    design_hash: str,
    context_fingerprint: str,
    authorization_payload: Mapping[str, Any],
    authorization_sha256: str,
    implementation_commit: str,
    matched_comparator_contract_sha256: str,
) -> dict[str, Any]:
    """Score one exact response prefix and emit a validated v2 artifact."""

    if response_width not in COMPARATOR_PROBE_WIDTHS:
        raise ValueError("resource probe width must be 1, 5 or 20")
    context = prepared.context
    panel_id = context.panel_id
    response_ids_tuple = _response_id_tuple(response_ids)
    bank_identity, prefix_identity = _response_identities(
        response_bank, response_ids_tuple, response_width
    )
    family_size = len(context.family.group_ids)
    copies = len(context.family.subgenomes)
    comparator_resource_limit(
        panel_id, family_size=family_size, copies=copies
    )
    if getattr(context, "sample_context", None) != "full":
        raise ValueError("resource probe requires the frozen full sample context")
    prepared_design_sha256 = prepared.scores.prepared_design_sha256
    if not all(
        _lower_hex(value, 64)
        for value in (design_hash, context_fingerprint, prepared_design_sha256)
    ):
        raise ValueError("resource probe design identity is invalid")
    if _context_fingerprint(context) != context_fingerprint:
        raise ValueError("resource probe context fingerprint is detached")
    response_bank_sha256 = sha256_payload(bank_identity)
    response_ids_sha256 = sha256_payload({
        "ordered_response_ids": list(response_ids_tuple),
    })
    _validate_authorization(
        authorization_payload,
        expected_sha256=authorization_sha256,
        panel_id=panel_id,
        response_width=response_width,
        implementation_commit=implementation_commit,
        matched_comparator_contract_sha256=matched_comparator_contract_sha256,
        design_hash=design_hash,
        context_fingerprint=context_fingerprint,
        prepared_design_sha256=prepared_design_sha256,
        response_bank_sha256=response_bank_sha256,
        response_ids_sha256=response_ids_sha256,
    )
    prefix = response_bank[:, :response_width]
    if not prefix.flags.c_contiguous:
        prefix = np.ascontiguousarray(prefix)

    def score() -> SNPxSNPScoreResult:
        return score_snpxsnp_family(
            prepared.scores,
            context.family,
            prepared.expanded,
            prepared.gene_blocks,
            prefix,
            max_offered_pairs=_snpxsnp_pair_ceiling(context),
        )

    # Imported here to keep audit/resource validation independent of CLI import
    # order while retaining the one reviewed measurement implementation.
    from .cli import _measure_comparator_operation

    root_pid = os.getpid()
    result, measurement = _measure_comparator_operation(
        score, declared_worker_pids=(root_pid,)
    )
    if not isinstance(result, SNPxSNPScoreResult):
        raise RuntimeError("resource probe scorer returned an invalid result")
    if (
        result.group_p.shape != (family_size, response_width)
        or result.failed_response_indices
        or result.nonfinite_pair_score_count != 0
        or not np.all(np.isfinite(result.group_p))
    ):
        raise ValueError("resource probe must complete every response")
    score_evidence = {
        "schema": "homoeogwas-snpxsnp-resource-score-evidence-v1",
        "panel_id": panel_id,
        "response_width": response_width,
        "response_ids": list(response_ids_tuple[:response_width]),
        "response_bank_identity": bank_identity,
        "response_prefix_identity": prefix_identity,
        "raw_score_evidence": result.evidence_payload(),
        "group_p": result.group_p.tolist(),
    }
    encoded_score_evidence = canonical_json(score_evidence).encode("utf-8")
    gated_marker_counts = {
        f"{subgenome}|{gene_id}": int(np.asarray(block).shape[1])
        for (subgenome, gene_id), block in sorted(prepared.gene_blocks.items())
    }
    record = ComparatorProbeRecordV2(
        schema="homoeogwas-snpxsnp-resource-probe-v2",
        panel_id=panel_id,
        sample_context="full",
        family_size=family_size,
        copies=copies,
        response_width=response_width,
        design_hash=design_hash,
        context_fingerprint=context_fingerprint,
        prepared_design_sha256=prepared_design_sha256,
        member_family_sha256=result.member_family_sha256,
        scorer_wall_seconds=measurement["scorer_wall_seconds"],
        scorer_cpu_seconds=measurement["scorer_cpu_seconds"],
        peak_parent_rss_bytes=measurement["peak_parent_rss_bytes"],
        peak_aggregate_pss_bytes=measurement["peak_aggregate_pss_bytes"],
        output_bytes=len(encoded_score_evidence),
        offered_pair_count=result.offered_pair_count,
        design_nonestimable_pair_count=result.design_nonestimable_pair_count,
        tested_pair_count=result.tested_pair_count,
        nonfinite_pair_score_count=result.nonfinite_pair_score_count,
        failed_response_indices=result.failed_response_indices,
        gated_marker_count_by_gene=gated_marker_counts,
        requested_jobs=1,
        effective_jobs=1,
        parallel_backend="serial",
        worker_pids=(root_pid,),
        inference_status="noninferential_resource_probe",
        execution_authorized=False,
        probe_authorization_sha256=authorization_sha256,
        implementation_commit=implementation_commit,
        matched_comparator_contract_sha256=matched_comparator_contract_sha256,
        input_family_sha256=result.input_family_sha256,
        response_bank_sha256=response_bank_sha256,
        response_ids=response_ids_tuple,
        response_ids_sha256=response_ids_sha256,
        response_prefix_sha256=sha256_payload(prefix_identity),
        score_evidence_sha256=hashlib.sha256(encoded_score_evidence).hexdigest(),
        root_pid=measurement["root_pid"],
        sampled_pids=tuple(measurement["sampled_pids"]),
        process_set_reconciled=measurement["process_set_reconciled"],
        aggregate_pss_missed_spike_strategy=measurement[
            "aggregate_pss_missed_spike_strategy"
        ],
        formal_execution_authorized=False,
    )
    artifact = {
        "schema": "homoeogwas-snpxsnp-resource-probe-artifact-v1",
        "record": record.to_payload(),
        "score_evidence": score_evidence,
    }
    return validate_snpxsnp_resource_probe_artifact(
        artifact,
        authorization_payload=authorization_payload,
        authorization_sha256=authorization_sha256,
        response_bank=response_bank,
        response_ids=response_ids_tuple,
    )


def validate_snpxsnp_resource_probe_artifact(
    payload: Mapping[str, Any],
    *,
    authorization_payload: Mapping[str, Any],
    authorization_sha256: str,
    response_bank: np.ndarray,
    response_ids: Sequence[str],
) -> dict[str, Any]:
    """Ground a v2 artifact in independent caller-supplied authorization/inputs.

    The witness must come from the trusted caller, never from the artifact.
    Hashes alone cannot establish that a stored prefix was authorized.
    """

    if (
        not isinstance(payload, Mapping)
        or set(payload) != {"schema", "record", "score_evidence"}
        or payload["schema"]
        != "homoeogwas-snpxsnp-resource-probe-artifact-v1"
        or not isinstance(payload["score_evidence"], Mapping)
        or set(payload["score_evidence"]) != _SCORE_EVIDENCE_FIELDS
    ):
        raise ValueError("resource probe artifact fields differ")
    record = ComparatorProbeRecordV2.from_payload(payload["record"])
    expected_ids = _response_id_tuple(response_ids)
    expected_bank, expected_prefix = _response_identities(
        response_bank, expected_ids, record.response_width,
    )
    _validate_authorization(
        authorization_payload,
        expected_sha256=authorization_sha256,
        panel_id=record.panel_id,
        response_width=record.response_width,
        implementation_commit=record.implementation_commit,
        matched_comparator_contract_sha256=record.matched_comparator_contract_sha256,
        design_hash=record.design_hash,
        context_fingerprint=record.context_fingerprint,
        prepared_design_sha256=record.prepared_design_sha256,
        response_bank_sha256=sha256_payload(expected_bank),
        response_ids_sha256=sha256_payload({"ordered_response_ids": list(expected_ids)}),
    )
    evidence = dict(payload["score_evidence"])
    encoded = canonical_json(evidence).encode("utf-8")
    bank_identity = evidence["response_bank_identity"]
    prefix_identity = evidence["response_prefix_identity"]
    raw = evidence["raw_score_evidence"]
    group_p = np.asarray(evidence["group_p"], dtype=float)
    bank_shape = (
        bank_identity.get("shape") if isinstance(bank_identity, Mapping) else None
    )
    prefix_shape = (
        prefix_identity.get("shape")
        if isinstance(prefix_identity, Mapping)
        else None
    )
    sample_count = (
        bank_shape[0]
        if isinstance(bank_shape, list)
        and len(bank_shape) == 2
        and isinstance(bank_shape[0], int)
        and not isinstance(bank_shape[0], bool)
        and bank_shape[0] > 0
        else None
    )
    if (
        record.probe_authorization_sha256 != authorization_sha256
        or record.response_ids != expected_ids
        or bank_identity != expected_bank
        or prefix_identity != expected_prefix
        or evidence["schema"]
        != "homoeogwas-snpxsnp-resource-score-evidence-v1"
        or evidence["panel_id"] != record.panel_id
        or evidence["response_width"] != record.response_width
        or evidence["response_ids"]
        != list(record.response_ids[:record.response_width])
        or not isinstance(bank_identity, Mapping)
        or set(bank_identity) != _RESPONSE_BANK_IDENTITY_FIELDS
        or bank_identity.get("schema") != "homoeogwas-snpxsnp-response-bank-v1"
        or bank_identity.get("dtype") != "float64"
        or bank_shape != [sample_count, max(COMPARATOR_PROBE_WIDTHS)]
        or not _lower_hex(bank_identity.get("array_sha256"), 64)
        or sha256_payload(bank_identity) != record.response_bank_sha256
        or bank_identity.get("ordered_response_ids") != list(record.response_ids)
        or not isinstance(prefix_identity, Mapping)
        or set(prefix_identity) != _RESPONSE_PREFIX_IDENTITY_FIELDS
        or prefix_identity.get("schema")
        != "homoeogwas-snpxsnp-response-prefix-v1"
        or prefix_identity.get("dtype") != "float64"
        or prefix_shape != [sample_count, record.response_width]
        or not _lower_hex(prefix_identity.get("array_sha256"), 64)
        or sha256_payload(prefix_identity) != record.response_prefix_sha256
        or prefix_identity.get("parent_response_bank_sha256")
        != record.response_bank_sha256
        or prefix_identity.get("response_width") != record.response_width
        or prefix_identity.get("ordered_response_ids") != evidence["response_ids"]
        or not isinstance(raw, Mapping)
        or raw.get("schema") != "snpxsnp_raw_stream_v2"
        or raw.get("member_family_sha256") != record.member_family_sha256
        or raw.get("input_family_sha256") != record.input_family_sha256
        or raw.get("offered_pair_count") != record.offered_pair_count
        or raw.get("design_nonestimable_pair_count")
        != record.design_nonestimable_pair_count
        or raw.get("tested_pair_count") != record.tested_pair_count
        or raw.get("nonfinite_pair_score_count")
        != record.nonfinite_pair_score_count
        or raw.get("failed_response_indices")
        != list(record.failed_response_indices)
        or group_p.shape != (record.family_size, record.response_width)
        or not np.all(np.isfinite(group_p))
        or hashlib.sha256(encoded).hexdigest() != record.score_evidence_sha256
        or len(encoded) != record.output_bytes
    ):
        raise ValueError("resource probe score evidence is detached or invalid")
    if _gated_marker_counts_from_bindings(raw.get("input_block_bindings")) != dict(
        record.gated_marker_count_by_gene
    ):
        raise ValueError("resource probe gated marker counts differ from input-block bindings")
    return {
        "schema": payload["schema"],
        "record": record.to_payload(),
        "score_evidence": evidence,
    }


def validate_snpxsnp_resource_probe_artifact_series(
    payloads: Sequence[Mapping[str, Any]],
    *,
    authorization_payload: Mapping[str, Any],
    authorization_sha256: str,
    response_bank: np.ndarray,
    response_ids: Sequence[str],
) -> dict[str, Any]:
    """Audit every width against one external witness, then project resources."""
    from .audit import _audit_comparator_probe_artifact

    records = [
        _audit_comparator_probe_artifact(
            payload,
            authorization_payload=authorization_payload,
            authorization_sha256=authorization_sha256,
            response_bank=response_bank,
            response_ids=response_ids,
        )["record"]
        for payload in payloads
    ]
    return _validate_comparator_probe_series_structure(records)


__all__ = [
    "produce_snpxsnp_resource_probe",
    "validate_snpxsnp_resource_probe_artifact",
    "validate_snpxsnp_resource_probe_artifact_series",
]
