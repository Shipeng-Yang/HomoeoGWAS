from __future__ import annotations

import csv
import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from homoeogwas.group_family import ExpandedEdgeFamily
from homoeogwas.interact import rank_int
from homoeogwas.omnib_family import OmniBFamilyScores, prepare_omnib_design
from scripts.benchmarks.v201.simulation import (
    compose_exact_pve,
    draw_null,
    interaction_signal,
)
from scripts.benchmarks.v201.track_omnib import (
    OmniBBenchmarkContext,
    _gene_blocks,
    _pc1,
    _root_from_covariance,
)

from .policy import canonical_interact, preparation_options


class MaterializationBlocked(RuntimeError):
    """Response materialization lacks a complete, reviewed authorization."""


@dataclass(frozen=True)
class PreparedAnchor:
    """Production-prepared design bound to an independent normal anchor."""

    context: OmniBBenchmarkContext
    scores: OmniBFamilyScores
    expanded: ExpandedEdgeFamily
    anchor: np.ndarray
    v_hat: np.ndarray
    root_v: np.ndarray
    pc1: np.ndarray
    gene_blocks: Mapping[tuple[str, str], np.ndarray]
    design_hash: str
    scenario_id: str


@dataclass(frozen=True)
class MaterializedResponse:
    """One deterministic QA response plus its transform-scale diagnostics."""

    response_id: str
    truth_id: str
    response_seed: int
    values: np.ndarray
    post_int_values: np.ndarray
    signal: np.ndarray | None
    metadata: Mapping[str, Any]


def materialize_one(
    authority: Mapping[str, Any],
    adapter: Callable[[], Any],
) -> Any:
    require_materialization_authority(authority)
    return adapter()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def require_materialization_authority(authority: Mapping[str, Any]) -> None:
    """Reject response work until a separate, fully bound approval exists."""

    if authority.get("response_materialization_authorized") is not True:
        raise MaterializationBlocked("response_materialization_authorized=false")
    required_hashes = {
        "amendment_sha256",
        "runner_review_sha256",
        "prospective_design_hash",
        "prospective_inventory_sha256",
        "source_input_reverification_sha256",
    }
    missing = sorted(required_hashes - authority.keys())
    invalid_hashes = sorted(
        field
        for field in required_hashes
        if field in authority and not _is_sha256(authority[field])
    )
    if authority.get("amendment_status") != "accepted_for_response_materialization":
        missing.append("amendment_status=accepted_for_response_materialization")
    if authority.get("runner_review_verdict") != "ACCEPT":
        missing.append("runner_review_verdict=ACCEPT")
    if missing or invalid_hashes:
        details = missing + [f"invalid:{field}" for field in invalid_hashes]
        raise MaterializationBlocked(
            "materialization evidence is incomplete: " + ", ".join(details)
        )


def validated_covariance_root(
    covariance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    v_hat = np.asarray(covariance, dtype=float)
    if v_hat.ndim != 2 or v_hat.shape[0] != v_hat.shape[1]:
        raise MaterializationBlocked("fitted V_hat must be square")
    if not np.all(np.isfinite(v_hat)):
        raise MaterializationBlocked("fitted V_hat contains non-finite values")
    if not np.array_equal(v_hat, v_hat.T):
        raise MaterializationBlocked("fitted V_hat is not exactly symmetric")
    eigenvalues = np.linalg.eigvalsh(v_hat)
    if float(eigenvalues.min()) <= 0.0:
        raise MaterializationBlocked("fitted V_hat is not strictly positive definite")
    return v_hat, _root_from_covariance(v_hat)


def prepare_anchor(
    context: OmniBBenchmarkContext,
    *,
    design_hash: str,
    scenario_id: str,
    anchor_seed: int,
) -> PreparedAnchor:
    """Fit the production design on an independent standard-normal anchor."""

    anchor = np.random.default_rng(anchor_seed).standard_normal(
        context.sample_idx.size
    )
    anchored_context = replace(context, phenotype=anchor)
    scores, expanded = prepare_omnib_design(
        anchored_context.subdata,
        anchored_context.family,
        anchored_context.phenotype,
        anchored_context.sample_idx,
        **preparation_options(
            feature_seed=anchored_context.feature_seed,
            retained_variant_masks=anchored_context.retained_variant_masks,
        ),
    )
    if scores.null_covariance is None:
        raise MaterializationBlocked(
            "production omniB preparation did not retain fitted V_hat"
        )
    v_hat, root_v = validated_covariance_root(scores.null_covariance)
    gene_blocks = _gene_blocks(anchored_context, scores)
    required = {
        (edge.sub_x, edge.gene_x) for edge in expanded.edges
    } | {
        (edge.sub_y, edge.gene_y) for edge in expanded.edges
    }
    missing = sorted(required - set(gene_blocks))
    if missing:
        raise MaterializationBlocked(
            f"production gating made QA gene non-estimable: {missing[0]!r}"
        )
    return PreparedAnchor(
        context=anchored_context,
        scores=scores,
        expanded=expanded,
        anchor=anchor,
        v_hat=v_hat,
        root_v=root_v,
        pc1=_pc1(anchored_context),
        gene_blocks=gene_blocks,
        design_hash=design_hash,
        scenario_id=scenario_id,
    )


def generate_response(
    prepared: PreparedAnchor,
    *,
    response_id: str,
    truth_id: str,
    response_seed: int,
) -> MaterializedResponse:
    """Generate one frozen truth without seed search or numerical retry."""

    residual, null_metadata = draw_null(
        "gaussian",
        prepared.root_v,
        np.random.default_rng(response_seed),
        prepared.pc1,
    )
    if truth_id == "gaussian_null":
        values = residual
        signal = None
        metadata: dict[str, Any] = {
            "truth_id": truth_id,
            "generator_scale_interaction_pve": 0.0,
            "generator_scale_status": "zero_by_construction",
            "null_generation": null_metadata,
        }
    elif truth_id == "mixed_sign_diagnostic_pve0p03":
        family = prepared.context.family
        if not family.group_ids:
            raise MaterializationBlocked("mixed-sign truth requires a nonempty family")
        group_index = 0
        blocks = {
            subgenome: prepared.gene_blocks[
                (subgenome, family.genes[group_index][copy_index])
            ]
            for copy_index, subgenome in enumerate(family.subgenomes)
        }
        declared_first_edge = tuple(family.subgenomes[:2])
        signal, signal_metadata = interaction_signal(
            "mixed_sign",
            blocks,
            causal_edges=(declared_first_edge,),
        )
        causal_edges = signal_metadata["causal_pair_edges"]
        if not isinstance(causal_edges, list) or len(causal_edges) != 1:
            raise MaterializationBlocked(
                "accepted mixed-sign primitive did not return exactly one causal edge"
            )
        values, pve_metadata = compose_exact_pve(signal, residual, 0.03)
        metadata = {
            "truth_id": truth_id,
            "generator_scale_interaction_pve": pve_metadata["realized_pve"],
            "generator_scale_target_pve": pve_metadata["target_pve"],
            "causal_group_id": family.group_ids[group_index],
            "causal_pair_edge": causal_edges[0],
            "null_generation": null_metadata,
            "signal_generation": signal_metadata,
        }
    else:
        raise MaterializationBlocked(f"unknown frozen QA truth: {truth_id!r}")

    post_int_values = rank_int(values)
    if signal is not None:
        squared_correlation = float(np.corrcoef(post_int_values, signal)[0, 1] ** 2)
        metadata["post_int_interaction_diagnostic"] = {
            "status": "descriptive_noninferential_only",
            "squared_correlation": squared_correlation,
        }
    return MaterializedResponse(
        response_id=response_id,
        truth_id=truth_id,
        response_seed=response_seed,
        values=np.asarray(values, dtype=np.float64),
        post_int_values=np.asarray(post_int_values, dtype=np.float64),
        signal=None if signal is None else np.asarray(signal, dtype=np.float64),
        metadata=metadata,
    )


def _float64_sha256(values: np.ndarray) -> str:
    little_endian = np.ascontiguousarray(values, dtype=np.dtype("<f8"))
    return hashlib.sha256(little_endian.tobytes()).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_roundtrip_response(
    response: MaterializedResponse,
    *,
    sample_ids: Sequence[str],
    npy_path: str | Path,
    tsv_path: str | Path,
    sample_col: str = "sample",
    trait: str = "qa_trait",
) -> dict[str, Any]:
    """Exclusively serialize a response and prove exact CLI TSV round-trip."""

    npy_path = Path(npy_path)
    tsv_path = Path(tsv_path)
    targets = (npy_path, tsv_path)
    existing = [path for path in targets if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite response target: {existing[0]}")
    ordered_samples = tuple(sample_ids)
    values = np.ascontiguousarray(response.values, dtype=np.dtype("<f8"))
    if values.ndim != 1 or values.size != len(ordered_samples):
        raise ValueError("sample IDs and response values must be one-dimensional and aligned")
    if not np.all(np.isfinite(values)):
        raise ValueError("response values must be finite")
    if any(
        not isinstance(sample_id, str)
        or not sample_id
        or "\t" in sample_id
        or "\n" in sample_id
        or "\r" in sample_id
        for sample_id in ordered_samples
    ):
        raise ValueError("sample IDs must be nonempty TSV-safe strings")
    if len(set(ordered_samples)) != len(ordered_samples):
        raise ValueError("sample IDs must be unique")
    science = canonical_interact()
    if sample_col != science["sample_col"] or trait != science["trait"]:
        raise ValueError("phenotype columns must match the frozen native config")

    with npy_path.open("xb") as npy_handle:
        np.save(npy_handle, values, allow_pickle=False)
    with tsv_path.open("x", encoding="utf-8", newline="") as tsv_handle:
        writer = csv.writer(tsv_handle, delimiter="\t", lineterminator="\n")
        writer.writerow((sample_col, trait))
        writer.writerows(
            (sample_id, format(float(value), ".17g"))
            for sample_id, value in zip(ordered_samples, values, strict=True)
        )

    loaded_npy = np.load(npy_path, allow_pickle=False)
    with tsv_path.open(encoding="utf-8", newline="") as tsv_handle:
        rows = list(csv.DictReader(tsv_handle, delimiter="\t"))
    loaded_samples = tuple(row[sample_col] for row in rows)
    loaded_tsv = np.asarray(
        [float(row[trait]) for row in rows], dtype=np.dtype("<f8")
    )
    expected_bits = values.view(np.uint64)
    if (
        loaded_samples != ordered_samples
        or loaded_npy.dtype.str != "<f8"
        or not np.array_equal(loaded_npy.view(np.uint64), expected_bits)
        or not np.array_equal(loaded_tsv.view(np.uint64), expected_bits)
    ):
        raise MaterializationBlocked("serialized response failed bitwise round-trip")

    return {
        "response_id": response.response_id,
        "sample_count": len(ordered_samples),
        "sample_col": sample_col,
        "trait": trait,
        "values_float64_sha256": _float64_sha256(values),
        "post_int_float64_sha256": _float64_sha256(response.post_int_values),
        "npy_path": str(npy_path),
        "npy_sha256": _file_sha256(npy_path),
        "tsv_path": str(tsv_path),
        "tsv_sha256": _file_sha256(tsv_path),
    }
