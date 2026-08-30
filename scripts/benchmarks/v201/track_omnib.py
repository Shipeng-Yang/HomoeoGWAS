"""Track B end-to-end, conditional-bank, power and encoding benchmarks.

The runner keeps the group family frozen, derives every response from an
auditable role-specific seed namespace, and never learns a decision threshold
from the response bank on which it is evaluated.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from homoeogwas.group_family import ExpandedEdgeFamily, MasterGroupFamily
from homoeogwas.interact import SubgenomeData
from homoeogwas.omnib_family import (
    OmniBFamilyScores,
    bootstrap_minp_calibration,
    score_omnib_family,
    score_omnib_responses,
)

from .comparators import (
    METHOD_NAMES,
    MethodScoreBank,
    _nested_snp_product_design,
    group_component_p,
    score_snpxsnp_family,
)
from .contracts import Scenario, derive_seed, sha256_payload
from .shards import ShardConflict, ShardKey, load_shard, write_shard_exclusive
from .simulation import compose_exact_pve, draw_null, interaction_signal, standardize

_ROLE_ALIASES = {
    "calibration": "calibration",
    "evaluation": "heldout",
    "heldout": "heldout",
    "power": "power",
}
_FORMAL_BOOTSTRAP_MINIMUM = 2_000
_DEFAULT_DESIGN_HASH = "0" * 64


@dataclass(frozen=True)
class OmniBBenchmarkContext:
    """One phenotype-independent genotype/family backbone plus an anchor trait."""

    subdata: dict[str, SubgenomeData]
    family: MasterGroupFamily
    phenotype: np.ndarray
    sample_idx: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.subdata, dict):
            raise ValueError("subdata must be a dictionary")
        if set(self.subdata) != set(self.family.subgenomes):
            raise ValueError("subdata labels must exactly match family subgenomes")
        object.__setattr__(
            self,
            "subdata",
            {label: self.subdata[label] for label in self.family.subgenomes},
        )
        if len(self.family.subgenomes) not in {2, 3, 4}:
            raise ValueError("Track B supports exactly two, three or four copies")
        phenotype = np.array(self.phenotype, dtype=float, copy=True)
        sample_idx = np.array(self.sample_idx, dtype=int, copy=True)
        if phenotype.ndim != 1 or sample_idx.ndim != 1 or phenotype.shape != sample_idx.shape:
            raise ValueError("phenotype and sample_idx must be aligned vectors")
        if phenotype.size < 8 or not np.all(np.isfinite(phenotype)):
            raise ValueError("phenotype must contain at least eight finite samples")
        if np.any(sample_idx < 0) or len(set(sample_idx.tolist())) != sample_idx.size:
            raise ValueError("sample_idx must contain unique non-negative indices")
        for label, data in self.subdata.items():
            values = np.asarray(data.X)
            if values.ndim != 2 or values.shape[0] <= int(sample_idx.max()):
                raise ValueError(f"subgenome {label} is not aligned to sample_idx")
        phenotype.setflags(write=False)
        sample_idx.setflags(write=False)
        object.__setattr__(self, "phenotype", phenotype)
        object.__setattr__(self, "sample_idx", sample_idx)


@dataclass(frozen=True)
class ConditionalBank:
    """One immutable response bank and all matched localizing-method scores."""

    requested_role: str
    canonical_role: str
    stage: str
    seed_ids: tuple[str, ...]
    seeds: tuple[int, ...]
    responses: np.ndarray
    response_hash: str
    response_metadata: tuple[dict[str, Any], ...]
    family_ids: tuple[str, ...]
    family_hash: str
    design_hash: str
    score_bank: MethodScoreBank
    tested_family_members: Mapping[str, tuple[str, ...]]
    tested_family_hashes: Mapping[str, str]
    score_matrix_hashes: Mapping[str, str]
    calibration_reference: Mapping[str, Any]
    execution: Mapping[str, Any]
    runtime_seconds: float
    null_covariance: Mapping[str, Any]
    failure: Mapping[str, Any]

    def __post_init__(self) -> None:
        values = np.array(self.responses, dtype=float, copy=True)
        if values.ndim != 2 or values.shape[1] != len(self.seed_ids):
            raise ValueError("responses must be sample-by-seed")
        if len(self.seed_ids) != len(self.seeds):
            raise ValueError("seed_ids and seeds must have the same length")
        if len(set(self.seed_ids)) != len(self.seed_ids):
            raise ValueError("seed_ids must be unique within a bank")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be unique within a bank")
        if self.canonical_role not in {"calibration", "heldout", "power"}:
            raise ValueError("invalid canonical response-bank role")
        methods = set(self.score_bank.p_by_method)
        if not (
            methods
            == set(self.tested_family_members)
            == set(self.tested_family_hashes)
            == set(self.score_matrix_hashes)
        ):
            raise ValueError("score and tested-family method sets must match")
        values.setflags(write=False)
        object.__setattr__(self, "responses", values)

    @property
    def p_by_method(self) -> Mapping[str, np.ndarray]:
        return self.score_bank.p_by_method

    def to_payload(self, *, include_scores: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "requested_role": self.requested_role,
            "canonical_role": self.canonical_role,
            "stage": self.stage,
            "formal": self.stage == "formal",
            "qa_only": self.stage == "pilot",
            "seed_ids": list(self.seed_ids),
            "seeds": list(self.seeds),
            "response_hash": self.response_hash,
            "response_shape": list(self.responses.shape),
            "response_metadata": list(self.response_metadata),
            "family_ids": list(self.family_ids),
            "family_hash": self.family_hash,
            "design_hash": self.design_hash,
            "tested_family_sizes": dict(self.score_bank.tested_family_sizes),
            "tested_family_members": {
                method: list(members)
                for method, members in self.tested_family_members.items()
            },
            "tested_family_hashes": dict(self.tested_family_hashes),
            "score_matrix_hashes": dict(self.score_matrix_hashes),
            "calibration_reference": dict(self.calibration_reference),
            "execution": dict(self.execution),
            "requested_jobs": self.execution.get("requested_jobs", 1),
            "effective_jobs": self.execution.get("effective_jobs", 1),
            "parallel_backend": self.execution.get("backend", "serial"),
            "worker_pids": list(self.execution.get("worker_pids", [])),
            "runtime_seconds": self.runtime_seconds,
            "null_covariance": dict(self.null_covariance),
            "failure": dict(self.failure),
        }
        if include_scores:
            payload["p_by_method"] = {
                method: _json_safe(values)
                for method, values in self.score_bank.p_by_method.items()
            }
        return payload


@dataclass(frozen=True)
class _PreparedScenario:
    context: OmniBBenchmarkContext
    scores: OmniBFamilyScores
    expanded: ExpandedEdgeFamily
    root_v: np.ndarray
    pc1: np.ndarray
    gene_blocks: Mapping[tuple[str, str], np.ndarray]
    family_hash: str
    setup_seed_id: str
    setup_seed: int


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return value


def _array_hash(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(repr(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _family_manifest(family: MasterGroupFamily) -> dict[str, Any]:
    return {
        "subgenomes": list(family.subgenomes),
        "group_ids": list(family.group_ids),
        "genes": [list(row) for row in family.genes],
    }


def _family_hash(family: MasterGroupFamily) -> str:
    return sha256_payload(_family_manifest(family))


def _context_manifest(context: OmniBBenchmarkContext) -> dict[str, Any]:
    subgenomes = []
    for label in context.family.subgenomes:
        data = context.subdata[label]
        values = np.asarray(data.X)
        subgenomes.append(
            {
                "label": label,
                "samples": [str(sample) for sample in data.samples],
                "X": {
                    "shape": list(values.shape),
                    "dtype": str(values.dtype),
                    "sha256": _array_hash(values),
                },
                "gene_snp": [
                    {
                        "gene": str(gene),
                        "indices": np.asarray(indices, dtype=int).tolist(),
                    }
                    for gene, indices in sorted(data.gene_snp.items())
                ],
            }
        )
    return {
        "subgenomes": subgenomes,
        "family": _family_manifest(context.family),
        "sample_idx": {
            "values": context.sample_idx.tolist(),
            "sha256": _array_hash(context.sample_idx),
        },
        "phenotype": {
            "shape": list(context.phenotype.shape),
            "dtype": str(context.phenotype.dtype),
            "sha256": _array_hash(context.phenotype),
        },
    }


def _context_fingerprint(context: OmniBBenchmarkContext) -> str:
    return sha256_payload(_context_manifest(context))


def _request_hash(
    *,
    design_hash: str,
    context: OmniBBenchmarkContext,
    request: Mapping[str, Any],
) -> str:
    return sha256_payload(
        {
            "design_hash": design_hash,
            "context_fingerprint": _context_fingerprint(context),
            "request": _json_safe(request),
        }
    )


def _seed(
    design_hash: str,
    scenario_id: str,
    replicate: int,
    stage: str,
    role: str,
) -> tuple[int, str]:
    namespace = f"{stage}:{role}"
    value = derive_seed(design_hash, "omnib", scenario_id, replicate, namespace)
    return value, f"{namespace}:{scenario_id}:{replicate}:{value:016x}"


def _validate_count(count: int, name: str = "count") -> int:
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or int(count) < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(count)


def _canonical_role(bank: str) -> str:
    if bank not in _ROLE_ALIASES:
        raise ValueError("bank must be calibration, heldout/evaluation, or power")
    return _ROLE_ALIASES[bank]


def _stage(qa_only: bool) -> str:
    return "pilot" if qa_only else "formal"


def _require_formal_budget(count: int, qa_only: bool, quantity: str) -> None:
    if not qa_only and count < _FORMAL_BOOTSTRAP_MINIMUM:
        raise ValueError(
            f"formal {quantity} requires at least {_FORMAL_BOOTSTRAP_MINIMUM} replicates; "
            "B=19/199 is pilot QA only"
        )


def _edge_count_per_group(context: OmniBBenchmarkContext) -> int:
    copies = len(context.family.subgenomes)
    return copies * (copies - 1) // 2


def build_synthetic_omnib_context(
    n: int,
    groups: int,
    copies: int,
    seed: int,
) -> OmniBBenchmarkContext:
    """Build a deterministic production-shaped synthetic Track B backbone."""

    n = _validate_count(n, "n")
    groups = _validate_count(groups, "groups")
    if copies not in {2, 3, 4}:
        raise ValueError("copies must be 2, 3 or 4")
    rng = np.random.default_rng(seed)
    labels = ("A", "B", "C", "D")[:copies]
    subdata: dict[str, SubgenomeData] = {}
    for label in labels:
        x = rng.integers(0, 3, size=(n, groups * 5)).astype(float)
        mapping = {
            f"g{index}": np.arange(index * 5, (index + 1) * 5)
            for index in range(groups)
        }
        subdata[label] = SubgenomeData(
            X=x,
            gene_snp=mapping,
            samples=[f"sample_{index}" for index in range(n)],
            chunk=None,
        )
    family = MasterGroupFamily(
        subgenomes=labels,
        group_ids=tuple(f"group_{index}" for index in range(groups)),
        genes=tuple((f"g{index}",) * copies for index in range(groups)),
    )
    return OmniBBenchmarkContext(
        subdata=subdata,
        family=family,
        phenotype=rng.standard_normal(n),
        sample_idx=np.arange(n),
    )


def _pc1(context: OmniBBenchmarkContext) -> np.ndarray:
    blocks = [
        np.asarray(context.subdata[label].X[context.sample_idx], dtype=float)
        for label in context.family.subgenomes
    ]
    combined = np.column_stack(blocks)
    means = np.nanmean(combined, axis=0)
    combined = np.where(np.isfinite(combined), combined, means)
    centered = combined - combined.mean(axis=0)
    scale = centered.std(axis=0, ddof=1)
    usable = np.isfinite(scale) & (scale > 0)
    if not np.any(usable):
        raise ValueError("synthetic genotype backbone has no variable marker")
    left, singular, _ = np.linalg.svd(centered[:, usable] / scale[usable], full_matrices=False)
    return standardize(left[:, 0] * singular[0])


def _context_root(context: OmniBBenchmarkContext) -> np.ndarray:
    blocks: list[np.ndarray] = []
    for label in context.family.subgenomes:
        values = np.asarray(context.subdata[label].X[context.sample_idx], dtype=float)
        means = np.nanmean(values, axis=0)
        values = np.where(np.isfinite(values), values, means)
        centered = values - values.mean(axis=0)
        scale = centered.std(axis=0, ddof=1)
        usable = np.isfinite(scale) & (scale > 0)
        if np.any(usable):
            blocks.append(centered[:, usable] / scale[usable])
    standardized = np.column_stack(blocks)
    kernel = standardized @ standardized.T / standardized.shape[1]
    covariance = 0.5 * kernel + 0.5 * np.eye(kernel.shape[0])
    values, vectors = np.linalg.eigh(0.5 * (covariance + covariance.T))
    return (vectors * np.sqrt(np.clip(values, 1e-12, None))) @ vectors.T


def _root_from_covariance(covariance: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(0.5 * (covariance + covariance.T))
    return (vectors * np.sqrt(np.clip(values, 1e-12, None))) @ vectors.T


def _gene_blocks(
    context: OmniBBenchmarkContext,
    scores: OmniBFamilyScores,
) -> dict[tuple[str, str], np.ndarray]:
    blocks: dict[tuple[str, str], np.ndarray] = {}
    for subgenome, row in zip(
        context.family.subgenomes,
        zip(*context.family.genes, strict=True),
        strict=True,
    ):
        for gene in row:
            key = (subgenome, gene)
            if key in blocks:
                continue
            indices = scores.gated_snp.get(key)
            if indices is None:
                continue
            blocks[key] = np.asarray(
                context.subdata[subgenome].X[
                    np.ix_(context.sample_idx, np.asarray(indices, dtype=int))
                ],
                dtype=float,
            )
    return blocks


def _prepare_scenario(
    context: OmniBBenchmarkContext,
    *,
    design_hash: str,
    stage: str,
    scenario_id: str,
    n_jobs: int,
) -> _PreparedScenario:
    setup_seed, setup_seed_id = _seed(design_hash, scenario_id, 0, stage, "scenario_setup")
    scores, expanded = score_omnib_family(
        context.subdata,
        context.family,
        context.phenotype,
        context.sample_idx,
        transform="INT",
        bootstrap_B=0,
        bootstrap_seed=setup_seed,
        n_jobs=n_jobs,
        grm_method="grm_from_X",
        maf_min=0.01,
        burden_maf=0.01,
        min_snp=3,
        cap=150,
        n_pc=3,
    )
    if scores.null_covariance is None:
        raise RuntimeError("production omniB preparation did not retain its null covariance")
    blocks = _gene_blocks(context, scores)
    required = {
        (edge.sub_x, edge.gene_x) for edge in expanded.edges
    } | {
        (edge.sub_y, edge.gene_y) for edge in expanded.edges
    }
    missing = sorted(required - set(blocks))
    if missing:
        raise RuntimeError(f"production gating made comparator gene non-estimable: {missing[0]!r}")
    return _PreparedScenario(
        context=context,
        scores=scores,
        expanded=expanded,
        root_v=_root_from_covariance(scores.null_covariance),
        pc1=_pc1(context),
        gene_blocks=blocks,
        family_hash=_family_hash(context.family),
        setup_seed_id=setup_seed_id,
        setup_seed=setup_seed,
    )


def _method_scores(
    prepared: _PreparedScenario,
    responses: np.ndarray,
    *,
    n_jobs: int,
    calibration_responses: np.ndarray | None = None,
) -> tuple[MethodScoreBank, dict[str, Any]]:
    _edge, group, components = score_omnib_responses(
        prepared.scores,
        prepared.context.family,
        prepared.expanded,
        responses,
        n_jobs=n_jobs,
    )
    response_execution = dict(prepared.scores.parallel_execution)
    component_scores = [
        group_component_p(components, prepared.expanded, component_index=index)
        for index in range(3)
    ]
    snpxsnp, snpxsnp_size = score_snpxsnp_family(
        prepared.scores,
        prepared.context.family,
        prepared.expanded,
        prepared.gene_blocks,
        responses,
        calibration_responses=(
            responses if calibration_responses is None else calibration_responses
        ),
    )
    group_count = len(prepared.context.family.group_ids)
    return (
        MethodScoreBank(
            family_ids=prepared.context.family.group_ids,
            p_by_method={
                "omnib": group,
                "minor_burden": component_scores[0],
                "pc1": component_scores[1],
                "kernel_hadamard": component_scores[2],
                "legacy_burden_product": component_scores[0].copy(),
                "snpxsnp": snpxsnp,
            },
            tested_family_sizes={
                "omnib": group_count,
                "minor_burden": group_count,
                "pc1": group_count,
                "kernel_hadamard": group_count,
                "legacy_burden_product": group_count,
                "snpxsnp": snpxsnp_size,
            },
        ),
        response_execution,
    )


def _tested_family_members(prepared: _PreparedScenario) -> dict[str, tuple[str, ...]]:
    local_members = tuple(
        f"{group_id}|edges="
        + ",".join(
            prepared.expanded.edges[edge_index].edge_id
            + f":estimable={int(prepared.scores.edge_estimable[edge_index])}"
            for edge_index in prepared.expanded.group_edge_indices[group_index]
        )
        for group_index, group_id in enumerate(prepared.context.family.group_ids)
    )
    snpxsnp_members: list[str] = []
    W = np.asarray(prepared.scores.W, dtype=float)
    Cw = W @ np.asarray(prepared.scores.null_design, dtype=float)
    for edge in prepared.expanded.edges:
        left_key = (edge.sub_x, edge.gene_x)
        right_key = (edge.sub_y, edge.gene_y)
        left = prepared.gene_blocks[left_key]
        right = prepared.gene_blocks[right_key]
        left_columns = np.asarray(prepared.scores.gated_snp[left_key], dtype=int)
        right_columns = np.asarray(prepared.scores.gated_snp[right_key], dtype=int)
        for left_local, left_column in enumerate(left_columns):
            for right_local, right_column in enumerate(right_columns):
                if _nested_snp_product_design(
                    W,
                    Cw,
                    left[:, left_local],
                    right[:, right_local],
                ) is None:
                    continue
                snpxsnp_members.append(
                    f"{edge.edge_id}|{int(left_column)}|{int(right_column)}"
                )
    members = {
        "omnib": local_members,
        "minor_burden": local_members,
        "pc1": local_members,
        "kernel_hadamard": local_members,
        "legacy_burden_product": local_members,
        "snpxsnp": tuple(snpxsnp_members),
    }
    return members


def _draw_bank_responses(
    prepared: _PreparedScenario,
    *,
    requested_role: str,
    canonical_role: str,
    count: int,
    design_hash: str,
    stage: str,
    scenario_id: str,
    replicate_offset: int,
    null_model: str,
    response_factory: Any = None,
) -> tuple[np.ndarray, tuple[int, ...], tuple[str, ...], tuple[dict[str, Any], ...]]:
    responses: list[np.ndarray] = []
    seeds: list[int] = []
    seed_ids: list[str] = []
    metadata: list[dict[str, Any]] = []
    for offset in range(count):
        replicate = replicate_offset + offset
        seed, seed_id = _seed(
            design_hash,
            scenario_id,
            replicate,
            stage,
            canonical_role,
        )
        rng = np.random.default_rng(seed)
        if response_factory is None:
            response, row_metadata = draw_null(
                null_model,
                prepared.root_v,
                rng,
                prepared.pc1,
            )
        else:
            response, row_metadata = response_factory(rng, replicate)
        response = np.array(response, dtype=float, copy=True)
        if response.ndim != 1 or response.size != prepared.context.sample_idx.size:
            raise ValueError("generated response is not aligned to the prepared samples")
        if not np.all(np.isfinite(response)):
            raise ValueError("generated response contains non-finite values")
        responses.append(response)
        seeds.append(seed)
        seed_ids.append(seed_id)
        metadata.append(
            {
                **row_metadata,
                "requested_role": requested_role,
                "canonical_role": canonical_role,
                "seed": seed,
                "seed_id": seed_id,
                "replicate": replicate,
            }
        )
    return (
        np.column_stack(responses).copy(),
        tuple(seeds),
        tuple(seed_ids),
        tuple(metadata),
    )


def _bank_from_prepared(
    prepared: _PreparedScenario,
    *,
    bank: str,
    count: int,
    design_hash: str,
    stage: str,
    scenario_id: str,
    replicate_offset: int,
    null_model: str,
    n_jobs: int,
    response_factory: Any = None,
    calibration_responses: np.ndarray | None = None,
    calibration_reference: Mapping[str, Any] | None = None,
) -> ConditionalBank:
    started = time.perf_counter()
    canonical_role = _canonical_role(bank)
    responses, seeds, seed_ids, response_metadata = _draw_bank_responses(
        prepared,
        requested_role=bank,
        canonical_role=canonical_role,
        count=count,
        design_hash=design_hash,
        stage=stage,
        scenario_id=scenario_id,
        replicate_offset=replicate_offset,
        null_model=null_model,
        response_factory=response_factory,
    )
    response_hash = _array_hash(responses)
    if canonical_role == "calibration" and calibration_responses is None:
        calibration_responses = responses
        calibration_reference = {
            "requested_role": bank,
            "canonical_role": "calibration",
            "seed_ids": list(seed_ids),
            "response_hash": response_hash,
            "shares_memory": True,
            "self_calibration_allowed": True,
        }
    elif calibration_responses is None or calibration_reference is None:
        raise ValueError(
            "heldout/power response scoring requires an independent calibration bank"
        )
    else:
        reference_seed_ids = set(calibration_reference.get("seed_ids", []))
        if set(seed_ids) & reference_seed_ids:
            raise RuntimeError("target and calibration response banks share seed IDs")
        if np.shares_memory(responses, calibration_responses):
            raise RuntimeError("target and calibration response banks share ndarray memory")
        calibration_hash = _array_hash(calibration_responses)
        if response_hash == calibration_hash:
            raise RuntimeError("target and calibration response banks share a response hash")
        if calibration_reference.get("response_hash") != calibration_hash:
            raise RuntimeError("calibration response hash does not match its provenance")
        calibration_reference = {
            **dict(calibration_reference),
            "shares_memory": False,
            "self_calibration_allowed": False,
        }
    failure: dict[str, Any] = {"failed": False, "failed_response_indices": []}
    try:
        score_bank, response_execution = _method_scores(
            prepared,
            responses,
            n_jobs=n_jobs,
            calibration_responses=calibration_responses,
        )
    except Exception as error:
        failure = {
            "failed": True,
            "failed_response_indices": list(range(count)),
            "error_type": type(error).__name__,
            "message": str(error),
        }
        raise RuntimeError(f"conditional bank scoring failed: {error}") from error
    nonfinite_by_method = {
        method: np.flatnonzero(~np.isfinite(values).all(axis=0)).astype(int).tolist()
        for method, values in score_bank.p_by_method.items()
    }
    all_nan_by_method = {
        method: np.flatnonzero(~np.isfinite(values).any(axis=0)).astype(int).tolist()
        for method, values in score_bank.p_by_method.items()
    }
    failed_indices = sorted(
        {
            index
            for indices in nonfinite_by_method.values()
            for index in indices
        }
    )
    failure.update(
        {
            "failed": bool(failed_indices),
            "status": (
                "failed"
                if len(failed_indices) == count
                else "partial_failure" if failed_indices else "completed"
            ),
            "failed_response_indices": failed_indices,
            "nonfinite_response_indices_by_method": nonfinite_by_method,
            "all_nan_response_indices_by_method": all_nan_by_method,
            "any_nonfinite": bool(failed_indices),
        }
    )
    members = _tested_family_members(prepared)
    for method in METHOD_NAMES:
        if len(members[method]) != score_bank.tested_family_sizes[method]:
            raise RuntimeError(
                f"tested-family member count disagrees for {method}: "
                f"{len(members[method])} != {score_bank.tested_family_sizes[method]}"
            )
    tested_hashes = {
        method: sha256_payload(
            {"method": method, "ordered_member_ids": list(members[method])}
        )
        for method in METHOD_NAMES
    }
    return ConditionalBank(
        requested_role=bank,
        canonical_role=canonical_role,
        stage=stage,
        seed_ids=seed_ids,
        seeds=seeds,
        responses=responses,
        response_hash=response_hash,
        response_metadata=response_metadata,
        family_ids=score_bank.family_ids,
        family_hash=prepared.family_hash,
        design_hash=design_hash,
        score_bank=score_bank,
        tested_family_members=members,
        tested_family_hashes=tested_hashes,
        score_matrix_hashes={
            method: _array_hash(values)
            for method, values in score_bank.p_by_method.items()
        },
        calibration_reference=dict(calibration_reference),
        execution=response_execution,
        runtime_seconds=float(time.perf_counter() - started),
        null_covariance={
            "components": dict(prepared.scores.covariance_components),
            "shape": list(prepared.scores.null_covariance.shape),
            "sha256": _array_hash(prepared.scores.null_covariance),
        },
        failure=failure,
    )


def _calibration_scenario_id(scenario_id: str) -> str:
    for suffix in (".heldout", ".evaluation", ".power"):
        if scenario_id.endswith(suffix):
            return scenario_id.removesuffix(suffix) + ".calibration"
    if scenario_id.endswith(".calibration"):
        return scenario_id
    return scenario_id + ".calibration"


def _calibration_reference(
    responses: np.ndarray,
    seed_ids: Sequence[str],
    scenario_id: str,
) -> dict[str, Any]:
    return {
        "requested_role": "calibration",
        "canonical_role": "calibration",
        "scenario_id": scenario_id,
        "seed_ids": list(seed_ids),
        "response_hash": _array_hash(responses),
    }


def run_conditional_bank(
    context: OmniBBenchmarkContext,
    *,
    bank: str,
    count: int,
    design_hash: str,
    n_jobs: int = 1,
    qa_only: bool = True,
    null_model: str = "gaussian",
    scenario_id: str = "B.conditional.synthetic",
    replicate_offset: int = 0,
) -> ConditionalBank:
    """Prepare once, then score one role-separated conditional response bank."""

    count = _validate_count(count)
    canonical_role = _canonical_role(bank)
    _require_formal_budget(count, qa_only, "conditional bank")
    stage = _stage(qa_only)
    prepared = _prepare_scenario(
        context,
        design_hash=design_hash,
        stage=stage,
        scenario_id=scenario_id,
        n_jobs=n_jobs,
    )
    calibration_responses = None
    calibration_reference = None
    if canonical_role != "calibration":
        reference_scenario_id = _calibration_scenario_id(scenario_id)
        (
            calibration_responses,
            _calibration_seeds,
            calibration_seed_ids,
            _calibration_metadata,
        ) = _draw_bank_responses(
            prepared,
            requested_role="calibration",
            canonical_role="calibration",
            count=count,
            design_hash=design_hash,
            stage=stage,
            scenario_id=reference_scenario_id,
            replicate_offset=0,
            null_model=null_model,
        )
        calibration_reference = _calibration_reference(
            calibration_responses,
            calibration_seed_ids,
            reference_scenario_id,
        )
    return _bank_from_prepared(
        prepared,
        bank=bank,
        count=count,
        design_hash=design_hash,
        stage=stage,
        scenario_id=scenario_id,
        replicate_offset=replicate_offset,
        null_model=null_model,
        n_jobs=n_jobs,
        calibration_responses=calibration_responses,
        calibration_reference=calibration_reference,
    )


def empirical_threshold(p_null: np.ndarray, alpha: float = 0.05) -> float | None:
    """Learn one strict experiment-wide min-P threshold from calibration only."""

    values = np.asarray(p_null, dtype=float)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 1:
        raise ValueError("p_null must be a non-empty hypothesis-by-response matrix")
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must be strictly between zero and one")
    finite = np.isfinite(values)
    all_nan_columns = np.flatnonzero(~finite.any(axis=0)).astype(int).tolist()
    if all_nan_columns:
        raise ValueError(
            "calibration contains all-NaN response columns: "
            + ",".join(str(index) for index in all_nan_columns)
        )
    family_min = np.where(finite, values, np.inf).min(axis=0)
    k = int(np.floor(float(alpha) * (family_min.size + 1)))
    return None if k < 1 else float(np.sort(family_min)[k - 1])


def apply_threshold(p_bank: np.ndarray, threshold: float | None) -> np.ndarray:
    """Apply a previously frozen strict min-P threshold column by column."""

    values = np.asarray(p_bank, dtype=float)
    if values.ndim != 2:
        raise ValueError("p_bank must be hypothesis-by-response")
    if threshold is None:
        return np.zeros(values.shape[1], dtype=bool)
    if not np.isfinite(threshold) or not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("threshold must be None or a finite probability")
    finite = np.isfinite(values)
    minimum = np.where(finite, values, np.inf).min(axis=0)
    minimum[~finite.any(axis=0)] = np.nan
    decisions = minimum < float(threshold)
    return decisions


def _base_replicate_payload(
    context: OmniBBenchmarkContext,
    *,
    replicate: int,
    design_hash: str,
    stage: str,
    scenario_id: str,
    n_jobs: int,
) -> dict[str, Any]:
    family_manifest = _family_manifest(context.family)
    context_manifest = _context_manifest(context)
    return {
        "track": "omnib",
        "scenario_id": scenario_id,
        "replicate": int(replicate),
        "design_hash": design_hash,
        "stage": stage,
        "formal": stage == "formal",
        "qa_only": stage == "pilot",
        "mode": "group",
        "statistic": "omniB",
        "hypothesis_unit": "group",
        "family_scope": "primary_only",
        "subset_order": 2,
        "pair_edges_per_group": _edge_count_per_group(context),
        "direct_higher_order_term": False,
        "requested_jobs": int(n_jobs),
        "family_ids": list(context.family.group_ids),
        "family_manifest": family_manifest,
        "family_hash": sha256_payload(family_manifest),
        "context_manifest": context_manifest,
        "context_fingerprint": sha256_payload(context_manifest),
    }


def _resume_existing_shard(
    shard_path: str | Path | None,
    *,
    scenario_id: str,
    replicate: int,
    design_hash: str,
    request_hash: str,
) -> dict[str, Any] | None:
    if shard_path is None or not Path(shard_path).exists():
        return None
    existing = load_shard(
        Path(shard_path),
        ShardKey("omnib", scenario_id, replicate),
    )
    if (
        existing["design_hash"] != design_hash
        or existing.get("request_hash") != request_hash
    ):
        raise ShardConflict(f"existing shard differs: {Path(shard_path)}")
    return existing


def run_end_to_end_null(
    context: OmniBBenchmarkContext,
    *,
    replicate: int,
    bootstrap_B: int,
    qa_only: bool,
    n_jobs: int = 1,
    design_hash: str = _DEFAULT_DESIGN_HASH,
    null_model: str = "gaussian",
    scenario_id: str = "B.end2end.synthetic.gaussian",
    shard_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run one complete released group-omniB null procedure.

    The production scorer is called exactly once for the phenotype. Its null
    covariance is re-estimated for that phenotype and reused for all bootstrap
    columns by the production implementation.
    """

    bootstrap_B = _validate_count(bootstrap_B, "bootstrap_B")
    stage = _stage(qa_only)
    request_hash = _request_hash(
        design_hash=design_hash,
        context=context,
        request={
            "entrypoint": "run_end_to_end_null",
            "scenario_id": scenario_id,
            "replicate": replicate,
            "stage": stage,
            "qa_only": qa_only,
            "bootstrap_B": bootstrap_B,
            "null_model": null_model,
            "n_jobs": n_jobs,
        },
    )
    existing = _resume_existing_shard(
        shard_path,
        scenario_id=scenario_id,
        replicate=replicate,
        design_hash=design_hash,
        request_hash=request_hash,
    )
    if existing is not None:
        return existing
    _require_formal_budget(bootstrap_B, qa_only, "end-to-end calibration")
    observed_seed, observed_seed_id = _seed(
        design_hash, scenario_id, replicate, stage, "heldout"
    )
    bootstrap_seed, bootstrap_seed_id = _seed(
        design_hash, scenario_id, replicate, stage, "calibration"
    )
    phenotype, null_metadata = draw_null(
        null_model,
        _context_root(context),
        np.random.default_rng(observed_seed),
        _pc1(context),
    )
    payload = _base_replicate_payload(
        context,
        replicate=replicate,
        design_hash=design_hash,
        stage=stage,
        scenario_id=scenario_id,
        n_jobs=n_jobs,
    )
    payload.update(
        {
            "experiment": "end2end",
            "request_hash": request_hash,
            "bootstrap_B": bootstrap_B,
            "response_role": "heldout",
            "response_seed": observed_seed,
            "response_seed_id": observed_seed_id,
            "response_hash": _array_hash(phenotype),
            "calibration_seed": bootstrap_seed,
            "calibration_seed_id": bootstrap_seed_id,
            "null_model": null_model,
            "null_generation": null_metadata,
            "formal_rejections": [],
            "inference_status": (
                "noninferential_do_not_threshold" if qa_only else "formal"
            ),
        }
    )
    started = time.perf_counter()
    try:
        scores, expanded = score_omnib_family(
            context.subdata,
            context.family,
            phenotype,
            context.sample_idx,
            transform="INT",
            bootstrap_B=bootstrap_B,
            bootstrap_seed=bootstrap_seed,
            n_jobs=n_jobs,
            grm_method="grm_from_X",
            maf_min=0.01,
            burden_maf=0.01,
            min_snp=3,
            cap=150,
            n_pc=3,
        )
        if len(expanded.group_edge_indices) != len(context.family.group_ids):
            raise RuntimeError("production scorer changed the frozen group family")
        calibration = bootstrap_minp_calibration(
            scores.group_p[:, 0],
            scores.group_p[:, 1:],
            alpha=0.05,
        )
        adjusted = np.asarray(calibration["adjusted_p_local"], dtype=float)
        qa_rejections = [
            context.family.group_ids[index]
            for index in calibration["rejected_local"]
        ]
        if not qa_only:
            payload["formal_rejections"] = qa_rejections
        execution = dict(scores.parallel_execution)
        payload.update(
            {
                "observed_group_p": _json_safe(scores.group_p[:, 0]),
                "adjusted_p": _json_safe(adjusted),
                "adjusted_decisions": (adjusted <= 0.05).tolist(),
                "qa_diagnostic_rejections": qa_rejections,
                "bootstrap_minp": _json_safe(calibration),
                "null_minima": _json_safe(
                    np.where(
                        np.isfinite(scores.group_p[:, 1:]).all(axis=0),
                        np.where(
                            np.isfinite(scores.group_p[:, 1:]),
                            scores.group_p[:, 1:],
                            np.inf,
                        ).min(axis=0),
                        0.0,
                    )
                ),
                "family_order_hash": sha256_payload(list(context.family.group_ids)),
                "observed_group_p_hash": _array_hash(scores.group_p[:, 0]),
                "bootstrap_group_p_hash": _array_hash(scores.group_p[:, 1:]),
                "null_covariance": {
                    "components": dict(scores.covariance_components),
                    "shape": list(scores.null_covariance.shape),
                    "sha256": _array_hash(scores.null_covariance),
                },
                "effective_jobs": execution.get("effective_jobs", 1),
                "parallel_backend": execution.get("backend", "serial"),
                "worker_pids": list(execution.get("worker_pids", [])),
                "parallel_execution": execution,
                "failure": {"failed": False, "error_type": None, "message": None},
            }
        )
    except Exception as error:
        payload.update(
            {
                "effective_jobs": 0,
                "parallel_backend": "failed",
                "worker_pids": [],
                "failure": {
                    "failed": True,
                    "error_type": type(error).__name__,
                    "message": str(error),
                },
            }
        )
    payload["runtime_seconds"] = float(time.perf_counter() - started)
    result = _json_safe(payload)
    if shard_path is not None:
        write_shard_exclusive(Path(shard_path), result)
    return result


def _group_signal(
    prepared: _PreparedScenario,
    architecture: str,
    causal_groups: int,
) -> tuple[np.ndarray, list[str], list[dict[str, Any]]]:
    family = prepared.context.family
    if causal_groups < 1 or causal_groups > len(family.group_ids):
        raise ValueError("causal_groups must select one or more declared groups")
    signals: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    causal_ids = list(family.group_ids[:causal_groups])
    for group_index in range(causal_groups):
        blocks = {
            subgenome: prepared.gene_blocks[(subgenome, family.genes[group_index][copy_index])]
            for copy_index, subgenome in enumerate(family.subgenomes)
        }
        kwargs: dict[str, Any] = {}
        if architecture == "mispaired":
            partner_group = (group_index + 1) % len(family.group_ids)
            if partner_group == group_index:
                wrong = np.roll(blocks[family.subgenomes[-1]], 1, axis=0)
            else:
                wrong_gene = family.genes[partner_group][-1]
                wrong = prepared.gene_blocks[(family.subgenomes[-1], wrong_gene)]
            kwargs["wrong_partner"] = wrong
        signal, row_metadata = interaction_signal(architecture, blocks, **kwargs)
        signals.append(signal)
        metadata.append(row_metadata)
    return standardize(np.sum(signals, axis=0)), causal_ids, metadata


def _assert_independent_banks(left: ConditionalBank, right: ConditionalBank) -> None:
    if set(left.seed_ids) & set(right.seed_ids):
        raise RuntimeError("response banks share a seed ID")
    if np.shares_memory(left.responses, right.responses):
        raise RuntimeError("response banks share ndarray memory")
    if left.response_hash == right.response_hash:
        raise RuntimeError("independent response banks have the same response hash")
    if left.family_ids != right.family_ids or left.family_hash != right.family_hash:
        raise RuntimeError("response banks do not share the frozen hypothesis family")


def run_power_replicate(
    context: OmniBBenchmarkContext,
    *,
    calibration_bank: ConditionalBank,
    calibration_scenario_id: str,
    replicate: int,
    architecture: str,
    interaction_pve: float,
    causal_groups: int,
    calibration_count: int,
    response_count: int = 1,
    design_hash: str = _DEFAULT_DESIGN_HASH,
    qa_only: bool = True,
    n_jobs: int = 1,
    null_model: str = "gaussian",
    scenario_id: str = "B.power.synthetic",
    shard_path: str | Path | None = None,
    request_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Learn thresholds on calibration nulls, freeze them, then score causal responses."""

    calibration_count = _validate_count(calibration_count, "calibration_count")
    response_count = _validate_count(response_count, "response_count")
    if response_count != 1:
        raise ValueError("each power shard must contain exactly one target response")
    stage = _stage(qa_only)
    if not isinstance(calibration_bank, ConditionalBank):
        raise ValueError("power requires a frozen ConditionalBank")
    if (
        calibration_bank.canonical_role != "calibration"
        or calibration_bank.stage != stage
        or calibration_bank.design_hash != design_hash
        or len(calibration_bank.seed_ids) != calibration_count
        or calibration_bank.family_ids != context.family.group_ids
        or calibration_bank.family_hash != _family_hash(context.family)
        or calibration_bank.failure.get("failed") is not False
        or any(
            metadata.get("canonical_kind") != null_model
            for metadata in calibration_bank.response_metadata
        )
        or not calibration_scenario_id.endswith(".gaussian.calibration")
        or null_model != "gaussian"
    ):
        raise ValueError("power calibration bank differs from the frozen Gaussian bank")
    calibration_payload = calibration_bank.to_payload(include_scores=False)
    calibration_manifest_hash = sha256_payload(calibration_payload)
    request_record = request_identity or {
        "entrypoint": "run_power_replicate",
        "scenario_id": scenario_id,
        "replicate": replicate,
        "stage": stage,
        "qa_only": qa_only,
        "architecture": architecture,
        "interaction_pve": interaction_pve,
        "causal_groups": causal_groups,
        "calibration_count": calibration_count,
        "response_count": response_count,
        "null_model": null_model,
        "calibration_scenario_id": calibration_scenario_id,
        "calibration_bank_manifest_hash": calibration_manifest_hash,
        "n_jobs": n_jobs,
    }
    request_hash = _request_hash(
        design_hash=design_hash,
        context=context,
        request=request_record,
    )
    existing = _resume_existing_shard(
        shard_path,
        scenario_id=scenario_id,
        replicate=replicate,
        design_hash=design_hash,
        request_hash=request_hash,
    )
    if existing is not None:
        return existing
    _require_formal_budget(calibration_count, qa_only, "power calibration bank")
    started = time.perf_counter()
    prepared = _prepare_scenario(
        context,
        design_hash=design_hash,
        stage=stage,
        scenario_id=scenario_id,
        n_jobs=n_jobs,
    )
    calibration = calibration_bank
    signal, causal_ids, signal_metadata = _group_signal(
        prepared, architecture, causal_groups
    )

    def causal_response(
        rng: np.random.Generator, _replicate: int
    ) -> tuple[np.ndarray, dict[str, Any]]:
        residual, residual_metadata = draw_null(
            null_model, prepared.root_v, rng, prepared.pc1
        )
        phenotype, pve_metadata = compose_exact_pve(signal, residual, interaction_pve)
        return phenotype, {
            "null": residual_metadata,
            "pve": pve_metadata,
            "architecture": architecture,
            "causal_group_ids": causal_ids,
        }

    target = _bank_from_prepared(
        prepared,
        bank="power",
        count=response_count,
        design_hash=design_hash,
        stage=stage,
        scenario_id=scenario_id,
        replicate_offset=replicate * response_count,
        null_model=null_model,
        n_jobs=n_jobs,
        response_factory=causal_response,
        calibration_responses=calibration.responses,
        calibration_reference=_calibration_reference(
            calibration.responses,
            calibration.seed_ids,
            calibration_scenario_id,
        ),
    )
    _assert_independent_banks(calibration, target)
    thresholds: dict[str, float | None] = {}
    threshold_failures: dict[str, str] = {}
    for method in METHOD_NAMES:
        try:
            thresholds[method] = empirical_threshold(calibration.p_by_method[method])
        except ValueError as error:
            thresholds[method] = None
            threshold_failures[method] = str(error)
    rejections = {
        method: apply_threshold(target.p_by_method[method], thresholds[method]).tolist()
        for method in METHOD_NAMES
    }
    causal_index = np.array(
        [context.family.group_ids.index(group_id) for group_id in causal_ids], dtype=int
    )
    causal_detection = {
        method: (
            np.nanmin(target.p_by_method[method][causal_index], axis=0)
            < thresholds[method]
        ).tolist()
        if thresholds[method] is not None
        else [False] * response_count
        for method in METHOD_NAMES
    }
    calibration_minima = {
        method: np.where(
            np.isfinite(values).any(axis=0),
            np.where(np.isfinite(values), values, np.inf).min(axis=0),
            np.nan,
        )
        for method, values in calibration.p_by_method.items()
    }
    target_minima = {
        method: np.where(
            np.isfinite(values).any(axis=0),
            np.where(np.isfinite(values), values, np.inf).min(axis=0),
            np.nan,
        )
        for method, values in target.p_by_method.items()
    }
    causal_minima = {
        method: np.asarray(values[causal_index], dtype=float)
        for method, values in target.p_by_method.items()
    }
    calibration_minima_hashes = {
        method: sha256_payload(_json_safe(values))
        for method, values in calibration_minima.items()
    }
    target_minima_hashes = {
        method: sha256_payload(_json_safe(values))
        for method, values in target_minima.items()
    }
    causal_minima_hashes = {
        method: sha256_payload(_json_safe(values))
        for method, values in causal_minima.items()
    }
    recall = {
        method: (
            np.mean(causal_minima[method] < thresholds[method], axis=0).tolist()
            if thresholds[method] is not None
            else [0.0] * response_count
        )
        for method in METHOD_NAMES
    }
    payload = _base_replicate_payload(
        context,
        replicate=replicate,
        design_hash=design_hash,
        stage=stage,
        scenario_id=scenario_id,
        n_jobs=n_jobs,
    )
    payload.update(
        {
            "experiment": "power",
            "request_hash": request_hash,
            "architecture": architecture,
            "interaction_pve": float(interaction_pve),
            "causal_group_ids": causal_ids,
            "signal_metadata": signal_metadata,
            "calibration_role": calibration.canonical_role,
            "target_role": target.canonical_role,
            "calibration_seed_ids": list(calibration.seed_ids),
            "target_seed_ids": list(target.seed_ids),
            "calibration_response_ids": list(calibration.seed_ids),
            "target_response_ids": list(target.seed_ids),
            "failed_calibration_response_indices": list(
                calibration.failure.get("failed_response_indices", [])
            ),
            "failed_target_response_indices": list(
                target.failure.get("failed_response_indices", [])
            ),
            "calibration_response_hash": calibration.response_hash,
            "calibration_scenario_id": calibration_scenario_id,
            "calibration_bank_manifest_hash": calibration_manifest_hash,
            "target_response_hash": target.response_hash,
            "threshold_source": "independent_calibration_bank",
            "thresholds": thresholds,
            "rejections_by_method": rejections,
            "causal_detection_by_method": causal_detection,
            "recall_by_method": recall,
            "calibration_minima_by_method": calibration_minima,
            "target_minima_by_method": target_minima,
            "causal_minima_by_method": causal_minima,
            "calibration_minima_hashes": calibration_minima_hashes,
            "target_minima_hashes": target_minima_hashes,
            "causal_minima_hashes": causal_minima_hashes,
            "calibration_bank": calibration.to_payload(include_scores=False),
            "target_bank": target.to_payload(include_scores=False),
            "effective_jobs": target.execution.get("effective_jobs", 1),
            "parallel_backend": target.execution.get("backend", "serial"),
            "worker_pids": list(target.execution.get("worker_pids", [])),
            "null_covariance": {
                "components": dict(prepared.scores.covariance_components),
                "shape": list(prepared.scores.null_covariance.shape),
                "sha256": _array_hash(prepared.scores.null_covariance),
            },
            "failure": {
                "failed": bool(
                    calibration.failure.get("failed")
                    or target.failure.get("failed")
                    or threshold_failures
                ),
                "status": (
                    "partial_failure"
                    if calibration.failure.get("failed")
                    or target.failure.get("failed")
                    or threshold_failures
                    else "completed"
                ),
                "threshold_failures": threshold_failures,
                "calibration": dict(calibration.failure),
                "target": dict(target.failure),
            },
            "inference_status": (
                "noninferential_do_not_threshold" if qa_only else "formal"
            ),
            "runtime_seconds": float(time.perf_counter() - started),
        }
    )
    result = _json_safe(payload)
    if shard_path is not None:
        write_shard_exclusive(Path(shard_path), result)
    return result


def _score_fixed_context(
    context: OmniBBenchmarkContext,
    phenotype: np.ndarray,
    *,
    bootstrap_B: int,
    bootstrap_seed: int,
    n_jobs: int,
) -> dict[str, Any]:
    scores, _expanded = score_omnib_family(
        context.subdata,
        context.family,
        phenotype,
        context.sample_idx,
        transform="INT",
        bootstrap_B=bootstrap_B,
        bootstrap_seed=bootstrap_seed,
        n_jobs=n_jobs,
        grm_method="grm_from_X",
        maf_min=0.01,
        burden_maf=0.01,
        min_snp=3,
        cap=150,
        n_pc=3,
    )
    calibration = bootstrap_minp_calibration(
        scores.group_p[:, 0], scores.group_p[:, 1:], alpha=0.05
    )
    adjusted = np.asarray(calibration["adjusted_p_local"], dtype=float)
    group_ids = context.family.group_ids
    order = sorted(
        range(len(group_ids)),
        key=lambda index: (
            float(scores.group_p[index, 0])
            if np.isfinite(scores.group_p[index, 0])
            else math.inf,
            group_ids[index],
        ),
    )
    return {
        "group_ids": group_ids,
        "response_hash": _array_hash(np.asarray(phenotype, dtype=float)),
        "observed": np.asarray(scores.group_p[:, 0], dtype=float),
        "adjusted": adjusted,
        "decisions": adjusted <= 0.05,
        "rejections": tuple(group_ids[index] for index in calibration["rejected_local"]),
        "ranking_hash": sha256_payload([group_ids[index] for index in order]),
        "execution": dict(scores.parallel_execution),
        "estimable": np.asarray(scores.group_estimable, dtype=bool),
        "null_covariance": {
            "components": dict(scores.covariance_components),
            "shape": list(scores.null_covariance.shape),
            "sha256": _array_hash(scores.null_covariance),
        },
    }


def _copy_context(
    context: OmniBBenchmarkContext,
    *,
    subdata: dict[str, SubgenomeData] | None = None,
    family: MasterGroupFamily | None = None,
    phenotype: np.ndarray | None = None,
) -> OmniBBenchmarkContext:
    return OmniBBenchmarkContext(
        subdata=context.subdata if subdata is None else subdata,
        family=context.family if family is None else family,
        phenotype=context.phenotype if phenotype is None else phenotype,
        sample_idx=context.sample_idx,
    )


def _replace_subdata(
    context: OmniBBenchmarkContext,
    matrices: Mapping[str, np.ndarray],
    mappings: Mapping[str, Mapping[str, np.ndarray]] | None = None,
) -> OmniBBenchmarkContext:
    subdata = {
        label: SubgenomeData(
            X=np.array(matrices[label], dtype=float, copy=True),
            gene_snp={
                gene: np.array(indices, dtype=int, copy=True)
                for gene, indices in (
                    data.gene_snp if mappings is None else mappings[label]
                ).items()
            },
            samples=list(data.samples),
            chunk=None,
        )
        for label, data in context.subdata.items()
    }
    return _copy_context(context, subdata=subdata)


def _allele_flip_context(
    context: OmniBBenchmarkContext, fraction: float
) -> OmniBBenchmarkContext:
    matrices = {
        label: np.array(data.X, dtype=float, copy=True)
        for label, data in context.subdata.items()
    }
    locations = [
        (label, column)
        for label in context.family.subgenomes
        for column in range(matrices[label].shape[1])
    ]
    selected = int(round(fraction * len(locations)))
    for label, column in locations[:selected]:
        values = matrices[label][:, column]
        finite = np.isfinite(values)
        values[finite] = 2.0 - values[finite]
    return _replace_subdata(context, matrices)


def _column_permutation_context(context: OmniBBenchmarkContext) -> OmniBBenchmarkContext:
    matrices = {label: data.X for label, data in context.subdata.items()}
    mappings = {
        label: {
            gene: np.asarray(indices, dtype=int)[::-1]
            for gene, indices in data.gene_snp.items()
        }
        for label, data in context.subdata.items()
    }
    return _replace_subdata(context, matrices, mappings)


def _group_permutation_context(context: OmniBBenchmarkContext) -> OmniBBenchmarkContext:
    order = tuple(reversed(range(len(context.family.group_ids))))
    family = MasterGroupFamily(
        context.family.subgenomes,
        tuple(context.family.group_ids[index] for index in order),
        tuple(context.family.genes[index] for index in order),
    )
    return _copy_context(context, family=family)


def _restore_result(result: dict[str, Any], canonical_ids: Sequence[str]) -> dict[str, Any]:
    indices = np.array([result["group_ids"].index(group_id) for group_id in canonical_ids])
    restored = dict(result)
    restored["group_ids"] = tuple(canonical_ids)
    for key in ("observed", "adjusted", "decisions", "estimable"):
        restored[key] = np.asarray(result[key])[indices]
    order = sorted(
        range(len(canonical_ids)),
        key=lambda index: (
            float(restored["observed"][index])
            if np.isfinite(restored["observed"][index])
            else math.inf,
            canonical_ids[index],
        ),
    )
    restored["ranking_hash"] = sha256_payload([canonical_ids[index] for index in order])
    restored["rejections"] = tuple(
        group_id for group_id in canonical_ids if group_id in set(result["rejections"])
    )
    return restored


def _exact_comparison(
    baseline: dict[str, Any], candidate: dict[str, Any], *, required: bool = True
) -> dict[str, Any]:
    observed_identical = np.array_equal(
        baseline["observed"], candidate["observed"], equal_nan=True
    ) and _array_hash(baseline["observed"]) == _array_hash(candidate["observed"])
    decisions_identical = np.array_equal(
        baseline["adjusted"], candidate["adjusted"], equal_nan=True
    ) and np.array_equal(baseline["decisions"], candidate["decisions"])
    return {
        "required": required,
        "observed_arrays_identical": bool(observed_identical),
        "adjusted_decisions_identical": bool(decisions_identical),
        "ranking_hash_identical": baseline["ranking_hash"] == candidate["ranking_hash"],
        "rejection_sets_identical": set(baseline["rejections"]) == set(candidate["rejections"]),
        "baseline_observed_hash": _array_hash(baseline["observed"]),
        "candidate_observed_hash": _array_hash(candidate["observed"]),
    }


def _rank_correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    finite = np.isfinite(left) & np.isfinite(right)
    if finite.sum() < 2:
        return None
    left_order = np.argsort(np.argsort(left[finite], kind="stable"), kind="stable")
    right_order = np.argsort(np.argsort(right[finite], kind="stable"), kind="stable")
    correlation = np.corrcoef(left_order, right_order)[0, 1]
    return float(correlation) if np.isfinite(correlation) else None


def _robustness_metrics(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    k = min(10, len(baseline["group_ids"]))
    base_order = np.argsort(np.where(np.isfinite(baseline["observed"]), baseline["observed"], np.inf))
    candidate_order = np.argsort(
        np.where(np.isfinite(candidate["observed"]), candidate["observed"], np.inf)
    )
    base_top = {baseline["group_ids"][index] for index in base_order[:k]}
    candidate_top = {candidate["group_ids"][index] for index in candidate_order[:k]}
    union = base_top | candidate_top
    return {
        "fwer": float(bool(candidate["rejections"])),
        "power": None,
        "rank_correlation": _rank_correlation(
            baseline["observed"], candidate["observed"]
        ),
        "top_k": k,
        "top_k_jaccard": len(base_top & candidate_top) / len(union) if union else 1.0,
        "non_estimable_rate": float(1.0 - np.mean(candidate["estimable"])),
        "absolute_power_regret": None,
        "note": "single-replicate robustness payload; power/regret are aggregated by scenario",
    }


def _basic_robustness_contexts(
    context: OmniBBenchmarkContext, seed: int
) -> dict[str, OmniBBenchmarkContext]:
    rng = np.random.default_rng(seed)
    outputs: dict[str, OmniBBenchmarkContext] = {}
    for fraction in (0.02, 0.05):
        matrices = {
            label: np.array(data.X, dtype=float, copy=True)
            for label, data in context.subdata.items()
        }
        for values in matrices.values():
            mask = rng.random(values.shape) < fraction
            values[mask] = np.nan
        outputs[f"missingness_{int(fraction * 100)}pct"] = _replace_subdata(
            context, matrices
        )
    matrices = {
        label: np.array(data.X, dtype=float, copy=True)
        for label, data in context.subdata.items()
    }
    for values in matrices.values():
        mask = rng.random(values.shape) < 0.01
        values[mask] = (values[mask] + 1.0) % 3.0
    outputs["miscoding_1pct"] = _replace_subdata(context, matrices)

    def marker_context(
        marker_counts: Sequence[int], allele_frequency: float, marker_seed: int
    ) -> OmniBBenchmarkContext:
        marker_rng = np.random.default_rng(marker_seed)
        generated: dict[str, SubgenomeData] = {}
        for copy_index, label in enumerate(context.family.subgenomes):
            per_group = int(marker_counts[copy_index])
            blocks = [
                marker_rng.binomial(
                    2,
                    allele_frequency,
                    size=(context.phenotype.size, per_group),
                ).astype(float)
                for _ in context.family.group_ids
            ]
            values = np.column_stack(blocks)
            generated[label] = SubgenomeData(
                X=values,
                gene_snp={
                    gene: np.arange(index * per_group, (index + 1) * per_group)
                    for index, gene in enumerate(
                        row[copy_index] for row in context.family.genes
                    )
                },
                samples=list(context.subdata[label].samples),
                chunk=None,
            )
        return _copy_context(context, subdata=generated)

    copies = len(context.family.subgenomes)
    for count in (3, 10, 30):
        outputs[f"markers_per_gene_{count}"] = marker_context(
            [count] * copies, 0.25, seed + count
        )
    outputs["maf_0p01_0p05"] = marker_context([5] * copies, 0.03, seed + 101)
    outputs["maf_0p05_0p20"] = marker_context([5] * copies, 0.12, seed + 102)
    outputs["maf_above_0p20"] = marker_context([5] * copies, 0.30, seed + 103)
    imbalanced = tuple((3, 10, 30, 5)[:copies])
    outputs["unbalanced_marker_counts"] = marker_context(imbalanced, 0.25, seed + 104)
    return outputs


def run_encoding_check(
    context: OmniBBenchmarkContext,
    *,
    bootstrap_B: int = 19,
    design_hash: str = _DEFAULT_DESIGN_HASH,
    n_jobs: int = 1,
    parallel_jobs: int = 2,
    include_robustness: bool = True,
    qa_only: bool = True,
    scenario_id: str = "B.encoding.synthetic",
    replicate: int = 0,
) -> dict[str, Any]:
    """Run exact invariance challenges and separately labelled robustness checks."""

    bootstrap_B = _validate_count(bootstrap_B, "bootstrap_B")
    started = time.perf_counter()
    stage = _stage(qa_only)
    request_hash = _request_hash(
        design_hash=design_hash,
        context=context,
        request={
            "entrypoint": "run_encoding_check",
            "bootstrap_B": bootstrap_B,
            "n_jobs": n_jobs,
            "parallel_jobs": parallel_jobs,
            "include_robustness": include_robustness,
            "qa_only": qa_only,
            "scenario_id": scenario_id,
            "replicate": replicate,
        },
    )
    seed, seed_id = _seed(
        design_hash, scenario_id, replicate, stage, "calibration"
    )
    baseline = _score_fixed_context(
        context,
        context.phenotype,
        bootstrap_B=bootstrap_B,
        bootstrap_seed=seed,
        n_jobs=n_jobs,
    )
    exact_contexts = {
        "allele_flip_25pct": _allele_flip_context(context, 0.25),
        "allele_flip_50pct": _allele_flip_context(context, 0.50),
        "allele_flip_100pct": _allele_flip_context(context, 1.00),
        "within_gene_snp_column_permutation": _column_permutation_context(context),
        "group_row_permutation_restored_ids": _group_permutation_context(context),
    }
    exact_checks: dict[str, dict[str, Any]] = {}
    for name, transformed in exact_contexts.items():
        candidate = _score_fixed_context(
            transformed,
            context.phenotype,
            bootstrap_B=bootstrap_B,
            bootstrap_seed=seed,
            n_jobs=n_jobs,
        )
        if name == "group_row_permutation_restored_ids":
            candidate = _restore_result(candidate, context.family.group_ids)
        exact_checks[name] = _exact_comparison(baseline, candidate)
    parallel = _score_fixed_context(
        context,
        context.phenotype,
        bootstrap_B=bootstrap_B,
        bootstrap_seed=seed,
        n_jobs=parallel_jobs,
    )
    parallel_execution = parallel["execution"]
    admitted = (
        parallel_execution.get("backend") == "fork_shared_memory"
        and int(parallel_execution.get("effective_jobs", 1)) > 1
    )
    exact_checks["serial_vs_admitted_parallel"] = _exact_comparison(
        baseline, parallel, required=admitted
    )
    exact_checks["serial_vs_admitted_parallel"].update(
        {
            "requested_jobs": parallel_jobs,
            "effective_jobs": parallel_execution.get("effective_jobs", 1),
            "backend": parallel_execution.get("backend", "serial"),
            "worker_pids": list(parallel_execution.get("worker_pids", [])),
            "skip_reason": None if admitted else "parallel_backend_not_admitted",
        }
    )
    robustness: dict[str, Any] = {}
    if include_robustness:
        for name, transformed in _basic_robustness_contexts(context, seed).items():
            try:
                candidate = _score_fixed_context(
                    transformed,
                    context.phenotype,
                    bootstrap_B=bootstrap_B,
                    bootstrap_seed=seed,
                    n_jobs=n_jobs,
                )
                robustness[name] = {
                    "status": "completed",
                    **_robustness_metrics(baseline, candidate),
                }
            except Exception as error:
                robustness[name] = {
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "message": str(error),
                    "fwer": None,
                    "power": None,
                    "rank_correlation": None,
                    "top_k_jaccard": None,
                    "non_estimable_rate": None,
                    "absolute_power_regret": None,
                }
    required_checks = [check for check in exact_checks.values() if check["required"]]
    exact_fields = (
        "observed_arrays_identical",
        "adjusted_decisions_identical",
        "ranking_hash_identical",
        "rejection_sets_identical",
    )
    all_required_exact = all(
        all(bool(check[field]) for field in exact_fields)
        for check in required_checks
    )
    return _json_safe(
        {
            **_base_replicate_payload(
                context,
                replicate=replicate,
                design_hash=design_hash,
                stage=stage,
                scenario_id=scenario_id,
                n_jobs=n_jobs,
            ),
            "experiment": "encoding",
            "inference_status": "noninferential_do_not_threshold",
            "request_hash": request_hash,
            "seed_id": seed_id,
            "response_hash": baseline["response_hash"],
            "bootstrap_B": bootstrap_B,
            "hypothesis_unit": "group",
            "family_scope": "primary_only",
            "pair_edges_per_group": _edge_count_per_group(context),
            "direct_higher_order_term": False,
            "null_covariance": baseline["null_covariance"],
            "exact_checks": exact_checks,
            "all_required_exact": all_required_exact,
            "failure": {
                "failed": not all_required_exact,
                "status": (
                    "completed" if all_required_exact else "failed_exact_invariance"
                ),
                "failed_checks": [
                    name
                    for name, check in exact_checks.items()
                    if check["required"]
                    and not all(bool(check[field]) for field in exact_fields)
                ],
            },
            "robustness_checks": robustness,
            "robustness_is_exact_invariance": False,
            "effective_jobs": baseline["execution"].get("effective_jobs", 1),
            "parallel_backend": baseline["execution"].get("backend", "serial"),
            "worker_pids": list(baseline["execution"].get("worker_pids", [])),
            "runtime_seconds": float(time.perf_counter() - started),
        }
    )


def run_omnib_replicate(
    scenario: Scenario,
    context: OmniBBenchmarkContext,
    *,
    replicate: int,
    design_hash: str,
    n_jobs: int = 1,
    shard_path: str | Path | None = None,
    power_calibration_bank: ConditionalBank | None = None,
) -> dict[str, Any]:
    """Dispatch one Track B scenario and optionally create its immutable shard."""

    if scenario.track != "omnib":
        raise ValueError("run_omnib_replicate requires an omnib scenario")
    if replicate < 0 or replicate >= scenario.replicates:
        raise ValueError("replicate is outside the scenario registry range")
    experiment = scenario.parameters.get("experiment")
    if experiment not in {"end2end", "conditional", "power", "encoding"}:
        raise ValueError(f"unsupported Track B experiment: {experiment!r}")
    canonical_bank = (
        _canonical_role(str(scenario.parameters.get("bank", "heldout")))
        if experiment == "conditional"
        else None
    )
    request_record = {
        "entrypoint": "run_omnib_replicate",
        "scenario": scenario.to_dict(),
        "replicate": replicate,
        "experiment": experiment,
        "canonical_bank": canonical_bank,
        "n_jobs": n_jobs,
    }
    if experiment == "power":
        request_record["calibration_bank_manifest_hash"] = (
            sha256_payload(power_calibration_bank.to_payload(include_scores=False))
            if isinstance(power_calibration_bank, ConditionalBank) else None
        )
    request_hash = _request_hash(
        design_hash=design_hash,
        context=context,
        request=request_record,
    )
    existing = _resume_existing_shard(
        shard_path,
        scenario_id=scenario.scenario_id,
        replicate=replicate,
        design_hash=design_hash,
        request_hash=request_hash,
    )
    if existing is not None:
        return existing
    qa_only = scenario.stage == "pilot"
    started = time.perf_counter()
    try:
        if experiment == "end2end":
            payload = run_end_to_end_null(
                context,
                replicate=replicate,
                bootstrap_B=scenario.bootstrap_B,
                qa_only=qa_only,
                n_jobs=n_jobs,
                design_hash=design_hash,
                null_model=str(scenario.parameters.get("null_model", "gaussian")),
                scenario_id=scenario.scenario_id,
            )
        elif experiment == "conditional":
            bank = run_conditional_bank(
                context,
                bank=str(scenario.parameters.get("bank", "heldout")),
                count=scenario.replicates,
                design_hash=design_hash,
                n_jobs=n_jobs,
                qa_only=qa_only,
                null_model=str(scenario.parameters.get("null_model", "gaussian")),
                scenario_id=scenario.scenario_id,
                replicate_offset=0,
            )
            payload = {
                **_base_replicate_payload(
                    context,
                    replicate=replicate,
                    design_hash=design_hash,
                    stage=scenario.stage,
                    scenario_id=scenario.scenario_id,
                    n_jobs=n_jobs,
                ),
                "experiment": "conditional",
                "bank": bank.to_payload(),
                "failure": dict(bank.failure),
            }
        elif experiment == "power":
            payload = run_power_replicate(
                context,
                calibration_bank=power_calibration_bank,
                calibration_scenario_id=str(
                    scenario.parameters["calibration_scenario_id"]
                ),
                replicate=replicate,
                architecture=str(scenario.parameters["architecture"]),
                interaction_pve=float(scenario.parameters["interaction_pve"]),
                causal_groups=int(scenario.parameters["causal_groups"]),
                calibration_count=int(scenario.parameters["calibration_count"]),
                response_count=1,
                design_hash=design_hash,
                qa_only=qa_only,
                n_jobs=n_jobs,
                null_model=str(scenario.parameters.get("null_model", "gaussian")),
                scenario_id=scenario.scenario_id,
                request_identity=request_record,
            )
        else:
            payload = run_encoding_check(
                context,
                bootstrap_B=scenario.bootstrap_B,
                design_hash=design_hash,
                n_jobs=n_jobs,
                parallel_jobs=int(
                    scenario.parameters.get("parallel_jobs", max(2, n_jobs))
                ),
                include_robustness=bool(
                    scenario.parameters.get("include_robustness", True)
                ),
                qa_only=qa_only,
                scenario_id=scenario.scenario_id,
                replicate=replicate,
            )
    except Exception as error:
        payload = {
            **_base_replicate_payload(
                context,
                replicate=replicate,
                design_hash=design_hash,
                stage=scenario.stage,
                scenario_id=scenario.scenario_id,
                n_jobs=n_jobs,
            ),
            "experiment": experiment,
            "runtime_seconds": float(time.perf_counter() - started),
            "effective_jobs": 0,
            "parallel_backend": "failed",
            "worker_pids": [],
            "failure": {
                "failed": True,
                "error_type": type(error).__name__,
                "message": str(error),
            },
        }
    if experiment == "power" and payload.get("request_hash") not in {
        None, request_hash,
    }:
        raise RuntimeError("inner omniB request identity differs from dispatcher")
    payload["request_hash"] = request_hash
    payload["context_fingerprint"] = _context_fingerprint(context)
    result = _json_safe(payload)
    if shard_path is not None:
        write_shard_exclusive(Path(shard_path), result)
    return result


__all__ = [
    "ConditionalBank",
    "OmniBBenchmarkContext",
    "apply_threshold",
    "build_synthetic_omnib_context",
    "empirical_threshold",
    "run_conditional_bank",
    "run_encoding_check",
    "run_end_to_end_null",
    "run_omnib_replicate",
    "run_power_replicate",
]
