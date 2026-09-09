"""Track B end-to-end, conditional-bank, power and encoding benchmarks.

The runner keeps the group family frozen, derives every response from an
auditable role-specific seed namespace, and never learns a decision threshold
from the response bank on which it is evaluated.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from homoeogwas.group_family import ExpandedEdgeFamily, MasterGroupFamily, load_master_group_family
from homoeogwas.interact import (
    SubgenomeData,
    _load_subgenome,
    build_retained_variant_mask,
)
from homoeogwas.omnib_family import (
    OmniBFamilyScores,
    _array_identity,
    _text_identity,
    bootstrap_minp_calibration,
    prepare_omnib_design,
    score_omnib_family,
    score_omnib_responses,
)

from .comparators import (
    GLOBAL_VC_METHOD,
    METHOD_NAMES,
    MethodScoreBank,
    SNPxSNPScoreResult,
    group_component_p,
    score_global_hadamard_vc,
    score_snpxsnp_family,
)
from .contracts import (
    Scenario,
    comparator_resource_limit,
    derive_seed,
    sha256_payload,
)
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
_BINDING_GAUSSIAN_FAILURE_RATE_MAX = 0.002
_DIAGNOSTIC_FAILURE_RATE_MAX = 0.01
_DEFAULT_SYNTHETIC_FEATURE_SEED = 20_260_830
_SYNTHETIC_FIXTURE_MAX_OFFERED_SNP_PAIRS = 750_000
_SNPXSNP_FIXTURE_PANEL_IDS = frozenset({
    "SYNTHETIC", "FIXTURE", "SYNTH.OVERCAP",
})


@dataclass(frozen=True)
class OmniBBenchmarkContext:
    """One phenotype-independent genotype/family backbone plus an anchor trait."""

    subdata: dict[str, SubgenomeData]
    family: MasterGroupFamily
    phenotype: np.ndarray
    sample_idx: np.ndarray
    panel_id: str = "SYNTHETIC"
    sample_context: str = "full"
    feature_seed: int = _DEFAULT_SYNTHETIC_FEATURE_SEED
    retained_variant_masks: Mapping[str, np.ndarray] | None = field(
        default=None, repr=False
    )
    marker_mask_identity: Mapping[str, Mapping[str, Any]] | None = None
    marker_mask_sha256: str = field(init=False)

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
        if not isinstance(self.panel_id, str) or not self.panel_id:
            raise ValueError("panel_id must be a non-empty string")
        if not isinstance(self.sample_context, str) or not self.sample_context:
            raise ValueError("sample_context must be a non-empty string")
        if (
            isinstance(self.feature_seed, bool)
            or not isinstance(self.feature_seed, (int, np.integer))
            or int(self.feature_seed) < 0
        ):
            raise ValueError("feature_seed must be a non-negative integer")
        for label, data in self.subdata.items():
            values = np.asarray(data.X)
            if values.ndim != 2 or values.shape[0] <= int(sample_idx.max()):
                raise ValueError(f"subgenome {label} is not aligned to sample_idx")
        supplied_masks = self.retained_variant_masks
        if supplied_masks is not None and (
            not isinstance(supplied_masks, Mapping)
            or set(supplied_masks) != set(self.family.subgenomes)
        ):
            raise ValueError(
                "retained_variant_masks must exactly match context subgenomes"
            )
        masks: dict[str, np.ndarray] = {}
        generated_identity: dict[str, dict[str, Any]] = {}
        sample_hash = _array_hash(sample_idx)
        for label in self.family.subgenomes:
            values = np.asarray(self.subdata[label].X)
            if supplied_masks is None:
                mask, qc = build_retained_variant_mask(
                    values[sample_idx],
                    call_rate_min=0.90,
                    maf_min=0.01,
                    mac_min=5,
                )
                source = "derived_context_qc"
            else:
                raw = supplied_masks[label]
                if not isinstance(raw, np.ndarray) or raw.dtype != np.bool_:
                    raise ValueError(
                        "retained_variant_masks entries must be boolean ndarrays"
                    )
                mask = np.array(raw, dtype=bool, copy=True)
                if mask.ndim != 1 or mask.size != values.shape[1]:
                    raise ValueError(
                        "retained_variant_masks entries must align to variants"
                    )
                qc = {
                    "n_samples": int(sample_idx.size),
                    "n_variants_input": int(mask.size),
                    "n_variants_retained": int(mask.sum()),
                    "retained_variant_mask_encoding": (
                        "uint8_input_variant_order"
                    ),
                    "retained_variant_mask_sha256": hashlib.sha256(
                        np.ascontiguousarray(mask, dtype=np.uint8).tobytes()
                    ).hexdigest(),
                }
                source = "explicit_context_mask"
            mask.setflags(write=False)
            masks[label] = mask
            generated_identity[label] = {
                "panel_id": self.panel_id,
                "sample_context": self.sample_context,
                "subgenome": label,
                "sample_index_sha256": sample_hash,
                "source": source,
                **qc,
            }
        identity = self.marker_mask_identity
        if identity is None:
            identity_copy = generated_identity
        else:
            if not isinstance(identity, Mapping) or set(identity) != set(masks):
                raise ValueError(
                    "marker_mask_identity must exactly match context subgenomes"
                )
            identity_copy = {
                label: dict(identity[label]) for label in self.family.subgenomes
            }
            for label, record in identity_copy.items():
                expected = generated_identity[label]
                for key in (
                    "panel_id",
                    "sample_context",
                    "subgenome",
                    "sample_index_sha256",
                    "n_variants_input",
                    "n_variants_retained",
                    "retained_variant_mask_sha256",
                ):
                    if record.get(key) != expected[key]:
                        raise ValueError(
                            f"marker_mask_identity {key} mismatch for {label}"
                        )
        phenotype.setflags(write=False)
        sample_idx.setflags(write=False)
        object.__setattr__(self, "phenotype", phenotype)
        object.__setattr__(self, "sample_idx", sample_idx)
        object.__setattr__(self, "feature_seed", int(self.feature_seed))
        object.__setattr__(
            self, "retained_variant_masks", MappingProxyType(masks)
        )
        frozen_identity = MappingProxyType({
            label: MappingProxyType(dict(identity_copy[label]))
            for label in self.family.subgenomes
        })
        object.__setattr__(self, "marker_mask_identity", frozen_identity)
        object.__setattr__(
            self,
            "marker_mask_sha256",
            sha256_payload({
                label: dict(frozen_identity[label])
                for label in self.family.subgenomes
            }),
        )


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
    panel_id: str
    sample_context: str
    feature_seed: int
    marker_mask_identity: Mapping[str, Mapping[str, Any]]
    marker_mask_sha256: str
    feature_cache_sha256: str
    fixed_mask_sha256: str
    null_fit_sha256: str
    prepared_design_sha256: str
    score_bank: MethodScoreBank
    tested_family_members: Mapping[str, tuple[str, ...]]
    tested_family_hashes: Mapping[str, str]
    score_matrix_hashes: Mapping[str, str]
    snpxsnp_evidence: Mapping[str, Any]
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
        if not isinstance(self.panel_id, str) or not self.panel_id:
            raise ValueError("conditional bank panel_id must be non-empty")
        if not isinstance(self.sample_context, str) or not self.sample_context:
            raise ValueError("conditional bank sample_context must be non-empty")
        if (
            isinstance(self.feature_seed, bool)
            or not isinstance(self.feature_seed, (int, np.integer))
            or int(self.feature_seed) < 0
        ):
            raise ValueError("conditional bank feature_seed must be non-negative")
        for name in (
            "marker_mask_sha256",
            "feature_cache_sha256",
            "fixed_mask_sha256",
            "null_fit_sha256",
            "prepared_design_sha256",
        ):
            digest = getattr(self, name)
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError(f"conditional bank {name} must be SHA-256")
        if (
            not isinstance(self.marker_mask_identity, Mapping)
            or not self.marker_mask_identity
            or self.marker_mask_sha256 != sha256_payload({
                str(label): dict(record)
                for label, record in self.marker_mask_identity.items()
            })
        ):
            raise ValueError("conditional bank marker-mask identity is invalid")
        methods = set(self.score_bank.p_by_method)
        if not (
            methods
            == set(self.tested_family_members)
            == set(self.tested_family_hashes)
            == set(self.score_matrix_hashes)
        ):
            raise ValueError("score and tested-family method sets must match")
        if (
            not isinstance(self.snpxsnp_evidence, Mapping)
            or self.snpxsnp_evidence.get("schema") != "snpxsnp_raw_stream_v2"
            or self.snpxsnp_evidence.get("tested_pair_count")
            != self.score_bank.tested_family_sizes["snpxsnp"]
            or self.snpxsnp_evidence.get("member_ids")
            != list(self.tested_family_members["snpxsnp"])
            or self.snpxsnp_evidence.get("member_family_sha256")
            != self.tested_family_hashes["snpxsnp"]
        ):
            raise ValueError("streaming SNPxSNP evidence differs from the score family")
        values.setflags(write=False)
        object.__setattr__(self, "responses", values)
        object.__setattr__(self, "feature_seed", int(self.feature_seed))
        object.__setattr__(
            self,
            "marker_mask_identity",
            MappingProxyType({
                str(label): MappingProxyType(dict(record))
                for label, record in self.marker_mask_identity.items()
            }),
        )
        object.__setattr__(
            self, "snpxsnp_evidence", MappingProxyType(dict(self.snpxsnp_evidence))
        )

    @property
    def p_by_method(self) -> Mapping[str, np.ndarray]:
        return self.score_bank.p_by_method

    def to_payload(
        self,
        *,
        include_scores: bool = True,
    ) -> dict[str, Any]:
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
            "responses": _json_safe(self.responses),
            "response_metadata": list(self.response_metadata),
            "family_ids": list(self.family_ids),
            "family_hash": self.family_hash,
            "design_hash": self.design_hash,
            "panel_id": self.panel_id,
            "sample_context": self.sample_context,
            "feature_seed": self.feature_seed,
            "marker_mask_identity": {
                label: dict(record)
                for label, record in self.marker_mask_identity.items()
            },
            "marker_mask_sha256": self.marker_mask_sha256,
            "feature_cache_sha256": self.feature_cache_sha256,
            "fixed_mask_sha256": self.fixed_mask_sha256,
            "null_fit_sha256": self.null_fit_sha256,
            "prepared_design_sha256": self.prepared_design_sha256,
            "tested_family_sizes": dict(self.score_bank.tested_family_sizes),
            "tested_family_members": {
                method: list(members)
                for method, members in self.tested_family_members.items()
            },
            "tested_family_hashes": dict(self.tested_family_hashes),
            "score_hypothesis_units": {
                method: (
                    "snp_pair_within_group" if method == "snpxsnp" else "group_score"
                )
                for method in METHOD_NAMES
            },
            "score_matrix_hashes": dict(self.score_matrix_hashes),
            "snpxsnp_evidence": dict(self.snpxsnp_evidence),
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

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> ConditionalBank:
        """Load a restart-stable bank and verify every serialized commitment."""

        if not isinstance(payload, Mapping):
            raise ValueError("conditional bank payload must be a mapping")
        responses = np.asarray(payload.get("responses"), dtype=float)
        p_by_method = payload.get("p_by_method")
        sizes = payload.get("tested_family_sizes")
        snpxsnp_evidence = payload.get("snpxsnp_evidence")
        if (
            responses.ndim != 2
            or list(responses.shape) != payload.get("response_shape")
            or _array_hash(responses) != payload.get("response_hash")
            or not isinstance(p_by_method, Mapping)
            or not isinstance(sizes, Mapping)
            or not isinstance(snpxsnp_evidence, Mapping)
        ):
            raise ValueError("conditional bank payload evidence is incomplete")
        score_bank = MethodScoreBank(
            family_ids=tuple(payload.get("family_ids", [])),
            p_by_method={
                method: np.asarray(p_by_method[method], dtype=float)
                for method in METHOD_NAMES
            },
            tested_family_sizes={method: int(sizes[method]) for method in METHOD_NAMES},
        )
        score_hashes = payload.get("score_matrix_hashes")
        if not isinstance(score_hashes, Mapping) or any(
            score_hashes.get(method)
            != sha256_payload(_json_safe(score_bank.p_by_method[method]))
            for method in METHOD_NAMES
        ):
            raise ValueError("conditional bank score hash mismatch")
        bank = cls(
            requested_role=str(payload.get("requested_role")),
            canonical_role=str(payload.get("canonical_role")),
            stage=str(payload.get("stage")),
            seed_ids=tuple(payload.get("seed_ids", [])),
            seeds=tuple(int(value) for value in payload.get("seeds", [])),
            responses=responses,
            response_hash=str(payload.get("response_hash")),
            response_metadata=tuple(dict(value) for value in payload.get("response_metadata", [])),
            family_ids=tuple(payload.get("family_ids", [])),
            family_hash=str(payload.get("family_hash")),
            design_hash=str(payload.get("design_hash")),
            panel_id=str(payload.get("panel_id")),
            sample_context=str(payload.get("sample_context")),
            feature_seed=payload.get("feature_seed"),
            marker_mask_identity=dict(payload.get("marker_mask_identity", {})),
            marker_mask_sha256=str(payload.get("marker_mask_sha256")),
            feature_cache_sha256=str(payload.get("feature_cache_sha256")),
            fixed_mask_sha256=str(payload.get("fixed_mask_sha256")),
            null_fit_sha256=str(payload.get("null_fit_sha256")),
            prepared_design_sha256=str(payload.get("prepared_design_sha256")),
            score_bank=score_bank,
            tested_family_members={
                method: tuple(payload["tested_family_members"][method])
                for method in METHOD_NAMES
            },
            tested_family_hashes=dict(payload.get("tested_family_hashes", {})),
            score_matrix_hashes=dict(score_hashes),
            snpxsnp_evidence=dict(snpxsnp_evidence),
            calibration_reference=dict(payload.get("calibration_reference", {})),
            execution=dict(payload.get("execution", {})),
            runtime_seconds=float(payload.get("runtime_seconds", 0.0)),
            null_covariance=dict(payload.get("null_covariance", {})),
            failure=dict(payload.get("failure", {})),
        )
        if bank.to_payload() != dict(payload):
            raise ValueError("conditional bank payload is not canonical")
        return bank

    @classmethod
    def from_shard(cls, path: str | Path) -> ConditionalBank:
        """Load either a bare bank artifact or a conditional shard after restart."""

        document = json.loads(Path(path).read_text(encoding="utf-8"))
        payload = document.get("bank", document) if isinstance(document, Mapping) else document
        return cls.from_payload(payload)

    def statistical_artifact(self) -> dict[str, Any]:
        """Restart-stable inferential commitment (no PID/runtime fields)."""

        minima = {
            method: _json_safe(np.where(
                np.isfinite(values).any(axis=0),
                np.where(np.isfinite(values), values, np.inf).min(axis=0),
                np.nan,
            ))
            for method, values in self.score_bank.p_by_method.items()
        }
        artifact = {
            "artifact_schema": "conditional_calibration_v1",
            "stage": self.stage,
            "canonical_role": self.canonical_role,
            "seed_ids": list(self.seed_ids),
            "seeds": list(self.seeds),
            "response_hash": self.response_hash,
            "response_metadata": [dict(item) for item in self.response_metadata],
            "family_ids": list(self.family_ids),
            "family_hash": self.family_hash,
            "design_hash": self.design_hash,
            "panel_id": self.panel_id,
            "sample_context": self.sample_context,
            "feature_seed": self.feature_seed,
            "marker_mask_sha256": self.marker_mask_sha256,
            "feature_cache_sha256": self.feature_cache_sha256,
            "fixed_mask_sha256": self.fixed_mask_sha256,
            "null_fit_sha256": self.null_fit_sha256,
            "prepared_design_sha256": self.prepared_design_sha256,
            "tested_family_sizes": dict(self.score_bank.tested_family_sizes),
            "tested_family_members": {
                method: list(value) for method, value in self.tested_family_members.items()
            },
            "tested_family_hashes": dict(self.tested_family_hashes),
            "score_matrix_hashes": dict(self.score_matrix_hashes),
            "snpxsnp_evidence": dict(self.snpxsnp_evidence),
            "method_minima": minima,
            "method_minima_hashes": {
                method: sha256_payload(value) for method, value in minima.items()
            },
            "failed_response_indices": list(
                self.failure.get("failed_response_indices", [])
            ),
        }
        return artifact


@dataclass(frozen=True)
class _PreparedScenario:
    context: OmniBBenchmarkContext
    scores: OmniBFamilyScores
    expanded: ExpandedEdgeFamily
    root_v: np.ndarray
    pc1: np.ndarray
    genotype_main_effect: np.ndarray
    omitted_kernel: np.ndarray
    omitted_subgenome: str
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
        "panel_id": context.panel_id,
        "sample_context": context.sample_context,
        "feature_seed": context.feature_seed,
        "marker_mask_identity": {
            label: dict(context.marker_mask_identity[label])
            for label in context.family.subgenomes
        },
        "marker_mask_sha256": context.marker_mask_sha256,
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


def _resolved_path(value: Any, base: Path, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"validated config {field} must be a non-empty path")
    path = Path(value)
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def load_omnib_benchmark_context(
    config_path: str | Path,
    *,
    context_artifact_path: str | Path,
    input_manifest: Mapping[str, str],
) -> OmniBBenchmarkContext:
    """Build a real benchmark context from the released interaction inputs.

    The caller supplies the already validated input-manifest identities.  The
    loader nevertheless re-reads BED/FAM/BIM, verified mapping NPZs, groups and
    phenotype, then requires the resulting manifest to equal the presealed
    context artifact.
    """

    import yaml

    config_file = Path(config_path).resolve()
    raw = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or not isinstance(raw.get("interact"), Mapping):
        raise ValueError("validated config must contain an interact mapping")
    ic = raw["interact"]
    benchmark_identity = ic.get("benchmark_identity")
    if (
        not isinstance(benchmark_identity, Mapping)
        or set(benchmark_identity)
        != {"panel_id", "sample_context", "feature_seed"}
    ):
        raise ValueError("validated config benchmark identity is invalid")
    exact = {
        "mode": "group", "statistic": "omniB", "hypothesis_unit": "group",
        "subset_order": 2, "family_scope": "primary_only",
        "primary_transform": "INT", "primary_multiplicity": "bootstrap_minp",
    }
    if any(ic.get(field) != value for field, value in exact.items()):
        raise ValueError("validated config differs from canonical group-omniB science")
    calibration = ic.get("calibration")
    grm = ic.get("grm")
    if (
        not isinstance(calibration, Mapping)
        or calibration.get("method") != "bootstrap"
        or isinstance(calibration.get("B"), bool)
        or not isinstance(calibration.get("B"), int)
        or calibration["B"] < 1
        or isinstance(calibration.get("seed"), bool)
        or not isinstance(calibration.get("seed"), int)
        or grm != {"method": "grm_from_X", "maf_min": 0.01,
                   "scope": "all_subgenomes"}
    ):
        raise ValueError("validated config calibration/GRM contract is invalid")
    subgenomes = ic.get("subgenomes")
    genotype, mappings = ic.get("genotype"), ic.get("snp_to_gene")
    if (
        not isinstance(subgenomes, list) or len(subgenomes) not in {2, 3, 4}
        or len(set(subgenomes)) != len(subgenomes)
        or not isinstance(genotype, Mapping) or set(genotype) != set(subgenomes)
        or not isinstance(mappings, Mapping) or set(mappings) != set(subgenomes)
    ):
        raise ValueError("validated config subgenome input maps are invalid")
    base = config_file.parent
    resolved_required: list[Path] = [config_file]
    subdata: dict[str, SubgenomeData] = {}
    for label in subgenomes:
        prefix = _resolved_path(genotype[label], base, f"genotype.{label}")
        mapping = _resolved_path(mappings[label], base, f"snp_to_gene.{label}")
        resolved_required.extend(
            [Path(str(prefix) + suffix) for suffix in (".bed", ".bim", ".fam")]
        )
        resolved_required.append(mapping)
        subdata[label] = _load_subgenome(
            str(prefix), str(mapping), verify_mapping=True
        )
    groups_path = _resolved_path(ic.get("groups"), base, "groups")
    phenotype_path = _resolved_path(ic.get("phenotype"), base, "phenotype")
    resolved_required.extend([groups_path, phenotype_path])
    normalized_manifest = {
        str(Path(path).resolve()): digest for path, digest in input_manifest.items()
    }
    for path in resolved_required:
        expected = normalized_manifest.get(str(path.resolve()))
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
        if expected != observed:
            raise ValueError(f"resolved input is absent or differs from manifest: {path}")
    family = load_master_group_family(groups_path, subgenomes, require_group_id=True)
    sample_col, trait = ic.get("sample_col"), ic.get("trait")
    if not isinstance(sample_col, str) or not isinstance(trait, str):
        raise ValueError("validated config phenotype columns are invalid")
    with phenotype_path.open(encoding="utf-8", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters="\t,")
        except csv.Error:
            dialect = csv.excel_tab
        reader = csv.DictReader(handle, dialect=dialect)
        if sample_col not in (reader.fieldnames or ()) or trait not in (reader.fieldnames or ()):
            raise ValueError("phenotype columns are absent")
        phenotype_by_id: dict[str, float] = {}
        for row in reader:
            sample_id = str(row[sample_col])
            if sample_id in phenotype_by_id:
                raise ValueError("phenotype sample IDs must be unique strings")
            try:
                value = float(row[trait])
            except (TypeError, ValueError):
                continue
            if np.isfinite(value):
                phenotype_by_id[sample_id] = value
    reference_samples = tuple(subdata[subgenomes[0]].samples)
    if any(tuple(subdata[label].samples) != reference_samples for label in subgenomes[1:]):
        raise ValueError("real PLINK subgenomes do not share the same ordered sample IDs")
    sample_idx = np.asarray(
        [index for index, sample_id in enumerate(reference_samples)
         if str(sample_id) in phenotype_by_id], dtype=int,
    )
    phenotype = np.asarray(
        [phenotype_by_id[str(reference_samples[index])] for index in sample_idx],
        dtype=float,
    )
    context = OmniBBenchmarkContext(
        subdata,
        family,
        phenotype,
        sample_idx,
        panel_id=benchmark_identity["panel_id"],
        sample_context=benchmark_identity["sample_context"],
        feature_seed=benchmark_identity["feature_seed"],
    )
    artifact_file = Path(context_artifact_path).resolve()
    artifact = json.loads(artifact_file.read_text(encoding="utf-8"))
    manifest = _context_manifest(context)
    if (
        not isinstance(artifact, Mapping)
        or artifact.get("schema") != "homoeogwas-v201-omnib-context-v1"
        or artifact.get("context_manifest") != manifest
        or artifact.get("context_fingerprint") != sha256_payload(manifest)
        or artifact.get("family_manifest") != _family_manifest(family)
        or artifact.get("family_hash") != _family_hash(family)
    ):
        raise ValueError("real context differs from the presealed context artifact")
    return context


def validate_real_omnib_context(
    config_path: str | Path,
    *,
    context_artifact_path: str | Path,
    input_manifest: Mapping[str, str],
    stage: str,
    backbone: str,
    family_size: int,
    expected_subgenomes: Sequence[str],
    manifest_root: str | Path | None = None,
) -> OmniBBenchmarkContext:
    """Strict Task-9 validation entry for one sealed real omniB context."""

    import yaml

    config_file = Path(config_path).resolve()
    try:
        raw = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ValueError("strict benchmark interaction config is unreadable") from error
    interact = raw.get("interact") if isinstance(raw, Mapping) else None
    calibration = interact.get("calibration") if isinstance(interact, Mapping) else None
    benchmark_identity = (
        interact.get("benchmark_identity") if isinstance(interact, Mapping) else None
    )
    expected_calibration = {
        "method": "bootstrap",
        "B": 199 if stage == "pilot" else 2_000,
        "seed": 2026,
        "qa_only": stage == "pilot",
    }
    if (
        stage not in {"pilot", "formal"}
        or not isinstance(backbone, str) or not backbone
        or isinstance(family_size, bool) or not isinstance(family_size, int)
        or family_size < 1
        or not isinstance(interact, Mapping)
        or interact.get("burden") != {
            "cap": 150, "min_snp": 3, "maf_min": 0.01, "n_pc": 3,
        }
        or calibration != expected_calibration
        or interact.get("subgenomes") != list(expected_subgenomes)
        or not isinstance(benchmark_identity, Mapping)
        or set(benchmark_identity)
        != {"panel_id", "sample_context", "feature_seed"}
    ):
        raise ValueError("strict benchmark interaction config contract is invalid")
    context = load_omnib_benchmark_context(
        config_file, context_artifact_path=context_artifact_path,
        input_manifest=input_manifest,
    )
    artifact_file = Path(context_artifact_path).resolve()
    try:
        artifact = json.loads(artifact_file.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("strict benchmark context artifact is unreadable") from error
    ordered_ids = list(context.family.group_ids)
    if (
        not isinstance(artifact, Mapping)
        or artifact.get("backbone") != backbone
        or artifact.get("group_count") != family_size
        or len(ordered_ids) != family_size
        or artifact.get("ordered_family_ids") != ordered_ids
        or artifact.get("ordered_family_ids_hash") != sha256_payload(ordered_ids)
        or artifact.get("context_fingerprint") != _context_fingerprint(context)
        or artifact.get("family_hash") != _family_hash(context.family)
    ):
        raise ValueError("strict benchmark context identity differs from sealed design")
    root = Path(manifest_root).resolve() if manifest_root is not None else config_file.parent
    normalized_manifest = {
        str(Path(path).resolve()): digest for path, digest in input_manifest.items()
    }
    sources = artifact.get("source_inputs")
    if not isinstance(sources, list) or not sources:
        raise ValueError("strict benchmark context source manifest is missing")
    seen: set[str] = set()
    for source in sources:
        if not isinstance(source, Mapping) or set(source) != {"path", "sha256", "type"}:
            raise ValueError("strict benchmark context source manifest is invalid")
        source_path = Path(str(source["path"]))
        resolved = (
            source_path.resolve() if source_path.is_absolute()
            else (root / source_path).resolve()
        )
        key = str(resolved)
        if (
            key in seen
            or normalized_manifest.get(key) != source.get("sha256")
            or not isinstance(source.get("type"), str) or not source["type"]
        ):
            raise ValueError("strict benchmark context source is not manifest-bound")
        seen.add(key)
    expected_sources = {
        str(_resolved_path(interact["groups"], config_file.parent, "groups")),
        str(_resolved_path(interact["phenotype"], config_file.parent, "phenotype")),
    }
    for label in expected_subgenomes:
        prefix = _resolved_path(
            interact["genotype"][label], config_file.parent, f"genotype.{label}"
        )
        expected_sources.update(
            str(Path(str(prefix) + suffix).resolve())
            for suffix in (".bed", ".bim", ".fam")
        )
        expected_sources.add(str(_resolved_path(
            interact["snp_to_gene"][label], config_file.parent,
            f"snp_to_gene.{label}",
        )))
    if seen != expected_sources:
        raise ValueError("strict benchmark context source manifest is incomplete")
    return context


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


def _subgenome_kernel(context: OmniBBenchmarkContext, label: str) -> np.ndarray:
    values = np.asarray(context.subdata[label].X[context.sample_idx], dtype=float)
    means = np.nanmean(values, axis=0)
    values = np.where(np.isfinite(values), values, means)
    centered = values - values.mean(axis=0)
    scale = centered.std(axis=0, ddof=1)
    usable = np.isfinite(scale) & (scale > 0)
    standardized = centered[:, usable] / scale[usable]
    kernel = standardized @ standardized.T / standardized.shape[1]
    return 0.5 * (kernel + kernel.T)


def _independent_main_effect(context: OmniBBenchmarkContext) -> np.ndarray:
    label = context.family.subgenomes[0]
    values = np.asarray(context.subdata[label].X[context.sample_idx], dtype=float)
    means = np.nanmean(values, axis=0)
    values = np.where(np.isfinite(values), values, means)
    centered = values - values.mean(axis=0)
    scale = centered.std(axis=0, ddof=1)
    usable = np.isfinite(scale) & (scale > 0)
    left, singular, _ = np.linalg.svd(centered[:, usable] / scale[usable], full_matrices=False)
    return standardize(left[:, 0] * singular[0])


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
    null_model: str = "gaussian",
) -> _PreparedScenario:
    setup_seed, setup_seed_id = _seed(design_hash, scenario_id, 0, stage, "scenario_setup")
    scores, expanded = prepare_omnib_design(
        context.subdata,
        context.family,
        context.phenotype,
        context.sample_idx,
        transform="INT",
        feature_seed=context.feature_seed,
        retained_variant_masks=context.retained_variant_masks,
        grm_method="grm_from_X",
        maf_min=0.01,
        burden_maf=0.01,
        min_snp=3,
        cap=150,
        n_pc=3,
    )
    if scores.null_covariance is None:
        raise RuntimeError("production omniB preparation did not retain its null covariance")
    omitted_label = context.family.subgenomes[-1]
    if null_model in {"omitted_kernel", "omitted_background_kernel"}:
        omitted_component = float(scores.covariance_components.get(omitted_label, 0.0))
        omitted_fit_kernel = np.asarray(scores.null_kernels[omitted_label], dtype=float)
        fitted_covariance = np.asarray(scores.null_covariance, dtype=float)
        excluded_covariance = 0.5 * (
            fitted_covariance - omitted_component * omitted_fit_kernel
            + (fitted_covariance - omitted_component * omitted_fit_kernel).T
        )
        eigenvalues, eigenvectors = np.linalg.eigh(excluded_covariance)
        if float(eigenvalues.min()) <= 0.0:
            raise RuntimeError("omitted-kernel fitted covariance is not positive definite")
        whitener = (
            eigenvectors * (1.0 / np.sqrt(eigenvalues))
        ) @ eigenvectors.T
        retained_components = {
            key: value for key, value in scores.covariance_components.items()
            if key != omitted_label
        }
        retained_kernels = {
            key: value for key, value in scores.null_kernels.items()
            if key != omitted_label
        }
        null_fit_identity = {
            "W": _array_identity(whitener),
            "covariance": _array_identity(excluded_covariance),
            "beta": _array_identity(np.asarray(scores.null_beta, float)),
            "design": _array_identity(np.asarray(scores.null_design, float)),
            "kernels": {
                sub: _array_identity(retained_kernels[sub])
                for sub in sorted(retained_kernels)
            },
            "components": {
                str(key): float(value)
                for key, value in sorted(retained_components.items())
            },
            "benchmark_covariance_override": {
                "kind": "omitted_kernel",
                "omitted_subgenome": omitted_label,
            },
        }
        null_fit_sha256 = _text_identity(null_fit_identity)
        prepared_design_identity = {
            **dict(scores.prepared_design_identity),
            "null_fit_sha256": null_fit_sha256,
            "benchmark_covariance_override": {
                "kind": "omitted_kernel",
                "omitted_subgenome": omitted_label,
            },
        }
        scores = replace(
            scores,
            W=whitener,
            null_covariance=excluded_covariance,
            covariance_components=retained_components,
            null_kernels=retained_kernels,
            null_fit_identity=null_fit_identity,
            null_fit_sha256=null_fit_sha256,
            prepared_design_identity=prepared_design_identity,
            prepared_design_sha256=_text_identity(prepared_design_identity),
            projection_cache={},
        )
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
        genotype_main_effect=_independent_main_effect(context),
        omitted_kernel=_subgenome_kernel(context, omitted_label),
        omitted_subgenome=omitted_label,
        gene_blocks=blocks,
        family_hash=_family_hash(context.family),
        setup_seed_id=setup_seed_id,
        setup_seed=setup_seed,
    )


def _snpxsnp_pair_ceiling(context: OmniBBenchmarkContext) -> int:
    """Resolve the frozen real-panel ceiling before any SNP-pair scoring."""

    if context.panel_id in _SNPXSNP_FIXTURE_PANEL_IDS:
        # These identities are reserved for bounded unit-test fixtures.
        # Explicit benchmark identities, including SYNTH.QUARTET, remain
        # fail-closed through comparator_resource_limit.
        return _SYNTHETIC_FIXTURE_MAX_OFFERED_SNP_PAIRS
    limit = comparator_resource_limit(
        context.panel_id,
        family_size=len(context.family.group_ids),
        copies=len(context.family.subgenomes),
    )
    return limit.max_offered_pairs


def _poison_failed_omnib_method_columns(
    group: np.ndarray,
    component_scores: Sequence[np.ndarray],
    diagnostics: Any,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Make every recorded fixed-component failure explicit in score banks."""

    group = np.asarray(group, dtype=float).copy()
    failed = np.asarray(diagnostics.failed_response_mask, dtype=bool)
    if failed.shape != (group.shape[1],):
        raise RuntimeError("omniB response diagnostics are not response-aligned")
    group[:, failed] = np.nan
    components = [np.asarray(values, dtype=float).copy() for values in component_scores]
    by_component = diagnostics.failed_response_indices_by_component
    for method, values in zip(
        ("minor_burden", "pc1", "kernel_hadamard"), components, strict=True
    ):
        indices = np.asarray(by_component[method], dtype=int)
        if indices.size:
            values[:, indices] = np.nan
    return group, components


def _frozen_response_failure_mask(
    response_count: int, indices: Sequence[int], *, method: str,
) -> np.ndarray:
    """Build one immutable response-aligned method failure mask."""

    mask = np.zeros(response_count, dtype=bool)
    selected = np.asarray(indices, dtype=int)
    if selected.size:
        if (
            selected.ndim != 1
            or np.any(selected < 0)
            or np.any(selected >= response_count)
            or np.unique(selected).size != selected.size
        ):
            raise RuntimeError(f"invalid response failure indices for {method}")
        mask[selected] = True
    mask.setflags(write=False)
    return mask


def _omnib_method_failure_masks(
    diagnostics: Any,
    *,
    response_count: int,
    snpxsnp_result: SNPxSNPScoreResult | None = None,
) -> dict[str, np.ndarray]:
    """Freeze method-specific failures before another bank overwrites diagnostics."""

    if diagnostics is None or diagnostics.attempted != response_count:
        raise RuntimeError("omniB response failure diagnostics are missing")
    by_component = diagnostics.failed_response_indices_by_component
    masks = {
        "omnib": _frozen_response_failure_mask(
            response_count, diagnostics.failed_response_indices, method="omnib"
        ),
        "minor_burden": _frozen_response_failure_mask(
            response_count, by_component["minor_burden"], method="minor_burden"
        ),
        "pc1": _frozen_response_failure_mask(
            response_count, by_component["pc1"], method="pc1"
        ),
        "kernel_hadamard": _frozen_response_failure_mask(
            response_count,
            by_component["kernel_hadamard"],
            method="kernel_hadamard",
        ),
    }
    if snpxsnp_result is not None:
        masks["snpxsnp"] = _frozen_response_failure_mask(
            response_count,
            snpxsnp_result.failed_response_indices,
            method="snpxsnp",
        )
    return masks


def _response_failure_record(
    masks: Mapping[str, np.ndarray],
    *,
    response_count: int,
    response_role: str,
    failure_rate_max: float,
) -> dict[str, Any]:
    """Serialize fixed-denominator, method-specific response failures."""

    checked: dict[str, np.ndarray] = {}
    for method, raw in masks.items():
        mask = np.asarray(raw)
        if mask.dtype != np.bool_ or mask.shape != (response_count,):
            raise RuntimeError(f"response failure mask is not aligned for {method}")
        checked[method] = mask
    failed_by_method = {
        method: np.flatnonzero(mask).astype(int).tolist()
        for method, mask in checked.items()
    }
    counts_by_method = {
        method: len(indices) for method, indices in failed_by_method.items()
    }
    rates_by_method = {
        method: count / response_count
        for method, count in counts_by_method.items()
    }
    failed_union = np.zeros(response_count, dtype=bool)
    for mask in checked.values():
        failed_union |= mask
    union_indices = np.flatnonzero(failed_union).astype(int).tolist()
    union_rate = len(union_indices) / response_count
    return {
        "attempted": response_count,
        "failed_response_indices": union_indices,
        "failed_response_indices_by_method": failed_by_method,
        "terminal_failures": len(union_indices),
        "terminal_failures_by_method": counts_by_method,
        "terminal_failure_rate": union_rate,
        "terminal_failure_rate_by_method": rates_by_method,
        "failure_rate_max": failure_rate_max,
        "within_failure_ceiling": union_rate <= failure_rate_max,
        "within_failure_ceiling_by_method": {
            method: rate <= failure_rate_max
            for method, rate in rates_by_method.items()
        },
        "worst_case_mapping": (
            "failure_counts_as_rejection"
            if response_role in {"calibration", "heldout", "null"}
            else "failure_counts_as_non_detection"
        ),
    }


def _method_scores(
    prepared: _PreparedScenario,
    responses: np.ndarray,
    *,
    n_jobs: int,
) -> tuple[
    MethodScoreBank,
    dict[str, Any],
    SNPxSNPScoreResult,
    dict[str, np.ndarray],
]:
    _edge, group, components, diagnostics = score_omnib_responses(
        prepared.scores,
        prepared.context.family,
        prepared.expanded,
        responses,
        n_jobs=n_jobs,
        return_diagnostics=True,
    )
    response_execution = dict(prepared.scores.parallel_execution)
    component_scores = [
        group_component_p(components, prepared.expanded, component_index=index)
        for index in range(3)
    ]
    group, component_scores = _poison_failed_omnib_method_columns(
        group, component_scores, diagnostics
    )
    snpxsnp_result = score_snpxsnp_family(
        prepared.scores,
        prepared.context.family,
        prepared.expanded,
        prepared.gene_blocks,
        responses,
        max_offered_pairs=_snpxsnp_pair_ceiling(prepared.context),
    )
    group_count = len(prepared.context.family.group_ids)
    failure_masks = _omnib_method_failure_masks(
        diagnostics,
        response_count=responses.shape[1],
        snpxsnp_result=snpxsnp_result,
    )
    return (
        MethodScoreBank(
            family_ids=prepared.context.family.group_ids,
            p_by_method={
                "omnib": group,
                "minor_burden": component_scores[0],
                "pc1": component_scores[1],
                "kernel_hadamard": component_scores[2],
                "snpxsnp": snpxsnp_result.group_p,
            },
            tested_family_sizes={
                "omnib": group_count,
                "minor_burden": group_count,
                "pc1": group_count,
                "kernel_hadamard": group_count,
                "snpxsnp": snpxsnp_result.tested_pair_count,
            },
        ),
        response_execution,
        snpxsnp_result,
        failure_masks,
    )


def _local_component_scores(
    prepared: _PreparedScenario,
    responses: np.ndarray,
    *,
    n_jobs: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    _edge, group, components, diagnostics = score_omnib_responses(
        prepared.scores, prepared.context.family, prepared.expanded,
        responses, n_jobs=n_jobs, return_diagnostics=True,
    )
    component = [
        group_component_p(components, prepared.expanded, component_index=index)
        for index in range(3)
    ]
    group, component = _poison_failed_omnib_method_columns(
        group, component, diagnostics
    )
    return (
        {
            "omnib": group,
            "minor_burden": component[0],
            "pc1": component[1],
            "kernel_hadamard": component[2],
        },
        _omnib_method_failure_masks(
            diagnostics, response_count=responses.shape[1]
        ),
    )


def run_family_size_stress(
    context: OmniBBenchmarkContext,
    *,
    family_size: int,
    response_count: int,
    calibration_count: int,
    design_hash: str,
    qa_only: bool,
    n_jobs: int,
    scenario_id: str,
) -> dict[str, Any]:
    """Run the statistical family-size stress without conflating Track C."""

    started = time.perf_counter()
    if len(context.family.group_ids) != family_size:
        raise ValueError("family-size context does not match declared family_size")
    prepared = _prepare_scenario(
        context, design_hash=design_hash, stage=_stage(qa_only),
        scenario_id=scenario_id, n_jobs=n_jobs,
    )
    calibration, calibration_seeds, calibration_ids, _ = _draw_bank_responses(
        prepared, requested_role="calibration", canonical_role="calibration",
        count=calibration_count, design_hash=design_hash, stage=_stage(qa_only),
        scenario_id=scenario_id + ".calibration", replicate_offset=0,
        null_model="gaussian",
    )
    target, target_seeds, target_ids, _ = _draw_bank_responses(
        prepared, requested_role="heldout", canonical_role="heldout",
        count=response_count, design_hash=design_hash, stage=_stage(qa_only),
        scenario_id=scenario_id + ".heldout", replicate_offset=0,
        null_model="gaussian",
    )
    if set(calibration_ids) & set(target_ids):
        raise RuntimeError("family-size calibration and heldout banks overlap")
    if family_size <= 80:
        calibration_scores, _, calibration_snpxsnp, calibration_failure_masks = _method_scores(
            prepared, calibration, n_jobs=n_jobs,
        )
        target_scores, _, target_snpxsnp, target_failure_masks = _method_scores(
            prepared, target, n_jobs=n_jobs,
        )
        if (
            calibration_snpxsnp.member_ids != target_snpxsnp.member_ids
            or calibration_snpxsnp.member_family_sha256
            != target_snpxsnp.member_family_sha256
            or calibration_snpxsnp.input_family_sha256
            != target_snpxsnp.input_family_sha256
        ):
            raise RuntimeError("family-size SNPxSNP response banks differ")
        calibration_map = dict(calibration_scores.p_by_method)
        target_map = dict(target_scores.p_by_method)
        tested_sizes = dict(target_scores.tested_family_sizes)
        tested_members = {
            method: tuple(context.family.group_ids)
            for method in calibration_map
        }
        tested_members["snpxsnp"] = target_snpxsnp.member_ids
        snpxsnp_status = "applicable"
    else:
        calibration_map, calibration_failure_masks = _local_component_scores(
            prepared, calibration, n_jobs=n_jobs
        )
        target_map, target_failure_masks = _local_component_scores(
            prepared, target, n_jobs=n_jobs
        )
        tested_sizes = {method: family_size for method in calibration_map}
        tested_members = {
            method: tuple(context.family.group_ids) for method in calibration_map
        }
        snpxsnp_status = "not_applicable_above_80"
    tested_hashes = {
        method: sha256_payload({
            "method": method, "ordered_member_ids": list(members),
        })
        for method, members in tested_members.items()
    }
    if family_size <= 80:
        tested_hashes["snpxsnp"] = target_snpxsnp.member_family_sha256
    thresholds = {
        method: empirical_threshold(
            values, failed_mask=calibration_failure_masks[method]
        )
        for method, values in calibration_map.items()
    }
    rejection = {
        method: apply_failure_policy(
            apply_threshold(target_map[method], thresholds[method]),
            target_failure_masks[method],
            response_role="heldout",
        )
        for method in calibration_map
    }
    calibration_minima = {
        method: _json_safe(np.where(
            np.isfinite(values).any(axis=0),
            np.where(np.isfinite(values), values, np.inf).min(axis=0),
            np.nan,
        ).copy())
        for method, values in calibration_map.items()
    }
    for method, failed_mask in calibration_failure_masks.items():
        for index in np.flatnonzero(failed_mask):
            calibration_minima[method][int(index)] = 0.0
    target_minima = {
        method: _json_safe(np.where(
            np.isfinite(values).any(axis=0),
            np.where(np.isfinite(values), values, np.inf).min(axis=0),
            np.nan,
        ))
        for method, values in target_map.items()
    }
    rejections_by_method = {
        method: values.astype(bool).tolist() for method, values in rejection.items()
    }
    inference = (
        {"qa_cutoffs": thresholds}
        if qa_only else {"thresholds": thresholds}
    )
    failure_rate_max = _DIAGNOSTIC_FAILURE_RATE_MAX
    calibration_failures = _response_failure_record(
        calibration_failure_masks,
        response_count=calibration_count,
        response_role="calibration",
        failure_rate_max=failure_rate_max,
    )
    target_failures = _response_failure_record(
        target_failure_masks,
        response_count=response_count,
        response_role="heldout",
        failure_rate_max=failure_rate_max,
    )
    within_failure_ceiling = bool(
        calibration_failures["within_failure_ceiling"]
        and target_failures["within_failure_ceiling"]
    )
    return {
        "experiment": "family_size", "family_size": family_size,
        "group_count": family_size, "snpxsnp_status": snpxsnp_status,
        "methods": sorted(calibration_map),
        "tested_family_sizes": tested_sizes,
        "tested_family_members": {
            method: list(members) for method, members in tested_members.items()
        },
        "tested_family_hashes": tested_hashes,
        **(
            {"snpxsnp_evidence": target_snpxsnp.evidence_payload()}
            if family_size <= 80 else {}
        ),
        "score_hypothesis_units": {
            method: (
                "snp_pair_within_group" if method == "snpxsnp" else "group_score"
            )
            for method in calibration_map
        },
        "calibration_minima_by_method": calibration_minima,
        "target_minima_by_method": target_minima,
        "calibration_minima_hashes": {
            method: sha256_payload(values)
            for method, values in calibration_minima.items()
        },
        "target_minima_hashes": {
            method: sha256_payload(values)
            for method, values in target_minima.items()
        },
        "fwer": {method: float(np.mean(values))
                 for method, values in rejections_by_method.items()},
        "failure_rate": max(
            calibration_failures["terminal_failure_rate"],
            target_failures["terminal_failure_rate"],
        ),
        "calibration_response_failures": calibration_failures,
        "target_response_failures": target_failures,
        "runtime_seconds": float(time.perf_counter() - started),
        "calibration_count": calibration_count, "response_count": response_count,
        "calibration_response_ids": list(calibration_ids),
        "target_response_ids": list(target_ids),
        "calibration_seeds": list(calibration_seeds),
        "target_seeds": list(target_seeds),
        "calibration_response_hash": _array_hash(calibration),
        "target_response_hash": _array_hash(target),
        "feature_cache_sha256": prepared.scores.feature_cache_sha256,
        "fixed_mask_sha256": prepared.scores.fixed_mask_sha256,
        "null_fit_sha256": prepared.scores.null_fit_sha256,
        "prepared_design_sha256": prepared.scores.prepared_design_sha256,
        "inference_status": "noninferential_do_not_threshold" if qa_only else "formal",
        "failure": {
            "failed": not within_failure_ceiling,
            "status": (
                "failed_response_ceiling"
                if not within_failure_ceiling
                else "completed_with_worst_case_response_failures"
                if calibration_failures["terminal_failures"]
                or target_failures["terminal_failures"]
                else "completed"
            ),
        },
        "requested_jobs": n_jobs,
        "effective_jobs": int(prepared.scores.parallel_execution.get("effective_jobs", 1)),
        "parallel_backend": prepared.scores.parallel_execution.get("backend", "serial"),
        "worker_pids": list(prepared.scores.parallel_execution.get("worker_pids", [])),
        **inference,
        **({"qa_rejections_by_method": rejections_by_method}
           if qa_only else {"rejections_by_method": rejections_by_method}),
    }


def _tested_family_members(
    prepared: _PreparedScenario,
    snpxsnp_result: SNPxSNPScoreResult,
) -> dict[str, tuple[str, ...]]:
    local_members = tuple(
        f"{group_id}|edges="
        + ",".join(
            prepared.expanded.edges[edge_index].edge_id
            + f":estimable={int(prepared.scores.edge_estimable[edge_index])}"
            for edge_index in prepared.expanded.group_edge_indices[group_index]
        )
        for group_index, group_id in enumerate(prepared.context.family.group_ids)
    )
    members = {
        "omnib": local_members,
        "minor_burden": local_members,
        "pc1": local_members,
        "kernel_hadamard": local_members,
        "snpxsnp": snpxsnp_result.member_ids,
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
                genotype_main_effect=prepared.genotype_main_effect,
                omitted_kernel=prepared.omitted_kernel,
                omitted_subgenome=prepared.omitted_subgenome,
                omitted_variance=0.25,
            )
        else:
            response, row_metadata = response_factory(rng, replicate)
        if row_metadata.get("canonical_kind") == "omitted_kernel":
            row_metadata = {
                **row_metadata,
                "fitted_null_covariance_sha256": _array_hash(
                    prepared.scores.null_covariance
                ),
                "fitted_null_components": sorted(
                    prepared.scores.covariance_components
                ),
            }
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
        score_bank, response_execution, snpxsnp_result, failure_masks = _method_scores(
            prepared,
            responses,
            n_jobs=n_jobs,
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
        method: np.flatnonzero(mask).astype(int).tolist()
        for method, mask in failure_masks.items()
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
    failure_rate_max = (
        _BINDING_GAUSSIAN_FAILURE_RATE_MAX
        if null_model == "gaussian" and canonical_role in {"calibration", "heldout"}
        else _DIAGNOSTIC_FAILURE_RATE_MAX
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
            "attempted": count,
            "successful": count - len(failed_indices),
            "retried": 0,
            "terminal_failures": len(failed_indices),
            "terminal_failure_rate": len(failed_indices) / count,
            "failure_rate_max": failure_rate_max,
            "within_failure_ceiling": len(failed_indices) / count <= failure_rate_max,
            "worst_case_mapping": (
                "failure_counts_as_rejection"
                if canonical_role in {"calibration", "heldout"}
                else "failure_counts_as_non_detection"
            ),
        }
    )
    members = _tested_family_members(prepared, snpxsnp_result)
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
    tested_hashes["snpxsnp"] = snpxsnp_result.member_family_sha256
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
        panel_id=prepared.context.panel_id,
        sample_context=prepared.context.sample_context,
        feature_seed=prepared.context.feature_seed,
        marker_mask_identity=prepared.context.marker_mask_identity,
        marker_mask_sha256=prepared.context.marker_mask_sha256,
        feature_cache_sha256=prepared.scores.feature_cache_sha256,
        fixed_mask_sha256=prepared.scores.fixed_mask_sha256,
        null_fit_sha256=prepared.scores.null_fit_sha256,
        prepared_design_sha256=prepared.scores.prepared_design_sha256,
        score_bank=score_bank,
        tested_family_members=members,
        tested_family_hashes=tested_hashes,
        score_matrix_hashes={
            method: sha256_payload(_json_safe(values))
            for method, values in score_bank.p_by_method.items()
        },
        snpxsnp_evidence=snpxsnp_result.evidence_payload(),
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
        null_model=null_model,
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


def _global_grms(prepared: _PreparedScenario) -> dict[str, np.ndarray]:
    return {
        label: _subgenome_kernel(prepared.context, label)
        for label in prepared.context.family.subgenomes
    }


def run_global_vc_bank(
    context: OmniBBenchmarkContext,
    *,
    bank: str,
    count: int,
    design_hash: str,
    qa_only: bool,
    scenario_id: str,
    replicate: int = 0,
    calibration_count: int | None = None,
) -> dict[str, Any]:
    """Run the standalone global K_hom VC comparator on frozen null banks."""

    started = time.perf_counter()
    count = _validate_count(count)
    role = _canonical_role(bank)
    if role not in {"calibration", "heldout", "power"}:
        raise ValueError("global VC bank must be calibration, heldout or power")
    frozen_calibration_count = count if calibration_count is None else _validate_count(
        calibration_count, "calibration_count"
    )
    if role != "power":
        _require_formal_budget(count, qa_only, "global VC bank")
    else:
        _require_formal_budget(
            frozen_calibration_count, qa_only, "global VC calibration bank"
        )
    stage = _stage(qa_only)
    prepared = _prepare_scenario(
        context, design_hash=design_hash, stage=stage,
        scenario_id=scenario_id, n_jobs=1,
    )
    response_factory = None
    if role == "power":
        signal, causal_ids, signal_metadata = _group_signal(
            prepared, "kernel_multidimensional", 1
        )

        def response_factory(
            rng: np.random.Generator, _replicate: int,
        ) -> tuple[np.ndarray, dict[str, Any]]:
            residual, residual_metadata = draw_null(
                "gaussian", prepared.root_v, rng, prepared.pc1,
                genotype_main_effect=prepared.genotype_main_effect,
                omitted_kernel=prepared.omitted_kernel,
                omitted_subgenome=prepared.omitted_subgenome,
                omitted_variance=0.25,
            )
            phenotype, pve_metadata = compose_exact_pve(signal, residual, 0.05)
            return phenotype, {
                "null": residual_metadata, "pve": pve_metadata,
                "architecture": "kernel_multidimensional",
                "causal_group_count": len(causal_ids),
                "signal_hash": _array_hash(signal),
                "signal_metadata": signal_metadata,
            }

    target, target_seeds, target_ids, target_meta = _draw_bank_responses(
        prepared, requested_role=role, canonical_role=role, count=count,
        design_hash=design_hash, stage=stage, scenario_id=scenario_id,
        replicate_offset=0, null_model="gaussian", response_factory=response_factory,
    )
    calibration_scenario_id = _calibration_scenario_id(scenario_id)
    if role == "calibration":
        calibration = target
        calibration_seeds = target_seeds
        calibration_ids = target_ids
        calibration_meta = target_meta
    else:
        calibration, calibration_seeds, calibration_ids, calibration_meta = _draw_bank_responses(
            prepared, requested_role="calibration", canonical_role="calibration",
            count=frozen_calibration_count, design_hash=design_hash, stage=stage,
            scenario_id=calibration_scenario_id, replicate_offset=0,
            null_model="gaussian",
        )
        if set(calibration_ids) & set(target_ids):
            raise RuntimeError("global VC calibration and heldout seed IDs overlap")
    grms = _global_grms(prepared)
    calibration_result = score_global_hadamard_vc(
        calibration, np.ones((calibration.shape[0], 1)), grms,
        fit_kwargs={"n_starts": 1},
    )
    target_result = (
        calibration_result if role == "calibration" else score_global_hadamard_vc(
            target, np.ones((target.shape[0], 1)), grms,
            fit_kwargs={"n_starts": 1},
        )
    )
    calibration_p = np.asarray(calibration_result["p_values"], dtype=float)
    target_p = np.asarray(target_result["p_values"], dtype=float)
    calibration_failed = sorted(set(
        int(index) for index in calibration_result.get("failed_response_indices", [])
    ))
    target_failed = sorted(set(
        int(index) for index in target_result.get("failed_response_indices", [])
    ))
    if (
        calibration_failed != np.flatnonzero(~np.isfinite(calibration_p)).tolist()
        or target_failed != np.flatnonzero(~np.isfinite(target_p)).tolist()
    ):
        raise RuntimeError("global VC failure indices differ from non-finite p-values")
    usable_calibration = calibration_p[np.isfinite(calibration_p)]
    threshold = (
        empirical_threshold(usable_calibration[None, :])
        if usable_calibration.size else None
    )
    evidence: dict[str, Any]
    if qa_only:
        evidence = {
            "qa_calibration_cutoff": threshold,
            "qa_detection_flags": apply_threshold(target_p[None, :], threshold).tolist(),
        }
    else:
        evidence = {
            "threshold": threshold,
            "rejected": apply_threshold(target_p[None, :], threshold).tolist(),
        }
    payload = _base_replicate_payload(
        context, replicate=replicate, design_hash=design_hash, stage=stage,
        scenario_id=scenario_id, n_jobs=1,
    )
    payload.update({
        "experiment": "global_vc",
        "scenario_id": scenario_id,
        "stage": stage,
        "method": GLOBAL_VC_METHOD,
        "hypothesis_unit": "global",
        "detection_only": True,
        "response_count": count,
        "target_role": role,
        "calibration_scenario_id": calibration_scenario_id,
        "calibration_response_ids": list(calibration_ids),
        "target_response_ids": list(target_ids),
        "calibration_seeds": list(calibration_seeds),
        "target_seeds": list(target_seeds),
        "calibration_response_metadata": list(calibration_meta),
        "target_response_metadata": list(target_meta),
        "calibration_response_hash": _array_hash(calibration),
        "target_response_hash": _array_hash(target),
        "calibration_p_values": _json_safe(calibration_p),
        "target_p_values": _json_safe(target_p),
        "calibration_p_hash": sha256_payload(_json_safe(calibration_p)),
        "target_p_hash": sha256_payload(_json_safe(target_p)),
        "feature_cache_sha256": prepared.scores.feature_cache_sha256,
        "fixed_mask_sha256": prepared.scores.fixed_mask_sha256,
        "null_fit_sha256": prepared.scores.null_fit_sha256,
        "prepared_design_sha256": prepared.scores.prepared_design_sha256,
        "failed_calibration_response_indices": calibration_failed,
        "failed_target_response_indices": target_failed,
        "calibration_lrt_evidence": calibration_result["lrt_evidence"],
        "target_lrt_evidence": target_result["lrt_evidence"],
        "kernel_manifest": target_result["kernel_manifest"],
        "inference_status": "noninferential_do_not_threshold" if qa_only else "formal",
        "failure": {
            "failed": bool(calibration_failed or target_failed),
            "status": ("partial_failure" if calibration_failed or target_failed
                       else "completed"),
            "failed_calibration_response_indices": calibration_failed,
            "failed_target_response_indices": target_failed,
        },
        "requested_jobs": 1,
        "effective_jobs": 1,
        "parallel_backend": "serial",
        "worker_pids": [os.getpid()],
        "runtime_seconds": float(time.perf_counter() - started),
        **evidence,
    })
    if role == "power":
        flags = payload.get("qa_detection_flags", payload.get("rejected", []))
        payload["detection_power"] = float(np.mean(np.asarray(flags, dtype=bool)))
        payload["positive_signal"] = {
            "architecture": "kernel_multidimensional",
            "interaction_pve": 0.05,
            "causal_groups": 1,
        }
    return _json_safe(payload)


def empirical_threshold(
    p_null: np.ndarray,
    alpha: float = 0.05,
    *,
    failed_mask: np.ndarray | None = None,
) -> float | None:
    """Learn one strict experiment-wide min-P threshold from calibration only."""

    values = np.asarray(p_null, dtype=float)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 1:
        raise ValueError("p_null must be a non-empty hypothesis-by-response matrix")
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must be strictly between zero and one")
    finite = np.isfinite(values)
    if failed_mask is None:
        failed = np.zeros(values.shape[1], bool)
    else:
        failed = np.asarray(failed_mask)
        if failed.dtype != np.bool_ or failed.shape != (values.shape[1],):
            raise ValueError("failed_mask must be a response-aligned boolean vector")
    all_nan_columns = np.flatnonzero(
        ~finite.any(axis=0) & ~failed
    ).astype(int).tolist()
    if all_nan_columns:
        raise ValueError(
            "calibration contains all-NaN response columns: "
            + ",".join(str(index) for index in all_nan_columns)
        )
    family_min = np.where(finite, values, np.inf).min(axis=0)
    family_min[failed] = 0.0
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


def apply_failure_policy(
    decisions: np.ndarray,
    failed_mask: np.ndarray,
    *,
    response_role: str,
) -> np.ndarray:
    """Map terminal response failures to the predeclared conservative outcome."""
    decisions = np.asarray(decisions)
    failed_mask = np.asarray(failed_mask)
    if decisions.dtype != np.bool_ or failed_mask.dtype != np.bool_:
        raise ValueError("decisions and failed_mask must have boolean dtype")
    if decisions.ndim != 1 or failed_mask.shape != decisions.shape:
        raise ValueError("decisions and failed_mask must be aligned vectors")
    output = decisions.copy()
    if response_role in {"calibration", "heldout", "null"}:
        output[failed_mask] = True
    elif response_role == "power":
        output[failed_mask] = False
    else:
        raise ValueError("response_role must be calibration, heldout/null, or power")
    return output


def _method_failure_mask(bank: ConditionalBank, method: str) -> np.ndarray:
    by_method = bank.failure.get("nonfinite_response_indices_by_method")
    if not isinstance(by_method, Mapping) or method not in by_method:
        raise RuntimeError(f"conditional bank lacks failure indices for {method}")
    mask = np.zeros(len(bank.seed_ids), dtype=bool)
    indices = np.asarray(by_method[method], dtype=int)
    if indices.size:
        if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= mask.size):
            raise RuntimeError(f"conditional bank has invalid failure indices for {method}")
        mask[indices] = True
    return mask


def _bank_failure_envelope(failure: Mapping[str, Any]) -> dict[str, Any]:
    """Translate response diagnostics into outer-shard failure semantics."""
    output = dict(failure)
    response_failures = bool(failure.get("failed"))
    contract_failed = failure.get("within_failure_ceiling") is not True
    output.update(
        {
            "failed": contract_failed,
            "response_failures_present": response_failures,
            "status": (
                "failed_response_ceiling"
                if contract_failed
                else (
                    "completed_with_worst_case_response_failures"
                    if response_failures
                    else "completed"
                )
            ),
        }
    )
    return output


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
        "panel_id": context.panel_id,
        "sample_context": context.sample_context,
        "feature_seed": context.feature_seed,
        "marker_mask_sha256": context.marker_mask_sha256,
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
    prepared = _prepare_scenario(
        context, design_hash=design_hash, stage=stage,
        scenario_id=scenario_id, n_jobs=n_jobs, null_model=null_model,
    )
    phenotype, null_metadata = draw_null(
        null_model, prepared.root_v, np.random.default_rng(observed_seed),
        prepared.pc1, genotype_main_effect=prepared.genotype_main_effect,
        omitted_kernel=prepared.omitted_kernel,
        omitted_subgenome=prepared.omitted_subgenome, omitted_variance=0.25,
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
            "inference_status": (
                "noninferential_do_not_threshold" if qa_only else "formal"
            ),
        }
    )
    started = time.perf_counter()
    try:
        if null_model in {"omitted_kernel", "omitted_background_kernel"}:
            calibration_rng = np.random.default_rng(bootstrap_seed)
            bootstrap = np.column_stack([
                draw_null(
                    "gaussian", prepared.root_v, calibration_rng, prepared.pc1,
                    genotype_main_effect=prepared.genotype_main_effect,
                )[0]
                for _ in range(bootstrap_B)
            ])
            edge_p, group_p, components, diagnostics = score_omnib_responses(
                prepared.scores, context.family, prepared.expanded,
                np.column_stack([phenotype, bootstrap]), n_jobs=n_jobs,
                return_diagnostics=True,
            )
            if diagnostics.failed_response_mask[0]:
                raise RuntimeError("native observed response failed fixed-mask scoring")
            failed = np.asarray(diagnostics.failed_response_mask, dtype=bool)
            edge_p[:, failed] = np.nan
            group_p[:, failed] = np.nan
            scores = replace(
                prepared.scores, edge_p=edge_p, group_p=group_p,
                edge_components_obs=components[:, :, 0], y=phenotype,
                response_diagnostics=diagnostics,
            )
            expanded = prepared.expanded
        else:
            scores, expanded = score_omnib_family(
                context.subdata,
                context.family,
                phenotype,
                context.sample_idx,
                transform="INT",
                bootstrap_B=bootstrap_B,
                bootstrap_seed=bootstrap_seed,
                feature_seed=context.feature_seed,
                retained_variant_masks=context.retained_variant_masks,
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
        diagnostics = scores.response_diagnostics
        if diagnostics is None or diagnostics.attempted != bootstrap_B + 1:
            raise RuntimeError("native response failure diagnostics are missing")
        if diagnostics.failed_response_mask[0]:
            raise RuntimeError("native observed response failed fixed-mask scoring")
        bootstrap_failed = [
            index - 1
            for index in diagnostics.failed_response_indices
            if index > 0
        ]
        bootstrap_failure_rate = len(bootstrap_failed) / bootstrap_B
        bootstrap_failure_rate_max = (
            _BINDING_GAUSSIAN_FAILURE_RATE_MAX
            if null_model == "gaussian"
            else _DIAGNOSTIC_FAILURE_RATE_MAX
        )
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
        payload[
            "qa_diagnostic_rejections" if qa_only else "formal_rejections"
        ] = qa_rejections
        execution = dict(scores.parallel_execution)
        payload.update(
            {
                "observed_group_p": _json_safe(scores.group_p[:, 0]),
                "adjusted_p": _json_safe(adjusted),
                (
                    "qa_adjusted_diagnostic_decisions"
                    if qa_only else "adjusted_decisions"
                ): (adjusted <= 0.05).tolist(),
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
                "feature_cache_sha256": scores.feature_cache_sha256,
                "fixed_mask_sha256": scores.fixed_mask_sha256,
                "null_fit_sha256": scores.null_fit_sha256,
                "prepared_design_sha256": scores.prepared_design_sha256,
                "effective_jobs": execution.get("effective_jobs", 1),
                "parallel_backend": execution.get("backend", "serial"),
                "worker_pids": list(execution.get("worker_pids", [])),
                "parallel_execution": execution,
                "failure": {
                    "failed": False,
                    "status": (
                        "completed_with_degenerate_bootstrap"
                        if bootstrap_failed else "completed"
                    ),
                    "error_type": None,
                    "message": None,
                    "observed_failed": False,
                    "response_diagnostics": diagnostics.as_dict(),
                    "bootstrap_attempted": bootstrap_B,
                    "bootstrap_successful": bootstrap_B - len(bootstrap_failed),
                    "bootstrap_retried": 0,
                    "bootstrap_terminal_failures": len(bootstrap_failed),
                    "bootstrap_terminal_failure_rate": bootstrap_failure_rate,
                    "bootstrap_failure_rate_max": bootstrap_failure_rate_max,
                    "bootstrap_within_failure_ceiling": (
                        bootstrap_failure_rate <= bootstrap_failure_rate_max
                    ),
                    "bootstrap_failed_response_indices": bootstrap_failed,
                    "bootstrap_degenerate_policy": (
                        "any_nonfinite_statistic_sets_null_min_to_zero"
                    ),
                },
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
    causal_ids = (
        [] if architecture in {"additive_only", "mispaired"}
        else list(family.group_ids[:causal_groups])
    )
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
    identity_fields = (
        "panel_id",
        "sample_context",
        "feature_seed",
        "marker_mask_sha256",
        "feature_cache_sha256",
        "fixed_mask_sha256",
        "null_fit_sha256",
        "prepared_design_sha256",
    )
    mismatched = [
        field
        for field in identity_fields
        if getattr(left, field) != getattr(right, field)
    ]
    if mismatched:
        raise RuntimeError(
            "response banks do not share prepared identity: "
            + ", ".join(mismatched)
        )


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
        or calibration_bank.panel_id != context.panel_id
        or calibration_bank.sample_context != context.sample_context
        or calibration_bank.feature_seed != context.feature_seed
        or calibration_bank.marker_mask_sha256 != context.marker_mask_sha256
        or calibration_bank.failure.get("within_failure_ceiling") is not True
        or any(
            metadata.get("canonical_kind") != null_model
            for metadata in calibration_bank.response_metadata
        )
        or not calibration_scenario_id.endswith(".gaussian.calibration")
        or null_model != "gaussian"
    ):
        raise ValueError("power calibration bank differs from the frozen Gaussian bank")
    calibration_artifact = calibration_bank.statistical_artifact()
    calibration_manifest_hash = sha256_payload(calibration_artifact)
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
        null_model=null_model,
    )
    calibration = calibration_bank
    signal, causal_ids, signal_metadata = _group_signal(
        prepared, architecture, causal_groups
    )

    def causal_response(
        rng: np.random.Generator, _replicate: int
    ) -> tuple[np.ndarray, dict[str, Any]]:
        residual, residual_metadata = draw_null(
            null_model, prepared.root_v, rng, prepared.pc1,
            genotype_main_effect=prepared.genotype_main_effect,
            omitted_kernel=prepared.omitted_kernel,
            omitted_subgenome=prepared.omitted_subgenome,
            omitted_variance=0.25,
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
            thresholds[method] = empirical_threshold(
                calibration.p_by_method[method],
                failed_mask=_method_failure_mask(calibration, method),
            )
        except ValueError as error:
            thresholds[method] = None
            threshold_failures[method] = str(error)
    negative_control = architecture in {"additive_only", "mispaired"}
    target_failure_role = "heldout" if negative_control else "power"
    rejections = {
        method: apply_failure_policy(
            apply_threshold(target.p_by_method[method], thresholds[method]),
            _method_failure_mask(target, method),
            response_role=target_failure_role,
        ).tolist()
        for method in METHOD_NAMES
    }
    causal_index = np.array(
        [context.family.group_ids.index(group_id) for group_id in causal_ids], dtype=int
    )
    causal_detection = {} if negative_control else {
        method: apply_failure_policy(
            (
                np.nanmin(target.p_by_method[method][causal_index], axis=0)
                < thresholds[method]
            )
            if thresholds[method] is not None
            else np.zeros(response_count, bool),
            _method_failure_mask(target, method),
            response_role="power",
        ).tolist()
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
    causal_minima = {} if negative_control else {
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
    recall = {} if negative_control else {
        method: (
            np.where(
                _method_failure_mask(target, method),
                0.0,
                np.mean(causal_minima[method] < thresholds[method], axis=0),
            ).tolist()
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
            "control_type": "negative" if negative_control else "positive",
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
            "calibration_artifact": calibration_artifact,
            "target_response_hash": target.response_hash,
            "feature_cache_sha256": target.feature_cache_sha256,
            "fixed_mask_sha256": target.fixed_mask_sha256,
            "null_fit_sha256": target.null_fit_sha256,
            "prepared_design_sha256": target.prepared_design_sha256,
            "threshold_source": "independent_calibration_bank",
            "calibration_minima_by_method": calibration_minima,
            "target_minima_by_method": target_minima,
            "calibration_minima_hashes": calibration_minima_hashes,
            "target_minima_hashes": target_minima_hashes,
            "calibration_bank": calibration.to_payload(include_scores=False),
            "target_bank": target.to_payload(include_scores=True),
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
                    calibration.failure.get("within_failure_ceiling") is not True
                    or target.failure.get("within_failure_ceiling") is not True
                    or threshold_failures
                ),
                "response_failures_present": bool(
                    calibration.failure.get("failed")
                    or target.failure.get("failed")
                ),
                "status": (
                    "failed_response_ceiling_or_threshold"
                    if calibration.failure.get("within_failure_ceiling") is not True
                    or target.failure.get("within_failure_ceiling") is not True
                    or threshold_failures
                    else (
                        "completed_with_worst_case_response_failures"
                        if calibration.failure.get("failed")
                        or target.failure.get("failed")
                        else "completed"
                    )
                ),
                "threshold_failures": threshold_failures,
                "calibration": dict(calibration.failure),
                "target": dict(target.failure),
            },
            "inference_status": (
                "noninferential_do_not_threshold" if qa_only else "formal"
            ),
            "runtime_seconds": float(time.perf_counter() - started),
            **(
                {
                    "qa_cutoffs_by_method": thresholds,
                    "qa_rejections_by_method": rejections,
                }
                if qa_only else {
                    "thresholds": thresholds,
                    "rejections_by_method": rejections,
                }
            ),
        }
    )
    if negative_control:
        false_positive = {
            method: np.asarray(values, dtype=bool).tolist()
            for method, values in rejections.items()
        }
        specificity = {
            method: [not value for value in false_positive[method]]
            for method in METHOD_NAMES
        }
        payload.update(
            {
                "qa_false_positive_by_method": false_positive,
                "qa_specificity_by_method": specificity,
            }
            if qa_only else {
                "false_positive_by_method": false_positive,
                "specificity_by_method": specificity,
            }
        )
    else:
        payload.update({
            "causal_minima_by_method": causal_minima,
            "causal_minima_hashes": causal_minima_hashes,
            **(
                {
                    "qa_causal_detection_by_method": causal_detection,
                    "qa_recall_by_method": recall,
                }
                if qa_only else {
                    "causal_detection_by_method": causal_detection,
                    "recall_by_method": recall,
                }
            ),
        })
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
        feature_seed=context.feature_seed,
        retained_variant_masks=context.retained_variant_masks,
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
        "feature_cache_sha256": scores.feature_cache_sha256,
        "fixed_mask_sha256": scores.fixed_mask_sha256,
        "null_fit_sha256": scores.null_fit_sha256,
        "prepared_design_sha256": scores.prepared_design_sha256,
    }


def _copy_context(
    context: OmniBBenchmarkContext,
    *,
    subdata: dict[str, SubgenomeData] | None = None,
    family: MasterGroupFamily | None = None,
    phenotype: np.ndarray | None = None,
) -> OmniBBenchmarkContext:
    changed_subdata = subdata is not None
    return OmniBBenchmarkContext(
        subdata=context.subdata if subdata is None else subdata,
        family=context.family if family is None else family,
        phenotype=context.phenotype if phenotype is None else phenotype,
        sample_idx=context.sample_idx,
        panel_id=context.panel_id,
        sample_context=context.sample_context,
        feature_seed=context.feature_seed,
        retained_variant_masks=(
            None if changed_subdata else context.retained_variant_masks
        ),
        marker_mask_identity=(
            None if changed_subdata else context.marker_mask_identity
        ),
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
        "status": "completed",
        "required": required,
        "observed_arrays_identical": bool(observed_identical),
        "adjusted_decisions_identical": bool(decisions_identical),
        "ranking_hash_identical": baseline["ranking_hash"] == candidate["ranking_hash"],
        "rejection_sets_identical": set(baseline["rejections"]) == set(candidate["rejections"]),
        "baseline_observed_hash": _array_hash(baseline["observed"]),
        "candidate_observed_hash": _array_hash(candidate["observed"]),
        "requested_jobs": None,
        "effective_jobs": None,
        "backend": None,
        "worker_pids": [],
        "skip_reason": None,
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


_ROBUSTNESS_METHODS = ("omnib", "minor_burden", "pc1", "kernel_hadamard")


def _robustness_score_bank(
    prepared: _PreparedScenario, responses: np.ndarray, *, n_jobs: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    scores, failure_masks = _local_component_scores(
        prepared, responses, n_jobs=n_jobs
    )
    return (
        {
            method: np.asarray(scores[method], dtype=float)
            for method in _ROBUSTNESS_METHODS
        },
        {method: failure_masks[method] for method in _ROBUSTNESS_METHODS},
    )


def _response_minima(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    return np.where(
        finite.any(axis=0), np.where(finite, values, np.inf).min(axis=0), np.nan
    )


def _ranking_metrics(
    baseline: np.ndarray, candidate: np.ndarray, group_ids: Sequence[str],
) -> tuple[list[float | None], list[float]]:
    correlations: list[float | None] = []
    overlaps: list[float] = []
    k = min(10, len(group_ids))
    for column in range(baseline.shape[1]):
        correlations.append(_rank_correlation(baseline[:, column], candidate[:, column]))
        left = set(np.argsort(np.where(
            np.isfinite(baseline[:, column]), baseline[:, column], np.inf
        ))[:k].tolist())
        right = set(np.argsort(np.where(
            np.isfinite(candidate[:, column]), candidate[:, column], np.inf
        ))[:k].tolist())
        union = left | right
        overlaps.append(len(left & right) / len(union) if union else 1.0)
    return correlations, overlaps


def _basic_robustness_contexts(
    context: OmniBBenchmarkContext, seed: int, *, with_metadata: bool = False,
) -> Any:
    rng = np.random.default_rng(seed)
    outputs: dict[str, OmniBBenchmarkContext] = {}
    marker_designs: dict[str, Mapping[str, Any] | None] = {}
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
            source = np.asarray(context.subdata[label].X, dtype=float)
            with np.errstate(invalid="ignore"):
                frequencies = np.nanmean(source[context.sample_idx], axis=0) / 2.0
            distance = np.abs(frequencies - allele_frequency)
            candidate_order = np.argsort(
                np.where(np.isfinite(distance), distance, np.inf), kind="stable"
            )
            blocks: list[np.ndarray] = []
            for gene in (row[copy_index] for row in context.family.genes):
                native = np.asarray(
                    context.subdata[label].gene_snp[gene], dtype=int
                )
                pool = np.unique(np.concatenate([native, candidate_order]))
                marker_rng.shuffle(pool)
                selected = np.resize(pool, per_group)
                blocks.append(source[:, selected])
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

    def maf_context(
        challenge_name: str, *, lower: float, upper: float,
        lower_inclusive: bool, upper_inclusive: bool,
    ) -> OmniBBenchmarkContext:
        generated: dict[str, SubgenomeData] = {}
        selected_ids: list[str] = []
        selected_maf: list[float] = []
        by_gene: dict[str, Any] = {}
        for copy_index, label in enumerate(context.family.subgenomes):
            source = np.asarray(context.subdata[label].X, dtype=float)
            blocks: list[np.ndarray] = []
            mapping: dict[str, np.ndarray] = {}
            offset = 0
            for gene in (row[copy_index] for row in context.family.genes):
                native = np.asarray(context.subdata[label].gene_snp[gene], dtype=int)
                with np.errstate(invalid="ignore"):
                    dosage_frequency = np.nanmean(
                        source[np.ix_(context.sample_idx, native)], axis=0
                    ) / 2.0
                minor_maf = np.minimum(dosage_frequency, 1.0 - dosage_frequency)
                in_band = np.isfinite(minor_maf) & (
                    minor_maf >= lower if lower_inclusive else minor_maf > lower
                )
                in_band &= minor_maf <= upper if upper_inclusive else minor_maf < upper
                eligible = np.unique(native[np.flatnonzero(in_band)])
                if eligible.size < 5:
                    raise ValueError(
                        f"{challenge_name} has fewer than five native markers in band "
                        f"for {label}:{gene}"
                    )
                chosen = np.sort(eligible, kind="stable")[:5]
                chosen_maf = np.asarray([
                    minor_maf[int(np.flatnonzero(native == marker)[0])]
                    for marker in chosen
                ], dtype=float)
                blocks.append(source[:, chosen])
                mapping[gene] = np.arange(offset, offset + chosen.size)
                offset += chosen.size
                ids = [f"{label}:{int(marker)}" for marker in chosen]
                selected_ids.extend(ids)
                selected_maf.extend(chosen_maf.tolist())
                by_gene[f"{label}:{gene}"] = {
                    "source_marker_ids": ids,
                    "source_marker_indices": chosen.astype(int).tolist(),
                    "count": int(chosen.size),
                    "realized_maf_min": float(chosen_maf.min()),
                    "realized_maf_max": float(chosen_maf.max()),
                }
            generated[label] = SubgenomeData(
                X=np.column_stack(blocks), gene_snp=mapping,
                samples=list(context.subdata[label].samples), chunk=None,
            )
        marker_designs[challenge_name] = {
            "band": {
                "lower": lower, "lower_inclusive": lower_inclusive,
                "upper": upper, "upper_inclusive": upper_inclusive,
            },
            "selected_marker_ids": selected_ids,
            "selected_marker_ids_hash": sha256_payload(selected_ids),
            "selected_marker_count": len(selected_ids),
            "realized_maf_min": float(min(selected_maf)),
            "realized_maf_max": float(max(selected_maf)),
            "by_gene": by_gene,
        }
        return _copy_context(context, subdata=generated)

    copies = len(context.family.subgenomes)
    for count in (3, 10, 30):
        outputs[f"markers_per_gene_{count}"] = marker_context(
            [count] * copies, 0.25, seed + count
        )
    maf_specs = {
        "maf_0p01_0p05": (0.01, True, 0.05, False),
        "maf_0p05_0p20": (0.05, True, 0.20, True),
        "maf_above_0p20": (0.20, False, 0.50, True),
    }
    for name, (lower, lower_inclusive, upper, upper_inclusive) in maf_specs.items():
        try:
            outputs[name] = maf_context(
                name, lower=lower, lower_inclusive=lower_inclusive, upper=upper,
                upper_inclusive=upper_inclusive,
            )
        except ValueError as error:
            outputs[name] = error  # type: ignore[assignment]
            marker_designs[name] = None
    imbalanced = tuple((3, 10, 30, 5)[:copies])
    outputs["unbalanced_marker_counts"] = marker_context(imbalanced, 0.25, seed + 104)
    return (outputs, marker_designs) if with_metadata else outputs


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
        response_count = 20 if qa_only else 500
        calibration_count = bootstrap_B
        architectures = [
            "minor_burden_aligned", "pc1_distributed", "kernel_multidimensional",
            "single_snp_pair", "mixed_sign",
        ]
        if len(context.family.subgenomes) > 2:
            architectures.append("multi_edge_group")
        canonical_prepared = _prepare_scenario(
            context, design_hash=design_hash, stage=stage,
            scenario_id=scenario_id + ".robustness", n_jobs=n_jobs,
        )
        calibration_responses, calibration_seeds, calibration_ids, _ = (
            _draw_bank_responses(
                canonical_prepared, requested_role="calibration",
                canonical_role="calibration", count=calibration_count,
                design_hash=design_hash, stage=stage,
                scenario_id=scenario_id + ".robustness.calibration",
                replicate_offset=0, null_model="gaussian",
            )
        )
        heldout_responses, heldout_seeds, heldout_ids, _ = _draw_bank_responses(
            canonical_prepared, requested_role="heldout", canonical_role="heldout",
            count=response_count, design_hash=design_hash, stage=stage,
            scenario_id=scenario_id + ".robustness.heldout", replicate_offset=0,
            null_model="gaussian",
        )
        power_banks: dict[str, tuple[np.ndarray, tuple[int, ...], tuple[str, ...], list[str]]] = {}
        for architecture in architectures:
            signal, causal_ids, _signal_metadata = _group_signal(
                canonical_prepared, architecture, 1
            )

            def response_factory(
                rng: np.random.Generator, _response_index: int,
                *, frozen_signal: np.ndarray = signal,
                frozen_architecture: str = architecture,
                frozen_causal_ids: list[str] = causal_ids,
            ) -> tuple[np.ndarray, dict[str, Any]]:
                residual, residual_metadata = draw_null(
                    "gaussian", canonical_prepared.root_v, rng,
                    canonical_prepared.pc1,
                    genotype_main_effect=canonical_prepared.genotype_main_effect,
                )
                phenotype, pve = compose_exact_pve(frozen_signal, residual, 0.05)
                return phenotype, {
                    "null": residual_metadata, "pve": pve,
                    "architecture": frozen_architecture,
                    "causal_group_ids": frozen_causal_ids,
                }

            values, seeds, ids, _ = _draw_bank_responses(
                canonical_prepared, requested_role="power", canonical_role="power",
                count=response_count, design_hash=design_hash, stage=stage,
                scenario_id=f"{scenario_id}.robustness.power.{architecture}",
                replicate_offset=0, null_model="gaussian",
                response_factory=response_factory,
            )
            power_banks[architecture] = (values, seeds, ids, causal_ids)
        if (
            set(calibration_ids) & set(heldout_ids)
            or any(set(calibration_ids) & set(item[2]) for item in power_banks.values())
        ):
            raise RuntimeError("robustness response banks overlap")
        robustness_contexts, marker_designs = _basic_robustness_contexts(
            context, seed, with_metadata=True
        )
        for name, transformed in robustness_contexts.items():
            try:
                if isinstance(transformed, Exception):
                    raise transformed
                challenge_prepared = _prepare_scenario(
                    transformed, design_hash=design_hash, stage=stage,
                    scenario_id=f"{scenario_id}.robustness.{name}", n_jobs=n_jobs,
                )
                calibration_scores, calibration_failure_masks = _robustness_score_bank(
                    challenge_prepared, calibration_responses, n_jobs=n_jobs
                )
                heldout_scores, heldout_failure_masks = _robustness_score_bank(
                    challenge_prepared, heldout_responses, n_jobs=n_jobs
                )
                thresholds = {
                    method: empirical_threshold(
                        values, failed_mask=calibration_failure_masks[method]
                    )
                    for method, values in calibration_scores.items()
                }
                heldout_decisions = {
                    method: apply_failure_policy(
                        apply_threshold(
                            heldout_scores[method], thresholds[method]
                        ),
                        heldout_failure_masks[method],
                        response_role="heldout",
                    )
                    for method in _ROBUSTNESS_METHODS
                }
                strata: dict[str, Any] = {}
                for architecture, (responses, power_seeds, power_ids, causal_ids) in power_banks.items():
                    candidate_scores, candidate_failure_masks = _robustness_score_bank(
                        challenge_prepared, responses, n_jobs=n_jobs
                    )
                    canonical_scores, canonical_failure_masks = _robustness_score_bank(
                        canonical_prepared, responses, n_jobs=n_jobs
                    )
                    causal_index = context.family.group_ids.index(causal_ids[0])
                    detection = {
                        method: apply_failure_policy(
                            np.asarray(
                                candidate_scores[method][causal_index]
                                < thresholds[method],
                                dtype=bool,
                            ),
                            candidate_failure_masks[method],
                            response_role="power",
                        ).tolist()
                        for method in _ROBUSTNESS_METHODS
                    }
                    correlations_by_method: dict[str, list[float | None]] = {}
                    top_k_by_method: dict[str, list[float]] = {}
                    for method in _ROBUSTNESS_METHODS:
                        correlations, top_k = _ranking_metrics(
                            canonical_scores[method], candidate_scores[method],
                            context.family.group_ids,
                        )
                        correlations_by_method[method] = correlations
                        top_k_by_method[method] = top_k
                    powers = {
                        method: float(np.mean(values))
                        for method, values in detection.items()
                    }
                    regret_by_method = {
                        "omnib": abs(
                            powers["omnib"] - max(
                                powers["minor_burden"], powers["pc1"],
                                powers["kernel_hadamard"],
                            )
                        ),
                        "minor_burden": None,
                        "pc1": None,
                        "kernel_hadamard": None,
                    }
                    strata[architecture] = {
                        "response_ids": list(power_ids), "seeds": list(power_seeds),
                        "response_hash": _array_hash(responses),
                        "causal_group_ids": causal_ids,
                        "response_failures": _response_failure_record(
                            candidate_failure_masks,
                            response_count=response_count,
                            response_role="power",
                            failure_rate_max=_DIAGNOSTIC_FAILURE_RATE_MAX,
                        ),
                        "baseline_response_failures": _response_failure_record(
                            canonical_failure_masks,
                            response_count=response_count,
                            response_role="power",
                            failure_rate_max=_DIAGNOSTIC_FAILURE_RATE_MAX,
                        ),
                        "p_by_method": {
                            method: _json_safe(values)
                            for method, values in candidate_scores.items()
                        },
                        "p_hashes": {
                            method: sha256_payload(_json_safe(values))
                            for method, values in candidate_scores.items()
                        },
                        "baseline_p_by_method": {
                            method: _json_safe(values)
                            for method, values in canonical_scores.items()
                        },
                        "baseline_p_hashes": {
                            method: sha256_payload(_json_safe(values))
                            for method, values in canonical_scores.items()
                        },
                        "rank_correlation_by_method": correlations_by_method,
                        "top_k_jaccard_by_method": top_k_by_method,
                        **(
                            {
                                "qa_detection_by_method": detection,
                                "qa_power_by_method": powers,
                                "qa_absolute_power_regret_by_method": regret_by_method,
                            }
                            if qa_only else {
                                "detection_by_method": detection,
                                "power_by_method": powers,
                                "absolute_power_regret_by_method": regret_by_method,
                            }
                        ),
                    }
                finite_correlations = [
                    value for record in strata.values()
                    for value in record["rank_correlation_by_method"]["omnib"]
                    if value is not None
                ]
                top_k_values = [
                    value for record in strata.values()
                    for value in record["top_k_jaccard_by_method"]["omnib"]
                ]
                fwer_value = float(np.mean(heldout_decisions["omnib"]))
                robustness[name] = {
                    "status": "completed",
                    "error_type": None,
                    "message": None,
                    "rank_correlation": (
                        float(np.mean(finite_correlations))
                        if finite_correlations else None
                    ),
                    "top_k": min(10, len(context.family.group_ids)),
                    "top_k_jaccard": float(np.mean(top_k_values)),
                    "non_estimable_rate": float(np.mean(
                        ~np.isfinite(heldout_scores["omnib"])
                    )),
                    "realized_marker_design": marker_designs.get(name),
                    "note": "response-level challenge evidence; metrics stratified by architecture",
                    "design_ruling": {
                        "interaction_pve": 0.05, "causal_groups": 1,
                        "architectures": architectures,
                        "calibration_count": calibration_count,
                        "heldout_count": response_count,
                        "power_count_per_architecture": response_count,
                        "stratify_by_architecture": True,
                        "component_regret_reference": [
                            "minor_burden", "pc1", "kernel_hadamard",
                        ],
                    },
                    "calibration": {
                        "response_ids": list(calibration_ids),
                        "seeds": list(calibration_seeds),
                        "response_hash": _array_hash(calibration_responses),
                        "p_by_method": {
                            method: _json_safe(values)
                            for method, values in calibration_scores.items()
                        },
                        "p_hashes": {
                            method: sha256_payload(_json_safe(values))
                            for method, values in calibration_scores.items()
                        },
                        "response_failures": _response_failure_record(
                            calibration_failure_masks,
                            response_count=calibration_count,
                            response_role="calibration",
                            failure_rate_max=_DIAGNOSTIC_FAILURE_RATE_MAX,
                        ),
                        **(
                            {"qa_cutoffs_by_method": thresholds}
                            if qa_only else {"thresholds": thresholds}
                        ),
                    },
                    "heldout": {
                        "response_ids": list(heldout_ids), "seeds": list(heldout_seeds),
                        "response_hash": _array_hash(heldout_responses),
                        "p_by_method": {
                            method: _json_safe(values)
                            for method, values in heldout_scores.items()
                        },
                        "p_hashes": {
                            method: sha256_payload(_json_safe(values))
                            for method, values in heldout_scores.items()
                        },
                        "response_failures": _response_failure_record(
                            heldout_failure_masks,
                            response_count=response_count,
                            response_role="heldout",
                            failure_rate_max=_DIAGNOSTIC_FAILURE_RATE_MAX,
                        ),
                        **(
                            {"qa_rejections_by_method": {
                                method: values.tolist()
                                for method, values in heldout_decisions.items()
                            }}
                            if qa_only else {"rejections_by_method": {
                                method: values.tolist()
                                for method, values in heldout_decisions.items()
                            }}
                        ),
                    },
                    **(
                        {
                            "qa_fwer": fwer_value,
                            "qa_power": None,
                            "qa_absolute_power_regret": None,
                            "qa_power_by_architecture": strata,
                        }
                        if qa_only else {
                            "fwer": fwer_value,
                            "power": None,
                            "absolute_power_regret": None,
                            "power_by_architecture": strata,
                        }
                    ),
                }
            except Exception as error:
                robustness[name] = {
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "message": str(error),
                    "rank_correlation": None,
                    "top_k": None,
                    "top_k_jaccard": None,
                    "non_estimable_rate": None,
                    "realized_marker_design": marker_designs.get(name),
                    "note": "robustness scoring failed; no metric was inferred",
                    "design_ruling": None,
                    "calibration": None,
                    "heldout": None,
                    **(
                        {
                            "qa_fwer": None,
                            "qa_power": None,
                            "qa_absolute_power_regret": None,
                            "qa_power_by_architecture": None,
                        }
                        if qa_only else {
                            "fwer": None,
                            "power": None,
                            "absolute_power_regret": None,
                            "power_by_architecture": None,
                        }
                    ),
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
            "inference_status": (
                "noninferential_do_not_threshold" if qa_only else "formal"
            ),
            "request_hash": request_hash,
            "seed_id": seed_id,
            "response_hash": baseline["response_hash"],
            "bootstrap_B": bootstrap_B,
            "hypothesis_unit": "group",
            "family_scope": "primary_only",
            "pair_edges_per_group": _edge_count_per_group(context),
            "direct_higher_order_term": False,
            "null_covariance": baseline["null_covariance"],
            "feature_cache_sha256": baseline["feature_cache_sha256"],
            "fixed_mask_sha256": baseline["fixed_mask_sha256"],
            "null_fit_sha256": baseline["null_fit_sha256"],
            "prepared_design_sha256": baseline["prepared_design_sha256"],
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
    context: OmniBBenchmarkContext | Mapping[str, OmniBBenchmarkContext],
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
    if isinstance(context, Mapping):
        backbone = str(scenario.parameters.get("backbone", ""))
        context_key = (
            f"{backbone}:g{scenario.parameters['family_size']}"
            if scenario.parameters.get("experiment") == "family_size" else backbone
        )
        selected = context.get(context_key)
        if not isinstance(selected, OmniBBenchmarkContext):
            raise ValueError(
                f"scenario requires presealed benchmark context {context_key!r}"
            )
        context = selected
    if not isinstance(context, OmniBBenchmarkContext):
        raise ValueError("run_omnib_replicate requires a benchmark context")
    if replicate < 0 or replicate >= scenario.replicates:
        raise ValueError("replicate is outside the scenario registry range")
    experiment = scenario.parameters.get("experiment")
    if experiment not in {
        "end2end", "conditional", "power", "encoding", "global_vc", "family_size",
    }:
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
            sha256_payload(power_calibration_bank.statistical_artifact())
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
                "feature_cache_sha256": bank.feature_cache_sha256,
                "fixed_mask_sha256": bank.fixed_mask_sha256,
                "null_fit_sha256": bank.null_fit_sha256,
                "prepared_design_sha256": bank.prepared_design_sha256,
                "failure": _bank_failure_envelope(bank.failure),
            }
        elif experiment == "global_vc":
            payload = run_global_vc_bank(
                context,
                bank=str(scenario.parameters["bank"]),
                count=scenario.replicates,
                design_hash=design_hash,
                qa_only=qa_only,
                scenario_id=scenario.scenario_id,
                replicate=replicate,
                calibration_count=(
                    int(scenario.parameters["calibration_count"])
                    if "calibration_count" in scenario.parameters else None
                ),
            )
        elif experiment == "family_size":
            declared_size = int(scenario.parameters["family_size"])
            payload = {
                **_base_replicate_payload(
                    context, replicate=replicate, design_hash=design_hash,
                    stage=scenario.stage, scenario_id=scenario.scenario_id,
                    n_jobs=n_jobs,
                ),
                **run_family_size_stress(
                    context, family_size=declared_size,
                    response_count=scenario.replicates,
                    calibration_count=scenario.bootstrap_B,
                    design_hash=design_hash, qa_only=qa_only, n_jobs=n_jobs,
                    scenario_id=scenario.scenario_id,
                ),
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
    "run_global_vc_bank",
    "run_family_size_stress",
    "run_end_to_end_null",
    "run_omnib_replicate",
    "run_power_replicate",
    "validate_real_omnib_context",
]
