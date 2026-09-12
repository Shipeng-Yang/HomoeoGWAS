"""Small, deterministic data contracts shared by benchmark tracks."""

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from itertools import pairwise
from numbers import Integral
from types import MappingProxyType
from typing import Any, Literal

Stage = Literal["pilot", "formal"]
Track = Literal["fit", "omnib", "scaling", "application"]
GIB = 1024 ** 3
COMPARATOR_PROBE_WIDTHS = (1, 5, 20)
COMPARATOR_TARGET_RESPONSE_COUNT = 2_000
COMPARATOR_PROJECTION_SAFETY_FACTOR = 2.0
COMPARATOR_TIME_PER_RESPONSE_TOLERANCE = 0.20


@dataclass(frozen=True)
class ComparatorResourceLimit:
    """Frozen exhaustive-SNP×SNP resource boundary for one real panel."""

    panel_id: str
    family_size: int
    copies: int
    max_offered_pairs: int
    max_parent_rss_bytes: int = 32 * GIB
    max_aggregate_pss_bytes: int = 128 * GIB

    def __post_init__(self) -> None:
        values = (
            self.family_size,
            self.copies,
            self.max_offered_pairs,
            self.max_parent_rss_bytes,
            self.max_aggregate_pss_bytes,
        )
        if (
            not isinstance(self.panel_id, str)
            or not self.panel_id
            or any(
                isinstance(value, bool)
                or not isinstance(value, Integral)
                or int(value) < 1
                for value in values
            )
        ):
            raise ValueError("invalid comparator resource limit")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_COMPARATOR_RESOURCE_LIMITS = {
    "REALG.CGVD1245": ComparatorResourceLimit(
        "REALG.CGVD1245", family_size=80, copies=2,
        max_offered_pairs=10_000,
    ),
    "REALG.WATKINS_F2143": ComparatorResourceLimit(
        "REALG.WATKINS_F2143", family_size=80, copies=3,
        max_offered_pairs=750_000,
    ),
}


def comparator_resource_limit(
    panel_id: str,
    *,
    family_size: int,
    copies: int,
) -> ComparatorResourceLimit:
    """Return the exact reviewed real-panel limit or fail as unfrozen."""

    limit = _COMPARATOR_RESOURCE_LIMITS.get(panel_id)
    if (
        limit is None
        or isinstance(family_size, bool)
        or not isinstance(family_size, Integral)
        or isinstance(copies, bool)
        or not isinstance(copies, Integral)
        or int(family_size) != limit.family_size
        or int(copies) != limit.copies
    ):
        raise ValueError(
            "SNPxSNP comparator context is unfrozen and not authorized"
        )
    return limit


def _finite_number(value: Any, field_name: str, *, positive: bool) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        or (positive and value <= 0)
    ):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{field_name} must be a finite {qualifier} number")
    return float(value)


def _sha256_string(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


@dataclass(frozen=True)
class ComparatorProbeRecord:
    """One noninferential exhaustive-SNP×SNP cost-probe measurement."""

    schema: str
    panel_id: str
    sample_context: str
    family_size: int
    copies: int
    response_width: int
    design_hash: str
    context_fingerprint: str
    prepared_design_sha256: str
    member_family_sha256: str
    scorer_wall_seconds: float
    scorer_cpu_seconds: float
    peak_parent_rss_bytes: int
    peak_aggregate_pss_bytes: int
    output_bytes: int
    offered_pair_count: int
    design_nonestimable_pair_count: int
    tested_pair_count: int
    nonfinite_pair_score_count: int
    failed_response_indices: tuple[int, ...]
    gated_marker_count_by_gene: Mapping[str, int]
    requested_jobs: int
    effective_jobs: int
    parallel_backend: str
    worker_pids: tuple[int, ...]
    inference_status: str
    execution_authorized: bool

    def _validate_fields(self, *, expected_schema: str) -> None:
        limit = comparator_resource_limit(
            self.panel_id, family_size=self.family_size, copies=self.copies,
        )
        integer_fields = {
            "response_width": self.response_width,
            "peak_parent_rss_bytes": self.peak_parent_rss_bytes,
            "peak_aggregate_pss_bytes": self.peak_aggregate_pss_bytes,
            "output_bytes": self.output_bytes,
            "offered_pair_count": self.offered_pair_count,
            "design_nonestimable_pair_count": self.design_nonestimable_pair_count,
            "tested_pair_count": self.tested_pair_count,
            "nonfinite_pair_score_count": self.nonfinite_pair_score_count,
            "requested_jobs": self.requested_jobs,
            "effective_jobs": self.effective_jobs,
        }
        if any(
            isinstance(value, bool)
            or not isinstance(value, Integral)
            or int(value) < (1 if name in {
                "response_width", "peak_parent_rss_bytes",
                "peak_aggregate_pss_bytes", "offered_pair_count",
                "tested_pair_count", "requested_jobs", "effective_jobs",
            } else 0)
            for name, value in integer_fields.items()
        ):
            raise ValueError("comparator probe integer fields are invalid")
        if (
            self.schema != expected_schema
            or self.sample_context != "full"
            or self.response_width not in COMPARATOR_PROBE_WIDTHS
            or self.inference_status != "noninferential_resource_probe"
            or self.execution_authorized is not False
            or not all(_sha256_string(value) for value in (
                self.design_hash,
                self.context_fingerprint,
                self.prepared_design_sha256,
                self.member_family_sha256,
            ))
        ):
            raise ValueError("comparator probe identity or role is invalid")
        wall = _finite_number(
            self.scorer_wall_seconds, "scorer_wall_seconds", positive=True,
        )
        cpu = _finite_number(
            self.scorer_cpu_seconds, "scorer_cpu_seconds", positive=False,
        )
        if self.offered_pair_count > limit.max_offered_pairs:
            raise ValueError("SNPxSNP offered pair ceiling exceeded")
        if self.peak_parent_rss_bytes > limit.max_parent_rss_bytes:
            raise ValueError("SNPxSNP parent RSS ceiling exceeded")
        if self.peak_aggregate_pss_bytes > limit.max_aggregate_pss_bytes:
            raise ValueError("SNPxSNP aggregate PSS ceiling exceeded")
        if (
            self.offered_pair_count - self.design_nonestimable_pair_count
            != self.tested_pair_count
        ):
            raise ValueError("comparator probe pair counts are inconsistent")
        failed = tuple(self.failed_response_indices)
        if (
            any(
                isinstance(index, bool)
                or not isinstance(index, Integral)
                or int(index) < 0
                or int(index) >= self.response_width
                for index in failed
            )
            or tuple(sorted(int(index) for index in failed))
            != tuple(int(index) for index in failed)
            or len(set(failed)) != len(failed)
            or failed
            or self.nonfinite_pair_score_count != 0
        ):
            raise ValueError("resource probe must complete every response")
        markers = self.gated_marker_count_by_gene
        if (
            not isinstance(markers, Mapping)
            or not markers
            or any(
                not isinstance(key, str)
                or not key
                or isinstance(value, bool)
                or not isinstance(value, Integral)
                or int(value) < 1
                for key, value in markers.items()
            )
        ):
            raise ValueError("gated marker counts are invalid")
        pids = tuple(self.worker_pids)
        if (
            self.requested_jobs != 1
            or self.effective_jobs != 1
            or self.parallel_backend != "serial"
            or len(pids) != 1
            or isinstance(pids[0], bool)
            or not isinstance(pids[0], Integral)
            or int(pids[0]) < 1
        ):
            raise ValueError("comparator probe execution provenance is invalid")
        object.__setattr__(self, "scorer_wall_seconds", wall)
        object.__setattr__(self, "scorer_cpu_seconds", cpu)
        object.__setattr__(
            self,
            "failed_response_indices",
            tuple(int(index) for index in failed),
        )
        object.__setattr__(
            self,
            "gated_marker_count_by_gene",
            MappingProxyType({
                key: int(value) for key, value in sorted(markers.items())
            }),
        )
        object.__setattr__(self, "worker_pids", tuple(int(pid) for pid in pids))

    def __post_init__(self) -> None:
        self._validate_fields(
            expected_schema="homoeogwas-snpxsnp-resource-probe-v1"
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ComparatorProbeRecord":
        if not isinstance(payload, Mapping):
            raise ValueError("comparator probe must be a mapping")
        fields = set(cls.__dataclass_fields__)
        if set(payload) != fields:
            raise ValueError("comparator probe schema fields differ")
        values = dict(payload)
        for field_name in (
            "failed_response_indices", "worker_pids", "response_ids",
            "sampled_pids",
        ):
            if field_name in values:
                values[field_name] = tuple(values[field_name])
        return cls(**values)

    def to_payload(self) -> dict[str, Any]:
        return {
            field_name: (
                dict(value) if field_name == "gated_marker_count_by_gene"
                else list(value) if field_name in {
                    "failed_response_indices", "worker_pids", "response_ids",
                    "sampled_pids",
                }
                else value
            )
            for field_name in self.__dataclass_fields__
            for value in (getattr(self, field_name),)
        }


@dataclass(frozen=True)
class ComparatorProbeRecordV2(ComparatorProbeRecord):
    """One authorization-, response- and process-bound resource probe."""

    probe_authorization_sha256: str
    implementation_commit: str
    matched_comparator_contract_sha256: str
    input_family_sha256: str
    response_bank_sha256: str
    response_ids: tuple[str, ...]
    response_ids_sha256: str
    response_prefix_sha256: str
    score_evidence_sha256: str
    root_pid: int
    sampled_pids: tuple[int, ...]
    process_set_reconciled: bool
    aggregate_pss_missed_spike_strategy: str
    formal_execution_authorized: bool

    def __post_init__(self) -> None:
        self._validate_fields(
            expected_schema="homoeogwas-snpxsnp-resource-probe-v2"
        )
        hashes = (
            self.probe_authorization_sha256,
            self.matched_comparator_contract_sha256,
            self.input_family_sha256,
            self.response_bank_sha256,
            self.response_ids_sha256,
            self.response_prefix_sha256,
            self.score_evidence_sha256,
        )
        response_ids = tuple(self.response_ids)
        sampled_pids = tuple(self.sampled_pids)
        if (
            not all(_sha256_string(value) for value in hashes)
            or not isinstance(self.implementation_commit, str)
            or len(self.implementation_commit) != 40
            or any(
                character not in "0123456789abcdef"
                for character in self.implementation_commit
            )
            or len(response_ids) != max(COMPARATOR_PROBE_WIDTHS)
            or len(set(response_ids)) != len(response_ids)
            or any(
                not isinstance(response_id, str) or not response_id
                for response_id in response_ids
            )
            or self.response_ids_sha256 != sha256_payload({
                "ordered_response_ids": list(response_ids),
            })
            or isinstance(self.root_pid, bool)
            or not isinstance(self.root_pid, Integral)
            or int(self.root_pid) < 1
            or any(
                isinstance(pid, bool)
                or not isinstance(pid, Integral)
                or int(pid) < 1
                for pid in sampled_pids
            )
            or tuple(sorted(int(pid) for pid in sampled_pids))
            != tuple(int(pid) for pid in sampled_pids)
            or len(set(sampled_pids)) != len(sampled_pids)
            or tuple(int(pid) for pid in sampled_pids) != (int(self.root_pid),)
            or self.worker_pids != (int(self.root_pid),)
            or self.process_set_reconciled is not True
            or self.aggregate_pss_missed_spike_strategy
            != "serial_singleton_parent_rss_upper_bound"
            or self.peak_aggregate_pss_bytes < self.peak_parent_rss_bytes
            or self.formal_execution_authorized is not False
        ):
            raise ValueError("comparator probe execution provenance or role is invalid")
        object.__setattr__(self, "response_ids", response_ids)
        object.__setattr__(
            self, "sampled_pids", tuple(int(pid) for pid in sampled_pids)
        )


def validate_comparator_probe_series(
    payloads: Sequence[Mapping[str, Any] | ComparatorProbeRecord],
) -> dict[str, Any]:
    """Validate the width anchors and return a safety-adjusted projection."""

    def parse_record(
        value: Mapping[str, Any] | ComparatorProbeRecord,
    ) -> ComparatorProbeRecord:
        if isinstance(value, ComparatorProbeRecord):
            return value
        if not isinstance(value, Mapping):
            raise ValueError("comparator probe must be a mapping")
        record_type = (
            ComparatorProbeRecordV2
            if value.get("schema") == "homoeogwas-snpxsnp-resource-probe-v2"
            else ComparatorProbeRecord
        )
        return record_type.from_payload(value)

    records = tuple(parse_record(value) for value in payloads)
    by_width = {record.response_width: record for record in records}
    if len(records) != 3 or tuple(sorted(by_width)) != COMPARATOR_PROBE_WIDTHS:
        raise ValueError("comparator probes require widths 1, 5 and 20")
    identity_fields = (
        "panel_id", "sample_context", "family_size", "copies", "design_hash",
        "context_fingerprint", "prepared_design_sha256", "member_family_sha256",
        "offered_pair_count", "design_nonestimable_pair_count",
        "tested_pair_count", "gated_marker_count_by_gene",
    )
    anchor = records[0]
    if any(type(record) is not type(anchor) for record in records[1:]):
        raise ValueError("comparator probes mix incompatible schema versions")
    if isinstance(anchor, ComparatorProbeRecordV2):
        identity_fields += (
            "probe_authorization_sha256", "implementation_commit",
            "matched_comparator_contract_sha256", "input_family_sha256",
            "response_bank_sha256", "response_ids", "response_ids_sha256",
        )
        if (
            len({record.response_prefix_sha256 for record in records}) != 3
            or len({record.score_evidence_sha256 for record in records}) != 3
        ):
            raise ValueError(
                "comparator probes do not bind distinct width-prefix evidence"
            )
    if any(
        any(getattr(record, field_name) != getattr(anchor, field_name)
            for field_name in identity_fields)
        for record in records[1:]
    ):
        raise ValueError("comparator probes do not use the same frozen context")
    per_response_5 = by_width[5].scorer_wall_seconds / 5
    per_response_20 = by_width[20].scorer_wall_seconds / 20
    relative_growth = (per_response_20 - per_response_5) / per_response_5
    if relative_growth > COMPARATOR_TIME_PER_RESPONSE_TOLERANCE:
        raise ValueError("comparator probe time per response worsens by more than 20%")
    limit = comparator_resource_limit(
        anchor.panel_id, family_size=anchor.family_size, copies=anchor.copies,
    )
    width20 = by_width[20]
    scale = COMPARATOR_TARGET_RESPONSE_COUNT / width20.response_width
    safety = COMPARATOR_PROJECTION_SAFETY_FACTOR

    def memory_upper_envelope(field_name: str) -> tuple[int, dict[str, Any]]:
        points = [
            (width, int(getattr(by_width[width], field_name)))
            for width in COMPARATOR_PROBE_WIDTHS
        ]
        if any(
            right[1] < left[1]
            for left, right in pairwise(points)
        ):
            raise ValueError(
                f"nonmonotone comparator memory anchors for {field_name}"
            )
        slopes = [
            Fraction(right_value - left_value, right_width - left_width)
            for left_index, (left_width, left_value) in enumerate(points)
            for right_width, right_value in points[left_index + 1:]
        ]
        slope = max(slopes, default=Fraction(0, 1))
        intercept = max(
            Fraction(value, 1) - slope * width for width, value in points
        )
        intercept = max(intercept, Fraction(0, 1))
        target = intercept + slope * COMPARATOR_TARGET_RESPONSE_COUNT
        target_bytes = math.ceil(target)
        model = {
            "anchor_bytes_by_width": {
                str(width): value for width, value in points
            },
            "intercept_bytes": math.ceil(intercept),
            "slope_bytes_per_response": float(slope),
            "target_without_safety_bytes": target_bytes,
        }
        return int(math.ceil(target_bytes * safety)), model

    projected_parent_rss, parent_model = memory_upper_envelope(
        "peak_parent_rss_bytes"
    )
    projected_aggregate_pss, pss_model = memory_upper_envelope(
        "peak_aggregate_pss_bytes"
    )
    projected = {
        "scorer_cpu_seconds": width20.scorer_cpu_seconds * scale * safety,
        "elapsed_seconds": width20.scorer_wall_seconds * scale * safety,
        "output_bytes": int(math.ceil(width20.output_bytes * scale * safety)),
        "peak_parent_rss_bytes": projected_parent_rss,
        "peak_aggregate_pss_bytes": projected_aggregate_pss,
    }
    limits = {
        "scorer_cpu_seconds": int(FORMAL_BUDGET.cpu_hours * 3_600),
        "elapsed_seconds": int(FORMAL_BUDGET.elapsed_hours * 3_600),
        "output_bytes": int(FORMAL_BUDGET.output_gb * GIB),
        "peak_parent_rss_bytes": limit.max_parent_rss_bytes,
        "peak_aggregate_pss_bytes": limit.max_aggregate_pss_bytes,
    }
    labels = {
        "scorer_cpu_seconds": "CPU",
        "elapsed_seconds": "elapsed",
        "output_bytes": "storage",
        "peak_parent_rss_bytes": "parent RSS",
        "peak_aggregate_pss_bytes": "aggregate PSS",
    }
    for field_name, value in projected.items():
        if value > limits[field_name]:
            raise ValueError(
                f"safety-adjusted {labels[field_name]} projection exceeds its ceiling"
            )
    return {
        "schema": "homoeogwas-snpxsnp-resource-projection-v2",
        "panel_id": anchor.panel_id,
        "sample_context": anchor.sample_context,
        "family_size": anchor.family_size,
        "response_widths": list(COMPARATOR_PROBE_WIDTHS),
        "target_response_count": COMPARATOR_TARGET_RESPONSE_COUNT,
        "time_per_response_growth_5_to_20": relative_growth,
        "time_amortization_observed": relative_growth < 0.0,
        "safety_factor": safety,
        "memory_projection_models": {
            "peak_parent_rss_bytes": parent_model,
            "peak_aggregate_pss_bytes": pss_model,
        },
        "projected": projected,
        "limits": limits,
        "accepted": True,
        "inference_status": "noninferential_resource_projection",
    }


def canonical_json(value: Any) -> str:
    """Serialize JSON values with one stable representation."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def sha256_payload(value: Any) -> str:
    """Return the SHA-256 fingerprint of a canonical JSON payload."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def derive_seed(
    design_hash: str,
    track: str,
    scenario_id: str,
    replicate_index: int,
    stage: str,
) -> int:
    """Derive an execution-order-independent 64-bit seed."""

    if not isinstance(design_hash, str) or len(design_hash) != 64:
        raise ValueError("invalid design hash or replicate index")
    if isinstance(replicate_index, bool) or replicate_index < 0:
        raise ValueError("invalid design hash or replicate index")
    material = "|".join(
        (design_hash, track, scenario_id, str(replicate_index), stage)
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


@dataclass(frozen=True)
class ScalingAnchor:
    """A fixed Track C workload definition."""

    anchor_id: str
    n: int
    groups: int
    copies: int
    edges: int
    bootstrap_B: int
    jobs: tuple[int, ...]
    repeats: int = 3

    def __post_init__(self) -> None:
        if not self.anchor_id or self.n < 1 or self.groups < 1:
            raise ValueError("scaling anchor dimensions must be positive")
        expected_edges = self.copies * (self.copies - 1) // 2
        if self.copies < 2 or self.edges != expected_edges or self.bootstrap_B < 0:
            raise ValueError("invalid scaling anchor edges or settings")
        if not self.jobs or any(job < 1 for job in self.jobs):
            raise ValueError("scaling anchor jobs must be positive")
        if self.repeats < 1:
            raise ValueError("scaling anchor repeats must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def name(self) -> str:
        """Human-readable alias retained for scaling table consumers."""

        return self.anchor_id

    @property
    def responses(self) -> int:
        """Alias for the number of prepared bootstrap responses."""

        return self.bootstrap_B

    @property
    def copies_per_group(self) -> int:
        return self.copies

    @property
    def edges_per_group(self) -> int:
        return self.edges


@dataclass(frozen=True)
class Budget:
    """Aggregate execution budget from the benchmark design lock."""

    stage: Stage
    cpu_hours: float
    elapsed_hours: float
    output_gb: float

    def __post_init__(self) -> None:
        if self.stage not in {"pilot", "formal"}:
            raise ValueError("stage must be pilot or formal")
        if self.cpu_hours < 0 or self.elapsed_hours < 0 or self.output_gb < 0:
            raise ValueError("budget values must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    track: Track
    stage: Stage
    replicates: int
    bootstrap_B: int
    parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.scenario_id:
            raise ValueError("scenario_id must not be empty")
        if self.track not in {"fit", "omnib", "scaling", "application"}:
            raise ValueError("invalid benchmark track")
        if self.stage not in {"pilot", "formal"}:
            raise ValueError("invalid benchmark stage")
        if self.replicates < 1 or self.bootstrap_B < 0:
            raise ValueError("replicates and bootstrap_B must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


PILOT_BUDGET = Budget("pilot", 1_000, 12, 200)
FORMAL_BUDGET = Budget("formal", 25_000, 14 * 24, 500)
