from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from types import MappingProxyType
from typing import Any

from scripts.benchmarks.v201.contracts import sha256_payload

from .plan import InvocationSpec

_THREAD_ENV = MappingProxyType(
    {
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }
)
_GIB = 1024**3
NATIVE_CAPS = MappingProxyType(
    {
        "wall_seconds": 7200,
        "cpu_seconds": 115200,
        "peak_aggregate_pss_gib": 128,
        "output_storage_gib": 20,
        "large_tmp_output_forbidden": True,
    }
)


@dataclass(frozen=True)
class CommandSpec:
    argv: tuple[str, ...]
    env: Mapping[str, str]
    purpose: str
    invocation_id: str


class ExecutionBlocked(RuntimeError):
    """Native execution lacks authority or violated a terminal constraint."""


@dataclass(frozen=True)
class StaticPreflight:
    status: str
    record: Mapping[str, Any]
    record_sha256: str


@dataclass(frozen=True)
class ResourceSample:
    wall_seconds: float
    cpu_seconds: float
    aggregate_pss_bytes: int
    output_bytes: int


@dataclass(frozen=True)
class RuntimeObservation:
    resources: ResourceSample
    exit_code: int | None


class CapBreach(ExecutionBlocked):
    def __init__(
        self,
        dimension: str,
        observed: float | int,
        limit: float | int,
    ) -> None:
        self.dimension = dimension
        self.observed = observed
        self.limit = limit
        super().__init__(
            f"native aggregate cap breached: {dimension}={observed} > {limit}"
        )


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def require_native_authority(authority: Mapping[str, Any]) -> None:
    """Require the complete post-materialization native execution gate."""

    if authority.get("execution_authorized") is not True:
        raise ExecutionBlocked("execution_authorized=false")
    hash_fields = {
        "qa_design_hash",
        "prospective_inventory_sha256",
        "anchor_manifest_sha256",
        "response_manifest_sha256",
        "config_manifest_sha256",
        "source_input_reverification_sha256",
        "artifact_review_sha256",
        "static_preflight_sha256",
        "anchor_manifest_design_hash",
        "response_manifest_design_hash",
        "config_manifest_design_hash",
    }
    details = sorted(hash_fields - authority.keys())
    details.extend(
        f"invalid:{field}"
        for field in sorted(hash_fields & authority.keys())
        if not _is_sha256(authority[field])
    )
    if authority.get("artifact_review_verdict") != "ACCEPT":
        details.append("artifact_review_verdict=ACCEPT")
    if authority.get("static_preflight_status") != "PASS":
        details.append("static_preflight_status=PASS")
    design_hash = authority.get("qa_design_hash")
    for field in (
        "anchor_manifest_design_hash",
        "response_manifest_design_hash",
        "config_manifest_design_hash",
    ):
        if field in authority and authority[field] != design_hash:
            details.append(f"mismatch:{field}")
    if details:
        raise ExecutionBlocked(
            "native execution evidence is incomplete: " + ", ".join(details)
        )


def _require_number(record: Mapping[str, Any], field: str) -> float:
    value = record.get(field)
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ExecutionBlocked(f"static preflight {field} is invalid")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0:
        raise ExecutionBlocked(f"static preflight {field} is invalid")
    return numeric


def evaluate_static_preflight(record: Mapping[str, Any]) -> StaticPreflight:
    """Evaluate only recorded host availability and supervisor configuration."""

    forbidden = {
        "supervisor_self_test",
        "resource_projection",
        "timing_workload",
        "genotype_computation",
        "phenotype_computation",
    } & set(record)
    if forbidden:
        raise ExecutionBlocked(
            "static preflight contains prohibited workload evidence: "
            + ", ".join(sorted(forbidden))
        )
    timestamp = record.get("utc_timestamp")
    if not isinstance(timestamp, str) or not timestamp.endswith("Z"):
        raise ExecutionBlocked("static preflight UTC timestamp is invalid")
    logical_cpus = _require_number(record, "logical_cpu_count")
    if not logical_cpus.is_integer() or logical_cpus < 1:
        raise ExecutionBlocked("static preflight logical CPU inventory is invalid")
    mem_available = _require_number(record, "mem_available_bytes")
    output_free = _require_number(record, "output_filesystem_free_bytes")
    _require_number(record, "temporary_filesystem_free_bytes")
    if mem_available < 128 * _GIB:
        raise ExecutionBlocked("static preflight MemAvailable is below 128 GiB")
    if output_free < 20 * _GIB:
        raise ExecutionBlocked("static preflight output filesystem is below 20 GiB")
    if record.get("output_root_exists") is not False:
        raise ExecutionBlocked("static preflight output root is occupied")
    if record.get("output_lock_exists") is not False:
        raise ExecutionBlocked("static preflight output lock is occupied")
    if record.get("supervisor_executable_present_and_executable") is not True:
        raise ExecutionBlocked("static preflight supervisor executable is unavailable")
    for field in (
        "supervisor_executable_sha256",
        "supervisor_configuration_sha256",
    ):
        if not _is_sha256(record.get(field)):
            raise ExecutionBlocked(f"static preflight {field} is invalid")
    if record.get("configured_thresholds") != dict(NATIVE_CAPS):
        raise ExecutionBlocked("static preflight supervisor thresholds differ from contract")
    normalized = deepcopy(dict(record))
    return StaticPreflight(
        status="PASS",
        record=MappingProxyType(normalized),
        record_sha256=sha256_payload(normalized),
    )


def _validate_resource_sample(sample: ResourceSample) -> None:
    for field, value in (
        ("wall_seconds", sample.wall_seconds),
        ("cpu_seconds", sample.cpu_seconds),
        ("peak_aggregate_pss_bytes", sample.aggregate_pss_bytes),
        ("output_storage_bytes", sample.output_bytes),
    ):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ExecutionBlocked(f"invalid supervisor sample: {field}")
        if not math.isfinite(float(value)) or float(value) < 0.0:
            raise ExecutionBlocked(f"invalid supervisor sample: {field}")


def evaluate_cap_sample(sample: ResourceSample) -> None:
    """Raise on the first frozen cap breach in deterministic contract order."""

    _validate_resource_sample(sample)
    checks: tuple[tuple[str, float | int, float | int], ...] = (
        ("wall_seconds", sample.wall_seconds, NATIVE_CAPS["wall_seconds"]),
        ("cpu_seconds", sample.cpu_seconds, NATIVE_CAPS["cpu_seconds"]),
        (
            "peak_aggregate_pss_bytes",
            sample.aggregate_pss_bytes,
            int(NATIVE_CAPS["peak_aggregate_pss_gib"]) * _GIB,
        ),
        (
            "output_storage_bytes",
            sample.output_bytes,
            int(NATIVE_CAPS["output_storage_gib"]) * _GIB,
        ),
    )
    for dimension, observed, limit in checks:
        if observed > limit:
            raise CapBreach(dimension, observed, limit)


def run_sequence(
    commands: Iterable[CommandSpec],
    *,
    authority: Mapping[str, Any],
    executor: Callable[[CommandSpec], Any],
    sampler: Callable[[Any], Iterable[Any]],
    terminator: Callable[[Any], None],
) -> tuple[str, ...]:
    """Run validate/interact/audit sequentially under injected supervision."""

    require_native_authority(authority)
    commands = tuple(commands)
    if tuple(command.purpose for command in commands) != (
        "validate",
        "interact",
        "audit",
    ):
        raise ExecutionBlocked("native sequence must be validate, interact, audit")
    invocation_ids = {command.invocation_id for command in commands}
    if len(invocation_ids) != 1:
        raise ExecutionBlocked("native sequence mixes invocation IDs")
    completed: list[str] = []
    for command in commands:
        handle = executor(command)
        saw_terminal_exit = False
        for observation in sampler(handle):
            if not isinstance(observation, RuntimeObservation):
                terminator(handle)
                raise ExecutionBlocked("supervisor returned an invalid observation")
            try:
                evaluate_cap_sample(observation.resources)
            except CapBreach:
                terminator(handle)
                raise
            if observation.exit_code is not None:
                saw_terminal_exit = True
                if observation.exit_code != 0:
                    raise ExecutionBlocked(
                        f"{command.purpose} exited with status {observation.exit_code}"
                    )
                break
        if not saw_terminal_exit:
            terminator(handle)
            raise ExecutionBlocked(
                f"supervisor ended before {command.purpose} reported an exit status"
            )
        completed.append(command.purpose)
    return tuple(completed)


def commands_for_invocation(
    invocation: InvocationSpec,
    *,
    executable: str | Path,
    config_path: str | Path,
    result_dir: str | Path,
    audit_dir: str | Path,
) -> tuple[CommandSpec, ...]:
    executable = str(executable)
    config_path = str(config_path)
    result_dir = str(result_dir)
    audit_dir = str(audit_dir)
    rows = (
        ("validate", (executable, "validate", "-c", config_path)),
        (
            "interact",
            (
                executable,
                "interact",
                "-c",
                config_path,
                "--n-jobs",
                str(invocation.jobs),
            ),
        ),
        ("audit", (executable, "audit", result_dir, "-o", audit_dir)),
    )
    return tuple(
        CommandSpec(
            argv=argv,
            env=_THREAD_ENV,
            purpose=purpose,
            invocation_id=invocation.invocation_id,
        )
        for purpose, argv in rows
    )
