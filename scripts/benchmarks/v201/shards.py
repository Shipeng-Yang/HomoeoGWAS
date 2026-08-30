"""Immutable benchmark replicate shards and conservative resource projections."""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import FORMAL_BUDGET, PILOT_BUDGET, Scenario, canonical_json

_BYTES_PER_GB = 1_000_000_000


class ShardConflict(RuntimeError):
    """Raised when an immutable shard cannot be resumed safely."""


class BudgetExceeded(RuntimeError):
    """Raised when a serializable budget projection exceeds its stage limit."""

    def __init__(self, message: str, projection: dict[str, Any]) -> None:
        super().__init__(message)
        self.projection = projection


@dataclass(frozen=True)
class ShardKey:
    """Identity of one benchmark replicate shard."""

    track: str
    scenario_id: str
    replicate: int

    def __post_init__(self) -> None:
        if not isinstance(self.track, str) or not self.track:
            raise ValueError("track must be a non-empty string")
        if not isinstance(self.scenario_id, str) or not self.scenario_id:
            raise ValueError("scenario_id must be a non-empty string")
        if isinstance(self.replicate, bool) or not isinstance(self.replicate, int) or self.replicate < 0:
            raise ValueError("replicate must be a non-negative integer")

    def relative_path(self) -> Path:
        return Path(self.track) / self.scenario_id / f"replicate-{self.replicate:06d}.json"


def _validate_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError("shard payload must be a mapping")
    result = dict(payload)
    design_hash = result.get("design_hash")
    if (
        not isinstance(design_hash, str)
        or len(design_hash) != 64
        or any(character not in "0123456789abcdef" for character in design_hash)
    ):
        raise ValueError("shard payload has an invalid design_hash")
    if not isinstance(result.get("track"), str) or not result["track"]:
        raise ValueError("shard payload has an invalid track")
    if not isinstance(result.get("scenario_id"), str) or not result["scenario_id"]:
        raise ValueError("shard payload has an invalid scenario_id")
    replicate = result.get("replicate")
    if isinstance(replicate, bool) or not isinstance(replicate, int) or replicate < 0:
        raise ValueError("shard payload has an invalid replicate")
    return result


def _validate_key_match(payload: Mapping[str, Any], expected_key: ShardKey) -> None:
    if (
        payload["track"] != expected_key.track
        or payload["scenario_id"] != expected_key.scenario_id
        or payload["replicate"] != expected_key.replicate
    ):
        raise ShardConflict(f"shard key mismatch: {expected_key.relative_path()}")


def write_shard_exclusive(path: Path, payload: dict[str, Any]) -> str:
    """Atomically write a canonical shard, or safely resume an identical one."""

    output = Path(path)
    checked_payload = _validate_payload(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = canonical_json(checked_payload) + "\n"
    if output.exists():
        if canonical_json(json.loads(output.read_text(encoding="utf-8"))) == canonical_json(checked_payload):
            return "existing_identical"
        raise ShardConflict(f"existing shard differs: {output}")
    with tempfile.NamedTemporaryFile(
        "w", dir=output.parent, delete=False, encoding="utf-8"
    ) as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    try:
        os.link(temporary, output)
    except FileExistsError:
        return write_shard_exclusive(output, checked_payload)
    finally:
        temporary.unlink(missing_ok=True)
    return "created"


def load_shard(path: Path, expected_key: ShardKey | None = None) -> dict[str, Any]:
    """Load one shard and optionally ensure its identity matches ``expected_key``."""

    try:
        decoded = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid shard JSON: {path}") from error
    payload = _validate_payload(decoded)
    if expected_key is not None:
        _validate_key_match(payload, expected_key)
    return payload


def expected_shards(scenarios: Iterable[Scenario]) -> list[ShardKey]:
    """Return registry-ordered, zero-based replicate identities."""

    return [
        ShardKey(scenario.track, scenario.scenario_id, replicate)
        for scenario in scenarios
        for replicate in range(scenario.replicates)
    ]


def _number(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if value < 0 or (positive and value <= 0):
        comparison = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be {comparison}")
    return float(value)


def project_budget(
    stage: str,
    pilot_measurements: Iterable[Mapping[str, Any]],
    *,
    effective_workers: int,
) -> dict[str, Any]:
    """Project stage resources from measured shards and per-scenario multipliers."""

    if stage == "pilot":
        budget = PILOT_BUDGET
    elif stage == "formal":
        budget = FORMAL_BUDGET
    else:
        raise ValueError("stage must be pilot or formal")
    if isinstance(effective_workers, bool) or not isinstance(effective_workers, int) or effective_workers < 1:
        raise ValueError("effective_workers must be a positive integer")

    breakdown: dict[str, dict[str, float]] = {}
    total_cpu_seconds = 0.0
    total_output_bytes = 0.0
    for measurement in pilot_measurements:
        if not isinstance(measurement, Mapping):
            raise ValueError("pilot measurement must be a mapping")
        scenario_id = measurement.get("scenario_id")
        if not isinstance(scenario_id, str) or not scenario_id:
            raise ValueError("pilot measurement has an invalid scenario_id")
        cpu_seconds = _number(measurement.get("cpu_seconds"), "cpu_seconds")
        output_bytes = _number(measurement.get("output_bytes"), "output_bytes")
        multiplier = _number(
            measurement.get("scenario_multiplier"), "scenario_multiplier", positive=True
        )
        row = breakdown.setdefault(
            scenario_id,
            {
                "scenario_multiplier": multiplier,
                "measured_cpu_seconds": 0.0,
                "projected_cpu_seconds": 0.0,
                "measured_output_bytes": 0.0,
                "projected_output_bytes": 0.0,
            },
        )
        if row["scenario_multiplier"] != multiplier:
            raise ValueError(f"scenario_multiplier differs within scenario: {scenario_id}")
        row["measured_cpu_seconds"] += cpu_seconds
        row["projected_cpu_seconds"] += cpu_seconds * multiplier
        row["measured_output_bytes"] += output_bytes
        row["projected_output_bytes"] += output_bytes * multiplier
        total_cpu_seconds += cpu_seconds * multiplier
        total_output_bytes += output_bytes * multiplier

    for row in breakdown.values():
        row["cpu_hours"] = row["projected_cpu_seconds"] / 3600
        row["output_gb"] = row["projected_output_bytes"] / _BYTES_PER_GB
    totals = {
        "cpu_seconds": total_cpu_seconds,
        "cpu_hours": total_cpu_seconds / 3600,
        "elapsed_hours": total_cpu_seconds / effective_workers / 3600,
        "output_bytes": total_output_bytes,
        "output_gb": total_output_bytes / _BYTES_PER_GB,
    }
    projection: dict[str, Any] = {
        "stage": stage,
        "effective_workers": effective_workers,
        "totals": totals,
        "limits": {
            "cpu_hours": budget.cpu_hours,
            "elapsed_hours": budget.elapsed_hours,
            "output_gb": budget.output_gb,
        },
        "breakdown": breakdown,
    }
    for name in ("cpu_hours", "elapsed_hours", "output_gb"):
        if totals[name] > projection["limits"][name]:
            raise BudgetExceeded(f"{name} exceeds {stage} budget", projection)
    return projection
