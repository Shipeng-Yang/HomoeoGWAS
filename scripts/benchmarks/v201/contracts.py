"""Small, deterministic data contracts shared by benchmark tracks."""

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Stage = Literal["pilot", "formal"]
Track = Literal["fit", "omnib", "scaling", "application"]


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
        if self.copies < 2 or self.edges < 1 or self.bootstrap_B < 0:
            raise ValueError("invalid scaling anchor settings")
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
