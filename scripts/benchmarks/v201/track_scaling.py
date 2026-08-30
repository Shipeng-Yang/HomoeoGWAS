"""Track C engineering-correctness and process-scaling benchmark.

Every measured workload crosses a fresh Python subprocess boundary.  The
release benchmark child therefore observes native-thread limits before it
imports NumPy/SciPy, matching the installed launcher's process contract.
"""

from __future__ import annotations

import statistics
from argparse import Namespace
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from homoeogwas_launcher import NUMERIC_THREAD_ENV
from scripts.benchmark_omnib_parallel import (
    _run_subprocess,
    evaluate_runs,
)

from .contracts import ScalingAnchor


@dataclass(frozen=True)
class ScalingAnchorRun:
    """One measured subprocess result for one anchor and worker count."""

    anchor_id: str
    jobs: int
    repeat: int
    effective_jobs: int
    backend: str
    worker_pids: tuple[int, ...]
    wall_seconds: float
    cpu_seconds: float
    aggregate_cpu_percent: float
    peak_aggregate_pss_bytes: int
    peak_aggregate_rss_bytes: int
    max_native_threads: int
    max_threads_by_pid: tuple[tuple[int, int], ...]
    result_sha256: str
    numeric_thread_env: tuple[tuple[str, str | None], ...]
    numeric_thread_limit_ok: bool
    command: tuple[str, ...]

    @classmethod
    def from_record(
        cls, anchor_id: str, repeat: int, record: Mapping[str, Any],
    ) -> ScalingAnchorRun:
        """Validate and freeze the release sampler's JSON-compatible record."""

        jobs = int(record["jobs"])
        wall = float(record["wall_seconds"])
        cpu = float(record["cpu_seconds"])
        pss = int(record.get(
            "peak_aggregate_pss_bytes", record["peak_rss_bytes"]
        ))
        rss = int(record["peak_aggregate_rss_bytes"])
        result_hash = str(record["result_sha256"])
        if jobs < 1 or repeat < 0 or wall <= 0 or cpu < 0 or pss <= 0 or rss <= 0:
            raise ValueError("invalid scaling measurement")
        if len(result_hash) != 64:
            raise ValueError("invalid scaling result hash")

        raw_threads = {
            int(pid): int(count)
            for pid, count in dict(record["max_threads_by_pid"]).items()
        }
        environment = {
            name: dict(record.get("numeric_thread_env", {})).get(name)
            for name in NUMERIC_THREAD_ENV
        }
        return cls(
            anchor_id=str(anchor_id),
            jobs=jobs,
            repeat=int(repeat),
            effective_jobs=int(record["effective_jobs"]),
            backend=str(record["backend"]),
            worker_pids=tuple(int(pid) for pid in record["worker_pids"]),
            wall_seconds=wall,
            cpu_seconds=cpu,
            aggregate_cpu_percent=float(record["cpu_percent"]),
            peak_aggregate_pss_bytes=pss,
            peak_aggregate_rss_bytes=rss,
            max_native_threads=max(raw_threads.values(), default=0),
            max_threads_by_pid=tuple(sorted(raw_threads.items())),
            result_sha256=result_hash,
            numeric_thread_env=tuple(
                (name, environment[name]) for name in NUMERIC_THREAD_ENV
            ),
            numeric_thread_limit_ok=all(
                environment[name] == "1" for name in NUMERIC_THREAD_ENV
            ),
            command=tuple(str(part) for part in record.get("command", ())),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["worker_pids"] = list(self.worker_pids)
        payload["max_threads_by_pid"] = {
            str(pid): count for pid, count in self.max_threads_by_pid
        }
        payload["numeric_thread_env"] = dict(self.numeric_thread_env)
        payload["command"] = list(self.command)
        return payload


def scaling_anchors() -> tuple[ScalingAnchor, ...]:
    """Return the four frozen workloads from the approved design."""

    return (
        ScalingAnchor("small_qa", 192, 192, 3, 3, 199, (1, 4, 8)),
        ScalingAnchor("cotton_like", 419, 500, 2, 1, 999, (1, 4, 8, 16)),
        ScalingAnchor("wheat_formal", 827, 2_143, 3, 3, 2_000, (1, 8, 16, 32)),
        ScalingAnchor("quartet_stress", 500, 500, 4, 6, 999, (1, 8, 16)),
    )


def _anchor_args(anchor: ScalingAnchor) -> Namespace:
    return Namespace(
        n=anchor.n,
        groups=anchor.groups,
        responses=anchor.responses,
        copies=anchor.copies,
    )


def run_anchor(anchor: ScalingAnchor) -> tuple[ScalingAnchorRun, ...]:
    """Run one discarded warm-up and three measured subprocesses per jobs value."""

    if anchor.repeats != 3:
        raise ValueError("Track C requires three measured repeats")
    measured: list[ScalingAnchorRun] = []
    args = _anchor_args(anchor)
    for jobs in anchor.jobs:
        _run_subprocess(jobs, args)
        for repeat in range(anchor.repeats):
            measured.append(ScalingAnchorRun.from_record(
                anchor.name, repeat, _run_subprocess(jobs, args)
            ))
    return tuple(measured)


def _median_int(values: Sequence[int]) -> int:
    return int(statistics.median(values))


def summarize_anchor(
    anchor: ScalingAnchor,
    measured: Sequence[ScalingAnchorRun],
) -> dict[str, Any]:
    """Summarize repeated measurements without fitting a scaling model."""

    grouped: dict[int, list[ScalingAnchorRun]] = {jobs: [] for jobs in anchor.jobs}
    if anchor.repeats != 3:
        raise ValueError("Track C requires three measured repeats")
    for run in measured:
        if run.anchor_id != anchor.name or run.jobs not in grouped:
            raise ValueError("scaling run does not belong to anchor")
        grouped[run.jobs].append(run)
    if any(len(runs) != anchor.repeats for runs in grouped.values()):
        raise ValueError("each jobs value requires exactly three measured repeats")

    hashes = [run.result_sha256 for run in measured]
    exact_identity = len(set(hashes)) == 1
    summarized: dict[str, dict[str, Any]] = {}
    release_runs: dict[int, dict[str, Any]] = {}
    for jobs in anchor.jobs:
        runs = sorted(grouped[jobs], key=lambda run: run.repeat)
        if [run.repeat for run in runs] != list(range(anchor.repeats)):
            raise ValueError("scaling repeat indices must be complete and unique")
        job_hashes = {run.result_sha256 for run in runs}
        effective_values = {run.effective_jobs for run in runs}
        backends = {run.backend for run in runs}
        job_summary = {
            "requested_jobs": jobs,
            "effective_jobs": sorted(effective_values),
            "backends": sorted(backends),
            "median_wall_seconds": statistics.median(
                run.wall_seconds for run in runs
            ),
            "median_cpu_seconds": statistics.median(
                run.cpu_seconds for run in runs
            ),
            "median_aggregate_cpu_percent": statistics.median(
                run.aggregate_cpu_percent for run in runs
            ),
            "median_peak_aggregate_pss_bytes": _median_int(
                [run.peak_aggregate_pss_bytes for run in runs]
            ),
            "median_peak_aggregate_rss_bytes": _median_int(
                [run.peak_aggregate_rss_bytes for run in runs]
            ),
            "max_native_threads": max(run.max_native_threads for run in runs),
            "worker_pids_by_repeat": [list(run.worker_pids) for run in runs],
            "numeric_thread_limits_valid": all(
                run.numeric_thread_limit_ok for run in runs
            ),
            "exact_result_hash_identity": len(job_hashes) == 1,
            "result_sha256": next(iter(job_hashes)) if len(job_hashes) == 1 else None,
            "measured_repeats": [run.to_dict() for run in runs],
        }
        summarized[str(jobs)] = job_summary
        release_runs[jobs] = {
            "jobs": jobs,
            "effective_jobs": runs[0].effective_jobs,
            "worker_pids": list(runs[0].worker_pids),
            "wall_seconds": job_summary["median_wall_seconds"],
            "cpu_percent": job_summary["median_aggregate_cpu_percent"],
            "peak_rss_bytes": job_summary["median_peak_aggregate_pss_bytes"],
            "result_sha256": runs[0].result_sha256,
        }

    native_thread_contract = all(
        run.numeric_thread_limit_ok and run.max_native_threads == 1
        for run in measured
    )
    if not exact_identity:
        release_policy = {
            "accepted": False,
            "selected_jobs": None,
            "reason": "result_hash_mismatch",
        }
    elif not native_thread_contract:
        release_policy = {
            "accepted": False,
            "selected_jobs": None,
            "reason": "native_thread_contract_failed",
        }
    else:
        release_policy = evaluate_runs(release_runs)
    baseline_wall = summarized["1"]["median_wall_seconds"]
    for jobs in anchor.jobs:
        row = summarized[str(jobs)]
        row["speedup_vs_serial"] = baseline_wall / row["median_wall_seconds"]
        row["parallel_efficiency"] = row["speedup_vs_serial"] / jobs

    return {
        "schema": "homoeogwas-v201-scaling-anchor-v1",
        "anchor": anchor.to_dict(),
        "warmups_per_jobs": 1,
        "measured_repeats_per_jobs": anchor.repeats,
        "exact_result_hash_identity": exact_identity,
        "native_thread_contract_valid": native_thread_contract,
        "result_sha256": hashes[0] if exact_identity and hashes else None,
        "runs": summarized,
        "release_policy": release_policy,
    }
