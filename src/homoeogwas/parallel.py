"""Memory-safe process blocks for large read-only numerical state."""

from __future__ import annotations

import multiprocessing as mp
import os
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from typing import Any

from threadpoolctl import threadpool_limits


@dataclass(frozen=True)
class ParallelExecution:
    """Observable execution details that do not affect statistical identity."""

    requested_jobs: int
    effective_jobs: int
    backend: str
    process_model: str
    inner_threads: int
    parent_pid: int
    worker_pids: tuple[int, ...]
    fallback_reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class ParallelBlockError(RuntimeError):
    """One indexed block failed in a worker process."""


_ACTIVE_WORKER: Callable[[Any], Any] | None = None


def _run_indexed_block(item: tuple[int, Any]) -> tuple:
    index, block = item
    worker = _ACTIVE_WORKER
    if worker is None:
        return index, os.getpid(), False, "RuntimeError", "worker state is not set", None
    try:
        with threadpool_limits(limits=1):
            value = worker(block)
    except Exception as exc:  # returned explicitly so the parent names the block
        return index, os.getpid(), False, type(exc).__name__, str(exc), None
    return index, os.getpid(), True, None, None, value


def _validate_n_jobs(n_jobs: int) -> int:
    if (
        isinstance(n_jobs, bool)
        or not isinstance(n_jobs, int)
        or n_jobs < 1
    ):
        raise ValueError("n_jobs must be an integer >= 1")
    return n_jobs


def run_fork_blocks(
    blocks: Iterable[Any],
    worker: Callable[[Any], Any],
    *,
    n_jobs: int,
    state_setter: Callable[[], None],
    state_clearer: Callable[[], None],
) -> tuple[list[Any], ParallelExecution]:
    """Run ordered blocks serially or in POSIX fork workers.

    Large numerical state is installed by ``state_setter`` before the pool is
    forked. Children inherit it through copy-on-write; queue messages contain
    only the supplied block objects.
    """
    global _ACTIVE_WORKER

    requested_jobs = _validate_n_jobs(n_jobs)
    indexed = list(enumerate(blocks))
    parent_pid = os.getpid()
    if not indexed:
        return [], ParallelExecution(
            requested_jobs=requested_jobs,
            effective_jobs=0,
            backend="serial",
            process_model="serial",
            inner_threads=1,
            parent_pid=parent_pid,
            worker_pids=(),
            fallback_reason="no_blocks",
        )

    cpu_count = os.cpu_count() or 1
    effective_jobs = min(requested_jobs, len(indexed), cpu_count)
    fork_available = os.name == "posix" and "fork" in mp.get_all_start_methods()
    use_processes = effective_jobs > 1 and fork_available
    fallback_reason = None
    if requested_jobs > 1 and not use_processes:
        fallback_reason = (
            "fork_unavailable" if not fork_available
            else "effective_jobs_bounded_to_one"
        )

    previous_worker = _ACTIVE_WORKER
    state_setter()
    _ACTIVE_WORKER = worker
    try:
        if use_processes:
            context = mp.get_context("fork")
            with context.Pool(processes=effective_jobs) as pool:
                records = pool.map(_run_indexed_block, indexed, chunksize=1)
            backend = "fork_shared_memory"
            process_model = "processes"
        else:
            records = [_run_indexed_block(item) for item in indexed]
            backend = "serial"
            process_model = "serial"

        for index, _pid, ok, error_type, message, _value in records:
            if not ok:
                raise ParallelBlockError(
                    f"parallel block {index} failed with {error_type}: {message}")
        records.sort(key=lambda record: record[0])
        values = [record[5] for record in records]
        worker_pids = tuple(sorted({int(record[1]) for record in records}))
        return values, ParallelExecution(
            requested_jobs=requested_jobs,
            effective_jobs=effective_jobs if use_processes else 1,
            backend=backend,
            process_model=process_model,
            inner_threads=1,
            parent_pid=parent_pid,
            worker_pids=worker_pids,
            fallback_reason=fallback_reason,
        )
    finally:
        _ACTIVE_WORKER = previous_worker
        state_clearer()
