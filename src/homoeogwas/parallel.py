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


class ForkBlockPool:
    """One forked worker pool reused across ordered block batches.

    Large numerical state is installed by ``state_setter`` before the pool is
    forked. Children inherit it through copy-on-write; queue messages contain
    only the supplied block objects.
    """

    def __init__(
        self,
        worker: Callable[[Any], Any],
        *,
        n_jobs: int,
        max_blocks: int,
        state_setter: Callable[[], None],
        state_clearer: Callable[[], None],
    ) -> None:
        self._requested_jobs = _validate_n_jobs(n_jobs)
        if (
            isinstance(max_blocks, bool)
            or not isinstance(max_blocks, int)
            or max_blocks < 1
        ):
            raise ValueError("max_blocks must be an integer >= 1")
        self._worker = worker
        self._state_setter = state_setter
        self._state_clearer = state_clearer
        self._parent_pid = os.getpid()
        cpu_count = os.cpu_count() or 1
        self._effective_jobs = min(self._requested_jobs, max_blocks, cpu_count)
        fork_available = (
            os.name == "posix" and "fork" in mp.get_all_start_methods())
        self._use_processes = self._effective_jobs > 1 and fork_available
        self._fallback_reason = None
        if self._requested_jobs > 1 and not self._use_processes:
            self._fallback_reason = (
                "fork_unavailable" if not fork_available
                else "effective_jobs_bounded_to_one"
            )
        self._pool = None
        self._previous_worker: Callable[[Any], Any] | None = None
        self._worker_pids: set[int] = set()

    def __enter__(self) -> ForkBlockPool:
        global _ACTIVE_WORKER
        self._previous_worker = _ACTIVE_WORKER
        self._state_setter()
        _ACTIVE_WORKER = self._worker
        if self._use_processes:
            context = mp.get_context("fork")
            self._pool = context.Pool(processes=self._effective_jobs)
        return self

    def __exit__(self, *_exc) -> None:
        global _ACTIVE_WORKER
        try:
            if self._pool is not None:
                self._pool.terminate()
        finally:
            self._pool = None
            _ACTIVE_WORKER = self._previous_worker
            self._state_clearer()

    def imap(self, blocks: Iterable[Any]):
        """Yield block values in submission order as they complete."""
        indexed = list(enumerate(blocks))
        if self._pool is not None:
            records = self._pool.imap(_run_indexed_block, indexed, chunksize=1)
        else:
            records = (_run_indexed_block(item) for item in indexed)
        for index, pid, ok, error_type, message, value in records:
            if not ok:
                raise ParallelBlockError(
                    f"parallel block {index} failed with {error_type}: {message}")
            self._worker_pids.add(int(pid))
            yield value

    def map(self, blocks: Iterable[Any]) -> list[Any]:
        return list(self.imap(blocks))

    @property
    def execution(self) -> ParallelExecution:
        return ParallelExecution(
            requested_jobs=self._requested_jobs,
            effective_jobs=self._effective_jobs if self._use_processes else 1,
            backend="fork_shared_memory" if self._use_processes else "serial",
            process_model="processes" if self._use_processes else "serial",
            inner_threads=1,
            parent_pid=self._parent_pid,
            worker_pids=tuple(sorted(self._worker_pids)),
            fallback_reason=self._fallback_reason,
        )


def run_fork_blocks(
    blocks: Iterable[Any],
    worker: Callable[[Any], Any],
    *,
    n_jobs: int,
    state_setter: Callable[[], None],
    state_clearer: Callable[[], None],
) -> tuple[list[Any], ParallelExecution]:
    """Run ordered blocks serially or in POSIX fork workers."""
    requested_jobs = _validate_n_jobs(n_jobs)
    indexed = list(blocks)
    if not indexed:
        return [], ParallelExecution(
            requested_jobs=requested_jobs,
            effective_jobs=0,
            backend="serial",
            process_model="serial",
            inner_threads=1,
            parent_pid=os.getpid(),
            worker_pids=(),
            fallback_reason="no_blocks",
        )
    with ForkBlockPool(
        worker,
        n_jobs=requested_jobs,
        max_blocks=len(indexed),
        state_setter=state_setter,
        state_clearer=state_clearer,
    ) as pool:
        values = pool.map(indexed)
        return values, pool.execution
