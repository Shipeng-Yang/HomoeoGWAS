"""Observable process-parallel execution contracts."""

from __future__ import annotations

import multiprocessing as mp
import os

import pytest

from homoeogwas.parallel import ParallelBlockError, run_fork_blocks

_TEST_STATE = None


def _set_state() -> None:
    global _TEST_STATE
    _TEST_STATE = {"barrier": mp.get_context("fork").Barrier(2)}


def _clear_state() -> None:
    global _TEST_STATE
    _TEST_STATE = None


def _get_state():
    return _TEST_STATE


def _identity_worker(block: int) -> int:
    assert _TEST_STATE is not None
    if block < 2:
        _TEST_STATE["barrier"].wait(timeout=10)
    return block


def _serial_worker(block: int) -> int:
    assert _TEST_STATE is not None
    return block * 2


def _failing_worker(block: int) -> int:
    assert _TEST_STATE is not None
    if block == 3:
        raise ValueError("deliberate worker failure")
    return block


def test_fork_runner_observes_multiple_worker_processes():
    results, execution = run_fork_blocks(
        list(range(16)), _identity_worker, n_jobs=2,
        state_setter=_set_state, state_clearer=_clear_state)

    assert results == list(range(16))
    assert execution.backend == "fork_shared_memory"
    assert execution.process_model == "processes"
    assert execution.effective_jobs == 2
    assert execution.inner_threads == 1
    assert execution.parent_pid == os.getpid()
    assert len(set(execution.worker_pids)) == 2
    assert os.getpid() not in execution.worker_pids
    assert _get_state() is None


def test_serial_runner_uses_parent_and_same_worker():
    results, execution = run_fork_blocks(
        list(range(4)), _serial_worker, n_jobs=1,
        state_setter=_set_state, state_clearer=_clear_state)

    assert results == [0, 2, 4, 6]
    assert execution.backend == "serial"
    assert execution.process_model == "serial"
    assert execution.effective_jobs == 1
    assert execution.worker_pids == (os.getpid(),)
    assert _get_state() is None


def test_worker_exception_is_raised_and_state_is_cleared():
    with pytest.raises(ParallelBlockError, match=r"block 3.*deliberate worker failure"):
        run_fork_blocks(
            list(range(8)), _failing_worker, n_jobs=2,
            state_setter=_set_state, state_clearer=_clear_state)

    assert _get_state() is None


@pytest.mark.parametrize("n_jobs", [True, 0, -1, 1.5])
def test_invalid_job_count_is_rejected(n_jobs):
    with pytest.raises(ValueError, match="n_jobs must be an integer >= 1"):
        run_fork_blocks(
            [0], _serial_worker, n_jobs=n_jobs,
            state_setter=_set_state, state_clearer=_clear_state)

