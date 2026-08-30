"""Release policy for canonical omniB process scaling."""

from __future__ import annotations

import importlib.util
import sys
from argparse import Namespace
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "benchmark_omnib_parallel.py"
SPEC = importlib.util.spec_from_file_location("benchmark_omnib_parallel", SCRIPT)
BENCHMARK = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = BENCHMARK
SPEC.loader.exec_module(BENCHMARK)


def _fixed_runs():
    return {
        1: {
            "jobs": 1, "effective_jobs": 1, "worker_pids": [101],
            "wall_seconds": 100.0, "cpu_percent": 99.0,
            "peak_rss_bytes": 10_000, "result_sha256": "a" * 64,
        },
        4: {
            "jobs": 4, "effective_jobs": 4,
            "worker_pids": [201, 202, 203, 204],
            "wall_seconds": 35.0, "cpu_percent": 350.0,
            "peak_rss_bytes": 12_000, "result_sha256": "a" * 64,
        },
        8: {
            "jobs": 8, "effective_jobs": 8,
            "worker_pids": list(range(301, 309)),
            "wall_seconds": 37.0, "cpu_percent": 620.0,
            "peak_rss_bytes": 13_000, "result_sha256": "a" * 64,
        },
    }


def test_acceptance_requires_identity_cpu_speed_and_memory():
    report = BENCHMARK.evaluate_runs(_fixed_runs())

    assert report["accepted"] is True
    assert report["selected_jobs"] == 4
    assert report["runs"]["4"]["result_identity"] is True
    assert report["runs"]["4"]["cpu_gate"] is True
    assert report["runs"]["4"]["oversubscription_gate"] is True
    assert report["runs"]["4"]["memory_gate"] is True


def test_memory_ratio_above_gate_is_rejected():
    runs = _fixed_runs()
    runs[4]["peak_rss_bytes"] = 13_501

    report = BENCHMARK.evaluate_runs(runs)

    assert report["runs"]["4"]["memory_gate"] is False
    assert report["accepted"] is False
    assert report["selected_jobs"] is None


def test_hash_mismatch_rejects_parallel_candidate():
    runs = _fixed_runs()
    runs[4]["result_sha256"] = "b" * 64
    runs[8]["result_sha256"] = "c" * 64

    report = BENCHMARK.evaluate_runs(runs)

    assert report["accepted"] is False
    assert report["selected_jobs"] is None


def test_missing_worker_pid_rejects_candidate():
    runs = _fixed_runs()
    runs[4]["worker_pids"] = [201]

    report = BENCHMARK.evaluate_runs(runs)

    assert report["runs"]["4"]["worker_pid_gate"] is False
    assert report["accepted"] is False
    assert report["selected_jobs"] is None


def test_native_thread_oversubscription_rejects_candidate():
    runs = _fixed_runs()
    runs[4]["cpu_percent"] = 700.0

    report = BENCHMARK.evaluate_runs(runs)

    assert report["runs"]["4"]["oversubscription_gate"] is False
    assert report["accepted"] is False


@pytest.mark.parametrize(("copies", "edges"), [(2, 2), (3, 6), (4, 12)])
def test_child_fixture_expands_all_pair_edges(copies, edges):
    record = BENCHMARK._child_run(
        1, n=48, groups=2, responses=2, copies=copies,
    )

    assert record["fixture"]["copies"] == copies
    assert record["fixture"]["unique_edges"] == edges
    assert len(record["family_sha256"]) == 64
    assert len(record["ranking_sha256"]) == 64


def test_child_fixture_rejects_unsupported_copy_counts():
    with pytest.raises(ValueError, match="copies must be 2, 3, or 4"):
        BENCHMARK._child_run(1, n=48, groups=2, responses=2, copies=5)


def test_release_fixture_keeps_three_copy_default():
    args = BENCHMARK.build_parser().parse_args([])

    assert args.copies == 3


def test_run_subprocess_observes_real_numeric_threadpools_and_process_tree():
    record = BENCHMARK._run_subprocess(
        1, Namespace(n=48, groups=2, responses=2, copies=4),
    )

    assert record["numeric_thread_env"] == {
        name: "1" for name in BENCHMARK.NUMERIC_THREAD_ENV
    }
    assert record["runtime_oversubscription_guard_passed"] is True
    assert record["numeric_threadpool_info"]
    assert record["numeric_threadpool_max_threads"] <= 1
    assert record["max_process_threads_by_pid"]
    assert record["peak_rss_bytes"] > 0
    assert record["peak_aggregate_rss_bytes"] > 0
    assert record["fixture"]["copies"] == 4
    assert record["fixture"]["unique_edges"] == 12
