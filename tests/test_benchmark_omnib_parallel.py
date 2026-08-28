"""Release policy for canonical omniB process scaling."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

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
