"""Track C scaling matrix and repetition policy."""

import pytest

from scripts.benchmarks.v201 import track_scaling
from scripts.benchmarks.v201.contracts import ScalingAnchor
from scripts.benchmarks.v201.track_scaling import (
    ScalingAnchorRun,
    run_anchor,
    scaling_anchors,
    summarize_anchor,
)


def test_scaling_anchors_match_the_approved_design():
    anchors = scaling_anchors()

    assert [
        (anchor.name, anchor.n, anchor.groups, anchor.copies, anchor.responses)
        for anchor in anchors
    ] == [
        ("small_qa", 192, 192, 3, 199),
        ("cotton_like", 419, 500, 2, 999),
        ("wheat_formal", 827, 2143, 3, 2000),
        ("quartet_stress", 500, 500, 4, 999),
    ]


def _record(jobs, measured_index):
    pid = 10_000 + jobs
    return {
        "jobs": jobs,
        "effective_jobs": jobs,
        "backend": "fork_shared_memory" if jobs > 1 else "serial",
        "worker_pids": list(range(pid, pid + jobs)),
        "wall_seconds": float(20 / jobs + measured_index),
        "cpu_seconds": float(10 + measured_index),
        "cpu_percent": float(100 * jobs),
        "peak_rss_bytes": 1_000 + measured_index,
        "peak_aggregate_rss_bytes": 2_000 + measured_index,
        "max_process_threads_by_pid": {str(pid): 1},
        "numeric_threadpool_max_threads": 1,
        "numeric_threadpool_info": [{
            "user_api": "blas",
            "internal_api": "openblas",
            "prefix": "libopenblas",
            "num_threads": 1,
        }],
        "runtime_oversubscription_guard_passed": True,
        "result_sha256": "a" * 64,
        "family_sha256": "f" * 64,
        "ranking_sha256": "e" * 64,
        "numeric_thread_env": {
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        },
        "command": ["python", "benchmark_omnib_parallel.py"],
    }


def test_run_anchor_discards_one_warmup_and_keeps_three_repeats(monkeypatch):
    anchor = ScalingAnchor("tiny", 48, 2, 2, 1, 2, (1, 4))
    calls = []

    def fake_subprocess(jobs, args):
        assert (args.n, args.groups, args.responses, args.copies) == (48, 2, 2, 2)
        calls.append(jobs)
        return _record(jobs, calls.count(jobs) - 1)

    monkeypatch.setattr(track_scaling, "_run_subprocess", fake_subprocess)

    runs = run_anchor(anchor)

    assert calls == [1, 1, 1, 1, 4, 4, 4, 4]
    assert [(run.jobs, run.repeat) for run in runs] == [
        (1, 0), (1, 1), (1, 2), (4, 0), (4, 1), (4, 2),
    ]
    assert all(run.numeric_thread_limit_ok for run in runs)


def test_summarize_anchor_reports_medians_and_exact_hash_identity():
    anchor = ScalingAnchor("tiny", 48, 2, 2, 1, 2, (1, 4))
    records = []
    for jobs in anchor.jobs:
        for repeat, wall in enumerate((3.0, 1.0, 2.0)):
            record = _record(jobs, repeat)
            record.update({
                "wall_seconds": wall / jobs,
                "cpu_seconds": (9.0, 3.0, 6.0)[repeat],
                "peak_rss_bytes": (300, 100, 200)[repeat],
                "peak_aggregate_rss_bytes": (600, 200, 400)[repeat],
            })
            records.append(ScalingAnchorRun.from_record(anchor.name, repeat, record))

    summary = summarize_anchor(anchor, records)

    assert summary["runs"]["1"]["median_wall_seconds"] == 2.0
    assert summary["runs"]["4"]["median_wall_seconds"] == 0.5
    assert summary["runs"]["1"]["median_cpu_seconds"] == 6.0
    assert summary["runs"]["1"]["median_peak_aggregate_pss_bytes"] == 200
    assert summary["runs"]["1"]["median_peak_aggregate_rss_bytes"] == 400
    assert summary["exact_result_hash_identity"] is True
    assert summary["result_sha256"] == "a" * 64
    assert "slope" not in summary
    assert "scaling_law" not in summary


def test_summarize_anchor_detects_any_hash_mismatch():
    anchor = ScalingAnchor("tiny", 48, 2, 2, 1, 2, (1,))
    records = []
    for repeat in range(3):
        record = _record(1, repeat)
        if repeat == 2:
            record["result_sha256"] = "b" * 64
        records.append(ScalingAnchorRun.from_record(anchor.name, repeat, record))

    summary = summarize_anchor(anchor, records)

    assert summary["exact_result_hash_identity"] is False
    assert summary["result_sha256"] is None


def test_run_anchor_refuses_noncanonical_repeat_count(monkeypatch):
    anchor = ScalingAnchor("tiny", 48, 2, 2, 1, 2, (1,), repeats=2)
    monkeypatch.setattr(
        track_scaling,
        "_run_subprocess",
        lambda *_args, **_kwargs: pytest.fail("subprocess must not start"),
    )

    with pytest.raises(ValueError, match="one pilot or three formal"):
        run_anchor(anchor)


def test_run_anchor_supports_one_repeat_pilot_after_one_warmup(monkeypatch):
    anchor = ScalingAnchor("tiny", 48, 2, 2, 1, 2, (1, 4), repeats=1)
    calls = []

    def fake_subprocess(jobs, _args):
        calls.append(jobs)
        return _record(jobs, calls.count(jobs) - 1)

    monkeypatch.setattr(track_scaling, "_run_subprocess", fake_subprocess)

    runs = run_anchor(anchor)
    summary = summarize_anchor(anchor, runs)

    assert calls == [1, 1, 4, 4]
    assert [(run.jobs, run.repeat) for run in runs] == [(1, 0), (4, 0)]
    assert summary["warmups_per_jobs"] == 1
    assert summary["measured_repeats_per_jobs"] == 1


def test_thread_limit_failure_cannot_pass_release_policy():
    anchor = ScalingAnchor("small_qa", 48, 2, 2, 1, 2, (1, 4))
    records = []
    for jobs in anchor.jobs:
        for repeat in range(3):
            record = _record(jobs, repeat)
            if jobs == 4 and repeat == 1:
                record["numeric_thread_env"]["OPENBLAS_NUM_THREADS"] = "8"
            records.append(ScalingAnchorRun.from_record(anchor.name, repeat, record))

    summary = summarize_anchor(anchor, records)

    assert summary["native_thread_contract_valid"] is False
    assert summary["release_policy"]["accepted"] is False
    assert summary["release_policy"]["reason"] == "native_thread_contract_failed"


def test_process_management_threads_are_not_numeric_threadpools():
    anchor = ScalingAnchor("tiny", 48, 2, 2, 1, 2, (1,))
    records = []
    for repeat in range(3):
        record = _record(1, repeat)
        record["max_process_threads_by_pid"] = {"10001": 7}
        records.append(ScalingAnchorRun.from_record(anchor.name, repeat, record))

    summary = summarize_anchor(anchor, records)

    assert summary["native_thread_contract_valid"] is True
    assert summary["runs"]["1"]["max_process_threads"] == 7


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ({"backend": "serial"}, "mixed_or_nonfork_backend"),
        ({"effective_jobs": 2}, "mixed_or_reduced_effective_jobs"),
        ({"worker_pids": [10004]}, "worker_pid_count_mismatch"),
    ],
)
def test_parallel_claims_require_consistent_real_fork_repeats(mutation, reason):
    anchor = ScalingAnchor("tiny", 48, 2, 2, 1, 2, (1, 4))
    records = []
    for jobs in anchor.jobs:
        for repeat in range(3):
            record = _record(jobs, repeat)
            if jobs == 4 and repeat == 1:
                record.update(mutation)
            records.append(ScalingAnchorRun.from_record(anchor.name, repeat, record))

    summary = summarize_anchor(anchor, records)
    parallel = summary["runs"]["4"]

    assert parallel["parallel_comparable"] is False
    assert reason in parallel["parallel_exclusion_reasons"]
    assert parallel["speedup_vs_serial"] is None
    assert parallel["parallel_efficiency"] is None
    assert summary["parallel_execution_contract_valid"] is False


def test_release_speed_pss_gate_is_not_applied_to_larger_anchors(monkeypatch):
    anchor = ScalingAnchor("cotton_like", 48, 2, 2, 1, 2, (1, 4))
    records = [
        ScalingAnchorRun.from_record(anchor.name, repeat, _record(jobs, repeat))
        for jobs in anchor.jobs
        for repeat in range(3)
    ]
    monkeypatch.setattr(
        track_scaling,
        "evaluate_runs",
        lambda _runs: pytest.fail("large anchors must not call release gate"),
    )

    summary = summarize_anchor(anchor, records)

    assert summary["release_policy"] == {
        "status": "not_applicable",
        "accepted": None,
        "selected_jobs": None,
        "reason": "small_qa_only_release_gate",
    }


def test_small_qa_parallel_fallback_fails_release_gate():
    anchor = ScalingAnchor("small_qa", 48, 2, 2, 1, 2, (1, 4))
    records = []
    for jobs in anchor.jobs:
        for repeat in range(3):
            record = _record(jobs, repeat)
            if jobs == 4:
                record.update({
                    "backend": "serial",
                    "effective_jobs": 1,
                    "worker_pids": [10004],
                })
            records.append(ScalingAnchorRun.from_record(anchor.name, repeat, record))

    summary = summarize_anchor(anchor, records)

    assert summary["release_policy"]["status"] == "applied"
    assert summary["release_policy"]["accepted"] is False
    assert summary["release_policy"]["reason"] == (
        "parallel_execution_contract_failed"
    )


@pytest.mark.parametrize("hash_field", ["family_sha256", "ranking_sha256"])
def test_family_and_ranking_hashes_must_match_every_repeat_and_jobs(hash_field):
    anchor = ScalingAnchor("small_qa", 48, 2, 2, 1, 2, (1, 4))
    records = []
    for jobs in anchor.jobs:
        for repeat in range(3):
            record = _record(jobs, repeat)
            if jobs == 4 and repeat == 2:
                record[hash_field] = "b" * 64
            records.append(ScalingAnchorRun.from_record(anchor.name, repeat, record))

    summary = summarize_anchor(anchor, records)

    assert summary["exact_family_ranking_identity"] is False
    assert summary[hash_field] is None
    assert summary["release_policy"]["accepted"] is False
    assert summary["release_policy"]["reason"] == (
        "family_or_ranking_hash_mismatch"
    )
