"""Release policy for canonical omniB process scaling."""

from __future__ import annotations

import importlib.util
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from homoeogwas.group_family import MasterGroupFamily, expand_pair_edges

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


def _ranking_family(group_ids=("group_z", "group_a", "group_m", "group_b")):
    family = MasterGroupFamily(
        subgenomes=("A", "B"),
        group_ids=group_ids,
        genes=tuple((f"a{index}", f"b{index}") for index in range(4)),
    )
    return family, expand_pair_edges(family)


def _ranking_identity(family, expanded, edge_p, group_p, components):
    return BENCHMARK._ordered_ranking_identity(
        family, expanded, edge_p, group_p, components, edge_p.shape[1],
    )


def _ranking_hash(family, expanded, edge_p, group_p, components):
    return BENCHMARK._ordered_ranking_hash(
        family, expanded, edge_p, group_p, components, edge_p.shape[1],
    )


def test_ranking_hash_changes_when_edge_and_group_orders_reverse():
    family, expanded = _ranking_family()
    ascending = np.array([[0.1], [0.2], [0.3], [0.4]])
    descending = ascending[::-1].copy()
    components = np.repeat(ascending[:, None, :], 3, axis=1)

    first = _ranking_hash(
        family, expanded, ascending, ascending, components,
    )
    reversed_order = _ranking_hash(
        family, expanded, descending, descending, components[::-1],
    )

    assert first != reversed_order


def test_ranking_ties_are_broken_by_stable_identifier():
    family, expanded = _ranking_family()
    tied = np.full((4, 1), 0.5)
    components = np.repeat(tied[:, None, :], 3, axis=1)

    identity = _ranking_identity(family, expanded, tied, tied, components)

    assert identity["edge"]["responses"][0]["finite_ordered_ids"] == sorted(
        edge.edge_id for edge in expanded.edges
    )
    assert identity["group"]["responses"][0]["finite_ordered_ids"] == sorted(
        family.group_ids
    )


def test_ranking_nonfinite_values_have_explicit_deterministic_partitions():
    family, expanded = _ranking_family()
    values = np.array([[np.nan], [np.inf], [-np.inf], [0.2]])
    components = np.repeat(values[:, None, :], 3, axis=1)

    identity = _ranking_identity(family, expanded, values, values, components)
    edge = identity["edge"]["responses"][0]
    again = _ranking_identity(
        family,
        expanded,
        np.array(values, order="F"),
        np.array(values, order="F"),
        np.array(components, order="F"),
    )
    two_response_values = np.column_stack((values[:, 0], values[::-1, 0]))
    two_response_components = np.repeat(
        two_response_values[:, None, :], 3, axis=1,
    )
    fortran_values = np.asfortranarray(two_response_values)
    fortran_components = np.asfortranarray(two_response_components)
    assert fortran_values.flags.f_contiguous and not fortran_values.flags.c_contiguous
    assert (
        fortran_components.flags.f_contiguous
        and not fortran_components.flags.c_contiguous
    )
    original_hash = _ranking_hash(
        family,
        expanded,
        two_response_values,
        two_response_values,
        two_response_components,
    )
    layout_changed_hash = _ranking_hash(
        family,
        expanded,
        fortran_values,
        fortran_values,
        fortran_components,
    )

    assert edge["nonfinite_ids"] == {
        "negative_infinity": [expanded.edges[2].edge_id],
        "positive_infinity": [expanded.edges[1].edge_id],
        "nan": [expanded.edges[0].edge_id],
    }
    assert edge["ordered_ids"] == [
        expanded.edges[2].edge_id,
        expanded.edges[3].edge_id,
        expanded.edges[1].edge_id,
        expanded.edges[0].edge_id,
    ]
    assert identity == again
    assert original_hash == layout_changed_hash


def test_ranking_hash_ignores_values_when_order_is_same_but_array_hash_does_not():
    family, expanded = _ranking_family()
    first = np.array([[0.1], [0.2], [0.3], [0.4]])
    changed = np.array([[0.01], [0.25], [0.7], [0.9]])
    first_components = np.repeat(first[:, None, :], 3, axis=1)
    changed_components = np.repeat(changed[:, None, :], 3, axis=1)

    first_ranking = _ranking_hash(
        family, expanded, first, first, first_components,
    )
    changed_ranking = _ranking_hash(
        family, expanded, changed, changed, changed_components,
    )

    assert first_ranking == changed_ranking
    assert BENCHMARK._array_hash(first, first, first_components) != (
        BENCHMARK._array_hash(changed, changed, changed_components)
    )
