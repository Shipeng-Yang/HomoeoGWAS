from __future__ import annotations

import copy
import os
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest

from scripts.benchmarks.v201 import contracts as resource_contracts
from scripts.benchmarks.v201 import (
    track_omnib,
)

GIB = 1024 ** 3


def _probe(
    width: int,
    *,
    panel_id: str = "REALG.CGVD1245",
    copies: int = 2,
    offered: int = 7_405,
    parent_rss_gib: int = 8,
    aggregate_pss_gib: int = 16,
) -> dict[str, object]:
    response_ids = [f"response-{index:02d}" for index in range(20)]
    return {
        "schema": "homoeogwas-snpxsnp-resource-probe-v2",
        "panel_id": panel_id,
        "sample_context": "full",
        "family_size": 80,
        "copies": copies,
        "response_width": width,
        "design_hash": "a" * 64,
        "context_fingerprint": "b" * 64,
        "prepared_design_sha256": "c" * 64,
        "member_family_sha256": "d" * 64,
        "probe_authorization_sha256": "0" * 64,
        "implementation_commit": "1" * 40,
        "matched_comparator_contract_sha256": "2" * 64,
        "input_family_sha256": "7" * 64,
        "response_bank_sha256": "3" * 64,
        "response_ids": response_ids,
        "response_ids_sha256": resource_contracts.sha256_payload({
            "ordered_response_ids": response_ids,
        }),
        "response_prefix_sha256": resource_contracts.sha256_payload({
            "response_width": width,
        }),
        "score_evidence_sha256": resource_contracts.sha256_payload({
            "score_width": width,
        }),
        "scorer_wall_seconds": float(width),
        "scorer_cpu_seconds": float(2 * width),
        "peak_parent_rss_bytes": parent_rss_gib * GIB,
        "peak_aggregate_pss_bytes": aggregate_pss_gib * GIB,
        "output_bytes": 100 * width,
        "offered_pair_count": offered,
        "design_nonestimable_pair_count": 5,
        "tested_pair_count": offered - 5,
        "nonfinite_pair_score_count": 0,
        "failed_response_indices": [],
        "gated_marker_count_by_gene": {"A|g1": 11, "D|g1": 9},
        "requested_jobs": 1,
        "effective_jobs": 1,
        "parallel_backend": "serial",
        "worker_pids": [12345],
        "root_pid": 12345,
        "sampled_pids": [12345],
        "process_set_reconciled": True,
        "aggregate_pss_missed_spike_strategy": (
            "serial_singleton_parent_rss_upper_bound"
        ),
        "inference_status": "noninferential_resource_probe",
        "execution_authorized": False,
        "formal_execution_authorized": False,
    }


def test_probe_schema_rejects_a_missing_measurement_field():
    payload = _probe(1)
    payload.pop("peak_aggregate_pss_bytes")
    with pytest.raises(ValueError, match="schema fields"):
        resource_contracts.ComparatorProbeRecordV2.from_payload(payload)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("offered_pair_count", 10_001, "offered pair ceiling"),
        ("peak_parent_rss_bytes", 32 * GIB + 1, "parent RSS ceiling"),
        ("peak_aggregate_pss_bytes", 128 * GIB + 1, "aggregate PSS ceiling"),
    ],
)
def test_probe_rejects_raw_resource_ceiling_exceedance(field, value, message):
    payload = _probe(1)
    payload[field] = value
    if field == "offered_pair_count":
        payload["tested_pair_count"] = value - payload[
            "design_nonestimable_pair_count"
        ]
    with pytest.raises(ValueError, match=message):
        resource_contracts.ComparatorProbeRecordV2.from_payload(payload)


@pytest.mark.parametrize(
    ("panel_id", "family_size", "copies"),
    [
        ("SYNTH.QUARTET", 80, 4),
        ("REALG.CGVD1245", 500, 2),
        ("REALG.WATKINS_F2143", 2_000, 3),
    ],
)
def test_unfrozen_comparator_contexts_are_rejected(panel_id, family_size, copies):
    with pytest.raises(ValueError, match="unfrozen"):
        resource_contracts.comparator_resource_limit(
            panel_id, family_size=family_size, copies=copies,
        )


def test_probe_series_requires_widths_1_5_20_and_projects_with_safety_factor_two():
    records = [_probe(width) for width in (1, 5, 20)]
    projection = resource_contracts.validate_comparator_probe_series(records)

    assert projection["response_widths"] == [1, 5, 20]
    assert projection["target_response_count"] == 2_000
    assert projection["safety_factor"] == 2.0
    assert projection["projected"] == {
        "scorer_cpu_seconds": 8_000.0,
        "elapsed_seconds": 4_000.0,
        "output_bytes": 400_000,
        "peak_parent_rss_bytes": 16 * GIB,
        "peak_aggregate_pss_bytes": 32 * GIB,
    }
    assert projection["accepted"] is True

    incomplete = records[:2]
    with pytest.raises(ValueError, match="widths 1, 5 and 20"):
        resource_contracts.validate_comparator_probe_series(incomplete)


def test_probe_series_rejects_time_nonlinearity_and_safety_adjusted_memory():
    nonlinear = [_probe(width) for width in (1, 5, 20)]
    nonlinear[-1]["scorer_wall_seconds"] = 25.0
    with pytest.raises(ValueError, match="time per response"):
        resource_contracts.validate_comparator_probe_series(nonlinear)

    memory = [
        _probe(width, parent_rss_gib=20, aggregate_pss_gib=24)
        for width in (1, 5, 20)
    ]
    with pytest.raises(ValueError, match="safety-adjusted parent RSS"):
        resource_contracts.validate_comparator_probe_series(memory)


def test_probe_series_allows_fixed_cost_amortization_across_response_widths():
    records = [_probe(width) for width in (1, 5, 20)]
    records[0]["scorer_wall_seconds"] = 6.0
    records[1]["scorer_wall_seconds"] = 10.0
    records[2]["scorer_wall_seconds"] = 20.0

    projection = resource_contracts.validate_comparator_probe_series(records)

    assert projection["time_per_response_growth_5_to_20"] == pytest.approx(-0.5)
    assert projection["time_amortization_observed"] is True


def test_probe_series_projects_width_varying_memory_with_upper_envelope():
    records = [_probe(width) for width in (1, 5, 20)]
    base_parent = 1 * GIB
    parent_slope = 1 * 1024 ** 2
    base_pss = 2 * GIB
    pss_slope = 2 * 1024 ** 2
    for record in records:
        width = int(record["response_width"])
        record["peak_parent_rss_bytes"] = base_parent + parent_slope * width
        record["peak_aggregate_pss_bytes"] = base_pss + pss_slope * width

    projection = resource_contracts.validate_comparator_probe_series(records)

    assert projection["projected"]["peak_parent_rss_bytes"] == 2 * (
        base_parent + 2_000 * parent_slope
    )
    assert projection["projected"]["peak_aggregate_pss_bytes"] == 2 * (
        base_pss + 2_000 * pss_slope
    )
    assert projection["memory_projection_models"]["peak_parent_rss_bytes"] == {
        "anchor_bytes_by_width": {
            "1": base_parent + parent_slope,
            "5": base_parent + 5 * parent_slope,
            "20": base_parent + 20 * parent_slope,
        },
        "intercept_bytes": base_parent,
        "slope_bytes_per_response": parent_slope,
        "target_without_safety_bytes": base_parent + 2_000 * parent_slope,
    }


@pytest.mark.parametrize(
    "field",
    ["peak_parent_rss_bytes", "peak_aggregate_pss_bytes"],
)
def test_probe_series_rejects_nonmonotone_memory_anchors(field):
    records = [_probe(width) for width in (1, 5, 20)]
    if field == "peak_aggregate_pss_bytes":
        for record in records:
            record["peak_parent_rss_bytes"] = 1 * GIB
    records[0][field] = 4 * GIB
    records[1][field] = 3 * GIB
    records[2][field] = 5 * GIB
    with pytest.raises(ValueError, match="nonmonotone.*memory"):
        resource_contracts.validate_comparator_probe_series(records)


def test_probe_series_rejects_a_detached_design_or_pair_family():
    records = [_probe(width) for width in (1, 5, 20)]
    records[-1] = copy.deepcopy(records[-1])
    records[-1]["member_family_sha256"] = "e" * 64
    with pytest.raises(ValueError, match="same frozen context"):
        resource_contracts.validate_comparator_probe_series(records)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("root_pid", 54321),
        ("sampled_pids", [12345, 54321]),
        ("worker_pids", [54321]),
        ("process_set_reconciled", False),
        ("aggregate_pss_missed_spike_strategy", "sampled_only"),
        ("formal_execution_authorized", True),
    ],
)
def test_probe_schema_rejects_unreconciled_process_or_role_evidence(field, value):
    payload = _probe(1)
    payload[field] = value
    with pytest.raises(ValueError, match="execution provenance|identity or role"):
        resource_contracts.ComparatorProbeRecordV2.from_payload(payload)


@pytest.mark.parametrize(
    "field",
    [
        "probe_authorization_sha256", "input_family_sha256",
        "response_bank_sha256", "response_ids",
    ],
)
def test_probe_series_rejects_detached_authorization_or_response_bank(field):
    records = [_probe(width) for width in (1, 5, 20)]
    if field == "response_ids":
        records[-1][field] = [*records[-1][field][:-1], "changed-response"]
        records[-1]["response_ids_sha256"] = resource_contracts.sha256_payload({
            "ordered_response_ids": records[-1][field],
        })
    else:
        records[-1][field] = "6" * 64
    with pytest.raises(ValueError, match="same frozen context"):
        resource_contracts.validate_comparator_probe_series(records)


def test_probe_series_rejects_reused_prefix_or_score_evidence():
    for field in ("response_prefix_sha256", "score_evidence_sha256"):
        records = [_probe(width) for width in (1, 5, 20)]
        records[-1][field] = records[0][field]
        with pytest.raises(ValueError, match="distinct width-prefix evidence"):
            resource_contracts.validate_comparator_probe_series(records)


@pytest.mark.parametrize(
    ("panel_id", "copies", "ceiling"),
    [
        ("REALG.CGVD1245", 2, 10_000),
        ("REALG.WATKINS_F2143", 3, 750_000),
    ],
)
def test_track_resolves_the_exact_frozen_pair_ceiling_before_scoring(
    panel_id, copies, ceiling,
):
    context = SimpleNamespace(
        panel_id=panel_id,
        family=SimpleNamespace(
            group_ids=tuple(f"g{index}" for index in range(80)),
            subgenomes=tuple(f"S{index}" for index in range(copies)),
        ),
    )
    assert track_omnib._snpxsnp_pair_ceiling(context) == ceiling


@pytest.mark.parametrize(
    ("panel_id", "family_size", "copies"),
    [
        ("REALG.CGVD1245", 500, 2),
        ("SYNTH.QUARTET", 80, 4),
    ],
)
def test_track_rejects_an_unfrozen_context_before_scoring(
    panel_id, family_size, copies,
):
    context = SimpleNamespace(
        panel_id=panel_id,
        family=SimpleNamespace(
            group_ids=tuple(f"g{index}" for index in range(family_size)),
            subgenomes=tuple(f"S{index}" for index in range(copies)),
        ),
    )
    with pytest.raises(ValueError, match="unfrozen"):
        track_omnib._snpxsnp_pair_ceiling(context)


@pytest.mark.parametrize(
    ("high_water_gib", "expected_parent_rss"),
    [
        ((8, 8), 100),
        ((8, 9), 9 * GIB),
    ],
)
def test_comparator_measurement_gates_high_water_to_operation_window(
    monkeypatch, high_water_gib, expected_parent_rss,
):
    from scripts.benchmarks.v201 import cli

    root_pid = os.getpid()
    monkeypatch.setattr(
        cli,
        "_comparator_memory_snapshot",
        lambda _pid: (100, 200, (root_pid,)),
    )
    high_water = iter(value * GIB for value in high_water_gib)
    monkeypatch.setattr(cli, "_self_peak_rss_bytes", lambda: next(high_water))
    _result, measurement = cli._measure_comparator_operation(
        lambda: "done", sample_interval_seconds=0.001
    )
    assert measurement["peak_parent_rss_bytes"] == expected_parent_rss
    assert measurement["peak_aggregate_pss_bytes"] == max(
        200, expected_parent_rss,
    )
    assert measurement["root_pid"] == root_pid
    assert measurement["sampled_pids"] == [root_pid]
    assert measurement["process_set_reconciled"] is True
    assert measurement["aggregate_pss_missed_spike_strategy"] == (
        "serial_singleton_parent_rss_upper_bound"
    )


def test_comparator_measurement_rejects_a_sampled_descendant_before_scoring(
    monkeypatch,
):
    from scripts.benchmarks.v201 import cli

    root_pid = os.getpid()
    monkeypatch.setattr(
        cli,
        "_comparator_memory_snapshot",
        lambda _pid: (100, 200, (root_pid, root_pid + 1)),
    )
    called = False

    def operation():
        nonlocal called
        called = True

    with pytest.raises(RuntimeError, match="descendant"):
        cli._measure_comparator_operation(operation)
    assert called is False


@pytest.mark.parametrize("declared", [(True,), (12345, 12345), ("12345",)])
def test_comparator_measurement_rejects_invalid_declared_workers_before_scoring(
    declared,
):
    from scripts.benchmarks.v201 import cli

    called = False

    def operation():
        nonlocal called
        called = True

    with pytest.raises(ValueError, match="worker PID|serial comparator"):
        cli._measure_comparator_operation(
            operation, declared_worker_pids=declared,
        )
    assert called is False


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux /proc contract")
def test_process_tree_finds_child_spawned_by_non_main_thread():
    from scripts.benchmarks.v201 import cli

    ready = threading.Event()
    release = threading.Event()
    holder = {}

    def spawn_child():
        process = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"]
        )
        holder["process"] = process
        ready.set()
        release.wait(timeout=10)
        process.terminate()
        process.wait(timeout=10)

    worker = threading.Thread(target=spawn_child)
    worker.start()
    try:
        assert ready.wait(timeout=10)
        assert holder["process"].pid in cli._process_tree(os.getpid())
    finally:
        release.set()
        worker.join(timeout=15)
        if worker.is_alive():
            holder["process"].kill()
            worker.join(timeout=5)
