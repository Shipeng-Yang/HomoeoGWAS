from __future__ import annotations

import copy
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
    return {
        "schema": "homoeogwas-snpxsnp-resource-probe-v1",
        "panel_id": panel_id,
        "sample_context": "full",
        "family_size": 80,
        "copies": copies,
        "response_width": width,
        "design_hash": "a" * 64,
        "context_fingerprint": "b" * 64,
        "prepared_design_sha256": "c" * 64,
        "member_family_sha256": "d" * 64,
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
        "inference_status": "noninferential_resource_probe",
        "execution_authorized": False,
    }


def test_probe_schema_rejects_a_missing_measurement_field():
    payload = _probe(1)
    payload.pop("peak_aggregate_pss_bytes")
    with pytest.raises(ValueError, match="schema fields"):
        resource_contracts.ComparatorProbeRecord.from_payload(payload)


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
        resource_contracts.ComparatorProbeRecord.from_payload(payload)


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

    memory = [_probe(width, parent_rss_gib=20) for width in (1, 5, 20)]
    with pytest.raises(ValueError, match="safety-adjusted parent RSS"):
        resource_contracts.validate_comparator_probe_series(memory)


def test_probe_series_rejects_a_detached_design_or_pair_family():
    records = [_probe(width) for width in (1, 5, 20)]
    records[-1] = copy.deepcopy(records[-1])
    records[-1]["member_family_sha256"] = "e" * 64
    with pytest.raises(ValueError, match="same frozen context"):
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
