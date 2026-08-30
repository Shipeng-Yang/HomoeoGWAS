import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from scripts.benchmarks.v201.contracts import Scenario
from scripts.benchmarks.v201.shards import (
    BudgetExceeded,
    ShardConflict,
    ShardKey,
    expected_shards,
    load_shard,
    project_budget,
    write_shard_exclusive,
)


def test_identical_resume_is_noop_and_conflict_is_rejected(tmp_path):
    path = tmp_path / "replicate-000007.json"
    payload = {"design_hash": "a" * 64, "track": "fit", "scenario_id": "s", "replicate": 7}
    assert write_shard_exclusive(path, payload) == "created"
    assert write_shard_exclusive(path, payload) == "existing_identical"
    with pytest.raises(ShardConflict, match="existing shard differs"):
        write_shard_exclusive(path, {**payload, "replicate": 8})
    assert json.loads(path.read_text()) == payload


def test_concurrent_identical_writers_create_one_immutable_shard(tmp_path):
    path = tmp_path / "replicate-000000.json"
    payload = {"design_hash": "b" * 64, "track": "fit", "scenario_id": "s", "replicate": 0}
    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(pool.map(lambda _: write_shard_exclusive(path, payload), range(8)))
    assert outcomes.count("created") == 1
    assert outcomes.count("existing_identical") == 7
    assert load_shard(path) == payload


def test_load_shard_rejects_payload_that_does_not_match_expected_key(tmp_path):
    path = tmp_path / "replicate-000002.json"
    payload = {"design_hash": "c" * 64, "track": "fit", "scenario_id": "scenario", "replicate": 2}
    write_shard_exclusive(path, payload)
    assert load_shard(path, ShardKey("fit", "scenario", 2)) == payload
    with pytest.raises(ShardConflict, match="shard key mismatch"):
        load_shard(path, ShardKey("fit", "scenario", 3))


def test_load_shard_rejects_missing_or_mismatched_expected_key_track(tmp_path):
    missing_track = tmp_path / "missing-track.json"
    missing_track.write_text(json.dumps({
        "design_hash": "c" * 64, "scenario_id": "scenario", "replicate": 2,
    }))
    with pytest.raises(ValueError, match="track"):
        load_shard(missing_track, ShardKey("fit", "scenario", 2))

    mismatched_track = tmp_path / "mismatched-track.json"
    write_shard_exclusive(mismatched_track, {
        "design_hash": "c" * 64, "track": "omnib", "scenario_id": "scenario", "replicate": 2,
    })
    with pytest.raises(ShardConflict, match="shard key mismatch"):
        load_shard(mismatched_track, ShardKey("fit", "scenario", 2))


@pytest.mark.parametrize(
    "payload",
    [
        {"design_hash": "short", "track": "fit", "scenario_id": "s", "replicate": 0},
        {"design_hash": "a" * 64, "track": "fit", "scenario_id": "", "replicate": 0},
        {"design_hash": "a" * 64, "track": "fit", "scenario_id": "s", "replicate": -1},
        {"design_hash": "a" * 64, "scenario_id": "s", "replicate": 0},
    ],
)
def test_write_shard_rejects_invalid_minimum_contract(tmp_path, payload):
    with pytest.raises(ValueError, match="payload"):
        write_shard_exclusive(tmp_path / "invalid.json", payload)


def test_expected_shards_follow_registry_order_and_zero_based_replicates():
    scenarios = [
        Scenario("first", "fit", "pilot", 2, 0),
        Scenario("second", "omnib", "pilot", 1, 0),
    ]
    assert expected_shards(scenarios) == [
        ShardKey("fit", "first", 0),
        ShardKey("fit", "first", 1),
        ShardKey("omnib", "second", 0),
    ]
    assert ShardKey("fit", "first", 7).relative_path().as_posix() == (
        "fit/first/replicate-000007.json"
    )


def test_project_budget_scales_each_measurement_and_reports_breakdown():
    projection = project_budget(
        "formal",
        [
            {"scenario_id": "a", "cpu_seconds": 3600, "output_bytes": 1_000_000_000,
             "scenario_multiplier": 3},
            {"scenario_id": "b", "cpu_seconds": 1800, "output_bytes": 2_000_000_000,
             "scenario_multiplier": 2},
        ],
        effective_workers=3,
    )
    assert set(projection) == {"stage", "effective_workers", "totals", "limits", "breakdown"}
    assert projection["totals"] == {
        "cpu_seconds": 14_400.0,
        "cpu_hours": 4.0,
        "elapsed_hours": pytest.approx(4 / 3),
        "output_bytes": 7_000_000_000.0,
        "output_gb": 7.0,
    }
    assert json.dumps(projection)
    assert projection["limits"] == {
        "cpu_hours": 25_000,
        "elapsed_hours": 336,
        "output_gb": 500,
    }
    assert projection["breakdown"]["a"] == {
        "scenario_multiplier": 3.0,
        "measured_cpu_seconds": 3600.0,
        "projected_cpu_seconds": 10_800.0,
        "measured_output_bytes": 1_000_000_000.0,
        "projected_output_bytes": 3_000_000_000.0,
        "cpu_hours": 3.0,
        "output_gb": 3.0,
    }
    assert projection["breakdown"]["b"]["scenario_multiplier"] == 2.0
    assert projection["breakdown"]["b"]["cpu_hours"] == 1.0
    assert projection["breakdown"]["b"]["output_gb"] == 4.0


@pytest.mark.parametrize(
    "stage, measurement, message",
    [
        ("pilot", {"cpu_seconds": 1_000 * 3600 + 1, "output_bytes": 0}, "cpu_hours"),
        ("pilot", {"cpu_seconds": 0, "output_bytes": 200_000_000_001}, "output_gb"),
        ("formal", {"cpu_seconds": 25_000 * 3600 + 1, "output_bytes": 0}, "cpu_hours"),
        ("formal", {"cpu_seconds": 0, "output_bytes": 500_000_000_001}, "output_gb"),
    ],
)
def test_project_budget_rejects_cpu_and_storage_overages(stage, measurement, message):
    measurement.update({"scenario_id": "s", "scenario_multiplier": 1})
    with pytest.raises(BudgetExceeded, match=message):
        project_budget(stage, [measurement], effective_workers=1)


@pytest.mark.parametrize("stage, workers", [("pilot", 0), ("formal", -1), ("pilot", True)])
def test_project_budget_rejects_invalid_worker_count(stage, workers):
    with pytest.raises(ValueError, match="effective_workers"):
        project_budget(
            stage,
            [{"scenario_id": "s", "cpu_seconds": 1, "output_bytes": 1, "scenario_multiplier": 1}],
            effective_workers=workers,
        )


def test_project_budget_rejects_elapsed_overage_using_admitted_workers():
    with pytest.raises(BudgetExceeded, match="elapsed_hours"):
        project_budget(
            "pilot",
            [{"scenario_id": "s", "cpu_seconds": 12 * 3600 + 1,
              "output_bytes": 0, "scenario_multiplier": 1}],
            effective_workers=1,
        )
