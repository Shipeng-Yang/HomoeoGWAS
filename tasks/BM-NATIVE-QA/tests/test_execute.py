from __future__ import annotations

from pathlib import Path

import pytest
from bm_native_qa_harness.execute import (
    NATIVE_CAPS,
    CapBreach,
    ResourceSample,
    RuntimeObservation,
    commands_for_invocation,
    evaluate_static_preflight,
    evaluate_successor_static_preflight,
    run_sequence,
)
from bm_native_qa_harness.plan import InvocationSpec


def _invocation() -> InvocationSpec:
    return InvocationSpec(
        invocation_id="qa_test.invocation.jobs4",
        response_id="qa_test.response",
        panel_id="REALG.CGVD1245",
        sample_context="pc1_spread_192",
        truth_id="gaussian_null",
        jobs=4,
    )


def test_commands_are_validate_interact_audit_with_fixed_native_threads(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config.yaml"
    result = tmp_path / "result"
    audit = tmp_path / "audit"

    commands = commands_for_invocation(
        _invocation(),
        executable="/accepted/bin/homoeogwas",
        config_path=config,
        result_dir=result,
        audit_dir=audit,
    )

    assert [command.argv for command in commands] == [
        ("/accepted/bin/homoeogwas", "validate", "-c", str(config)),
        (
            "/accepted/bin/homoeogwas",
            "interact",
            "-c",
            str(config),
            "--n-jobs",
            "4",
        ),
        ("/accepted/bin/homoeogwas", "audit", str(result), "-o", str(audit)),
    ]
    assert [command.purpose for command in commands] == [
        "validate",
        "interact",
        "audit",
    ]
    assert all(
        command.env
        == {
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
        for command in commands
    )


def _native_authority() -> dict[str, object]:
    design_hash = "a" * 64
    return {
        "execution_authorized": True,
        "qa_design_hash": design_hash,
        "prospective_inventory_sha256": "b" * 64,
        "anchor_manifest_sha256": "c" * 64,
        "response_manifest_sha256": "d" * 64,
        "config_manifest_sha256": "e" * 64,
        "source_input_reverification_sha256": "f" * 64,
        "artifact_review_verdict": "ACCEPT",
        "artifact_review_sha256": "1" * 64,
        "static_preflight_status": "PASS",
        "static_preflight_sha256": "2" * 64,
        "anchor_manifest_design_hash": design_hash,
        "response_manifest_design_hash": design_hash,
        "config_manifest_design_hash": design_hash,
    }


@pytest.mark.parametrize(
    "missing_field",
    (
        "qa_design_hash",
        "prospective_inventory_sha256",
        "anchor_manifest_sha256",
        "response_manifest_sha256",
        "config_manifest_sha256",
        "source_input_reverification_sha256",
        "artifact_review_sha256",
        "static_preflight_sha256",
        "anchor_manifest_design_hash",
        "response_manifest_design_hash",
        "config_manifest_design_hash",
    ),
)
def test_native_gate_rejects_missing_evidence_before_executor_call(
    tmp_path: Path,
    missing_field: str,
) -> None:
    commands = commands_for_invocation(
        _invocation(),
        executable="/accepted/bin/homoeogwas",
        config_path=tmp_path / "config.yaml",
        result_dir=tmp_path / "result",
        audit_dir=tmp_path / "audit",
    )
    authority = _native_authority()
    del authority[missing_field]
    calls = []

    with pytest.raises(RuntimeError, match="native execution evidence is incomplete"):
        run_sequence(
            commands,
            authority=authority,
            executor=lambda command: calls.append(command),
            sampler=lambda _handle: (),
            terminator=lambda _handle: None,
        )

    assert calls == []


def test_native_gate_rejects_false_authorization_before_executor_call(
    tmp_path: Path,
) -> None:
    commands = commands_for_invocation(
        _invocation(),
        executable="/accepted/bin/homoeogwas",
        config_path=tmp_path / "config.yaml",
        result_dir=tmp_path / "result",
        audit_dir=tmp_path / "audit",
    )
    authority = _native_authority()
    authority["execution_authorized"] = False
    calls = []

    with pytest.raises(RuntimeError, match="execution_authorized=false"):
        run_sequence(
            commands,
            authority=authority,
            executor=lambda command: calls.append(command),
            sampler=lambda _handle: (),
            terminator=lambda _handle: None,
        )

    assert calls == []


def _static_preflight_record() -> dict[str, object]:
    gib = 1024**3
    return {
        "utc_timestamp": "2026-09-10T08:00:00Z",
        "logical_cpu_count": 64,
        "mem_available_bytes": 128 * gib,
        "output_filesystem_free_bytes": 20 * gib,
        "temporary_filesystem_free_bytes": 30 * gib,
        "output_root_exists": False,
        "output_lock_exists": False,
        "supervisor_executable_present_and_executable": True,
        "supervisor_executable_sha256": "3" * 64,
        "supervisor_configuration_sha256": "4" * 64,
        "configured_thresholds": dict(NATIVE_CAPS),
    }


def test_static_preflight_accepts_exact_floors_without_running_a_probe() -> None:
    result = evaluate_static_preflight(_static_preflight_record())

    assert result.status == "PASS"
    assert len(result.record_sha256) == 64
    assert set(result.record).isdisjoint(
        {"supervisor_self_test", "resource_projection", "timing_workload"}
    )


def test_successor_static_preflight_requires_192_gib_memavailable() -> None:
    record = _static_preflight_record()
    record["mem_available_bytes"] = 191 * 1024**3

    with pytest.raises(RuntimeError, match="below 192 GiB"):
        evaluate_successor_static_preflight(record)

    record["mem_available_bytes"] = 192 * 1024**3
    assert evaluate_successor_static_preflight(record).status == "PASS"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("mem_available_bytes", 128 * 1024**3 - 1, "MemAvailable"),
        ("output_filesystem_free_bytes", 20 * 1024**3 - 1, "output filesystem"),
        ("output_root_exists", True, "output root"),
        ("output_lock_exists", True, "output lock"),
        (
            "supervisor_executable_present_and_executable",
            False,
            "supervisor executable",
        ),
    ),
)
def test_static_preflight_fails_closed_below_each_literal_floor(
    field: str,
    value: object,
    message: str,
) -> None:
    record = _static_preflight_record()
    record[field] = value

    with pytest.raises(RuntimeError, match=message):
        evaluate_static_preflight(record)


def test_static_preflight_rejects_threshold_drift() -> None:
    record = _static_preflight_record()
    thresholds = dict(NATIVE_CAPS)
    thresholds["wall_seconds"] = 7201
    record["configured_thresholds"] = thresholds

    with pytest.raises(RuntimeError, match="thresholds"):
        evaluate_static_preflight(record)


def test_static_preflight_rejects_every_unknown_field() -> None:
    record = _static_preflight_record()
    record["estimated_runtime_seconds"] = 123

    with pytest.raises(RuntimeError, match="unknown fields.*estimated_runtime_seconds"):
        evaluate_static_preflight(record)


def _commands(tmp_path: Path):
    return commands_for_invocation(
        _invocation(),
        executable="/accepted/bin/homoeogwas",
        config_path=tmp_path / "config.yaml",
        result_dir=tmp_path / "result",
        audit_dir=tmp_path / "audit",
    )


@pytest.mark.parametrize(
    ("sample", "dimension"),
    (
        (ResourceSample(7200.1, 1.0, 1, 1), "wall_seconds"),
        (ResourceSample(1.0, 115200.1, 1, 1), "cpu_seconds"),
        (
            ResourceSample(1.0, 1.0, 128 * 1024**3 + 1, 1),
            "peak_aggregate_pss_bytes",
        ),
        (
            ResourceSample(1.0, 1.0, 1, 20 * 1024**3 + 1),
            "output_storage_bytes",
        ),
    ),
)
def test_first_cap_breach_terminates_process_group_and_prevents_audit(
    tmp_path: Path,
    sample: ResourceSample,
    dimension: str,
) -> None:
    calls: list[str] = []
    terminated: list[str] = []

    def executor(command):
        calls.append(command.purpose)
        return command.purpose

    def sampler(handle):
        observed = ResourceSample(1.0, 1.0, 1, 1)
        if handle == "interact":
            observed = sample
        return (RuntimeObservation(observed, exit_code=0),)

    with pytest.raises(CapBreach) as raised:
        run_sequence(
            _commands(tmp_path),
            authority=_native_authority(),
            executor=executor,
            sampler=sampler,
            terminator=terminated.append,
        )

    assert raised.value.dimension == dimension
    assert calls == ["validate", "interact"]
    assert terminated == ["interact"]


def test_simultaneous_cap_breaches_report_wall_first(tmp_path: Path) -> None:
    sample = ResourceSample(
        wall_seconds=7201,
        cpu_seconds=115201,
        aggregate_pss_bytes=128 * 1024**3 + 1,
        output_bytes=20 * 1024**3 + 1,
    )

    with pytest.raises(CapBreach) as raised:
        run_sequence(
            _commands(tmp_path),
            authority=_native_authority(),
            executor=lambda command: command.purpose,
            sampler=lambda _handle: (RuntimeObservation(sample, exit_code=0),),
            terminator=lambda _handle: None,
        )

    assert raised.value.dimension == "wall_seconds"


def test_nonzero_interact_exit_is_terminal_without_retry_or_audit(tmp_path: Path) -> None:
    calls: list[str] = []

    def executor(command):
        calls.append(command.purpose)
        return command.purpose

    def sampler(handle):
        exit_code = 2 if handle == "interact" else 0
        return (
            RuntimeObservation(
                ResourceSample(1.0, 1.0, 1, 1), exit_code=exit_code
            ),
        )

    with pytest.raises(RuntimeError, match="interact exited with status 2"):
        run_sequence(
            _commands(tmp_path),
            authority=_native_authority(),
            executor=executor,
            sampler=sampler,
            terminator=lambda _handle: None,
        )

    assert calls == ["validate", "interact"]


def test_sequence_reaches_audit_only_after_validate_and_interact_succeed(
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    def executor(command):
        calls.append(command.purpose)
        return command.purpose

    completed = run_sequence(
        _commands(tmp_path),
        authority=_native_authority(),
        executor=executor,
        sampler=lambda _handle: (
            RuntimeObservation(ResourceSample(1.0, 1.0, 1, 1), exit_code=0),
        ),
        terminator=lambda _handle: None,
    )

    assert completed == ("validate", "interact", "audit")
    assert calls == ["validate", "interact", "audit"]


def test_sequence_rejects_audit_without_validate_and_interact_before_executor(
    tmp_path: Path,
) -> None:
    audit_only = (_commands(tmp_path)[-1],)
    calls = []

    with pytest.raises(RuntimeError, match="validate, interact, audit"):
        run_sequence(
            audit_only,
            authority=_native_authority(),
            executor=lambda command: calls.append(command),
            sampler=lambda _handle: (),
            terminator=lambda _handle: None,
        )

    assert calls == []
