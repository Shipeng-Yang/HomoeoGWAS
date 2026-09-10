from __future__ import annotations

import pytest
from bm_native_qa_harness.plan import build_inventory


def test_build_inventory_expands_six_contexts_to_twelve_responses_and_sixteen_invocations(
    amendment: dict,
) -> None:
    inventory = build_inventory(amendment)

    assert len(inventory.contexts) == 6
    assert len(inventory.responses) == 12
    assert len(inventory.invocations) == 16
    assert [
        (row.sample_context, row.jobs) for row in inventory.invocations[:4]
    ] == [
        ("pc1_spread_192", 1),
        ("pc1_spread_192", 4),
        ("seeded_random_192", 4),
        ("holdout_192", 4),
    ]


def test_inventory_rejects_open_authorization_flag(amendment: dict) -> None:
    amendment["execution_authorized"] = True

    with pytest.raises(RuntimeError, match="execution_authorized must remain false"):
        build_inventory(amendment)


def test_inventory_rejects_open_response_materialization_flag(amendment: dict) -> None:
    amendment["response_materialization_authorized"] = True

    with pytest.raises(
        RuntimeError,
        match="response_materialization_authorized must remain false",
    ):
        build_inventory(amendment)


def test_inventory_rejects_worker_matrix_drift(amendment: dict) -> None:
    amendment["matrix"]["context_job_rows"][0]["jobs"] = [1, 2]

    with pytest.raises(RuntimeError, match="frozen context/job matrix"):
        build_inventory(amendment)


def test_inventory_rejects_duplicate_response_and_invocation_ids(amendment: dict) -> None:
    amendment["matrix"]["panels"] = ["REALG.CGVD1245", "REALG.CGVD1245"]

    with pytest.raises(RuntimeError, match="duplicate prospective IDs"):
        build_inventory(amendment)


def test_inventory_rejects_declared_count_mismatch(amendment: dict) -> None:
    amendment["matrix"]["native_invocations"] = 15

    with pytest.raises(RuntimeError, match="declared matrix counts"):
        build_inventory(amendment)


def test_inventory_rejects_worker_invariance_scope_drift(amendment: dict) -> None:
    amendment["worker_invariance_pair"]["contexts"] = ["cotton.pc1_spread_192"]

    with pytest.raises(RuntimeError, match="worker-invariance contexts"):
        build_inventory(amendment)
