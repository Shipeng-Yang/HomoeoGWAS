from __future__ import annotations

from bm_native_qa_harness.identity import freeze_identity
from bm_native_qa_harness.plan import build_inventory
from bm_native_qa_harness.policy import NATIVE_CAPS, canonical_interact


def test_frozen_identity_uses_the_shared_science_and_cap_policy(
    amendment: dict,
) -> None:
    frozen = freeze_identity(
        build_inventory(amendment),
        fixture_manifest_sha256="7" * 64,
        amendment_sha256="a" * 64,
        runner_test_sha256s={"runner.py": "b" * 64},
    )

    assert frozen.design_payload["canonical_interact"] == canonical_interact()
    assert frozen.design_payload["aggregate_caps"] == dict(NATIVE_CAPS)
    assert frozen.design_payload["aggregate_caps"][
        "large_tmp_output_forbidden"
    ] is True
