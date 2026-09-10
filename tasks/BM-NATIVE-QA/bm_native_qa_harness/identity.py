from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from scripts.benchmarks.v201.contracts import derive_seed, sha256_payload

from .plan import ProspectiveInventory
from .policy import NATIVE_CAPS, canonical_interact

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


class IdentityError(RuntimeError):
    """A prospective identity binding is malformed or inconsistent."""


def _require_sha256(value: str, name: str) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise IdentityError(f"invalid SHA-256 for {name}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class FrozenIdentity:
    qa_design_hash: str
    design_payload: Mapping[str, object]
    seeds: tuple[object, ...] = ()


@dataclass(frozen=True)
class SeedRecord:
    purpose: str
    context_key: str
    truth_id: str | None
    scenario_id: str
    stage_or_role: str
    value: int
    seed_id: str


def _plain_record(value: Any) -> dict[str, object]:
    return {
        key: str(item) if key.endswith("_path") else item
        for key, item in asdict(value).items()
    }


def _future_artifact_layout(inventory: ProspectiveInventory) -> dict[str, object]:
    root = "tasks/BM-NATIVE-QA/materialized/v1"
    return {
        "root": root,
        "anchors": [
            {
                "anchor_id": (
                    f"qa_real80_v4.{context.panel_id}."
                    f"{context.sample_context}.anchor"
                ),
                "context_key": context.key,
                "anchor_npy": f"{root}/anchors/{context.key}/anchor.npy",
                "anchor_int_npy": f"{root}/anchors/{context.key}/anchor_int.npy",
                "v_hat_npy": f"{root}/anchors/{context.key}/v_hat.npy",
                "root_v_npy": f"{root}/anchors/{context.key}/root_v.npy",
                "manifest_json": f"{root}/anchors/{context.key}/manifest.json",
            }
            for context in inventory.contexts
        ],
        "responses": [
            {
                "response_id": response.response_id,
                "response_npy": (
                    f"{root}/responses/{response.response_id}/response.npy"
                ),
                "phenotype_tsv": (
                    f"{root}/responses/{response.response_id}/phenotype.tsv"
                ),
                "manifest_json": (
                    f"{root}/responses/{response.response_id}/manifest.json"
                ),
            }
            for response in inventory.responses
        ],
        "invocations": [
            {
                "invocation_id": invocation.invocation_id,
                "config_yaml": f"{root}/configs/{invocation.invocation_id}.yaml",
                "checkpoint_root": (
                    f"{root}/checkpoints/{invocation.invocation_id}"
                ),
                "result_dir": f"{root}/results/{invocation.invocation_id}",
                "audit_dir": f"{root}/audits/{invocation.invocation_id}",
            }
            for invocation in inventory.invocations
        ],
    }


def freeze_identity(
    inventory: ProspectiveInventory,
    *,
    fixture_manifest_sha256: str,
    amendment_sha256: str,
    runner_test_sha256s: Mapping[str, str],
) -> FrozenIdentity:
    _require_sha256(fixture_manifest_sha256, "fixture manifest")
    _require_sha256(amendment_sha256, "amendment")
    if not runner_test_sha256s:
        raise IdentityError("runner/test SHA-256 mapping must not be empty")
    for name, value in runner_test_sha256s.items():
        _require_sha256(value, name)
    for context in inventory.contexts:
        if context.subgenomes and not context.input_file_sha256s:
            raise IdentityError(
                f"prospective input-file SHA-256 bindings missing for {context.key}"
            )
        paths = [path for path, _digest in context.input_file_sha256s]
        if len(paths) != len(set(paths)):
            raise IdentityError(f"duplicate prospective input path for {context.key}")
        for path, digest in context.input_file_sha256s:
            _require_sha256(digest, f"{context.key}:{path}")
    science = canonical_interact()
    payload: dict[str, object] = {
        "fixture_manifest_sha256": fixture_manifest_sha256,
        "amendment_sha256": amendment_sha256,
        "runner_test_sha256s": {
            name: runner_test_sha256s[name] for name in sorted(runner_test_sha256s)
        },
        "contexts": [_plain_record(value) for value in inventory.contexts],
        "responses": [_plain_record(value) for value in inventory.responses],
        "invocations": [_plain_record(value) for value in inventory.invocations],
        "response_generation_policy_id": (
            "fitted_vhat_from_independent_standard_normal_anchor_v1"
        ),
        "canonical_interact": science,
        "bootstrap": {
            "B": science["calibration"]["B"],
            "checkpoint_mode": "indexed_required",
        },
        "aggregate_caps": dict(NATIVE_CAPS),
        "future_artifacts": _future_artifact_layout(inventory),
    }
    design_hash = sha256_payload(payload)
    return FrozenIdentity(
        qa_design_hash=design_hash,
        design_payload=payload,
        seeds=build_seed_ledger(design_hash, inventory),
    )


def _seed_record(
    design_hash: str,
    *,
    purpose: str,
    context_key: str,
    truth_id: str | None,
    scenario_id: str,
    stage_or_role: str,
) -> SeedRecord:
    value = derive_seed(design_hash, "omnib", scenario_id, 0, stage_or_role)
    return SeedRecord(
        purpose=purpose,
        context_key=context_key,
        truth_id=truth_id,
        scenario_id=scenario_id,
        stage_or_role=stage_or_role,
        value=value,
        seed_id=f"{stage_or_role}:{scenario_id}:0:{value:016x}",
    )


def build_seed_ledger(
    design_hash: str,
    inventory: ProspectiveInventory,
) -> tuple[SeedRecord, ...]:
    responses_by_context = {
        context.key: tuple(
            response
            for response in inventory.responses
            if response.context_key == context.key
        )
        for context in inventory.contexts
    }
    records: list[SeedRecord] = []
    for context in inventory.contexts:
        base = f"qa_real80_v4.{context.panel_id}.{context.sample_context}"
        records.append(
            _seed_record(
                design_hash,
                purpose="anchor",
                context_key=context.key,
                truth_id=None,
                scenario_id=base,
                stage_or_role="pilot:qa_anchor",
            )
        )
        for response in responses_by_context[context.key]:
            records.append(
                _seed_record(
                    design_hash,
                    purpose="observed",
                    context_key=context.key,
                    truth_id=response.truth_id,
                    scenario_id=response.response_id,
                    stage_or_role="pilot:qa_observed",
                )
            )
            records.append(
                _seed_record(
                    design_hash,
                    purpose="bootstrap",
                    context_key=context.key,
                    truth_id=response.truth_id,
                    scenario_id=response.response_id,
                    stage_or_role="pilot:native_bootstrap",
                )
            )
        context_values = [
            record.value for record in records if record.context_key == context.key
        ]
        if len(set(context_values)) != len(context_values):
            raise RuntimeError(f"seed collision within context {context.key}")
    return tuple(records)
