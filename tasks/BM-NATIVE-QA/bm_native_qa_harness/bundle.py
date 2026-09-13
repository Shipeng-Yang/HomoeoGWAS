from __future__ import annotations

import csv
import json
import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .authority import VerifiedMaterializationAuthority
from .config import (
    build_interact_config,
    dump_config,
    scientific_config_sha256,
)
from .identity import SeedRecord, sha256_file
from .materialize import (
    _float64_sha256,
    generate_response,
    prepare_anchor,
    write_roundtrip_response,
)
from .plan import ContextSpec, ProspectiveInventory

ContextLoader = Callable[[ContextSpec], tuple[Any, Sequence[str]]]


def _read_ordered_samples(
    path: Path,
    *,
    reference_samples: Sequence[str],
    expected_count: int = 192,
) -> tuple[tuple[str, ...], np.ndarray]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not {"sample_id", "source_order"}.issubset(reader.fieldnames or ()):
            raise ValueError("sample selection lacks sample_id/source_order columns")
        rows = list(reader)
    sample_ids = tuple(str(row["sample_id"]) for row in rows)
    try:
        source_order = np.asarray(
            [int(row["source_order"]) for row in rows], dtype=np.int64
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("sample selection source_order is not integral") from exc
    if (
        len(sample_ids) != expected_count
        or len(set(sample_ids)) != expected_count
        or source_order.size != expected_count
        or len(set(source_order.tolist())) != expected_count
        or np.any(source_order < 0)
        or np.any(source_order >= len(reference_samples))
        or np.any(np.diff(source_order) <= 0)
    ):
        raise ValueError("sample selection is not the exact ordered unique subset")
    selected = tuple(str(reference_samples[index]) for index in source_order)
    if selected != sample_ids:
        raise ValueError("sample IDs differ from FAM source_order")
    return sample_ids, source_order


def make_real_context_loader(
    *,
    project_root: Path,
    context_evidence: Mapping[str, Any],
) -> ContextLoader:
    """Create a cache-aware loader for the six reviewed real contexts."""

    from homoeogwas.group_family import load_master_group_family
    from homoeogwas.interact import _load_subgenome
    from scripts.benchmarks.v201.track_omnib import OmniBBenchmarkContext

    root = project_root.resolve()
    evidence_contexts = context_evidence.get("contexts")
    if not isinstance(evidence_contexts, dict):
        raise ValueError("context evidence lacks contexts mapping")
    subdata_cache: dict[tuple[tuple[str, str, str], ...], dict[str, Any]] = {}
    family_cache: dict[tuple[str, tuple[str, ...]], Any] = {}

    def resolve(path: str | Path) -> Path:
        value = Path(path)
        return value if value.is_absolute() else root / value

    def load(context_spec: ContextSpec) -> tuple[Any, Sequence[str]]:
        evidence = evidence_contexts.get(context_spec.key)
        if not isinstance(evidence, dict):
            raise ValueError(f"context evidence is absent: {context_spec.key}")
        if evidence.get("feature_seed") != context_spec.feature_seed:
            raise ValueError(f"context feature seed differs: {context_spec.key}")
        copy_evidence = evidence.get("copies")
        if not isinstance(copy_evidence, dict) or tuple(copy_evidence) != (
            context_spec.subgenomes
        ):
            raise ValueError(f"context subgenome order differs: {context_spec.key}")
        bed_prefixes = dict(context_spec.bed_prefixes)
        mappings = dict(context_spec.snp_to_gene)
        cache_key = tuple(
            (label, str(resolve(bed_prefixes[label])), str(resolve(mappings[label])))
            for label in context_spec.subgenomes
        )
        if cache_key not in subdata_cache:
            subdata_cache[cache_key] = {
                label: _load_subgenome(
                    str(resolve(bed_prefixes[label])),
                    str(resolve(mappings[label])),
                    verify_mapping=True,
                )
                for label in context_spec.subgenomes
            }
        subdata = subdata_cache[cache_key]
        reference = tuple(str(value) for value in subdata[context_spec.subgenomes[0]].samples)
        if any(
            tuple(str(value) for value in subdata[label].samples) != reference
            for label in context_spec.subgenomes[1:]
        ):
            raise ValueError("real PLINK subgenomes lack identical ordered samples")
        sample_ids, sample_idx = _read_ordered_samples(
            resolve(context_spec.samples_path),
            reference_samples=reference,
        )
        family_key = (str(resolve(context_spec.groups_path)), context_spec.subgenomes)
        if family_key not in family_cache:
            family_cache[family_key] = load_master_group_family(
                family_key[0],
                context_spec.subgenomes,
                require_group_id=True,
            )
        family = family_cache[family_key]
        evidence_group_ids = tuple(
            str(row["group_id"]) for row in evidence.get("groups", ())
        )
        if tuple(family.group_ids) != evidence_group_ids:
            raise ValueError(f"context family order differs: {context_spec.key}")
        if evidence.get("n_groups_all_edges_estimable") != (
            context_spec.expected_calibrated_groups
        ):
            raise ValueError(f"context expected group count differs: {context_spec.key}")
        context = OmniBBenchmarkContext(
            subdata,
            family,
            np.zeros(sample_idx.size, dtype=np.float64),
            sample_idx,
            panel_id=context_spec.panel_id,
            sample_context=context_spec.sample_context,
            feature_seed=context_spec.feature_seed,
        )
        for label in context_spec.subgenomes:
            observed = context.marker_mask_identity[label]["n_variants_retained"]
            if observed != copy_evidence[label].get("retained_markers"):
                raise ValueError(
                    f"context retained marker count differs: {context_spec.key}:{label}"
                )
        return context, sample_ids

    return load


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        _jsonable(payload),
        sort_keys=True,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"
    with path.open("x", encoding="utf-8", newline="") as handle:
        handle.write(text)
    return sha256_file(path)


def _write_npy(path: Path, values: np.ndarray) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.ascontiguousarray(values, dtype=np.dtype("<f8"))
    with path.open("xb") as handle:
        np.save(handle, array, allow_pickle=False)
    loaded = np.load(path, allow_pickle=False)
    if loaded.dtype.str != "<f8" or not np.array_equal(
        loaded.view(np.uint64), array.view(np.uint64)
    ):
        raise RuntimeError(f"array failed bitwise round-trip: {path}")
    return {
        "file_sha256": sha256_file(path),
        "float64_sha256": _float64_sha256(array),
        "shape": list(array.shape),
        "dtype": loaded.dtype.str,
    }


def _attempt_path(final_path: str | Path, *, root: Path, attempt: Path) -> Path:
    path = Path(final_path)
    if not path.is_absolute() or not path.is_relative_to(root):
        raise ValueError(f"frozen artifact path escapes root: {path}")
    return attempt / path.relative_to(root)


def _seed_lookup(
    seeds: Sequence[SeedRecord],
) -> dict[tuple[str, str, str | None], SeedRecord]:
    lookup = {
        (record.purpose, record.context_key, record.truth_id): record
        for record in seeds
    }
    if len(lookup) != len(seeds):
        raise ValueError("duplicate seed roles are forbidden")
    return lookup


def require_replica_identity(config_records: Sequence[Mapping[str, Any]]) -> None:
    """Fail closed unless every PC1 replica pair shares scientific identity."""

    pairs: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in config_records:
        invocation_id = str(row.get("invocation_id", ""))
        replica = invocation_id.rsplit(".", 1)[-1]
        if replica not in {"replica_a", "replica_b"}:
            continue
        base = invocation_id.rsplit(".", 1)[0]
        pairs.setdefault(base, {})[replica] = row
    if len(pairs) != 4 or any(
        set(rows) != {"replica_a", "replica_b"} for rows in pairs.values()
    ):
        raise ValueError("replica identity mismatch: expected four complete pairs")
    identity_fields = (
        "response_id",
        "jobs",
        "bootstrap_seed",
        "phenotype_tsv",
        "scientific_config_sha256",
    )
    for base, rows in pairs.items():
        left = tuple(rows["replica_a"].get(field) for field in identity_fields)
        right = tuple(rows["replica_b"].get(field) for field in identity_fields)
        if left != right:
            raise ValueError(f"replica identity mismatch: {base}")


def _written_file_inventory(root: Path) -> list[dict[str, str]]:
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": sha256_file(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "materialization-abort.json"
    ]


def _validate_bundle_inputs(
    verified: VerifiedMaterializationAuthority,
    inventory: ProspectiveInventory,
    seeds: Sequence[SeedRecord],
) -> tuple[dict[str, Any], dict[tuple[str, str, str | None], SeedRecord]]:
    if (
        len(inventory.contexts) != 6
        or len(inventory.responses) != 12
        or len(inventory.invocations) != 16
        or {row.jobs for row in inventory.invocations} != {128}
    ):
        raise ValueError("materialization requires the exact 6/12/16 workers128 inventory")
    layout = verified.inventory.get("design_payload", {}).get("future_artifacts")
    if not isinstance(layout, dict) or layout.get("root") != str(
        verified.artifact_root
    ):
        raise ValueError("verified future artifact layout is invalid")
    if len(layout.get("anchors", ())) != 6 or len(layout.get("responses", ())) != 12:
        raise ValueError("frozen anchor/response layout has incorrect cardinality")
    if len(layout.get("invocations", ())) != 16:
        raise ValueError("frozen invocation layout has incorrect cardinality")
    lookup = _seed_lookup(seeds)
    required = {
        ("anchor", context.key, None) for context in inventory.contexts
    } | {
        (purpose, response.context_key, response.truth_id)
        for response in inventory.responses
        for purpose in ("observed", "bootstrap")
    }
    if set(lookup) != required:
        raise ValueError("seed ledger differs from the exact six/twelve/twelve roles")
    return layout, lookup


def materialize_bundle(
    verified: VerifiedMaterializationAuthority,
    *,
    inventory: ProspectiveInventory,
    seeds: Sequence[SeedRecord],
    context_loader: ContextLoader,
    context_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write the reviewed bundle once and stop before every native command."""

    layout, seed_by_role = _validate_bundle_inputs(verified, inventory, seeds)
    root = verified.artifact_root
    attempt = root.parent / f".{root.name}.materialization-attempt"
    lock = root.parent / f".{root.name}.materialization-lock"
    root.parent.mkdir(parents=True, exist_ok=True)
    if root.exists():
        raise FileExistsError(f"exclusive artifact root already exists: {root}")
    if attempt.exists():
        raise FileExistsError(f"materialization attempt already exists: {attempt}")
    if lock.exists():
        raise FileExistsError(f"materialization lock already exists: {lock}")
    try:
        _write_json(
            lock,
            {
                "schema": "homoeogwas-bm-native-qa-materialization-lock-v1",
                "status": "CLAIMED_EXCLUSIVE_NO_RETRY",
                "qa_design_hash": verified.qa_design_hash,
                "authority_bindings": verified.binding_hashes,
                "final_artifact_root": str(root),
            },
        )
    except FileExistsError as exc:
        raise FileExistsError(f"materialization lock already exists: {lock}") from exc
    attempt.mkdir()
    _write_json(
        attempt / "materialization-attempt.json",
        {
            "schema": "homoeogwas-bm-native-qa-materialization-attempt-v1",
            "status": "STARTED_EXCLUSIVE_NO_RETRY",
            "qa_design_hash": verified.qa_design_hash,
            "authority_bindings": verified.binding_hashes,
            "final_artifact_root": str(root),
        },
    )

    anchor_layout = {row["context_key"]: row for row in layout["anchors"]}
    response_layout = {row["response_id"]: row for row in layout["responses"]}
    invocation_layout = {
        row["invocation_id"]: row for row in layout["invocations"]
    }
    contexts = {row.key: row for row in inventory.contexts}
    responses = {row.response_id: row for row in inventory.responses}
    anchor_records: list[dict[str, Any]] = []
    response_records: list[dict[str, Any]] = []
    config_records: list[dict[str, Any]] = []
    prepared_by_context: dict[str, Any] = {}
    samples_by_context: dict[str, tuple[str, ...]] = {}

    try:
        for context_spec in inventory.contexts:
            context, raw_sample_ids = context_loader(context_spec)
            sample_ids = tuple(raw_sample_ids)
            if (
                context.panel_id != context_spec.panel_id
                or context.sample_context != context_spec.sample_context
                or context.feature_seed != context_spec.feature_seed
                or tuple(context.family.subgenomes) != context_spec.subgenomes
                or len(sample_ids) != context.sample_idx.size
                or len(set(sample_ids)) != len(sample_ids)
            ):
                raise ValueError(f"loaded context differs from inventory: {context_spec.key}")
            anchor_seed = seed_by_role[("anchor", context_spec.key, None)]
            prepared = prepare_anchor(
                context,
                design_hash=verified.qa_design_hash,
                scenario_id=anchor_seed.scenario_id,
                anchor_seed=anchor_seed.value,
            )
            estimable = int(np.asarray(prepared.scores.group_estimable, bool).sum())
            if estimable != context_spec.expected_calibrated_groups:
                raise ValueError(
                    f"prepared group count differs from inventory: {context_spec.key}"
                )
            if context_evidence is not None:
                evidence = context_evidence["contexts"][context_spec.key]
                failed = tuple(
                    group_id
                    for group_id, passed in zip(
                        prepared.context.family.group_ids,
                        np.asarray(prepared.scores.group_estimable, bool),
                        strict=True,
                    )
                    if not passed
                )
                if failed != tuple(evidence["failed_group_ids"]):
                    raise ValueError(
                        f"prepared failed-group identity differs: {context_spec.key}"
                    )
            row = anchor_layout[context_spec.key]
            arrays = {}
            for label, final_path, values in (
                ("anchor", row["anchor_npy"], prepared.anchor),
                ("anchor_int", row["anchor_int_npy"], prepared.scores.y),
                ("v_hat", row["v_hat_npy"], prepared.v_hat),
                ("root_v", row["root_v_npy"], prepared.root_v),
            ):
                physical = _attempt_path(final_path, root=root, attempt=attempt)
                arrays[label] = {
                    "path": str(final_path),
                    **_write_npy(physical, values),
                }
            manifest = {
                "schema": "homoeogwas-bm-native-qa-anchor-v2",
                "qa_design_hash": verified.qa_design_hash,
                "anchor_id": row["anchor_id"],
                "context_key": context_spec.key,
                "anchor_seed": anchor_seed.value,
                "anchor_seed_id": anchor_seed.seed_id,
                "sample_ids": list(sample_ids),
                "sample_count": len(sample_ids),
                "marker_mask_sha256": context.marker_mask_sha256,
                "marker_mask_identity": context.marker_mask_identity,
                "expected_calibrated_groups": context_spec.expected_calibrated_groups,
                "prepared_estimable_groups": estimable,
                "arrays": arrays,
                "fit_identity": {
                    "feature_cache_sha256": prepared.scores.feature_cache_sha256,
                    "fixed_mask_sha256": prepared.scores.fixed_mask_sha256,
                    "null_fit_sha256": prepared.scores.null_fit_sha256,
                    "prepared_design_sha256": prepared.scores.prepared_design_sha256,
                    "grm_provenance": prepared.scores.grm_provenance,
                    "retained_variant_mask_identity": (
                        prepared.scores.retained_variant_mask_identity
                    ),
                },
            }
            manifest_final = Path(row["manifest_json"])
            manifest_physical = _attempt_path(
                manifest_final, root=root, attempt=attempt
            )
            manifest_sha = _write_json(manifest_physical, manifest)
            anchor_records.append(
                {
                    "anchor_id": row["anchor_id"],
                    "context_key": context_spec.key,
                    "manifest_path": str(manifest_final),
                    "manifest_sha256": manifest_sha,
                }
            )
            prepared_by_context[context_spec.key] = prepared
            samples_by_context[context_spec.key] = sample_ids

        for response_spec in inventory.responses:
            prepared = prepared_by_context[response_spec.context_key]
            response_seed = seed_by_role[
                ("observed", response_spec.context_key, response_spec.truth_id)
            ]
            generated = generate_response(
                prepared,
                response_id=response_spec.response_id,
                truth_id=response_spec.truth_id,
                response_seed=response_seed.value,
            )
            row = response_layout[response_spec.response_id]
            npy_physical = _attempt_path(
                row["response_npy"], root=root, attempt=attempt
            )
            tsv_physical = _attempt_path(
                row["phenotype_tsv"], root=root, attempt=attempt
            )
            npy_physical.parent.mkdir(parents=True, exist_ok=True)
            serialized = write_roundtrip_response(
                generated,
                sample_ids=samples_by_context[response_spec.context_key],
                npy_path=npy_physical,
                tsv_path=tsv_physical,
            )
            serialized["npy_path"] = str(row["response_npy"])
            serialized["tsv_path"] = str(row["phenotype_tsv"])
            response_manifest = {
                "schema": "homoeogwas-bm-native-qa-response-v2",
                "qa_design_hash": verified.qa_design_hash,
                "context_key": response_spec.context_key,
                "response_seed": response_seed.value,
                "response_seed_id": response_seed.seed_id,
                "serialization": serialized,
                "generation": generated.metadata,
            }
            manifest_final = Path(row["manifest_json"])
            manifest_sha = _write_json(
                _attempt_path(manifest_final, root=root, attempt=attempt),
                response_manifest,
            )
            response_records.append(
                {
                    "response_id": response_spec.response_id,
                    "context_key": response_spec.context_key,
                    "phenotype_tsv": str(row["phenotype_tsv"]),
                    "manifest_path": str(manifest_final),
                    "manifest_sha256": manifest_sha,
                }
            )

        for invocation in inventory.invocations:
            response_spec = responses[invocation.response_id]
            context_spec = contexts[response_spec.context_key]
            response_row = response_layout[invocation.response_id]
            row = invocation_layout[invocation.invocation_id]
            bootstrap_seed = seed_by_role[
                ("bootstrap", response_spec.context_key, response_spec.truth_id)
            ]
            generated_config = build_interact_config(
                context_spec,
                response_spec,
                invocation,
                bootstrap_seed=bootstrap_seed.value,
                phenotype_path=Path(response_row["phenotype_tsv"]),
                checkpoint_root=Path(row["checkpoint_root"]),
                output_root=Path(row["result_dir"]),
            )
            config_final = Path(row["config_yaml"])
            config_physical = _attempt_path(
                config_final, root=root, attempt=attempt
            )
            config_physical.parent.mkdir(parents=True, exist_ok=True)
            config_sha = dump_config(generated_config, config_physical)
            config_records.append(
                {
                    "invocation_id": invocation.invocation_id,
                    "response_id": invocation.response_id,
                    "jobs": invocation.jobs,
                    "bootstrap_seed": bootstrap_seed.value,
                    "bootstrap_seed_id": bootstrap_seed.seed_id,
                    "phenotype_tsv": str(response_row["phenotype_tsv"]),
                    "config_path": str(config_final),
                    "config_sha256": config_sha,
                    "scientific_config_sha256": scientific_config_sha256(
                        generated_config
                    ),
                    "checkpoint_root": str(row["checkpoint_root"]),
                    "result_dir": str(row["result_dir"]),
                    "audit_dir": str(row["audit_dir"]),
                }
            )

        require_replica_identity(config_records)
        counts = {
            "anchors": len(anchor_records),
            "responses": len(response_records),
            "configs": len(config_records),
        }
        if counts != {"anchors": 6, "responses": 12, "configs": 16}:
            raise ValueError("physical materialization counts differ from 6/12/16")
        prohibited = [attempt / name for name in ("checkpoints", "results", "audits")]
        if any(path.exists() for path in prohibited):
            raise ValueError("prohibited native output directory exists")
        physical_counts = {
            "anchors": len(list(attempt.glob("anchors/*/manifest.json"))),
            "responses": len(list(attempt.glob("responses/*/manifest.json"))),
            "configs": len(list(attempt.glob("configs/*.yaml"))),
        }
        if physical_counts != counts:
            raise ValueError("physical file counts differ from materialization records")
        manifest = {
            "schema": "homoeogwas-bm-native-qa-materialization-manifest-v2",
            "status": "MATERIALIZED_AWAITING_INDEPENDENT_PHYSICAL_REVIEW",
            "qa_design_hash": verified.qa_design_hash,
            "run_namespace": "qa_real80_njobs128_v1",
            "response_materialization_authorized": True,
            "execution_authorized": False,
            "authority_bindings": verified.binding_hashes,
            "counts": counts,
            "physical_counts": physical_counts,
            "materialization_lock": str(lock),
            "anchors": anchor_records,
            "responses": response_records,
            "configs": config_records,
            "prohibited_outputs_confirmed_absent": [
                str(root / name) for name in ("checkpoints", "results", "audits")
            ],
        }
        _write_json(attempt / "materialization-manifest.json", manifest)
        os.replace(attempt, root)
        return _jsonable(manifest)
    except Exception as exc:
        abort = {
            "schema": "homoeogwas-bm-native-qa-materialization-abort-v1",
            "status": "ABORTED_NO_RETRY",
            "qa_design_hash": verified.qa_design_hash,
            "authority_bindings": verified.binding_hashes,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "written_files": _written_file_inventory(attempt),
        }
        try:
            _write_json(attempt / "materialization-abort.json", abort)
        except Exception as abort_exc:
            exc.add_note(f"could not persist materialization abort record: {abort_exc}")
        raise
