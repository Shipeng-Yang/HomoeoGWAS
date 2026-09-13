from __future__ import annotations

import hashlib
import importlib.machinery
import json
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class AuthorityBlocked(RuntimeError):
    """The reviewed response-materialization authority is not exact."""


V1_QA_DESIGN_HASH = (
    "65ba6556d86db4d27b03169ff70d5b39796c5968ce8ece9f782e866995cf5d5e"
)
V1_INVENTORY_SHA256 = (
    "7ded836fa5348ba0b24ca6fdfa061393172f5813105f16e31dbd6bb85ea25e66"
)
# Canonical JSON SHA-256 of runner_test_sha256s from
# 3aa5e07f0c87575fae90463943b17ae141662cc1:tasks/BM-NATIVE-QA/
# prospective-inventory.v1.json. A test derives this literal from those bytes.
V1_RUNNER_TEST_MAPPING_SHA256 = (
    "342fb67de133dc812a10a34e68b97dc04bfa16e538f6e9a79b452d92cd0132bb"
)
ACCEPTED_PRODUCT_COMMIT = "9d7faee655c020d84bdd6a6fba33b42a8428aea6"
ACCEPTED_PRODUCT_TREE = "92355ddcbb2cd3d749169291fcbf6065eb556311"
ACCEPTED_HOMOEO_GWAS_PACKAGE_SOURCE_SHA256 = (
    "4f249d698d94123721f9c94b6544dbd8cc2f2df9aaf04dd6776a351c035e945b"
)
ACCEPTED_V201_HELPER_SOURCE_SHA256 = (
    "b7935d3b51a59efcb1d107cdcf517c1c6a5621bffd025fae26477a645e1067ab"
)


@dataclass(frozen=True)
class VerifiedMaterializationAuthority:
    qa_design_hash: str
    inventory: Mapping[str, Any]
    artifact_root: Path
    paths: Mapping[str, Path]
    binding_hashes: Mapping[str, str]
    source_reverification: Mapping[str, Any]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(path: Path, *, json_format: bool = False) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
        value = json.loads(text) if json_format else yaml.safe_load(text)
    except (OSError, json.JSONDecodeError, yaml.YAMLError) as exc:
        raise AuthorityBlocked(f"cannot load authority binding: {path}") from exc
    if not isinstance(value, dict):
        raise AuthorityBlocked(f"authority binding is not a mapping: {path}")
    return value


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _mapping_sha256(value: Mapping[str, object]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def enumerate_runner_test_sources(task_root: Path) -> dict[str, Path]:
    """Enumerate the exact importable Python source set without following links."""

    root = task_root.resolve()
    members: dict[str, Path] = {}
    for relative_root in ("bm_native_qa_harness", "tests"):
        source_root = root / relative_root
        if not source_root.is_dir() or source_root.is_symlink():
            raise AuthorityBlocked(f"runner/test source root is invalid: {relative_root}")
        for current, directory_names, file_names in os.walk(
            source_root,
            followlinks=False,
        ):
            current_path = Path(current)
            for name in tuple(directory_names):
                child = current_path / name
                if child.is_symlink():
                    raise AuthorityBlocked(f"runner/test symlink is forbidden: {child}")
            directory_names[:] = sorted(
                name for name in directory_names if name != "__pycache__"
            )
            for name in sorted(file_names):
                path = current_path / name
                if path.is_symlink():
                    raise AuthorityBlocked(f"runner/test symlink is forbidden: {path}")
                if path.suffix in {".so", ".pyc", ".pyo"}:
                    raise AuthorityBlocked(
                        f"runner/test import artifact is forbidden: {path}"
                    )
                if path.suffix != ".py":
                    continue
                relative = path.relative_to(root).as_posix()
                members[relative] = path
    if not members:
        raise AuthorityBlocked("runner/test source inventory is empty")
    return members


def python_source_tree_sha256(root: Path, *, domain: str) -> str:
    """Hash ordered relative Python paths and bytes for one source tree."""

    resolved = root.resolve()
    files = sorted(
        (path for path in resolved.rglob("*.py") if path.is_file()),
        key=lambda path: path.relative_to(resolved).as_posix(),
    )
    if not files:
        raise AuthorityBlocked(f"Python source tree is empty: {resolved}")
    digest = hashlib.sha256(domain.encode("utf-8") + b"\0")
    for path in files:
        relative = path.relative_to(resolved).as_posix().encode("utf-8")
        body = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(body).to_bytes(8, "big"))
        digest.update(body)
    return digest.hexdigest()


def _runtime_path(runtime: Mapping[str, Any], field: str) -> Path:
    return _absolute_path(runtime.get(field), f"runtime.{field}").resolve()


def verify_runtime_import_isolation(
    verified: VerifiedMaterializationAuthority,
    *,
    task_root: Path,
) -> None:
    """Reject source shadowing before importing any numerical module."""

    runtime = verified.source_reverification.get("runtime")
    if not isinstance(runtime, dict):
        raise AuthorityBlocked("source reverification lacks runtime isolation binding")
    if (
        runtime.get("accepted_product_commit") != ACCEPTED_PRODUCT_COMMIT
        or runtime.get("accepted_product_tree") != ACCEPTED_PRODUCT_TREE
        or runtime.get("homoeogwas_package_source_sha256")
        != ACCEPTED_HOMOEO_GWAS_PACKAGE_SOURCE_SHA256
        or runtime.get("v201_helper_source_sha256")
        != ACCEPTED_V201_HELPER_SOURCE_SHA256
    ):
        raise AuthorityBlocked("runtime numerical source identity is not accepted r3")
    expected_python = _runtime_path(runtime, "python_executable_realpath")
    expected_prefix = _runtime_path(runtime, "sys_prefix_realpath")
    expected_task_root = _runtime_path(runtime, "harness_task_root")
    helper_root = _runtime_path(runtime, "accepted_helper_source_root")
    site_packages = _runtime_path(runtime, "installed_site_packages")
    package_root = _runtime_path(runtime, "installed_homoeogwas_package_root")
    if Path(sys.executable).resolve() != expected_python:
        raise AuthorityBlocked("runtime Python executable differs from authority")
    if Path(sys.prefix).resolve() != expected_prefix:
        raise AuthorityBlocked("runtime sys.prefix differs from authority")
    if task_root.resolve() != expected_task_root:
        raise AuthorityBlocked("runtime harness root differs from authority")
    if package_root != site_packages / "homoeogwas" or not package_root.is_dir():
        raise AuthorityBlocked("installed HomoeoGWAS package root is invalid")
    if not (helper_root / "scripts/benchmarks/v201").is_dir():
        raise AuthorityBlocked("accepted helper source root is invalid")
    editable = tuple(site_packages.glob("__editable__*homoeogwas*.pth")) + tuple(
        site_packages.glob("*homoeogwas*.egg-link")
    )
    if editable:
        raise AuthorityBlocked("editable HomoeoGWAS installation is forbidden")
    forbidden_loaded = sorted(
        name
        for name in sys.modules
        if name == "homoeogwas"
        or name.startswith("homoeogwas.")
        or name == "scripts"
        or name.startswith("scripts.benchmarks")
    )
    if forbidden_loaded:
        raise AuthorityBlocked(
            "numerical module was loaded before authority gate: " + forbidden_loaded[0]
        )
    allowed = {expected_task_root, helper_root, site_packages}
    for raw_entry in sys.path:
        entry = Path(raw_entry or os.getcwd()).resolve()
        if entry in allowed:
            continue
        for top_level in ("homoeogwas", "scripts"):
            spec = importlib.machinery.PathFinder.find_spec(top_level, [str(entry)])
            if spec is not None:
                raise AuthorityBlocked(
                    f"unreviewed sys.path entry resolves {top_level}: {entry}"
                )
    helper_spec = importlib.machinery.PathFinder.find_spec("scripts", [str(helper_root)])
    package_spec = importlib.machinery.PathFinder.find_spec(
        "homoeogwas", [str(site_packages)]
    )
    if helper_spec is None or package_spec is None:
        raise AuthorityBlocked("reviewed numerical import roots are not resolvable")


def verify_loaded_numerical_origins(
    verified: VerifiedMaterializationAuthority,
) -> None:
    """Verify loaded module origins and complete Python source trees."""

    runtime = verified.source_reverification["runtime"]
    helper_root = _runtime_path(runtime, "accepted_helper_source_root")
    package_root = _runtime_path(runtime, "installed_homoeogwas_package_root")
    import homoeogwas
    import scripts

    package_portions = tuple(Path(path).resolve() for path in homoeogwas.__path__)
    script_portions = tuple(Path(path).resolve() for path in scripts.__path__)
    if package_portions != (package_root,):
        raise AuthorityBlocked("loaded HomoeoGWAS package path is not exclusive")
    if script_portions != (helper_root / "scripts",):
        raise AuthorityBlocked("loaded scripts namespace path is not exclusive")
    for name, expected_root in (
        ("homoeogwas.group_family", package_root),
        ("homoeogwas.interact", package_root),
        ("homoeogwas.omnib_family", package_root),
        ("scripts.benchmarks.v201.contracts", helper_root),
        ("scripts.benchmarks.v201.simulation", helper_root),
        ("scripts.benchmarks.v201.track_omnib", helper_root),
    ):
        module = sys.modules.get(name)
        origin = Path(str(getattr(module, "__file__", ""))).resolve()
        if module is None or not origin.is_relative_to(expected_root):
            raise AuthorityBlocked(f"loaded numerical module origin is invalid: {name}")
    package_hash = python_source_tree_sha256(
        package_root,
        domain="homoeogwas-package-source-v1",
    )
    helper_hash = python_source_tree_sha256(
        helper_root / "scripts/benchmarks/v201",
        domain="homoeogwas-v201-helper-source-v1",
    )
    if package_hash != ACCEPTED_HOMOEO_GWAS_PACKAGE_SOURCE_SHA256 or (
        package_hash != runtime.get("homoeogwas_package_source_sha256")
    ):
        raise AuthorityBlocked("installed HomoeoGWAS package tree hash differs")
    if helper_hash != ACCEPTED_V201_HELPER_SOURCE_SHA256 or (
        helper_hash != runtime.get("v201_helper_source_sha256")
    ):
        raise AuthorityBlocked("accepted v201 helper tree hash differs")


def _require_sha256(value: object, label: str) -> str:
    if not _is_sha256(value):
        raise AuthorityBlocked(f"invalid SHA-256 for {label}")
    return str(value)


def _absolute_path(value: object, label: str) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        raise AuthorityBlocked(f"{label} must use an absolute path")
    return path


def verify_materialization_authority(
    authority_path: Path,
    *,
    expected_authority_path: Path,
    task_root: Path,
    expected_successor_design_sha256: str,
    expected_worker_decision_sha256: str,
) -> VerifiedMaterializationAuthority:
    """Rehash every authority dependency without importing numerical code."""

    if authority_path.resolve() != expected_authority_path.resolve():
        raise AuthorityBlocked("authority file is not at the exact reviewed path")
    record = _mapping(authority_path)
    if record.get("schema") != "homoeogwas-bm-native-qa-materialization-authority-v1":
        raise AuthorityBlocked("materialization authority schema is invalid")
    if record.get("response_materialization_authorized") is not True:
        raise AuthorityBlocked("response_materialization_authorized=false")
    if record.get("execution_authorized") is not False:
        raise AuthorityBlocked("execution_authorized must remain false")
    if record.get("run_namespace") != "qa_real80_njobs128_v1":
        raise AuthorityBlocked("materialization run namespace is invalid")

    flat_hash_fields = (
        "successor_design_sha256",
        "decision_sha256",
        "qa_design_hash",
        "prospective_inventory_sha256",
        "runner_review_sha256",
        "source_input_reverification_sha256",
    )
    hashes = {
        field: _require_sha256(record.get(field), field) for field in flat_hash_fields
    }
    if hashes["qa_design_hash"] == V1_QA_DESIGN_HASH or (
        hashes["prospective_inventory_sha256"] == V1_INVENTORY_SHA256
    ):
        raise AuthorityBlocked("v1 identity is forbidden for successor materialization")
    if hashes["successor_design_sha256"] != expected_successor_design_sha256:
        raise AuthorityBlocked("successor design hash is not the reviewed identity")
    if hashes["decision_sha256"] != expected_worker_decision_sha256:
        raise AuthorityBlocked("worker decision hash is not the reviewed identity")

    raw_paths = record.get("paths")
    if not isinstance(raw_paths, dict):
        raise AuthorityBlocked("authority paths mapping is missing")
    paths = {
        key: _absolute_path(raw_paths.get(key), key)
        for key in (
            "successor_design",
            "decision",
            "prospective_inventory",
            "runner_review",
            "source_input_reverification",
            "artifact_root",
            "context_evidence",
            "amendment",
            "fixture_manifest",
        )
    }
    for path_key, hash_key in (
        ("successor_design", "successor_design_sha256"),
        ("decision", "decision_sha256"),
        ("prospective_inventory", "prospective_inventory_sha256"),
        ("runner_review", "runner_review_sha256"),
        ("source_input_reverification", "source_input_reverification_sha256"),
    ):
        try:
            observed = _sha256_file(paths[path_key])
        except OSError as exc:
            raise AuthorityBlocked(f"bound file is unavailable: {paths[path_key]}") from exc
        if observed != hashes[hash_key]:
            raise AuthorityBlocked(f"bound file hash mismatch: {path_key}")

    inventory = _mapping(paths["prospective_inventory"], json_format=True)
    if inventory.get("schema") != "homoeogwas-bm-native-qa-prospective-inventory-v2":
        raise AuthorityBlocked("prospective inventory is not successor v2")
    if inventory.get("response_materialization_authorized") is not False or (
        inventory.get("execution_authorized") is not False
    ):
        raise AuthorityBlocked("prospective inventory authorization flags are invalid")
    if inventory.get("qa_design_hash") != hashes["qa_design_hash"]:
        raise AuthorityBlocked("authority and inventory design hashes differ")
    bindings = inventory.get("source_bindings")
    if not isinstance(bindings, dict) or (
        bindings.get("successor_design_sha256")
        != hashes["successor_design_sha256"]
        or bindings.get("worker_decision_sha256") != hashes["decision_sha256"]
    ):
        raise AuthorityBlocked("inventory source bindings differ from authority")
    design_payload = inventory.get("design_payload")
    if not isinstance(design_payload, dict):
        raise AuthorityBlocked("inventory design payload is missing")
    if _mapping_sha256(design_payload) != hashes["qa_design_hash"]:
        raise AuthorityBlocked("inventory design payload hash is invalid")
    future = design_payload.get("future_artifacts")
    if not isinstance(future, dict) or future.get("root") != str(
        paths["artifact_root"]
    ):
        raise AuthorityBlocked("artifact root differs from the frozen inventory")
    if paths["artifact_root"].exists():
        raise AuthorityBlocked("exclusive artifact root already exists")
    lock = paths["artifact_root"].parent / (
        f".{paths['artifact_root'].name}.materialization-lock"
    )
    if lock.exists():
        raise AuthorityBlocked("exclusive materialization lock already exists")

    authority_runner_hashes = record.get("runner_test_sha256s")
    inventory_runner_hashes = inventory.get("runner_test_sha256s")
    if (
        not isinstance(authority_runner_hashes, dict)
        or not authority_runner_hashes
        or authority_runner_hashes != inventory_runner_hashes
    ):
        raise AuthorityBlocked("runner/test hash mappings differ")
    if _mapping_sha256(authority_runner_hashes) == V1_RUNNER_TEST_MAPPING_SHA256:
        raise AuthorityBlocked("v1 runner/test identity is forbidden")
    root = task_root.resolve()
    observed_members = enumerate_runner_test_sources(root)
    if set(observed_members) != set(authority_runner_hashes):
        raise AuthorityBlocked("runner/test source path set differs from authority")
    for relative, expected in authority_runner_hashes.items():
        _require_sha256(expected, f"runner/test {relative}")
        member = Path(str(relative))
        if member.is_absolute() or ".." in member.parts:
            raise AuthorityBlocked("runner/test member path escapes task root")
        path = observed_members[str(relative)].resolve()
        try:
            observed = _sha256_file(path)
        except OSError as exc:
            raise AuthorityBlocked(
                f"runner/test source is unavailable: {relative}"
            ) from exc
        if not path.is_relative_to(root) or observed != expected:
            raise AuthorityBlocked(f"runner/test hash mismatch: {relative}")

    if record.get("runner_review_verdict") != "ACCEPT":
        raise AuthorityBlocked("runner review verdict is not ACCEPT")
    try:
        review_text = paths["runner_review"].read_text(encoding="utf-8")
    except OSError as exc:
        raise AuthorityBlocked("runner review is unavailable") from exc
    if "FINAL_CODE_REVIEW_VERDICT: ACCEPT" not in review_text:
        raise AuthorityBlocked("runner review lacks the final ACCEPT marker")

    for path_key, binding_key in (
        ("context_evidence", "context_evidence_sha256"),
        ("amendment", "amendment_sha256"),
        ("fixture_manifest", "fixture_manifest_sha256"),
    ):
        expected = _require_sha256(bindings.get(binding_key), binding_key)
        try:
            observed = _sha256_file(paths[path_key])
        except OSError as exc:
            raise AuthorityBlocked(f"bound file is unavailable: {path_key}") from exc
        if observed != expected:
            raise AuthorityBlocked(f"inventory source binding mismatch: {path_key}")
    context_evidence = _mapping(paths["context_evidence"], json_format=True)
    if context_evidence.get("phenotype_values_read") is not False:
        raise AuthorityBlocked("context evidence is not phenotype-blind")

    reverify = _mapping(paths["source_input_reverification"])
    if reverify.get("schema") != (
        "homoeogwas-bm-native-qa-source-input-reverification-v2"
    ) or reverify.get("status") != "VERIFIED":
        raise AuthorityBlocked("source/input reverification status is not VERIFIED")
    files = reverify.get("files")
    if not isinstance(files, list) or not files:
        raise AuthorityBlocked("source/input reverification file list is empty")
    for row in files:
        if not isinstance(row, dict):
            raise AuthorityBlocked("source/input reverification row is invalid")
        path = _absolute_path(row.get("path"), "reverified file")
        expected = _require_sha256(row.get("sha256"), f"reverified file {path}")
        try:
            observed = _sha256_file(path)
        except OSError as exc:
            raise AuthorityBlocked(f"reverified file is unavailable: {path}") from exc
        if observed != expected:
            raise AuthorityBlocked(f"reverified file hash mismatch: {path}")

    return VerifiedMaterializationAuthority(
        qa_design_hash=hashes["qa_design_hash"],
        inventory=inventory,
        artifact_root=paths["artifact_root"],
        paths=paths,
        binding_hashes={
            "materialization_authority_sha256": _sha256_file(authority_path),
            **hashes,
        },
        source_reverification=reverify,
    )
