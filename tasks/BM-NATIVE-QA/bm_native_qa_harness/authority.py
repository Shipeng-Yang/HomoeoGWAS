from __future__ import annotations

import hashlib
import importlib
import importlib.machinery
import importlib.metadata
import json
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from . import SUCCESSOR_RUN_NAMESPACE, THREAD_ENV_NAMES


class AuthorityBlocked(RuntimeError):
    """The reviewed response-materialization authority is not exact."""


V1_QA_DESIGN_HASH = (
    "65ba6556d86db4d27b03169ff70d5b39796c5968ce8ece9f782e866995cf5d5e"
)
V1_INVENTORY_SHA256 = (
    "7ded836fa5348ba0b24ca6fdfa061393172f5813105f16e31dbd6bb85ea25e66"
)
FAILED_NJOBS128_V1_QA_DESIGN_HASH = (
    "dd89dde4cd9d364ca431cbae33b4fc9cf550d1ad887e4ab88999635574a33ac0"
)
FAILED_NJOBS128_V1_INVENTORY_SHA256 = (
    "be2d091e0a788cb8f0831546f0e95c0ba25613d7de09cb0fc57794666078c7d7"
)
# Canonical JSON SHA-256 of runner_test_sha256s from
# 3aa5e07f0c87575fae90463943b17ae141662cc1:tasks/BM-NATIVE-QA/
# prospective-inventory.v1.json. A test derives this literal from those bytes.
V1_RUNNER_TEST_MAPPING_SHA256 = (
    "342fb67de133dc812a10a34e68b97dc04bfa16e538f6e9a79b452d92cd0132bb"
)
FAILED_NJOBS128_V1_RUNNER_TEST_MAPPING_SHA256 = (
    "5fddf212d0c190a27ed1f9a68277b16bc335b9a889049008218e60bdadfa6a2e"
)
REJECTED_V2_QA_DESIGN_HASHES = frozenset(
    {
        "523194f359e8fc0f07c30d345d5cef9eb3968456c8890277fd6c2bbc3ea3eca2",
        "383e6322c3cd6e7deccc8542f22f8a31d3e6638d92a0e03a62f1017542ff0c09",
        "8a527004528387f4872f31030a33d6b0129b81f2d891565e88974713fc41562e",
        "62c8cd9a1ca3ccbfae8664813cb34c3b0166ba74efd00b738a4645b565172b75",
    }
)
REJECTED_V2_INVENTORY_SHA256S = frozenset(
    {
        "584358ded7e8c7f3e028af5afac5834ee06fc4c85a1284bc6f966b039629dbb3",
        "16b67d9030ce32cc8a67f3551640a7d7470063494b3155eb7ac1b8ef2c204d96",
        "c0361c4baab8b7b5ffe9781abb14c9305ed357c66625bdf595313e0c0ffa8e85",
        "177a7db1104ba907eb78a9eca4be5bb2713046ce2fcc37061299871f21a79bc6",
    }
)
REJECTED_V2_RUNNER_TEST_MAPPING_SHA256S = frozenset(
    {
        "41c01cce14bfb214cfb654af1e9ec14f48bfd9e7dd2317984a8373ec42e0efd3",
        "6a59398b21e32228fb9456e11a8a71a9ea1bf41e2493b18a5ec201d260b94931",
        "931c4b5cd88b960e971f7ebf37d3fe3231c73db68c28f9d4531acf3d5ba1fbca",
        "13d8ef76a5e2c3bb9031c825ed485f761987189499393f875a80bfdcbfef9dbe",
    }
)
REJECTED_V3_QA_DESIGN_HASHES = frozenset(
    {
        "a78e2eca7ee9beab05d83167ac2b051b8db02d10b428cf08251d7852e56f1968",
        "f741abbd968bb76bd56340cf715a0cc7031119388cb0de6d95d7b44cae5eaf7e",
        "0b668c77c6792d566a18bc068564ec7736b177ad50105199e3f9490aed5b77fa",
        "ff8e917324aa5227b6fd4d1501ed3c08966b9f45cb5d10ec45f369d66c9d9215",
    }
)
REJECTED_V3_INVENTORY_SHA256S = frozenset(
    {
        "9470941e68661f32a583fb04c72dd6aacdb5c6f35dc5f0aa6f1a5ca677150662",
        "ed97d365a0b5f0cde7d4997ced0753ab44a14bb1cda4da758316569cc0c976db",
        "1408fe7e9d5846cc7d785bda6d66be981b4f48346ca7de8903bb4fe3415a675b",
        "e6d8a3ed2f2ab75bddafdcd4628526f15fe70d68d9e17a1f6f53e833c78b81ca",
    }
)
REJECTED_V3_RUNNER_TEST_MAPPING_SHA256S = frozenset(
    {
        "af2328831e45d5903700381fd5960e883074d1f6da4e67f7ea0e1c1299e27800",
        "d0bd1cc0a9c12dfd0b4b8aa0acadc0470a18b304a0fe891baa748de1d4e661fd",
        "13261cc62cdd3e9396f9a6cf236a3291fcec0f4735b82ab6f96401fb9c588482",
        "6253c131a42ca6402b17736ac0076a367a41232a5e96b3b954f30fcaa87de6f6",
    }
)
REJECTED_V4_QA_DESIGN_HASHES = frozenset(
    {"947aab2d9c1a1682e42f03e76ea71a4447df43e6451094d93a42ac04aea13d15"}
)
REJECTED_V4_INVENTORY_SHA256S = frozenset(
    {"899ad4efd07383553f87560bf981d847ca4bb9d67bb9e2aa8e035497788df3f4"}
)
REJECTED_V4_RUNNER_TEST_MAPPING_SHA256S = frozenset(
    {"c169100739ddcc36d3343da0e34b4e39116e8f3b19f8581e490fd44223335fe1"}
)
ACCEPTED_PRODUCT_COMMIT = "9d7faee655c020d84bdd6a6fba33b42a8428aea6"
ACCEPTED_PRODUCT_TREE = "92355ddcbb2cd3d749169291fcbf6065eb556311"
ACCEPTED_HOMOEO_GWAS_PACKAGE_SOURCE_SHA256 = (
    "4f249d698d94123721f9c94b6544dbd8cc2f2df9aaf04dd6776a351c035e945b"
)
ACCEPTED_V201_HELPER_SOURCE_SHA256 = (
    "b7935d3b51a59efcb1d107cdcf517c1c6a5621bffd025fae26477a645e1067ab"
)
REQUIRED_RUNTIME_DEPENDENCIES = (
    ("bed_reader", "bed-reader"),
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("pandas", "pandas"),
    ("sklearn", "scikit-learn"),
    ("matplotlib", "matplotlib"),
    ("yaml", "PyYAML"),
    ("pydantic", "pydantic"),
    ("joblib", "joblib"),
    ("threadpoolctl", "threadpoolctl"),
    ("pysam", "pysam"),
)


@dataclass(frozen=True)
class VerifiedMaterializationAuthority:
    run_namespace: str
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


def verify_materialization_launch_contract(
    verified: VerifiedMaterializationAuthority,
    *,
    task_root: Path,
    authority_path: Path,
    out_path: Path,
    observed_cwd: Path | None = None,
    observed_pythonpath: str | None = None,
    observed_python_no_user_site: str | None = None,
) -> None:
    """Verify the complete, authority-bound materialization launch contract."""

    runtime = verified.source_reverification.get("runtime")
    if not isinstance(runtime, dict):
        raise AuthorityBlocked("source reverification lacks launch contract")
    expected_task_root = _runtime_path(runtime, "harness_task_root")
    helper_root = _runtime_path(runtime, "accepted_helper_source_root")
    expected_argv = [
        "materialize",
        "--authority",
        str(authority_path.resolve()),
        "--out",
        str(out_path.resolve()),
    ]
    expected_record = {
        "cwd": str(expected_task_root),
        "pythonpath": str(helper_root),
        "python_no_user_site": True,
        "argv": expected_argv,
    }
    if runtime.get("materialization_launch") != expected_record:
        raise AuthorityBlocked("recorded materialization launch contract differs")
    actual_cwd = (observed_cwd or Path.cwd()).resolve()
    raw_pythonpath = (
        observed_pythonpath
        if observed_pythonpath is not None
        else os.environ.get("PYTHONPATH", "")
    )
    actual_pythonpath = tuple(
        Path(entry).resolve() for entry in raw_pythonpath.split(os.pathsep) if entry
    )
    no_user_site = (
        observed_python_no_user_site
        if observed_python_no_user_site is not None
        else os.environ.get("PYTHONNOUSERSITE")
    )
    if (
        task_root.resolve() != expected_task_root
        or actual_cwd != expected_task_root
        or actual_pythonpath != (helper_root,)
        or no_user_site != "1"
    ):
        raise AuthorityBlocked("effective materialization launch contract differs")


def _dependency_file_rows(reverification: Mapping[str, Any]) -> dict[Path, str]:
    rows = reverification.get("files")
    if not isinstance(rows, list):
        raise AuthorityBlocked("source reverification file list is missing")
    bound: dict[Path, str] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise AuthorityBlocked("source reverification file row is invalid")
        path = _absolute_path(row.get("path"), "runtime dependency file").resolve()
        digest = _require_sha256(row.get("sha256"), f"runtime dependency {path}")
        previous = bound.setdefault(path, digest)
        if previous != digest:
            raise AuthorityBlocked(f"conflicting runtime dependency hash: {path}")
    return bound


def _require_bound_dependency_file(
    path_value: object,
    digest_value: object,
    *,
    label: str,
    bound_files: Mapping[Path, str],
) -> Path:
    path = _absolute_path(path_value, label).resolve()
    expected = _require_sha256(digest_value, label)
    if bound_files.get(path) != expected:
        raise AuthorityBlocked(f"runtime dependency file is not authority-bound: {path}")
    try:
        observed = _sha256_file(path)
    except OSError as exc:
        raise AuthorityBlocked(f"runtime dependency file is unavailable: {path}") from exc
    if observed != expected:
        raise AuthorityBlocked(f"runtime dependency file hash mismatch: {path}")
    return path


def _native_distribution_files(distribution: Any) -> tuple[Path, ...]:
    native_suffixes = (".dylib", ".dll", ".pyd")
    return tuple(
        sorted(
            {
                Path(distribution.locate_file(item)).resolve()
                for item in distribution.files or ()
                if ".so" in Path(str(item)).name
                or Path(str(item)).name.endswith(native_suffixes)
            },
            key=str,
        )
    )


def _distribution_control_files(
    distribution: Any,
    filename: str,
) -> tuple[Path, ...]:
    return tuple(
        sorted(
            {
                Path(distribution.locate_file(item)).resolve()
                for item in distribution.files or ()
                if Path(str(item)).name == filename
                and any(part.endswith(".dist-info") for part in Path(str(item)).parts)
            },
            key=str,
        )
    )


def _no_user_site_active() -> bool:
    import site as site_module

    return sys.flags.no_user_site == 1 and site_module.ENABLE_USER_SITE is False


def _canonical_threadpools(rows: object) -> list[dict[str, object]]:
    if not isinstance(rows, list):
        raise AuthorityBlocked("runtime dependency threadpool snapshot is invalid")
    fields = (
        "user_api",
        "internal_api",
        "num_threads",
        "prefix",
        "filepath",
        "version",
        "threading_layer",
        "architecture",
    )
    canonical = [
        {field: row.get(field) for field in fields if field in row}
        for row in rows
        if isinstance(row, dict)
    ]
    if len(canonical) != len(rows):
        raise AuthorityBlocked("runtime dependency threadpool row is invalid")
    return sorted(
        canonical,
        key=lambda row: (
            str(row.get("filepath", "")),
            str(row.get("prefix", "")),
            str(row.get("internal_api", "")),
        ),
    )


def verify_runtime_dependencies(
    verified: VerifiedMaterializationAuthority,
) -> tuple[dict[str, object], ...]:
    """Import and verify the exact isolated dependency stack before the lock."""

    reverification = verified.source_reverification
    runtime = reverification.get("runtime")
    dependency_record = reverification.get("runtime_dependencies")
    if not isinstance(runtime, dict) or not isinstance(dependency_record, dict):
        raise AuthorityBlocked("runtime dependency identity is missing")
    raw_required = dependency_record.get("required")
    if (
        not isinstance(raw_required, list)
        or len(raw_required) != len(REQUIRED_RUNTIME_DEPENDENCIES)
        or not all(isinstance(row, dict) for row in raw_required)
    ):
        raise AuthorityBlocked("runtime dependency set is invalid")
    observed_names = tuple(
        (row.get("module"), row.get("distribution"))
        for row in raw_required
    )
    if observed_names != REQUIRED_RUNTIME_DEPENDENCIES:
        raise AuthorityBlocked("runtime dependency set differs from reviewed design")

    thread_environment = dependency_record.get("thread_environment")
    if not isinstance(thread_environment, dict) or thread_environment != {
        name: "1" for name in THREAD_ENV_NAMES
    }:
        raise AuthorityBlocked("recorded native thread environment is invalid")
    if any(os.environ.get(name) != "1" for name in THREAD_ENV_NAMES):
        raise AuthorityBlocked("effective native thread environment is not one")

    prefix = _runtime_path(runtime, "sys_prefix_realpath")
    site_packages = _runtime_path(runtime, "installed_site_packages")
    try:
        config = (prefix / "pyvenv.cfg").read_text(encoding="utf-8")
    except OSError as exc:
        raise AuthorityBlocked("isolated runtime pyvenv.cfg is unavailable") from exc
    config_values = {
        key.strip().lower(): value.strip().lower()
        for line in config.splitlines()
        if "=" in line
        for key, value in (line.split("=", 1),)
    }
    if dependency_record.get("include_system_site_packages") is not False or (
        config_values.get("include-system-site-packages") != "false"
    ):
        raise AuthorityBlocked("runtime inherits system site-packages")
    if (
        dependency_record.get("python_no_user_site") is not True
        or not _no_user_site_active()
    ):
        raise AuthorityBlocked("runtime user site must be disabled")

    bound_files = _dependency_file_rows(reverification)
    verified_rows: list[dict[str, object]] = []
    imported_modules: dict[str, Any] = {}
    for row in raw_required:
        module_name = str(row["module"])
        distribution_name = str(row["distribution"])
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            raise AuthorityBlocked(
                f"runtime dependency import failed: {module_name}: {exc}"
            ) from exc
        imported_modules[module_name] = module
        try:
            distribution = importlib.metadata.distribution(distribution_name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise AuthorityBlocked(
                f"runtime dependency distribution is missing: {distribution_name}"
            ) from exc
        version = str(row.get("version", ""))
        if not version or distribution.version != version:
            raise AuthorityBlocked(
                f"runtime dependency version mismatch: {distribution_name}"
            )
        module_origin = Path(str(getattr(module, "__file__", ""))).resolve()
        expected_origin = _absolute_path(
            row.get("module_origin"), f"runtime dependency origin {module_name}"
        ).resolve()
        if module_origin != expected_origin or not module_origin.is_relative_to(
            site_packages
        ):
            raise AuthorityBlocked(
                f"runtime dependency origin mismatch: {module_name}"
            )
        metadata_path = _require_bound_dependency_file(
            row.get("metadata_path"),
            row.get("metadata_sha256"),
            label=f"runtime dependency METADATA {distribution_name}",
            bound_files=bound_files,
        )
        record_path = _require_bound_dependency_file(
            row.get("record_path"),
            row.get("record_sha256"),
            label=f"runtime dependency RECORD {distribution_name}",
            bound_files=bound_files,
        )
        if (
            metadata_path.parent != record_path.parent
            or not metadata_path.parent.is_relative_to(site_packages)
            or _distribution_control_files(distribution, "METADATA")
            != (metadata_path,)
            or _distribution_control_files(distribution, "RECORD") != (record_path,)
        ):
            raise AuthorityBlocked(
                f"runtime dependency dist-info origin mismatch: {distribution_name}"
            )
        raw_native = row.get("native_extensions")
        if not isinstance(raw_native, list):
            raise AuthorityBlocked(
                f"runtime dependency native extension list is invalid: {distribution_name}"
            )
        expected_native = []
        for native in raw_native:
            if not isinstance(native, dict):
                raise AuthorityBlocked("runtime dependency native extension row is invalid")
            expected_native.append(
                _require_bound_dependency_file(
                    native.get("path"),
                    native.get("sha256"),
                    label=f"runtime dependency native extension {distribution_name}",
                    bound_files=bound_files,
                )
            )
        if tuple(sorted(expected_native, key=str)) != _native_distribution_files(
            distribution
        ):
            raise AuthorityBlocked(
                f"runtime dependency native extension set mismatch: {distribution_name}"
            )
        verified_rows.append(
            {
                "module": module_name,
                "distribution": distribution_name,
                "version": version,
                "module_origin": str(module_origin),
            }
        )

    threadpoolctl = imported_modules["threadpoolctl"]
    observed_pools = _canonical_threadpools(threadpoolctl.threadpool_info())
    expected_pools = _canonical_threadpools(dependency_record.get("threadpool_info"))
    if not observed_pools or not expected_pools:
        raise AuthorityBlocked("no native thread pool was observed")
    if any(row.get("num_threads") != 1 for row in observed_pools):
        raise AuthorityBlocked("loaded native thread pool is not limited to one")
    if observed_pools != expected_pools:
        raise AuthorityBlocked("loaded native threadpool identity differs")
    return tuple(verified_rows)


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
    if record.get("schema") != "homoeogwas-bm-native-qa-materialization-authority-v2":
        raise AuthorityBlocked("materialization authority schema is invalid")
    if record.get("response_materialization_authorized") is not True:
        raise AuthorityBlocked("response_materialization_authorized=false")
    if record.get("execution_authorized") is not False:
        raise AuthorityBlocked("execution_authorized must remain false")
    run_namespace = record.get("run_namespace")
    if run_namespace != SUCCESSOR_RUN_NAMESPACE:
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
    if hashes["qa_design_hash"] in {
        V1_QA_DESIGN_HASH,
        FAILED_NJOBS128_V1_QA_DESIGN_HASH,
    } or hashes["prospective_inventory_sha256"] in {
        V1_INVENTORY_SHA256,
        FAILED_NJOBS128_V1_INVENTORY_SHA256,
    }:
        raise AuthorityBlocked(
            "v1 identity is forbidden for successor materialization"
        )
    if (
        hashes["qa_design_hash"] in REJECTED_V2_QA_DESIGN_HASHES
        or hashes["prospective_inventory_sha256"]
        in REJECTED_V2_INVENTORY_SHA256S
    ):
        raise AuthorityBlocked("rejected v2 identity is forbidden")
    if (
        hashes["qa_design_hash"] in REJECTED_V3_QA_DESIGN_HASHES
        or hashes["prospective_inventory_sha256"]
        in REJECTED_V3_INVENTORY_SHA256S
    ):
        raise AuthorityBlocked("rejected v3 identity is forbidden")
    if (
        hashes["qa_design_hash"] in REJECTED_V4_QA_DESIGN_HASHES
        or hashes["prospective_inventory_sha256"]
        in REJECTED_V4_INVENTORY_SHA256S
    ):
        raise AuthorityBlocked("rejected v4 identity is forbidden")
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
    if inventory.get("schema") != "homoeogwas-bm-native-qa-prospective-inventory-v5":
        raise AuthorityBlocked("prospective inventory is not the active successor")
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
    runner_mapping_sha256 = _mapping_sha256(authority_runner_hashes)
    if runner_mapping_sha256 in {
        V1_RUNNER_TEST_MAPPING_SHA256,
        FAILED_NJOBS128_V1_RUNNER_TEST_MAPPING_SHA256,
    }:
        raise AuthorityBlocked("v1 runner/test identity is forbidden")
    if runner_mapping_sha256 in REJECTED_V2_RUNNER_TEST_MAPPING_SHA256S:
        raise AuthorityBlocked("rejected v2 runner/test identity is forbidden")
    if runner_mapping_sha256 in REJECTED_V3_RUNNER_TEST_MAPPING_SHA256S:
        raise AuthorityBlocked("rejected v3 runner/test identity is forbidden")
    if runner_mapping_sha256 in REJECTED_V4_RUNNER_TEST_MAPPING_SHA256S:
        raise AuthorityBlocked("rejected v4 runner/test identity is forbidden")
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
    review_mapping_marker = (
        f"REVIEWED_RUNNER_TEST_MAPPING_SHA256: {runner_mapping_sha256}"
    )
    if review_mapping_marker not in review_text.splitlines():
        raise AuthorityBlocked("runner review does not bind runner/test mapping")

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
        "homoeogwas-bm-native-qa-source-input-reverification-v3"
    ):
        raise AuthorityBlocked("source/input reverification schema is not v3")
    if reverify.get("status") != "VERIFIED":
        raise AuthorityBlocked("source/input reverification status is not VERIFIED")
    runtime_dependencies = reverify.get("runtime_dependencies")
    required_dependency_keys = {
        "include_system_site_packages",
        "python_no_user_site",
        "required",
        "thread_environment",
        "threadpool_info",
    }
    if not isinstance(runtime_dependencies, dict) or (
        set(runtime_dependencies) != required_dependency_keys
    ):
        raise AuthorityBlocked("source/input runtime dependency record is incomplete")
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
        run_namespace=str(run_namespace),
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
