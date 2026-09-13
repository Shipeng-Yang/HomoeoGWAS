from __future__ import annotations

import hashlib
import importlib
import os
import site
import subprocess
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest
from bm_native_qa_harness import authority, cli

REQUIRED = (
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
TASK_ROOT = Path(__file__).resolve().parents[1]
WORKTREE_ROOT = Path(__file__).resolve().parents[3]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verified_dependency_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> authority.VerifiedMaterializationAuthority:
    monkeypatch.setattr(site, "ENABLE_USER_SITE", False)
    monkeypatch.setattr(authority, "_no_user_site_active", lambda: True)
    prefix = tmp_path / "venv"
    site_packages = prefix / "lib/python3.12/site-packages"
    site_packages.mkdir(parents=True)
    (prefix / "pyvenv.cfg").write_text(
        "include-system-site-packages = false\n", encoding="utf-8"
    )
    files = []
    required = []
    for module, distribution in REQUIRED:
        dist_info = site_packages / f"{distribution}-1.0.dist-info"
        dist_info.mkdir()
        metadata = dist_info / "METADATA"
        record = dist_info / "RECORD"
        metadata.write_text(f"Name: {distribution}\nVersion: 1.0\n", encoding="utf-8")
        record.write_text("bound\n", encoding="utf-8")
        for path in (metadata, record):
            files.append({"path": str(path), "sha256": _sha256(path)})
        native_extensions = []
        if module == "numpy":
            native_path = site_packages / "numpy/core/_multiarray_test.so"
            native_path.parent.mkdir(parents=True)
            native_path.write_bytes(b"native-extension\n")
            native_sha256 = _sha256(native_path)
            files.append({"path": str(native_path), "sha256": native_sha256})
            native_extensions.append(
                {"path": str(native_path), "sha256": native_sha256}
            )
        required.append(
            {
                "module": module,
                "distribution": distribution,
                "version": "1.0",
                "module_origin": str(site_packages / module / "__init__.py"),
                "metadata_path": str(metadata),
                "metadata_sha256": _sha256(metadata),
                "record_path": str(record),
                "record_sha256": _sha256(record),
                "native_extensions": native_extensions,
            }
        )
    source_reverification = {
        "runtime": {
            "sys_prefix_realpath": str(prefix),
            "installed_site_packages": str(site_packages),
        },
        "runtime_dependencies": {
            "include_system_site_packages": False,
            "python_no_user_site": True,
            "required": required,
            "thread_environment": {name: "1" for name in cli.THREAD_ENV_NAMES},
            "threadpool_info": [
                {
                    "user_api": "blas",
                    "internal_api": "openblas",
                    "num_threads": 1,
                    "prefix": "libopenblas",
                    "filepath": str(
                        site_packages / "numpy/core/_multiarray_test.so"
                    ),
                    "version": "test",
                    "threading_layer": "pthreads",
                    "architecture": "test",
                }
            ],
        },
        "files": files,
    }
    return authority.VerifiedMaterializationAuthority(
        run_namespace="qa_real80_njobs128_v4",
        qa_design_hash="1" * 64,
        inventory={},
        artifact_root=tmp_path / "materialized" / "njobs128-v4",
        paths={},
        binding_hashes={},
        source_reverification=source_reverification,
    )


class _FakeDistribution:
    def __init__(self, site_packages: Path, distribution: str) -> None:
        self.version = "1.0"
        self._site_packages = site_packages
        control_root = Path(f"{distribution}-1.0.dist-info")
        files = [control_root / "METADATA", control_root / "RECORD"]
        if distribution == "numpy":
            files.append(Path("numpy/core/_multiarray_test.so"))
        self.files = tuple(files)

    def locate_file(self, item: Path) -> Path:
        return self._site_packages / item


def _install_dependency_stubs(
    verified: authority.VerifiedMaterializationAuthority,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dependency_record = verified.source_reverification["runtime_dependencies"]
    required = dependency_record["required"]
    site_packages = Path(
        verified.source_reverification["runtime"]["installed_site_packages"]
    )
    modules = {
        row["module"]: SimpleNamespace(
            __file__=str(site_packages / row["module"] / "__init__.py")
        )
        for row in required
    }
    modules["threadpoolctl"].threadpool_info = lambda: [
        {
            "user_api": "blas",
            "internal_api": "openblas",
            "num_threads": 1,
            "prefix": "libopenblas",
            "filepath": str(site_packages / "numpy/core/_multiarray_test.so"),
            "version": "test",
            "threading_layer": "pthreads",
            "architecture": "test",
        }
    ]

    monkeypatch.setattr(importlib, "import_module", modules.__getitem__)
    monkeypatch.setattr(
        importlib.metadata,
        "distribution",
        lambda name: _FakeDistribution(site_packages, name),
    )
    monkeypatch.setitem(sys.modules, "threadpoolctl", modules["threadpoolctl"])


def test_cli_forces_all_native_thread_variables_to_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in cli.THREAD_ENV_NAMES:
        monkeypatch.setenv(name, "9")

    cli.enforce_single_native_thread_before_imports()

    assert {name: os.environ[name] for name in cli.THREAD_ENV_NAMES} == {
        name: "1" for name in cli.THREAD_ENV_NAMES
    }


def test_package_init_forces_threads_before_direct_bundle_import() -> None:
    env = os.environ.copy()
    env.update({name: "9" for name in cli.THREAD_ENV_NAMES})
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(
        (str(TASK_ROOT), str(WORKTREE_ROOT / "src"), str(WORKTREE_ROOT))
    )
    raw = subprocess.check_output(
        [
            sys.executable,
            "-c",
            "import os; import bm_native_qa_harness.bundle; "
            "print(','.join(os.environ[name] for name in "
            "bm_native_qa_harness.THREAD_ENV_NAMES))",
        ],
        env=env,
        text=True,
    )

    assert raw.strip() == "1,1,1,1"


def test_no_user_site_contract_requires_python_startup_flag() -> None:
    env = os.environ.copy()
    env.pop("PYTHONNOUSERSITE", None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = str(TASK_ROOT)
    command = (
        "from bm_native_qa_harness.authority import _no_user_site_active; "
        "print(_no_user_site_active())"
    )

    ordinary = subprocess.check_output(
        [sys.executable, "-c", command], env=env, text=True
    ).strip()
    isolated = subprocess.check_output(
        [sys.executable, "-s", "-c", command], env=env, text=True
    ).strip()

    assert ordinary == "False"
    assert isolated == "True"


def test_cli_maps_materialization_import_error_to_clean_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    verified = object()
    monkeypatch.setattr(cli, "_verify_materialization_cli", lambda _args: verified)
    monkeypatch.setattr(
        authority,
        "verify_runtime_import_isolation",
        lambda _verified, **_kwargs: None,
    )
    monkeypatch.setattr(
        authority,
        "verify_materialization_launch_contract",
        lambda _verified, **_kwargs: None,
    )
    monkeypatch.setattr(
        cli,
        "_activate_materialization",
        lambda _args, _verified: (_ for _ in ()).throw(
            ModuleNotFoundError("No module named 'scripts'", name="scripts")
        ),
    )

    exit_code = cli.main(
        [
            "materialize",
            "--authority",
            str(tmp_path / "authority.yaml"),
            "--out",
            str(tmp_path / "out"),
        ]
    )

    assert exit_code == 2
    assert "No module named 'scripts'" in capsys.readouterr().err


def test_dependency_gate_names_missing_bed_reader_before_v4_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verified = _verified_dependency_fixture(tmp_path, monkeypatch)
    original = importlib.import_module

    def missing_bed_reader(name: str):
        if name == "bed_reader":
            raise ModuleNotFoundError("No module named 'bed_reader'", name=name)
        return original(name)

    monkeypatch.setattr(importlib, "import_module", missing_bed_reader)

    with pytest.raises(authority.AuthorityBlocked, match="bed_reader"):
        authority.verify_runtime_dependencies(verified)

    lock = verified.artifact_root.parent / ".njobs128-v4.materialization-lock"
    assert not lock.exists()


def test_cli_dependency_failure_precedes_bundle_import_and_v4_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verified = _verified_dependency_fixture(tmp_path, monkeypatch)
    monkeypatch.delitem(sys.modules, "bm_native_qa_harness.bundle", raising=False)

    def fail_dependency_gate(_verified) -> None:
        raise authority.AuthorityBlocked("bed_reader is missing")

    monkeypatch.setattr(authority, "verify_runtime_dependencies", fail_dependency_gate)

    with pytest.raises(authority.AuthorityBlocked, match="bed_reader"):
        cli._activate_materialization(Namespace(), verified)

    assert "bm_native_qa_harness.bundle" not in sys.modules
    lock = verified.artifact_root.parent / ".njobs128-v4.materialization-lock"
    assert not lock.exists()


def test_dependency_gate_rejects_omitted_required_distribution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verified = _verified_dependency_fixture(tmp_path, monkeypatch)
    verified.source_reverification["runtime_dependencies"]["required"].pop()

    with pytest.raises(authority.AuthorityBlocked, match="dependency set"):
        authority.verify_runtime_dependencies(verified)


def test_dependency_gate_rejects_enabled_user_site(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verified = _verified_dependency_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(authority, "_no_user_site_active", lambda: False)

    with pytest.raises(authority.AuthorityBlocked, match="user site"):
        authority.verify_runtime_dependencies(verified)


def test_dependency_gate_requires_recorded_no_user_site_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verified = _verified_dependency_fixture(tmp_path, monkeypatch)
    verified.source_reverification["runtime_dependencies"].pop(
        "python_no_user_site"
    )

    with pytest.raises(authority.AuthorityBlocked, match="user site"):
        authority.verify_runtime_dependencies(verified)


def test_dependency_gate_accepts_exact_bound_isolated_stack(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verified = _verified_dependency_fixture(tmp_path, monkeypatch)
    _install_dependency_stubs(verified, monkeypatch)

    rows = authority.verify_runtime_dependencies(verified)

    assert len(rows) == 11
    assert tuple((row["module"], row["distribution"]) for row in rows) == REQUIRED


def test_dependency_gate_rejects_non_mapping_extra_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verified = _verified_dependency_fixture(tmp_path, monkeypatch)
    verified.source_reverification["runtime_dependencies"]["required"].append("junk")

    with pytest.raises(authority.AuthorityBlocked, match="dependency set"):
        authority.verify_runtime_dependencies(verified)


def test_dependency_gate_rejects_empty_threadpool_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verified = _verified_dependency_fixture(tmp_path, monkeypatch)
    verified.source_reverification["runtime_dependencies"]["threadpool_info"] = []
    _install_dependency_stubs(verified, monkeypatch)

    with pytest.raises(authority.AuthorityBlocked, match="no native thread pool"):
        authority.verify_runtime_dependencies(verified)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("version", "version mismatch"),
        ("origin", "origin mismatch"),
        ("native", "native extension set mismatch"),
        ("threadpool", "threadpool identity differs"),
    ],
)
def test_dependency_gate_rejects_bound_identity_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    match: str,
) -> None:
    verified = _verified_dependency_fixture(tmp_path, monkeypatch)
    dependency_record = verified.source_reverification["runtime_dependencies"]
    if mutation == "version":
        dependency_record["required"][0]["version"] = "2.0"
    elif mutation == "origin":
        dependency_record["required"][0]["module_origin"] = str(
            tmp_path / "outside.py"
        )
    elif mutation == "native":
        dependency_record["required"][1]["native_extensions"] = []
    else:
        dependency_record["threadpool_info"][0]["prefix"] = "different"
    _install_dependency_stubs(verified, monkeypatch)

    with pytest.raises(authority.AuthorityBlocked, match=match):
        authority.verify_runtime_dependencies(verified)
