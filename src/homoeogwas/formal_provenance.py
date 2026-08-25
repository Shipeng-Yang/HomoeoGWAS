"""Immutable executable/config identity for formal checkpointed launches."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import scipy

from . import __version__

PRE_RUN_MANIFEST_SCHEMA = "homoeogwas-wheat-f2143-edge-pre-run-manifest-v2"
BLAS_THREAD_VARIABLES = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
FORMAL_BLAS_THREAD_POLICY = {name: "1" for name in BLAS_THREAD_VARIABLES}


class FormalLaunchError(ValueError):
    """A formal launch is not identical to its frozen preparation record."""


@dataclass(frozen=True)
class VerifiedFormalLaunch:
    checkpoint_context: dict[str, Any]
    attestation_path: str


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_source_sha256(package_root: str | Path | None = None) -> str:
    """Hash ordered package-relative Python paths and bytes, not a version tag."""
    root = (
        Path(package_root).resolve()
        if package_root is not None else Path(__file__).resolve().parent
    )
    files = sorted(
        (path for path in root.rglob("*.py") if path.is_file()),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    if not files:
        raise FormalLaunchError(f"no HomoeoGWAS Python sources found under {root}")
    digest = hashlib.sha256(b"homoeogwas-package-source-v1\0")
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        body = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(body).to_bytes(8, "big"))
        digest.update(body)
    return digest.hexdigest()


def _git_output(root: Path, *args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", "") or str(exc)
        raise FormalLaunchError(
            f"cannot resolve formal git source identity: {detail.strip()}"
        ) from exc
    return completed.stdout.strip()


def capture_source_identity() -> dict[str, Any]:
    """Return committed source identity plus current worktree cleanliness."""
    root = Path(__file__).resolve().parents[2]
    commit = _git_output(root, "rev-parse", "HEAD")
    tree = _git_output(root, "rev-parse", "HEAD^{tree}")
    status = _git_output(root, "status", "--porcelain", "--untracked-files=normal")
    return {
        "git_commit": commit,
        "git_tree": tree,
        "package_source_sha256": package_source_sha256(),
        "source_clean": not bool(status),
    }


def _normalize_blas_policy(policy: Any) -> dict[str, str]:
    if not isinstance(policy, dict) or set(policy) != set(BLAS_THREAD_VARIABLES):
        raise FormalLaunchError(
            "formal provenance.blas_thread_policy must declare exactly "
            + ", ".join(BLAS_THREAD_VARIABLES)
        )
    normalized = {name: str(policy[name]) for name in BLAS_THREAD_VARIABLES}
    if any(value != "1" for value in normalized.values()):
        raise FormalLaunchError(
            "formal BLAS thread policy requires every declared thread count to be 1")
    return normalized


def runtime_fingerprint(policy: Any) -> dict[str, Any]:
    """Return stable inference-relevant runtime versions and declared policy."""
    return {
        "homoeogwas": str(__version__),
        "python": platform.python_version(),
        "numpy": str(np.__version__),
        "scipy": str(scipy.__version__),
        "blas_thread_policy": _normalize_blas_policy(policy),
    }


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                payload, handle, sort_keys=True, ensure_ascii=False,
                allow_nan=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _require_equal(label: str, expected: Any, actual: Any) -> None:
    if expected != actual:
        raise FormalLaunchError(
            f"formal {label} mismatch: expected {expected!r}, observed {actual!r}; "
            "regenerate the preparation bundle from this clean executable source")


def _formal_checkpoint_requested(raw_config: dict[str, Any]) -> bool:
    """Return whether raw config requests the canonical checkpointed omniB path."""
    interact = raw_config.get("interact")
    if not isinstance(interact, dict):
        return False
    if str(interact.get("statistic", "omniB")).lower() != "omnib":
        return False
    calibration = interact.get("calibration")
    if not isinstance(calibration, dict):
        return False
    checkpoint = calibration.get("checkpoint")
    return isinstance(checkpoint, dict) and checkpoint.get("enabled") is True


def verify_formal_launch(
    config_path: str | Path,
    raw_config: dict[str, Any],
) -> VerifiedFormalLaunch | None:
    """Verify and attest a formal config before any large genotype is loaded.

    Binding rule: the config stores only the pre-run-manifest reference; the
    manifest stores the SHA-256 of the final raw config bytes. Consequently the
    manifest does not hash itself, while any later config-byte change fails.
    """
    if not isinstance(raw_config, dict):
        return None
    provenance = raw_config.get("provenance")
    if not isinstance(provenance, dict) or not provenance.get("pre_run_manifest"):
        if _formal_checkpoint_requested(raw_config):
            raise FormalLaunchError(
                "checkpointed group omniB requires "
                "provenance.pre_run_manifest; regenerate the formal config "
                "instead of removing its provenance binding")
        return None

    config_path = Path(config_path).resolve(strict=True)
    manifest_path = Path(provenance["pre_run_manifest"])
    if not manifest_path.is_absolute():
        manifest_path = (config_path.parent / manifest_path).resolve()
    try:
        manifest_body = manifest_path.read_bytes()
        manifest = json.loads(manifest_body)
    except (OSError, json.JSONDecodeError) as exc:
        raise FormalLaunchError(
            f"cannot read formal pre-run manifest {manifest_path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise FormalLaunchError("formal pre-run manifest must be a JSON object")

    raw_config_sha256 = sha256_file(config_path)
    manifest_sha256 = hashlib.sha256(manifest_body).hexdigest()
    schema = manifest.get("schema")
    _require_equal("pre-run manifest schema", PRE_RUN_MANIFEST_SCHEMA, schema)
    expected_config_sha = (manifest.get("config") or {}).get("sha256")
    if expected_config_sha != raw_config_sha256:
        raise FormalLaunchError(
            "formal raw config SHA-256 mismatch (pre-run/config mismatch): "
            f"manifest expects {expected_config_sha!r}, "
            f"observed {raw_config_sha256}; regenerate the preparation bundle")

    expected_source = manifest.get("source_identity")
    if not isinstance(expected_source, dict):
        raise FormalLaunchError("formal manifest is missing source_identity")
    actual_source = capture_source_identity()
    if actual_source.get("source_clean") is not True:
        raise FormalLaunchError(
            "formal source_clean mismatch: the runtime worktree is dirty; "
            "launch from a clean detached snapshot")
    for field in ("git_commit", "git_tree", "package_source_sha256"):
        _require_equal(
            f"source_identity.{field}", expected_source.get(field),
            actual_source.get(field))
    _require_equal(
        "source_identity.source_clean", expected_source.get("source_clean"), True)

    policy = _normalize_blas_policy(provenance.get("blas_thread_policy"))
    actual_runtime = runtime_fingerprint(policy)
    expected_runtime = manifest.get("runtime_fingerprint")
    _require_equal("runtime_fingerprint", expected_runtime, actual_runtime)
    for name, expected in policy.items():
        observed = os.environ.get(name)
        if observed != expected:
            raise FormalLaunchError(
                f"formal BLAS thread policy mismatch for {name}: expected "
                f"{expected!r}, observed {observed!r}; export the declared policy "
                "before starting Python")

    context = {
        "raw_config_sha256": raw_config_sha256,
        "pre_run_manifest_sha256": manifest_sha256,
        "pre_run_manifest_schema": schema,
        "source_identity": actual_source,
        "runtime_fingerprint": actual_runtime,
    }
    out_dir = Path((raw_config.get("outputs") or {}).get("out_dir", ".")).resolve()
    attestation_path = out_dir / "provenance" / "launch_attestation.json"
    checks = {
        "config_matches_pre_run_manifest": {
            "passed": True,
            "expected": expected_config_sha,
            "actual": raw_config_sha256,
        },
        "pre_run_manifest_schema": {
            "passed": True,
            "expected": PRE_RUN_MANIFEST_SCHEMA,
            "actual": schema,
        },
        "source_identity": {
            "passed": True,
            "expected": expected_source,
            "actual": actual_source,
        },
        "runtime_fingerprint": {
            "passed": True,
            "expected": expected_runtime,
            "actual": actual_runtime,
        },
        "blas_thread_environment": {
            "passed": True,
            "expected": policy,
            "actual": {name: os.environ.get(name) for name in policy},
        },
    }
    _atomic_write_json(attestation_path, {
        "schema": "homoeogwas-formal-launch-attestation-v1",
        "status": "passed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "pre_run_manifest_path": str(manifest_path),
        "pre_run_manifest_sha256": manifest_sha256,
        "checks": checks,
        "checkpoint_manifest_context": context,
    })
    return VerifiedFormalLaunch(context, str(attestation_path))
