"""Opt-in, hash-keyed, fail-closed cache for phenotype-independent subgenome GRMs."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy

from . import __version__
from .formal_provenance import BLAS_THREAD_VARIABLES, sha256_file
from .io import plink_bim_sha256, plink_path

GRM_CACHE_ENTRY_SCHEMA = "homoeogwas-grm-cache-entry-v1"
GRM_CACHE_KEY_SCHEMA = "homoeogwas-grm-cache-key-v1"


class GRMCacheError(RuntimeError):
    """The GRM cache cannot be used or an entry cannot be trusted."""


def _canonical_json(values) -> bytes:
    return json.dumps(
        values, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()


def _array_identity(values: np.ndarray) -> dict:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode())
    digest.update(b"\0")
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode())
    digest.update(b"\0")
    digest.update(memoryview(array).cast("B"))
    return {
        "shape": list(array.shape),
        "dtype": array.dtype.str,
        "sha256": digest.hexdigest(),
    }


def runtime_fingerprint() -> dict:
    return {
        "homoeogwas": str(__version__),
        "numpy": str(np.__version__),
        "scipy": str(scipy.__version__),
        "blas_thread_env": {
            name: os.environ.get(name) for name in BLAS_THREAD_VARIABLES
        },
    }


def resolve_genotype_identity(source) -> dict:
    """Byte identity of the genotype behind a subgenome, re-verified on disk."""
    if not isinstance(source, Mapping):
        raise GRMCacheError(
            "GRM cache requires file-backed subgenome genotype identity; "
            "load subgenomes with _load_subgenome")
    prefix = source.get("plink_prefix")
    if prefix is None:
        missing = {"bed_sha256", "bim_sha256", "n_variants"} - set(source)
        if missing:
            raise GRMCacheError(
                "GRM cache genotype identity is missing "
                + ", ".join(sorted(missing)))
        return {
            "bed_sha256": str(source["bed_sha256"]),
            "bim_sha256": str(source["bim_sha256"]),
            "n_variants": int(source["n_variants"]),
        }
    recorded = str(source["bim_sha256"])
    observed = plink_bim_sha256(prefix)
    if observed != recorded:
        raise GRMCacheError(
            f"BIM fingerprint of {prefix} no longer matches the bound "
            "snp_to_gene mapping; rerun `homoeogwas prep-snps`")
    return {
        "bed_sha256": sha256_file(plink_path(prefix, ".bed")),
        "bim_sha256": observed,
        "n_variants": int(source["n_variants"]),
    }


def mask_policy_inputs(
    retained_variant_mask: np.ndarray | None,
    mask_policy: Mapping | None,
) -> dict:
    thresholds = None
    if mask_policy is not None and mask_policy.get("thresholds") is not None:
        thresholds = {
            str(name): value.item() if isinstance(value, np.generic) else value
            for name, value in dict(mask_policy["thresholds"]).items()
        }
    if retained_variant_mask is None:
        return {"filter_policy": "legacy_maf_only"}
    mask = np.asarray(retained_variant_mask)
    if mask.dtype != np.bool_ or mask.ndim != 1:
        raise GRMCacheError(
            "retained_variant_mask must be a one-dimensional boolean array")
    return {
        "filter_policy": "explicit_retained_variant_mask",
        "retained_variant_mask_sha256": hashlib.sha256(
            np.ascontiguousarray(mask, dtype=np.uint8).tobytes()).hexdigest(),
        "retained_variant_count": int(mask.sum()),
        "thresholds": thresholds,
    }


def grm_cache_key_inputs(
    *,
    subgenome: str,
    genotype_identity: Mapping,
    sample_row_indices,
    sample_ids,
    method: str,
    maf_min: float,
    mask_policy: Mapping,
) -> dict:
    rows = [int(index) for index in np.asarray(sample_row_indices).tolist()]
    ids = [str(value) for value in sample_ids]
    if len(rows) != len(ids):
        raise GRMCacheError("sample rows and sample IDs must align")
    return {
        "schema": GRM_CACHE_KEY_SCHEMA,
        "subgenome": str(subgenome),
        "genotype": {
            "bed_sha256": str(genotype_identity["bed_sha256"]),
            "bim_sha256": str(genotype_identity["bim_sha256"]),
            "n_variants": int(genotype_identity["n_variants"]),
        },
        "samples": {"n": len(rows), "row_indices": rows, "ids": ids},
        "method": str(method),
        "maf_min": float(maf_min),
        "mask_policy": dict(mask_policy),
    }


def grm_cache_key(inputs: Mapping) -> str:
    return hashlib.sha256(_canonical_json(inputs)).hexdigest()


def _provenance_mask_sha256(provenance: Mapping) -> str:
    mask = np.asarray(provenance["retained_variant_mask"], dtype=np.uint8)
    return hashlib.sha256(np.ascontiguousarray(mask).tobytes()).hexdigest()


class GRMCache:
    """Directory of ``<key>.npy`` matrices with ``<key>.json`` provenance."""

    def __init__(self, cache_dir, *, incomplete_grace_seconds: float = 2.0):
        path = Path(cache_dir)
        if not path.is_absolute():
            raise GRMCacheError(
                f"GRM cache_dir must be an absolute path, got {cache_dir!r}")
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise GRMCacheError(
                f"GRM cache_dir {path} is not a creatable directory: {exc}"
            ) from exc
        if not path.is_dir():
            raise GRMCacheError(f"GRM cache_dir {path} is not a directory")
        self.root = path
        self.incomplete_grace_seconds = float(incomplete_grace_seconds)

    def entry_paths(self, key: str) -> tuple[Path, Path]:
        return self.root / f"{key}.npy", self.root / f"{key}.json"

    def fetch_or_compute(
        self,
        *,
        genotype_source,
        subgenome,
        sample_idx,
        sample_ids,
        method: str,
        maf_min: float,
        retained_variant_mask,
        mask_policy,
        compute: Callable[[], tuple[np.ndarray, dict]],
    ) -> tuple[np.ndarray, dict]:
        if method != "grm_from_X":
            raise GRMCacheError(
                "GRM cache supports only grm.method=grm_from_X")
        if not subgenome:
            raise GRMCacheError("GRM cache requires the subgenome label")
        inputs = grm_cache_key_inputs(
            subgenome=subgenome,
            genotype_identity=resolve_genotype_identity(genotype_source),
            sample_row_indices=sample_idx,
            sample_ids=sample_ids,
            method=method,
            maf_min=maf_min,
            mask_policy=mask_policy_inputs(retained_variant_mask, mask_policy),
        )
        key = grm_cache_key(inputs)
        entry = self.lookup(key, inputs)
        if entry is None:
            K, provenance = compute()
            entry = self.store(key, inputs, K, provenance)
        K = entry["matrix"]
        provenance = dict(entry["grm_provenance"])
        provenance["cache"] = {
            "enabled": True,
            "hit": bool(entry["hit"]),
            "key": key,
            "npy_sha256": entry["npy_sha256"],
        }
        return K, provenance

    def lookup(self, key: str, inputs: Mapping) -> dict | None:
        """Verified entry for ``key`` or ``None`` when no entry exists."""
        npy_path, json_path = self.entry_paths(key)
        deadline = time.monotonic() + self.incomplete_grace_seconds
        while True:
            has_npy, has_json = npy_path.exists(), json_path.exists()
            if has_npy == has_json:
                break
            if time.monotonic() >= deadline:
                raise GRMCacheError(
                    f"GRM cache entry {key} is incomplete "
                    f"(npy={has_npy}, json={has_json}); remove both files "
                    "to recompute")
            time.sleep(0.05)
        if not has_npy:
            return None
        try:
            record = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise GRMCacheError(
                f"GRM cache entry {key} has unreadable provenance: {exc}"
            ) from exc
        if not isinstance(record, dict) or record.get(
                "schema") != GRM_CACHE_ENTRY_SCHEMA:
            raise GRMCacheError(
                f"GRM cache entry {key} has an unsupported schema")
        if record.get("key") != key:
            raise GRMCacheError(
                f"GRM cache entry {key} records a different key")
        observed_sha256 = sha256_file(npy_path)
        if record.get("npy_sha256") != observed_sha256:
            raise GRMCacheError(
                f"GRM cache entry {key} npy sha256 mismatch: expected "
                f"{record.get('npy_sha256')}, observed {observed_sha256}")
        if record.get("key_inputs") != json.loads(_canonical_json(inputs)):
            raise GRMCacheError(
                f"GRM cache entry {key} identity mismatch: stored key inputs "
                "differ from the freshly derived genotype/sample/policy identity")
        current_runtime = runtime_fingerprint()
        if record.get("runtime") != current_runtime:
            raise GRMCacheError(
                f"GRM cache entry {key} runtime mismatch: stored "
                f"{record.get('runtime')!r}, current {current_runtime!r}")
        try:
            K = np.load(npy_path, allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise GRMCacheError(
                f"GRM cache entry {key} matrix is unreadable: {exc}") from exc
        n = int(inputs["samples"]["n"])
        if K.dtype != np.float64 or K.shape != (n, n):
            raise GRMCacheError(
                f"GRM cache entry {key} matrix has dtype {K.dtype} shape "
                f"{K.shape}; expected float64 ({n}, {n})")
        K = np.ascontiguousarray(K)
        if record.get("matrix") != _array_identity(K):
            raise GRMCacheError(
                f"GRM cache entry {key} matrix identity mismatch")
        provenance = record.get("grm_provenance")
        self._verify_provenance(key, inputs, provenance)
        return {
            "matrix": K,
            "grm_provenance": provenance,
            "npy_sha256": observed_sha256,
            "hit": True,
        }

    @staticmethod
    def _verify_provenance(key: str, inputs: Mapping, provenance) -> None:
        if not isinstance(provenance, dict):
            raise GRMCacheError(
                f"GRM cache entry {key} provenance is not a mapping")
        problems = []
        try:
            mask_sha256 = _provenance_mask_sha256(provenance)
        except (KeyError, TypeError, ValueError):
            mask_sha256 = None
        if mask_sha256 != provenance.get("retained_variant_mask_sha256"):
            problems.append("retained_variant_mask does not match its sha256")
        if provenance.get("n_variants_input") != inputs["genotype"]["n_variants"]:
            problems.append("n_variants_input differs from genotype identity")
        used = provenance.get("n_variants_used")
        if mask_sha256 is not None and used != int(
                np.count_nonzero(provenance["retained_variant_mask"])):
            problems.append("n_variants_used differs from the retained mask")
        if provenance.get("maf_min") != inputs["maf_min"]:
            problems.append("maf_min differs from key inputs")
        policy = inputs["mask_policy"]
        if provenance.get("filter_policy") != policy["filter_policy"]:
            problems.append("filter_policy differs from key inputs")
        expected_mask = policy.get("retained_variant_mask_sha256")
        if expected_mask is not None and expected_mask != mask_sha256:
            problems.append("explicit mask sha256 differs from key inputs")
        if problems:
            raise GRMCacheError(
                f"GRM cache entry {key} provenance mismatch: "
                + "; ".join(problems))

    def store(
        self,
        key: str,
        inputs: Mapping,
        K: np.ndarray,
        provenance: Mapping,
    ) -> dict:
        """Publish ``K`` atomically; an existing entry is verified, never replaced."""
        K = np.ascontiguousarray(np.asarray(K, dtype=np.float64))
        provenance = json.loads(_canonical_json(dict(provenance)))
        self._verify_provenance(key, inputs, provenance)
        npy_path, json_path = self.entry_paths(key)
        if npy_path.exists() or json_path.exists():
            return self._reconcile(key, inputs, K, provenance)
        temporary_npy = temporary_json = None
        try:
            descriptor, name = tempfile.mkstemp(
                dir=self.root, prefix=f".{key}.", suffix=".npy.tmp")
            temporary_npy = Path(name)
            with os.fdopen(descriptor, "wb") as handle:
                np.save(handle, K, allow_pickle=False)
                handle.flush()
                os.fsync(handle.fileno())
            npy_sha256 = sha256_file(temporary_npy)
            record = {
                "schema": GRM_CACHE_ENTRY_SCHEMA,
                "key": key,
                "key_inputs": json.loads(_canonical_json(inputs)),
                "npy_sha256": npy_sha256,
                "matrix": _array_identity(K),
                "grm_provenance": provenance,
                "runtime": runtime_fingerprint(),
                "created_utc": datetime.now(timezone.utc).isoformat(),
            }
            descriptor, name = tempfile.mkstemp(
                dir=self.root, prefix=f".{key}.", suffix=".json.tmp")
            temporary_json = Path(name)
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(
                    record, handle, sort_keys=True, ensure_ascii=False,
                    allow_nan=False, indent=1)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            if npy_path.exists() or json_path.exists():
                return self._reconcile(key, inputs, K, provenance)
            os.replace(temporary_npy, npy_path)
            temporary_npy = None
            os.replace(temporary_json, json_path)
            temporary_json = None
        finally:
            for temporary in (temporary_npy, temporary_json):
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        return {
            "matrix": K,
            "grm_provenance": provenance,
            "npy_sha256": npy_sha256,
            "hit": False,
        }

    def _reconcile(
        self, key: str, inputs: Mapping, K: np.ndarray, provenance: Mapping,
    ) -> dict:
        entry = self.lookup(key, inputs)
        if entry is None:
            raise GRMCacheError(
                f"GRM cache entry {key} disappeared while being published")
        if not np.array_equal(
                entry["matrix"].view(np.uint64), K.view(np.uint64)):
            raise GRMCacheError(
                f"GRM cache entry {key} differs bitwise from the freshly "
                "computed matrix; the runtime is not reproducing GRM bytes")
        if entry["grm_provenance"] != provenance:
            raise GRMCacheError(
                f"GRM cache entry {key} differs from the freshly computed "
                "GRM provenance")
        return entry
