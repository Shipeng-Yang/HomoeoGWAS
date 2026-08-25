"""Strict deterministic checkpoints for indexed omniB bootstrap blocks."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np

CHECKPOINT_SCHEMA_VERSION = 1
_BLOCK_RE = re.compile(r"^block_(\d+)_(\d+)\.npz$")
_BLOCK_FIELDS = {
    "schema_version", "manifest_id", "start", "stop",
    "replicate_seed_ids", "matrix_shape", "matrix_dtype",
    "primary_null_p", "matrix_sha256",
}
_OBSERVED_FIELDS = {
    "schema_version", "manifest_id", "hypothesis_ids", "vector_shape",
    "vector_dtype", "observed_p", "vector_sha256",
}


class CheckpointError(ValueError):
    """A checkpoint is incomplete, corrupt, or belongs to another run."""


def replicate_seed(base_seed: int, replicate_index: int) -> int:
    """Return the frozen SHA-256-derived seed for one bootstrap replicate."""
    if isinstance(replicate_index, bool) or not isinstance(
            replicate_index, (int, np.integer)):
        raise ValueError("replicate_index must be a non-negative integer")
    replicate_index = int(replicate_index)
    if replicate_index < 0:
        raise ValueError("replicate_index must be non-negative")
    body = (
        f"homoeogwas-omnib-bootstrap-v1\0{base_seed}\0{replicate_index}"
        .encode()
    )
    return int.from_bytes(hashlib.sha256(body).digest()[:16], "little")


def _canonical_json(value) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CheckpointError(f"manifest is not canonical JSON: {exc}") from exc


def canonical_manifest_id(manifest: dict) -> str:
    """Hash a JSON manifest with stable ordering and no non-finite numbers."""
    if not isinstance(manifest, dict):
        raise CheckpointError("manifest must be a mapping")
    return hashlib.sha256(_canonical_json(manifest)).hexdigest()


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write(path: Path, writer) -> None:
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _fsynced_temporary(path: Path, writer) -> Path:
    """Write and fsync one same-directory temporary file."""
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    return temporary


@contextmanager
def _block_publication_lock(root: Path):
    """Serialize range validation and no-replace publication across writers."""
    path = root / ".block-publication.lock"
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


class CheckpointStore:
    """Atomic primary-null-p blocks bound to one immutable run manifest."""

    def __init__(
        self,
        root,
        manifest_id: str,
        B: int,
        block_size: int,
        *,
        base_seed: int | None = None,
    ) -> None:
        if not isinstance(manifest_id, str) or not manifest_id:
            raise CheckpointError("manifest_id must be a non-empty string")
        if isinstance(B, bool) or not isinstance(B, (int, np.integer)) or int(B) < 1:
            raise CheckpointError("B must be an integer >= 1")
        if (
            isinstance(block_size, bool)
            or not isinstance(block_size, (int, np.integer))
            or int(block_size) < 1
        ):
            raise CheckpointError("block_size must be an integer >= 1")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_id = manifest_id
        self.B = int(B)
        self.block_size = int(block_size)
        self.base_seed = base_seed

    def planned_ranges(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            (start, min(start + self.block_size, self.B))
            for start in range(0, self.B, self.block_size)
        )

    def _path(self, start: int, stop: int) -> Path:
        return self.root / f"block_{start}_{stop}.npz"

    def _validate_range(self, start: int, stop: int) -> tuple[int, int]:
        if any(isinstance(value, bool) for value in (start, stop)):
            raise CheckpointError("checkpoint range values must be integers")
        try:
            start, stop = int(start), int(stop)
        except (TypeError, ValueError) as exc:
            raise CheckpointError("checkpoint range values must be integers") from exc
        if not (0 <= start < stop <= self.B):
            raise CheckpointError(
                f"checkpoint range [{start}, {stop}) is out of bounds for B={self.B}")
        return start, stop

    def _seed_ids(self, start: int, stop: int) -> np.ndarray:
        if self.base_seed is None:
            values = [f"index:{index}" for index in range(start, stop)]
        else:
            values = [
                f"{replicate_seed(self.base_seed, index):032x}"
                for index in range(start, stop)
            ]
        return np.asarray(values, dtype="U40")

    def bind_manifest(self, manifest: dict) -> Path:
        """Persist the canonical manifest or verify the existing binding."""
        actual_id = canonical_manifest_id(manifest)
        if actual_id != self.manifest_id:
            raise CheckpointError(
                f"manifest content hashes to {actual_id}, expected {self.manifest_id}")
        path = self.root / "manifest.json"
        payload = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "manifest_id": self.manifest_id,
            "manifest": manifest,
        }
        body = _canonical_json(payload) + b"\n"
        if path.exists():
            try:
                existing = path.read_bytes()
            except OSError as exc:
                raise CheckpointError(f"cannot read checkpoint manifest: {exc}") from exc
            if existing != body:
                raise CheckpointError(
                    "checkpoint manifest does not match this run")
            return path
        _atomic_write(path, lambda handle: handle.write(body))
        return path

    def write_observed(self, values: np.ndarray, hypothesis_ids) -> Path:
        """Persist the observed primary-family vector before any null blocks."""
        vector = np.asarray(values)
        ids = tuple(hypothesis_ids)
        if vector.ndim != 1:
            raise CheckpointError("observed primary p-values must be one-dimensional")
        if len(ids) != vector.size or not all(isinstance(value, str) for value in ids):
            raise CheckpointError(
                "observed hypothesis IDs must be strings aligned to the p-value vector")
        if vector.dtype.hasobject:
            raise CheckpointError("observed primary p-value dtype cannot be object")
        vector = np.ascontiguousarray(vector)
        path = self.root / "observed.npz"
        if path.exists():
            existing = self.read_observed(ids)
            if not np.array_equal(existing, vector, equal_nan=True):
                raise CheckpointError(
                    "observed primary family does not match the existing checkpoint")
            return path
        payload = {
            "schema_version": np.array(CHECKPOINT_SCHEMA_VERSION, dtype=np.int64),
            "manifest_id": np.array(self.manifest_id),
            "hypothesis_ids": np.asarray(ids, dtype="U"),
            "vector_shape": np.asarray(vector.shape, dtype=np.int64),
            "vector_dtype": np.array(vector.dtype.str),
            "observed_p": vector,
            "vector_sha256": np.array(hashlib.sha256(
                vector.tobytes(order="C")).hexdigest()),
        }
        _atomic_write(path, lambda handle: np.savez(handle, **payload))
        return path

    def read_observed(self, hypothesis_ids) -> np.ndarray:
        """Strictly read the manifest-bound observed primary-family vector."""
        path = self.root / "observed.npz"
        if not path.exists():
            raise CheckpointError("observed primary family checkpoint is missing")
        expected_ids = tuple(hypothesis_ids)
        try:
            with np.load(path, allow_pickle=False) as loaded:
                if set(loaded.files) != _OBSERVED_FIELDS:
                    raise CheckpointError(
                        "observed checkpoint schema fields mismatch")
                schema = int(np.asarray(loaded["schema_version"]).item())
                manifest = str(np.asarray(loaded["manifest_id"]).item())
                ids = tuple(np.asarray(loaded["hypothesis_ids"]).astype(str).tolist())
                shape = tuple(
                    int(value) for value in np.asarray(loaded["vector_shape"]).tolist())
                dtype = str(np.asarray(loaded["vector_dtype"]).item())
                vector = np.asarray(loaded["observed_p"]).copy()
                digest = str(np.asarray(loaded["vector_sha256"]).item())
        except CheckpointError:
            raise
        except Exception as exc:
            raise CheckpointError(
                f"cannot read observed checkpoint: {exc}") from exc
        if schema != CHECKPOINT_SCHEMA_VERSION:
            raise CheckpointError("observed checkpoint schema version mismatch")
        if manifest != self.manifest_id:
            raise CheckpointError("observed checkpoint manifest does not match")
        if ids != expected_ids:
            raise CheckpointError("observed checkpoint hypothesis IDs do not match")
        if shape != vector.shape or vector.ndim != 1:
            raise CheckpointError("observed checkpoint vector shape does not match")
        if dtype != vector.dtype.str:
            raise CheckpointError("observed checkpoint vector dtype does not match")
        actual = hashlib.sha256(
            np.ascontiguousarray(vector).tobytes(order="C")).hexdigest()
        if digest != actual:
            raise CheckpointError("observed checkpoint SHA-256 does not match")
        return vector

    def write_block(self, start: int, stop: int, matrix: np.ndarray) -> Path:
        start, stop = self._validate_range(start, stop)
        values = np.asarray(matrix)
        if values.ndim != 2 or values.shape[1] != stop - start:
            raise CheckpointError(
                "primary null-p matrix shape must be "
                f"(n_hypotheses, {stop - start}); got {values.shape}")
        if values.dtype.hasobject:
            raise CheckpointError("primary null-p matrix dtype cannot be object")
        values = np.ascontiguousarray(values)
        path = self._path(start, stop)
        digest = hashlib.sha256(values.tobytes(order="C")).hexdigest()
        payload = {
            "schema_version": np.array(CHECKPOINT_SCHEMA_VERSION, dtype=np.int64),
            "manifest_id": np.array(self.manifest_id),
            "start": np.array(start, dtype=np.int64),
            "stop": np.array(stop, dtype=np.int64),
            "replicate_seed_ids": self._seed_ids(start, stop),
            "matrix_shape": np.asarray(values.shape, dtype=np.int64),
            "matrix_dtype": np.array(values.dtype.str),
            "primary_null_p": values,
            "matrix_sha256": np.array(digest),
        }
        temporary = _fsynced_temporary(
            path, lambda handle: np.savez(handle, **payload))
        try:
            with _block_publication_lock(self.root):
                ranges = self.completed_ranges()
                if path.exists():
                    existing = self.read_block(start, stop)
                    existing_digest = hashlib.sha256(
                        np.ascontiguousarray(existing).tobytes(order="C")
                    ).hexdigest()
                    if (
                        existing.dtype.str != values.dtype.str
                        or existing_digest != digest
                    ):
                        raise CheckpointError(
                            f"checkpoint block publication conflict for range "
                            f"[{start}, {stop})")
                    return path
                overlap = next((
                    current for current in ranges
                    if start < current[1] and current[0] < stop
                ), None)
                if overlap is not None:
                    raise CheckpointError(
                        f"checkpoint block publication conflict: range "
                        f"[{start}, {stop}) overlaps existing {overlap}")
                try:
                    os.link(temporary, path)
                except FileExistsError as exc:
                    existing = self.read_block(start, stop)
                    existing_digest = hashlib.sha256(
                        np.ascontiguousarray(existing).tobytes(order="C")
                    ).hexdigest()
                    if (
                        existing.dtype.str != values.dtype.str
                        or existing_digest != digest
                    ):
                        raise CheckpointError(
                            f"checkpoint block publication conflict for range "
                            f"[{start}, {stop})") from exc
                return path
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
            _fsync_directory(self.root)

    def read_block(self, start: int, stop: int) -> np.ndarray:
        start, stop = self._validate_range(start, stop)
        path = self._path(start, stop)
        if not path.exists():
            raise CheckpointError(
                f"checkpoint block [{start}, {stop}) is incomplete/missing")
        try:
            with np.load(path, allow_pickle=False) as loaded:
                if set(loaded.files) != _BLOCK_FIELDS:
                    missing = sorted(_BLOCK_FIELDS - set(loaded.files))
                    extra = sorted(set(loaded.files) - _BLOCK_FIELDS)
                    raise CheckpointError(
                        f"checkpoint block schema fields mismatch; missing={missing}, "
                        f"extra={extra}")
                schema = int(np.asarray(loaded["schema_version"]).item())
                manifest = str(np.asarray(loaded["manifest_id"]).item())
                block_start = int(np.asarray(loaded["start"]).item())
                block_stop = int(np.asarray(loaded["stop"]).item())
                seed_ids = np.asarray(loaded["replicate_seed_ids"]).astype(str)
                stored_shape = tuple(
                    int(value) for value in np.asarray(loaded["matrix_shape"]).tolist())
                stored_dtype = str(np.asarray(loaded["matrix_dtype"]).item())
                matrix = np.asarray(loaded["primary_null_p"]).copy()
                stored_hash = str(np.asarray(loaded["matrix_sha256"]).item())
        except CheckpointError:
            raise
        except Exception as exc:
            raise CheckpointError(
                f"cannot read checkpoint block {path.name}: {exc}") from exc

        if schema != CHECKPOINT_SCHEMA_VERSION:
            raise CheckpointError(
                f"checkpoint schema {schema} != {CHECKPOINT_SCHEMA_VERSION}")
        if manifest != self.manifest_id:
            raise CheckpointError(
                f"checkpoint manifest {manifest!r} != {self.manifest_id!r}")
        if (block_start, block_stop) != (start, stop):
            raise CheckpointError(
                "checkpoint filename range disagrees with stored range")
        if not np.array_equal(seed_ids, self._seed_ids(start, stop)):
            raise CheckpointError("checkpoint replicate seed IDs do not match")
        if stored_shape != matrix.shape or matrix.shape[1] != stop - start:
            raise CheckpointError("checkpoint matrix shape metadata does not match")
        if stored_dtype != matrix.dtype.str:
            raise CheckpointError("checkpoint matrix dtype metadata does not match")
        actual_hash = hashlib.sha256(
            np.ascontiguousarray(matrix).tobytes(order="C")).hexdigest()
        if stored_hash != actual_hash:
            raise CheckpointError("checkpoint matrix SHA-256 does not match")
        return matrix

    def has_range(self, start: int, stop: int) -> bool:
        start, stop = self._validate_range(start, stop)
        if not self._path(start, stop).exists():
            return False
        self.read_block(start, stop)
        return True

    def completed_ranges(self) -> tuple[tuple[int, int], ...]:
        ranges: list[tuple[int, int]] = []
        for path in sorted(self.root.glob("block_*.npz")):
            match = _BLOCK_RE.fullmatch(path.name)
            if match is None:
                raise CheckpointError(
                    f"checkpoint block filename is invalid: {path.name}")
            start, stop = (int(match.group(1)), int(match.group(2)))
            self.read_block(start, stop)
            ranges.append((start, stop))
        ranges.sort()
        for previous, current in zip(ranges, ranges[1:], strict=False):
            if current[0] < previous[1]:
                raise CheckpointError(
                    f"checkpoint blocks overlap: {previous} and {current}")
            if current == previous:
                raise CheckpointError(f"duplicate checkpoint range: {current}")
        return tuple(ranges)

    def missing_ranges(self) -> tuple[tuple[int, int], ...]:
        """Partition uncovered replicate indices using the current block size."""
        missing: list[tuple[int, int]] = []
        cursor = 0
        for start, stop in self.completed_ranges():
            while cursor < start:
                block_stop = min(cursor + self.block_size, start)
                missing.append((cursor, block_stop))
                cursor = block_stop
            cursor = stop
        while cursor < self.B:
            block_stop = min(cursor + self.block_size, self.B)
            missing.append((cursor, block_stop))
            cursor = block_stop
        return tuple(missing)

    def concatenate(self, *, require_complete: bool = True) -> np.ndarray:
        ranges = self.completed_ranges()
        if not ranges:
            raise CheckpointError("checkpoint is incomplete: no blocks exist")
        if require_complete:
            cursor = 0
            for start, stop in ranges:
                if start != cursor:
                    raise CheckpointError(
                        "checkpoint is incomplete or has a gap; completed ranges "
                        f"are {ranges}")
                cursor = stop
            if cursor != self.B:
                raise CheckpointError(
                    "checkpoint is incomplete or has a gap; completed ranges "
                    f"are {ranges}")
        matrices = [self.read_block(start, stop) for start, stop in ranges]
        rows = {matrix.shape[0] for matrix in matrices}
        if len(rows) != 1:
            raise CheckpointError("checkpoint blocks have inconsistent row count/shape")
        dtypes = {matrix.dtype.str for matrix in matrices}
        if len(dtypes) != 1:
            raise CheckpointError("checkpoint blocks have inconsistent dtype")
        return np.concatenate(matrices, axis=1)
