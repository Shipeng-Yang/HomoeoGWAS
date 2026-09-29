import hashlib
import json
import tracemalloc

import numpy as np
import pytest

from homoeogwas import omnib_family as F


def reference_identity(values):
    array = np.asarray(values)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode())
    digest.update(b"\0")
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode())
    digest.update(b"\0")
    digest.update(np.ascontiguousarray(array).tobytes(order="C"))
    return {"shape": list(array.shape), "dtype": array.dtype.str, "sha256": digest.hexdigest()}


@pytest.mark.parametrize("shape", [(0, 5), (1, 7), (37, 11), (300, 1), (5, 4, 3)])
@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.int8, np.int64])
@pytest.mark.parametrize("layout", ["C", "F", "strided"])
def test_streamed_digest_matches_reference(shape, dtype, layout):
    rng = np.random.default_rng(7)
    base = rng.normal(size=tuple(2 * s if layout == "strided" else s for s in shape)) * 50
    base = base.astype(dtype)
    if layout == "F":
        array = np.asfortranarray(base)
    elif layout == "strided":
        array = base[tuple(slice(None, None, 2) for _ in shape)]
    else:
        array = np.ascontiguousarray(base)
    assert F._array_identity(array) == reference_identity(array)


def test_fortran_array_is_hashed_without_full_copy(monkeypatch):
    monkeypatch.setattr(F, "_HASH_CHUNK_BYTES", 1 << 20)
    array = np.asfortranarray(np.ones((2000, 4000), dtype=np.float64))
    tracemalloc.start()
    try:
        F._array_identity(array)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < array.nbytes / 4


def test_fortran_dosage_matrix_digest_unchanged():
    rng = np.random.default_rng(3)
    X = np.asfortranarray(rng.integers(0, 3, size=(50, 400)).astype(np.float64))
    assert F._array_identity(X) == reference_identity(X)
