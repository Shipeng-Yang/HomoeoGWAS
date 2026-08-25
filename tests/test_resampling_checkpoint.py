"""Deterministic, strict checkpoint storage for indexed bootstrap blocks."""

from __future__ import annotations

import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest

from homoeogwas.resampling_checkpoint import (
    CheckpointError,
    CheckpointStore,
    canonical_manifest_id,
    replicate_seed,
)


def _run_fake_blocks(root, *, block_size=3, stop_after=None, order=None):
    store = CheckpointStore(
        root, manifest_id="fixture-v1", B=10, block_size=block_size,
        base_seed=2026)
    ranges = list(store.planned_ranges())
    if order is not None:
        ranges = [ranges[index] for index in order]
    written = 0
    for start, stop in ranges:
        if store.has_range(start, stop):
            continue
        cols = [
            np.random.default_rng(replicate_seed(2026, index)).uniform(size=7)
            for index in range(start, stop)
        ]
        store.write_block(start, stop, np.column_stack(cols))
        written += 1
        if stop_after is not None and written == stop_after:
            break
    null_p = store.concatenate(require_complete=stop_after is None)
    return SimpleNamespace(
        null_p=null_p,
        null_p_sha256=hashlib.sha256(
            null_p.tobytes(order="C")).hexdigest(),
        store=store,
    )


def _rewrite_npz(path, **updates):
    with np.load(path, allow_pickle=False) as loaded:
        payload = {key: loaded[key] for key in loaded.files}
    payload.update(updates)
    with path.open("wb") as handle:
        np.savez(handle, **payload)


def test_replicate_seed_uses_frozen_sha256_domain_and_index_only():
    assert [replicate_seed(2026, index) for index in range(4)] == [
        304997635691478729550130864679682032482,
        103485488584519421142365000500424458016,
        223279822643388472131429537684126288246,
        143578937669811089123113135169596014939,
    ]
    forward = [replicate_seed(2026, index) for index in range(8)]
    reverse = [replicate_seed(2026, index) for index in reversed(range(8))]
    assert forward == list(reversed(reverse))
    assert len(set(forward)) == 8
    with pytest.raises(ValueError, match="non-negative"):
        replicate_seed(2026, -1)


def test_checkpoint_resume_and_completion_order_are_byte_identical(tmp_path):
    full = _run_fake_blocks(tmp_path / "full", block_size=5)
    _run_fake_blocks(tmp_path / "resume", block_size=3, stop_after=2)
    resumed = _run_fake_blocks(tmp_path / "resume", block_size=3)
    reverse = _run_fake_blocks(
        tmp_path / "reverse", block_size=7, order=[1, 0])

    assert full.null_p_sha256 == resumed.null_p_sha256 == reverse.null_p_sha256
    np.testing.assert_array_equal(full.null_p, resumed.null_p)
    np.testing.assert_array_equal(full.null_p, reverse.null_p)
    assert resumed.store.completed_ranges() == ((0, 3), (3, 6), (6, 9), (9, 10))


def test_checkpoint_resume_discovers_existing_partition_with_new_block_size(tmp_path):
    root = tmp_path / "resume"
    interrupted = CheckpointStore(
        root, manifest_id="fixture-v1", B=10, block_size=3,
        base_seed=2026)
    for start, stop in interrupted.planned_ranges()[:2]:
        columns = [
            np.random.default_rng(replicate_seed(2026, index)).uniform(size=7)
            for index in range(start, stop)
        ]
        interrupted.write_block(start, stop, np.column_stack(columns))

    resumed = CheckpointStore(
        root, manifest_id="fixture-v1", B=10, block_size=5,
        base_seed=2026)
    assert resumed.missing_ranges() == ((6, 10),)
    for start, stop in resumed.missing_ranges():
        columns = [
            np.random.default_rng(replicate_seed(2026, index)).uniform(size=7)
            for index in range(start, stop)
        ]
        resumed.write_block(start, stop, np.column_stack(columns))

    baseline = _run_fake_blocks(tmp_path / "full", block_size=5)
    restored = resumed.concatenate(require_complete=True)
    np.testing.assert_array_equal(restored, baseline.null_p)
    assert hashlib.sha256(restored.tobytes(order="C")).hexdigest() == (
        baseline.null_p_sha256)
    assert resumed.completed_ranges() == ((0, 3), (3, 6), (6, 10))


def test_block_binds_manifest_range_seed_ids_shape_dtype_and_hash(tmp_path):
    store = CheckpointStore(
        tmp_path, "manifest-a", B=5, block_size=3, base_seed=9)
    matrix = np.arange(12, dtype=np.float64).reshape(4, 3)
    path = store.write_block(0, 3, matrix)

    assert path.name == "block_0_3.npz"
    with np.load(path, allow_pickle=False) as block:
        assert set(block.files) == {
            "schema_version", "manifest_id", "start", "stop",
            "replicate_seed_ids", "matrix_shape", "matrix_dtype",
            "primary_null_p", "matrix_sha256",
        }
        assert block["manifest_id"].item() == "manifest-a"
        assert block["matrix_shape"].tolist() == [4, 3]
        assert block["matrix_dtype"].item() == matrix.dtype.str
        assert block["replicate_seed_ids"].tolist() == [
            f"{replicate_seed(9, index):032x}" for index in range(3)]
        assert block["matrix_sha256"].item() == hashlib.sha256(
            matrix.tobytes(order="C")).hexdigest()
    np.testing.assert_array_equal(store.read_block(0, 3), matrix)


def test_concurrent_block_writers_accept_identical_loser(monkeypatch, tmp_path):
    barrier = threading.Barrier(2)
    original_savez = np.savez

    def synchronized_savez(*args, **kwargs):
        original_savez(*args, **kwargs)
        barrier.wait(timeout=10)

    monkeypatch.setattr(np, "savez", synchronized_savez)
    values = np.arange(12, dtype=np.float64).reshape(4, 3)
    stores = [
        CheckpointStore(
            tmp_path, "manifest-a", B=3, block_size=3, base_seed=9)
        for _ in range(2)
    ]
    with ThreadPoolExecutor(max_workers=2) as pool:
        paths = list(pool.map(
            lambda store: store.write_block(0, 3, values.copy()), stores))

    assert paths == [tmp_path / "block_0_3.npz"] * 2
    np.testing.assert_array_equal(stores[0].read_block(0, 3), values)


def test_concurrent_block_writers_reject_conflicting_loser(monkeypatch, tmp_path):
    barrier = threading.Barrier(2)
    original_savez = np.savez

    def synchronized_savez(*args, **kwargs):
        original_savez(*args, **kwargs)
        barrier.wait(timeout=10)

    monkeypatch.setattr(np, "savez", synchronized_savez)
    stores = [
        CheckpointStore(
            tmp_path, "manifest-a", B=3, block_size=3, base_seed=9)
        for _ in range(2)
    ]
    matrices = [np.zeros((4, 3)), np.ones((4, 3))]

    def publish(index):
        try:
            return stores[index].write_block(0, 3, matrices[index])
        except CheckpointError as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(publish, range(2)))

    assert sum(isinstance(value, CheckpointError) for value in outcomes) == 1
    assert sum(value == tmp_path / "block_0_3.npz" for value in outcomes) == 1
    error = next(value for value in outcomes if isinstance(value, CheckpointError))
    assert "conflict" in str(error).lower()
    stored = stores[0].read_block(0, 3)
    assert any(np.array_equal(stored, values) for values in matrices)


def test_strict_reads_reject_corruption_wrong_manifest_and_seed_ids(tmp_path):
    store = CheckpointStore(
        tmp_path, "manifest-a", B=5, block_size=3, base_seed=9)
    path = store.write_block(0, 3, np.arange(12.0).reshape(4, 3))

    with pytest.raises(CheckpointError, match="manifest"):
        CheckpointStore(
            tmp_path, "manifest-b", B=5, block_size=3,
            base_seed=9).read_block(0, 3)

    _rewrite_npz(path, replicate_seed_ids=np.array(["bad"] * 3))
    with pytest.raises(CheckpointError, match="seed"):
        store.read_block(0, 3)

    store = CheckpointStore(
        tmp_path / "hash", "manifest-a", B=5, block_size=3, base_seed=9)
    path = store.write_block(0, 3, np.arange(12.0).reshape(4, 3))
    changed = store.read_block(0, 3)
    changed[0, 0] = 999.0
    _rewrite_npz(path, primary_null_p=changed)
    with pytest.raises(CheckpointError, match="SHA-256"):
        store.read_block(0, 3)

    path.write_bytes(b"not-an-npz")
    with pytest.raises(CheckpointError, match="cannot read"):
        store.read_block(0, 3)


def test_store_rejects_overlap_gap_incomplete_and_out_of_bounds(tmp_path):
    store = CheckpointStore(
        tmp_path, "manifest-a", B=10, block_size=3, base_seed=9)
    store.write_block(0, 3, np.zeros((2, 3)))
    store.write_block(6, 9, np.zeros((2, 3)))

    assert store.completed_ranges() == ((0, 3), (6, 9))
    with pytest.raises(CheckpointError, match="gap|incomplete"):
        store.concatenate(require_complete=True)
    with pytest.raises(CheckpointError, match="planned|overlap"):
        store.write_block(2, 5, np.zeros((2, 3)))
    with pytest.raises(CheckpointError, match="bounds|range"):
        store.write_block(9, 11, np.zeros((2, 2)))
    assert store.write_block(0, 3, np.zeros((2, 3))) == (
        tmp_path / "block_0_3.npz")
    with pytest.raises(CheckpointError, match="conflict"):
        store.write_block(0, 3, np.ones((2, 3)))


def test_concatenate_rejects_inconsistent_shape_dtype_and_filename_range(tmp_path):
    shape_store = CheckpointStore(
        tmp_path / "shape", "m", B=6, block_size=3, base_seed=1)
    shape_store.write_block(0, 3, np.zeros((2, 3), dtype=np.float64))
    shape_store.write_block(3, 6, np.zeros((3, 3), dtype=np.float64))
    with pytest.raises(CheckpointError, match="row count|shape"):
        shape_store.concatenate()

    dtype_store = CheckpointStore(
        tmp_path / "dtype", "m", B=6, block_size=3, base_seed=1)
    dtype_store.write_block(0, 3, np.zeros((2, 3), dtype=np.float64))
    dtype_store.write_block(3, 6, np.zeros((2, 3), dtype=np.float32))
    with pytest.raises(CheckpointError, match="dtype"):
        dtype_store.concatenate()

    range_store = CheckpointStore(
        tmp_path / "range", "m", B=6, block_size=3, base_seed=1)
    path = range_store.write_block(0, 3, np.zeros((2, 3)))
    _rewrite_npz(path, start=np.array(1, dtype=np.int64))
    with pytest.raises(CheckpointError, match="filename|range"):
        range_store.completed_ranges()


def test_manifest_file_is_canonical_and_refuses_changed_inputs(tmp_path):
    manifest = {
        "hypothesis_unit": "edge",
        "bootstrap": {"B": 10, "seed": 2026},
        "family_sha256": "f" * 64,
    }
    manifest_id = canonical_manifest_id(manifest)
    store = CheckpointStore(tmp_path, manifest_id, B=10, block_size=3)
    path = store.bind_manifest(manifest)

    assert path.name == "manifest.json"
    assert store.bind_manifest(dict(reversed(list(manifest.items())))) == path
    changed = manifest | {"hypothesis_unit": "group"}
    with pytest.raises(CheckpointError, match="manifest"):
        store.bind_manifest(changed)


def test_observed_primary_family_is_hash_bound_and_idempotent(tmp_path):
    store = CheckpointStore(
        tmp_path, "fixture-v1", B=5, block_size=3, base_seed=2026)
    observed = np.array([0.2, np.nan, 0.7])
    ids = ["edge:AB:a:b", "edge:AB:c:d", "edge:AB:e:f"]
    path = store.write_observed(observed, ids)

    assert path.name == "observed.npz"
    np.testing.assert_allclose(
        store.read_observed(ids), observed, equal_nan=True, rtol=0, atol=0)
    assert store.write_observed(observed.copy(), ids) == path
    with pytest.raises(CheckpointError, match="hypothesis"):
        store.read_observed(list(reversed(ids)))
    with pytest.raises(CheckpointError, match="observed"):
        store.write_observed(observed + 0.01, ids)


def _small_group_fixture(seed=441):
    from homoeogwas.group_family import MasterGroupFamily
    from homoeogwas.interact import SubgenomeData

    rng = np.random.default_rng(seed)
    n, groups, snps_per_gene = 48, 4, 5
    subdata = {}
    for sub in ("A", "B", "D"):
        X = rng.integers(
            0, 3, size=(n, groups * snps_per_gene)).astype(float)
        gene_snp = {
            f"g{index}": np.arange(
                index * snps_per_gene, (index + 1) * snps_per_gene)
            for index in range(groups)
        }
        subdata[sub] = SubgenomeData(
            X=X, gene_snp=gene_snp,
            samples=[f"s{index}" for index in range(n)], chunk=None)
    family = MasterGroupFamily(
        subgenomes=("A", "B", "D"),
        group_ids=tuple(f"group_{index}" for index in range(groups)),
        genes=tuple((f"g{index}",) * 3 for index in range(groups)),
    )
    return subdata, family, rng.normal(size=n), np.arange(n)


def _checkpoint_group_run(
        root, *, n_jobs, block_size, y=None, alpha=0.05, inferential=True):
    from homoeogwas.omnib_family import run_group_scan_omnib

    subdata, family, fixture_y, sample_idx = _small_group_fixture()
    return run_group_scan_omnib(
        subdata, family, fixture_y if y is None else y, sample_idx,
        hypothesis_unit="edge", family_scope="primary_only",
        cap=150, n_pc=3, transform="INT", bootstrap_B=19,
        bootstrap_seed=2026, n_jobs=n_jobs, grm_method="grm_from_X",
        maf_min=0.01, burden_maf=0.01, min_snp=3,
        checkpoint_dir=root, checkpoint_block_size=block_size,
        alpha=alpha, inferential=inferential,
    )


def _result_bytes(result):
    from homoeogwas.interact import _json_safe

    return json.dumps(
        _json_safe(asdict(result)), sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()


@pytest.mark.parametrize(
    ("n_jobs", "block_size"), [(1, 5), (2, 3), (4, 7)])
def test_checkpoint_group_scan_is_worker_and_block_size_invariant(
        tmp_path, n_jobs, block_size):
    result = _checkpoint_group_run(
        tmp_path / f"run_{n_jobs}_{block_size}",
        n_jobs=n_jobs, block_size=block_size)
    baseline = _checkpoint_group_run(
        tmp_path / "baseline", n_jobs=1, block_size=5)

    got = result.model_diagnostics["resampling_checkpoint"]
    want = baseline.model_diagnostics["resampling_checkpoint"]
    assert got["primary_null_p_sha256"] == want["primary_null_p_sha256"]
    assert result.minp_boot_threshold == baseline.minp_boot_threshold
    assert result.model_diagnostics["bootstrap_fwer"]["adjusted_p"] == (
        baseline.model_diagnostics["bootstrap_fwer"]["adjusted_p"])
    assert [hit["hypothesis_id"] for hit in result.sig] == [
        hit["hypothesis_id"] for hit in baseline.sig]


def test_checkpoint_group_scan_resume_is_byte_identical(monkeypatch, tmp_path):
    original = CheckpointStore.write_block
    calls = {"count": 0}

    def interrupt_after_two(self, start, stop, matrix):
        path = original(self, start, stop, matrix)
        calls["count"] += 1
        if calls["count"] == 2:
            raise RuntimeError("simulated interruption")
        return path

    monkeypatch.setattr(CheckpointStore, "write_block", interrupt_after_two)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        _checkpoint_group_run(
            tmp_path / "resume", n_jobs=2, block_size=3)
    monkeypatch.setattr(CheckpointStore, "write_block", original)

    resumed = _checkpoint_group_run(
        tmp_path / "resume", n_jobs=4, block_size=3)
    uninterrupted = _checkpoint_group_run(
        tmp_path / "full", n_jobs=1, block_size=3)
    assert _result_bytes(resumed) == _result_bytes(uninterrupted)


def test_checkpoint_group_scan_resume_with_new_block_size_is_calibration_identical(
        monkeypatch, tmp_path):
    original = CheckpointStore.write_block
    calls = {"count": 0}

    def interrupt_after_two(self, start, stop, matrix):
        path = original(self, start, stop, matrix)
        calls["count"] += 1
        if calls["count"] == 2:
            raise RuntimeError("simulated interruption")
        return path

    monkeypatch.setattr(CheckpointStore, "write_block", interrupt_after_two)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        _checkpoint_group_run(
            tmp_path / "resume", n_jobs=2, block_size=3)
    monkeypatch.setattr(CheckpointStore, "write_block", original)

    resumed = _checkpoint_group_run(
        tmp_path / "resume", n_jobs=4, block_size=5)
    uninterrupted = _checkpoint_group_run(
        tmp_path / "full", n_jobs=1, block_size=5)
    resumed_checkpoint = resumed.model_diagnostics["resampling_checkpoint"]
    full_checkpoint = uninterrupted.model_diagnostics["resampling_checkpoint"]
    assert resumed_checkpoint["block_size"] == full_checkpoint["block_size"] == 5
    assert resumed_checkpoint["primary_null_p_sha256"] == (
        full_checkpoint["primary_null_p_sha256"])
    assert resumed.model_diagnostics["bootstrap_fwer"] == (
        uninterrupted.model_diagnostics["bootstrap_fwer"])
    assert resumed.minp_boot_threshold == uninterrupted.minp_boot_threshold
    assert resumed.sig == uninterrupted.sig


def test_checkpoint_group_scan_refuses_changed_inference_manifest(tmp_path):
    root = tmp_path / "run"
    _checkpoint_group_run(root, n_jobs=1, block_size=5)
    _, _, y, _ = _small_group_fixture()
    changed = y.copy()
    changed[0] += 0.25
    with pytest.raises(CheckpointError, match="manifest"):
        _checkpoint_group_run(
            root, n_jobs=1, block_size=5, y=changed)

    alpha_root = tmp_path / "alpha"
    _checkpoint_group_run(alpha_root, n_jobs=1, block_size=5, alpha=0.05)
    with pytest.raises(CheckpointError, match="manifest"):
        _checkpoint_group_run(
            alpha_root, n_jobs=1, block_size=5, alpha=0.01)

    authority_root = tmp_path / "authority"
    _checkpoint_group_run(
        authority_root, n_jobs=1, block_size=5, inferential=False)
    with pytest.raises(CheckpointError, match="manifest"):
        _checkpoint_group_run(
            authority_root, n_jobs=1, block_size=5, inferential=True)
