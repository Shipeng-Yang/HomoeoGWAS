"""Chunk-parallel streaming scan reproduces the serial streaming scans."""
from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pytest
from bed_reader import to_bed

from homoeogwas.grm import compute_loco_grm_parts, loco_grm_from_parts
from homoeogwas.kernel import normalize_kernel
from homoeogwas.scan import (
    build_loco_scan_contexts,
    build_scan_context,
    scan_bed_stream,
    scan_bed_stream_loco,
    scan_bed_stream_parallel,
)
from homoeogwas.io import GenoChunk


def _chunk(n=90, m_per_chrom=55, chroms=("cA", "cB", "cC"), seed=11):
    rng = np.random.default_rng(seed)
    m = m_per_chrom * len(chroms)
    p = rng.uniform(0.02, 0.5, size=m)
    X = rng.binomial(2, np.tile(p, (n, 1))).astype(np.float32)
    X[:, 3] = 0.0
    X[rng.random((n, m)) < 0.03] = np.nan
    X[: n // 2, 7] = np.nan
    return GenoChunk(
        samples=np.array([f"s{i:03d}" for i in range(n)], dtype=object),
        variant_ids=np.array([f"v{j:04d}" for j in range(m)], dtype=object),
        chrom=np.array([c for c in chroms for _ in range(m_per_chrom)], dtype=object),
        pos=np.array([j + 1 for _ in chroms for j in range(m_per_chrom)], dtype=np.int64),
        dosage=X,
    )


def _write_bed(chunk, prefix):
    to_bed(
        f"{prefix}.bed",
        np.where(np.isnan(chunk.dosage), -127, chunk.dosage).astype(np.int8),
        properties={
            "iid": [str(s) for s in chunk.samples],
            "sid": [str(v) for v in chunk.variant_ids],
            "chromosome": [str(c) for c in chunk.chrom],
            "bp_position": chunk.pos.astype(np.int32),
        },
        count_A1=True,
    )


def _contexts(chunk):
    n = chunk.dosage.shape[0]
    filled = np.where(np.isnan(chunk.dosage), 0.0, chunk.dosage)
    clean = GenoChunk(chunk.samples, chunk.variant_ids, chunk.chrom, chunk.pos, filled)
    global_part, parts = compute_loco_grm_parts(clean, maf_min=0.01)
    chroms = sorted(set(chunk.chrom))
    kbc = {c: {"g": normalize_kernel(loco_grm_from_parts(global_part, parts, c)[0], mode="trace")}
           for c in chroms}
    rng = np.random.default_rng(5)
    y = rng.normal(size=n)
    X = np.ones((n, 1))
    order = rng.permutation(n)[: n - 7]
    ids = chunk.samples[order]
    sigma2 = {"g": 0.5, "e": 0.5}
    loco = build_loco_scan_contexts(
        y[order], X[order], {c: {"g": k["g"][np.ix_(order, order)]} for c, k in kbc.items()},
        sigma2, sample_ids=ids)
    K = normalize_kernel(global_part.grm, mode="trace")
    plain = build_scan_context(y[order], X[order], {"g": K[np.ix_(order, order)]}, sigma2,
                               sample_ids=ids)
    return loco, plain


def _text(path, gz):
    if gz:
        with gzip.open(path, "rt") as fh:
            return fh.read()
    return path.read_text()


@pytest.mark.parametrize("gz", [True, False])
@pytest.mark.parametrize("chunk_size", [40, 70, 1000])
def test_parallel_loco_equals_serial(tmp_path, gz, chunk_size):
    chunk = _chunk()
    _write_bed(chunk, tmp_path / "g")
    loco, _ = _contexts(chunk)
    ext = ".tsv.gz" if gz else ".tsv"
    ser = scan_bed_stream_loco(loco, tmp_path / "g", tmp_path / f"s{ext}", backend="cpu",
                               chunk_size=chunk_size, subgenome="A", gzip_out=gz)
    par = scan_bed_stream_parallel(loco, tmp_path / "g", tmp_path / f"p{ext}", n_jobs=3,
                                   chunk_size=chunk_size, subgenome="A", gzip_out=gz)
    assert _text(tmp_path / f"p{ext}", gz) == _text(tmp_path / f"s{ext}", gz)
    assert (par.n_input, par.n_kept, par.n_chunks) == (ser.n_input, ser.n_kept, ser.n_chunks)
    assert par.filter_counts == ser.filter_counts
    assert sum(ser.filter_counts.values()) > 0


@pytest.mark.parametrize("chunk_size", [40, 1000])
def test_parallel_plain_equals_serial(tmp_path, chunk_size):
    chunk = _chunk(seed=12)
    _write_bed(chunk, tmp_path / "g")
    _, plain = _contexts(chunk)
    ser = scan_bed_stream(plain, tmp_path / "g", tmp_path / "s.tsv.gz", backend="cpu",
                          chunk_size=chunk_size, subgenome="B")
    par = scan_bed_stream_parallel(plain, tmp_path / "g", tmp_path / "p.tsv.gz", n_jobs=4,
                                   chunk_size=chunk_size, subgenome="B")
    assert _text(tmp_path / "p.tsv.gz", True) == _text(tmp_path / "s.tsv.gz", True)
    assert par.filter_counts == ser.filter_counts
    assert (par.n_input, par.n_kept) == (ser.n_input, ser.n_kept)


def test_parallel_rejects_unknown_chrom(tmp_path):
    chunk = _chunk(seed=13)
    _write_bed(chunk, tmp_path / "g")
    loco, _ = _contexts(_chunk(seed=13, chroms=("cA", "cB", "cX")))
    with pytest.raises(KeyError):
        scan_bed_stream_parallel(loco, tmp_path / "g", tmp_path / "p.tsv.gz", n_jobs=2,
                                 chunk_size=50)


def test_parallel_rejects_missing_samples(tmp_path):
    chunk = _chunk(seed=14)
    _write_bed(chunk, tmp_path / "g")
    loco, _ = _contexts(chunk)
    loco = loco.__class__(**{**loco.__dict__, "sample_ids": np.array(
        list(loco.sample_ids[:-1]) + ["absent"], dtype=object)})
    with pytest.raises(ValueError, match="absent from BED"):
        scan_bed_stream_parallel(loco, tmp_path / "g", tmp_path / "p.tsv.gz", n_jobs=2)


def test_parallel_rejects_bad_n_jobs(tmp_path):
    chunk = _chunk(seed=15)
    _write_bed(chunk, tmp_path / "g")
    loco, _ = _contexts(chunk)
    with pytest.raises(ValueError):
        scan_bed_stream_parallel(loco, tmp_path / "g", tmp_path / "p.tsv.gz", n_jobs=0)


def test_run_scan_routes_n_jobs_and_matches_serial(tmp_path):
    from homoeogwas import cli
    chunk = _chunk(seed=16)
    (tmp_path / "A").mkdir()
    _write_bed(chunk, tmp_path / "A" / "all")
    loco, _ = _contexts(chunk)
    outs = {}
    for jobs in (1, 3):
        cfg = {"genotype": {"scan_bed_prefix_template": str(tmp_path / "{subgenome}" / "all")},
               "scan": {"backend": "cpu", "chunk_size": 45, "n_jobs": jobs}}
        out = tmp_path / f"o{jobs}"
        out.mkdir()
        res = cli.run_scan(cfg, loco, ["A"], "cpu", out, "t", "stream")
        assert res["n_jobs"] == jobs
        outs[jobs] = (res, _text(Path(res["sumstats"][0]), True))
    assert outs[1][1] == outs[3][1]
    assert outs[1][0]["filter_counts"] == outs[3][0]["filter_counts"]


@pytest.mark.parametrize("value", [0, True, 2.5])
def test_validate_config_rejects_bad_scan_n_jobs(value):
    from homoeogwas import cli
    cfg = {"panel": {"subgenomes": ["A"]}, "phenotype": {"path": "p", "trait": "t"},
           "genotype": {"scan_bed_prefix_template": "x/{subgenome}"},
           "scan": {"n_jobs": value}}
    with pytest.raises(SystemExit, match="scan.n_jobs"):
        cli.validate_config(cfg)


def test_dead_worker_fails_instead_of_hanging(tmp_path, monkeypatch):
    import os
    from concurrent.futures.process import BrokenProcessPool

    from homoeogwas import scan as scan_mod
    chunk = _chunk(seed=17)
    _write_bed(chunk, tmp_path / "g")
    loco, _ = _contexts(chunk)
    real = scan_mod._parallel_chunk_body

    def dying(task):
        if task[0] == 50:
            os._exit(9)
        return real(task)

    monkeypatch.setattr(scan_mod, "_parallel_chunk_body", dying)
    with pytest.raises(BrokenProcessPool):
        scan_bed_stream_parallel(loco, tmp_path / "g", tmp_path / "p.tsv.gz", n_jobs=2,
                                 chunk_size=50)
    assert scan_mod._PARALLEL_STATE == {}
