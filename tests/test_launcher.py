from __future__ import annotations

import os

import pytest

NUMERIC_THREAD_ENV = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def test_interact_launcher_forces_numeric_thread_env_before_import(monkeypatch):
    import homoeogwas_launcher as launcher

    for name in NUMERIC_THREAD_ENV:
        monkeypatch.setenv(name, "64")
    imported = []

    def fake_import():
        imported.append({name: os.environ.get(name) for name in NUMERIC_THREAD_ENV})
        return lambda _argv: 0

    monkeypatch.setattr(launcher, "_import_cli_main", fake_import)
    assert launcher.main(["interact", "--help"]) == 0
    assert imported == [{name: "1" for name in NUMERIC_THREAD_ENV}]


def test_non_interact_launcher_preserves_numeric_thread_env(monkeypatch):
    import homoeogwas_launcher as launcher

    for name in NUMERIC_THREAD_ENV:
        monkeypatch.setenv(name, "64")
    monkeypatch.setattr(launcher, "_import_cli_main", lambda: lambda _argv: 0)
    assert launcher.main(["fit", "--help"]) == 0
    assert {name: os.environ[name] for name in NUMERIC_THREAD_ENV} == {
        name: "64" for name in NUMERIC_THREAD_ENV
    }


def test_runtime_firewall_rejects_oversubscribed_canonical_omnib():
    from homoeogwas.interact import _validate_canonical_parallel_runtime

    libraries = [{"user_api": "blas", "prefix": "libopenblas", "num_threads": 64}]
    with pytest.raises(RuntimeError, match="installed.*homoeogwas interact"):
        _validate_canonical_parallel_runtime(n_jobs=4, libraries=libraries)


def test_runtime_firewall_allows_one_thread_libraries_and_serial():
    from homoeogwas.interact import _validate_canonical_parallel_runtime

    one_thread = [{"user_api": "blas", "prefix": "libopenblas", "num_threads": 1}]
    hostile = [{"user_api": "blas", "prefix": "libopenblas", "num_threads": 64}]
    _validate_canonical_parallel_runtime(n_jobs=4, libraries=one_thread)
    _validate_canonical_parallel_runtime(n_jobs=1, libraries=hostile)
