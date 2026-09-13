"""Fail-closed orchestration for the bounded BM-NATIVE-QA task."""

from __future__ import annotations

import os

SUCCESSOR_RUN_NAMESPACE = "qa_real80_njobs128_v2"
THREAD_ENV_NAMES = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def enforce_single_native_thread_before_imports() -> None:
    """Set native pools to one thread before any package submodule import."""

    for name in THREAD_ENV_NAMES:
        os.environ[name] = "1"


enforce_single_native_thread_before_imports()
