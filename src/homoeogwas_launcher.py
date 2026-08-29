"""Lightweight console launcher with pre-import numeric thread control."""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence

NUMERIC_THREAD_ENV = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _configure_numeric_threads(argv: Sequence[str]) -> None:
    """Keep each canonical interaction worker single-threaded.

    This must run before importing :mod:`homoeogwas`, because NumPy/SciPy may
    initialize a native thread pool while those modules are imported.
    """
    if argv and argv[0] in {"interact", "follow-up"}:
        for name in NUMERIC_THREAD_ENV:
            os.environ[name] = "1"


def _import_cli_main():
    from homoeogwas.cli import main

    return main


def main(argv: Sequence[str] | None = None) -> int:
    actual_argv = list(sys.argv[1:] if argv is None else argv)
    _configure_numeric_threads(actual_argv)
    return _import_cli_main()(actual_argv)


if __name__ == "__main__":
    raise SystemExit(main())
