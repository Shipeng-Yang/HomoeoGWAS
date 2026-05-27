"""Pytest bootstrap.

Ensures the in-tree ``src/`` layout is importable even when the package has not
been installed (``pip install -e .``). Runs before test collection, so individual
test modules can import ``homoeogwas`` at the top of the file without per-file
``sys.path`` manipulation (which previously triggered E402).
"""
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
