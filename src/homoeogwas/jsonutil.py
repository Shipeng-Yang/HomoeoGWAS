"""Strict RFC 8259 JSON helpers used by every public result writer."""
from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path

import numpy as np


def json_safe(value):
    """Recursively convert numpy objects and non-finite numbers to JSON values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    raise TypeError(f"not JSON serializable: {type(value)}")


def dumps_strict(value, **kwargs) -> str:
    """Serialize after normalization and reject any remaining NaN/Infinity."""
    return json.dumps(json_safe(value), allow_nan=False, **kwargs)


def dump_strict(value, fp, **kwargs) -> None:
    """Write strict JSON to an open text file."""
    json.dump(json_safe(value), fp, allow_nan=False, **kwargs)
