from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

TASK_ROOT = Path(__file__).resolve().parents[1]
MANAGEMENT_ROOT = Path("/mnt/7302share/fast_ysp/U7_GWAS")
if str(TASK_ROOT) not in sys.path:
    sys.path.insert(0, str(TASK_ROOT))


@pytest.fixture
def amendment() -> dict:
    path = MANAGEMENT_ROOT / "tasks/BM-NATIVE-QA/QA-EXECUTION-AMENDMENT-v1-20260910.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@pytest.fixture
def context_evidence() -> dict:
    path = (
        MANAGEMENT_ROOT
        / "tasks/BM-INPUTS/staging/real-core-v4-primary80-estimability/manifest.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))
