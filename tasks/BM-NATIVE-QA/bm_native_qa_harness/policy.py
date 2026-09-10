from __future__ import annotations

from copy import deepcopy
from types import MappingProxyType
from typing import Any

_CANONICAL_INTERACT: dict[str, Any] = {
    "mode": "group",
    "statistic": "omniB",
    "hypothesis_unit": "group",
    "subset_order": 2,
    "family_scope": "primary_only",
    "primary_transform": "INT",
    "primary_multiplicity": "bootstrap_minp",
    "burden": {"cap": 150, "min_snp": 3, "maf_min": 0.01, "n_pc": 3},
    "grm": {
        "method": "grm_from_X",
        "maf_min": 0.01,
        "scope": "all_subgenomes",
    },
    "calibration": {
        "method": "bootstrap",
        "B": 199,
        "qa_only": True,
        "checkpoint_enabled": True,
        "checkpoint_block_size": 25,
    },
    "sample_col": "sample",
    "trait": "qa_trait",
    "full_ranking": True,
}

NATIVE_CAPS = MappingProxyType(
    {
        "wall_seconds": 7200,
        "cpu_seconds": 115200,
        "peak_aggregate_pss_gib": 128,
        "output_storage_gib": 20,
        "large_tmp_output_forbidden": True,
    }
)


def canonical_interact() -> dict[str, Any]:
    """Return an isolated copy of the one frozen scientific policy."""

    return deepcopy(_CANONICAL_INTERACT)


def preparation_options(
    *,
    feature_seed: int,
    retained_variant_masks: Any,
) -> dict[str, Any]:
    """Map the frozen policy onto ``prepare_omnib_design`` keyword names."""

    science = canonical_interact()
    burden = science["burden"]
    grm = science["grm"]
    return {
        "transform": science["primary_transform"],
        "feature_seed": feature_seed,
        "retained_variant_masks": retained_variant_masks,
        "grm_method": grm["method"],
        "maf_min": grm["maf_min"],
        "burden_maf": burden["maf_min"],
        "min_snp": burden["min_snp"],
        "cap": burden["cap"],
        "n_pc": burden["n_pc"],
    }
