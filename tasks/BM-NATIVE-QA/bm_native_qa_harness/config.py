from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml

from scripts.benchmarks.v201.contracts import sha256_payload

from .identity import sha256_file
from .plan import ContextSpec, InvocationSpec, ResponseSpec


class ConfigError(RuntimeError):
    """A generated config would violate the frozen prospective identity."""


def dump_config(config: dict[str, object], path: Path) -> str:
    text = yaml.safe_dump(config, sort_keys=True, allow_unicode=True)
    try:
        with path.open("x", encoding="utf-8", newline="") as handle:
            handle.write(text)
    except FileExistsError as exc:
        raise ConfigError(f"config target already exists: {path}") from exc
    return sha256_file(path)


def scientific_config_sha256(config: dict[str, object]) -> str:
    payload = deepcopy(config)
    interact = payload["interact"]
    assert isinstance(interact, dict)
    calibration = interact["calibration"]
    assert isinstance(calibration, dict)
    checkpoint = calibration["checkpoint"]
    assert isinstance(checkpoint, dict)
    checkpoint.pop("root")
    outputs = payload["outputs"]
    assert isinstance(outputs, dict)
    outputs.pop("out_dir")
    return sha256_payload(payload)


def build_interact_config(
    context: ContextSpec,
    response: ResponseSpec,
    invocation: InvocationSpec,
    *,
    bootstrap_seed: int,
    phenotype_path: Path,
    checkpoint_root: Path,
    output_root: Path,
) -> dict[str, object]:
    if (
        response.context_key != context.key
        or invocation.response_id != response.response_id
        or invocation.panel_id != context.panel_id
        or invocation.sample_context != context.sample_context
        or invocation.truth_id != response.truth_id
    ):
        raise ConfigError("response/invocation identity mismatch")
    if output_root.exists():
        raise ConfigError(f"output target already exists: {output_root}")
    if checkpoint_root.exists():
        raise ConfigError(f"checkpoint target already exists: {checkpoint_root}")
    return {
        "interact": {
            "mode": "group",
            "subgenomes": list(context.subgenomes),
            "groups": str(context.groups_path),
            "statistic": "omniB",
            "hypothesis_unit": "group",
            "subset_order": 2,
            "family_scope": "primary_only",
            "primary_transform": "INT",
            "primary_multiplicity": "bootstrap_minp",
            "benchmark_identity": {
                "panel_id": context.panel_id,
                "sample_context": context.sample_context,
                "feature_seed": context.feature_seed,
            },
            "genotype": dict(context.bed_prefixes),
            "snp_to_gene": dict(context.snp_to_gene),
            "phenotype": str(phenotype_path),
            "sample_col": "sample",
            "trait": "qa_trait",
            "burden": {
                "cap": 150,
                "min_snp": 3,
                "maf_min": 0.01,
                "n_pc": 3,
                "feature_seed": context.feature_seed,
            },
            "grm": {
                "method": "grm_from_X",
                "maf_min": 0.01,
                "scope": "all_subgenomes",
            },
            "calibration": {
                "method": "bootstrap",
                "B": 199,
                "seed": int(bootstrap_seed),
                "qa_only": True,
                "checkpoint": {
                    "enabled": True,
                    "root": str(checkpoint_root),
                    "block_size": 25,
                },
            },
        },
        "outputs": {"out_dir": str(output_root), "full_ranking": True},
    }
