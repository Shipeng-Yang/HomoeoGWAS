from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml

from scripts.benchmarks.v201.contracts import sha256_payload

from .identity import sha256_file
from .plan import ContextSpec, InvocationSpec, ResponseSpec
from .policy import canonical_interact


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
    science = canonical_interact()
    burden = dict(science["burden"])
    calibration = science["calibration"]
    return {
        "interact": {
            "mode": science["mode"],
            "subgenomes": list(context.subgenomes),
            "groups": str(context.groups_path),
            "statistic": science["statistic"],
            "hypothesis_unit": science["hypothesis_unit"],
            "subset_order": science["subset_order"],
            "family_scope": science["family_scope"],
            "primary_transform": science["primary_transform"],
            "primary_multiplicity": science["primary_multiplicity"],
            "benchmark_identity": {
                "panel_id": context.panel_id,
                "sample_context": context.sample_context,
                "feature_seed": context.feature_seed,
            },
            "genotype": dict(context.bed_prefixes),
            "snp_to_gene": dict(context.snp_to_gene),
            "phenotype": str(phenotype_path),
            "sample_col": science["sample_col"],
            "trait": science["trait"],
            "burden": {**burden, "feature_seed": context.feature_seed},
            "grm": dict(science["grm"]),
            "calibration": {
                "method": calibration["method"],
                "B": calibration["B"],
                "seed": int(bootstrap_seed),
                "qa_only": calibration["qa_only"],
                "checkpoint": {
                    "enabled": calibration["checkpoint_enabled"],
                    "root": str(checkpoint_root),
                    "block_size": calibration["checkpoint_block_size"],
                },
            },
        },
        "outputs": {
            "out_dir": str(output_root),
            "full_ranking": science["full_ranking"],
        },
    }
