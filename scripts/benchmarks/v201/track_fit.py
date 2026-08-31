"""Track A fit benchmark primitives for the v2.0.1 benchmark."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import shutil
import subprocess
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from homoeogwas.sim import kernel_factor

from .contracts import Scenario, derive_seed, sha256_payload
from .shards import ShardConflict, ShardKey, load_shard, write_shard_exclusive


def select_pilot_samples(
    sample_ids: Sequence[str] | np.ndarray,
    pc1: Sequence[float] | np.ndarray,
    n: int = 192,
    *,
    return_metadata: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    """Select deterministic PC1-spanning sample indices, including extremes."""

    ids = np.asarray(sample_ids)
    scores = np.asarray(pc1, dtype=float)
    if ids.ndim != 1 or scores.ndim != 1 or ids.shape != scores.shape:
        raise ValueError("sample_ids and pc1 must be aligned one-dimensional arrays")
    if ids.size < 2 or not np.all(np.isfinite(scores)):
        raise ValueError("sample_ids and pc1 must contain at least two finite rows")
    if ids.dtype.kind not in {"U", "S", "O"} or any(
        not isinstance(value, (str, np.str_)) for value in ids.tolist()
    ):
        raise ValueError("sample IDs must be strings")
    ids = ids.astype(str, copy=False)
    if np.unique(ids).size != ids.size:
        raise ValueError("sample IDs must be unique")
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or not 2 <= int(n) <= ids.size:
        raise ValueError("n must select between two and all available samples")

    ordered = np.lexsort((ids, scores))
    ranks = np.rint(np.linspace(0, ids.size - 1, int(n))).astype(int)
    selected = ordered[ranks]
    if np.unique(selected).size != int(n):  # defensive for unusual integer rounding
        raise RuntimeError("evenly spaced PC1 ranks were not unique")
    selected = selected.astype(np.int64, copy=False)
    if not return_metadata:
        return selected
    ordered_ids = ids[selected].tolist()
    metadata = {
        "method": "pc1_evenly_spaced_ranks_string_id_tiebreak",
        "n_available": int(ids.size),
        "n_selected": int(selected.size),
        "ordered_sample_ids": ordered_ids,
        "ordered_sample_ids_sha256": sha256_payload(ordered_ids),
        "includes_pc1_min": True,
        "includes_pc1_max": True,
    }
    return selected, metadata


def _bed_template(fixture: Mapping[str, Any], out_dir: Path) -> str:
    supplied = fixture.get("bed_template", fixture.get("scan_bed_prefix_template"))
    if supplied is not None:
        template = str(supplied)
        if "{subgenome}" not in template:
            raise ValueError("bed template must contain {subgenome}")
        return template

    subgenomes = [str(value) for value in fixture["subgenomes"]]
    prefixes = fixture.get("bed_prefixes")
    if not isinstance(prefixes, Mapping) or set(prefixes) != set(subgenomes):
        raise ValueError("bed_prefixes must contain exactly the declared subgenomes")
    missing = [
        f"{prefixes[subgenome]}{extension}"
        for subgenome in subgenomes
        for extension in (".bed", ".bim", ".fam")
        if not Path(f"{prefixes[subgenome]}{extension}").is_file()
    ]
    if missing:
        raise FileNotFoundError(f"missing PLINK component: {missing[0]}")
    parts = [Path(str(prefixes[subgenome])).parts for subgenome in subgenomes]
    if len({len(value) for value in parts}) == 1:
        differing = [
            index
            for index in range(len(parts[0]))
            if len({value[index] for value in parts}) > 1
        ]
        if len(differing) == 1:
            index = differing[0]
            if [value[index] for value in parts] == subgenomes:
                template_parts = list(parts[0])
                template_parts[index] = "{subgenome}"
                return str(Path(*template_parts))
    for subgenome in subgenomes:
        destination_dir = out_dir / "geno" / subgenome
        destination_dir.mkdir(parents=True, exist_ok=True)
        for extension in (".bed", ".bim", ".fam"):
            source = Path(f"{prefixes[subgenome]}{extension}").resolve()
            destination = destination_dir / f"all{extension}"
            if destination.is_symlink():
                if destination.resolve(strict=False) == source:
                    continue
                raise FileExistsError(
                    f"managed PLINK symlink points elsewhere: {destination}"
                )
            if destination.exists():
                raise FileExistsError(
                    f"managed PLINK path already exists: {destination}"
                )
            destination.symlink_to(source)
    return str(out_dir / "geno" / "{subgenome}" / "all")


def build_fit_config(
    fixture: Mapping[str, Any],
    out_dir: str | Path,
    *,
    coverage: bool = False,
    bootstrap_B: int = 200,
    bootstrap_jobs: int = 1,
    loco: bool = False,
) -> dict[str, Any]:
    """Build the locked, portable Track A HomoeoGWAS fit configuration."""

    root = Path(out_dir)
    output = root / "results"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"results directory is not empty: {output}")
    subgenomes = [str(value) for value in fixture["subgenomes"]]
    template = _bed_template(fixture, root)
    reml: dict[str, Any] = {"n_starts": 10, "seed": 2026}
    if coverage:
        if type(bootstrap_B) is not int or bootstrap_B < 1:
            raise ValueError("bootstrap_B must be a positive integer")
        reml["pve_bootstrap"] = {
            "enabled": True,
            "B": bootstrap_B,
            "level": 0.95,
            "n_jobs": int(bootstrap_jobs),
            "n_starts": 10,
        }
    cfg = {
        "fit_version": 1,
        "panel": {"name": str(fixture["panel"]), "subgenomes": subgenomes},
        "phenotype": {
            "path": str(fixture["phenotype"]),
            "sample_col": str(fixture["sample_col"]),
            "trait": str(fixture["trait"]),
        },
        "genotype": {
            "scan_bed_prefix_template": template,
            "grm": {
                "source": "bed",
                "bed_prefix_template": template,
                "maf_min": 0.05,
            },
        },
        "kernels": {
            "normalize": "trace",
            "include_hadamard": False,
            "hadamard_name": "hom",
        },
        "reml": reml,
        "scan": {
            "mode": "memory",
            "backend": "cpu",
            "maf_min": 0.05,
            "call_rate_min": 0.9,
            "loco": {"enabled": bool(loco), **({"fallback": "error"} if loco else {})},
        },
        "plots": {"enabled": True},
        "outputs": {"out_dir": str(output), "prefix": str(fixture["trait"])},
    }
    from homoeogwas.cli import validate_config
    from homoeogwas.workflow import write_config

    config_path = root / "configs" / "fit.generated.yaml"
    if Path(write_config(cfg, config_path)) != config_path:
        raise RuntimeError("fit config was not written as YAML")
    validate_config(cfg)
    return cfg


def _checked_kernels(
    kernels: Mapping[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], int]:
    if not isinstance(kernels, Mapping) or not kernels:
        raise ValueError("kernels must be a non-empty ordered mapping")
    checked: dict[str, np.ndarray] = {}
    n: int | None = None
    for name, raw in kernels.items():
        if not isinstance(name, str) or not name or name == "e":
            raise ValueError("kernel names must be non-empty strings other than e")
        value = np.asarray(raw, dtype=np.float64)
        if (
            value.ndim != 2
            or value.shape[0] != value.shape[1]
            or value.shape[0] < 2
            or not np.all(np.isfinite(value))
        ):
            raise ValueError(f"kernel {name!r} must be a finite square matrix")
        if n is None:
            n = value.shape[0]
        elif value.shape != (n, n):
            raise ValueError("all kernels must have the same sample axis")
        checked[name] = np.ascontiguousarray(value)
    assert n is not None
    return checked, n


def _kernel_fingerprint(value: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(value, dtype="<f8")
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return {
        "shape": [int(axis) for axis in array.shape],
        "dtype": array.dtype.str,
        "sha256": digest.hexdigest(),
        "trace": float(np.trace(array)),
    }


def _target_pve(
    names: list[str],
    allocation: str | Mapping[str, float],
    total_pve: float | None,
    dominant_subgenomes: Sequence[str] | None,
    null_subgenome: str | None,
) -> tuple[str, dict[str, float], float]:
    if isinstance(allocation, Mapping):
        if set(allocation) != set(names):
            raise ValueError("allocation must contain exactly the kernel names")
        targets = {name: float(allocation[name]) for name in names}
        if any(not np.isfinite(value) or value < 0.0 for value in targets.values()):
            raise ValueError("allocation PVE values must be finite and non-negative")
        inferred = float(sum(targets.values()))
        if total_pve is not None and not np.isclose(inferred, float(total_pve)):
            raise ValueError("allocation PVE values do not sum to total_pve")
        total = inferred
        allocation_id = "explicit"
    else:
        allocation_id = str(allocation)
        total = float(
            0.15
            if allocation_id == "low_total_pve" and total_pve is None
            else 0.40 if total_pve is None else total_pve
        )
        count = len(names)
        if allocation_id in {"balanced", "low_total_pve"}:
            weights = {name: 1.0 / count for name in names}
        elif allocation_id == "single_dominant":
            dominant = list(dominant_subgenomes or names[:1])
            if len(dominant) != 1 or dominant[0] not in names or count < 2:
                raise ValueError("single_dominant requires one declared subgenome")
            weights = {
                name: 0.8 if name == dominant[0] else 0.2 / (count - 1)
                for name in names
            }
        elif allocation_id == "two_dominant":
            dominant = list(dominant_subgenomes or names[:2])
            if len(dominant) != 2 or len(set(dominant)) != 2 or not set(dominant) <= set(names):
                raise ValueError("two_dominant requires two distinct subgenomes")
            if count == 2:
                weights = {name: 0.5 for name in names}
            else:
                weights = {
                    name: 0.45 if name in dominant else 0.10 / (count - 2)
                    for name in names
                }
        elif allocation_id == "null_component":
            null_name = null_subgenome or names[0]
            if null_name not in names or count < 2:
                raise ValueError("null_component requires one declared subgenome")
            weights = {
                name: 0.0 if name == null_name else 1.0 / (count - 1)
                for name in names
            }
        else:
            raise ValueError(f"unknown Track A allocation: {allocation_id!r}")
        targets = {name: round(total * weights[name], 15) for name in names}
    if not 0.0 <= total < 1.0:
        raise ValueError("total genetic PVE must be in [0, 1)")
    return allocation_id, targets, total


def _scaled_component(
    rng: np.random.Generator, factor: np.ndarray, target_variance: float
) -> np.ndarray:
    if target_variance == 0.0:
        return np.zeros(factor.shape[0], dtype=np.float64)
    value = factor @ rng.standard_normal(factor.shape[1])
    value -= value.mean()
    variance = float(np.var(value, ddof=1))
    if variance <= 0.0:
        raise ValueError("degenerate kernel component draw")
    return value * np.sqrt(target_variance / variance)


def simulate_fit_truth(
    kernels: Mapping[str, np.ndarray],
    allocation: str | Mapping[str, float],
    *,
    seed: int,
    total_pve: float | None = None,
    dominant_subgenomes: Sequence[str] | None = None,
    null_subgenome: str | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Draw one finite-sample-scaled phenotype from fingerprinted kernels."""

    checked, n = _checked_kernels(kernels)
    names = list(checked)
    allocation_id, genetic_targets, genetic_total = _target_pve(
        names,
        allocation,
        total_pve,
        dominant_subgenomes,
        null_subgenome,
    )
    residual_target = round(1.0 - genetic_total, 15)
    rng = np.random.default_rng(int(seed))
    relative_components = {
        name: _scaled_component(rng, kernel_factor(checked[name]), genetic_targets[name])
        for name in names
    }
    genetic_relative = np.sum(
        np.vstack(list(relative_components.values())), axis=0
    )
    relative_genetic_variance = float(np.var(genetic_relative, ddof=1))
    if genetic_total > 0.0:
        if relative_genetic_variance <= 0.0:
            raise ValueError("degenerate total genetic component draw")
        common_genetic_scale = math.sqrt(
            genetic_total / relative_genetic_variance
        )
    else:
        common_genetic_scale = 0.0
    components = {
        name: value * common_genetic_scale
        for name, value in relative_components.items()
    }
    genetic = np.sum(np.vstack(list(components.values())), axis=0)

    residual = rng.standard_normal(n)
    residual -= residual.mean()
    genetic_ss = float(genetic @ genetic)
    if genetic_ss > 0.0:
        residual -= genetic * float(genetic @ residual) / genetic_ss
    residual -= residual.mean()
    residual_variance = float(np.var(residual, ddof=1))
    if residual_target > 0.0:
        if residual_variance <= 0.0:
            raise ValueError("degenerate residual draw")
        residual *= math.sqrt(residual_target / residual_variance)
    else:
        residual = np.zeros(n, dtype=np.float64)
    components["e"] = residual
    phenotype = genetic + residual
    total_variance = float(np.var(phenotype, ddof=1))
    if not np.isclose(total_variance, 1.0, rtol=1e-10, atol=1e-12):
        raise RuntimeError("finite-sample phenotype variance is not one")

    target = {**genetic_targets, "e": residual_target}
    component_effect_variance = {
        name: float(np.var(value, ddof=1)) for name, value in components.items()
    }
    marginal_effect_pve = {
        name: float(value / total_variance)
        for name, value in component_effect_variance.items()
    }
    cross_covariance = {
        left: {
            right: float(
                np.dot(components[left], components[right]) / (n - 1)
            )
            for right in components
        }
        for left in components
    }
    observed_total_genetic_pve = float(
        np.var(genetic, ddof=1) / total_variance
    )
    kernel_fingerprints = {
        name: _kernel_fingerprint(value) for name, value in checked.items()
    }
    context_hash = sha256_payload(
        {"kernel_order": names, "kernel_fingerprints": kernel_fingerprints}
    )
    allocation_record = {
        "allocation": allocation_id,
        "target_pve": target,
        "dominant_subgenomes": list(dominant_subgenomes or []),
        "null_subgenome": null_subgenome,
    }
    allocation_hash = sha256_payload(allocation_record)
    truth = {
        "allocation": allocation_id,
        "allocation_hash": allocation_hash,
        "context_hash": context_hash,
        "kernel_order": names,
        "kernel_fingerprints": kernel_fingerprints,
        "seed": int(seed),
        "target_pve": target,
        "target_allocation": dict(genetic_targets),
        "component_effect_sample_variance": component_effect_variance,
        "marginal_component_effect_pve": marginal_effect_pve,
        "component_cross_covariance": cross_covariance,
        "common_genetic_scale": float(common_genetic_scale),
        "total_genetic_effect_sample_variance": float(
            np.var(genetic, ddof=1)
        ),
        "observed_total_genetic_pve": observed_total_genetic_pve,
        "observed_phenotype_sample_variance": total_variance,
    }
    truth["truth_hash"] = sha256_payload(truth)
    return phenotype, truth


def fit_pve_replicate(
    y: Sequence[float] | np.ndarray,
    kernels: Mapping[str, np.ndarray],
    seed: int,
) -> dict[str, Any]:
    """Fit one full multi-kernel REML replicate without dropping failures."""

    from homoeogwas.lmm import fit_multi_reml

    checked, n = _checked_kernels(kernels)
    response = np.asarray(y, dtype=float)
    if response.ndim != 1 or response.shape[0] != n or not np.all(np.isfinite(response)):
        raise ValueError("y must be a finite vector aligned to kernels")
    started = time.perf_counter()
    result = fit_multi_reml(
        response,
        np.ones((n, 1), dtype=float),
        checked,
        n_starts=10,
        random_state=int(seed % (2**32)),
    )
    return {
        "estimated_pve": {name: float(value) for name, value in result.pve.items()},
        "estimated_sigma2": {
            name: float(value) for name, value in result.sigma2.items()
        },
        "boundary_components": list(result.boundary_components),
        "optimizer_status": bool(result.optimizer_status),
        "optimizer_message": str(result.optimizer_message),
        "converged": bool(result.optimizer_status),
        "n_iter": int(result.n_iter),
        "n_starts": int(result.n_starts),
        "runtime_seconds": float(time.perf_counter() - started),
    }


def _true_pve(truth: Mapping[str, Any] | None) -> dict[str, float] | None:
    if truth is None:
        return None
    source = truth.get("target_pve", truth)
    if not isinstance(source, Mapping):
        raise ValueError("truth must be a PVE mapping or contain realized_pve")
    result = {str(name): float(value) for name, value in source.items()}
    if any(not math.isfinite(value) for value in result.values()):
        raise ValueError("true PVE values must be finite")
    return result


def _finite_pve(
    values: Mapping[str, Any],
) -> tuple[dict[str, float | None], list[str]]:
    parsed: dict[str, float | None] = {}
    nonfinite: list[str] = []
    for name, raw in values.items():
        value = float(raw)
        if math.isfinite(value):
            parsed[str(name)] = value
        else:
            parsed[str(name)] = None
            nonfinite.append(str(name))
    return parsed, nonfinite


def _fit_layout(root: str | Path) -> tuple[Path, Path]:
    replicate_root = Path(root).resolve()
    return replicate_root, replicate_root / "results"


def _read_yaml_mapping(path: Path) -> Mapping[str, Any]:
    import yaml

    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"cannot read fit config: {path}") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"fit config is not a mapping: {path}")
    return value


def _configured_bed_sources(
    config: Mapping[str, Any],
) -> dict[str, Path]:
    panel = config.get("panel")
    genotype = config.get("genotype")
    if not isinstance(panel, Mapping) or not isinstance(genotype, Mapping):
        raise ValueError("fit config panel/genotype provenance is invalid")
    subgenomes = panel.get("subgenomes")
    if not isinstance(subgenomes, list) or not subgenomes:
        raise ValueError("fit config subgenomes are invalid")
    scan_template = genotype.get("scan_bed_prefix_template")
    grm = genotype.get("grm")
    if not isinstance(scan_template, str) or "{subgenome}" not in scan_template:
        raise ValueError("fit scan BED template provenance is invalid")
    if not isinstance(grm, Mapping) or grm.get("source", "bed") != "bed":
        raise ValueError("Track A released provenance requires BED-sourced GRMs")
    grm_template = grm.get("bed_prefix_template", scan_template)
    if not isinstance(grm_template, str) or "{subgenome}" not in grm_template:
        raise ValueError("fit GRM BED template provenance is invalid")
    sources: dict[str, Path] = {}
    for source_name, template in (("scan", scan_template), ("grm", grm_template)):
        for subgenome in subgenomes:
            prefix = Path(template.format(subgenome=subgenome)).resolve()
            for extension in ("bed", "bim", "fam"):
                sources[f"{source_name}_{subgenome}_{extension}"] = Path(
                    f"{prefix}.{extension}"
                )
    return sources


def _released_config_context(root: str | Path) -> dict[str, Any]:
    replicate_root, output = _fit_layout(root)
    generated_path = replicate_root / "configs" / "fit.generated.yaml"
    resolved_path = output / "resolved_config.yaml"
    generated = _read_yaml_mapping(generated_path)
    resolved = _read_yaml_mapping(resolved_path)
    return {
        "root": replicate_root,
        "output": output,
        "generated_path": generated_path,
        "resolved_path": resolved_path,
        "generated": generated,
        "resolved": resolved,
        "bed_sources": _configured_bed_sources(resolved),
    }


def _fam_iids(path: Path) -> list[str]:
    """Read one PLINK FAM IID column without numeric coercion."""

    try:
        rows = [line.split() for line in path.read_text(encoding="utf-8").splitlines()]
    except OSError as error:
        raise ValueError(f"cannot read FAM sample provenance: {path}") from error
    if not rows or any(len(row) < 2 or not row[1] for row in rows):
        raise ValueError(f"FAM sample provenance is invalid: {path}")
    iids = [row[1] for row in rows]
    if len(set(iids)) != len(iids):
        raise ValueError(f"FAM contains duplicate sample IIDs: {path}")
    return iids


def _released_fit_inputs(
    context: Mapping[str, Any],
    analysis_samples: Sequence[str],
    *,
    expected_phenotype: Sequence[float] | np.ndarray | None,
    expected_kernels: Mapping[str, np.ndarray] | None,
    expected_truth_hash: str | None = None,
) -> dict[str, Any]:
    """Reproduce production sample joining and BED-derived kernel construction."""

    from homoeogwas.cli import build_kernels, join_samples
    from homoeogwas.io import load_bed_hardcall

    analysis_ids = list(analysis_samples)
    bed_sources = context["bed_sources"]
    observed_prefixes: set[Path] = set()
    for key, fam_path in bed_sources.items():
        if not key.endswith("_fam"):
            continue
        fam_iids = _fam_iids(fam_path)
        missing = [sample for sample in analysis_ids if sample not in set(fam_iids)]
        if missing:
            raise ValueError(
                f"FAM sample join is missing analysis IIDs for {key}: {missing[:3]}"
            )
        prefix = Path(str(fam_path)[: -len(".fam")])
        if prefix in observed_prefixes:
            continue
        observed_prefixes.add(prefix)
        try:
            genotype = load_bed_hardcall(prefix)
        except Exception as error:
            raise ValueError(f"FAM/BED sample provenance is inconsistent: {prefix}") from error
        if genotype.samples.astype(str).tolist() != fam_iids:
            raise ValueError(f"FAM IID order differs from BED sample axis: {prefix}")

    try:
        joined_samples, joined_phenotype, _ = join_samples(dict(context["resolved"]))
    except SystemExit as error:
        raise ValueError(f"production fit sample join failed: {error}") from error
    except Exception as error:
        raise ValueError("production fit sample join failed") from error
    joined_ids = [str(sample) for sample in joined_samples]
    joined_y = np.ascontiguousarray(np.asarray(joined_phenotype, dtype=np.float64))
    if joined_ids != analysis_ids:
        raise ValueError(
            "production fit duplicate-averaging/inner sample join differs from analysis samples"
        )
    requested_y: np.ndarray | None = None
    phenotype_max_abs: float | None = None
    if expected_phenotype is not None:
        requested_y = np.ascontiguousarray(
            np.asarray(expected_phenotype, dtype=np.float64)
        )
        phenotype_max_abs = (
            float(np.max(np.abs(joined_y - requested_y)))
            if requested_y.shape == joined_y.shape
            else None
        )
        roundtrip_tolerance = 8.0 * np.finfo(np.float64).eps * max(
            1.0,
            float(np.max(np.abs(requested_y))) if requested_y.size else 1.0,
        )
        if (
            requested_y.shape != joined_y.shape
            or not np.all(np.isfinite(requested_y))
            or phenotype_max_abs is None
            or phenotype_max_abs > roundtrip_tolerance
        ):
            raise ValueError(
                "production-joined phenotype differs from request simulation "
                f"(joined={_exact_array_fingerprint(joined_y)['sha256']}, "
                f"requested={_exact_array_fingerprint(requested_y)['sha256']}, "
                f"max_abs={phenotype_max_abs})"
            )

    try:
        released_kernels, grm_info = build_kernels(
            dict(context["resolved"]), joined_samples
        )
    except SystemExit as error:
        raise ValueError(f"production BED-derived kernel build failed: {error}") from error
    except Exception as error:
        raise ValueError("production BED-derived kernel build failed") from error
    released_checked, _ = _checked_kernels(released_kernels)
    if expected_kernels is not None:
        requested_checked, requested_n = _checked_kernels(expected_kernels)
        if requested_n != len(joined_ids) or list(requested_checked) != list(
            released_checked
        ):
            raise ValueError("production BED-derived kernel order differs from request")
        for name in requested_checked:
            if _kernel_fingerprint(requested_checked[name]) != _kernel_fingerprint(
                released_checked[name]
            ):
                raise ValueError(
                    f"production BED-derived kernel differs from request: {name}"
                )

    binding = {
        "analysis_sample_ids": joined_ids,
        "analysis_sample_ids_sha256": sha256_payload(joined_ids),
        "joined_phenotype": _exact_array_fingerprint(joined_y),
        "requested_phenotype": (
            _exact_array_fingerprint(requested_y)
            if requested_y is not None
            else None
        ),
        "phenotype_max_abs_difference": phenotype_max_abs,
        "truth_hash": expected_truth_hash,
        "kernel_order": list(released_checked),
        "kernel_fingerprints": {
            name: _kernel_fingerprint(value)
            for name, value in released_checked.items()
        },
        "grm_info": _json_safe(grm_info),
        "kernel_source": "production_build_kernels_from_resolved_bed",
    }
    binding["sha256"] = sha256_payload(binding)
    return binding


def parse_fit_metrics(
    out_dir: str | Path,
    trait: str,
    *,
    truth: Mapping[str, Any] | None = None,
    expected_subgenomes: Sequence[str] | None = None,
    expected_sample_ids: Sequence[str] | None = None,
    expected_phenotype: Sequence[float] | np.ndarray | None = None,
    expected_kernels: Mapping[str, np.ndarray] | None = None,
    expected_scenario: Scenario | None = None,
    experiment: str | None = None,
) -> dict[str, Any]:
    """Parse only released fit JSON/TSV outputs into replicate metrics."""

    context = _released_config_context(out_dir)
    output = context["output"]
    summary_path = output / f"summary_{trait}.json"
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read fit summary: {summary_path}") from error
    if not isinstance(summary, Mapping) or not isinstance(summary.get("reml"), Mapping):
        raise ValueError("fit summary does not contain a REML result")
    if summary.get("tool") != "homoeogwas" or summary.get("command") != "fit":
        raise ValueError("fit summary tool/command provenance is invalid")
    if summary.get("trait") != trait:
        raise ValueError("fit summary trait does not match the request")
    subgenomes = summary.get("subgenomes")
    if (
        not isinstance(subgenomes, list)
        or not subgenomes
        or any(not isinstance(value, str) or not value for value in subgenomes)
        or len(set(subgenomes)) != len(subgenomes)
    ):
        raise ValueError("fit summary subgenomes are invalid")
    if expected_subgenomes is not None and subgenomes != list(expected_subgenomes):
        raise ValueError("fit summary subgenome order differs from the request")
    if summary.get("kernel_names") != subgenomes:
        raise ValueError("fit summary kernel order differs from subgenomes")

    generated_config_path = context["generated_path"]
    resolved_config_path = output / "resolved_config.yaml"
    analysis_samples_path = output / "analysis_samples.tsv"
    reported_config = Path(str(summary.get("config", ""))).resolve()
    outputs = summary.get("outputs")
    if not isinstance(outputs, Mapping):
        raise ValueError("fit summary outputs provenance is missing")
    if reported_config != generated_config_path or not generated_config_path.is_file():
        raise ValueError("fit summary generated config path is invalid")
    if Path(str(outputs.get("out_dir", ""))).resolve() != output:
        raise ValueError("fit summary output directory is invalid")
    if Path(str(outputs.get("resolved_config", ""))).resolve() != resolved_config_path:
        raise ValueError("fit summary resolved config path is invalid")
    if Path(str(outputs.get("analysis_samples", ""))).resolve() != analysis_samples_path:
        raise ValueError("fit summary analysis sample path is invalid")
    if not resolved_config_path.is_file() or not analysis_samples_path.is_file():
        raise ValueError("fit summary required provenance files are missing")
    resolved_config = context["resolved"]
    generated_config = context["generated"]
    if (
        resolved_config.get("panel", {}).get("subgenomes") != subgenomes
        or resolved_config.get("phenotype", {}).get("trait") != trait
        or Path(str(resolved_config.get("outputs", {}).get("out_dir", ""))).resolve()
        != output
    ):
        raise ValueError("fit resolved config does not match summary provenance")
    if generated_config != resolved_config:
        raise ValueError("generated and resolved fit configs differ")
    grm_info = summary.get("grm_info")
    kernels_config = resolved_config.get("kernels", {})
    grm_config = resolved_config.get("genotype", {}).get("grm", {})
    expected_kernel_names = list(subgenomes)
    if kernels_config.get("include_hadamard", False):
        expected_kernel_names.append(str(kernels_config.get("hadamard_name", "hom")))
    if (
        not isinstance(grm_info, Mapping)
        or grm_info.get("source") != grm_config.get("source", "bed")
        or grm_info.get("normalize") != kernels_config.get("normalize", "trace")
        or grm_info.get("kernel_names") != expected_kernel_names
        or summary.get("kernel_names") != expected_kernel_names
    ):
        raise ValueError("fit summary GRM provenance differs from resolved config")
    marker_input = grm_info.get("marker_input")
    if not isinstance(marker_input, Mapping):
        raise ValueError("fit summary GRM marker-source provenance is missing")
    for source_name in ("scan", "grm"):
        source_record = marker_input.get(source_name)
        if not isinstance(source_record, Mapping) or not isinstance(
            source_record.get("subgenomes"), Mapping
        ):
            raise ValueError("fit summary GRM marker-source provenance is invalid")
        for subgenome in subgenomes:
            item = source_record["subgenomes"].get(subgenome)
            expected_bed = context["bed_sources"][
                f"{source_name}_{subgenome}_bed"
            ]
            expected_prefix = str(expected_bed)[: -len(".bed")]
            if (
                not isinstance(item, Mapping)
                or Path(str(item.get("bed_prefix", ""))).resolve()
                != Path(expected_prefix)
            ):
                raise ValueError(
                    "fit summary GRM marker source differs from resolved config"
                )
    for source_path in context["bed_sources"].values():
        if not source_path.is_file():
            raise ValueError(f"fit BED provenance file is missing: {source_path}")

    with analysis_samples_path.open(encoding="utf-8", newline="") as handle:
        sample_reader = csv.DictReader(handle, delimiter="\t")
        if sample_reader.fieldnames != ["sample"]:
            raise ValueError("fit analysis sample schema is invalid")
        analysis_samples = [row["sample"] for row in sample_reader]
    if (
        any(not value for value in analysis_samples)
        or len(set(analysis_samples)) != len(analysis_samples)
        or summary.get("n_analysis") != len(analysis_samples)
    ):
        raise ValueError("fit analysis sample order/count is invalid")
    if expected_sample_ids is not None:
        expected_ids = list(expected_sample_ids)
        if any(not isinstance(value, str) for value in expected_ids):
            raise ValueError("expected sample IDs must be strings")
        if analysis_samples != expected_ids:
            raise ValueError("fit analysis sample order differs from the request")
    released_input_binding = _released_fit_inputs(
        context,
        analysis_samples,
        expected_phenotype=expected_phenotype,
        expected_kernels=expected_kernels,
        expected_truth_hash=(
            str(truth["truth_hash"])
            if isinstance(truth, Mapping) and isinstance(truth.get("truth_hash"), str)
            else None
        ),
    )

    declared_experiment = experiment or (
        str(expected_scenario.parameters.get("experiment"))
        if expected_scenario is not None
        else None
    )
    if expected_scenario is not None:
        if expected_scenario.track != "fit":
            raise ValueError("expected scenario is not Track A")
        if declared_experiment != expected_scenario.parameters.get("experiment"):
            raise ValueError("fit experiment differs from the scenario")
    if declared_experiment not in {None, "coverage", "scan", "loco"}:
        raise ValueError("released fit parsing requires coverage, scan, or loco")
    loco_enabled = bool(summary.get("scan", {}).get("loco_enabled", False))
    config_loco = bool(
        resolved_config.get("scan", {}).get("loco", {}).get("enabled", False)
    )
    if loco_enabled != config_loco:
        raise ValueError("fit summary and config LOCO state differ")
    if declared_experiment == "loco" and not loco_enabled:
        raise ValueError("released LOCO output is not a production LOCO scan")
    if declared_experiment == "scan" and loco_enabled:
        raise ValueError("primary scan output must not be LOCO")

    reml = summary["reml"]
    if not isinstance(reml.get("pve"), Mapping):
        raise ValueError("fit summary REML result does not contain PVE")
    estimated, nonfinite = _finite_pve(reml["pve"])
    true = _true_pve(truth)
    if true is not None and set(true) != set(estimated):
        raise ValueError("true and estimated PVE components differ")
    bias = (
        {
            name: None if estimated[name] is None else float(estimated[name] - true[name])
            for name in estimated
        }
        if true is not None
        else None
    )
    if type(reml.get("optimizer_status")) is not bool:
        raise ValueError("fit optimizer status evidence must be boolean")
    optimizer_status = reml["optimizer_status"]
    acceptance = summary.get("acceptance")
    if type(summary.get("acceptance_all_passed")) is not bool or not isinstance(
        acceptance, list
    ) or any(
        not isinstance(item, Mapping)
        or not isinstance(item.get("check"), str)
        or not isinstance(item.get("passed"), bool)
        for item in acceptance
    ):
        raise ValueError("fit acceptance evidence is invalid")
    failed_acceptance = [
        str(item["check"]) for item in acceptance if not item["passed"]
    ]
    acceptance_ok = summary["acceptance_all_passed"] and not (
        failed_acceptance
    )
    if not acceptance_ok:
        failure_type = "AcceptanceFailure"
        failure_message = (
            f"failed acceptance checks: {failed_acceptance}"
            if failed_acceptance
            else "acceptance_all_passed is false"
        )
    elif nonfinite:
        failure_type = "NonFinitePVE"
        failure_message = f"non-finite PVE components: {nonfinite}"
    elif not optimizer_status:
        failure_type = "OptimizerFailure"
        failure_message = str(
            reml.get("optimizer_message", "optimizer did not converge")
        )
    else:
        failure_type = None
        failure_message = None
    metrics: dict[str, Any] = {
        "true_pve": true,
        "estimated_pve": estimated,
        "pve_bias": bias,
        "estimated_sigma2": {
            str(name): float(value)
            for name, value in dict(reml.get("sigma2") or {}).items()
        },
        "boundary_components": [
            str(value) for value in reml.get("boundary_components", [])
        ],
        "optimizer_status": optimizer_status,
        "optimizer_message": str(reml.get("optimizer_message", "")),
        "converged": optimizer_status,
        "n_starts": int(reml.get("n_starts", 0)),
        "runtime_seconds": float(summary.get("runtime_sec", 0.0)),
        "acceptance_all_passed": acceptance_ok,
        "acceptance": list(acceptance),
        "failed_acceptance_checks": failed_acceptance,
        "nonfinite_pve_components": nonfinite,
        "analysis_samples": analysis_samples,
        "summary_path": str(summary_path),
        "summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
        "released_input_binding": released_input_binding,
        "released_source_manifest": _released_source_manifest(
            out_dir, trait, declared_experiment or "fit"
        ),
        "failure": {
            "failed": failure_type is not None,
            "error_type": failure_type,
            "message": failure_message,
        },
    }

    uncertainty = reml.get("pve_uncertainty")
    bootstrap_path = output / f"pve_bootstrap_{trait}.tsv"
    if uncertainty is not None:
        if not isinstance(uncertainty, Mapping):
            raise ValueError("reml.pve_uncertainty must be a mapping")
        reported_bootstrap = outputs.get("pve_bootstrap_samples")
        if (
            not bootstrap_path.is_file()
            or Path(str(reported_bootstrap or "")).resolve() != bootstrap_path
        ):
            raise ValueError("bootstrap TSV provenance is missing or invalid")
        with bootstrap_path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            rows = list(reader)
        components = uncertainty.get("components")
        component_names = list(estimated)
        if not isinstance(components, Mapping) or set(components) != set(component_names):
            raise ValueError("bootstrap component intervals do not match REML PVE")
        expected_columns = ["replicate", *component_names] + [
            f"boundary_{name}" for name in component_names
        ]
        if reader.fieldnames != expected_columns:
            raise ValueError("bootstrap TSV schema is invalid")
        try:
            requested = int(uncertainty["B_requested"])
            succeeded = int(uncertainty["B_success"])
            failed = int(uncertainty["B_failed"])
            level = float(uncertainty["level"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("bootstrap count/level metadata is invalid") from error
        config_bootstrap = resolved_config.get("reml", {}).get("pve_bootstrap")
        if not isinstance(config_bootstrap, Mapping) or not config_bootstrap.get(
            "enabled", True
        ):
            raise ValueError("bootstrap was not enabled in the resolved config")
        if (
            requested < 1
            or succeeded < 0
            or failed < 0
            or succeeded + failed != requested
            or int(config_bootstrap.get("B", -1)) != requested
            or not np.isclose(float(config_bootstrap.get("level", -1)), level)
            or len(rows) != succeeded
        ):
            raise ValueError("bootstrap requested/success/failure counts are inconsistent")
        if expected_scenario is not None:
            if expected_scenario.stage == "formal" and requested != 200:
                raise ValueError("bootstrap formal coverage requires B=200")
            if (
                expected_scenario.stage == "pilot"
                and expected_scenario.bootstrap_B > 0
                and requested != expected_scenario.bootstrap_B
            ):
                raise ValueError("bootstrap pilot B differs from the scenario")
        replicate_ids: list[int] = []
        sample_values: dict[str, list[float]] = {
            name: [] for name in component_names
        }
        for row in rows:
            try:
                replicate_id = int(row["replicate"])
            except (TypeError, ValueError) as error:
                raise ValueError("bootstrap replicate IDs are invalid") from error
            replicate_ids.append(replicate_id)
            for name in component_names:
                try:
                    value = float(row[name])
                except (TypeError, ValueError) as error:
                    raise ValueError("bootstrap PVE value is invalid") from error
                if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                    raise ValueError("bootstrap PVE values must be finite in [0,1]")
                sample_values[name].append(value)
                if row[f"boundary_{name}"] not in {"True", "False"}:
                    raise ValueError("bootstrap boundary values must be boolean")
        if replicate_ids != list(range(succeeded)):
            raise ValueError("bootstrap replicate IDs must be unique and consecutive")
        intervals: dict[str, dict[str, float]] = {}
        for name in component_names:
            item = components[name]
            if not isinstance(item, Mapping):
                raise ValueError("bootstrap component interval is invalid")
            try:
                estimate = float(item["estimate"])
                low = float(item["ci_low"])
                high = float(item["ci_high"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("bootstrap component interval is invalid") from error
            if not all(math.isfinite(value) for value in (estimate, low, high)) or not (
                0.0 <= estimate <= 1.0 and 0.0 <= low <= high <= 1.0
            ):
                raise ValueError("bootstrap interval bounds are invalid")
            if estimated[name] is None or not np.isclose(estimate, estimated[name]):
                raise ValueError("bootstrap interval estimate differs from REML PVE")
            intervals[name] = {
                "estimate": estimate,
                "ci_low": low,
                "ci_high": high,
            }
        genetic_names = [name for name in component_names if name != "e"]
        total_samples = np.sum(
            np.asarray([sample_values[name] for name in genetic_names], dtype=float),
            axis=0,
        )
        alpha = 1.0 - level
        total_estimate = float(
            sum(float(estimated[name]) for name in genetic_names)
        )
        total_low = float(np.quantile(total_samples, alpha / 2.0))
        total_high = float(np.quantile(total_samples, 1.0 - alpha / 2.0))
        if not (
            0.0 <= total_estimate <= 1.0
            and 0.0 <= total_low <= total_high <= 1.0
        ):
            raise ValueError("bootstrap total-genetic interval is invalid")
        intervals["total_genetic"] = {
            "estimate": total_estimate,
            "ci_low": total_low,
            "ci_high": total_high,
        }
        coverage = (
            {
                name: bool(
                    intervals[name]["ci_low"] <= true[name] <= intervals[name]["ci_high"]
                )
                for name in component_names
            }
            if true is not None
            else None
        )
        if coverage is not None:
            target_total = float(sum(true[name] for name in genetic_names))
            coverage["total_genetic"] = bool(
                total_low <= target_total <= total_high
            )
        metrics["pve_bootstrap"] = {
            "method": uncertainty.get("method"),
            "level": level,
            "B_requested": requested,
            "B_success": succeeded,
            "B_failed": failed,
            "intervals": intervals,
            "coverage": coverage,
            "rows": len(rows),
            "columns": list(rows[0]) if rows else [],
            "path": str(bootstrap_path),
            "sha256": hashlib.sha256(bootstrap_path.read_bytes()).hexdigest(),
        }
    elif declared_experiment == "coverage":
        raise ValueError("coverage released output is missing bootstrap evidence")
    elif bootstrap_path.exists():
        raise ValueError("bootstrap TSV exists without summary PVE uncertainty")
    else:
        metrics["pve_bootstrap"] = None
    return metrics


def _sample_manifest(
    sample_ids: Sequence[str] | np.ndarray | None,
    n: int,
) -> dict[str, Any]:
    if sample_ids is None:
        ids = [f"sample_{index:06d}" for index in range(n)]
    else:
        raw = np.asarray(sample_ids)
        if raw.ndim != 1 or raw.size != n or any(
            not isinstance(value, (str, np.str_)) for value in raw.tolist()
        ):
            raise ValueError("sample_ids must be an aligned string vector")
        ids = raw.astype(str, copy=False).tolist()
    if len(set(ids)) != len(ids):
        raise ValueError("sample_ids must be unique")
    return {
        "n": n,
        "ordered_sample_ids": ids,
        "ordered_sample_ids_sha256": sha256_payload(ids),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, Path):
        return str(value)
    return value


def _truth_manifest(truth: Mapping[str, Any]) -> dict[str, Any]:
    """Return and validate the complete serialized input to ``truth_hash``."""

    if not isinstance(truth, Mapping):
        raise ValueError("truth must be a mapping")
    declared = truth.get("truth_hash")
    manifest = _json_safe(
        {name: value for name, value in truth.items() if name != "truth_hash"}
    )
    if not isinstance(declared, str) or sha256_payload(manifest) != declared:
        raise ValueError("truth hash differs from its serialized manifest")
    return manifest


def _released_source_manifest(
    out_dir: str | Path,
    trait: str,
    experiment: str,
) -> dict[str, Any]:
    """Fingerprint every released artifact that may affect a resumed result."""

    root, output = _fit_layout(out_dir)
    paths = {
        "summary": output / f"summary_{trait}.json",
        "generated_config": root / "configs" / "fit.generated.yaml",
        "resolved_config": output / "resolved_config.yaml",
        "analysis_samples": output / "analysis_samples.tsv",
    }
    if experiment == "coverage":
        paths["bootstrap"] = output / f"pve_bootstrap_{trait}.tsv"
    if experiment in {"scan", "loco"}:
        paths["sumstats"] = output / f"sumstats_{trait}.tsv"
        paths["lambda_gc"] = output / f"lambda_gc_{trait}.tsv"
    discovery_error = None
    try:
        context = _released_config_context(root)
        phenotype_path = Path(
            str(context["resolved"].get("phenotype", {}).get("path", ""))
        ).resolve()
        paths["phenotype"] = phenotype_path
        paths.update(context["bed_sources"])
    except Exception as error:  # missing evidence remains bound into the request
        discovery_error = f"{type(error).__name__}: {error}"
    files: dict[str, dict[str, Any]] = {}
    for name, path in paths.items():
        exists = path.is_file()
        files[name] = {
            "path": str(path),
            "exists": exists,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest() if exists else None,
        }
    manifest = {
        "replicate_root": str(root),
        "output_dir": str(output),
        "experiment": experiment,
        "files": files,
        "config_discovery_error": discovery_error,
        "kernel_provenance": {
            "matrix_hash_available": False,
            "evidence_boundary": (
                "ordered samples plus resolved QC/config and all scan/GRM "
                "BED/BIM/FAM source bytes"
            ),
        },
    }
    manifest["sha256"] = sha256_payload(manifest)
    return manifest


def _parse_released_scan_evidence(
    out_dir: str | Path,
    trait: str,
    subgenomes: Sequence[str],
) -> dict[str, Any]:
    _, output = _fit_layout(out_dir)
    summary_path = output / f"summary_{trait}.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    outputs = summary.get("outputs")
    scan_summary = summary.get("scan")
    if not isinstance(outputs, Mapping) or not isinstance(scan_summary, Mapping):
        raise ValueError("released scan summary evidence is missing")
    sumstats_path = output / f"sumstats_{trait}.tsv"
    lambda_path = output / f"lambda_gc_{trait}.tsv"
    reported_sumstats = outputs.get("sumstats")
    if (
        not isinstance(reported_sumstats, list)
        or len(reported_sumstats) != 1
        or Path(str(reported_sumstats[0])).resolve() != sumstats_path
        or Path(str(outputs.get("lambda_gc_tsv", ""))).resolve() != lambda_path
        or not sumstats_path.is_file()
        or not lambda_path.is_file()
    ):
        raise ValueError("released scan output paths are missing or stale")
    with sumstats_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"snp_id", "subgenome", "chrom", "pos", "p"}
        if not required <= set(reader.fieldnames or []):
            raise ValueError("released scan sumstats schema is invalid")
        rows = list(reader)
    if not rows or int(scan_summary.get("n_markers_kept", -1)) != len(rows):
        raise ValueError("released scan marker count is empty or inconsistent")
    pvalues: dict[str, list[float]] = {name: [] for name in subgenomes}
    variant_ids: dict[str, list[str]] = {name: [] for name in subgenomes}
    variant_positions_bp: dict[str, list[int]] = {
        name: [] for name in subgenomes
    }
    seen: set[str] = set()
    for row in rows:
        name = row["subgenome"]
        marker = row["snp_id"]
        if name not in pvalues or not marker or marker in seen:
            raise ValueError("released scan variant family is invalid")
        try:
            pvalue = float(row["p"])
            position = int(row["pos"])
        except (TypeError, ValueError) as error:
            raise ValueError("released scan sumstats values are invalid") from error
        if position < 0 or math.isinf(pvalue) or (
            math.isfinite(pvalue) and not 0.0 <= pvalue <= 1.0
        ):
            raise ValueError("released scan p-values must be in [0,1] or NaN")
        seen.add(marker)
        variant_ids[name].append(marker)
        variant_positions_bp[name].append(position)
        pvalues[name].append(pvalue)
    fwer = experiment_wide_scan_fwer(
        pvalues,
        variant_ids=variant_ids,
        variant_positions_bp=variant_positions_bp,
    )
    with lambda_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != ["scope", "level", "n_markers", "lambda_gc"]:
            raise ValueError("released scan lambda-GC schema is invalid")
        lambda_rows = list(reader)
    if not lambda_rows:
        raise ValueError("released scan lambda-GC evidence is empty")
    lambda_gc: list[dict[str, Any]] = []
    for row in lambda_rows:
        try:
            value = float(row["lambda_gc"])
            n_markers = int(row["n_markers"])
        except (TypeError, ValueError) as error:
            raise ValueError("released scan lambda-GC value is invalid") from error
        if not math.isfinite(value) or value < 0.0 or n_markers < 0:
            raise ValueError("released scan lambda-GC value must be finite")
        lambda_gc.append({**row, "n_markers": n_markers, "lambda_gc": value})
    all_rows = [row for row in lambda_gc if row["scope"] == "all"]
    if len(all_rows) != 1 or all_rows[0]["n_markers"] != len(rows):
        raise ValueError("released scan all-marker lambda-GC count is inconsistent")
    return {
        "fwer": fwer,
        "lambda_gc": lambda_gc,
        "sumstats_path": str(sumstats_path),
        "sumstats_sha256": hashlib.sha256(sumstats_path.read_bytes()).hexdigest(),
        "lambda_gc_path": str(lambda_path),
        "lambda_gc_sha256": hashlib.sha256(lambda_path.read_bytes()).hexdigest(),
    }


def run_scan_replicate(
    comparators: Mapping[str, Any] | None = None,
    *,
    released_output_dir: str | Path | None = None,
    trait: str = "simulated_trait",
    expected_subgenomes: Sequence[str] | None = None,
    expected_sample_ids: Sequence[str] | None = None,
    expected_kernels: Mapping[str, np.ndarray] | None = None,
    expected_scenario: Scenario | None = None,
    loco: bool = False,
    seed: int = 2026,
) -> dict[str, Any]:
    """Run matched primary scans, or strictly parse a released LOCO arm."""

    if loco:
        if comparators is not None or released_output_dir is None:
            raise ValueError("LOCO requires a strictly bound released production output")
        metrics = parse_fit_metrics(
            released_output_dir,
            trait,
            expected_subgenomes=expected_subgenomes,
            expected_sample_ids=expected_sample_ids,
            expected_kernels=expected_kernels,
            expected_scenario=expected_scenario,
            experiment="loco",
        )
        evidence = _parse_released_scan_evidence(
            released_output_dir,
            trait,
            list(expected_subgenomes or []),
        )
        metrics["scan_arm"] = "loco_sensitivity"
        metrics["separate_sensitivity_arm"] = True
        metrics["comparators"] = {"canonical_multi_kernel": evidence}
        if evidence["fwer"]["failure"]["failed"]:
            metrics["failure"] = dict(evidence["fwer"]["failure"])
        return metrics
    if released_output_dir is not None:
        raise ValueError("primary scan requires matched in-memory comparator inputs")
    if comparators is None:
        raise ValueError("primary scan comparator inputs are required")
    required = {
        "context_fingerprint",
        "context",
        "sample_ids",
        "phenotype",
        "covariates",
        "standardized_variants",
        "canonical_multi_kernel",
        "pooled_trace_sum",
        "independent_subgenome",
    }
    if not required <= set(comparators):
        raise ValueError("primary scan comparator context is incomplete")
    context_record = comparators["context"]
    if not isinstance(context_record, Mapping):
        raise ValueError("primary scan context record is invalid")
    sample_ids = np.asarray(comparators["sample_ids"])
    y = np.asarray(comparators["phenotype"], dtype=float)
    X = np.asarray(comparators["covariates"], dtype=float)
    variants = comparators["standardized_variants"]
    variant_ids = context_record.get("variant_ids")
    variant_positions_bp = context_record.get("variant_positions_bp")
    if (
        not isinstance(variants, Mapping)
        or not isinstance(variant_ids, Mapping)
        or not isinstance(variant_positions_bp, Mapping)
        or context_record.get("distance_unit") != "bp"
        or type(context_record.get("bp_positions_explicit")) is not bool
    ):
        raise ValueError("primary scan variants are invalid")
    if expected_subgenomes is not None and list(variant_ids) != list(
        expected_subgenomes
    ):
        raise ValueError("primary scan subgenome order differs from the request")
    if expected_sample_ids is not None and sample_ids.tolist() != list(
        expected_sample_ids
    ):
        raise ValueError("primary scan sample order differs from the request")
    if expected_scenario is not None and expected_scenario.parameters.get(
        "experiment"
    ) != "scan":
        raise ValueError("primary scan scenario does not declare scan")
    rebuilt = build_scan_comparators(
        comparators["canonical_multi_kernel"],
        sample_ids=sample_ids,
        variant_ids=variant_ids,
        variant_positions_bp=(
            variant_positions_bp
            if context_record["bp_positions_explicit"]
            else None
        ),
        standardized_variants=variants,
        phenotype=y,
        covariates=X,
        qc_declaration=context_record.get("qc_declaration"),
        truth_metadata=context_record.get("truth"),
    )
    if rebuilt["context_fingerprint"] != comparators["context_fingerprint"]:
        raise ValueError("primary scan comparator context fingerprint differs")
    if comparators.get("truth") != rebuilt["truth"]:
        raise ValueError("primary scan truth metadata/hash differs from context")
    if _array_mapping_fingerprint(
        comparators["pooled_trace_sum"]
    ) != _array_mapping_fingerprint(rebuilt["pooled_trace_sum"]):
        raise ValueError("primary pooled kernel differs from canonical derivation")
    if _nested_array_mapping_fingerprint(
        comparators["independent_subgenome"]
    ) != _nested_array_mapping_fingerprint(rebuilt["independent_subgenome"]):
        raise ValueError("primary independent kernels differ from canonical derivation")
    if expected_scenario is not None:
        truth = rebuilt["truth"]
        if truth["placement"] != expected_scenario.parameters.get("placement"):
            raise ValueError("primary scan truth placement differs from scenario")
        if not np.isclose(
            truth["scan_pve"],
            float(expected_scenario.parameters.get("scan_pve", float("nan"))),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("primary scan truth scan_pve differs from scenario")
        if (
            expected_scenario.stage == "pilot"
            and abs(truth["realized_signal_pve"] - truth["scan_pve"]) > 0.01
        ):
            raise ValueError("pilot realized signal PVE differs by more than 0.01")

    from homoeogwas.io import GenoChunk
    from homoeogwas.lmm import fit_multi_reml
    from homoeogwas.scan import build_scan_context, scan_snps

    def scan_with_fit(
        kernel_set: Mapping[str, np.ndarray],
        blocks: Sequence[str],
        method_seed: int,
    ) -> tuple[dict[str, list[float]], dict[str, Any]]:
        fitted = fit_multi_reml(
            y,
            X,
            dict(kernel_set),
            n_starts=10,
            random_state=int(method_seed % (2**32)),
        )
        scan_context = build_scan_context(
            y,
            X,
            dict(kernel_set),
            dict(fitted.sigma2),
            sample_ids=sample_ids,
        )
        pvalues: dict[str, list[float]] = {}
        for name in blocks:
            ids = list(variant_ids[name])
            geno = GenoChunk(
                samples=np.asarray(sample_ids, dtype=object),
                variant_ids=np.asarray(ids, dtype=object),
                chrom=np.asarray([name] * len(ids), dtype=object),
                pos=np.asarray(variant_positions_bp[name], dtype=np.int64),
                dosage=np.asarray(variants[name], dtype=np.float32),
            )
            scanned = scan_snps(
                scan_context,
                geno,
                backend="cpu",
                maf_min=float("-inf"),
                call_rate_min=0.0,
            )
            by_id = {
                str(marker): float(pvalue)
                for marker, pvalue in zip(scanned.snp_id, scanned.p, strict=True)
            }
            pvalues[name] = [by_id.get(marker, float("nan")) for marker in ids]
        fit_record = {
            "estimated_pve": {
                str(name): float(value) for name, value in fitted.pve.items()
            },
            "estimated_sigma2": {
                str(name): float(value) for name, value in fitted.sigma2.items()
            },
            "optimizer_status": bool(fitted.optimizer_status),
            "optimizer_message": str(fitted.optimizer_message),
            "n_starts": int(fitted.n_starts),
        }
        return pvalues, fit_record

    names = list(variant_ids)
    started = time.perf_counter()
    canonical_p, canonical_fit = scan_with_fit(
        comparators["canonical_multi_kernel"], names, seed
    )
    pooled_p, pooled_fit = scan_with_fit(
        comparators["pooled_trace_sum"], names, seed + 1
    )
    independent_p: dict[str, list[float]] = {}
    independent_fits: dict[str, Any] = {}
    for offset, name in enumerate(names, start=2):
        pvalues, fit_record = scan_with_fit(
            comparators["independent_subgenome"][name], [name], seed + offset
        )
        independent_p[name] = pvalues[name]
        independent_fits[name] = fit_record

    method_records = {
        "canonical_multi_kernel": {
            "fit": canonical_fit,
            "fwer": experiment_wide_scan_fwer(
                canonical_p,
                variant_ids=variant_ids,
                variant_positions_bp=variant_positions_bp,
            ),
        },
        "pooled_trace_sum": {
            "fit": pooled_fit,
            "fwer": experiment_wide_scan_fwer(
                pooled_p,
                variant_ids=variant_ids,
                variant_positions_bp=variant_positions_bp,
            ),
        },
        "independent_subgenome": {
            "fit_by_subgenome": independent_fits,
            "fwer": experiment_wide_scan_fwer(
                independent_p,
                variant_ids=variant_ids,
                variant_positions_bp=variant_positions_bp,
            ),
            "one_experiment_wide_family": True,
        },
    }
    failed = any(
        record["fwer"]["failure"]["failed"]
        for record in method_records.values()
    ) or any(
        not fit["optimizer_status"]
        for fit in [canonical_fit, pooled_fit, *independent_fits.values()]
    )
    return {
        "scan_arm": "primary",
        "separate_sensitivity_arm": False,
        "context_fingerprint": comparators["context_fingerprint"],
        "comparators": method_records,
        "runtime_seconds": float(time.perf_counter() - started),
        "failure": {
            "failed": failed,
            "error_type": "ScanOrOptimizerFailure" if failed else None,
            "message": "non-finite scan results or optimizer failure" if failed else None,
        },
    }


def _preflight_request_evidence(
    path: str | Path | None,
    expected_hash: str | None,
) -> dict[str, Any] | None:
    if path is None and expected_hash is None:
        return None
    source = Path(path).resolve() if path is not None else None
    evidence: dict[str, Any] = {
        "path": str(source) if source is not None else None,
        "expected_hash": expected_hash,
        "exists": bool(source is not None and source.is_file()),
        "sha256": None,
        "binaries": [],
        "read_error": None,
    }
    if source is None or not source.is_file():
        return evidence
    try:
        evidence["sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
        with source.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        for row in rows:
            raw_binary = row.get("executable_path", "")
            binary = Path(raw_binary) if raw_binary else None
            exists = bool(binary is not None and binary.is_file())
            evidence["binaries"].append(
                {
                    "comparator": row.get("comparator"),
                    "path": str(binary) if binary is not None else None,
                    "exists": exists,
                    "sha256": (
                        hashlib.sha256(binary.read_bytes()).hexdigest()
                        if exists and binary is not None
                        else None
                    ),
                }
            )
    except Exception as error:
        evidence["read_error"] = f"{type(error).__name__}: {error}"
    return evidence


def run_fit_replicate(
    scenario: Scenario,
    kernels: Mapping[str, np.ndarray],
    *,
    replicate: int,
    design_hash: str,
    sample_ids: Sequence[str] | np.ndarray | None = None,
    shard_path: str | Path | None = None,
    fit_output_dir: str | Path | None = None,
    trait: str = "simulated_trait",
    comparator_preflight_path: str | Path | None = None,
    comparator_preflight_hash: str | None = None,
    scan_comparators: Mapping[str, Any] | None = None,
    released_scan_truth: Mapping[str, Any] | None = None,
    loco_design_root: str | Path | None = None,
) -> dict[str, Any]:
    """Run or parse one Track A replicate and write one immutable shard."""

    if scenario.track != "fit":
        raise ValueError("run_fit_replicate requires a fit scenario")
    if replicate < 0 or replicate >= scenario.replicates:
        raise ValueError("replicate is outside the scenario registry range")
    checked, n = _checked_kernels(kernels)
    samples = _sample_manifest(sample_ids, n)
    fingerprints = {
        name: _kernel_fingerprint(value) for name, value in checked.items()
    }
    fit_context_manifest = {
        "kernel_order": list(checked),
        "kernel_fingerprints": fingerprints,
        "sample_manifest": samples,
    }
    context_fingerprint = sha256_payload(fit_context_manifest)
    seed = derive_seed(
        design_hash, "fit", scenario.scenario_id, replicate, scenario.stage
    )
    preflight = None
    preflight_evidence = _preflight_request_evidence(
        comparator_preflight_path, comparator_preflight_hash
    )
    experiment = scenario.parameters.get("experiment")
    if not isinstance(experiment, str):
        experiment = "invalid"
    if experiment == "loco" and released_scan_truth is not None:
        raise ValueError(
            "LOCO truth must be loaded from the presealed design-lock artifact"
        )
    coverage_y: np.ndarray | None = None
    coverage_truth: dict[str, Any] | None = None
    coverage_request_binding: dict[str, Any] | None = None
    coverage_setup_error: str | None = None
    recovery_y: np.ndarray | None = None
    recovery_truth: dict[str, Any] | None = None
    recovery_setup_error: str | None = None
    if experiment in {"recovery", "coverage"}:
        try:
            if "allocation" not in scenario.parameters:
                raise ValueError(
                    f"{experiment} scenario is missing its allocation"
                )
            simulated_y, simulated_truth = simulate_fit_truth(
                checked,
                scenario.parameters["allocation"],
                seed=seed,
                total_pve=scenario.parameters.get("total_pve"),
                dominant_subgenomes=scenario.parameters.get("dominant_subgenomes"),
                null_subgenome=scenario.parameters.get("null_subgenome"),
            )
            if experiment == "coverage":
                coverage_y, coverage_truth = simulated_y, simulated_truth
                coverage_request_binding = {
                    "sample_ids_sha256": samples["ordered_sample_ids_sha256"],
                    "phenotype": _exact_array_fingerprint(coverage_y),
                    "truth_hash": coverage_truth["truth_hash"],
                    "kernel_order": list(checked),
                    "kernel_fingerprints": fingerprints,
                }
                coverage_request_binding["sha256"] = sha256_payload(
                    coverage_request_binding
                )
            else:
                recovery_y, recovery_truth = simulated_y, simulated_truth
        except Exception as error:
            setup_error = f"{type(error).__name__}: {error}"
            if experiment == "coverage":
                coverage_setup_error = setup_error
            else:
                recovery_setup_error = setup_error
    released_source = (
        _released_source_manifest(fit_output_dir, trait, experiment)
        if fit_output_dir is not None
        else None
    )
    released_request_binding: dict[str, Any] | None = None
    released_binding_error: str | None = None
    if fit_output_dir is not None:
        try:
            released_request_binding = _released_fit_inputs(
                _released_config_context(fit_output_dir),
                samples["ordered_sample_ids"],
                expected_phenotype=coverage_y if experiment == "coverage" else None,
                expected_kernels=checked,
                expected_truth_hash=(
                    coverage_truth["truth_hash"]
                    if coverage_truth is not None
                    else None
                ),
            )
        except Exception as error:
            released_binding_error = f"{type(error).__name__}: {error}"
    released_scan_truth_binding: dict[str, Any] | None = None
    released_scan_truth_error: str | None = None
    released_loco_evidence: dict[str, Any] | None = None
    if experiment == "loco" and loco_design_root is not None:
        try:
            design_root = Path(loco_design_root).resolve()
            lock_path = design_root / "design_lock.json"
            if lock_path.is_symlink():
                raise ValueError("LOCO design lock must not be a symlink")
            lock = json.loads(lock_path.read_text(encoding="utf-8"))
            seed_design = lock.get("seed_design")
            seed_design_hash = lock.get("seed_design_hash")
            if (
                not isinstance(seed_design, Mapping)
                or not isinstance(seed_design_hash, str)
                or sha256_payload(seed_design) != seed_design_hash
            ):
                raise ValueError("LOCO seed design identity is invalid")
            scenario_records = lock.get("loco_truth_artifacts", {}).get(
                scenario.scenario_id
            )
            required = {
                "path", "sha256", "truth_hash", "source", "seed",
                "generated_config_sha256", "phenotype_sha256",
            }
            if isinstance(scenario_records, Mapping) and set(scenario_records) == required:
                raise ValueError(
                    "legacy single-artifact LOCO lock is unsupported; regenerate "
                    "the design with one presealed artifact per replicate"
                )
            record = (
                scenario_records.get(str(replicate))
                if isinstance(scenario_records, Mapping) else None
            )
            if (
                lock.get("design_hash") != design_hash
                or not isinstance(record, Mapping) or set(record) != required
            ):
                raise ValueError("LOCO truth artifact is absent from the design lock")
            expected_seed = derive_seed(
                seed_design_hash, "fit", scenario.scenario_id, replicate,
                f"{scenario.stage}:loco_truth",
            )
            if record.get("seed") != expected_seed:
                raise ValueError("LOCO truth artifact derived seed differs from design")
            truth_path = (design_root / str(record["path"])).resolve()
            if (
                design_root not in truth_path.parents
                or truth_path.is_symlink() or not truth_path.is_file()
                or hashlib.sha256(truth_path.read_bytes()).hexdigest() != record["sha256"]
            ):
                raise ValueError("LOCO truth artifact path/hash differs from design lock")
            loaded_truth = json.loads(truth_path.read_text(encoding="utf-8"))
            manifest = _truth_manifest(loaded_truth)
            if (
                loaded_truth.get("truth_hash") != record["truth_hash"]
                or manifest.get("source") != record["source"]
                or manifest.get("seed") != record["seed"]
            ):
                raise ValueError("LOCO truth artifact provenance differs from design lock")
            released_scan_truth = loaded_truth
        except Exception as error:
            released_scan_truth_error = f"{type(error).__name__}: {error}"
    if experiment == "loco" and released_scan_truth is not None:
        try:
            if fit_output_dir is None or released_source is None or released_request_binding is None:
                raise ValueError("released LOCO truth requires bound production output")
            released_truth_manifest = _truth_manifest(released_scan_truth)
            expected_analysis_context = {
                "analysis_sample_ids_sha256": released_request_binding[
                    "analysis_sample_ids_sha256"
                ],
                "joined_phenotype": released_request_binding["joined_phenotype"],
                "kernel_order": released_request_binding["kernel_order"],
                "kernel_fingerprints": released_request_binding[
                    "kernel_fingerprints"
                ],
                "generated_config_sha256": released_source["files"][
                    "generated_config"
                ]["sha256"],
            }
            if (
                record["generated_config_sha256"]
                != expected_analysis_context["generated_config_sha256"]
                or record["phenotype_sha256"]
                != expected_analysis_context["joined_phenotype"]["sha256"]
            ):
                raise ValueError("LOCO config/phenotype differs from design lock")
            if released_truth_manifest.get("analysis_context") != expected_analysis_context:
                raise ValueError(
                    "released LOCO truth is not bound to the pre-run analysis context"
                )
            released_request_binding = {
                **released_request_binding,
                "truth_hash": released_scan_truth["truth_hash"],
            }
            released_request_binding["sha256"] = sha256_payload(
                {
                    key: value
                    for key, value in released_request_binding.items()
                    if key != "sha256"
                }
            )
            if released_truth_manifest.get("distance_unit") != "bp":
                raise ValueError("released LOCO truth distance unit must be bp")
            if released_truth_manifest.get("scan_pve") != scenario.parameters.get("scan_pve"):
                raise ValueError("released LOCO truth PVE differs from scenario")
            causal = released_truth_manifest.get("causal_variants")
            if not isinstance(causal, list) or any(
                not isinstance(item, Mapping)
                or set(item) != {"variant_id", "subgenome", "position_bp"}
                or not isinstance(item["variant_id"], str)
                or not isinstance(item["subgenome"], str)
                or isinstance(item["position_bp"], bool)
                or not isinstance(item["position_bp"], int)
                or item["position_bp"] < 0
                for item in causal
            ):
                raise ValueError("released LOCO causal variant truth is invalid")
            if (scenario.parameters.get("scan_pve") == 0.0) is not (len(causal) == 0):
                raise ValueError("released LOCO null/causal truth differs from scan PVE")
            released_loco_evidence = _parse_released_scan_evidence(
                fit_output_dir, trait, list(checked)
            )
            ordered = released_loco_evidence["fwer"]["ordered_family"]
            by_id = {item["variant_id"]: item for item in ordered}
            if any(by_id.get(item["variant_id"]) != dict(item) for item in causal):
                raise ValueError("released LOCO causal truth differs from output family")
            released_scan_truth_binding = {
                "truth_hash": released_scan_truth["truth_hash"],
                "truth_artifact_path": str(record["path"]),
                "truth_artifact_sha256": str(record["sha256"]),
                "source": record["source"],
                "seed": record["seed"],
                "generated_config_sha256": record["generated_config_sha256"],
                "phenotype_sha256": record["phenotype_sha256"],
                "released_source_manifest_sha256": released_source["sha256"],
                "released_request_binding_sha256": released_request_binding["sha256"],
                "ordered_family": ordered,
                "ordered_family_hash": released_loco_evidence["fwer"][
                    "ordered_family_hash"
                ],
                "distance_unit": "bp",
            }
            released_scan_truth_binding["sha256"] = sha256_payload(
                released_scan_truth_binding
            )
        except Exception as error:
            released_scan_truth_error = f"{type(error).__name__}: {error}"
    scan_context_hash = None
    scan_request_manifest = None
    if scan_comparators is not None:
        try:
            scan_request_manifest = _scan_request_manifest(scan_comparators)
            scan_context_hash = scan_request_manifest["sha256"]
        except Exception as error:  # retained as an immutable failure request
            scan_context_hash = sha256_payload(
                {
                    "invalid_scan_context": type(error).__name__,
                    "message": str(error),
                    "declared": _json_safe(
                        scan_comparators.get("context_fingerprint")
                    ),
                }
            )
    elif released_scan_truth_binding is not None:
        scan_request_manifest = {
            "source": "released_loco_truth",
            "bp_positions_explicit": True,
            "distance_unit": "bp",
            "ordered_family": released_scan_truth_binding["ordered_family"],
            "ordered_family_hash": released_scan_truth_binding[
                "ordered_family_hash"
            ],
            "released_scan_truth_binding_sha256": (
                released_scan_truth_binding["sha256"]
            ),
        }
        scan_request_manifest["sha256"] = sha256_payload(scan_request_manifest)
        scan_context_hash = scan_request_manifest["sha256"]
    source = (
        "homoeogwas_outputs"
        if fit_output_dir is not None
        else "matched_scan_context"
        if scan_comparators is not None
        else "fit_multi_reml"
    )
    request_truth: Mapping[str, Any] | None = (
        recovery_truth
        if recovery_truth is not None
        else coverage_truth
        if coverage_truth is not None
        else scan_comparators.get("truth")
        if scan_comparators is not None
        and isinstance(scan_comparators.get("truth"), Mapping)
        else released_scan_truth
        if experiment == "loco" and released_scan_truth is not None
        else None
    )
    truth_manifest: dict[str, Any] | None = None
    truth_hash: str | None = None
    truth_manifest_error: str | None = None
    if request_truth is not None:
        try:
            truth_manifest = _truth_manifest(request_truth)
            truth_hash = str(request_truth["truth_hash"])
        except Exception as error:
            truth_manifest_error = f"{type(error).__name__}: {error}"
    request_manifest = {
        "design_hash": design_hash,
        "context_fingerprint": context_fingerprint,
        "scenario": scenario.to_dict(),
        "replicate": replicate,
        "seed": int(seed),
        "source": source,
        "released_source_manifest": released_source,
        "released_request_binding": released_request_binding,
        "released_binding_error": released_binding_error,
        "coverage_request_binding": coverage_request_binding,
        "coverage_setup_error": coverage_setup_error,
        "recovery_setup_error": recovery_setup_error,
        "truth_hash": truth_hash,
        "truth_manifest": truth_manifest,
        "truth_manifest_error": truth_manifest_error,
        "released_scan_truth_binding": released_scan_truth_binding,
        "released_scan_truth_error": released_scan_truth_error,
        "scan_context_fingerprint": scan_context_hash,
        "scan_request_manifest": scan_request_manifest,
        "trait": trait,
        "comparator_preflight": preflight_evidence,
    }
    request_hash = sha256_payload(request_manifest)
    if shard_path is not None and Path(shard_path).exists():
        existing = load_shard(
            Path(shard_path), ShardKey("fit", scenario.scenario_id, replicate)
        )
        if (
            existing["design_hash"] != design_hash
            or existing.get("request_hash") != request_hash
            or existing.get("context_fingerprint") != context_fingerprint
        ):
            raise ShardConflict(f"existing shard differs: {Path(shard_path)}")
        return existing

    base = {
        "track": "fit",
        "scenario_id": scenario.scenario_id,
        "replicate": int(replicate),
        "design_hash": design_hash,
        "stage": scenario.stage,
        "formal": scenario.stage == "formal",
        "qa_only": scenario.stage == "pilot",
        "experiment": experiment,
        "seed": int(seed),
        "sample_manifest": samples,
        "kernel_fingerprints": fingerprints,
        "fit_context_manifest": fit_context_manifest,
        "context_fingerprint": context_fingerprint,
        "request_manifest": request_manifest,
        "request_hash": request_hash,
        "truth_manifest": truth_manifest,
        "truth_hash": truth_hash,
        "result_source": source,
        "released_source_manifest": released_source,
        "released_request_binding": released_request_binding,
        "released_scan_truth_binding": released_scan_truth_binding,
        "coverage_request_binding": coverage_request_binding,
        "scan_context_fingerprint": scan_context_hash,
        "scan_request_manifest": scan_request_manifest,
        "scan_truth": (
            _json_safe(scan_comparators.get("truth"))
            if scan_comparators is not None
            else _json_safe(released_scan_truth)
            if released_scan_truth is not None
            else None
        ),
        "comparator_preflight_hash": comparator_preflight_hash,
        "comparator_preflight_evidence": preflight_evidence,
        "external_comparators": [],
    }
    started = time.perf_counter()
    try:
        if comparator_preflight_path is None and comparator_preflight_hash is not None:
            raise ValueError("comparator_preflight_hash requires its frozen TSV")
        if released_binding_error is not None:
            raise ValueError(released_binding_error)
        if truth_manifest_error is not None:
            raise ValueError(truth_manifest_error)
        if released_scan_truth_error is not None:
            raise ValueError(released_scan_truth_error)
        if comparator_preflight_path is not None:
            if comparator_preflight_hash is None:
                raise ValueError("comparator preflight requires its frozen hash")
            preflight = read_comparator_preflight(
                comparator_preflight_path,
                expected_hash=comparator_preflight_hash,
            )
            base["external_comparators"] = preflight["rows"]
            if scan_comparators is None:
                raise ValueError("comparator preflight requires actual scan inputs")
            scan_record = scan_comparators.get("context")
            if not isinstance(scan_record, Mapping):
                raise ValueError("comparator preflight scan context is missing")
            current_hashes = _preflight_hashes(
                samples["ordered_sample_ids"],
                scan_record.get("qc_declaration"),
                scan_comparators["covariates"],
                scan_record.get("variant_ids"),
                scan_comparators["standardized_variants"],
            )
            preflight_mismatches: list[str] = []
            for row in preflight["rows"]:
                binary_path = row["executable_path"]
                if binary_path:
                    current_binary_hash = (
                        hashlib.sha256(Path(binary_path).read_bytes()).hexdigest()
                        if Path(binary_path).is_file()
                        else None
                    )
                    if current_binary_hash != row["binary_sha256"]:
                        preflight_mismatches.append(
                            f"{row['comparator']} binary_hash differs"
                        )
                for field, value in current_hashes.items():
                    if row[field] != value:
                        preflight_mismatches.append(
                            f"{row['comparator']} {field} differs"
                        )
            if preflight_mismatches:
                raise ValueError(
                    "comparator preflight differs from this run: "
                    + "; ".join(preflight_mismatches)
                )
        if experiment not in {"recovery", "coverage", "scan", "loco"}:
            raise ValueError(f"unknown Track A experiment: {experiment!r}")
        if (
            experiment in {"scan", "loco"}
            and scenario.stage == "formal"
            and (
                scan_request_manifest is None
                or scan_request_manifest.get("bp_positions_explicit") is not True
            )
        ):
            raise ValueError("formal scan requires explicit bp positions")
        if experiment == "recovery":
            if fit_output_dir is not None or scan_comparators is not None:
                raise ValueError("recovery does not accept released or scan inputs")
            if "allocation" not in scenario.parameters:
                raise ValueError("recovery scenario is missing its allocation")
            if recovery_setup_error is not None:
                raise ValueError(recovery_setup_error)
            if recovery_y is None or recovery_truth is None:
                raise ValueError("recovery request simulation binding is missing")
            y, truth = recovery_y, recovery_truth
            fitted = fit_pve_replicate(y, checked, seed)
            true = dict(truth["target_pve"])
            estimated = dict(fitted["estimated_pve"])
            metrics = {
                **fitted,
                "true_pve": true,
                "target_pve": dict(truth["target_pve"]),
                "pve_bias": {
                    name: float(estimated[name] - true[name]) for name in estimated
                },
                "pve_bootstrap": None,
                "failure": {
                    "failed": not fitted["optimizer_status"],
                    "error_type": (
                        None if fitted["optimizer_status"] else "OptimizerFailure"
                    ),
                    "message": (
                        None
                        if fitted["optimizer_status"]
                        else fitted["optimizer_message"]
                    ),
                },
            }
            payload = {**base, "truth": truth, **metrics}
        elif experiment == "coverage":
            if fit_output_dir is None or scan_comparators is not None:
                raise ValueError("coverage requires one released fit output")
            if coverage_setup_error is not None:
                raise ValueError(coverage_setup_error)
            if coverage_y is None or coverage_truth is None:
                raise ValueError("coverage request simulation binding is missing")
            truth = coverage_truth
            metrics = parse_fit_metrics(
                fit_output_dir,
                trait,
                truth=truth,
                expected_subgenomes=list(checked),
                expected_sample_ids=samples["ordered_sample_ids"],
                expected_phenotype=coverage_y,
                expected_kernels=checked,
                expected_scenario=scenario,
                experiment="coverage",
            )
            metrics["target_pve"] = dict(truth["target_pve"])
            payload = {**base, "truth": truth, **metrics}
        else:
            if experiment == "loco" and scan_comparators is not None:
                raise ValueError("LOCO forbids comparator injection")
            metrics = run_scan_replicate(
                scan_comparators,
                released_output_dir=fit_output_dir,
                trait=trait,
                expected_subgenomes=list(checked),
                expected_sample_ids=samples["ordered_sample_ids"],
                expected_kernels=checked,
                expected_scenario=scenario,
                loco=experiment == "loco",
                seed=seed,
            )
            if experiment == "loco" and released_loco_evidence is not None:
                if metrics.get("comparators", {}).get(
                    "canonical_multi_kernel"
                ) != released_loco_evidence:
                    raise RuntimeError("released LOCO evidence changed within request")
            payload = {**base, **metrics}
    except Exception as error:
        payload = {
            **base,
            "failure": {
                "failed": True,
                "error_type": type(error).__name__,
                "message": str(error),
            },
        }
    orchestration_runtime = float(time.perf_counter() - started)
    payload["orchestration_runtime_seconds"] = orchestration_runtime
    payload.setdefault("runtime_seconds", orchestration_runtime)
    result = _json_safe(payload)
    if shard_path is not None:
        write_shard_exclusive(Path(shard_path), result)
    return result


def _numeric_fingerprint(value: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(value, dtype="<f8")
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return {
        "shape": [int(axis) for axis in array.shape],
        "dtype": array.dtype.str,
        "sha256": digest.hexdigest(),
    }


def _exact_array_fingerprint(value: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return {
        "shape": [int(axis) for axis in array.shape],
        "dtype": array.dtype.str,
        "sha256": digest.hexdigest(),
    }


def _array_mapping_fingerprint(
    values: Mapping[str, np.ndarray],
) -> list[dict[str, Any]]:
    if not isinstance(values, Mapping) or not values:
        raise ValueError("scan array mapping is missing")
    if any(not isinstance(name, str) or not name for name in values):
        raise ValueError("scan array mapping names are invalid")
    return [
        {"name": name, **_exact_array_fingerprint(np.asarray(value))}
        for name, value in values.items()
    ]


def _nested_array_mapping_fingerprint(
    values: Mapping[str, Mapping[str, np.ndarray]],
) -> list[dict[str, Any]]:
    if not isinstance(values, Mapping) or not values:
        raise ValueError("nested scan array mapping is missing")
    if any(not isinstance(name, str) or not name for name in values):
        raise ValueError("nested scan array mapping names are invalid")
    return [
        {"name": name, "values": _array_mapping_fingerprint(inner)}
        for name, inner in values.items()
    ]


def _scan_request_manifest(comparators: Mapping[str, Any]) -> dict[str, Any]:
    context = comparators.get("context")
    if not isinstance(context, Mapping):
        raise ValueError("scan request context is missing")
    sample_ids = np.asarray(comparators["sample_ids"])
    if sample_ids.ndim != 1 or any(
        not isinstance(value, (str, np.str_)) for value in sample_ids.tolist()
    ):
        raise ValueError("scan request sample IDs are invalid")
    manifest = {
        "sample_ids": sample_ids.astype(str).tolist(),
        "variant_ids": _json_safe(context.get("variant_ids")),
        "variant_positions_bp": _json_safe(
            context.get("variant_positions_bp")
        ),
        "distance_unit": context.get("distance_unit"),
        "bp_positions_explicit": context.get("bp_positions_explicit"),
        "qc_declaration": _json_safe(context.get("qc_declaration")),
        "truth": _json_safe(context.get("truth")),
        "top_level_truth": _json_safe(comparators.get("truth")),
        "declared_context_fingerprint": comparators.get("context_fingerprint"),
        "phenotype": _exact_array_fingerprint(comparators["phenotype"]),
        "covariates": _exact_array_fingerprint(comparators["covariates"]),
        "standardized_variants": _array_mapping_fingerprint(
            comparators["standardized_variants"]
        ),
        "canonical_multi_kernel": _array_mapping_fingerprint(
            comparators["canonical_multi_kernel"]
        ),
        "pooled_trace_sum": _array_mapping_fingerprint(
            comparators["pooled_trace_sum"]
        ),
        "independent_subgenome": _nested_array_mapping_fingerprint(
            comparators["independent_subgenome"]
        ),
    }
    manifest["sha256"] = sha256_payload(manifest)
    return manifest


def _scan_truth_record(
    truth_metadata: Mapping[str, Any],
    variant_ids: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    if not isinstance(truth_metadata, Mapping):
        raise ValueError("scan truth_metadata must be a mapping")
    placement = truth_metadata.get("placement")
    if placement not in {"one_subgenome", "two_subgenomes"}:
        raise ValueError("scan truth placement is invalid")
    try:
        scan_pve = float(truth_metadata["scan_pve"])
        realized = float(truth_metadata["realized_signal_pve"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("scan truth PVE metadata is invalid") from error
    if not all(math.isfinite(value) and 0.0 <= value < 1.0 for value in (scan_pve, realized)):
        raise ValueError("scan truth PVE metadata must be finite in [0,1)")
    causal = truth_metadata.get("causal_variants")
    if not isinstance(causal, list):
        raise ValueError("scan truth causal_variants must be a list")
    normalized_causal: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in causal:
        if not isinstance(item, Mapping):
            raise ValueError("scan truth causal variant metadata is invalid")
        marker = item.get("variant_id")
        subgenome = item.get("subgenome")
        if (
            not isinstance(marker, str)
            or not marker
            or marker in seen
            or subgenome not in variant_ids
            or marker not in variant_ids[subgenome]
        ):
            raise ValueError("scan truth causal variant is absent or duplicated")
        seen.add(marker)
        normalized_causal.append(
            {"variant_id": marker, "subgenome": str(subgenome)}
        )
    causal_subgenomes = {item["subgenome"] for item in normalized_causal}
    expected_count = 1 if placement == "one_subgenome" else 2
    if scan_pve > 0.0 and len(causal_subgenomes) != expected_count:
        raise ValueError("scan truth causal subgenomes differ from placement")
    if scan_pve == 0.0 and normalized_causal:
        raise ValueError("null scan truth must not declare causal variants")
    record = {
        "placement": placement,
        "scan_pve": scan_pve,
        "causal_variants": normalized_causal,
        "causal_variant_ids": [item["variant_id"] for item in normalized_causal],
        "causal_subgenomes": sorted(causal_subgenomes),
        "realized_signal_pve": realized,
    }
    record["truth_hash"] = sha256_payload(record)
    return record


def build_scan_comparators(
    kernels: Mapping[str, np.ndarray],
    *,
    sample_ids: Sequence[str] | np.ndarray,
    variant_ids: Mapping[str, Sequence[str]],
    variant_positions_bp: Mapping[str, Sequence[int]] | None = None,
    standardized_variants: Mapping[str, np.ndarray],
    phenotype: Sequence[float] | np.ndarray,
    covariates: np.ndarray,
    qc_declaration: Mapping[str, Any] | None = None,
    truth_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Freeze matched canonical, pooled, and independent scan kernel sets."""

    checked, n = _checked_kernels(kernels)
    samples = _sample_manifest(sample_ids, n)
    response = np.asarray(phenotype, dtype=float)
    fixed = np.asarray(covariates, dtype=float)
    if response.ndim != 1 or response.shape[0] != n or not np.all(np.isfinite(response)):
        raise ValueError("phenotype must be a finite vector aligned to samples")
    if (
        fixed.ndim != 2
        or fixed.shape[0] != n
        or fixed.shape[1] < 1
        or not np.all(np.isfinite(fixed))
    ):
        raise ValueError("covariates must be a finite sample-aligned matrix")
    if not isinstance(variant_ids, Mapping) or set(variant_ids) != set(checked):
        raise ValueError("variant_ids must contain exactly the kernel subgenomes")
    if (
        not isinstance(standardized_variants, Mapping)
        or set(standardized_variants) != set(checked)
    ):
        raise ValueError(
            "standardized_variants must contain exactly the kernel subgenomes"
        )
    ordered_variants: dict[str, list[str]] = {}
    ordered_positions: dict[str, list[int]] = {}
    frozen_variants: dict[str, np.ndarray] = {}
    all_variants: list[str] = []
    positions_explicit = variant_positions_bp is not None
    if variant_positions_bp is not None and (
        not isinstance(variant_positions_bp, Mapping)
        or set(variant_positions_bp) != set(checked)
    ):
        raise ValueError(
            "variant_positions_bp must contain exactly the kernel subgenomes"
        )
    for name in checked:
        raw = list(variant_ids[name])
        if not raw or any(not isinstance(value, str) or not value for value in raw):
            raise ValueError("variant IDs must be non-empty strings")
        block = np.asarray(standardized_variants[name], dtype=float)
        if (
            block.ndim != 2
            or block.shape != (n, len(raw))
            or not np.all(np.isfinite(block))
        ):
            raise ValueError(
                "standardized variant blocks must match samples and variant IDs"
            )
        frozen_variants[name] = np.array(block, copy=True)
        frozen_variants[name].setflags(write=False)
        ordered_variants[name] = raw
        raw_positions = (
            list(variant_positions_bp[name])
            if variant_positions_bp is not None
            else list(range(1, len(raw) + 1))
        )
        if (
            len(raw_positions) != len(raw)
            or any(
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or int(value) < 0
                for value in raw_positions
            )
        ):
            raise ValueError(
                "variant_positions_bp must be aligned non-negative integers"
            )
        ordered_positions[name] = [int(value) for value in raw_positions]
        all_variants.extend(raw)
    if len(set(all_variants)) != len(all_variants):
        raise ValueError("variant IDs must be unique across the experiment")
    truth = _scan_truth_record(truth_metadata, ordered_variants)

    frozen = {name: np.array(value, copy=True) for name, value in checked.items()}
    for value in frozen.values():
        value.setflags(write=False)
    pooled = np.sum(np.stack(list(frozen.values())), axis=0)
    trace_scale = float(np.trace(pooled) / n)
    if not math.isfinite(trace_scale) or trace_scale <= 0.0:
        raise ValueError("pooled kernel has non-positive trace")
    pooled = np.asarray(pooled / trace_scale, dtype=float)
    pooled.setflags(write=False)
    context = {
        "sample_manifest": samples,
        "variant_ids": ordered_variants,
        "variant_positions_bp": ordered_positions,
        "distance_unit": "bp",
        "bp_positions_explicit": positions_explicit,
        "variant_order_hash": sha256_payload(ordered_variants),
        "standardized_variant_fingerprints": {
            name: _numeric_fingerprint(value)
            for name, value in frozen_variants.items()
        },
        "phenotype": _numeric_fingerprint(response),
        "covariates": _numeric_fingerprint(fixed),
        "qc_declaration": _json_safe(
            qc_declaration
            or {"maf_min": 0.05, "call_rate_min": 0.9}
        ),
        "truth": truth,
        "kernel_fingerprints": {
            name: _kernel_fingerprint(value) for name, value in frozen.items()
        },
    }
    frozen_samples = np.asarray(samples["ordered_sample_ids"], dtype=object)
    frozen_response = np.array(response, copy=True)
    frozen_fixed = np.array(fixed, copy=True)
    frozen_samples.setflags(write=False)
    frozen_response.setflags(write=False)
    frozen_fixed.setflags(write=False)
    return {
        "context_fingerprint": sha256_payload(context),
        "context": context,
        "sample_ids": frozen_samples,
        "phenotype": frozen_response,
        "covariates": frozen_fixed,
        "truth": truth,
        "standardized_variants": frozen_variants,
        "canonical_multi_kernel": frozen,
        "pooled_trace_sum": {"pooled": pooled},
        "independent_subgenome": {
            name: {name: value} for name, value in frozen.items()
        },
        "loco": {
            "enabled": False,
            "separate_sensitivity_arm": True,
            "mixed_with_primary_family": False,
        },
    }


def experiment_wide_scan_fwer(
    pvalues_by_subgenome: Mapping[str, Sequence[float] | np.ndarray],
    *,
    variant_ids: Mapping[str, Sequence[str]],
    variant_positions_bp: Mapping[str, Sequence[int]],
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Bonferroni-correct independent scans as one subgenome-union family."""

    if not isinstance(pvalues_by_subgenome, Mapping) or not pvalues_by_subgenome:
        raise ValueError("pvalues_by_subgenome must be a non-empty mapping")
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    names = list(pvalues_by_subgenome)
    if not isinstance(variant_ids, Mapping) or set(variant_ids) != set(names):
        raise ValueError("variant_ids must match pvalue subgenomes")
    if (
        not isinstance(variant_positions_bp, Mapping)
        or set(variant_positions_bp) != set(names)
    ):
        raise ValueError("variant_positions_bp must match pvalue subgenomes")
    arrays: dict[str, np.ndarray] = {}
    ordered_members: list[dict[str, Any]] = []
    for name in names:
        values = np.asarray(pvalues_by_subgenome[name], dtype=float)
        ids = list(variant_ids[name])
        positions = list(variant_positions_bp[name])
        if values.ndim != 1 or values.size != len(ids) or len(positions) != len(ids):
            raise ValueError(
                "p-values, variant IDs, and bp positions must be aligned vectors"
            )
        finite = values[np.isfinite(values)]
        if np.any((finite < 0.0) | (finite > 1.0)) or np.isinf(values).any():
            raise ValueError("p-values must be in [0, 1] or NaN")
        if any(not isinstance(value, str) or not value for value in ids):
            raise ValueError("variant IDs must be non-empty strings")
        if any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or int(value) < 0
            for value in positions
        ):
            raise ValueError("variant bp positions must be non-negative integers")
        arrays[name] = values
        ordered_members.extend(
            {
                "subgenome": name,
                "variant_id": variant_id,
                "position_bp": int(position),
            }
            for variant_id, position in zip(ids, positions, strict=True)
        )
    member_ids = [item["variant_id"] for item in ordered_members]
    if len(set(member_ids)) != len(member_ids):
        raise ValueError("variant IDs must be unique across the experiment")
    family_size = len(ordered_members)
    if family_size == 0:
        raise ValueError("experiment-wide FWER family is empty")
    nonfinite_ids = [
        member
        for name, values in arrays.items()
        for member, value in zip(
            [item for item in ordered_members if item["subgenome"] == name],
            values,
            strict=True,
        )
        if not math.isfinite(float(value))
    ]
    finite_count = family_size - len(nonfinite_ids)
    adjusted: dict[str, list[float | None]] = {}
    rejected: dict[str, list[bool]] = {}
    for name, values in arrays.items():
        adj = np.minimum(values * family_size, 1.0)
        adjusted[name] = [
            float(value) if math.isfinite(float(value)) else None for value in adj
        ]
        rejected[name] = [
            bool(math.isfinite(float(value)) and value <= alpha) for value in adj
        ]
    return {
        "method": "bonferroni",
        "family_scope": "experiment_wide_subgenome_union",
        "distance_unit": "bp",
        "alpha": float(alpha),
        "family_size": family_size,
        "planned_count": family_size,
        "finite_count": finite_count,
        "nonfinite_ids": nonfinite_ids,
        "missingness": {
            "planned_count": family_size,
            "finite_count": finite_count,
            "nonfinite_count": len(nonfinite_ids),
        },
        "ordered_family": ordered_members,
        "ordered_family_hash": sha256_payload(ordered_members),
        "raw_p": {
            name: [
                float(value) if math.isfinite(float(value)) else None
                for value in values
            ]
            for name, values in arrays.items()
        },
        "adjusted_p": adjusted,
        "rejected": rejected,
        "directionwise_alpha_families": False,
        "failure": {
            "failed": bool(nonfinite_ids),
            "error_type": "MissingPValues" if nonfinite_ids else None,
            "message": (
                "planned variants have non-finite p-values"
                if nonfinite_ids
                else None
            ),
        },
    }


_PREFLIGHT_COLUMNS = (
    "comparator",
    "executable_path",
    "executable_basename",
    "binary_sha256",
    "version_output",
    "parsed_identity",
    "parsed_version",
    "version_returncode",
    "smoke_status",
    "sample_order_hash",
    "genotype_hash",
    "qc_hash",
    "covariates_hash",
    "variant_manifest_hash",
    "input_context_hash",
    "status",
    "reason",
)


def _validated_sha256(value: str, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lower-case SHA-256")
    return value


def _resolve_executable(candidate: str | Path) -> str | None:
    value = str(candidate)
    path = Path(value)
    if path.is_absolute() or path.parent != Path("."):
        return str(path.resolve()) if path.is_file() and path.stat().st_mode & 0o111 else None
    found = shutil.which(value)
    return str(Path(found).resolve()) if found else None


def _probe_executable_version(path: str, comparator: str) -> dict[str, Any]:
    basename = Path(path).name.lower()
    if basename.endswith(".exe"):
        basename = basename[:-4]
    allowed = {"gcta", "gcta64"} if comparator == "GCTA" else {"gemma"}
    last_returncode: int | None = None
    version_output = ""
    for flag in ("--version", "-v"):
        try:
            completed = subprocess.run(
                [path, flag],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        last_returncode = int(completed.returncode)
        version_output = " ".join(
            (completed.stdout or completed.stderr).split()
        )[:500]
        if completed.returncode == 0:
            break
    match = re.search(r"\b\d+(?:\.\d+)+\b", version_output)
    identity_ok = basename in allowed and comparator.lower() in version_output.lower()
    return {
        "executable_basename": basename,
        "version_output": version_output,
        "parsed_identity": comparator if identity_ok else "",
        "parsed_version": match.group(0) if identity_ok and match else "",
        "version_returncode": last_returncode,
    }


def _preflight_hashes(
    sample_ids: Sequence[str],
    qc_declaration: Mapping[str, Any],
    covariates: np.ndarray,
    variant_ids: Mapping[str, Sequence[str]],
    genotypes: Mapping[str, np.ndarray],
) -> dict[str, str]:
    samples = list(sample_ids)
    if not samples or any(not isinstance(value, str) or not value for value in samples):
        raise ValueError("preflight sample IDs must be non-empty strings")
    if len(set(samples)) != len(samples):
        raise ValueError("preflight sample IDs must be unique")
    if not isinstance(qc_declaration, Mapping) or not qc_declaration:
        raise ValueError("preflight QC declaration must be a non-empty mapping")
    fixed = np.asarray(covariates)
    if (
        fixed.ndim != 2
        or fixed.shape[0] != len(samples)
        or fixed.shape[1] < 1
        or not np.issubdtype(fixed.dtype, np.number)
        or not np.all(np.isfinite(fixed))
    ):
        raise ValueError("preflight covariates must be finite and sample-aligned")
    if not isinstance(variant_ids, Mapping) or not variant_ids:
        raise ValueError("preflight variant manifest must be a non-empty mapping")
    if not isinstance(genotypes, Mapping) or list(genotypes) != list(variant_ids):
        raise ValueError("preflight genotypes must match ordered variant subgenomes")
    ordered_variants: dict[str, list[str]] = {}
    flat: list[str] = []
    for name, raw_ids in variant_ids.items():
        if not isinstance(name, str) or not name:
            raise ValueError("preflight subgenome names must be non-empty strings")
        ids = list(raw_ids)
        if not ids or any(not isinstance(value, str) or not value for value in ids):
            raise ValueError("preflight variant IDs must be non-empty strings")
        ordered_variants[name] = ids
        flat.extend(ids)
        block = np.asarray(genotypes[name])
        if (
            block.ndim != 2
            or block.shape != (len(samples), len(ids))
            or not np.issubdtype(block.dtype, np.number)
            or not np.all(np.isfinite(block))
        ):
            raise ValueError("preflight genotypes must be finite and variant-aligned")
    if len(set(flat)) != len(flat):
        raise ValueError("preflight variant IDs must be experiment-wide unique")
    hashes = {
        "sample_order_hash": sha256_payload(samples),
        "qc_hash": sha256_payload(_json_safe(qc_declaration)),
        "covariates_hash": _exact_array_fingerprint(fixed)["sha256"],
        "variant_manifest_hash": sha256_payload(ordered_variants),
        "genotype_hash": sha256_payload(
            {
                name: _exact_array_fingerprint(np.asarray(genotypes[name]))
                for name in ordered_variants
            }
        ),
    }
    hashes["input_context_hash"] = sha256_payload(hashes)
    return hashes


def write_comparator_preflight(
    path: str | Path,
    *,
    sample_ids: Sequence[str],
    qc_declaration: Mapping[str, Any],
    covariates: np.ndarray,
    variant_ids: Mapping[str, Sequence[str]],
    genotypes: Mapping[str, np.ndarray],
    executables: Mapping[str, str | Path] | None = None,
    outcomes_exist: bool = False,
) -> dict[str, Any]:
    """Probe and freeze the two external comparator rows before outcomes."""

    output = Path(path)
    if outcomes_exist:
        raise RuntimeError("comparator preflight must be frozen before outcomes exist")
    if output.exists():
        raise FileExistsError(f"comparator preflight is already frozen: {output}")
    hashes = _preflight_hashes(
        sample_ids, qc_declaration, covariates, variant_ids, genotypes
    )
    configured = dict(executables or {"GCTA": "gcta64", "GEMMA": "gemma"})
    if set(configured) != {"GCTA", "GEMMA"}:
        raise ValueError("executables must contain exactly GCTA and GEMMA")
    rows: list[dict[str, Any]] = []
    for comparator in ("GCTA", "GEMMA"):
        executable = _resolve_executable(configured[comparator])
        if executable:
            probe = _probe_executable_version(executable, comparator)
            binary_sha256 = hashlib.sha256(Path(executable).read_bytes()).hexdigest()
        else:
            probe = {
                "executable_basename": "",
                "version_output": "",
                "parsed_identity": "",
                "parsed_version": "",
                "version_returncode": None,
            }
            binary_sha256 = ""
        version_ok = bool(
            probe["version_returncode"] == 0
            and probe["parsed_identity"] == comparator
            and probe["parsed_version"]
        )
        smoke_status = "NO_ADAPTER" if version_ok else "UNAVAILABLE"
        if not executable:
            reason = "executable unavailable"
        elif not version_ok:
            reason = "executable identity/version probe failed"
        else:
            reason = "no outcome-independent smoke adapter is implemented"
        rows.append(
            {
                "comparator": comparator,
                "executable_path": executable or "",
                **probe,
                "binary_sha256": binary_sha256,
                "smoke_status": smoke_status,
                **hashes,
                "status": "UNAVAILABLE_OR_NONCOMPARABLE",
                "reason": reason,
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=_PREFLIGHT_COLUMNS, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    return read_comparator_preflight(output)


def read_comparator_preflight(
    path: str | Path,
    *,
    expected_hash: str | None = None,
) -> dict[str, Any]:
    """Read a frozen preflight and optionally enforce its outcome-time hash."""

    source = Path(path)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if expected_hash is not None and digest != _validated_sha256(
        expected_hash, "expected_hash"
    ):
        raise ShardConflict(f"comparator preflight hash differs: {source}")
    with source.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != _PREFLIGHT_COLUMNS:
            raise ValueError("comparator preflight has an invalid schema")
        raw_rows = list(reader)
    if [row["comparator"] for row in raw_rows] != ["GCTA", "GEMMA"]:
        raise ValueError("comparator preflight must contain ordered GCTA/GEMMA rows")
    rows: list[dict[str, Any]] = []
    for raw in raw_rows:
        row: dict[str, Any] = dict(raw)
        for field in (
            "sample_order_hash",
            "genotype_hash",
            "qc_hash",
            "covariates_hash",
            "variant_manifest_hash",
            "input_context_hash",
        ):
            _validated_sha256(raw[field], field)
        if raw["binary_sha256"]:
            _validated_sha256(raw["binary_sha256"], "binary_sha256")
        if raw["version_returncode"] == "":
            row["version_returncode"] = None
        else:
            try:
                row["version_returncode"] = int(raw["version_returncode"])
            except ValueError as error:
                raise ValueError("preflight version returncode is invalid") from error
        if raw["smoke_status"] not in {"UNAVAILABLE", "NO_ADAPTER", "PASS"}:
            raise ValueError("comparator preflight smoke status is invalid")
        if raw["status"] not in {
            "COMPARABLE",
            "UNAVAILABLE_OR_NONCOMPARABLE",
        }:
            raise ValueError("comparator preflight has an invalid status")
        can_compare = raw["smoke_status"] == "PASS"
        if (raw["status"] == "COMPARABLE") != can_compare:
            raise ValueError("comparator preflight status contradicts its evidence")
        rows.append(row)
    return {"path": str(source), "sha256": digest, "rows": rows}


__all__ = [
    "build_scan_comparators",
    "build_fit_config",
    "experiment_wide_scan_fwer",
    "fit_pve_replicate",
    "parse_fit_metrics",
    "read_comparator_preflight",
    "run_fit_replicate",
    "run_scan_replicate",
    "select_pilot_samples",
    "simulate_fit_truth",
    "write_comparator_preflight",
]
