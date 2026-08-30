"""Track A fit benchmark primitives for the v2.0.1 benchmark."""

from __future__ import annotations

import csv
import hashlib
import json
import math
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
    return str(out_dir / "geno" / "{subgenome}" / "all")


def build_fit_config(
    fixture: Mapping[str, Any],
    out_dir: str | Path,
    *,
    coverage: bool = False,
    bootstrap_jobs: int = 1,
    loco: bool = False,
) -> dict[str, Any]:
    """Build the locked, portable Track A HomoeoGWAS fit configuration."""

    output = Path(out_dir)
    subgenomes = [str(value) for value in fixture["subgenomes"]]
    template = _bed_template(fixture, output)
    reml: dict[str, Any] = {"n_starts": 10, "seed": 2026}
    if coverage:
        reml["pve_bootstrap"] = {
            "enabled": True,
            "B": 200,
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
    components = {
        name: _scaled_component(rng, kernel_factor(checked[name]), genetic_targets[name])
        for name in names
    }
    components["e"] = _scaled_component(rng, np.eye(n), residual_target)
    phenotype = np.sum(np.vstack(list(components.values())), axis=0)
    phenotype -= phenotype.mean()
    total_variance = float(np.var(phenotype, ddof=1))
    if total_variance <= 0.0:
        raise ValueError("degenerate phenotype draw")
    phenotype /= np.sqrt(total_variance)

    target = {**genetic_targets, "e": residual_target}
    realized_variance = {
        name: float(np.var(value, ddof=1)) for name, value in components.items()
    }
    realized_pve = {
        name: float(value / total_variance)
        for name, value in realized_variance.items()
    }
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
        "target_component_variance": dict(target),
        "realized_component_variance": realized_variance,
        "realized_pve": realized_pve,
        "total_variance_prescale": total_variance,
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
    source = truth.get("realized_pve", truth)
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


def parse_fit_metrics(
    out_dir: str | Path,
    trait: str,
    *,
    truth: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Parse only released fit JSON/TSV outputs into replicate metrics."""

    output = Path(out_dir)
    summary_path = output / f"summary_{trait}.json"
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read fit summary: {summary_path}") from error
    if not isinstance(summary, Mapping) or not isinstance(summary.get("reml"), Mapping):
        raise ValueError("fit summary does not contain a REML result")
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
    optimizer_status = bool(reml.get("optimizer_status", False))
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
        "acceptance_all_passed": bool(summary.get("acceptance_all_passed", False)),
        "acceptance": list(summary.get("acceptance", [])),
        "nonfinite_pve_components": nonfinite,
        "summary_path": str(summary_path),
        "summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
        "failure": {
            "failed": not optimizer_status or bool(nonfinite),
            "error_type": (
                "NonFinitePVE" if nonfinite else "OptimizerFailure" if not optimizer_status else None
            ),
            "message": (
                f"non-finite PVE components: {nonfinite}"
                if nonfinite
                else str(reml.get("optimizer_message", "optimizer did not converge"))
                if not optimizer_status
                else None
            ),
        },
    }

    uncertainty = reml.get("pve_uncertainty")
    bootstrap_path = output / f"pve_bootstrap_{trait}.tsv"
    if uncertainty is not None:
        if not isinstance(uncertainty, Mapping):
            raise ValueError("reml.pve_uncertainty must be a mapping")
        if not bootstrap_path.exists():
            raise ValueError(f"PVE uncertainty is missing bootstrap TSV: {bootstrap_path}")
        with bootstrap_path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        components = uncertainty.get("components")
        if not isinstance(components, Mapping):
            raise ValueError("PVE uncertainty does not contain component intervals")
        intervals = {
            str(name): {
                "ci_low": float(item["ci_low"]),
                "ci_high": float(item["ci_high"]),
            }
            for name, item in components.items()
        }
        coverage = (
            {
                name: bool(
                    intervals[name]["ci_low"] <= true[name] <= intervals[name]["ci_high"]
                )
                for name in intervals
            }
            if true is not None
            else None
        )
        metrics["pve_bootstrap"] = {
            "method": uncertainty.get("method"),
            "level": float(uncertainty.get("level", 0.95)),
            "B_requested": int(uncertainty.get("B_requested", 0)),
            "B_success": int(uncertainty.get("B_success", len(rows))),
            "B_failed": int(uncertainty.get("B_failed", 0)),
            "intervals": intervals,
            "coverage": coverage,
            "rows": len(rows),
            "columns": list(rows[0]) if rows else [],
            "path": str(bootstrap_path),
            "sha256": hashlib.sha256(bootstrap_path.read_bytes()).hexdigest(),
        }
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
    context_fingerprint = sha256_payload(
        {
            "kernel_order": list(checked),
            "kernel_fingerprints": fingerprints,
            "sample_manifest": samples,
        }
    )
    if comparator_preflight_path is None:
        if comparator_preflight_hash is not None:
            raise ValueError("comparator_preflight_hash requires its frozen TSV")
        preflight = None
    else:
        if comparator_preflight_hash is None:
            raise ValueError("comparator preflight requires its frozen hash")
        preflight = read_comparator_preflight(
            comparator_preflight_path,
            expected_hash=comparator_preflight_hash,
        )
    source = "homoeogwas_outputs" if fit_output_dir is not None else "fit_multi_reml"
    request_hash = sha256_payload(
        {
            "design_hash": design_hash,
            "context_fingerprint": context_fingerprint,
            "scenario": scenario.to_dict(),
            "replicate": replicate,
            "source": source,
            "trait": trait,
            "comparator_preflight_hash": (
                preflight["sha256"] if preflight is not None else None
            ),
        }
    )
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

    seed = derive_seed(design_hash, "fit", scenario.scenario_id, replicate, scenario.stage)
    base = {
        "track": "fit",
        "scenario_id": scenario.scenario_id,
        "replicate": int(replicate),
        "design_hash": design_hash,
        "stage": scenario.stage,
        "formal": scenario.stage == "formal",
        "qa_only": scenario.stage == "pilot",
        "experiment": str(scenario.parameters.get("experiment", "recovery")),
        "seed": int(seed),
        "sample_manifest": samples,
        "kernel_fingerprints": fingerprints,
        "context_fingerprint": context_fingerprint,
        "request_hash": request_hash,
        "result_source": source,
        "comparator_preflight_hash": (
            preflight["sha256"] if preflight is not None else None
        ),
        "external_comparators": (
            preflight["rows"] if preflight is not None else []
        ),
    }
    started = time.perf_counter()
    try:
        allocation = scenario.parameters.get("allocation", "balanced")
        y, truth = simulate_fit_truth(
            checked,
            allocation,
            seed=seed,
            total_pve=scenario.parameters.get("total_pve"),
            dominant_subgenomes=scenario.parameters.get("dominant_subgenomes"),
            null_subgenome=scenario.parameters.get("null_subgenome"),
        )
        if fit_output_dir is None:
            fitted = fit_pve_replicate(y, checked, seed)
            true = dict(truth["realized_pve"])
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
        else:
            metrics = parse_fit_metrics(fit_output_dir, trait, truth=truth)
            metrics["target_pve"] = dict(truth["target_pve"])
        payload = {**base, "truth": truth, **metrics}
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


def build_scan_comparators(
    kernels: Mapping[str, np.ndarray],
    *,
    sample_ids: Sequence[str] | np.ndarray,
    variant_ids: Mapping[str, Sequence[str]],
    standardized_variants: Mapping[str, np.ndarray],
    phenotype: Sequence[float] | np.ndarray,
    covariates: np.ndarray,
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
    frozen_variants: dict[str, np.ndarray] = {}
    all_variants: list[str] = []
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
        all_variants.extend(raw)
    if len(set(all_variants)) != len(all_variants):
        raise ValueError("variant IDs must be unique across the experiment")

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
        "variant_order_hash": sha256_payload(ordered_variants),
        "standardized_variant_fingerprints": {
            name: _numeric_fingerprint(value)
            for name, value in frozen_variants.items()
        },
        "phenotype": _numeric_fingerprint(response),
        "covariates": _numeric_fingerprint(fixed),
        "kernel_fingerprints": {
            name: _kernel_fingerprint(value) for name, value in frozen.items()
        },
    }
    return {
        "context_fingerprint": sha256_payload(context),
        "context": context,
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
    arrays: dict[str, np.ndarray] = {}
    ordered_members: list[dict[str, str]] = []
    for name in names:
        values = np.asarray(pvalues_by_subgenome[name], dtype=float)
        ids = list(variant_ids[name])
        if values.ndim != 1 or values.size != len(ids):
            raise ValueError("p-values and variant IDs must be aligned vectors")
        finite = values[np.isfinite(values)]
        if np.any((finite < 0.0) | (finite > 1.0)) or np.isinf(values).any():
            raise ValueError("p-values must be in [0, 1] or NaN")
        if any(not isinstance(value, str) or not value for value in ids):
            raise ValueError("variant IDs must be non-empty strings")
        arrays[name] = values
        ordered_members.extend(
            {"subgenome": name, "variant_id": variant_id} for variant_id in ids
        )
    member_ids = [item["variant_id"] for item in ordered_members]
    if len(set(member_ids)) != len(member_ids):
        raise ValueError("variant IDs must be unique across the experiment")
    family_size = len(ordered_members)
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
        "alpha": float(alpha),
        "family_size": family_size,
        "ordered_family": ordered_members,
        "ordered_family_hash": sha256_payload(ordered_members),
        "adjusted_p": adjusted,
        "rejected": rejected,
        "directionwise_alpha_families": False,
    }


_PREFLIGHT_COLUMNS = (
    "comparator",
    "executable_path",
    "version",
    "matched_sample_order",
    "matched_qc",
    "matched_covariates",
    "sample_order_hash",
    "qc_hash",
    "covariates_hash",
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


def _executable_version(path: str) -> str:
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
        text = (completed.stdout or completed.stderr).strip()
        if text:
            return " ".join(text.split())[:500]
    return ""


def write_comparator_preflight(
    path: str | Path,
    *,
    sample_order_hash: str,
    qc_hash: str,
    covariates_hash: str,
    executables: Mapping[str, str | Path] | None = None,
    matched: Mapping[str, Mapping[str, bool]] | None = None,
    outcomes_exist: bool = False,
) -> dict[str, Any]:
    """Probe and freeze the two external comparator rows before outcomes."""

    output = Path(path)
    if outcomes_exist:
        raise RuntimeError("comparator preflight must be frozen before outcomes exist")
    if output.exists():
        raise FileExistsError(f"comparator preflight is already frozen: {output}")
    hashes = {
        "sample_order_hash": _validated_sha256(sample_order_hash, "sample_order_hash"),
        "qc_hash": _validated_sha256(qc_hash, "qc_hash"),
        "covariates_hash": _validated_sha256(covariates_hash, "covariates_hash"),
    }
    configured = dict(executables or {"GCTA": "gcta64", "GEMMA": "gemma"})
    if set(configured) != {"GCTA", "GEMMA"}:
        raise ValueError("executables must contain exactly GCTA and GEMMA")
    matches = dict(matched or {})
    rows: list[dict[str, Any]] = []
    for comparator in ("GCTA", "GEMMA"):
        executable = _resolve_executable(configured[comparator])
        version = _executable_version(executable) if executable else ""
        flags = matches.get(comparator, {})
        sample_match = bool(flags.get("sample_order", False))
        qc_match = bool(flags.get("qc", False))
        covariate_match = bool(flags.get("covariates", False))
        comparable = bool(
            executable and version and sample_match and qc_match and covariate_match
        )
        if not executable:
            reason = "executable unavailable"
        elif not version:
            reason = "version unavailable"
        elif not (sample_match and qc_match and covariate_match):
            reason = "sample order, QC, or covariates are not exactly matched"
        else:
            reason = "matched external comparator"
        rows.append(
            {
                "comparator": comparator,
                "executable_path": executable or "",
                "version": version,
                "matched_sample_order": sample_match,
                "matched_qc": qc_match,
                "matched_covariates": covariate_match,
                **hashes,
                "status": (
                    "COMPARABLE" if comparable else "UNAVAILABLE_OR_NONCOMPARABLE"
                ),
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
            "matched_sample_order",
            "matched_qc",
            "matched_covariates",
        ):
            if raw[field] not in {"True", "False"}:
                raise ValueError(f"comparator preflight {field} must be boolean")
            row[field] = raw[field] == "True"
        for field in ("sample_order_hash", "qc_hash", "covariates_hash"):
            _validated_sha256(raw[field], field)
        if raw["status"] not in {
            "COMPARABLE",
            "UNAVAILABLE_OR_NONCOMPARABLE",
        }:
            raise ValueError("comparator preflight has an invalid status")
        can_compare = bool(
            raw["executable_path"]
            and raw["version"]
            and row["matched_sample_order"]
            and row["matched_qc"]
            and row["matched_covariates"]
        )
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
    "select_pilot_samples",
    "simulate_fit_truth",
    "write_comparator_preflight",
]
