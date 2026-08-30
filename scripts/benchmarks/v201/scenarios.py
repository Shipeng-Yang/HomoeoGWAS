"""Frozen v2.0.1 benchmark scenario registry."""

import csv
from collections.abc import Iterable
from pathlib import Path

from .contracts import (
    FORMAL_BUDGET,
    PILOT_BUDGET,
    Budget,
    ScalingAnchor,
    Scenario,
    canonical_json,
)

__all__ = [
    "Budget",
    "FORMAL_BUDGET",
    "PILOT_BUDGET",
    "ScalingAnchor",
    "Scenario",
    "build_scenarios",
    "write_scenario_registry",
]


def _scenario(
    scenario_id: str,
    track: str,
    stage: str,
    replicates: int,
    bootstrap_B: int,
    **parameters: object,
) -> Scenario:
    parameters = {"qa_only": stage == "pilot", **parameters}
    return Scenario(scenario_id, track, stage, replicates, bootstrap_B, parameters)


def _pve_label(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def _fit_scenarios(stage: str) -> list[Scenario]:
    rows: list[Scenario] = []
    panels = ("cotton", "wheat")
    allocations = ("balanced", "single_dominant", "two_dominant", "null_component", "low_total_pve")
    for panel in panels:
        for allocation in allocations:
            total_pve = 0.15 if allocation == "low_total_pve" else 0.40
            rows.append(_scenario(
                f"A.recovery.{panel}.{allocation}", "fit", stage,
                20 if stage == "pilot" else 300, 199 if stage == "pilot" else 0,
                panel=panel, experiment="recovery", allocation=allocation,
                total_pve=total_pve,
            ))

        for allocation in ("balanced", "single_dominant", "null_component"):
            rows.append(_scenario(
                f"A.coverage.{panel}.{allocation}", "fit", stage,
                20 if stage == "pilot" else 200, 199 if stage == "pilot" else 200,
                panel=panel, experiment="coverage", allocation=allocation,
                interval_level=0.95,
            ))

        for placement in ("one_subgenome", "two_subgenomes"):
            for pve in (0.0, 0.02, 0.05, 0.10):
                reps = 1_000 if pve == 0 else 300
                rows.append(_scenario(
                    f"A.scan.{panel}.{placement}.pve_{_pve_label(pve)}",
                    "fit", stage, 20 if stage == "pilot" else reps,
                    199 if stage == "pilot" else 0,
                    panel=panel, experiment="scan", placement=placement,
                    scan_pve=pve,
                ))

        for pve, reps in ((0.0, 100), (0.05, 100)):
            rows.append(_scenario(
                f"A.loco.{panel}.pve_{_pve_label(pve)}", "fit", stage,
                20 if stage == "pilot" else reps, 199 if stage == "pilot" else 0,
                panel=panel, experiment="loco", scan_pve=pve,
            ))
    return rows


def _omnib_scenarios(stage: str) -> list[Scenario]:
    rows: list[Scenario] = []
    core_backbones = ("cotton", "wheat")
    null_models = ("gaussian", "additive_only", "structure_aligned")

    def omnib_row(scenario_id: str, replicates: int, bootstrap_B: int, **parameters: object) -> Scenario:
        copies, edges = {"cotton": (2, 1), "wheat": (3, 3), "quartet": (4, 6)}[parameters["backbone"]]
        return _scenario(
            scenario_id, "omnib", stage, replicates, bootstrap_B,
            mode="group", statistic="omniB", hypothesis_unit="group",
            subset_order=2, pair_edges="common_primitive", copies=copies,
            edges_per_group=edges, direct_four_way=False, **parameters,
        )

    for backbone in core_backbones:
        for null_model in null_models:
            rows.append(omnib_row(
                f"B.end2end.{backbone}.{null_model}",
                20 if stage == "pilot" else 500, 199 if stage == "pilot" else 2_000,
                backbone=backbone, experiment="end2end", null_model=null_model,
            ))

    # The calibration and held-out banks are disjoint, score-level banks.
    for backbone in ("cotton", "wheat", "quartet"):
        for bank in ("calibration", "heldout"):
            rows.append(omnib_row(
                f"B.conditional.{backbone}.{bank}",
                20 if stage == "pilot" else 2_000, 199 if stage == "pilot" else 0,
                backbone=backbone, experiment="conditional", bank=bank,
            ))

    architectures = (
        "minor_burden_aligned", "pc1_distributed", "kernel_multidimensional",
        "single_snp_pair", "mixed_sign", "multi_edge_group", "additive_only",
        "mispaired",
    )
    for backbone in ("cotton", "wheat", "quartet"):
        for architecture in architectures:
            for causal_groups in (1, 4):
                for pve in (0.02, 0.05, 0.10):
                    rows.append(omnib_row(
                        f"B.power.{backbone}.{architecture}.g{causal_groups}.pve_{_pve_label(pve)}",
                        20 if stage == "pilot" else 500, 199 if stage == "pilot" else 0,
                        backbone=backbone, experiment="power",
                        architecture=architecture, causal_groups=causal_groups,
                        interaction_pve=pve,
                    ))
    return rows


def _scaling_scenarios(stage: str) -> list[Scenario]:
    anchors = (
        ScalingAnchor("small_qa", 192, 192, 3, 3, 199, (1, 4, 8)),
        ScalingAnchor("cotton_like", 419, 500, 2, 1, 999, (1, 4, 8, 16)),
        ScalingAnchor("wheat_formal", 827, 2_143, 3, 3, 2_000, (1, 8, 16, 32)),
        ScalingAnchor("quartet_stress", 500, 500, 4, 6, 999, (1, 8, 16)),
    )
    return [
        _scenario(
            f"C.{anchor.anchor_id}", "scaling", stage,
            1 if stage == "pilot" else anchor.repeats,
            anchor.bootstrap_B, anchor=anchor,
        )
        for anchor in anchors
    ]


def _application_scenarios(stage: str) -> list[Scenario]:
    return [
        _scenario(
            f"D.{species}", "application", stage, 1, 0,
            species=species, read_only=True, rescan=False,
        )
        for species in ("wheat", "cotton", "rapeseed", "peanut")
    ]


def build_scenarios(stage: str) -> list[Scenario]:
    """Build the immutable ordered registry for ``pilot`` or ``formal``."""

    if stage not in {"pilot", "formal"}:
        raise ValueError("stage must be pilot or formal")
    return (
        _fit_scenarios(stage)
        + _omnib_scenarios(stage)
        + _scaling_scenarios(stage)
        + _application_scenarios(stage)
    )


def write_scenario_registry(
    scenarios: Iterable[Scenario], path: str | Path,
) -> Path:
    """Write a deterministic tab-separated scenario registry and return its path."""

    rows = list(scenarios)
    ids = [row.scenario_id for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("scenario IDs must be unique")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("scenario_id", "track", "stage", "replicates", "bootstrap_B", "parameters"))
        for row in rows:
            writer.writerow((
                row.scenario_id, row.track, row.stage, row.replicates,
                row.bootstrap_B,
                canonical_json(row.to_dict()["parameters"]),
            ))
    return output
