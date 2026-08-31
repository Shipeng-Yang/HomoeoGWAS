from pathlib import Path

import pytest

from scripts.benchmarks.v201.aggregate import _expected_replicates
from scripts.benchmarks.v201.contracts import (
    Budget,
    ScalingAnchor,
    canonical_json,
    derive_seed,
    sha256_payload,
)
from scripts.benchmarks.v201.scenarios import build_scenarios, write_scenario_registry


def test_seed_is_stable_and_stage_separated():
    args = ("a" * 64, "omnib", "B.end2end.cotton.gaussian", 7)
    assert derive_seed(*args, "pilot") == derive_seed(*args, "pilot")
    assert derive_seed(*args, "pilot") != derive_seed(*args, "formal")


def test_registry_ids_are_unique_and_counts_are_locked():
    pilot = build_scenarios("pilot")
    formal = build_scenarios("formal")
    assert len({row.scenario_id for row in pilot}) == len(pilot)
    assert len({row.scenario_id for row in formal}) == len(formal)
    assert all(
        row.replicates == 20
        for row in pilot
        if row.track in {"fit", "omnib"}
        and row.parameters.get("experiment") != "encoding"
    )
    assert any(row.replicates == 500 and row.bootstrap_B == 2000 for row in formal)
    assert sha256_payload([row.to_dict() for row in formal]) == sha256_payload(
        [row.to_dict() for row in formal]
    )


def test_scenario_stage_and_parameters_are_serializable():
    rows = build_scenarios("formal")
    assert rows
    assert all(row.to_dict()["scenario_id"] == row.scenario_id for row in rows)
    assert all(canonical_json(row.to_dict()) for row in rows)
    assert {row.track for row in rows} == {"fit", "omnib", "scaling", "application"}


def test_fixed_scaling_anchors_and_budgets():
    rows = build_scenarios("formal")
    anchors = [row.parameters["anchor"] for row in rows if row.track == "scaling"]
    assert [anchor.anchor_id for anchor in anchors] == [
        "small_qa",
        "cotton_like",
        "wheat_formal",
        "quartet_stress",
    ]
    assert anchors[0] == ScalingAnchor(
        "small_qa", n=192, groups=192, copies=3, edges=3,
        bootstrap_B=199, jobs=(1, 4, 8), repeats=3,
    )
    assert Budget("pilot", 1000, 12, 200).to_dict() == {
        "stage": "pilot", "cpu_hours": 1000, "elapsed_hours": 12,
        "output_gb": 200,
    }


def test_scaling_anchor_repeats_match_stage_execution_budget():
    """Pilot anchors carrying formal repeats would overstate the QA contract."""

    pilot = [row.parameters["anchor"] for row in build_scenarios("pilot")
             if row.track == "scaling"]
    formal = [row.parameters["anchor"] for row in build_scenarios("formal")
              if row.track == "scaling"]
    assert {anchor.repeats for anchor in pilot} == {1}
    assert {anchor.repeats for anchor in formal} == {3}


def test_pilot_uses_qa_settings_without_changing_scenario_identity():
    pilot = {row.scenario_id: row for row in build_scenarios("pilot")}
    formal = {row.scenario_id: row for row in build_scenarios("formal")}
    assert set(pilot) == set(formal)
    assert all(row.parameters["qa_only"] is True for row in pilot.values())
    assert all(row.bootstrap_B == 199 for row in pilot.values() if row.track in {"fit", "omnib"})
    assert all(row.stage == "pilot" for row in pilot.values())


def test_write_scenario_registry_is_deterministic(tmp_path: Path):
    path = tmp_path / "scenario_registry.tsv"
    written = write_scenario_registry(build_scenarios("formal"), path)
    assert written == path
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "scenario_id\ttrack\tstage\treplicates\tbootstrap_B\tparameters"
    assert len(lines) == len(build_scenarios("formal")) + 1


@pytest.mark.parametrize("stage", ["invalid", "FORMAL", ""])
def test_build_scenarios_rejects_unknown_stage(stage):
    with pytest.raises(ValueError, match="stage"):
        build_scenarios(stage)


@pytest.mark.parametrize("copies, edges", [(2, 3), (3, 1), (4, 3)])
def test_scaling_anchor_rejects_inconsistent_pair_edge_count(copies, edges):
    with pytest.raises(ValueError, match="edges"):
        ScalingAnchor("invalid", 100, 10, copies, edges, 199, (1,))


def test_omnib_scenarios_carry_canonical_group_edge_contract():
    expected = {"cotton": (2, 1), "wheat": (3, 3), "quartet": (4, 6)}
    for row in build_scenarios("formal"):
        if row.track != "omnib":
            continue
        parameters = row.parameters
        if parameters["experiment"] == "global_vc":
            assert parameters["hypothesis_unit"] == "global"
            assert parameters["detection_only"] is True
            continue
        assert parameters["mode"] == "group"
        assert parameters["statistic"] == "omniB"
        assert parameters["hypothesis_unit"] == "group"
        assert parameters["subset_order"] == 2
        assert parameters["pair_edges"] == "common_primitive"
        assert (parameters["copies"], parameters["edges_per_group"]) == expected[parameters["backbone"]]
        assert parameters["direct_four_way"] is False


def test_power_scenarios_declare_independent_calibration_bank_size():
    for stage, expected in (("pilot", 20), ("formal", 2_000)):
        power = [
            row for row in build_scenarios(stage)
            if row.track == "omnib" and row.parameters["experiment"] == "power"
        ]
        assert power
        assert {row.parameters["calibration_count"] for row in power} == {expected}
        assert {row.parameters["response_count"] for row in power} == {1}
        assert {row.parameters["null_model"] for row in power} == {"gaussian"}
        assert all(
            row.parameters["calibration_scenario_id"]
            == f"B.conditional.{row.parameters['backbone']}.gaussian.calibration"
            for row in power
        )


def test_registry_has_global_vc_and_locked_family_size_stress_matrix():
    rows = [row for row in build_scenarios("formal") if row.track == "omnib"]
    global_rows = [row for row in rows if row.parameters["experiment"] == "global_vc"]
    assert {(row.parameters["backbone"], row.parameters["bank"]) for row in global_rows} == {
        (backbone, bank)
        for backbone in ("cotton", "wheat", "quartet")
        for bank in ("calibration", "heldout", "power")
    }
    assert all(row.parameters["hypothesis_unit"] == "global" for row in global_rows)
    assert all(row.parameters["detection_only"] is True for row in global_rows)
    assert all(list(_expected_replicates(row)) == [0] for row in global_rows)

    stress = [row for row in rows if row.parameters["experiment"] == "family_size"]
    assert {(row.parameters["backbone"], row.parameters["family_size"]) for row in stress} == {
        (backbone, size)
        for backbone in ("cotton", "wheat", "quartet")
        for size in (80, 500, 2_000)
    }
    assert all(
        row.parameters["snpxsnp_status"]
        == ("applicable" if row.parameters["family_size"] == 80 else "not_applicable_above_80")
        for row in stress
    )
    assert all(list(_expected_replicates(row)) == [0] for row in stress)


def test_cotton_never_registers_multi_edge_group_and_negative_controls_are_explicit():
    power = [
        row for row in build_scenarios("formal")
        if row.track == "omnib" and row.parameters["experiment"] == "power"
    ]
    assert not any(
        row.parameters["backbone"] == "cotton"
        and row.parameters["architecture"] == "multi_edge_group"
        for row in power
    )
    for row in power:
        negative = row.parameters["architecture"] in {"additive_only", "mispaired"}
        assert row.parameters["control_type"] == ("negative" if negative else "positive")
        assert row.parameters["report_metric"] == ("specificity" if negative else "power")


def test_conditional_registry_locks_complete_null_matrix_and_gate_roles():
    """Dropping a null arm or promoting a stress arm into core must fail."""

    expected_nulls = {
        "gaussian",
        "student_t5",
        "heteroscedastic_pc1",
        "contamination_1pct_6sd",
        "additive_only",
        "omitted_kernel",
    }
    rows = [
        row for row in build_scenarios("formal")
        if row.track == "omnib" and row.parameters["experiment"] == "conditional"
    ]
    assert len(rows) == 3 * 6 * 2
    observed = {
        (
            row.parameters["backbone"],
            row.parameters["null_model"],
            row.parameters["bank"],
        )
        for row in rows
    }
    assert observed == {
        (backbone, null_model, bank)
        for backbone in ("cotton", "wheat", "quartet")
        for null_model in expected_nulls
        for bank in ("calibration", "heldout")
    }
    for row in rows:
        null_model = row.parameters["null_model"]
        assert row.parameters["core"] is (
            null_model in {"gaussian", "additive_only"}
        )
        assert row.parameters["stress"] is (
            null_model
            in {
                "student_t5",
                "heteroscedastic_pc1",
                "contamination_1pct_6sd",
                "omitted_kernel",
            }
        )
        assert row.parameters["descriptive"] is (not row.parameters["core"])
        assert row.parameters["acceptance_eligible"] is row.parameters["core"]
        assert row.parameters["failure_boundary"] is (
            null_model == "omitted_kernel"
        )


def test_registry_has_one_encoding_batch_per_backbone():
    """Omitting a ploidy backbone from the encoding dispatcher must fail."""

    rows = [
        row for row in build_scenarios("formal")
        if row.track == "omnib" and row.parameters["experiment"] == "encoding"
    ]
    assert [row.scenario_id for row in rows] == [
        "B.encoding.cotton",
        "B.encoding.wheat",
        "B.encoding.quartet",
    ]
    assert [row.parameters["backbone"] for row in rows] == [
        "cotton", "wheat", "quartet",
    ]
    assert all(row.replicates == 1 for row in rows)


def test_application_scenarios_are_read_only_and_never_rescan():
    rows = [row for row in build_scenarios("formal") if row.track == "application"]
    assert len(rows) == 4
    assert all(row.parameters["read_only"] is True for row in rows)
    assert all(row.parameters["rescan"] is False for row in rows)
