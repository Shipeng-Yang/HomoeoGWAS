import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from homoeogwas.audit import audit_result, cmd_audit, run_audit
from homoeogwas.cli import main


def _write(path, payload):
    path.write_text(json.dumps(payload))
    return path


def _canonical_group_omnib_payload(*, hypothesis_unit="group", n_groups=4,
                                   n_edges=12):
    n_hypotheses = n_edges if hypothesis_unit == "edge" else n_groups
    hypothesis_ids = [
        f"{hypothesis_unit}:h{i}" for i in range(n_hypotheses)]
    family_order_sha256 = hashlib.sha256(
        "\x00".join(hypothesis_ids).encode()).hexdigest()
    group_sha256 = "a" * 64
    edge_sha256 = "b" * 64
    fwer = {
        "alpha": 0.05,
        "B": 2000,
        "method": "parametric_bootstrap_minp_plus_one",
        "inferential": True,
        "formal_discovery_layer": True,
        "family_id": hypothesis_unit,
        "family_scope": "primary_only",
        "declared_hypothesis_unit": hypothesis_unit,
        "calibrated_layers": [hypothesis_unit],
        "n_hypotheses": n_hypotheses,
        "n_calibrated": n_hypotheses,
        "n_unestimable": 0,
        "hypothesis_ids": hypothesis_ids,
        "family_order_sha256": family_order_sha256,
        "observed_p": [0.2] * n_hypotheses,
        "adjusted_p": [1.0] * n_hypotheses,
        "threshold": 0.01,
        "threshold_comparator": "strict_less_than",
        "rejected": False,
        "rejected_indices": [],
        "rejected_hypothesis_ids": [],
        "n_rejected": 0,
        "empirical_p": 0.2,
        "sig": [],
    }
    family_provenance = {
        "group_family_sha256": group_sha256,
        "edge_family_sha256": edge_sha256,
        "n_groups_raw": n_groups,
        "n_unique_edges": n_edges,
    }
    primary = {
        "statistic": "omniB",
        "calibration_method": "bootstrap",
        "bootstrap_B": 2000,
        "n": 100,
        "G": n_hypotheses,
        "n_planned": n_hypotheses,
        "n_valid": n_hypotheses,
        "n_unestimable": 0,
        "n_sig": 0,
        "sig": [],
        "top": [],
        "lambda_gc_obs": 1.0,
        "minp_boot_emp": 0.2,
        "minp_boot_threshold": 0.01,
        "minp_boot_rejected": False,
        "component_diagnostics": {"role": "descriptive_localization"},
        "model_diagnostics": {
            "bootstrap_fwer": fwer,
            "family_provenance": family_provenance,
        },
    }
    return {
        "tool": "homoeogwas",
        "command": "interact",
        "mode": "group",
        "trait": "trait",
        "provenance": {
            "mode": "group",
            "statistic": "omniB",
            "primary_transform": "INT",
            "primary_multiplicity": "bootstrap_minp",
            "calibration_method": "bootstrap",
            "hypothesis_unit": hypothesis_unit,
            "family_scope": "primary_only",
            **family_provenance,
        },
        "results": {"INT": primary},
    }


def test_audit_group_claim_boundary(tmp_path):
    payload = _canonical_group_omnib_payload(hypothesis_unit="group")
    record = audit_result(_write(tmp_path / "interact_trait.json", payload))
    assert record.discovery_count == 0
    assert any(
        "within a homoeolog group" in text
        for text in record.evidence_boundary)
    assert any(
        "not" in text.lower() and "fourth-order" in text
        for text in record.evidence_boundary)


def test_audit_edge_claim_boundary(tmp_path):
    payload = _canonical_group_omnib_payload(hypothesis_unit="edge")
    record = audit_result(_write(tmp_path / "interact_trait.json", payload))
    assert any(
        "for a homoeolog pair" in text
        for text in record.evidence_boundary)


def test_audit_canonical_group_requires_family_hashes_and_counts(tmp_path):
    payload = _canonical_group_omnib_payload()
    del payload["provenance"]["edge_family_sha256"]
    payload["provenance"]["n_groups_raw"] = 5
    record = audit_result(_write(tmp_path / "interact_trait.json", payload))
    codes = {flag.code for flag in record.flags}
    assert "OMNIB_FAMILY_PROVENANCE_MISSING" in codes
    assert "OMNIB_FWER_HYPOTHESIS_COUNT_MISMATCH" in codes
    assert record.discovery_count is None


def test_audit_canonical_group_refuses_uncalibrated_second_layer(tmp_path):
    payload = _canonical_group_omnib_payload()
    payload["results"]["INT"]["model_diagnostics"]["bootstrap_fwer"][
        "calibrated_layers"] = ["group", "edge"]
    record = audit_result(_write(tmp_path / "interact_trait.json", payload))
    assert "OMNIB_FWER_UNCALIBRATED_SECOND_PRIMARY_LAYER" in {
        flag.code for flag in record.flags}
    assert record.discovery_count is None


def test_audit_canonical_group_refuses_low_resolution_or_wrong_method(tmp_path):
    payload = _canonical_group_omnib_payload()
    primary = payload["results"]["INT"]
    primary["bootstrap_B"] = 99
    primary["model_diagnostics"]["bootstrap_fwer"]["B"] = 99
    primary["model_diagnostics"]["bootstrap_fwer"]["method"] = "bootstrap"
    record = audit_result(_write(tmp_path / "interact_trait.json", payload))
    codes = {flag.code for flag in record.flags}
    assert "OMNIB_BOOTSTRAP_MONTE_CARLO_RESOLUTION" in codes
    assert "OMNIB_FWER_METHOD_MISMATCH" in codes
    assert record.discovery_count is None


@pytest.mark.parametrize("field", [
    "n_sig", "sig", "minp_boot_rejected", "inferential",
    "formal_discovery_layer", "rejected", "rejected_hypothesis_ids",
])
def test_audit_canonical_group_rejects_raw_second_authority(tmp_path, field):
    payload = _canonical_group_omnib_payload()
    sibling = {
        "statistic": "omniB",
        "n_sig": None,
        "sig": None,
        "minp_boot_rejected": None,
        "model_diagnostics": {"bootstrap_fwer": {
            "inferential": False,
            "formal_discovery_layer": False,
            "rejected": None,
            "n_rejected": None,
            "rejected_hypothesis_ids": None,
        }},
    }
    if field in {"inferential", "formal_discovery_layer", "rejected"}:
        sibling["model_diagnostics"]["bootstrap_fwer"][field] = True
    elif field == "rejected_hypothesis_ids":
        sibling["model_diagnostics"]["bootstrap_fwer"][field] = ["raw:rogue"]
    elif field == "sig":
        sibling[field] = [{"hypothesis_id": "raw:rogue", "p": 1e-9}]
    elif field == "n_sig":
        sibling[field] = 1
    else:
        sibling[field] = True
    payload["results"]["raw"] = sibling

    record = audit_result(_write(tmp_path / f"raw_{field}.json", payload))
    flags = {flag.code: flag.severity for flag in record.flags}
    assert flags["OMNIB_FWER_UNCALIBRATED_SECOND_PRIMARY_LAYER"] == "error"
    assert record.status == "ANALYSIS_INVALID"
    assert record.discovery_count is None


def test_audit_canonical_group_accepts_nonprimary_noninferential_diagnostics(
        tmp_path):
    payload = _canonical_group_omnib_payload()
    payload["results"]["raw"] = {
        "statistic": "omniB",
        "n_sig": None,
        "sig": None,
        "minp_boot_rejected": None,
        "model_diagnostics": {"bootstrap_fwer": {
            "inferential": False,
            "formal_discovery_layer": False,
            "rejected": None,
            "n_rejected": None,
            "sig": None,
            "rejected_indices": None,
            "rejected_hypothesis_ids": None,
            "qa_diagnostics": {"role": "noninferential_engineering"},
        }},
    }
    record = audit_result(_write(tmp_path / "qa_raw.json", payload))
    assert record.status == "NO_FAMILYWISE_DISCOVERY"
    assert record.discovery_count == 0
    assert "OMNIB_FWER_UNCALIBRATED_SECOND_PRIMARY_LAYER" not in {
        flag.code for flag in record.flags}


def test_audit_canonical_group_accepts_primary_qa_only_serialization(tmp_path):
    payload = _canonical_group_omnib_payload()
    primary = payload["results"]["INT"]
    fwer = primary["model_diagnostics"]["bootstrap_fwer"]
    primary.update(
        bootstrap_B=3,
        n_sig=None,
        sig=None,
        minp_boot_rejected=None,
        minp_boot_emp=None,
        minp_boot_threshold=None,
    )
    fwer.update(
        B=3,
        inferential=False,
        formal_discovery_layer=False,
        adjusted_p=[None] * len(fwer["hypothesis_ids"]),
        threshold=None,
        threshold_comparator=None,
        rejected=None,
        rejected_indices=None,
        rejected_hypothesis_ids=None,
        n_rejected=None,
        empirical_p=None,
        sig=None,
        qa_diagnostics={
            "role": "noninferential_do_not_threshold",
            "empirical_p": 0.2,
            "threshold": 0.01,
            "threshold_comparator": "strict_less_than",
            "adjusted_p": [1.0] * len(fwer["hypothesis_ids"]),
        },
    )
    record = audit_result(_write(tmp_path / "qa_primary.json", payload))
    assert record.status != "ANALYSIS_INVALID"
    assert record.discovery_count is None
    assert "OMNIB_QA_ONLY" in {flag.code for flag in record.flags}


@pytest.mark.parametrize("bad", [4.9, "4", True, None])
def test_audit_canonical_group_rejects_noninteger_family_counts(tmp_path, bad):
    payload = _canonical_group_omnib_payload()
    payload["provenance"]["n_groups_raw"] = bad
    payload["results"]["INT"]["model_diagnostics"]["family_provenance"][
        "n_groups_raw"] = bad
    record = audit_result(_write(tmp_path / "bad_count.json", payload))
    assert record.status == "ANALYSIS_INVALID"
    assert "OMNIB_FAMILY_COUNTS_INVALID" in {f.code for f in record.flags}


@pytest.mark.parametrize("bad", [2000.9, "2000", True, 0, None])
def test_audit_canonical_group_rejects_invalid_bootstrap_B(tmp_path, bad):
    payload = _canonical_group_omnib_payload()
    primary = payload["results"]["INT"]
    primary["bootstrap_B"] = bad
    primary["model_diagnostics"]["bootstrap_fwer"]["B"] = bad
    record = audit_result(_write(tmp_path / "bad_b.json", payload))
    assert record.status == "ANALYSIS_INVALID"
    assert "OMNIB_FWER_BOOTSTRAP_B_INVALID" in {f.code for f in record.flags}


@pytest.mark.parametrize("vector", ["observed_p", "adjusted_p"])
@pytest.mark.parametrize("bad", ["bad", True, float("nan"), -0.01, 1.01])
def test_audit_canonical_group_rejects_invalid_probability_vectors(
        tmp_path, vector, bad):
    payload = _canonical_group_omnib_payload()
    payload["results"]["INT"]["model_diagnostics"]["bootstrap_fwer"][
        vector][0] = bad
    record = audit_result(_write(tmp_path / f"bad_{vector}.json", payload))
    assert record.status == "ANALYSIS_INVALID"
    expected = f"OMNIB_FWER_{vector.upper()}_INVALID"
    assert expected in {f.code for f in record.flags}


@pytest.mark.parametrize("bad_ids", [["group:h0", "group:h0"],
                                      ["group:h0", 2]])
def test_audit_canonical_group_rejects_invalid_hypothesis_ids(tmp_path, bad_ids):
    payload = _canonical_group_omnib_payload(n_groups=2)
    fwer = payload["results"]["INT"]["model_diagnostics"]["bootstrap_fwer"]
    fwer["hypothesis_ids"] = bad_ids
    fwer["family_order_sha256"] = hashlib.sha256(
        "\x00".join(map(str, bad_ids)).encode()).hexdigest()
    record = audit_result(_write(tmp_path / "bad_ids.json", payload))
    assert record.status == "ANALYSIS_INVALID"
    assert "OMNIB_FWER_HYPOTHESIS_IDS_INVALID" in {f.code for f in record.flags}


def test_cmd_audit_malformed_vectors_returns_controlled_nonzero(tmp_path):
    payload = _canonical_group_omnib_payload()
    payload["results"]["INT"]["model_diagnostics"]["bootstrap_fwer"][
        "adjusted_p"][0] = "bad"
    _write(tmp_path / "interact_trait.json", payload)
    assert cmd_audit(SimpleNamespace(results=str(tmp_path), out_dir=None)) == 1


def _make_positive_canonical_hit(payload, *, unit):
    primary = payload["results"]["INT"]
    fwer = primary["model_diagnostics"]["bootstrap_fwer"]
    hit = {
        "hypothesis_id": f"{unit}:h0",
        "p": 0.001,
        "p_interaction": 0.001,
        "p_adjusted_bootstrap_minp": 0.02,
        "primary_sig": True,
        "p_unestimable": False,
    }
    if unit == "group":
        hit.update(group_id="h0", driving_edge="A:B|a|b",
                   driving_component="pc1")
    else:
        hit.update(edge_id="A:B|a|b", smallest_component="kernel_hadamard")
    primary.update(n_sig=1, sig=[hit], top=[hit], minp_boot_emp=0.02,
                   minp_boot_threshold=0.01, minp_boot_rejected=True)
    fwer.update(
        observed_p=[0.001, *fwer["observed_p"][1:]],
        adjusted_p=[0.02, *fwer["adjusted_p"][1:]],
        rejected=True,
        rejected_indices=[0],
        rejected_hypothesis_ids=[f"{unit}:h0"],
        n_rejected=1,
        empirical_p=0.02,
        sig=[hit],
    )


def test_audit_canonical_group_hit_renders_identity_and_driver(tmp_path):
    payload = _canonical_group_omnib_payload(hypothesis_unit="group")
    _make_positive_canonical_hit(payload, unit="group")
    result = _write(tmp_path / "interact_trait.json", payload)
    record = audit_result(result)
    assert record.discovery_count == 1
    assert record.status == "INTERNAL_DISCOVERY_REPLICATION_REQUIRED"
    component_flags = [f.message for f in record.flags
                       if f.code == "COMPONENT_SPECIFIC_EVIDENCE"]
    assert component_flags and "group:h0" in component_flags[0]
    assert "pc1" in component_flags[0]
    audit = run_audit(result, tmp_path / "audit")
    markdown = Path(audit["outputs"]["markdown"]).read_text()
    assert "group:h0" in markdown and "driving component=pc1" in markdown
    assert "`None`" not in markdown


def test_audit_canonical_edge_hit_renders_identity_and_driver(tmp_path):
    payload = _canonical_group_omnib_payload(hypothesis_unit="edge")
    _make_positive_canonical_hit(payload, unit="edge")
    result = _write(tmp_path / "interact_trait.json", payload)
    record = audit_result(result)
    assert record.discovery_count == 1
    markdown = Path(run_audit(result, tmp_path / "audit")["outputs"][
        "markdown"]).read_text()
    assert "edge:h0" in markdown
    assert "smallest component=kernel_hadamard" in markdown


def _omnib_payload(*, components=True, n_sig=1):
    hit = {
        "pair": ["geneA", "geneD"],
        "p": 1e-7,
        "smallest_component": "kernel_hadamard",
    }
    primary = {
        "statistic": "omniB",
        "calibration_method": "bootstrap",
        "bootstrap_B": 200,
        "n": 300,
        "G": 1000,
        "n_planned": 1000,
        "n_valid": 998,
        "n_unestimable": 2,
        "n_sig": n_sig,
        "lambda_gc_obs": 1.01,
        "top": [hit],
    }
    if components:
        primary["component_diagnostics"] = {
            "names": ["minor_burden", "pc1", "kernel_hadamard"]}
        hit["component_p"] = {
            "minor_burden": 0.4, "pc1": 0.2, "kernel_hadamard": 1e-8}
    return {
        "tool": "homoeogwas",
        "command": "interact",
        "mode": "pairwise",
        "trait": "flowering",
        "provenance": {
            "primary_transform": "INT",
            "statistic": "omniB",
            "calibration_method": "bootstrap",
        },
        "results": {"INT": primary},
    }


def test_audit_omnib_separates_discovery_component_and_replication(tmp_path):
    path = _write(tmp_path / "interact_flowering.json", _omnib_payload())
    record = audit_result(path)
    assert record.status == "INTERNAL_DISCOVERY_REPLICATION_REQUIRED"
    assert record.replication_status == "NOT_ASSESSED"
    codes = {flag.code for flag in record.flags}
    assert "COMPONENT_SPECIFIC_EVIDENCE" in codes
    assert "REPLICATION_REQUIRED" in codes
    assert "OMNIB_COMPONENTS_NOT_RECORDED" not in codes


def test_audit_old_omnib_artifact_flags_missing_components(tmp_path):
    path = _write(
        tmp_path / "interact_old.json",
        _omnib_payload(components=False, n_sig=0))
    record = audit_result(path)
    assert record.status == "NO_FAMILYWISE_DISCOVERY_REVIEW_REQUIRED"
    assert "OMNIB_COMPONENTS_NOT_RECORDED" in {
        flag.code for flag in record.flags}


def test_audit_pairwise_omnib_accepts_consistent_formal_bootstrap_minp(tmp_path):
    # The calibrated object is authoritative even if a stale top-level count
    # still contains an older analytic-screen value.
    payload = _omnib_payload(n_sig=7)
    hit = payload["results"]["INT"]["top"][0]
    hit["p_adjusted_bootstrap_minp"] = 0.02
    primary = payload["results"]["INT"]
    primary["bootstrap_B"] = 2000
    primary["model_diagnostics"] = {
        "bootstrap_fwer": {
            "alpha": 0.05,
            "method": "parametric_bootstrap_minp_plus_one",
            "inferential": True,
            "rejected": True,
            "n_rejected": 1,
            "empirical_p": 0.02,
            "n_degenerate_replicates": 0,
            "sig": [hit],
        }
    }
    payload["provenance"]["primary_multiplicity"] = "bootstrap_minp"

    record = audit_result(_write(tmp_path / "formal_pair_omnib.json", payload))
    codes = {flag.code for flag in record.flags}
    assert record.discovery_count == 1
    assert "OMNIB_FWER_INCONSISTENT" not in codes
    assert "OMNIB_FWER_DECISION_MISSING" not in codes


def test_audit_uses_canonical_omnib_family_consistency_guard(tmp_path):
    payload = _omnib_payload(n_sig=2)
    primary = payload["results"]["INT"]
    hit = primary["top"][0] | {
        "hypothesis_id": "edge:AB|geneA|geneD",
        "p_interaction": 1e-7,
        "p_adjusted_bootstrap_minp": 0.02,
    }
    primary["bootstrap_B"] = 2000
    primary["model_diagnostics"] = {
        "bootstrap_fwer": {
            "alpha": 0.05,
            "method": "parametric_bootstrap_minp_plus_one",
            "inferential": True,
            "formal_discovery_layer": True,
            "family_id": "edge",
            "family_scope": "primary_only",
            "declared_hypothesis_unit": "edge",
            "calibrated_layers": ["edge"],
            "n_hypotheses": 1,
            "hypothesis_ids": [hit["hypothesis_id"]],
            "observed_p": [hit["p_interaction"]],
            "adjusted_p": [hit["p_adjusted_bootstrap_minp"]],
            "threshold": 1e-6,
            "rejected": True,
            "n_rejected": 1,
            "empirical_p": 0.02,
            "rejected_hypothesis_ids": [hit["hypothesis_id"]],
            "sig": [hit],
        }
    }
    payload["provenance"]["primary_multiplicity"] = "bootstrap_minp"

    record = audit_result(_write(tmp_path / "canonical_edge.json", payload))
    assert "OMNIB_FWER_TOPLEVEL_COUNT_MISMATCH" in {
        flag.code for flag in record.flags}


def test_audit_pairwise_omnib_refuses_low_resolution_formal_claim(tmp_path):
    payload = _omnib_payload(n_sig=1)
    primary = payload["results"]["INT"]
    primary["bootstrap_B"] = 99
    primary["model_diagnostics"] = {
        "bootstrap_fwer": {
            "alpha": 0.05,
            "method": "parametric_bootstrap_minp_plus_one",
            "inferential": True,
            "rejected": True,
            "n_rejected": 1,
            "empirical_p": 0.01,
            "sig": primary["top"],
        }
    }
    payload["provenance"]["primary_multiplicity"] = "bootstrap_minp"

    record = audit_result(_write(tmp_path / "low_b_pair_omnib.json", payload))
    codes = {flag.code for flag in record.flags}
    assert record.discovery_count is None
    assert "OMNIB_BOOTSTRAP_MONTE_CARLO_RESOLUTION" in codes


def _burden_permutation_payload(*, perm_B=2000, inferential=True, n_rejected=1):
    hit = {
        "pair": ["geneA", "geneC"],
        "p": 1e-7,
        "p_adjusted_permutation_minp": 0.02,
    }
    fwer = {
        "alpha": 0.05,
        "method": "freedman_lane_permutation_minp_plus_one",
        "inferential": inferential,
        "rejected": bool(n_rejected),
        "n_rejected": n_rejected,
        "empirical_p": 0.02 if n_rejected else 0.2,
        "n_degenerate_replicates": 0,
        "sig": [hit] if n_rejected else [],
    }
    return {
        "tool": "homoeogwas", "command": "interact", "mode": "pairwise",
        "trait": "flowering",
        "provenance": {
            "primary_transform": "INT", "statistic": "burden",
            "calibration_method": "permutation",
            "primary_multiplicity": "permutation_minp",
        },
        "results": {"INT": {
            "statistic": "burden", "calibration_method": "permutation",
            "n": 926, "G": 17404, "n_planned": 17404, "n_valid": 17404,
            "n_unestimable": 0,
            "n_sig": n_rejected if inferential else None,
            "top": [hit], "lambda_gc_obs": 0.98,
            "permutation": {"status": "completed", "B_requested": perm_B,
                            "n_used": perm_B, "n_degenerate": 0},
            "model_diagnostics": {"permutation_fwer": fwer},
        }},
    }


def test_audit_burden_accepts_consistent_formal_permutation_minp(tmp_path):
    payload = _burden_permutation_payload()
    record = audit_result(_write(tmp_path / "formal_burden.json", payload))
    codes = {flag.code for flag in record.flags}
    assert record.discovery_count == 1
    assert record.status == "INTERNAL_DISCOVERY_REPLICATION_REQUIRED"
    assert "BURDEN_FWER_INCONSISTENT" not in codes
    assert "BURDEN_FWER_DECISION_MISSING" not in codes


def test_audit_burden_refuses_low_resolution_formal_permutation_minp(tmp_path):
    payload = _burden_permutation_payload(perm_B=99)
    record = audit_result(_write(tmp_path / "low_b_burden.json", payload))
    codes = {flag.code for flag in record.flags}
    assert record.discovery_count is None
    assert "BURDEN_PERMUTATION_MONTE_CARLO_RESOLUTION" in codes


def test_audit_triad3_records_hierarchy_and_bootstrap(tmp_path):
    payload = {
        "tool": "homoeogwas",
        "command": "interact",
        "mode": "triad",
        "trait": "flowering",
        "provenance": {
            "primary_transform": "INT",
            "statistic": "triad3",
            "calibration_method": "bootstrap",
        },
        "results": {
            "INT": {
                "statistic": "triad3",
                "calibration_method": "bootstrap",
                "bootstrap_B": 999,
                "G": 100,
                "n_planned": 100,
                "n_valid": 100,
                "n_unestimable": 0,
                "n_sig": 0,
                "analytic_screen_n": 1,
                "lambda_gc_obs": 1.0,
                "top": [],
                "model_diagnostics": {
                    "tested_term": "A:B:D",
                    "target_residual_ratio_min": 0.2,
                    "bootstrap_fwer": {
                        "alpha": 0.05,
                        "method": "parametric_bootstrap_minp_plus_one",
                        "inferential": True,
                        "rejected": False,
                        "n_rejected": 0,
                        "empirical_p": 0.2,
                        "n_degenerate_replicates": 0,
                        "sig": [],
                    },
                },
            }
        },
    }
    record = audit_result(_write(tmp_path / "interact_triad3.json", payload))
    assert record.calibration == "triad3+bootstrap(B=999)"
    assert "TRIAD3_MODEL_NOT_RECORDED" not in {
        flag.code for flag in record.flags}
    assert record.discovery_count == 0
    assert "TRIAD3_ANALYTIC_ONLY_CANDIDATE" in {
        flag.code for flag in record.flags}


def test_audit_triad3_accepts_internally_consistent_positive_fwer(tmp_path):
    hit = {
        "triad": ["gA", "gB", "gD"],
        "p": 1e-6,
        "p_adjusted_bootstrap_minp": 0.02,
        "target_residual_ratio": 0.4,
        "target_information_max_fraction": 0.03,
        "target_information_effective_n": 80.0,
    }
    payload = {
        "tool": "homoeogwas",
        "command": "interact",
        "mode": "triad",
        "subgenomes": ["A", "B", "D"],
        "trait": "flowering",
        "provenance": {
            "primary_transform": "INT",
            "statistic": "triad3",
            "calibration_method": "bootstrap",
        },
        "results": {
            "INT": {
                "statistic": "triad3",
                "bootstrap_B": 999,
                "G": 100,
                "n_valid": 100,
                "n_sig": 1,
                "top": [hit],
                "model_diagnostics": {
                    "tested_term": "A:B:D",
                    "bootstrap_fwer": {
                        "alpha": 0.05,
                        "method": "parametric_bootstrap_minp_plus_one",
                        "inferential": True,
                        "rejected": True,
                        "n_rejected": 1,
                        "empirical_p": 0.02,
                        "n_degenerate_replicates": 0,
                        "sig": [hit],
                    },
                },
            }
        },
    }
    record = audit_result(_write(tmp_path / "positive_triad3.json", payload))
    assert record.discovery_count == 1
    assert record.status == "INTERNAL_DISCOVERY_REPLICATION_REQUIRED"
    assert "TRIAD3_FWER_INCONSISTENT" not in {
        flag.code for flag in record.flags}
    assert "TRIAD3_DISCOVERY_SPARSE_SUPPORT" not in {
        flag.code for flag in record.flags}


def test_audit_triad3_flags_sparse_support_in_a_rejected_hit(tmp_path):
    hit = {
        "triad": ["gA", "gB", "gD"],
        "p": 1e-6,
        "p_adjusted_bootstrap_minp": 0.02,
        "target_residual_ratio": 0.3,
        "target_information_max_fraction": 0.48,
        "target_information_effective_n": 2.5,
    }
    payload = {
        "tool": "homoeogwas",
        "command": "interact",
        "mode": "triad",
        "subgenomes": ["A", "B", "D"],
        "trait": "flowering",
        "provenance": {
            "primary_transform": "INT",
            "statistic": "triad3",
        },
        "results": {
            "INT": {
                "statistic": "triad3",
                "bootstrap_B": 999,
                "G": 100,
                "n_valid": 100,
                "n_sig": 1,
                "model_diagnostics": {
                    "tested_term": "A:B:D",
                    "bootstrap_fwer": {
                        "alpha": 0.05,
                        "method": "parametric_bootstrap_minp_plus_one",
                        "inferential": True,
                        "rejected": True,
                        "n_rejected": 1,
                        "empirical_p": 0.02,
                        "n_degenerate_replicates": 0,
                        "sig": [hit],
                    },
                },
            }
        },
    }
    record = audit_result(_write(tmp_path / "sparse_triad3.json", payload))
    assert "TRIAD3_DISCOVERY_SPARSE_SUPPORT" in {
        flag.code for flag in record.flags}


def test_audit_triad3_rejects_inconsistent_positive_fwer(tmp_path):
    payload = {
        "tool": "homoeogwas",
        "command": "interact",
        "mode": "triad",
        "subgenomes": ["A", "B", "D"],
        "trait": "flowering",
        "provenance": {
            "primary_transform": "INT",
            "statistic": "triad3",
        },
        "results": {
            "INT": {
                "statistic": "triad3",
                "bootstrap_B": 999,
                "G": 100,
                "n_valid": 100,
                "n_sig": 3,
                "model_diagnostics": {
                    "tested_term": "A:B:D",
                    "bootstrap_fwer": {
                        "alpha": 0.05,
                        "method": "parametric_bootstrap_minp_plus_one",
                        "inferential": True,
                        "rejected": False,
                        "n_rejected": 3,
                        "empirical_p": 0.2,
                    },
                },
            }
        },
    }
    record = audit_result(_write(tmp_path / "bad_triad3.json", payload))
    assert record.discovery_count is None
    assert "TRIAD3_FWER_INCONSISTENT" in {
        flag.code for flag in record.flags}


def test_audit_triad3_qa_only_is_not_a_missing_formal_decision(tmp_path):
    payload = {
        "tool": "homoeogwas",
        "command": "interact",
        "mode": "triad",
        "subgenomes": ["A", "B", "D"],
        "trait": "flowering",
        "provenance": {
            "primary_transform": "INT",
            "statistic": "triad3",
        },
        "results": {
            "INT": {
                "statistic": "triad3",
                "bootstrap_B": 99,
                "G": 100,
                "n_valid": 100,
                "n_sig": None,
                "model_diagnostics": {
                    "tested_term": "A:B:D",
                    "bootstrap_fwer": {
                        "alpha": 0.05,
                        "method": "parametric_bootstrap_minp_plus_one",
                        "inferential": False,
                        "rejected": None,
                        "n_rejected": None,
                        "empirical_p": 0.02,
                        "n_degenerate_replicates": 0,
                        "sig": None,
                    },
                },
            }
        },
    }
    record = audit_result(_write(tmp_path / "qa_triad3.json", payload))
    codes = {flag.code for flag in record.flags}
    assert record.discovery_count is None
    assert "TRIAD3_QA_ONLY" in codes
    assert "TRIAD3_FWER_DECISION_MISSING" not in codes


def test_audit_triad3_never_reconstructs_discovery_from_old_empirical_p(tmp_path):
    payload = {
        "tool": "homoeogwas",
        "command": "interact",
        "mode": "triad",
        "subgenomes": ["A", "B", "C"],
        "trait": "flowering",
        "provenance": {
            "primary_transform": "INT",
            "statistic": "triad3",
            "calibration_method": "bootstrap",
        },
        "results": {
            "INT": {
                "statistic": "triad3",
                "bootstrap_B": 200,
                "minp_boot_emp": 0.01,
                "n_sig": 2,
                "G": 100,
                "n_valid": 100,
                "model_diagnostics": {
                    "tested_term": "A:B:C",
                },
            }
        },
    }
    record = audit_result(_write(tmp_path / "old_triad3.json", payload))
    assert record.discovery_count is None
    assert "TRIAD3_FWER_DECISION_MISSING" in {
        flag.code for flag in record.flags}


def test_audit_fit_does_not_treat_point_pve_as_uncertainty(tmp_path):
    path = _write(tmp_path / "summary_height.json", {
        "trait": "height",
        "n_samples": 500,
        "reml": {"pve": {"A": 0.2, "B": 0.3, "D": 0.1, "e": 0.4}},
        "lambda_gc": {"all": 1.02},
        "acceptance_all_passed": True,
        "acceptance": [],
    })
    record = audit_result(path)
    assert record.command == "fit"
    assert record.status == "INTERNAL_RESULT_REVIEW_REQUIRED"
    assert "PVE_UNCERTAINTY_NOT_ASSESSED" in {
        flag.code for flag in record.flags}


def test_run_audit_writes_json_tsv_and_markdown(tmp_path):
    _write(tmp_path / "interact_flowering.json", _omnib_payload())
    out = tmp_path / "review"
    result = run_audit(tmp_path, out)
    assert result["overall_status"] == "INTERNAL_DISCOVERY_REPLICATION_REQUIRED"
    for name in ("homoeogwas_audit.json", "homoeogwas_audit.tsv",
                 "homoeogwas_audit.md"):
        assert (out / name).exists()
    assert main(["audit", str(tmp_path), "-o", str(tmp_path / "cli_audit")]) == 0
