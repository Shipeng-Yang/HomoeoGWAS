"""v2.1.3 item 9: report_card JSON written next to fit/interact results."""
import json

from homoeogwas.report_card import build_report_card, write_report_card
from tests.test_audit import _canonical_group_omnib_payload, _write


def _fit_payload(**extra):
    return {"trait": "height", "n_samples": 500,
            "reml": {"pve": {"A": 0.2, "B": 0.3, "e": 0.5}, "pve_ci": {"A": [0.1, 0.3]}},
            "lambda_gc": {"all": 1.4}, "acceptance_all_passed": True, "acceptance": [], **extra}


def test_report_card_for_fit_lists_numbers_flags_and_must_state(tmp_path):
    path = _write(tmp_path / "summary_height.json", _fit_payload())
    card = build_report_card(path)
    assert card["command"] == "fit" and card["status"] == "INTERNAL_RESULT_REVIEW_REQUIRED"
    assert card["numbers"]["n"] == 500 and card["numbers"]["pve"]["B"] == 0.3
    assert card["numbers"]["pve_ci"]["A"] == [0.1, 0.3] and card["numbers"]["lambda_gc"] == 1.4
    codes = {f["code"]: f["severity"] for f in card["flags"]}
    assert codes["LAMBDA_OUTSIDE_SCREEN"] == "review"
    assert any("lambda_GC" in s for s in card["must_state"])


def test_report_card_for_interact_quotes_familywise_decision(tmp_path):
    path = _write(tmp_path / "interact_height.json", _canonical_group_omnib_payload())
    card = build_report_card(path)
    assert card["command"] == "interact" and card["status"] == "NO_FAMILYWISE_DISCOVERY"
    fw = card["numbers"]["familywise"]
    assert {"threshold", "empirical_p", "rejected", "rejected_hypothesis_ids"} <= set(fw)
    assert card["numbers"]["discovery_count"] == 0
    assert card["must_state"]                          # evidence boundary statements


def test_write_report_card_names_file_after_result(tmp_path):
    path = _write(tmp_path / "interact_height.json", _canonical_group_omnib_payload())
    out = write_report_card(path)
    assert out.name == "report_card_interact_height.json"
    assert json.loads(out.read_text())["source"].endswith("interact_height.json")


def test_report_card_is_strict_json_with_nan_numbers(tmp_path):
    path = _write(tmp_path / "summary_height.json", _fit_payload(lambda_gc={"all": float("nan")}))
    out = write_report_card(path)
    text = out.read_text()
    assert "NaN" not in text and "Infinity" not in text
    json.loads(text)
