"""Report card: the numbers to quote, the audit flags with severity and the statements a
report must carry, from one fit or interact result JSON (written next to it as
``report_card_<result stem>.json``). It summarises; it computes no new statistics."""
from __future__ import annotations

from pathlib import Path

from .audit import _load_json, _primary_interact, audit_result
from .jsonutil import dumps_strict, json_safe

SCHEMA = "homoeogwas-report-card-v1"


def build_report_card(result_path) -> dict:
    path = Path(result_path)
    record = audit_result(path)
    payload = _load_json(path)
    numbers: dict = {"n": record.n, "n_planned": record.n_planned, "n_valid": record.n_valid,
                     "discovery_count": record.discovery_count, "lambda_gc": record.lambda_gc,
                     "calibration": record.calibration}
    if record.command == "fit":
        reml = payload.get("reml") or {}
        numbers["pve"] = reml.get("pve")
        numbers["pve_ci"] = reml.get("pve_ci")
    elif record.command == "interact":
        primary, transform = _primary_interact(payload)
        diag = (primary or {}).get("model_diagnostics") or {}
        fw = diag.get("bootstrap_fwer") or {}
        numbers["primary_transform"] = transform
        numbers["familywise"] = {k: fw.get(k) for k in ("method", "alpha", "threshold", "empirical_p",
                                                          "rejected", "n_rejected")}
        numbers["familywise"]["rejected_hypothesis_ids"] = (fw.get("rejected_hypothesis_ids") or [])[:20]
        numbers["null_variance_model"] = (diag.get("null_variance_model") or {}).get("model")
    flags = [{"code": f.code, "severity": f.severity, "message": f.message} for f in record.flags]
    must_state = [f.message for f in record.flags if f.severity in ("review", "error")]
    must_state += list(record.evidence_boundary)
    return {"schema": SCHEMA, "source": str(path), "command": record.command,
            "trait": record.trait, "status": record.status, "numbers": numbers,
            "flags": flags, "must_state": must_state, "top": list(record.top[:10])}


def write_report_card(result_path) -> Path:
    path = Path(result_path)
    out = path.with_name(f"report_card_{path.stem}.json")
    out.write_text(dumps_strict(json_safe(build_report_card(path)), indent=2))
    return out


def try_write_report_card(result_path) -> None:
    """Write the card after a run; a failure here never fails the run, it is printed."""
    try:
        out = write_report_card(result_path)
        print(f"  report card -> {out}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"  WARN: report card not written for {result_path}: {exc}", flush=True)
