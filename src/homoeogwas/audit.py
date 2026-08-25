"""Evidence-aware audit of finished HomoeoGWAS result artifacts.

The audit deliberately separates computational validity, statistical discovery,
biological interpretation and replication.  It never upgrades an internally
significant result to a causal or replicated claim.
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np

from .jsonutil import dump_strict


@dataclass
class AuditFlag:
    code: str
    severity: str
    message: str


@dataclass
class AuditRecord:
    source: str
    command: str
    trait: str | None
    mode: str | None
    statistic: str | None
    status: str
    discovery_count: int | None
    n: int | None
    n_planned: int | None
    n_valid: int | None
    lambda_gc: float | None
    calibration: str | None
    replication_status: str
    flags: list[AuditFlag] = field(default_factory=list)
    top: list[dict] = field(default_factory=list)
    evidence_boundary: list[str] = field(default_factory=list)


def _finite(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _integer(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _strict_integer(value: Any, *, minimum: int = 0) -> int | None:
    """Return a serialized integer count without coercing floats/strings/bools."""
    if isinstance(value, int) and not isinstance(value, bool) and value >= minimum:
        return value
    return None


def _probability(value: Any, *, nullable: bool = False) -> bool:
    """Whether value is a finite JSON probability, explicitly excluding bool."""
    if value is None:
        return nullable
    return (
        isinstance(value, Real)
        and not isinstance(value, (bool, np.bool_))
        and np.isfinite(value)
        and 0.0 <= float(value) <= 1.0
    )


def _add(flags: list[AuditFlag], code: str, severity: str, message: str) -> None:
    flags.append(AuditFlag(code=code, severity=severity, message=message))


def _find_result_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"audit input does not exist: {path}")
    patterns = ("summary_*.json", "interact_*.json", "predict_*.json")
    found: list[Path] = []
    for pattern in patterns:
        found.extend(path.glob(pattern))
    return sorted({p.resolve() for p in found if not p.name.startswith("audit_")})


def _load_json(path: Path) -> dict:
    try:
        with path.open() as fh:
            value = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read result JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"result JSON must contain an object: {path}")
    return value


def _fit_record(path: Path, payload: dict) -> AuditRecord:
    flags: list[AuditFlag] = []
    reml = payload.get("reml") or {}
    acceptance = payload.get("acceptance") or []
    passed = payload.get("acceptance_all_passed")
    trait = payload.get("trait") or payload.get("phenotype", {}).get("trait")
    n = _integer(payload.get("n") or payload.get("n_samples"))

    lam_obj = payload.get("lambda_gc")
    if isinstance(lam_obj, dict):
        lam = _finite(lam_obj.get("all") or lam_obj.get("genomewide"))
    else:
        lam = _finite(lam_obj)
    if lam is not None and not 0.8 <= lam <= 1.2:
        _add(flags, "LAMBDA_OUTSIDE_SCREEN", "review",
             f"Genome-wide lambda_GC={lam:.3f} is outside the advisory 0.8-1.2 screen.")

    if passed is False or any(item.get("passed") is False for item in acceptance):
        _add(flags, "ACCEPTANCE_FAILED", "error",
             "One or more fit acceptance checks failed; do not interpret the run.")

    pve = reml.get("pve") or {}
    if not pve:
        _add(flags, "PVE_MISSING", "error", "No REML PVE partition was recorded.")
    if not reml.get("pve_uncertainty") and not reml.get("pve_ci"):
        _add(flags, "PVE_UNCERTAINTY_NOT_ASSESSED", "review",
             "Point PVE estimates are present without confidence intervals or bootstrap stability.")
    boundary = reml.get("boundary_components") or []
    if boundary:
        _add(flags, "BOUNDARY_COMPONENT", "review",
             f"REML placed {len(boundary)} component(s) on the boundary: {boundary}.")

    if any(flag.severity == "error" for flag in flags):
        status = "ANALYSIS_INVALID"
    elif any(flag.severity == "review" for flag in flags):
        status = "INTERNAL_RESULT_REVIEW_REQUIRED"
    else:
        status = "INTERNAL_RESULT_COMPLETE"
    return AuditRecord(
        source=str(path), command="fit", trait=str(trait) if trait is not None else None,
        mode="subgenome_lmm", statistic="multi_kernel_REML+LMM",
        status=status, discovery_count=None, n=n, n_planned=None, n_valid=None,
        lambda_gc=lam, calibration="model_acceptance_checks",
        replication_status="NOT_ASSESSED", flags=flags,
        evidence_boundary=[
            "PVE is a panel- and model-specific variance partition, not a causal allocation.",
            "GWAS hits require locus-level robustness and independent or external validation.",
        ],
    )


def _primary_interact(payload: dict) -> tuple[dict, str]:
    results = payload.get("results") or {}
    provenance = payload.get("provenance") or {}
    requested = str(provenance.get("primary_transform", "INT"))
    for key in (requested, requested.upper(), requested.lower(), "INT", "raw"):
        value = results.get(key)
        if isinstance(value, dict):
            return value, key
    return {}, requested


def _group_fields(primary: dict) -> tuple[int | None, int | None, int | None, int | None, list]:
    group = primary.get("group_omnibus")
    if not isinstance(group, dict):
        return None, None, None, None, []
    return (
        _integer(group.get("n_sig")),
        _integer(group.get("n_planned")),
        _integer(group.get("n_valid")),
        _integer(group.get("n_unestimable")),
        list(group.get("top") or []),
    )


def _sha256_hex(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value.lower())
    )


def _canonical_omnib_contract_codes(
        payload: dict, provenance: dict, primary: dict) -> tuple[str, ...]:
    """Validate canonical family identity beyond the pure FWER serializer."""
    codes: list[str] = []
    diagnostics = primary.get("model_diagnostics") or {}
    fwer = diagnostics.get("bootstrap_fwer") or {}
    family = diagnostics.get("family_provenance") or {}
    required_hashes = ("group_family_sha256", "edge_family_sha256")
    if not all(
        _sha256_hex(family.get(key)) and _sha256_hex(provenance.get(key))
        for key in required_hashes
    ):
        codes.append("OMNIB_FAMILY_PROVENANCE_MISSING")
    elif any(family.get(key) != provenance.get(key) for key in required_hashes):
        codes.append("OMNIB_FAMILY_PROVENANCE_MISMATCH")

    raw_counts = (
        family.get("n_groups_raw"), family.get("n_unique_edges"),
        provenance.get("n_groups_raw"), provenance.get("n_unique_edges"),
    )
    n_groups, n_edges, top_groups, top_edges = (
        _strict_integer(value, minimum=1) for value in raw_counts)
    if any(value is None or value < 1 for value in (
            n_groups, n_edges, top_groups, top_edges)):
        if any(value is None for value in raw_counts):
            codes.append("OMNIB_FAMILY_COUNTS_MISSING")
        codes.append("OMNIB_FAMILY_COUNTS_INVALID")

    unit = provenance.get("hypothesis_unit")
    if unit not in {"edge", "group"}:
        codes.append("OMNIB_PRIMARY_UNIT_MISSING")
    if fwer.get("declared_hypothesis_unit") != unit:
        codes.append("OMNIB_FWER_PRIMARY_UNIT_MISMATCH")
    scope = provenance.get("family_scope")
    if scope not in {"primary_only", "joint"} or fwer.get("family_scope") != scope:
        codes.append("OMNIB_FWER_FAMILY_SCOPE_MISMATCH")

    if scope == "joint":
        expected = (
            n_groups + n_edges
            if n_groups is not None and n_edges is not None else None)
    elif unit == "edge":
        expected = n_edges
    else:
        expected = n_groups
    counts_agree = (
        n_groups == top_groups
        and n_edges == top_edges
        and expected is not None
        and _strict_integer(fwer.get("n_hypotheses"), minimum=1) == expected
        and _strict_integer(primary.get("G"), minimum=1) == expected
        and _strict_integer(primary.get("n_planned"), minimum=1) == expected
    )
    if not counts_agree:
        codes.append("OMNIB_FWER_HYPOTHESIS_COUNT_MISMATCH")

    if fwer.get("method") != "parametric_bootstrap_minp_plus_one":
        codes.append("OMNIB_FWER_METHOD_MISMATCH")
    if provenance.get("primary_multiplicity") != "bootstrap_minp":
        codes.append("OMNIB_FWER_PRIMARY_MULTIPLICITY_MISMATCH")
    if payload.get("mode") != "group" or provenance.get("mode") != "group":
        codes.append("OMNIB_CANONICAL_MODE_MISMATCH")
    if provenance.get("primary_transform") != "INT":
        codes.append("OMNIB_PRIMARY_TRANSFORM_MISMATCH")
    return tuple(dict.fromkeys(codes))


def _canonical_omnib_serialization_codes(primary: dict) -> tuple[str, ...]:
    """Validate untrusted scalar/vector types before the shared math checker."""
    codes: list[str] = []
    diagnostics = primary.get("model_diagnostics") or {}
    fwer = diagnostics.get("bootstrap_fwer") or {}
    if not isinstance(fwer, dict):
        return ("OMNIB_FWER_OBJECT_MISSING",)

    primary_b = _strict_integer(primary.get("bootstrap_B"), minimum=1)
    fwer_b = _strict_integer(fwer.get("B"), minimum=1)
    if primary_b is None or fwer_b is None or primary_b != fwer_b:
        codes.append("OMNIB_FWER_BOOTSTRAP_B_INVALID")

    primary_counts = ("G", "n_planned", "n_valid", "n_unestimable")
    if any(_strict_integer(primary.get(key), minimum=(1 if key in {
            "G", "n_planned"} else 0)) is None for key in primary_counts):
        codes.append("OMNIB_FWER_COUNTS_INVALID")
    if fwer.get("inferential") is True and _strict_integer(
            primary.get("n_sig"), minimum=0) is None:
        codes.append("OMNIB_FWER_COUNTS_INVALID")
    inferential = fwer.get("inferential")
    count_fields = [
        ("n_hypotheses", 1), ("n_calibrated", 0), ("n_unestimable", 0)]
    if inferential is True:
        count_fields.append(("n_rejected", 0))
    for key, minimum in count_fields:
        if _strict_integer(fwer.get(key), minimum=minimum) is None:
            codes.append("OMNIB_FWER_COUNTS_INVALID")
            break

    ids = fwer.get("hypothesis_ids")
    ids_valid = (
        isinstance(ids, list)
        and all(isinstance(value, str) for value in ids)
        and len(ids) == len(set(ids))
    )
    if not ids_valid:
        codes.append("OMNIB_FWER_HYPOTHESIS_IDS_INVALID")

    observed = fwer.get("observed_p")
    adjusted = fwer.get("adjusted_p")
    if not isinstance(observed, list) or not (
            ids_valid and len(observed) == len(ids)):
        codes.append("OMNIB_FWER_VECTOR_LENGTH_MISMATCH")
    elif not all(_probability(value, nullable=True) for value in observed):
        codes.append("OMNIB_FWER_OBSERVED_P_INVALID")
    if not isinstance(adjusted, list) or not (
            ids_valid and len(adjusted) == len(ids)):
        codes.append("OMNIB_FWER_VECTOR_LENGTH_MISMATCH")
    elif not all(_probability(value, nullable=True) for value in adjusted):
        codes.append("OMNIB_FWER_ADJUSTED_P_INVALID")

    if not _probability(fwer.get("alpha"), nullable=False):
        codes.append("OMNIB_FWER_PROBABILITY_SCALAR_INVALID")
    if inferential is True:
        if any(not _probability(fwer.get(key), nullable=False)
               for key in ("empirical_p", "threshold")):
            codes.append("OMNIB_FWER_PROBABILITY_SCALAR_INVALID")
    elif inferential is False:
        qa = fwer.get("qa_diagnostics")
        qa_adjusted = qa.get("adjusted_p") if isinstance(qa, dict) else None
        qa_valid = (
            isinstance(qa, dict)
            and qa.get("role") == "noninferential_do_not_threshold"
            and _probability(qa.get("empirical_p"), nullable=False)
            and _probability(qa.get("threshold"), nullable=False)
            and isinstance(qa_adjusted, list)
            and ids_valid
            and len(qa_adjusted) == len(ids)
            and all(_probability(value, nullable=True) for value in qa_adjusted)
        )
        if not qa_valid:
            codes.append("OMNIB_FWER_QA_DIAGNOSTICS_INVALID")
    else:
        codes.append("OMNIB_FWER_AUTHORITY_MODE_INVALID")
    return tuple(dict.fromkeys(codes))


def _nonprimary_authority_codes(payload: dict, primary_key: str) -> tuple[str, ...]:
    """Reject any formal discovery authority outside the selected transform."""
    results = payload.get("results") or {}
    if not isinstance(results, dict):
        return ()
    for key, result in results.items():
        if key == primary_key or not isinstance(result, dict):
            continue
        fwer = ((result.get("model_diagnostics") or {}).get(
            "bootstrap_fwer") or {})
        top_authority = any(result.get(field) is not None for field in (
            "n_sig", "sig", "minp_boot_rejected"))
        fwer_authority = not isinstance(fwer, dict) or any((
            fwer.get("inferential") is not False,
            fwer.get("formal_discovery_layer") is not False,
            *(fwer.get(field) is not None for field in (
                "rejected", "n_rejected", "sig", "rejected_indices",
                "rejected_hypothesis_ids")),
        ))
        if top_authority or fwer_authority:
            return ("OMNIB_FWER_UNCALIBRATED_SECOND_PRIMARY_LAYER",)
    return ()


def _hit_identity(hit: dict) -> Any:
    for key in ("hypothesis_id", "group_id", "edge_id", "pair", "triad"):
        if hit.get(key) is not None:
            return hit[key]
    return "UNIDENTIFIED"


def _hit_component(hit: dict) -> tuple[Any, str | None]:
    if hit.get("driving_component") is not None:
        return hit["driving_component"], "driving component"
    if hit.get("smallest_component") is not None:
        return hit["smallest_component"], "smallest component"
    return None, None


def _interact_record(path: Path, payload: dict) -> AuditRecord:
    flags: list[AuditFlag] = []
    provenance = payload.get("provenance") or {}
    primary, primary_key = _primary_interact(payload)
    statistic = str(
        primary.get("statistic") or provenance.get("statistic") or "burden")
    calibration_method = str(
        primary.get("calibration_method")
        or provenance.get("calibration_method")
        or ("bootstrap" if statistic.lower() in {"omnib", "triad3"}
            else "permutation"))

    n_sig = _integer(primary.get("n_sig"))
    n_planned = _integer(primary.get("n_planned") or primary.get("G"))
    n_valid = _integer(primary.get("n_valid"))
    n_unestimable = _integer(primary.get("n_unestimable"))
    top = list(primary.get("top") or [])
    group_n_sig, group_planned, group_valid, group_unestimable, group_top = (
        _group_fields(primary))
    if group_n_sig is not None:
        n_sig, n_planned, n_valid, n_unestimable, top = (
            group_n_sig, group_planned, group_valid, group_unestimable, group_top)

    if not primary:
        _add(flags, "PRIMARY_RESULT_MISSING", "error",
             f"No result object was found for primary transform {primary_key!r}.")
    if n_valid == 0:
        _add(flags, "NOT_ESTIMABLE", "error",
             "The declared primary family contains no estimable test.")
    if n_planned is not None and n_valid is None:
        _add(flags, "ESTIMABILITY_COUNTS_NOT_RECORDED", "review",
             "The artifact records the planned family but not how many tests were estimable.")
    if n_planned and n_valid is not None:
        fraction = n_valid / n_planned
        if fraction < 0.95:
            _add(flags, "LOW_VALID_FRACTION", "review",
                 f"Only {n_valid}/{n_planned} ({fraction:.1%}) planned tests were estimable.")
    if n_unestimable and n_unestimable > 0:
        _add(flags, "UNESTIMABLE_UNITS_RECORDED", "info",
             f"{n_unestimable} planned test(s) were design-unestimable and retained in the audit.")

    lam = _finite(primary.get("lambda_gc_obs"))
    if lam is not None and not 0.8 <= lam <= 1.2:
        _add(flags, "LAMBDA_OUTSIDE_SCREEN", "review",
             f"Primary lambda_GC={lam:.3f} is outside the advisory 0.8-1.2 screen.")

    if statistic.lower() == "omnib":
        from .interact import PAIRWISE_OMNIB_FORMAL_BOOTSTRAP_MIN_B
        from .omnib_family import omnib_fwer_consistency_flags

        formal_fwer = (
            (primary.get("model_diagnostics") or {}).get("bootstrap_fwer") or {})
        is_canonical_group = payload.get("mode") == "group"
        serialization_codes: tuple[str, ...] = ()
        cross_layer_codes: tuple[str, ...] = ()
        if is_canonical_group:
            serialization_codes = _canonical_omnib_serialization_codes(primary)
            cross_layer_codes = _nonprimary_authority_codes(payload, primary_key)
            for code in (*serialization_codes, *cross_layer_codes):
                _add(
                    flags, code, "error",
                    "The canonical omniB artifact has invalid serialized family "
                    "values or a formal discovery authority outside its INT primary layer.")
        consistency_codes: tuple[str, ...] = ()
        if (
            formal_fwer.get("formal_discovery_layer") is True
            or is_canonical_group
        ) and not serialization_codes:
            try:
                consistency_codes = omnib_fwer_consistency_flags(primary)
            except (TypeError, ValueError, OverflowError):
                consistency_codes = ("OMNIB_FWER_SERIALIZATION_INVALID",)
            for code in consistency_codes:
                _add(
                    flags, code, "error" if is_canonical_group else "review",
                    "The canonical omniB result has inconsistent family, adjusted-p, "
                    "threshold or serialized rejection metadata.")
        contract_codes: tuple[str, ...] = ()
        if is_canonical_group:
            contract_codes = _canonical_omnib_contract_codes(
                payload, provenance, primary)
            for code in contract_codes:
                _add(
                    flags, code, "error",
                    "The canonical omniB family provenance, hypothesis count, "
                    "primary unit or bootstrap method is not publication-valid.")

        component_diag = primary.get("component_diagnostics")
        if not component_diag:
            _add(flags, "OMNIB_COMPONENTS_NOT_RECORDED", "review",
                 "omniB component p-values were not recorded; the signal cannot be localized "
                 "to minor-burden, PC1 or kernel-Hadamard evidence from this artifact.")
        for hit in top[:5]:
            driver, _ = _hit_component(hit)
            if driver and driver != "minor_burden":
                _add(flags, "COMPONENT_SPECIFIC_EVIDENCE", "info",
                     f"Top unit {_hit_identity(hit)} is most strongly supported by {driver}; "
                     "do not relabel the omnibus hit as a burden-product interaction.")
        boot_b = (
            _strict_integer(primary.get("bootstrap_B"), minimum=1)
            if is_canonical_group else _integer(primary.get("bootstrap_B")))
        if not boot_b:
            _add(flags, "BOOTSTRAP_NOT_RUN", "review",
                 "omniB kinship-preserving bootstrap calibration was not run.")
        primary_multiplicity = str(
            provenance.get("primary_multiplicity", "bonferroni")).lower()
        if primary_multiplicity == "bootstrap_minp":
            bootstrap_fwer = (
                (primary.get("model_diagnostics") or {})
                .get("bootstrap_fwer") or {})
            calibrated_n = _integer(bootstrap_fwer.get("n_rejected"))
            rejected = bootstrap_fwer.get("rejected")
            inferential = bootstrap_fwer.get("inferential")
            alpha = _finite(bootstrap_fwer.get("alpha"))
            empirical_p = _finite(bootstrap_fwer.get("empirical_p"))
            method_ok = (
                bootstrap_fwer.get("method")
                == "parametric_bootstrap_minp_plus_one")
            decision_ok = (
                isinstance(rejected, bool)
                and calibrated_n is not None
                and rejected == (calibrated_n > 0)
                and alpha == 0.05
                and empirical_p is not None
                and rejected == (empirical_p <= alpha)
                and method_ok)
            authoritative = (
                inferential is True
                and bool(boot_b)
                and boot_b >= PAIRWISE_OMNIB_FORMAL_BOOTSTRAP_MIN_B
                and decision_ok
                and not consistency_codes
                and not contract_codes
                and not serialization_codes
                and not cross_layer_codes)
            if inferential is False:
                _add(flags, "OMNIB_QA_ONLY", "info",
                     "The omniB bootstrap was marked QA-only; no "
                     "formal discovery is emitted.")
                calibrated_n = None
            elif boot_b and boot_b < PAIRWISE_OMNIB_FORMAL_BOOTSTRAP_MIN_B:
                _add(flags, "OMNIB_BOOTSTRAP_MONTE_CARLO_RESOLUTION", "review",
                     f"omniB used B={boot_b}; formal inference requires "
                     f"B>={PAIRWISE_OMNIB_FORMAL_BOOTSTRAP_MIN_B}. Treat this as QA.")
                calibrated_n = None
            elif not decision_ok:
                _add(flags, "OMNIB_FWER_INCONSISTENT", "review",
                     "The omniB bootstrap-FWER method, alpha, empirical p, "
                     "rejection boolean and rejection count are not internally consistent.")
                calibrated_n = None
            if not authoritative and inferential is not False:
                _add(flags, "OMNIB_FWER_DECISION_MISSING", "review",
                     "No authoritative omniB bootstrap-FWER decision was "
                     "recorded; the audit will not reconstruct a discovery from an "
                     "analytic screen or low-resolution bootstrap.")
            if (
                authoritative and n_sig is not None
                and n_sig != calibrated_n
            ):
                _add(flags, "OMNIB_FWER_TOPLEVEL_COUNT_MISMATCH", "review",
                     f"The top-level n_sig={n_sig} differs from the authoritative "
                     f"bootstrap-FWER n_rejected={calibrated_n}; the latter is used.")
            n_degenerate = _integer(
                bootstrap_fwer.get("n_degenerate_replicates")) or 0
            if n_degenerate:
                _add(flags, "OMNIB_BOOTSTRAP_DEGENERATE", "review",
                     f"{n_degenerate}/{boot_b} bootstrap replicate(s) contained a "
                     "non-finite statistic and were handled conservatively.")
            if authoritative:
                for hit in list(bootstrap_fwer.get("sig") or []):
                    if hit.get("p_adjusted_bootstrap_minp") is None:
                        _add(flags, "OMNIB_MINP_ADJUSTED_P_MISSING", "review",
                             f"Rejected unit {hit.get('hypothesis_id') or hit.get('pair')} "
                             "lacks its single-step "
                             "bootstrap min-P adjusted p-value.")
            n_sig = calibrated_n if authoritative else None
        calibration = f"omniB+{calibration_method}(B={boot_b or 0})"
    elif statistic.lower() == "triad3":
        from .interact import TRIAD3_FORMAL_BOOTSTRAP_MIN_B

        model_diag = primary.get("model_diagnostics") or {}
        subgenomes = list(payload.get("subgenomes") or [])
        expected_term = (
            ":".join(str(value) for value in subgenomes)
            if len(subgenomes) == 3 else "A:B:D")
        if model_diag.get("tested_term") != expected_term:
            _add(flags, "TRIAD3_MODEL_NOT_RECORDED", "review",
                 f"The conditional {expected_term} model hierarchy was not recorded.")
        boot_b = _integer(primary.get("bootstrap_B"))
        if not boot_b:
            _add(flags, "BOOTSTRAP_NOT_RUN", "review",
                 "triad3 kinship-preserving bootstrap calibration was not run.")
            n_sig = None
        elif boot_b < 19:
            _add(flags, "BOOTSTRAP_LOW_RESOLUTION", "review",
                 f"triad3 used B={boot_b}; at least 19 null replicates are needed "
                 "for a possible plus-one p <= 0.05.")
            n_sig = None
        if boot_b and boot_b >= 19:
            bootstrap_fwer = model_diag.get("bootstrap_fwer") or {}
            calibrated_n = _integer(bootstrap_fwer.get("n_rejected"))
            rejected = bootstrap_fwer.get("rejected")
            inferential = bootstrap_fwer.get("inferential")
            alpha = _finite(bootstrap_fwer.get("alpha"))
            empirical_p = _finite(bootstrap_fwer.get("empirical_p"))
            method_ok = (
                bootstrap_fwer.get("method")
                == "parametric_bootstrap_minp_plus_one")
            decision_ok = (
                isinstance(rejected, bool)
                and calibrated_n is not None
                and rejected == (calibrated_n > 0)
                and alpha == 0.05
                and empirical_p is not None
                and rejected == (empirical_p <= alpha)
                and method_ok
            )
            authoritative = (
                inferential is True
                and boot_b >= TRIAD3_FORMAL_BOOTSTRAP_MIN_B
                and decision_ok
            )
            analytic_n = (
                _integer(primary.get("analytic_screen_n"))
                or _integer((model_diag.get("analytic_screen") or {}).get(
                    "n_screened"))
                or 0
            )
            if inferential is False:
                _add(flags, "TRIAD3_QA_ONLY", "info",
                     "The bootstrap scan was explicitly marked QA-only; no "
                     "formal discovery is emitted.")
            elif boot_b < TRIAD3_FORMAL_BOOTSTRAP_MIN_B:
                _add(flags, "BOOTSTRAP_MONTE_CARLO_RESOLUTION", "review",
                     f"triad3 used B={boot_b}; formal inference requires "
                     f"B>={TRIAD3_FORMAL_BOOTSTRAP_MIN_B}. Treat this as QA.")
            elif not decision_ok:
                _add(flags, "TRIAD3_FWER_INCONSISTENT", "review",
                     "The bootstrap-FWER method, alpha, empirical p, rejection "
                     "boolean and rejection count are not internally consistent.")
            if not authoritative and inferential is not False:
                _add(flags, "TRIAD3_FWER_DECISION_MISSING", "review",
                     "No authoritative inferential bootstrap-FWER decision "
                     "was recorded; the audit will not reconstruct a discovery "
                     "from an empirical p-value or an analytic screen.")
                calibrated_n = None
            n_degenerate = _integer(
                bootstrap_fwer.get("n_degenerate_replicates")) or 0
            if n_degenerate:
                _add(flags, "TRIAD3_BOOTSTRAP_DEGENERATE", "review",
                     f"{n_degenerate}/{boot_b} bootstrap replicate(s) contained "
                     "a non-finite statistic and were handled conservatively.")
            if analytic_n > 0 and calibrated_n == 0:
                _add(flags, "TRIAD3_ANALYTIC_ONLY_CANDIDATE", "review",
                     f"{analytic_n} triad(s) passed the analytic Bonferroni screen, "
                     "but the bootstrap min-P family did not reject.")
            # For experimental triad3, the bootstrap min-P layer is the
            # calibrated discovery decision; Bonferroni remains a screen.
            n_sig = calibrated_n
        ratio = _finite(model_diag.get("target_residual_ratio_min"))
        if ratio is not None and ratio < 1e-4:
            _add(flags, "TRIAD3_FAMILY_WEAK_IDENTIFICATION", "info",
                 f"The tested family contains an estimable {expected_term} "
                 f"target retaining only {ratio:.2g} of its norm after "
                 "lower-order conditioning.")
        bootstrap_sig = list(
            (model_diag.get("bootstrap_fwer") or {}).get("sig") or [])
        for hit in bootstrap_sig:
            hit_ratio = _finite(hit.get("target_residual_ratio"))
            if hit_ratio is not None and hit_ratio < 1e-4:
                _add(flags, "TRIAD3_DISCOVERY_WEAK_IDENTIFICATION", "review",
                     f"Rejected triad {hit.get('triad')} retains only "
                     f"{hit_ratio:.2g} of its target norm.")
            if hit.get("p_adjusted_bootstrap_minp") is None:
                _add(flags, "TRIAD3_MINP_ADJUSTED_P_MISSING", "review",
                     f"Rejected triad {hit.get('triad')} lacks its single-step "
                     "bootstrap min-P adjusted p-value.")
            max_information = _finite(
                hit.get("target_information_max_fraction"))
            effective_n = _finite(
                hit.get("target_information_effective_n"))
            if (
                (max_information is not None and max_information > 0.10)
                or (effective_n is not None and effective_n < 20)
            ):
                _add(flags, "TRIAD3_DISCOVERY_SPARSE_SUPPORT", "review",
                     f"Rejected triad {hit.get('triad')} has concentrated "
                     "conditional-target information "
                     f"(max sample fraction={max_information}, "
                     f"effective n={effective_n}); case-deletion and "
                     "independent replication are required.")
        calibration = f"triad3+{calibration_method}(B={boot_b or 0})"
    else:
        from .interact import PAIRWISE_BURDEN_FORMAL_PERMUTATION_MIN_B

        perm = primary.get("permutation") or {}
        perm_b = (_integer(perm.get("B_requested") or perm.get("B_used")
                           or perm.get("n_used"))
                  or _integer(provenance.get("perm_B")) or 0)
        if perm_b == 0:
            _add(flags, "RESAMPLING_NOT_RUN", "info",
                 "No empirical resampling was run; analytic familywise inference may still be "
                 "valid when Bonferroni was predeclared.")
        primary_multiplicity = str(
            provenance.get("primary_multiplicity", "bonferroni")).lower()
        if primary_multiplicity == "permutation_minp":
            permutation_fwer = (
                (primary.get("model_diagnostics") or {})
                .get("permutation_fwer") or {})
            calibrated_n = _integer(permutation_fwer.get("n_rejected"))
            rejected = permutation_fwer.get("rejected")
            inferential = permutation_fwer.get("inferential")
            alpha = _finite(permutation_fwer.get("alpha"))
            empirical_p = _finite(permutation_fwer.get("empirical_p"))
            method_ok = (
                permutation_fwer.get("method")
                == "freedman_lane_permutation_minp_plus_one")
            decision_ok = (
                isinstance(rejected, bool)
                and calibrated_n is not None
                and rejected == (calibrated_n > 0)
                and alpha == 0.05
                and empirical_p is not None
                and rejected == (empirical_p <= alpha)
                and method_ok)
            authoritative = (
                inferential is True
                and perm_b >= PAIRWISE_BURDEN_FORMAL_PERMUTATION_MIN_B
                and decision_ok)
            if inferential is False:
                _add(flags, "BURDEN_QA_ONLY", "info",
                     "The burden permutation-minP scan was marked QA-only; no formal "
                     "discovery is emitted.")
                calibrated_n = None
            elif perm_b < PAIRWISE_BURDEN_FORMAL_PERMUTATION_MIN_B:
                _add(flags, "BURDEN_PERMUTATION_MONTE_CARLO_RESOLUTION", "review",
                     f"pairwise burden used B={perm_b}; formal inference requires "
                     f"B>={PAIRWISE_BURDEN_FORMAL_PERMUTATION_MIN_B}. Treat this as QA.")
                calibrated_n = None
            elif not decision_ok:
                _add(flags, "BURDEN_FWER_INCONSISTENT", "review",
                     "The burden permutation-FWER method, alpha, empirical p, rejection "
                     "boolean and rejection count are not internally consistent.")
                calibrated_n = None
            if not authoritative and inferential is not False:
                _add(flags, "BURDEN_FWER_DECISION_MISSING", "review",
                     "No authoritative burden permutation-minP FWER decision was recorded; "
                     "the audit will not reconstruct a discovery from the analytic screen.")
            if authoritative and n_sig is not None and n_sig != calibrated_n:
                _add(flags, "BURDEN_FWER_TOPLEVEL_COUNT_MISMATCH", "review",
                     f"The top-level n_sig={n_sig} differs from the authoritative "
                     f"permutation-FWER n_rejected={calibrated_n}; the latter is used.")
            n_degenerate = _integer(
                permutation_fwer.get("n_degenerate_replicates")) or 0
            if n_degenerate:
                _add(flags, "BURDEN_PERMUTATION_DEGENERATE", "review",
                     f"{n_degenerate}/{perm_b} permutation replicate(s) contained a "
                     "non-finite statistic and were handled conservatively.")
            if authoritative:
                for hit in list(permutation_fwer.get("sig") or []):
                    if hit.get("p_adjusted_permutation_minp") is None:
                        _add(flags, "BURDEN_MINP_ADJUSTED_P_MISSING", "review",
                             f"Rejected pair {hit.get('pair')} lacks its single-step "
                             "permutation min-P adjusted p-value.")
            n_sig = calibrated_n if authoritative else None
        calibration = f"burden+{calibration_method}(B={perm_b})"

    discovery = bool(n_sig and n_sig > 0)
    replication = payload.get("replication") or provenance.get("replication")
    replication_status = "RECORDED" if replication else "NOT_ASSESSED"
    if discovery and not replication:
        _add(flags, "REPLICATION_REQUIRED", "review",
             "An internal familywise discovery is present but no replication manifest is attached.")

    if any(flag.severity == "error" for flag in flags):
        status = "ANALYSIS_INVALID"
    elif discovery and not replication:
        status = "INTERNAL_DISCOVERY_REPLICATION_REQUIRED"
    elif discovery:
        status = "INTERNAL_DISCOVERY_REPLICATION_RECORDED"
    elif any(flag.severity == "review" for flag in flags):
        status = "NO_FAMILYWISE_DISCOVERY_REVIEW_REQUIRED"
    else:
        status = "NO_FAMILYWISE_DISCOVERY"

    evidence_boundary = [
        "A homoeolog interaction association does not establish the causal gene pair, "
        "molecular mechanism or breeding portability.",
        "Independent population/environment validation remains a separate evidence layer.",
    ]
    if statistic.lower() == "omnib":
        unit = provenance.get("hypothesis_unit")
        if payload.get("mode") == "group" and unit == "edge":
            evidence_boundary.insert(
                0,
                "The formal result is encoding-robust omnibus interaction evidence "
                "for a homoeolog pair.")
        elif payload.get("mode") == "group" and unit == "group":
            evidence_boundary.insert(
                0,
                "The formal result is encoding-robust omnibus pairwise interaction "
                "evidence within a homoeolog group.")
        else:
            evidence_boundary.insert(
                0,
                "omniB significance supports a combined interaction test, not "
                "automatically a minor-burden mechanism.")
        evidence_boundary.insert(
            1,
            "This pair-edge omnibus is not a third- or fourth-order causal or "
            "physical interaction mechanism.")
    elif statistic.lower() == "triad3":
        evidence_boundary.insert(
            0,
            "triad3 significance supports statistical third-order non-additivity conditional "
            "on lower-order burden terms; it does not establish a physical A/B/D complex.")
    else:
        evidence_boundary.insert(
            0,
            "Burden-product significance is conditional on the chosen SNP-to-gene mapping, "
            "burden coding, phenotype transform and fitted kinship model.")

    return AuditRecord(
        source=str(path), command="interact", trait=payload.get("trait"),
        mode=payload.get("mode"), statistic=statistic, status=status,
        discovery_count=n_sig, n=_integer(primary.get("n") or provenance.get("n_samples")),
        n_planned=n_planned, n_valid=n_valid, lambda_gc=lam,
        calibration=calibration, replication_status=replication_status,
        flags=flags, top=top[:5], evidence_boundary=evidence_boundary,
    )


def _predict_record(path: Path, payload: dict) -> AuditRecord:
    flags: list[AuditFlag] = []
    trait = payload.get("trait")
    delta = payload.get("delta_vs_tier0") or {}
    if not delta:
        _add(flags, "PAIRED_CV_CONTRAST_MISSING", "review",
             "Prediction output has no paired tier-vs-tier0 contrast.")
    return AuditRecord(
        source=str(path), command="predict", trait=trait, mode="cross_validated_GBLUP",
        statistic="paired_cross_validation", status=(
            "INTERNAL_RESULT_REVIEW_REQUIRED" if flags else "INTERNAL_RESULT_COMPLETE"),
        discovery_count=None, n=_integer(payload.get("n")), n_planned=None, n_valid=None,
        lambda_gc=None, calibration="repeated_cross_validation",
        replication_status="NOT_ASSESSED", flags=flags,
        evidence_boundary=[
            "Cross-validation estimates within-panel prediction, not portability to a new panel "
            "or environment.",
        ],
    )


def audit_result(path: Path) -> AuditRecord:
    payload = _load_json(path)
    command = payload.get("command")
    if command == "interact" or "results" in payload and "provenance" in payload:
        return _interact_record(path, payload)
    if command == "predict" or path.name.startswith("predict_"):
        return _predict_record(path, payload)
    return _fit_record(path, payload)


def _overall_status(records: list[AuditRecord]) -> str:
    statuses = {record.status for record in records}
    if "ANALYSIS_INVALID" in statuses:
        return "ANALYSIS_INVALID"
    if "INTERNAL_DISCOVERY_REPLICATION_REQUIRED" in statuses:
        return "INTERNAL_DISCOVERY_REPLICATION_REQUIRED"
    if any("REVIEW_REQUIRED" in value for value in statuses):
        return "REVIEW_REQUIRED"
    return "AUDIT_COMPLETE"


def _write_tsv(path: Path, records: list[AuditRecord]) -> None:
    header = [
        "source", "command", "trait", "mode", "statistic", "status",
        "discovery_count", "n", "n_planned", "n_valid", "lambda_gc",
        "calibration", "replication_status", "flag_codes",
    ]
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(header)
        for record in records:
            row = asdict(record)
            writer.writerow([
                row.get(key) if row.get(key) is not None else "NA"
                for key in header[:-1]
            ] + [";".join(flag.code for flag in record.flags) or "NONE"])


def _write_markdown(path: Path, overall: str, records: list[AuditRecord]) -> None:
    lines = [
        "# HomoeoGWAS result audit",
        "",
        f"- Overall status: `{overall}`",
        f"- Results inspected: {len(records)}",
        "",
    ]
    for record in records:
        lines.extend([
            f"## {record.command}: {record.trait or Path(record.source).stem}",
            "",
            f"- Status: `{record.status}`",
            f"- Statistic: `{record.statistic or 'NA'}`",
            f"- Calibration: `{record.calibration or 'NA'}`",
            f"- Planned/valid tests: `{record.n_planned or 'NA'}` / "
            f"`{record.n_valid if record.n_valid is not None else 'NA'}`",
            f"- Familywise discoveries: "
            f"`{record.discovery_count if record.discovery_count is not None else 'NA'}`",
            f"- Replication: `{record.replication_status}`",
            "",
            "Flags:",
            "",
        ])
        if record.flags:
            lines.extend([
                f"- `{flag.severity}:{flag.code}` — {flag.message}"
                for flag in record.flags
            ])
        else:
            lines.append("- None.")
        lines.extend(["", "Evidence boundary:", ""])
        lines.extend([f"- {text}" for text in record.evidence_boundary])
        if record.top:
            lines.extend(["", "Top reported units (descriptive):", ""])
            for hit in record.top:
                unit = _hit_identity(hit)
                p = _finite(hit.get("p_interaction"))
                if p is None:
                    p = _finite(hit.get("p"))
                driver, driver_label = _hit_component(hit)
                suffix = (
                    f"; {driver_label}={driver}"
                    if driver is not None and driver_label else "")
                lines.append(f"- `{unit}`: p={p if p is not None else 'NA'}{suffix}")
        lines.append("")
    path.write_text("\n".join(lines))


def run_audit(input_path: str | Path, out_dir: str | Path | None = None) -> dict:
    source = Path(input_path)
    files = _find_result_files(source)
    if not files:
        raise ValueError(f"no summary_*.json, interact_*.json or predict_*.json in {source}")
    records = [audit_result(path) for path in files]
    out = Path(out_dir) if out_dir else (source if source.is_dir() else source.parent) / "audit"
    out.mkdir(parents=True, exist_ok=True)
    overall = _overall_status(records)
    payload = {
        "audit_schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(source),
        "overall_status": overall,
        "n_results": len(records),
        "records": [asdict(record) for record in records],
    }
    json_path = out / "homoeogwas_audit.json"
    tsv_path = out / "homoeogwas_audit.tsv"
    md_path = out / "homoeogwas_audit.md"
    with json_path.open("w") as fh:
        dump_strict(payload, fh, indent=2)
    _write_tsv(tsv_path, records)
    _write_markdown(md_path, overall, records)
    payload["outputs"] = {
        "json": str(json_path), "tsv": str(tsv_path), "markdown": str(md_path)}
    return payload


def cmd_audit(args) -> int:
    try:
        result = run_audit(args.results, args.out_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: homoeogwas audit: {exc}")
        return 2
    print(f"homoeogwas audit: {result['overall_status']}")
    for name, path in result["outputs"].items():
        print(f"  {name}: {path}")
    return 1 if result["overall_status"] == "ANALYSIS_INVALID" else 0
