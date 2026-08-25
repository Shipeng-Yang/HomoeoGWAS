"""Canonicalize legacy homoeolog-interaction configuration shapes.

The public configuration accepts the historical ``pairs``/``triads`` table
keys for backwards compatibility. New omniB analyses use the generic ``group``
shape so all copy counts share one vocabulary.
"""

from __future__ import annotations

import copy

_NORMALIZED_VERSION = 1


def normalize_interact_config(cfg: dict) -> dict:
    """Return a non-mutating, idempotently normalized interaction config.

    Legacy pairwise/triad omniB inputs become canonical group inputs. Legacy
    burden and experimental triad3 inputs retain their established execution
    paths. Missing legacy table keys are deliberately left for validation to
    report as concrete missing ``interact.groups`` errors rather than KeyError.
    """
    out = copy.deepcopy(cfg)
    if not isinstance(out, dict):
        return out
    ic = out.get("interact")
    if not isinstance(ic, dict):
        return out

    mode = str(ic.get("mode", "pairwise")).lower()
    statistic = str(ic.get("statistic", "omniB")).lower()
    if statistic == "omnib" and mode in {"pairwise", "triad"}:
        legacy_key = "pairs" if mode == "pairwise" else "triads"
        if "groups" not in ic and legacy_key in ic:
            ic["groups"] = ic[legacy_key]
        ic.pop(legacy_key, None)
        ic["mode"] = "group"
        ic.setdefault("hypothesis_unit", "edge" if mode == "pairwise" else "group")
        ic.setdefault("subset_order", 2)

    if statistic == "omnib" and str(ic.get("mode", mode)).lower() == "group":
        grm = ic.setdefault("grm", {})
        if isinstance(grm, dict):
            grm.setdefault("method", "grm_from_X")
            grm.setdefault("maf_min", 0.01)
            grm.setdefault("scope", "all_subgenomes")
        burden = ic.setdefault("burden", {})
        if isinstance(burden, dict):
            burden.setdefault("maf_min", 0.01)

    ic.setdefault("family_scope", "primary_only")
    ic["_normalized_version"] = _NORMALIZED_VERSION
    return out
