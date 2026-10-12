"""Input screens that report problems without changing data: extreme phenotype values and
genes repeated across homoeolog groups."""
from __future__ import annotations

from collections import Counter
from typing import Iterable, Sequence

import numpy as np

WARN_SD = 5
REVIEW_SD = 10
MAX_LISTED = 10


def screen_phenotype(samples: Sequence, values) -> dict:
    """Standardized distance of each analysed value from the mean (SD, ddof 1).

    ``warn`` when any value is beyond WARN_SD, ``review`` beyond REVIEW_SD; the most extreme
    samples beyond WARN_SD are listed. Values are never removed or altered."""
    y = np.asarray(values, dtype=float)
    ids = [str(s) for s in samples]
    keep = np.isfinite(y)
    y, ids = y[keep], [s for s, k in zip(ids, keep) if k]
    out = {"n": int(len(y)), "warn_sd": WARN_SD, "review_sd": REVIEW_SD,
           "max_abs_z": None, "flagged": [], "status": "ok"}
    if len(y) < 3:
        return out
    sd = float(np.std(y, ddof=1))
    if not np.isfinite(sd) or sd == 0:
        return out
    z = (y - y.mean()) / sd
    order = np.argsort(-np.abs(z))
    out["max_abs_z"] = round(float(abs(z[order[0]])), 3)
    out["flagged"] = [{"sample": ids[i], "value": float(y[i]), "z": round(float(z[i]), 3)}
                      for i in order[:MAX_LISTED] if abs(z[i]) > WARN_SD]
    if out["max_abs_z"] > REVIEW_SD:
        out["status"] = "review"
    elif out["max_abs_z"] > WARN_SD:
        out["status"] = "warn"
    return out


def phenotype_message(qc: dict, trait: str) -> str | None:
    if qc.get("status") == "ok":
        return None
    listed = ", ".join(f"{f['sample']} ({f['z']:+.1f} SD)" for f in qc["flagged"][:5])
    return (f"trait {trait!r} has values beyond {qc['warn_sd']} SD of the mean: {listed}. "
            "Check these records (data entry, unit or trial errors) before interpreting results; "
            "nothing was removed.")


def duplicated_genes(rows: Iterable[Sequence[str]], subgenomes: Sequence[str]) -> dict:
    """Genes that appear in more than one row within a subgenome column."""
    counts = [Counter() for _ in subgenomes]
    for row in rows:
        for c, gene in zip(counts, row):
            if gene is None or (isinstance(gene, float) and gene != gene):
                continue
            gene = str(gene).strip()
            if gene:
                c[gene] += 1
    return {s: sorted(g for g, k in c.items() if k > 1)
            for s, c in zip(subgenomes, counts) if any(k > 1 for k in c.values())}


def duplicate_message(dup: dict) -> str:
    parts = []
    for s, genes in dup.items():
        shown = ", ".join(genes[:5]) + (f" (+{len(genes) - 5} more)" if len(genes) > 5 else "")
        parts.append(f"{s}: {shown}")
    return ("genes duplicated across homoeolog groups (paralog contamination or a broken "
            f"homoeolog table): {'; '.join(parts)}. Rebuild the table one-to-one, or set "
            "interact.allow_duplicate_genes: true to run anyway (recorded and flagged in the audit)")
