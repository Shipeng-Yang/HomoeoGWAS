"""Input screens that report problems without changing data: extreme phenotype values and
genes repeated across homoeolog groups."""
from __future__ import annotations

import hashlib
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

WARN_SD = 5
REVIEW_SD = 10
MAX_LISTED = 10
WAIVER_MIN_REASON = 30
JOIN_LOSS_WARN = 0.05
TRIAL_SHIFT_SD = 3.0


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
            f"homoeolog table): {'; '.join(parts)}. Rebuild the table one-to-one (one gene per "
            "subgenome per group) or report it to the data owner; nothing was run.")


def raw_trial_average(df, sample_col: str, trait: str, max_cols: int = 200) -> dict | None:
    """Detect a trait that equals the row mean of >= 2 other numeric columns (trials) whose
    means differ by more than TRIAL_SHIFT_SD pooled within-column SD: a raw average that
    mixes trial/environment effects into the phenotype. Returns a record or None.

    One least-squares solve of the trait on the other numeric columns (complete rows): an exact
    row mean leaves no residual and gives weight 1/k to its k columns and 0 to the rest."""
    import pandas as pd
    if trait not in df.columns:
        return None
    y = pd.to_numeric(df[trait], errors="coerce")
    num = []
    for c in df.columns:
        if c in (sample_col, trait) or len(num) >= max_cols:
            continue
        col = pd.to_numeric(df[c], errors="coerce")
        if col.notna().mean() > 0.5 and col.nunique(dropna=True) > 1:
            num.append(c)
    if len(num) < 2:
        return None
    vals = df[num].apply(pd.to_numeric, errors="coerce")
    mask = vals.notna().all(axis=1) & y.notna()
    if mask.sum() < max(10, 0.5 * len(df), len(num) + 2):
        return None
    X, yy = vals[mask].to_numpy(float), y[mask].to_numpy(float)
    coef, *_ = np.linalg.lstsq(X, yy, rcond=None)
    scale = float(np.std(yy)) or 1.0
    if float(np.max(np.abs(X @ coef - yy))) > 1e-6 * scale:
        return None
    used = [k for k, b in enumerate(coef) if abs(b) > 1e-6]
    if len(used) < 2 or not np.allclose(coef[used], 1.0 / len(used), atol=1e-6):
        return None
    cols = [num[k] for k in used]
    sub = vals.loc[mask, cols]
    means, sds = sub.mean(axis=0), sub.std(axis=0, ddof=1)
    pooled = float(np.sqrt(np.mean(np.square(sds))))
    if not pooled > 0:
        return None
    spread = float((means.max() - means.min()) / pooled)
    if spread <= TRIAL_SHIFT_SD:
        return None
    return {"columns": cols, "mean_range_sd": spread, "n_rows": int(mask.sum()),
            "column_means": {c: float(means[c]) for c in cols}}


def raw_trial_average_message(record: dict | None, trait: str) -> str | None:
    if not record:
        return None
    return (f"trait {trait!r} equals the raw average of {record['columns']}, whose means differ "
            f"by {record['mean_range_sd']:.1f} pooled SD. If these columns are trials or "
            "environments, a raw average carries their effects into the phenotype: use adjusted "
            "means (BLUEs/BLUPs from a trial model) or analyse trials separately. Nothing was changed.")


def _norm_id(sample) -> str:
    s = str(sample).strip().lower()
    s = re.sub(r"\.0+$", "", s)
    return (s.lstrip("0") or "0") if s.isdigit() else s


def sample_join(phenotyped: Iterable, genotyped: Iterable, analysis: Iterable) -> dict:
    """Record of the phenotype-genotype sample join.

    ``refuse`` when unmatched phenotype IDs match unmatched genotype IDs after normalisation
    (leading zeros, case, whitespace, trailing ``.0``): the IDs were damaged, e.g. by a
    spreadsheet; ``warn`` when more than JOIN_LOSS_WARN of the phenotyped samples are not
    analysed; otherwise ``ok``. Nothing is renamed or matched fuzzily."""
    ph = {str(s) for s in phenotyped}
    ge = {str(s) for s in genotyped}
    an = {str(s) for s in analysis}
    un_p, un_g = sorted(ph - ge), sorted(ge - ph)
    by_norm: dict[str, str] = {}
    for g in un_g:
        by_norm.setdefault(_norm_id(g), g)
    pairs = [(p, by_norm[_norm_id(p)]) for p in un_p if _norm_id(p) in by_norm]
    n_dropped = len(ph - an)
    frac = n_dropped / len(ph) if ph else 0.0
    status = "refuse" if pairs else ("warn" if frac > JOIN_LOSS_WARN else "ok")
    return {"n_phenotyped": len(ph), "n_genotyped": len(ge), "n_analysis": len(an & ph),
            "n_dropped": n_dropped, "dropped_fraction": frac,
            "examples_unmatched_phenotype": un_p[:5], "examples_unmatched_genotype": un_g[:5],
            "normalisable_pairs": len(pairs), "pair_examples": [list(p) for p in pairs[:5]],
            "status": status}


def sample_join_message(join: dict) -> str | None:
    if join.get("status") == "refuse":
        shown = ", ".join(f"{p!r} vs {g!r}" for p, g in join["pair_examples"])
        return (f"{join['normalisable_pairs']} phenotype sample IDs match only after normalisation "
                f"(leading zeros, case, whitespace or a trailing .0) to genotype IDs, e.g. {shown}. "
                "The IDs were altered (often by a spreadsheet); restore them exactly as in the "
                ".fam file. Nothing was run.")
    if join.get("status") == "warn":
        return (f"{join['n_dropped']} of {join['n_phenotyped']} phenotyped samples "
                f"({100 * join['dropped_fraction']:.1f}%) have no genotype and are not analysed, "
                f"e.g. {join['examples_unmatched_phenotype']}. Check the sample IDs.")
    return None


def file_sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def waiver_problem(waiver) -> str | None:
    """Why a duplicate-gene waiver is incomplete, or None. A waiver needs a written reason
    (>= WAIVER_MIN_REASON characters), a named approver and the sha256 of the exact table."""
    if not isinstance(waiver, dict):
        return "must be a mapping with reason, approver and table_sha256"
    reason, approver, sha = waiver.get("reason"), waiver.get("approver"), waiver.get("table_sha256")
    if not isinstance(reason, str) or len(reason.strip()) < WAIVER_MIN_REASON:
        return f"reason must be a written justification of at least {WAIVER_MIN_REASON} characters"
    if not isinstance(approver, str) or not approver.strip():
        return "approver must name the person who approved running the table as is"
    if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha):
        return "table_sha256 must be the 64-character sha256 of the homoeolog table"
    return None
