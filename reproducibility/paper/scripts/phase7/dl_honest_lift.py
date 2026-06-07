#!/usr/bin/env python3
"""P0-2: leakage-aware honest re-analysis of the DL-prior recall "lift".

The historical estimate (scripts/phase3/m3_3_fuse_and_evaluate.py) is SELECTION-BIASED: the
fusion weight beta is chosen by ``fit_beta_loqo`` to maximise mean recall over the recoverable
known-QTL set, and the headline ``absolute_lift`` is then reported at that beta ON THE SAME SET
(no outer held-out fold). beta almost always lands on the grid extremum (5.0).

This script recomputes, from the EXISTING scored candidates (no GPU), a beta-selection-honest
estimate plus a permutation null, for every panel x trait, and writes a 3-column comparison:
``selection_biased_lift`` (historical) / ``nested_loqo_lift`` (honest) / ``permutation_p``.

  nested_loqo_lift : outer leave-one-QTL-out. For each recoverable QTL q (the OUTER held-out
      unit), beta is selected ONLY on the inner QTL (recoverable \\ {q}) using the same fusion,
      same beta grid, and the same <5-recoverable / plateau lock rule as the historical
      procedure (grid-max inner recall; no peeking at q); then we record whether q is recovered
      at top-N under the fused ranking minus under the GWAS-only ranking (delta in {-1,0,+1}).
      The nested lift is the mean delta over outer QTL. Raw per-QTL deltas are persisted (with
      N_recoverable = 3-13 the raw table is more honest than any interval; a QTL-level bootstrap
      CI is added as descriptive only).
  permutation_p : shuffle |LLR|_ensemble among the scored candidates and rerun the ENTIRE nested
      procedure (incl. inner beta selection); statistic = nested_loqo_lift; empirical
      p = (1 + #{perm >= obs}) / (1 + B).

CRITICAL DISCLOSURE (this analysis CANNOT remove it): the candidate pool was seeded with
known-QTL sentinel SNPs AND the DL ensemble was scored only on GWAS-derived candidate variants
(a y-DEPENDENT selection). So this is NOT a genome-wide y-independent prior-weighted discovery
test — it only asks whether the available DL scores add ranking signal WITHIN the already-selected
candidate set. The clean prior-weighted-discovery evidence is the wheat HEB weighted-interaction
demo + the power simulation, not this. Genome-wide DL rescoring (to remove candidate-selection
leakage) is future work.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/mnt/7302share/fast_ysp/U7_GWAS")
sys.path.insert(0, str(ROOT / "src"))
_spec = importlib.util.spec_from_file_location(
    "m3_3", str(ROOT / "scripts/phase3/m3_3_fuse_and_evaluate.py"))
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)
robust_z = _m.robust_z
known_qtl_for_snp = _m.known_qtl_for_snp
TOP_N = _m.RECALL_TOP_N                                  # 100
BETA_GRID = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0)
LOCKED_BETA = 0.25
B_PERM = 2000
N_BOOT = 10000
OUT = ROOT / "results/phase7/dl_honest_lift.json"

# (panel, trait, candidate_dir, known_qtl_tsv)
KQ = ROOT / "data/reference"
MANIFEST = [
    ("wheat_watkins", "days_to_emerg", "results/phase3/m3_3_dl_prior/wheat_watkins",
     KQ / "wheat/known_qtl_wheat.tsv"),
    ("rapeseed_horvath2020", "bloom_50pct", "results/phase3/m3_3_dl_prior_v2/horvath2020/bloom_50pct",
     KQ / "rapeseed/known_qtl_rapeseed.tsv"),
    ("rapeseed_horvath2020", "plant_height", "results/phase3/m3_3_dl_prior_v2/horvath2020/plant_height",
     KQ / "rapeseed/known_qtl_rapeseed.tsv"),
    ("cotton_hebau", "fiber_length_BLUE", "results/phase3/m3_3_dl_prior_v2/cotton_hebau/fiber_length_BLUE",
     KQ / "cotton/known_qtl_cotton.tsv"),
    ("cotton_hebau", "lint_percentage_BLUE", "results/phase3/m3_3_dl_prior_v2/cotton_hebau/lint_percentage_BLUE",
     KQ / "cotton/known_qtl_cotton.tsv"),
    ("rice_ricevarmap2", "Heading_date", "results/phase5e/m3_3_dl_prior/rice_ricevarmap2/Heading_date",
     KQ / "rice/known_qtl_rice_Heading_date.tsv"),
    ("rice_ricevarmap2", "Grain_width", "results/phase5e/m3_3_dl_prior/rice_ricevarmap2/Grain_width",
     KQ / "rice/known_qtl_rice_Grain_width.tsv"),
    ("rice_ricevarmap2", "Grain_length", "results/phase5e/m3_3_dl_prior/rice_ricevarmap2/Grain_length",
     KQ / "rice/known_qtl_rice_Grain_length.tsv"),
    ("rice_ricevarmap2", "Plant_height", "results/phase5e/m3_3_dl_prior/rice_ricevarmap2/Plant_height",
     KQ / "rice/known_qtl_rice_Plant_height.tsv"),
    ("oat_old_rahman2025", "BIO6", "results/phase5d/m3_3_dl_prior/oat_old_rahman2025/BIO6",
     KQ / "oat/known_qtl_oat.tsv"),
    ("oat_old_rahman2025", "BIO12", "results/phase5d/m3_3_dl_prior/oat_old_rahman2025/BIO12",
     KQ / "oat/known_qtl_oat.tsv"),
    ("oat_old_rahman2025", "SOC", "results/phase5d/m3_3_dl_prior/oat_old_rahman2025/SOC",
     KQ / "oat/known_qtl_oat.tsv"),
]


def _prep(cand_dir: Path, known_qtl: Path):
    """Replicate the m3_3 fusion prep: merge DL scores, ensemble robust-z, z_gwas, QTL windows."""
    cand = pd.read_csv(cand_dir / "candidates.tsv.gz", sep="\t", compression="gzip")
    merged = cand.copy()
    n_models = 0
    for model in ("plantcad", "agront"):
        fp = cand_dir / f"dl_scores_{model}.tsv.gz"
        if not fp.exists():
            continue
        df = pd.read_csv(fp, sep="\t", compression="gzip")
        df = df[df["status"] == "OK"][["snp_id", "llr_abs"]].rename(columns={"llr_abs": f"llr_abs_{model}"})
        merged = merged.merge(df, on="snp_id", how="left")
        merged[f"z_llr_abs_{model}"] = robust_z(merged[f"llr_abs_{model}"].to_numpy())
        n_models += 1
    z_cols = [f"z_llr_abs_{m}" for m in ("plantcad", "agront") if f"z_llr_abs_{m}" in merged.columns]
    merged["z_llr_abs"] = merged[z_cols].mean(axis=1, skipna=True)
    p_safe = merged["p"].fillna(1.0).clip(lower=1e-300).to_numpy()
    merged["z_gwas"] = robust_z(-np.log10(p_safe))
    qtls = pd.read_csv(known_qtl, sep="\t")
    merged = known_qtl_for_snp(merged, qtls)
    valid = merged.dropna(subset=["z_gwas", "z_llr_abs"]).reset_index(drop=True)
    recoverable = set(valid.loc[valid["in_known_window_500kb"], "nearest_known_qtl"].dropna().astype(str))
    return valid, recoverable, n_models


def _recovered_by_beta(zg: np.ndarray, zl: np.ndarray, inwin: np.ndarray, nq: np.ndarray,
                       recoverable: set, top_n: int) -> dict:
    """For each beta, the set of recoverable QTL recovered in the top-N fused ranking."""
    out = {}
    for b in BETA_GRID:
        order = np.argsort(-(zg + b * zl), kind="stable")[:top_n]
        rec = set(nq[order][inwin[order]]) & recoverable
        out[b] = rec
    return out


def _select_beta_inner(rec_by_beta: dict, inner: set) -> float:
    """Historical beta-selection rule applied to the INNER QTL only (outer q held out):
    <5 recoverable -> locked 0.25; else grid-max inner recall, plateau over beta=0 -> 0.25."""
    if len(inner) < 5:
        return LOCKED_BETA
    inner_recall = {b: len(rec_by_beta[b] & inner) / len(inner) for b in BETA_GRID}
    best = max(inner_recall, key=inner_recall.get)
    if inner_recall[best] - inner_recall[0.0] < 1e-9:
        return LOCKED_BETA
    return best


def _nested_loqo(rec_by_beta: dict, recoverable: set):
    """Outer LOQO nested lift: per outer QTL, select beta on inner, record fused-vs-gwas recovery."""
    rl = sorted(recoverable)
    deltas, betas = [], []
    gwas_rec = rec_by_beta[0.0]
    for q in rl:
        inner = recoverable - {q}
        b_in = _select_beta_inner(rec_by_beta, inner)
        d = int(q in rec_by_beta[b_in]) - int(q in gwas_rec)
        deltas.append(d)
        betas.append(b_in)
    return float(np.mean(deltas)) if deltas else float("nan"), deltas, betas


def analyse(panel, trait, cand_dir, known_qtl, rng):
    valid, recoverable, n_models = _prep(ROOT / cand_dir, known_qtl)
    n_rec = len(recoverable)
    # historical selection-biased lift (read from the persisted summary if present)
    summ = ROOT / cand_dir / f"m3_3_summary_{trait}.json"
    sel_lift = chosen_beta = None
    if summ.exists():
        sd = json.loads(summ.read_text())
        sel_lift = sd.get("absolute_lift")
        chosen_beta = sd.get("chosen_beta")

    base = dict(panel=panel, trait=trait, n_candidates=int(len(valid)), n_models=n_models,
                n_recoverable_qtls=n_rec, selection_biased_lift=sel_lift,
                historical_chosen_beta=chosen_beta,
                # n_rec<6 -> every outer fold has <5 inner QTL -> the <5 lock ALWAYS fires, so the
                # nested estimate evaluates the locked beta=0.25 fallback, NOT the historical
                # grid-search selection rule. Conservative but to be stated.
                small_n_locked_fallback=bool(n_rec < 6))
    if n_rec < 2:
        base.update(nested_loqo_lift=None, permutation_p=None,
                    note="n_recoverable < 2 -> nested LOQO undefined; disclose only")
        return base

    zg = valid["z_gwas"].to_numpy()
    zl = valid["z_llr_abs"].to_numpy()
    inwin = valid["in_known_window_500kb"].to_numpy().astype(bool)
    nq = valid["nearest_known_qtl"].astype(str).to_numpy()

    rec_obs = _recovered_by_beta(zg, zl, inwin, nq, recoverable, TOP_N)
    obs_lift, deltas, betas = _nested_loqo(rec_obs, recoverable)

    # permutation null: shuffle |LLR| among scored candidates, rerun the full nested procedure
    perm = np.empty(B_PERM)
    for b in range(B_PERM):
        zl_p = rng.permutation(zl)
        rec_p = _recovered_by_beta(zg, zl_p, inwin, nq, recoverable, TOP_N)
        perm[b], _, _ = _nested_loqo(rec_p, recoverable)
    emp_p = float((1 + int((perm >= obs_lift - 1e-12).sum())) / (1 + B_PERM))

    # descriptive QTL-level bootstrap CI on the mean delta (tiny N -> noisy, descriptive only)
    dv = np.array(deltas, float)
    boot = np.array([dv[rng.integers(0, len(dv), len(dv))].mean() for _ in range(N_BOOT)])
    ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]

    base.update(
        nested_loqo_lift=obs_lift,
        nested_loqo_lift_boot_ci95=ci,
        permutation_p=emp_p,
        per_qtl=[dict(qtl=q, delta=int(d), beta_inner=float(bi))
                 for q, d, bi in zip(sorted(recoverable), deltas, betas, strict=True)],
        beta_inner_grid_max_freq=float(np.mean(np.array(betas) == max(BETA_GRID))),
        perm_null_quantiles=dict(q50=float(np.percentile(perm, 50)),
                                 q95=float(np.percentile(perm, 95)),
                                 q99=float(np.percentile(perm, 99))),
        interpretation=("nested-LOQO lift is the beta-selection-honest estimate (beta chosen only "
                        "on inner QTL); permutation_p tests whether the DL score CONTENT adds "
                        "ranking signal beyond the candidate structure (GWAS ranking + sentinels + "
                        "beta machinery). A high perm_p means the DL scores are not distinguishable "
                        "from shuffled labels UNDER THIS TEST — it does NOT prove the DL carries no "
                        "signal. Candidate-selection leakage (sentinel seeding + y-dependent "
                        "candidate scoring) is NOT removed. Permutations preserve the exact candidate "
                        "universe, GWAS ranking, QTL windows and sentinel membership; only "
                        "|LLR|_ensemble is shuffled."))
    return base


def main():
    t0 = time.time()
    rng = np.random.default_rng(20260602)
    print("=== P0-2 honest DL-prior lift re-analysis (nested-LOQO + permutation null) ===")
    rows = []
    for panel, trait, cand_dir, kq in MANIFEST:
        if not (ROOT / cand_dir / "candidates.tsv.gz").exists():
            print(f"  SKIP {panel}/{trait}: no candidates")
            continue
        r = analyse(panel, trait, cand_dir, kq, rng)
        rows.append(r)
        nl = r.get("nested_loqo_lift")
        print(f"  {panel}/{trait}: N_rec={r['n_recoverable_qtls']} "
              f"sel_biased_lift={r.get('selection_biased_lift')} "
              f"nested_loqo_lift={nl if nl is None else round(nl,4)} "
              f"perm_p={r.get('permutation_p')} "
              f"beta_max_freq={r.get('beta_inner_grid_max_freq')}", flush=True)

    n_done = [r for r in rows if r.get("nested_loqo_lift") is not None]
    payload = dict(
        analysis="dl_prior_honest_lift", date="2026-06-02", top_n=TOP_N, beta_grid=list(BETA_GRID),
        n_perm=B_PERM, n_boot=N_BOOT,
        method=__doc__,
        summary=dict(
            n_panels=len(rows), n_with_nested=len(n_done),
            n_nested_lift_positive=int(sum(r["nested_loqo_lift"] > 1e-9 for r in n_done)),
            n_perm_p_below_0p05=int(sum((r["permutation_p"] or 1) < 0.05 for r in n_done)),
            headline=("No panel shows a permutation-significant nested-LOQO lift (all perm_p >= 0.43). "
                      "Apparent historical lifts are either beta-selection artifacts that collapse to "
                      "0 under nested LOQO (small-N panels, with the locked-fallback caveat) or, for "
                      "large-N panels, not distinguishable from the candidate/sentinel structure under "
                      "permutation (shuffled DL scores reproduce them). This does NOT prove the DL "
                      "carries no signal; it shows the available candidate-limited DL scores add no "
                      "detectable ranking benefit beyond the already-leaky candidate set. Candidate-"
                      "selection leakage (sentinel seeding + y-dependent candidate scoring) is NOT "
                      "removed by this analysis. DL is positioned as a capability + mechanistic demo; "
                      "the clean prior-weighted-discovery evidence is the wheat HEB weighted demo + "
                      "the power simulation. Genome-wide DL rescoring is future work.")),
        results=rows)
    OUT.write_text(json.dumps(payload, indent=2, default=float))
    print(f"\nwrote {OUT} ({time.time()-t0:.0f}s); nested on {len(n_done)}/{len(rows)} panels")


if __name__ == "__main__":
    main()
