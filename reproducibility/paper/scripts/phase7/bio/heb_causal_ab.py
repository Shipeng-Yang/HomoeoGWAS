#!/usr/bin/env python3
"""Task #2 (a)+(b): multi-study stage/tissue-specific HEB + matched-null for the wheat chr1 triad.

Pre-registered in reports/heb_causal_ladder_prereg.md (FROZEN before chr1 was examined in any window
or matched set). Upgrades the single-study, all-sample-mean imbalance result toward causal:

(a) Stage/tissue-specific, multi-study: recompute the chr1 triad A/B/D imbalance restricted to the
    FROZEN seedling-tight emergence window across the expVIP compendium; study-balanced weighting is
    primary; percentile vs the window background; multi-study consistency (fraction of studies where
    chr1 is B-dominant).
(b) Matched-null: match chr1 to background 1:1:1 triads on detectability covariates (expression
    level/variability/missingness, gene length, callable-SNP count) that EXCLUDE the imbalance, then
    test whether chr1 stays extreme -> refutes the "imbalanced genes are merely easier to detect"
    confound.

All TPM is local (expVIP kallisto matrices in the zip); gene_len + callable-SNP count are joined from
the task-#1 genome-wide ranking (GFF-derived, no fabrication). No Salmon, no download.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/mnt/7302share/fast_ysp/U7_GWAS")
ZIP = ROOT / "data/raw/expression/wheat/iwgsc_refseqv1.1_rnaseq_TPM.zip"
ZBASE = "iwgsc_refseqv1.1_rnaseq_mapping_2017July20/ByGene"
META = "iwgsc_refseqv1.1_rnaseq_mapping_2017July20/public_metadata_20170518.txt"
HCTRIADS = ROOT / "data/raw/expression/wheat/HCTriads.csv"
GW_RANKING = ROOT / "results/phase7/bio_wheat/wheat_GW_ranking_flank_INT.tsv"
OUT = ROOT / "results/phase7/bio_wheat/heb_causal_ladder"

CHR1 = dict(A="TraesCS1A02G125400", B="TraesCS1B02G143800", D="TraesCS1D02G128400")
# FROZEN seedling-tight emergence window (parent prereg §2): age==seedling AND tissue in this set
EMG_TISSUES = {"radicle", "coleoptile", "seedling", "roots", "shoot apical meristem", "stem axis",
               "first leaf sheath", "first leaf blade", "second leaf"}
# heading-window sensitivity tissue sets (addendum §2): flag leaf is heading-ASSOCIATED (sens only);
# gametophytic tissues excluded in the gametophyte-exclusion sensitivity.
GAMETO_TISSUES = {"anther", "microspores", "ovary", "stigma & ovary"}
EXPR_MIN = 1.0           # triad "expressed" in a sample if A+B+D >= 1 TPM
MIN_SAMP = 3             # a study/triad needs >=3 expressed window samples to count

# FROZEN window predicates (addendum 2026-06-06). "seedling" reproduces the original result
# bit-for-bit; the heading windows are the phenotype-aligned correction (trait = days-to-heading).
WINDOWS = {
    "seedling": lambda r: r["High level age"] == "seedling" and r["Tissue"] in EMG_TISSUES,
    "heading": lambda r: r["High level age"] == "reproductive" and r["High level tissue"] == "spike",
    "heading_spike_anyage": lambda r: r["High level tissue"] == "spike",
    "heading_spike_flagleaf": lambda r: (
        (r["High level age"] == "reproductive" and r["High level tissue"] == "spike")
        or ("flag leaf" in r["Tissue"].lower())),
    "heading_spike_nogam": lambda r: (
        r["High level age"] == "reproductive" and r["High level tissue"] == "spike"
        and r["Tissue"] not in GAMETO_TISSUES),
}


def _v10_to_v11(g):
    return g.replace("01G", "02G", 1) if "01G" in g else g


def _read_meta_window(window):
    """Return {study_title: [sample_id,...]} for the FROZEN selection predicate `window`."""
    pred = WINDOWS[window]
    with zipfile.ZipFile(ZIP) as z, z.open(META) as fh:
        rows = list(csv.DictReader(io.TextIOWrapper(fh, "utf-8"), delimiter="\t"))
    by_study = {}
    for r in rows:
        if pred(r):
            by_study.setdefault(r["study_title"], []).append(r["Sample IDs"])
    return by_study


def _triad_balance(a, b, d):
    tot = a + b + d
    if tot <= 1e-9:
        return None
    frac = np.array([a / tot, b / tot, d / tot])
    dist = float(np.linalg.norm(frac - np.array([1 / 3, 1 / 3, 1 / 3])))
    ideals = {"balanced": [1 / 3, 1 / 3, 1 / 3], "A.dominant": [1, 0, 0], "B.dominant": [0, 1, 0],
              "D.dominant": [0, 0, 1], "A.suppressed": [0, 1 / 2, 1 / 2],
              "B.suppressed": [1 / 2, 0, 1 / 2], "D.suppressed": [1 / 2, 1 / 2, 0]}
    cat = min(ideals, key=lambda k: np.linalg.norm(frac - np.array(ideals[k])))
    return dict(fa=float(frac[0]), fb=float(frac[1]), fd=float(frac[2]), dist_center=dist, category=cat)


def _load_study_window_tpm(study, sample_ids, needed_genes):
    """Read one study's TPM matrix (only `needed_genes` rows), return DataFrame genes x window-samples.

    Missing samples (not all metadata samples are necessarily columns) are skipped with a warning."""
    path = f"{ZBASE}/{study}_tpm.tsv.gz"
    with zipfile.ZipFile(ZIP) as z:
        if path not in z.namelist():
            return None
        with z.open(path) as raw, gzip.open(raw, "rt") as fh:
            header = fh.readline().rstrip("\n").split("\t")
            col_idx = {c: i for i, c in enumerate(header)}
            keep = [s for s in sample_ids if s in col_idx]
            if not keep:
                return None
            sel = [col_idx[s] for s in keep]
            data = {}
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                g = parts[0]
                if g in needed_genes:
                    data[g] = [float(parts[i]) if parts[i] != "" else np.nan for i in sel]
    if not data:
        return None
    return pd.DataFrame.from_dict(data, orient="index", columns=keep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", choices=list(WINDOWS), default="seedling")
    args = ap.parse_args()
    window = args.window
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"=== HEB causal ladder (a)+(b): window={window} + matched-null ===")

    # ---- triad universe (1:1:1 HCTriads, v11) + needed gene set ----
    hc = pd.read_csv(HCTRIADS)
    hc = hc[hc["cardinality_abs"] == "1:1:1"].copy()
    triads = []
    for _, r in hc.iterrows():
        triads.append((_v10_to_v11(str(r["A"])), _v10_to_v11(str(r["B"])), _v10_to_v11(str(r["D"]))))
    needed = set()
    for a, b, d in triads:
        needed.update([a, b, d])
    needed.update(CHR1.values())
    print(f"  {len(triads)} background 1:1:1 triads; {len(needed)} genes needed")

    # ---- load window TPM per study ----
    emg = _read_meta_window(window)
    print(f"  {window}-window studies: { {k: len(v) for k, v in emg.items()} }")
    # per study: gene -> array of window-sample TPM
    study_tpm = {}
    for study, sids in emg.items():
        df = _load_study_window_tpm(study, sids, needed)
        if df is None or df.shape[1] == 0:
            print(f"    {study}: no usable columns, skipped")
            continue
        study_tpm[study] = df
        print(f"    {study}: {df.shape[0]} genes x {df.shape[1]} window samples")
    studies = list(study_tpm.keys())

    # ---- per-study, SAMPLE-ALIGNED triad stats with MIN_SAMP enforcement (dual-eval fixes #1,#2) ----
    def triad_study_stats(a, b, d, study):
        """Per-study triad means over EXPRESSED, sample-ALIGNED window samples (all three copies
        finite AND A+B+D >= EXPR_MIN in that sample). Returns (mA, mB, mD, n_expressed, totals) or
        None if the study lacks the genes or has < MIN_SAMP expressed samples. Enforces the declared
        MIN_SAMP rule in the PRIMARY estimate (not just the consistency display) and aligns A/B/D by
        sample column (a missing gene -> study skipped, no per-gene length truncation)."""
        df = study_tpm[study]
        if not all(g in df.index for g in (a, b, d)):
            return None
        M = np.vstack([df.loc[a].to_numpy(float), df.loc[b].to_numpy(float), df.loc[d].to_numpy(float)])
        tot = M.sum(0)
        expr = np.isfinite(M).all(0) & (tot >= EXPR_MIN)     # per-sample expressed, aligned across A/B/D
        n = int(expr.sum())
        if n < MIN_SAMP:
            return None
        m = M[:, expr].mean(1)
        return float(m[0]), float(m[1]), float(m[2]), n, tot[expr]

    def pooled_finite_totals(a, b, d):
        """Concatenated per-sample total TPM (A+B+D) across studies over samples where all three
        copies are FINITE (sample-aligned; NO expression threshold, so the missingness covariate =
        fraction with total < EXPR_MIN is meaningful). One total per sample => no per-gene
        misalignment (dual-eval fix #2)."""
        out = []
        for study in studies:
            df = study_tpm[study]
            if not all(g in df.index for g in (a, b, d)):
                continue
            M = np.vstack([df.loc[a].to_numpy(float), df.loc[b].to_numpy(float),
                           df.loc[d].to_numpy(float)])
            fin = np.isfinite(M).all(0)
            if fin.any():
                out.append(M[:, fin].sum(0))
        return np.concatenate(out) if out else np.array([])

    def triad_studybalanced(a, b, d):
        """study-balanced mean TPM per copy: mean over studies (each >= MIN_SAMP expressed samples)
        of that study's expressed-sample mean. Equal weight per study (batch-robust)."""
        means = {"A": [], "B": [], "D": []}
        per_study_fracB, per_study_cat, per_study_n, per_study_name = [], [], [], []
        for study in studies:
            st = triad_study_stats(a, b, d, study)
            if st is None:
                continue
            ta, tb, td, nexp, _ = st
            means["A"].append(ta)
            means["B"].append(tb)
            means["D"].append(td)
            bal = _triad_balance(ta, tb, td)
            if bal:
                per_study_fracB.append(bal["fb"])
                per_study_cat.append(bal["category"])
                per_study_n.append(nexp)
                per_study_name.append(study)
        if not means["A"]:
            return None
        mA, mB, mD = np.mean(means["A"]), np.mean(means["B"]), np.mean(means["D"])
        bal = _triad_balance(mA, mB, mD)
        if bal is None:
            return None
        bal.update(tpmA=float(mA), tpmB=float(mB), tpmD=float(mD), n_studies=len(means["A"]),
                   per_study_fracB=per_study_fracB, per_study_cat=per_study_cat,
                   per_study_n=per_study_n, per_study_name=per_study_name)
        return bal

    # ---- background imbalance distribution in the frozen window ----
    bg_rows = []
    for a, b, d in triads:
        bal = triad_studybalanced(a, b, d)
        if bal is None or bal["n_studies"] < 1:
            continue
        bg_rows.append(dict(A=a, B=b, D=d, **{k: bal[k] for k in
                       ("fa", "fb", "fd", "dist_center", "category", "tpmA", "tpmB", "tpmD",
                        "n_studies")}))
    bg = pd.DataFrame(bg_rows)
    print(f"  {len(bg)} background triads expressed in the emergence window")

    # ---- (a) chr1 in the window ----
    c = CHR1
    chr1_bal = triad_studybalanced(c["A"], c["B"], c["D"])
    a_out = {}
    if chr1_bal is None:
        print("  chr1 NOT expressed in the emergence window")
    else:
        dist_pct = float((bg["dist_center"] < chr1_bal["dist_center"]).mean())
        fracb_pct = float((bg["fb"] < chr1_bal["fb"]).mean())
        # conservative empirical upper-tail p (ties via >=)
        emp_p_dist = float((1 + int((bg["dist_center"] >= chr1_bal["dist_center"]).sum())) / (len(bg) + 1))
        nB = sum(1 for cat in chr1_bal["per_study_cat"] if cat == "B.dominant")
        # strict consistency: studies with >=3 expressed window samples (Codex fix #4)
        strict = [(nm, cat, n) for nm, cat, n in zip(chr1_bal["per_study_name"],
                  chr1_bal["per_study_cat"], chr1_bal["per_study_n"], strict=True) if n >= MIN_SAMP]
        nB_strict = sum(1 for _, cat, _ in strict if cat == "B.dominant")
        a_out = dict(window=window, weighting="study_balanced_primary",
                     n_window_studies_chr1=chr1_bal["n_studies"],
                     tpmA=chr1_bal["tpmA"], tpmB=chr1_bal["tpmB"], tpmD=chr1_bal["tpmD"],
                     fracA=chr1_bal["fa"], fracB=chr1_bal["fb"], fracD=chr1_bal["fd"],
                     dist_center=chr1_bal["dist_center"], category=chr1_bal["category"],
                     imbalance_percentile=dist_pct, fracB_percentile=fracb_pct,
                     empirical_p_dist=emp_p_dist, n_background=len(bg),
                     consistency=dict(n_studies_expressed=chr1_bal["n_studies"],
                                      n_B_dominant=nB,
                                      frac_studies_B_dominant=(nB / chr1_bal["n_studies"]),
                                      per_study_fracB=chr1_bal["per_study_fracB"],
                                      per_study_category=chr1_bal["per_study_cat"],
                                      per_study_n=chr1_bal["per_study_n"],
                                      per_study_name=chr1_bal["per_study_name"],
                                      strict_min3=dict(n_studies=len(strict), n_B_dominant=nB_strict,
                                                       frac_B_dominant=(nB_strict / len(strict)
                                                                        if strict else None),
                                                       studies=[(nm, cat, n) for nm, cat, n in strict])))
        print(f"  (a) chr1 window: fracB={chr1_bal['fb']:.3f} cat={chr1_bal['category']} "
              f"dist_pct={dist_pct:.4f} fracB_pct={fracb_pct:.4f} emp_p={emp_p_dist:.3g} "
              f"B-dominant in {nB}/{chr1_bal['n_studies']} studies "
              f"(strict>=3samp: {nB_strict}/{len(strict)})")

    # ---- (b) matched-null: covariates joined from #1 ranking (gene_len + n_snp) ----
    rk = pd.read_csv(GW_RANKING, sep="\t",
                     usecols=["gene_A", "gene_B", "gene_D", "n_snp_triad", "gene_len_triad_sum"])
    rk = rk[(rk["gene_len_triad_sum"] != "NA")].copy()
    rk["gene_len_triad_sum"] = rk["gene_len_triad_sum"].astype(float)
    rk_key = {(r.gene_A, r.gene_B, r.gene_D): (float(r.n_snp_triad), float(r.gene_len_triad_sum))
              for r in rk.itertuples(index=False)}

    def covariates(a, b, d):
        """[log1p(meanTPM), CV, missingness, gene_len_sum, n_snp_triad] or None if covariates absent."""
        if (a, b, d) not in rk_key:
            return None
        nsnp, glen = rk_key[(a, b, d)]
        tot = pooled_finite_totals(a, b, d)              # sample-aligned per-sample totals (finite)
        if tot.size < MIN_SAMP:
            return None
        mean_tpm = float(tot.mean())
        cv = float(tot.std() / mean_tpm) if mean_tpm > 1e-9 else 0.0
        miss = float((tot < EXPR_MIN).mean())
        return np.array([np.log1p(mean_tpm), cv, miss, glen, nsnp])

    # build covariate matrix + imbalance over the intersection (expression background AND covariates)
    bg_idx = {(r["A"], r["B"], r["D"]): r["dist_center"] for _, r in bg.iterrows()}
    Xcov, dist_vec, keys = [], [], []
    for a, b, d in triads:
        if (a, b, d) not in bg_idx:
            continue
        cov = covariates(a, b, d)
        if cov is None or not np.all(np.isfinite(cov)):
            continue
        Xcov.append(cov)
        dist_vec.append(bg_idx[(a, b, d)])
        keys.append((a, b, d))
    Xcov = np.array(Xcov)
    dist_vec = np.array(dist_vec)
    chr1_key = (c["A"], c["B"], c["D"])
    b_out = {}
    if chr1_key not in keys:
        print("  (b) chr1 lacks covariates/expression in the matched-null universe")
    else:
        ci = keys.index(chr1_key)
        # standardize covariates; coarsened strata on expression decile x missingness quintile x n_snp quintile
        mu, sd = Xcov.mean(0), Xcov.std(0)
        sd = np.where(sd > 1e-12, sd, 1.0)
        Z = (Xcov - mu) / sd
        expr_dec = np.asarray(pd.qcut(Xcov[:, 0], 10, labels=False, duplicates="drop"))
        miss_q = np.asarray(pd.qcut(Xcov[:, 2].argsort().argsort(), 5, labels=False, duplicates="drop"))
        snp_q = np.asarray(pd.qcut(Xcov[:, 4].argsort().argsort(), 5, labels=False, duplicates="drop"))
        strata = np.array([f"{expr_dec[i]}_{miss_q[i]}_{snp_q[i]}" for i in range(len(keys))])
        in_stratum = np.where(strata == strata[ci])[0]
        in_stratum = in_stratum[in_stratum != ci]
        # candidate pool = chr1's coarsened stratum if well-populated, else all other triads (the
        # caliper then does the matching); avoids a degenerate tiny matched set.
        all_others = np.array([i for i in range(len(keys)) if i != ci])
        pool = in_stratum if in_stratum.size >= 50 else all_others
        pool_mode = "stratum" if in_stratum.size >= 50 else "all_caliper"

        # caliper: per-covariate standardized abs diff <= caliper
        def matched(caliper):
            d2 = np.abs(Z[pool] - Z[ci])
            return pool[(d2 <= caliper).all(1)]
        matched_idx = matched(0.25)
        caliper_used = 0.25
        if matched_idx.size < 200:
            matched_idx = matched(0.35)
            caliper_used = 0.35
        if matched_idx.size > 1000:
            order = np.argsort(np.sqrt(((Z[matched_idx] - Z[ci]) ** 2).sum(1)))
            matched_idx = matched_idx[order[:1000]]
        m_dist = dist_vec[matched_idx]
        matched_pct = float((m_dist < dist_vec[ci]).mean())
        emp_p = float((1 + int((m_dist >= dist_vec[ci]).sum())) / (m_dist.size + 1))
        # covariate balance (post-match standardized mean diff)
        bal_smd = {nm: float(Z[matched_idx, j].mean() - Z[ci, j])
                   for j, nm in enumerate(["logTPM", "CV", "missing", "gene_len", "n_snp"])}
        # PRE-REGISTERED rank-distance sensitivity (powered): K nearest background triads by
        # standardized Mahalanobis distance (no hard caliper) -> empirical p not floored by a tiny N.
        # chr1 sits in a sparse high-expression/high-SNP corner, so the caliper match is small (N~12,
        # so its empirical p is floored at 1/(N+1)); the K-NN gives a powered test with a reported
        # covariate-balance table so match quality is auditable.
        maha = np.sqrt(((Z[all_others] - Z[ci]) ** 2).sum(1))
        knn = {}
        for K in (200, 500):
            sel = all_others[np.argsort(maha)[:K]]
            m_pct = float((dist_vec[sel] < dist_vec[ci]).mean())
            ep = float((1 + int((dist_vec[sel] >= dist_vec[ci]).sum())) / (sel.size + 1))
            smd = {nm: float(Z[sel, j].mean() - Z[ci, j])
                   for j, nm in enumerate(["logTPM", "CV", "missing", "gene_len", "n_snp"])}
            knn[f"K{K}"] = dict(matched_n=int(sel.size), matched_percentile=m_pct, empirical_p=ep,
                                covariate_balance_smd=smd)
            print(f"      rank-dist K={K}: matched_pct={m_pct:.4f} emp_p={ep:.3g}")

        # covariate-set sensitivity (Codex fix #2): drop CV (may partly track dominance), redo K-NN
        keepc = [0, 2, 3, 4]                              # logTPM, missingness, gene_len, n_snp
        Zb = (Xcov[:, keepc] - Xcov[:, keepc].mean(0)) / np.where(Xcov[:, keepc].std(0) > 1e-12,
                                                                  Xcov[:, keepc].std(0), 1.0)
        maha_b = np.sqrt(((Zb[all_others] - Zb[ci]) ** 2).sum(1))
        knn_noCV = {}
        for K in (200, 500):
            sel = all_others[np.argsort(maha_b)[:K]]
            knn_noCV[f"K{K}"] = dict(
                matched_percentile=float((dist_vec[sel] < dist_vec[ci]).mean()),
                empirical_p=float((1 + int((dist_vec[sel] >= dist_vec[ci]).sum())) / (sel.size + 1)))
        print(f"      no-CV K200: pct={knn_noCV['K200']['matched_percentile']:.4f} "
              f"emp_p={knn_noCV['K200']['empirical_p']:.3g}")

        # parametric sensitivity: regress dist ~ covariates, chr1 residual percentile
        import numpy.linalg as la
        Xd = np.column_stack([np.ones(len(keys)), Z])
        beta, *_ = la.lstsq(Xd, dist_vec, rcond=None)
        resid = dist_vec - Xd @ beta
        resid_pct = float((resid < resid[ci]).mean())
        b_out = dict(universe_n=len(keys), chr1_dist=float(dist_vec[ci]),
                     stratum_n=int(in_stratum.size), pool_mode=pool_mode,
                     matched_n=int(matched_idx.size),
                     caliper_used=caliper_used, matched_percentile=matched_pct, empirical_p=emp_p,
                     caliper_note=("chr1 in a sparse high-expression/high-SNP corner -> small caliper "
                                   "match; empirical_p floored at 1/(matched_n+1). See rank_distance_knn "
                                   "(powered) and residual_percentile_parametric."),
                     rank_distance_knn=knn, rank_distance_knn_noCV=knn_noCV,
                     covariate_balance_smd=bal_smd, residual_percentile_parametric=resid_pct,
                     covariates=["log1p_meanTPM", "CV", "missingness", "gene_len_sum", "n_snp_triad"])
        print(f"  (b) matched-null: matched_n={matched_idx.size} (caliper {caliper_used}) "
              f"matched_pct={matched_pct:.4f} emp_p={emp_p:.3g} resid_pct={resid_pct:.4f}")

    prereg = ("reports/heb_causal_ladder_prereg.md" if window == "seedling"
              else "reports/heb_causal_ladder_prereg_addendum.md")
    payload = dict(task="heb_causal_ladder_ab", window=window, prereg=prereg,
                   window_studies={k: len(v) for k, v in emg.items()},
                   studies_used=studies, n_background_window=len(bg),
                   chr1=CHR1, a_stage_specific=a_out, b_matched_null=b_out,
                   note=("study-balanced weighting primary; expVIP kallisto TPM (no Salmon); gene_len "
                         "+ n_snp joined from task-#1 GFF-derived ranking (no fabrication); chr5 A-D "
                         "remains expression-balanced (separate, no convergence claim). trait = "
                         "days-to-heading (ear emergence, GS55); seedling window superseded by the "
                         "heading-window addendum."))
    suffix = "" if window == "seedling" else f"_{window}"
    fp = OUT / f"heb_causal_ab{suffix}.json"
    fp.write_text(json.dumps(payload, indent=2, default=float))
    print(f"  wrote {fp}")


if __name__ == "__main__":
    main()
