#!/usr/bin/env python3
"""Phase 3 M3.3 — fuse DL-prior with GWAS p, evaluate, framing decision.

Fusion (Codex plan §5):  fusion_score = z(-log10(p_gwas)) + beta * z(|LLR|)
  - z = robust z-score (median + MAD-based)
  - beta = leave-one-known-QTL-out grid-search in {0, 0.1, 0.25, 0.5, 1.0}
  - if optimisation unstable (<5 known recovered), lock conservative beta=0.25

Acceptance (charter §2.2 v1.5):
  recall@100 absolute lift ≥ 0.05  →  "DL-enhanced GWAS"
  else                              →  "variant prioritization"

Inputs:
  results/phase3/m3_3_dl_prior/wheat_watkins/dl_scores_{plantcad,agront}.tsv.gz
  results/phase3/m3_3_dl_prior/wheat_watkins/candidates.tsv.gz
  data/reference/wheat/known_qtl_wheat.tsv

Outputs in same dir:
  dl_prior_scores_ensemble.tsv.gz
  reranked_top_loci.tsv
  m3_3_summary_<trait>.json
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
KNOWN_WINDOW_BP = 500_000                       # ±500 kb per QTL
RECALL_TOP_N = 100                              # recall@100


def _json_default(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.bool_,)): return bool(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


# ---------------------------------------------------------------------
# Robust z-score
# ---------------------------------------------------------------------


def robust_z(x: np.ndarray) -> np.ndarray:
    """Median-MAD robust z-score (NaN-preserving)."""
    x = np.asarray(x, dtype=np.float64)
    finite = np.isfinite(x)
    if finite.sum() < 2:
        return np.full_like(x, np.nan)
    med = np.median(x[finite])
    mad = np.median(np.abs(x[finite] - med))
    if mad <= 0:
        sd = np.std(x[finite]) or 1.0
        return (x - med) / sd
    return (x - med) / (1.4826 * mad)


# ---------------------------------------------------------------------
# Candidate / known QTL mapping
# ---------------------------------------------------------------------


def known_qtl_for_snp(cand_df: pd.DataFrame, qtls: pd.DataFrame,
                       window_bp: int = KNOWN_WINDOW_BP) -> pd.DataFrame:
    """Add columns: nearest_known_qtl, in_known_window_500kb.

    For each candidate SNP, find the nearest QTL on the same chrom and flag
    if within ±window_bp.
    """
    out = cand_df.copy()
    # Normalise chrom to str on BOTH sides: a panel with purely numeric chrom
    # names ("1".."12", e.g. rice) is read as int64, so the previous
    # str(cand) == int64(qtl) comparison silently matched nothing → 0 recoverable.
    qtls = qtls.copy()
    qtls["_chrom_s"] = qtls["chrom"].astype(str)
    near: list[str | None] = []
    inwin: list[bool] = []
    for _, r in out.iterrows():
        ch = str(r["chrom"])
        pos = int(r["pos"])
        sub = qtls[qtls["_chrom_s"] == ch]
        if len(sub) == 0:
            near.append(None); inwin.append(False); continue
        d = (sub["qtl_pos"] - pos).abs()
        i = d.idxmin()
        near.append(str(sub.loc[i, "qtl_name"]))
        inwin.append(bool(d[i] <= window_bp))
    out["nearest_known_qtl"] = near
    out["in_known_window_500kb"] = inwin
    return out


# ---------------------------------------------------------------------
# Fusion + recall
# ---------------------------------------------------------------------


def recall_at_n(ranking: pd.DataFrame, top_n: int,
                  recoverable_qtls: set[str]) -> tuple[float, int, int]:
    """Top-N recall of known QTLs:  unique nearest_known_qtl in top N rows.

    Only counts ``recoverable_qtls`` (QTL with at least one in-window SNP).
    """
    if len(ranking) == 0 or not recoverable_qtls:
        return 0.0, 0, len(recoverable_qtls)
    top = ranking.head(top_n)
    recovered = set(top.loc[top["in_known_window_500kb"], "nearest_known_qtl"]) \
        & recoverable_qtls
    return (len(recovered) / max(len(recoverable_qtls), 1),
            len(recovered), len(recoverable_qtls))


def fit_beta_loqo(cand_with_scores: pd.DataFrame,
                   recoverable_qtls: set[str], top_n: int = RECALL_TOP_N,
                   beta_grid: tuple = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0)
                   ) -> tuple[float, dict]:
    """Leave-one-QTL-out grid search over beta to maximise mean recall@N.

    For each beta value: for each QTL, hold it out, rank by fusion_score,
    measure recall@N over the remaining QTLs. Pick beta with max mean recall.
    If fewer than 5 recoverable QTLs, lock beta=0.25 (Codex plan §5).
    """
    diagnostics: dict = {"grid": list(beta_grid), "recalls": {}}
    if len(recoverable_qtls) < 5:
        diagnostics["mode"] = "locked_beta_low_recoverable"
        diagnostics["recoverable_count"] = len(recoverable_qtls)
        return 0.25, diagnostics
    mean_recalls: dict[float, float] = {}
    for beta in beta_grid:
        recalls = []
        for qtl in recoverable_qtls:
            others = recoverable_qtls - {qtl}
            cand_with_scores["fusion_score"] = (
                cand_with_scores["z_gwas"] + beta * cand_with_scores["z_llr_abs"])
            ranked = cand_with_scores.sort_values(
                "fusion_score", ascending=False).reset_index(drop=True)
            r, _, _ = recall_at_n(ranked, top_n, others)
            recalls.append(r)
        mean_recalls[beta] = float(np.mean(recalls)) if recalls else 0.0
        diagnostics["recalls"][f"beta_{beta}"] = mean_recalls[beta]
    best_beta = max(mean_recalls, key=mean_recalls.get)
    baseline_recall = mean_recalls.get(0.0, 0.0)
    if mean_recalls[best_beta] - baseline_recall < 1e-6:
        # recall@N plateau: LOQO cannot discriminate beta. Fall back to
        # conservative beta=0.25 (Codex plan §5) — flagged in diagnostics.
        diagnostics["mode"] = "locked_beta_recall_plateau"
        diagnostics["plateau_recall"] = baseline_recall
        diagnostics["loqo_search_recalls"] = mean_recalls
        return 0.25, diagnostics
    diagnostics["mode"] = "loqo_search"
    diagnostics["best_beta"] = best_beta
    diagnostics["best_mean_recall"] = mean_recalls[best_beta]
    return best_beta, diagnostics


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description="M3.3 fuse DL prior + GWAS")
    ap.add_argument("--trait", default="days_to_emerg")
    ap.add_argument("--out-dir", default=str(
        ROOT / "results/phase3/m3_3_dl_prior/wheat_watkins"))
    ap.add_argument("--candidates", default=None,
                    help="default: <out-dir>/candidates.tsv.gz")
    ap.add_argument("--known-qtl", default=str(
        ROOT / "data/reference/wheat/known_qtl_wheat.tsv"))
    ap.add_argument("--scores-plantcad", default=None,
                    help="default: <out-dir>/dl_scores_plantcad.tsv.gz")
    ap.add_argument("--scores-agront", default=None,
                    help="default: <out-dir>/dl_scores_agront.tsv.gz")
    ap.add_argument("--top-n", type=int, default=RECALL_TOP_N)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cand_path = Path(args.candidates) if args.candidates \
        else out_dir / "candidates.tsv.gz"
    plant_path = Path(args.scores_plantcad) if args.scores_plantcad \
        else out_dir / "dl_scores_plantcad.tsv.gz"
    agr_path = Path(args.scores_agront) if args.scores_agront \
        else out_dir / "dl_scores_agront.tsv.gz"

    t0 = time.time()
    print(f"=== M3.3 fuse + evaluate — trait={args.trait} ===")

    # 1. Load
    cand = pd.read_csv(cand_path, sep="\t", compression="gzip")
    qtls = pd.read_csv(args.known_qtl, sep="\t")
    print(f"[1] {len(cand)} candidates, {len(qtls)} known QTLs")

    # 2. Per-model score load + abs-z
    n_models = 0
    score_parts = {}
    for model, path in [("plantcad", plant_path), ("agront", agr_path)]:
        if path.exists():
            df = pd.read_csv(path, sep="\t", compression="gzip")
            df = df[df["status"] == "OK"][["snp_id", "llr_signed", "llr_abs"]]
            df = df.rename(columns={
                "llr_signed": f"llr_signed_{model}",
                "llr_abs": f"llr_abs_{model}",
            })
            score_parts[model] = df
            n_models += 1
            print(f"  loaded {model}: {len(df)} scored SNPs")
        else:
            print(f"  {model}: file missing ({path}), skipping")
    if n_models == 0:
        sys.exit("ERR: no model scores found")

    # 3. Merge scores onto candidates
    merged = cand.copy()
    for model, df in score_parts.items():
        merged = merged.merge(df, on="snp_id", how="left")

    # robust z per model + ensemble
    for model in score_parts:
        merged[f"z_llr_abs_{model}"] = robust_z(
            merged[f"llr_abs_{model}"].to_numpy())
    z_cols = [f"z_llr_abs_{m}" for m in score_parts]
    merged["z_llr_abs_ensemble"] = merged[z_cols].mean(axis=1, skipna=True)
    # primary fusion uses ensemble (or single if only one model)
    merged["z_llr_abs"] = merged["z_llr_abs_ensemble"]

    # GWAS p → -log10 → z
    p_safe = merged["p"].fillna(1.0).clip(lower=1e-300).to_numpy()
    nlogp = -np.log10(p_safe)
    merged["nlog10_p"] = nlogp
    merged["z_gwas"] = robust_z(nlogp)

    # 4. Augment with known QTL distance + recoverable QTL set
    merged = known_qtl_for_snp(merged, qtls)
    recoverable_qtls = set(
        merged.loc[merged["in_known_window_500kb"], "nearest_known_qtl"]
            .dropna().astype(str))
    print(f"[4] {len(recoverable_qtls)} recoverable known QTLs in candidate set "
          f"(those with ≥1 SNP within ±{KNOWN_WINDOW_BP//1000}kb)")

    # 5. β fitting
    print("[5] LOQO grid-search for fusion beta")
    # need both z scores valid for fusion
    valid = merged.dropna(subset=["z_gwas", "z_llr_abs"]).reset_index(drop=True)
    print(f"  {len(valid)} SNPs with both z_gwas + z_llr_abs valid")
    beta, beta_diag = fit_beta_loqo(valid, recoverable_qtls, top_n=args.top_n)
    print(f"  best beta = {beta}  (diag: {beta_diag.get('mode')})")

    # 6. Final ranking with chosen beta
    merged["fusion_score"] = merged["z_gwas"].fillna(0) \
        + beta * merged["z_llr_abs"].fillna(0)

    # baseline = pure GWAS ranking
    gwas_ranked = merged.sort_values("nlog10_p", ascending=False).reset_index(drop=True)
    fused_ranked = merged.sort_values("fusion_score", ascending=False).reset_index(drop=True)

    recall_gwas, n_rec_g, n_pos = recall_at_n(
        gwas_ranked, args.top_n, recoverable_qtls)
    recall_fused, n_rec_f, _ = recall_at_n(
        fused_ranked, args.top_n, recoverable_qtls)
    lift = recall_fused - recall_gwas
    print(f"\n[6a] recall@{args.top_n}: GWAS={recall_gwas:.4f} "
          f"({n_rec_g}/{n_pos})  Fused={recall_fused:.4f} "
          f"({n_rec_f}/{n_pos})  lift={lift:+.4f}")

    # Multi-N recall sweep (v1.5 variant prioritization framing)
    recall_sweep: dict[int, dict] = {}
    for n in (100, 300, 500, 1000, 2000, 5000, 10000):
        rg, ng, _ = recall_at_n(gwas_ranked, n, recoverable_qtls)
        rf, nf, _ = recall_at_n(fused_ranked, n, recoverable_qtls)
        recall_sweep[n] = {"recall_gwas": rg, "recall_fused": rf,
                            "lift": rf - rg, "n_rec_gwas": ng, "n_rec_fused": nf}
        print(f"  recall@{n}:  GWAS={rg:.3f} ({ng}/{n_pos})  "
              f"Fused={rf:.3f} ({nf}/{n_pos})  lift={rf-rg:+.3f}")

    # β sensitivity sweep: report per-QTL rank improvement at multiple β
    print(f"\n[6b'] β sensitivity sweep — per-QTL best rank improvement")
    beta_sensitivity: dict[str, dict] = {}
    for beta_test in (0.25, 1.0, 5.0):
        s_score = merged["z_gwas"].fillna(0) + beta_test * merged["z_llr_abs"].fillna(0)
        s_ranked = merged.assign(_s=s_score).sort_values("_s", ascending=False).reset_index(drop=True)
        s_rank_map = {sid: i + 1 for i, sid in enumerate(s_ranked["snp_id"].astype(str))}
        qtl_s = merged[merged["in_known_window_500kb"] == True].copy()
        qtl_s["s_rank"] = qtl_s["snp_id"].astype(str).map(s_rank_map)
        qtl_s["g_rank"] = qtl_s["snp_id"].astype(str).map(
            {sid: i + 1 for i, sid in enumerate(gwas_ranked["snp_id"].astype(str))})
        per_q = qtl_s.groupby("nearest_known_qtl").agg(
            best_g=("g_rank", "min"), best_s=("s_rank", "min"))
        n_imp = int((per_q["best_s"] < per_q["best_g"]).sum())
        med_g = float(per_q["best_g"].median())
        med_s = float(per_q["best_s"].median())
        beta_sensitivity[f"beta_{beta_test}"] = {
            "n_qtl_improved": n_imp,
            "median_best_rank_gwas": med_g,
            "median_best_rank_fused": med_s,
            "median_rank_ratio": med_g / max(med_s, 1),
        }
        print(f"  β={beta_test}:  {n_imp}/{len(per_q)} QTLs improved, "
              f"median rank GWAS={med_g:.0f} Fused={med_s:.0f}  "
              f"(ratio {med_g/max(med_s,1):.2f}x)")

    # Per-QTL best-rank improvement (the variant-prioritization metric)
    print(f"\n[6c] per-QTL best rank (GWAS vs Fused using chosen β={beta})")
    qtl_in = merged[merged["in_known_window_500kb"] == True].copy()
    gwas_rank_map = {sid: i + 1 for i, sid in
                     enumerate(gwas_ranked["snp_id"].astype(str))}
    fused_rank_map = {sid: i + 1 for i, sid in
                      enumerate(fused_ranked["snp_id"].astype(str))}
    qtl_in["gwas_rank"] = qtl_in["snp_id"].astype(str).map(gwas_rank_map)
    qtl_in["fused_rank"] = qtl_in["snp_id"].astype(str).map(fused_rank_map)
    per_qtl = qtl_in.groupby("nearest_known_qtl").agg(
        n_in_window=("snp_id", "count"),
        best_gwas_rank=("gwas_rank", "min"),
        best_fused_rank=("fused_rank", "min"))
    per_qtl["rank_improvement"] = (
        per_qtl["best_gwas_rank"] - per_qtl["best_fused_rank"])
    per_qtl_sorted = per_qtl.sort_values("best_gwas_rank")
    print(per_qtl_sorted.to_string())
    n_qtl_improved = int((per_qtl["rank_improvement"] > 0).sum())
    median_gwas_rank = float(per_qtl["best_gwas_rank"].median())
    median_fused_rank = float(per_qtl["best_fused_rank"].median())
    print(f"  {n_qtl_improved}/{len(per_qtl)} QTLs improved with DL prior")
    print(f"  median best rank: GWAS={median_gwas_rank:.0f}, "
          f"Fused={median_fused_rank:.0f}  "
          f"(ratio {median_gwas_rank/max(median_fused_rank,1):.2f}x)")
    per_qtl_path = out_dir / "per_qtl_rank_compare.tsv"
    per_qtl_sorted.to_csv(per_qtl_path, sep="\t")
    print(f"  wrote {per_qtl_path}")

    # 7. Framing decision (charter §2.2 v1.5)
    framing = ("DL-enhanced GWAS" if lift >= 0.05
               else "variant prioritization")
    framing_evidence = {
        "recall100_lift": float(lift),
        "n_qtl_improved": int(n_qtl_improved),
        "median_rank_ratio": float(median_gwas_rank / max(median_fused_rank, 1)),
    }
    print(f"[7] framing = {framing}")
    if framing == "variant prioritization":
        print(f"     (quantitative support: {n_qtl_improved}/{len(per_qtl)} "
              f"QTLs improved by DL prior; median rank "
              f"{median_gwas_rank:.0f}→{median_fused_rank:.0f})")

    # 8. Write outputs
    ensemble_path = out_dir / "dl_prior_scores_ensemble.tsv.gz"
    keep_cols = ["snp_id","chrom","pos","subgenome","source","p","nlog10_p","z_gwas"]
    for m in score_parts:
        keep_cols += [f"llr_signed_{m}", f"llr_abs_{m}", f"z_llr_abs_{m}"]
    keep_cols += ["z_llr_abs_ensemble", "fusion_score",
                   "nearest_known_qtl", "in_known_window_500kb"]
    keep_cols = [c for c in keep_cols if c in merged.columns]
    merged[keep_cols].to_csv(ensemble_path, sep="\t", index=False, compression="gzip")
    print(f"  wrote {ensemble_path}")

    # reranked_top_loci.tsv: per-rank both methods
    top_fused = fused_ranked.head(args.top_n)[
        ["snp_id","chrom","pos","subgenome","source","p","nlog10_p","z_gwas",
         "z_llr_abs_ensemble","fusion_score","nearest_known_qtl",
         "in_known_window_500kb"]].reset_index(drop=True)
    top_fused["rank_fused"] = top_fused.index + 1
    # add gwas rank as comparison
    gwas_rank_map = {sid: i+1 for i, sid in enumerate(gwas_ranked["snp_id"].astype(str))}
    top_fused["rank_gwas"] = top_fused["snp_id"].astype(str).map(gwas_rank_map).fillna(-1).astype(int)
    top_fused["rank_delta"] = top_fused["rank_gwas"] - top_fused["rank_fused"]
    top_path = out_dir / "reranked_top_loci.tsv"
    top_fused.to_csv(top_path, sep="\t", index=False)
    print(f"  wrote {top_path}")

    summary = {
        "script": "m3_3_fuse_and_evaluate.py",
        "milestone": "M3.3 v1.5",
        "trait": args.trait,
        "n_candidates": int(len(cand)),
        "n_with_valid_z": int(len(valid)),
        "models_used": list(score_parts.keys()),
        "n_models_used": int(n_models),
        "top_n": int(args.top_n),
        "known_window_bp": int(KNOWN_WINDOW_BP),
        "recoverable_known_qtls": sorted(recoverable_qtls),
        "n_recoverable_qtls": int(len(recoverable_qtls)),
        "beta_fit_diag": beta_diag,
        "chosen_beta": float(beta),
        "recall_gwas": float(recall_gwas),
        "n_recovered_gwas": int(n_rec_g),
        "recall_fused": float(recall_fused),
        "n_recovered_fused": int(n_rec_f),
        "absolute_lift": float(lift),
        "recall_sweep": {str(k): v for k, v in recall_sweep.items()},
        "n_qtl_improved_by_dl": int(n_qtl_improved),
        "median_best_rank_gwas": float(median_gwas_rank),
        "median_best_rank_fused": float(median_fused_rank),
        "median_rank_improvement_ratio": float(
            median_gwas_rank / max(median_fused_rank, 1)),
        "beta_sensitivity_sweep": beta_sensitivity,
        "framing": framing,
        "framing_evidence": framing_evidence,
        "runtime_sec": round(time.time() - t0, 1),
        "outputs": {
            "ensemble_scores": str(ensemble_path),
            "reranked_top_loci": str(top_path),
        },
    }
    summary_path = out_dir / f"m3_3_summary_{args.trait}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=_json_default)
    print(f"\nwrote {summary_path}")
    print(f"acceptance: lift={lift:+.4f} → framing={framing}")


if __name__ == "__main__":
    main()
