#!/usr/bin/env python3
"""P0 deliverable: 4-panel permutation-calibration summary table + figure.
Harvests the already-persisted per-pair lambda_gc (observed + permutation median/IQR) from
each panel's deploy JSON into tables/lambda_perm_summary.tsv, and builds a 3-panel calibration figure:
(A) wheat per-pair interaction QQ, (B) cotton (Hit2 length-uniformity) per-pair QQ — both from the
already-dumped OBSERVED p-vectors (NOT permutation-empirical p; labelled as such) — and (C) a λ forest
of λ_obs vs λ_perm(median, IQR) across all 4 panels × modes, the cross-ploidy calibration claim.

Honesty: QQ = observed scan p, explicitly not permutation QQ; the sparse boundary panels
(rapeseed/oat, G≈2–69) have too few callable pairs for an informative QQ and are shown only in the λ
forest; the degenerate rapeseed body_same mode (G=2) is flagged and excluded from the forest.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path("/mnt/7302share/fast_ysp/U7_GWAS")
OUT_TBL = ROOT / "tables/lambda_perm_summary.tsv"
OUT_FIG = ROOT / "figures/calibration_4panel"
WHEAT_RANK = ROOT / "results/phase7/bio_wheat/wheat_GW_ranking_flank_INT.tsv"
COTTON_RANK = ROOT / "results/phase7/bio/cotton_ranking/interact_length_uniformity_BLUE_ranking_pairwise_INT.tsv"


def _med_iqr(xs):
    xs = np.array([x for x in xs if x is not None and np.isfinite(x)], float)
    if xs.size == 0:
        return (None, None, None, 0)
    return (float(np.median(xs)), float(np.percentile(xs, 25)), float(np.percentile(xs, 75)), int(xs.size))


def harvest():
    """One row per (panel, mode): median λ_obs, median λ_perm + IQR across the per-pair/per-trait units."""
    rows = []
    # cotton + rapeseed: by_trait/{t}/INT_primary/{lambda_gc_obs, lambda_gc_perm_median}
    for panel, base, ploidy in [("cotton", "results/phase7/bio_full", "4n AADD"),
                                ("rapeseed", "results/phase7/bio_rapeseed", "4n AACC")]:
        for mode in ("body_same", "body_cross", "flank_same", "flank_cross"):
            f = ROOT / base / f"deploy_{mode}.json"
            if not f.exists():
                continue
            d = json.loads(f.read_text())
            bt = d.get("by_trait", {})
            lo = [v["INT_primary"]["lambda_gc_obs"] for v in bt.values() if "INT_primary" in v]
            lp = [v["INT_primary"]["lambda_gc_perm_median"] for v in bt.values() if "INT_primary" in v]
            G = d.get("n_pairs") or d.get("G") or d.get("n_triad")
            mo, mp = _med_iqr(lo), _med_iqr(lp)
            rows.append(dict(panel=panel, ploidy=ploidy, mode=mode, n_unit=mp[3], G=G,
                             lambda_obs_med=mo[0], lambda_perm_med=mp[0],
                             lambda_perm_q25=mp[1], lambda_perm_q75=mp[2]))
    # oat: results/{trait}/INT_primary/calibration/{AC,AD,CD}/lambda_perm_median + per_family/.../lambda_gc
    for mode in ("flank", "body"):
        f = ROOT / f"results/phase7/bio_oat/deploy_oat_{mode}.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        lo, lp = [], []
        for tv in d.get("results", {}).values():
            cal = tv.get("INT_primary", {}).get("calibration", {})
            pf = tv.get("INT_primary", {}).get("per_family", {})
            for fam in ("AC", "AD", "CD"):
                if "lambda_perm_median" in cal.get(fam, {}):
                    lp.append(cal[fam]["lambda_perm_median"])
                if "lambda_gc" in pf.get(fam, {}):
                    lo.append(pf[fam]["lambda_gc"])
        mo, mp = _med_iqr(lo), _med_iqr(lp)
        rows.append(dict(panel="oat", ploidy="6n AACCDD", mode=mode + "_all", n_unit=mp[3], G=None,
                         lambda_obs_med=mo[0], lambda_perm_med=mp[0],
                         lambda_perm_q25=mp[1], lambda_perm_q75=mp[2]))
    # wheat: INT_primary/pairwise_emp/{AB,AD,BD}/lambda_gc_perm_median + pairwise/.../lambda_gc
    f = ROOT / "results/phase7/bio_wheat/deploy_wheat_GW_CURATED_flank.json"
    d = json.loads(f.read_text())
    pe = d["INT_primary"]["pairwise_emp"]
    pw = d["INT_primary"]["pairwise"]
    lp = [pe[p]["lambda_gc_perm_median"] for p in ("AB", "AD", "BD")]
    lo = [pw[p]["lambda_gc"] for p in ("AB", "AD", "BD")]
    mo, mp = _med_iqr(lo), _med_iqr(lp)
    rows.append(dict(panel="wheat", ploidy="6n AABBDD", mode="flank_GW(heading)", n_unit=mp[3],
                     G=d.get("n_triad"), lambda_obs_med=mo[0], lambda_perm_med=mp[0],
                     lambda_perm_q25=mp[1], lambda_perm_q75=mp[2]))
    df = pd.DataFrame(rows)
    df = df[df["n_unit"] > 0].reset_index(drop=True)  # drop modes with no permutation calibration
    # honest degeneracy flag: a mode with G<5 callable units gives an uninterpretable λ
    df["note"] = np.where((df["panel"] == "rapeseed") & (df["mode"] == "body_same"),
                          "DEGENERATE: G=2 callable pairs, λ uninterpretable (boundary panel)", "")
    return df


def _qq(ax, p, label, color):
    p = np.sort(np.asarray(p, float))
    p = p[np.isfinite(p) & (p > 0)]
    n = p.size
    exp = -np.log10((np.arange(1, n + 1) - 0.5) / n)
    obs = -np.log10(p)
    lim = max(exp.max(), obs.max()) * 1.05
    ax.plot([0, lim], [0, lim], "--", color="0.6", lw=1)
    ax.scatter(exp, obs, s=6, color=color, alpha=0.5, edgecolors="none")
    ax.set_xlabel("expected $-\\log_{10}p$")
    ax.set_ylabel("observed $-\\log_{10}p$")
    ax.set_title(label, fontsize=9)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)


def main():
    OUT_TBL.parent.mkdir(parents=True, exist_ok=True)
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    df = harvest()
    df.to_csv(OUT_TBL, sep="\t", index=False, float_format="%.4f")
    print(f"wrote {OUT_TBL}\n{df.to_string(index=False)}")

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    # (A) wheat per-pair QQ (pool AB/AD/BD across 12768 triads)
    wr = pd.read_csv(WHEAT_RANK, sep="\t", usecols=["p_AB", "p_AD", "p_BD"])
    wp = np.concatenate([wr["p_AB"].values, wr["p_AD"].values, wr["p_BD"].values])
    _qq(axes[0], wp, f"A  wheat heading — observed scan p\n(6n; AB/AD/BD pooled, G={len(wr)}×3)", "#1f77b4")
    # (B) cotton Hit2 length-uniformity per-pair QQ (representative dense A-D panel, not best-QQ)
    cr = pd.read_csv(COTTON_RANK, sep="\t", usecols=["p_interaction"])
    _qq(axes[1], cr["p_interaction"].values,
        f"B  cotton length-uniformity — observed scan p\n(4n; representative dense A–D, G={len(cr)})", "#d62728")

    # (C) λ forest: λ_perm median (IQR) + λ_obs, all panels×modes (exclude degenerate G=2)
    fdf = df[df["note"] == ""].copy().reset_index(drop=True)
    fdf = fdf.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(fdf))
    ax = axes[2]
    ax.axvline(1.0, color="0.6", ls="--", lw=1)
    pal = {"cotton": "#d62728", "wheat": "#1f77b4", "rapeseed": "#2ca02c", "oat": "#9467bd"}
    for i, r in fdf.iterrows():
        lo, hi = r["lambda_perm_q25"], r["lambda_perm_q75"]
        ax.plot([lo, hi], [i, i], color=pal[r["panel"]], lw=2, alpha=0.7)
        ax.scatter([r["lambda_perm_med"]], [i], color=pal[r["panel"]], s=28, zorder=3,
                   label=r["panel"] if r["panel"] not in ax.get_legend_handles_labels()[1] else "")
        ax.scatter([r["lambda_obs_med"]], [i], facecolors="none", edgecolors=pal[r["panel"]],
                   s=40, marker="D", zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r['panel']} {r['mode']} (G={int(r['G']) if pd.notna(r['G']) else '36×3'})"
                        for _, r in fdf.iterrows()], fontsize=6.5)
    ax.set_xlabel("$\\lambda_{GC}$  (● perm median + IQR;  ◇ observed)")
    ax.set_title("C  permutation-calibrated $\\lambda$ across\nploidies & species (dashed = 1.0)",
                 fontsize=9)
    ax.set_xlim(0.7, 1.3)
    fig.text(0.5, -0.02, "Panels A–B: observed-scan p-values (illustrative diagnostics for the two "
             "dense, discovery-feasible panels), NOT permutation-empirical p. Panel C is the calibration "
             "evidence. rapeseed body_same (G=2) is tabulated in lambda_perm_summary.tsv but omitted "
             "from the forest (λ degenerate at G<5).", ha="center", fontsize=6, color="0.35")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT_FIG}.{ext}", dpi=200, bbox_inches="tight")
    print(f"wrote {OUT_FIG}.{{pdf,png}}")
    print(f"  wheat per-pair λ(pooled QQ) median p-points={len(wp)}; cotton pairs={len(cr)}")


if __name__ == "__main__":
    main()
