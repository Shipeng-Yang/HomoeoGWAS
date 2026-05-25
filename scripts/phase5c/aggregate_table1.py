"""Phase 5c Week 2 — aggregate per-panel GBLUP JSONs into paper Table 1.

Reads ``results/phase5c/<panel>/<trait>/gblup_<panel>_<trait>.json`` for each
of the 4 Tier-1 panels and emits:

  Table 1 TSV (panel × trait × {tier0/1/2 r², r, Δr², CI, top10_enrich})
  bar plot PNG (paper main figure)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "results/phase5c/aggregate"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_PANELS = [
    ("wheat_watkins", "days_to_emerg"),
    ("cotton_hebau", "fiber_length_BLUE"),
    ("rapeseed_horvath2020", "bloom_50pct"),
    ("strawberry_pincot2018", "mean_score"),
]


def collect_rows(panels: list[tuple[str, str]]) -> list[dict]:
    rows = []
    for panel, trait in panels:
        json_path = ROOT / f"results/phase5c/{panel}/{trait}/gblup_{panel}_{trait}.json"
        if not json_path.exists():
            print(f"  SKIP: missing {json_path}")
            continue
        with json_path.open() as f:
            d = json.load(f)
        r = d["result"]
        for tname, t in r["tiers"].items():
            delta = r["delta_vs_tier0"].get(tname, {})
            rows.append({
                "panel": d["panel"],
                "trait": d["trait"],
                "n_samples": d["n_samples"],
                "n_subgenomes": len(d["subgenomes"]),
                "n_pool_snp": d.get("n_pool_snp", "?"),
                "tier": tname,
                "mean_r2": t["mean_r2"],
                "se_r2": t["se_r2"],
                "ci_lo_vs_tier0": delta.get("ci_lo", float("nan")),
                "ci_hi_vs_tier0": delta.get("ci_hi", float("nan")),
                "delta_r2_vs_tier0": delta.get("delta_r2_mean", float("nan")),
                "sig_95_vs_tier0": delta.get("significant_95", False),
                "mean_r_pearson": t["mean_r"],
                "mean_top10_enrich": t["mean_top10_enrichment"],
                "n_dropped_kernels": t.get("n_dropped_kernels", 0),
            })
    return rows


def write_table1(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        print("  WARN: empty rows — no panel JSON found")
        return df
    # Reshape to one row per panel-trait, columns per tier
    pivot_r2 = df.pivot_table(index=["panel","trait","n_samples","n_subgenomes"],
                              columns="tier", values="mean_r2").reset_index()
    pivot_top10 = df.pivot_table(index=["panel","trait"],
                                  columns="tier", values="mean_top10_enrich").reset_index()
    delta = df[df["tier"] != "tier0"][[
        "panel","trait","tier","delta_r2_vs_tier0","ci_lo_vs_tier0","ci_hi_vs_tier0","sig_95_vs_tier0"
    ]]
    out_path = OUT_DIR / "table1_long.tsv"
    df.to_csv(out_path, sep="\t", index=False)
    print(f"  wrote long-form TSV: {out_path}")
    wide_path = OUT_DIR / "table1_wide_r2.tsv"
    pivot_r2.to_csv(wide_path, sep="\t", index=False)
    print(f"  wrote wide-form r² TSV: {wide_path}")
    delta_path = OUT_DIR / "table1_delta.tsv"
    delta.to_csv(delta_path, sep="\t", index=False)
    print(f"  wrote delta TSV: {delta_path}")
    return df


def plot_table1(df: pd.DataFrame):
    if df.empty:
        return
    tiers_order = ["tier0", "tier1", "tier2"]
    panels = df[["panel", "trait"]].drop_duplicates().itertuples(index=False, name=None)
    panels = list(panels)
    fig, ax = plt.subplots(figsize=(max(8, 2 * len(panels)), 5))
    x = np.arange(len(panels))
    width = 0.27
    colors = {"tier0": "#9aa0a6", "tier1": "#1a73e8", "tier2": "#d93025"}
    for k, tier in enumerate(tiers_order):
        bars = []
        errs = []
        for p, t in panels:
            row = df[(df["panel"] == p) & (df["trait"] == t) & (df["tier"] == tier)]
            if row.empty:
                bars.append(np.nan); errs.append(0)
            else:
                bars.append(row["mean_r2"].iloc[0])
                # use 1 SE as visual error bar (CI is paired-bootstrap of Δ, not absolute)
                errs.append(row["se_r2"].iloc[0])
        ax.bar(x + (k - 1) * width, bars, width, label=tier, color=colors[tier], yerr=errs, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{p}\n{t}" for p, t in panels], fontsize=8)
    ax.set_ylabel("CV Pearson r²")
    ax.set_title("HomoeoGWAS GBLUP across 4 allopolyploid panels (5-fold × 20 repeats)")
    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()
    fig_path = OUT_DIR / "table1_barplot.png"
    plt.savefig(fig_path, dpi=200)
    print(f"  wrote {fig_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panels", default=None, help="optional comma-sep panel:trait pairs")
    args = ap.parse_args()
    if args.panels:
        panels = []
        for entry in args.panels.split(","):
            p, t = entry.split(":")
            panels.append((p.strip(), t.strip()))
    else:
        panels = DEFAULT_PANELS
    print(f"aggregating {len(panels)} panels …")
    rows = collect_rows(panels)
    df = write_table1(rows)
    plot_table1(df)
    print(f"\n=== Final Table 1 (paper-grade) ===")
    if not df.empty:
        view = df.copy()
        for c in ("mean_r2","se_r2","delta_r2_vs_tier0","ci_lo_vs_tier0","ci_hi_vs_tier0"):
            view[c] = view[c].astype(float).round(4)
        cols = ["panel","trait","n_samples","tier","mean_r2","se_r2",
                "delta_r2_vs_tier0","ci_lo_vs_tier0","ci_hi_vs_tier0","sig_95_vs_tier0",
                "mean_top10_enrich","n_dropped_kernels"]
        print(view[cols].to_string(index=False))


if __name__ == "__main__":
    main()
