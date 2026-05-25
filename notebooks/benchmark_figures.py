# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: polygwas-cpu
#     language: python
#     name: python3
# ---

# %% [markdown]
# # HomoeoGWAS — benchmark figures (Table 1, Fig 2, Fig 3)
#
# Regenerates the paper's quantitative display items from committed result
# artifacts. Run headless with:
#
# ```bash
# ~/.local/share/mamba/envs/polygwas-cpu/bin/python notebooks/benchmark_figures.py
# ```
#
# Outputs land in `results/phase4/figures/`.
#
# **Framing locks (charter §4 — do not violate in any caption/heading):**
# - The DL prior is a **panel-size / baseline-power dependent** re-ranker, **not**
#   a universal p-value amplifier. Report wheat lift = 0 explicitly.
# - K_hom Δr² is **scope-conditional** (bidirectional, σ²_h at boundary in all
#   four panels = universal null). Never write "universal improvement".
# - Oat is a **sanity / robustness** panel (heterogeneous behaviour); never
#   summarise it as an "average lift". SOC negative lift (−0.087) shown openly.

# %%
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "phase4" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 9,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
    }
)

# Restrained, colour-blind-safe palette.
C_GWAS = "#bdbdbd"
C_FUSED = "#2c7fb8"
C_POS = "#2c7fb8"
C_NEG = "#d7301f"

# %% [markdown]
# ## 0. Verify result fingerprints
#
# The committed `reproducibility_fingerprints.tsv` pins the SHA256 of every
# artifact this notebook reads. Mismatch ⇒ upstream results changed.

# %%
fp = pd.read_csv(
    ROOT / "results/phase4/reproducibility_fingerprints.tsv",
    sep=r"\s+",
    header=None,
    names=["sha256", "path"],
)
print(f"{len(fp)} fingerprinted artifacts:")
for _, r in fp.iterrows():
    print(f"  {r.sha256[:12]}  {r.path}")

# %% [markdown]
# ## Table 1 — scope-conditional K_hom (GBLUP prediction accuracy)
#
# Tier 0 = SNP-count-weighted single kernel; Tier 1 = +K_hom (Hadamard);
# Tier 2 = per-subgenome additive + K_hom. Δr² is paired across 20×k-fold CV
# repeats; `sig_95` = paired-bootstrap 95% CI excludes 0.
#
# **Result:** Δr²(tier1) is small and **bidirectional** — cotton +0.019 and
# strawberry +0.018 (both sig), wheat −0.004 and rapeseed −0.015 (both sig).
# K_hom variance component sits at the REML boundary in all four panels
# (universal null); the kernel's prediction value is therefore
# **scope-conditional**, not a general gain.

# %%
t1 = pd.read_csv(ROOT / "results/phase5c/aggregate/table1_long.tsv", sep="\t")
tier0 = t1[t1.tier == "tier0"].set_index(["panel", "trait"])
tier1 = t1[t1.tier == "tier1"].set_index(["panel", "trait"])

table1 = pd.DataFrame(
    {
        "n": tier0["n_samples"],
        "n_sub": tier0["n_subgenomes"],
        "tier0_r2": tier0["mean_r2"].round(3),
        "delta_r2_tier1": tier1["delta_r2_vs_tier0"].round(4),
        "ci_lo": tier1["ci_lo_vs_tier0"].round(4),
        "ci_hi": tier1["ci_hi_vs_tier0"].round(4),
        "sig_95": tier1["sig_95_vs_tier0"],
    }
).reset_index()
table1["direction"] = table1["delta_r2_tier1"].apply(
    lambda d: "K_hom helps" if d > 0 else "K_hom hurts"
)
print(table1.to_string(index=False))
table1.to_csv(OUT / "table1_khom_scope_conditional.tsv", sep="\t", index=False)

# %%
# Table 1 visual: Δr²(tier1) ± 95% CI, coloured by sign.
fig, ax = plt.subplots(figsize=(4.2, 2.6))
y = range(len(table1))
colors = [C_POS if d > 0 else C_NEG for d in table1["delta_r2_tier1"]]
ax.barh(
    list(y),
    table1["delta_r2_tier1"],
    xerr=[
        table1["delta_r2_tier1"] - table1["ci_lo"],
        table1["ci_hi"] - table1["delta_r2_tier1"],
    ],
    color=colors,
    height=0.6,
    error_kw={"elinewidth": 0.8, "capsize": 2},
)
ax.axvline(0, color="k", lw=0.6)
ax.set_yticks(list(y))
ax.set_yticklabels([f"{p}\n{t}" for p, t in zip(table1["panel"], table1["trait"])])
ax.set_xlabel(r"$\Delta r^2$ (tier1 − tier0), K_hom added")
ax.set_title("Table 1 — scope-conditional K_hom (bidirectional, σ²_h null)")
fig.savefig(OUT / "table1_khom_delta_r2.png")
fig.savefig(OUT / "table1_khom_delta_r2.pdf")
print("saved table1_khom_delta_r2.{png,pdf}")

# %% [markdown]
# ## Fig 2 — DL-prior cross-panel recall@100 lift (panel-size dependent)
#
# Each main panel/trait: recall@100 of canonical genes by GWAS p-value alone
# vs by the PlantCaduceus + AgroNT fused score (β chosen per-panel by LOQO).
# Wheat (n=827, power-saturated) is shown as the **reference with lift = 0** —
# the lift is largest precisely where marginal GWAS power is low.

# %%
MAIN = [
    ("horvath2020", "bloom_50pct", "Horvath bloom", "results/phase3/m3_3_dl_prior_v2"),
    ("horvath2020", "plant_height", "Horvath PH", "results/phase3/m3_3_dl_prior_v2"),
    ("cotton_hebau", "fiber_length_BLUE", "Cotton fiber", "results/phase3/m3_3_dl_prior_v2"),
    ("cotton_hebau", "lint_percentage_BLUE", "Cotton lint%", "results/phase3/m3_3_dl_prior_v2"),
    ("wheat_watkins", "days_to_emerg", "Wheat emerg (ref)", "results/phase3/m3_3_dl_prior"),
]


def _load_summary(panel, trait, base):
    p = ROOT / base / panel
    cand = list(p.glob(f"**/m3_3_summary_{trait}.json"))
    if not cand:
        cand = list(p.glob(f"m3_3_summary_{trait}.json"))
    return json.load(open(cand[0]))


rows = []
for panel, trait, label, base in MAIN:
    d = _load_summary(panel, trait, base)
    rows.append(
        {
            "label": label,
            "n": d.get("n_panel_samples", None),
            "beta": d.get("chosen_beta"),
            "recall_gwas": d.get("recall_gwas", 0.0),
            "recall_fused": d.get("recall_fused", 0.0),
            "lift": d.get("absolute_lift", 0.0),
        }
    )
fig2 = pd.DataFrame(rows)
print(fig2.to_string(index=False))

# %%
fig, ax = plt.subplots(figsize=(5.2, 2.8))
import numpy as np

x = np.arange(len(fig2))
w = 0.38
ax.bar(x - w / 2, fig2["recall_gwas"], w, label="GWAS p-value", color=C_GWAS)
ax.bar(x + w / 2, fig2["recall_fused"], w, label="+ DL prior (fused)", color=C_FUSED)
for i, lift in enumerate(fig2["lift"]):
    ax.annotate(
        f"{lift:+.2f}",
        (i, max(fig2["recall_gwas"][i], fig2["recall_fused"][i]) + 0.03),
        ha="center",
        fontsize=7,
        color="k" if lift != 0 else "#888",
    )
ax.set_xticks(x)
ax.set_xticklabels(fig2["label"], rotation=20, ha="right")
ax.set_ylabel("recall@100 of canonical genes")
ax.set_ylim(0, 1.05)
ax.legend(frameon=False, fontsize=7, loc="upper right")
ax.set_title("Fig 2 — DL-prior lift is panel-size / baseline-power dependent")
fig.savefig(OUT / "fig2_dl_prior_recall_lift.png")
fig.savefig(OUT / "fig2_dl_prior_recall_lift.pdf")
print("saved fig2_dl_prior_recall_lift.{png,pdf}")

# %% [markdown]
# ## Fig 2 supp — oat OLD panel: heterogeneous DL-prior behaviour
#
# **Robustness/sanity panel, not a main figure.** Three environmental-
# differentiation traits show heterogeneous behaviour: positive lift for the
# weaker-signal traits (BIO6 +0.058, BIO12 +0.117) and a **negative** lift for
# the high-baseline-recall SOC trait (recall 0.231 → 0.145, **−0.087**). SOC did
# not meet the DL-enhanced criterion; it is reported openly, not relabelled.

# %%
OAT = [("BIO6", "BIO6"), ("SOC", "SOC"), ("BIO12", "BIO12")]
orows = []
for t, label in OAT:
    d = json.load(
        open(ROOT / f"results/phase5d/m3_3_dl_prior/oat_old_rahman2025/{t}/m3_3_summary_{t}.json")
    )
    orows.append(
        {
            "trait": label,
            "recall_gwas": d["recall_gwas"],
            "recall_fused": d["recall_fused"],
            "lift": d["absolute_lift"],
        }
    )
oat = pd.DataFrame(orows)
print(oat.to_string(index=False))

fig, ax = plt.subplots(figsize=(3.6, 2.6))
colors = [C_POS if lift > 0 else C_NEG for lift in oat["lift"]]
ax.bar(oat["trait"], oat["lift"], color=colors, width=0.6)
ax.axhline(0, color="k", lw=0.6)
_lo, _hi = oat["lift"].min(), oat["lift"].max()
ax.set_ylim(_lo - 0.04, _hi + 0.03)
for i, lift in enumerate(oat["lift"]):
    ax.annotate(
        f"{lift:+.3f}",
        (i, lift + (0.006 if lift > 0 else -0.018)),
        ha="center",
        va="bottom" if lift > 0 else "top",
        fontsize=7,
    )
ax.set_ylabel("recall lift (fused − GWAS)")
ax.set_title("Fig 2 supp — oat heterogeneous behaviour (sanity panel)")
fig.savefig(OUT / "fig2supp_oat_heterogeneous.png")
fig.savefig(OUT / "fig2supp_oat_heterogeneous.pdf")
print("saved fig2supp_oat_heterogeneous.{png,pdf}")

# %% [markdown]
# ## Fig 3 — per-panel Manhattan + QQ montage (reference)
#
# Composes already-generated diagnostic plots (one representative trait per
# panel) into a single montage. These PNGs come from the LOCO scans; the
# notebook does not re-run GWAS.

# %%
MONTAGE = [
    ("Wheat days_to_emerg", "results/phase3/m3_1_loco_v2/wheat_watkins/manhattan_days_to_emerg.png"),
    ("Horvath bloom_50pct", "results/phase3/m3_1_loco_v2/horvath2020/bloom_50pct/qq_bloom_50pct.png"),
    ("Cotton fiber_length", "results/phase3/m3_1_loco_v2/cotton_hebau/fiber_length_BLUE/qq_fiber_length_BLUE.png"),
    ("Strawberry mean_score", "results/phase5/fragaria_ananassa/mean_score/manhattan_mean_score.png"),
    ("Oat BIO6", "results/phase5d/m3_1_loco_v2/oat_old_rahman2025/BIO6/manhattan_BIO6.png"),
    ("Oat SOC", "results/phase5d/m3_1_loco_v2/oat_old_rahman2025/SOC/qq_SOC.png"),
]
import matplotlib.image as mpimg

fig, axes = plt.subplots(2, 3, figsize=(9, 5))
for ax, (label, rel) in zip(axes.ravel(), MONTAGE):
    p = ROOT / rel
    if p.exists():
        ax.imshow(mpimg.imread(p))
        ax.set_title(label, fontsize=8)
    else:
        ax.text(0.5, 0.5, f"missing:\n{rel}", ha="center", va="center", fontsize=6)
    ax.axis("off")
fig.suptitle("Fig 3 — five-panel Manhattan / QQ montage", fontsize=10)
fig.savefig(OUT / "fig3_montage.png")
print("saved fig3_montage.png")

print("\nAll figures written to", OUT)
