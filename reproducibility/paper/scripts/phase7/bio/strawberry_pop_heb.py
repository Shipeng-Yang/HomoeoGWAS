#!/usr/bin/env python
"""Stage 4: population homoeolog expression bias (HEB), reference-bias corrected.

Per strict 4/4 quartet, per genotype, the 4 Camarosa subgenome-copy TPMs give a
within-library expression share (pA,pB,pC,pD). The naive test (share vs 0.25) is
confounded: the in-silico equal-TPM simulation shows salmon's per-copy share
deviates from 0.25 by a median of ~0.15 PURELY from reference-mapping bias, and
k-mer mappability tier does NOT predict this. So we do not test against 0.25.

Instead we use the in-silico per-quartet share `b=(bA..bD)` (the mapping-bias-only
expectation under known-balanced truth) as the BASELINE and define the
bias-corrected HEB signal per copy and genotype as d_i(g) = p_i(g) - b_i. Real
homoeolog bias = consistent deviation of the OBSERVED share from this mapping
baseline, not from uniformity.

Stats (per Codex review): all four copies are tested vs 0 (pre-specified, no
winner-take-all selection), BH-FDR across copies x quartets; conserved/variable
require an effect-size floor so low-count argmax noise does not inflate classes.

Framing: descriptive population HEB atlas across an octoploid ripe-fruit panel;
expression assigned to Camarosa homoeolog models; single-reference + baseline-
correction caveats; NOT a GWAS discovery. Runs after Stage 3 + in-silico.
"""
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import stats

PROJ = Path("/mnt/7302share/fast_ysp/U7_GWAS")
IN = PROJ / "results/phase7/bio_strawberry"
MIN_TPM = 1.0           # min quartet total TPM in a genotype to use it
MIN_GENOS = 30          # min covered genotypes to evaluate a quartet
EFFECT_MIN = 0.10       # min |mean corrected deviation| for a real bias call
SIGN_CONSISTENCY = 0.70  # frac genotypes with same-sign deviation => conserved
VAR_SD_MIN = 0.10       # min across-genotype SD of corrected dominant => variable
SUBS = ["A", "B", "C", "D"]


def load_tsv(path):
    with open(path) as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


# TPM matrix (length-normalised expression; right scale for expression share)
tpm = {}
with open(IN / "expr_tpm_gene_x_genotype.tsv") as fh:
    rd = csv.reader(fh, delimiter="\t")
    genos = next(rd)[1:]
    for row in rd:
        tpm[row[0]] = np.array([float(x) for x in row[1:]])
print(f"[heb] TPM matrix {len(tpm)} genes x {len(genos)} genotypes")

quartet_rows = [r for r in load_tsv(IN / "quartets.tsv")
                if all(r[s] != "." for s in SUBS)]
tier = {i: r["tier"] for i, r in enumerate(load_tsv(IN / "quartet_mappability.tsv"))}
# in-silico baseline shares b_i per quartet (keyed by quartet_idx) + baseline
# severity for the confidence tier.
base = {}
bsev = {}
for r in load_tsv(IN / "insilico_fairness.tsv"):
    qi = int(r["quartet_idx"])
    base[qi] = np.array([float(r[f"p{s}"]) for s in SUBS])
    bsev[qi] = float(r["max_dev_from_0.25"])


def conf_tier(md):
    return "fair" if md <= 0.10 else "moderate" if md <= 0.25 else "severe"


rows = []
for qi, r in enumerate(quartet_rows):
    genes = [r[s] for s in SUBS]
    if any(g not in tpm for g in genes) or qi not in base:
        continue
    mat = np.vstack([tpm[g] for g in genes])           # 4 x genotypes
    tot = mat.sum(axis=0)
    keep = tot >= MIN_TPM
    if keep.sum() < MIN_GENOS:
        continue
    props = (mat[:, keep] / tot[keep]).T               # n_keep x 4 observed shares
    dev = props - base[qi]                              # bias-corrected deviation
    mean_d = dev.mean(axis=0)
    sd_d = dev.std(axis=0)
    # all four copies tested vs 0 (pre-specified, no winner selection)
    tps = [float(stats.ttest_1samp(dev[:, k], 0.0).pvalue) for k in range(4)]
    dom = int(np.abs(mean_d).argmax())                 # report-only label
    eff = float(np.abs(mean_d[dom]))
    sign_consistency = float((np.sign(dev[:, dom]) == np.sign(mean_d[dom])).mean())
    rows.append({
        "quartet_idx": qi, "group": r["group"], "n_geno": int(keep.sum()),
        "dom": SUBS[dom], "mean_dev_dom": float(mean_d[dom]), "eff": eff,
        "sd_dom": float(sd_d[dom]), "sign_consistency": sign_consistency,
        "t_p_dom": tps[dom], "min_t_p": min(tps),
        "tier": tier.get(qi, "?"), "base_sev": conf_tier(bsev.get(qi, 1.0)),
        "_devdom": dev[:, dom],
    })

# BH-FDR across the per-quartet dominant-copy tests
pv = np.array([x["t_p_dom"] for x in rows])
order = pv.argsort()
m = len(pv)
q = np.empty(m)
prev = 1.0
for rank in range(m - 1, -1, -1):
    idx = order[rank]
    prev = min(prev, pv[idx] * m / (rank + 1))
    q[idx] = prev
for x, qq in zip(rows, q, strict=True):
    x["q_dom"] = float(qq)
    del x["_devdom"]


def classify(x):
    if x["eff"] < EFFECT_MIN or x["q_dom"] >= 0.05:
        return "balanced"
    if x["sign_consistency"] >= SIGN_CONSISTENCY:
        return "conserved_HEB"
    if x["sd_dom"] >= VAR_SD_MIN:
        return "variable_HEB"
    return "balanced"


for x in rows:
    x["class"] = classify(x)

# headline set excludes 'severe' baseline-bias quartets (correction least reliable)
reliable = [x for x in rows if x["base_sev"] in ("fair", "moderate")]
conserved = [x for x in reliable if x["class"] == "conserved_HEB"]
variable = [x for x in reliable if x["class"] == "variable_HEB"]
balanced = [x for x in reliable if x["class"] == "balanced"]
dom_dist = Counter(x["dom"] for x in conserved)

with open(IN / "pop_heb_quartets.tsv", "w") as o:
    cols = ["quartet_idx", "group", "n_geno", "dom", "mean_dev_dom", "eff",
            "sd_dom", "sign_consistency", "t_p_dom", "q_dom", "class", "tier",
            "base_sev"]
    w = csv.DictWriter(o, fieldnames=cols, delimiter="\t", extrasaction="ignore")
    w.writeheader()
    for x in sorted(rows, key=lambda z: z["q_dom"]):
        w.writerow({k: (f"{x[k]:.4f}" if isinstance(x[k], float) else x[k]) for k in cols})

summary = {
    "tool": "strawberry_pop_heb",
    "framing": "reference-bias-corrected population HEB across octoploid ripe-fruit "
               "panel; observed per-copy TPM share compared to the in-silico "
               "mapping-bias baseline (NOT to 0.25); expression assigned to "
               "Camarosa homoeolog models; descriptive atlas, NOT GWAS; single-"
               "reference + baseline-correction caveats.",
    "params": {"min_tpm": MIN_TPM, "min_genos": MIN_GENOS, "effect_min": EFFECT_MIN,
               "sign_consistency": SIGN_CONSISTENCY, "var_sd_min": VAR_SD_MIN},
    "n_quartets_evaluated": len(rows),
    "n_reliable_baseline": len(reliable),
    "n_conserved_HEB": len(conserved),
    "n_variable_HEB": len(variable),
    "n_balanced": len(balanced),
    "conserved_subgenome_dominance": dict(dom_dist),
    "caveat": "in-silico baseline corrects reference-structure mapping bias but "
              "not allele-specific mapping differences of real panel individuals "
              "vs Camarosa; 'severe' baseline quartets excluded from headline.",
}
(IN / "pop_heb_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
print(f"[done] -> {IN}/pop_heb_quartets.tsv + pop_heb_summary.json")
