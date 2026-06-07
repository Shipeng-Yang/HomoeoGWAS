#!/usr/bin/env python3
"""Wheat landrace MULTI-TRAIT genome-wide homoeolog-triad interaction scan (pre-registered:
reports/multitrait_prereg.md). ETL of the JIC Watkins landrace phenotype workbook -> 14 continuous
environment-averaged traits aligned to our 827 WATDE genotype, then ONE genotype extraction +
per-gene capped burdens + subgenome GRMs (trait-independent, built once), then a per-trait
observed-only scan: 3 pairwise (A-B/A-D/B-D) + per-triad ACAT over ~12,768 curated 1:1:1 HCTriads.

CALIBRATION + BREADTH is the claim (lambda_GC per trait; anchors=positive control). Discovery uses
asymptotic p (perm-B=0, fast); candidates (triad ACAT p<1e-4 OR any pairwise p<1e-5) are flagged for
follow-up permutation. emergence chr1 B-D is the positive-control anchor (must reproduce ~2.4e-6).
Multiplicity: within-trait Bonferroni 0.05/G on triad ACAT + BH-FDR across all (trait x triad) ACAT p.
"""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/mnt/7302share/fast_ysp/U7_GWAS")
LAND = ROOT / "data/raw/wheat/phenotype/Natural_Populations/Watkins_Collection_WGIN_WISP_DFW_watseq_phenotype_data_JIC.xlsx"
GENO_SAMPLES = ROOT / "data/processed/wheat/pheno_clean.tsv"
HCTRIADS = ROOT / "data/raw/expression/wheat/HCTriads.csv"
OUT = ROOT / "results/phase7/bio_wheat/multitrait"
PHENO_OUT = ROOT / "data/processed/wheat/pheno_landrace_multitrait.tsv"
SHEETS = ["WGIN_Watkins_JIC_CFLN06", "WISP_Watkins_JIC_CFLN10", "DFW_Watkins_JI_CFLN14",
          "DFW_Watkins_JI_CFLN20"]
# clean trait -> regex matching the per-env summary column(s); multi-match cols are AVERAGED per accession
TRAITS = {
    "heading_date": r"Hd_dto", "plant_height": r"PH_M_cm", "flag_leaf_sen": r"FleafSen",
    "stem_sen": r"StemSen", "grain_weight_1000": r"GW_M_g1000", "grain_surfarea": r"GSurfA",
    "grain_width": r"^GWid", "grain_length": r"^GLng", "grain_hardness_pct": r"GHrd_M_pct",
    "grain_moisture": r"GMoi", "grain_diameter": r"GDia", "grain_protein_NIR": r"Gprot_NIR",
    "grain_starch_NIR": r"GStarch_NIR", "grain_fibre_NIR": r"GFibre_NIR",
}
MIN_N = 500
N_JOBS = 96

_s4 = importlib.util.spec_from_file_location("w4", str(ROOT / "scripts/phase7/bio_wheat/04w_wheat_genomewide.py"))
_w4 = importlib.util.module_from_spec(_s4)
_s4.loader.exec_module(_w4)
_w2 = _w4._w2  # 02w deploy machinery (whiten3, pairwise_pvals, acat, rank_int, scols_safe)


def _storecode_col(df):
    for c in df.columns:
        if "StoreCode" in str(c):
            return c
    return None


def etl():
    """Build the 14-trait environment-averaged landrace phenotype table aligned to 827 WATDE."""
    geno = list(pd.read_csv(GENO_SAMPLES, sep="\t")["sample"])
    xl = pd.ExcelFile(LAND)
    # per trait: collect {WATDE: [values across matching cols/sheets]}
    acc = {t: {} for t in TRAITS}
    for sh in SHEETS:
        df = xl.parse(sh)
        sc = _storecode_col(df)
        if sc is None:
            continue
        codes = df[sc].astype(str).str.strip()
        for t, pat in TRAITS.items():
            cols = [c for c in df.columns if re.search(pat, str(c)) and "Rep" not in str(c)
                    and "date_ymd" not in str(c) and not str(c).startswith(("Min.", "Max."))]
            for c in cols:
                vals = pd.to_numeric(df[c], errors="coerce")
                for code, v in zip(codes, vals, strict=True):
                    if code in geno and np.isfinite(v):
                        acc[t].setdefault(code, []).append(float(v))
    rows = {"sample": geno}
    for t in TRAITS:
        rows[t] = [np.mean(acc[t][s]) if s in acc[t] else np.nan for s in geno]
    ph = pd.DataFrame(rows)
    PHENO_OUT.parent.mkdir(parents=True, exist_ok=True)
    ph.to_csv(PHENO_OUT, sep="\t", index=False)
    print(f"  ETL -> {PHENO_OUT}  (n_samples={len(geno)})")
    for t in TRAITS:
        print(f"    {t:20s} n_nonNA={int(ph[t].notna().sum()):4d}")
    return ph


def _bh(p):
    p = np.asarray(p, float)
    n = p.size
    o = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for k in range(n - 1, -1, -1):
        i = o[k]
        prev = min(prev, p[i] * n / (k + 1))
        q[i] = prev
    return q


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("=== wheat landrace MULTI-TRAIT genome-wide scan ===")
    ph = etl()

    # ---- ONE genotype extraction + per-gene burdens + GRMs (trait-independent) ----
    rng = np.random.default_rng(7)
    burdens, grms, nsnp, samples_ref = {}, {}, {}, None
    for s in ("A", "B", "D"):
        genes = _w4._parse_gff(s, 2000)
        g2b, n_snp, K, samp = _w4._extract_and_burden(s, genes, "flank", 2000, 150, 3, rng)
        burdens[s] = g2b
        grms[s] = K
        nsnp[s] = n_snp
        if samples_ref is None:
            samples_ref = samp
        assert samp == samples_ref
        print(f"  {s}: {len(g2b)} retained genes")
    common = samples_ref

    hc = pd.read_csv(HCTRIADS)
    hc = hc[hc["cardinality_abs"] == "1:1:1"]
    kept = []
    for _, r in hc.iterrows():
        a, b, d = (_w4._v10_to_v11(str(r["A"])), _w4._v10_to_v11(str(r["B"])), _w4._v10_to_v11(str(r["D"])))
        if a in burdens["A"] and b in burdens["B"] and d in burdens["D"]:
            kept.append((a, b, d))
    G = len(kept)
    print(f"  curated 1:1:1 triads with full coverage: {G}")
    Bfull = {s: _w2._scols_safe(np.column_stack([burdens[s][k[i]] for k in kept]))
             for i, s in enumerate(("A", "B", "D"))}

    phi = ph.set_index("sample")
    pairwise_defs = [("A", "B"), ("A", "D"), ("B", "D")]
    grid_rows = []
    per_trait = []
    for t in TRAITS:
        valid = [s for s in common if s in phi.index and pd.notna(phi.loc[s, t])]
        if len(valid) < MIN_N:
            print(f"  [{t}] SKIP n={len(valid)}<{MIN_N}")
            continue
        idx = np.array([common.index(s) for s in valid])
        n_t = len(valid)
        KA = grms["A"][np.ix_(idx, idx)]
        KB = grms["B"][np.ix_(idx, idx)]
        KD = grms["D"][np.ix_(idx, idx)]
        KA, KB, KD = (K / (np.trace(K) / n_t) for K in (KA, KB, KD))
        Bd = {s: _w2._scols_safe(Bfull[s][idx]) for s in ("A", "B", "D")}
        y = _w2._rank_int(np.array([float(phi.loc[s, t]) for s in valid]))
        Wh, cv = _w2._whiten3(KA, KB, KD, y, seed=42)
        pw = {f"{x}{z}": _w2._pairwise_pvals(Wh, y, Bd[x], Bd[z]) for x, z in pairwise_defs}
        tri_acat = np.array([_w2._acat([pw[f"{x}{z}"][i] for x, z in pairwise_defs]) for i in range(G)])
        lam = {tag: _w2._lambda_gc(pw[tag]) for tag in pw}
        h2 = (cv.get("A", 0) + cv.get("B", 0) + cv.get("D", 0)) / max(
            cv.get("A", 0) + cv.get("B", 0) + cv.get("D", 0) + cv.get("e", 1e-9), 1e-9)
        bonf = 0.05 / G
        order = np.argsort(tri_acat)
        besti = int(order[0])
        nsig = int((tri_acat < bonf).sum())
        # candidate flag for follow-up permutation
        cand = bool(tri_acat.min() < 1e-4 or min(p.min() for p in pw.values()) < 1e-5)
        per_trait.append(dict(trait=t, n=n_t, G=G, min_triad_acat=float(tri_acat.min()),
                              bonferroni_alpha=bonf, n_sig_triad=nsig,
                              top_triad="|".join(kept[besti]),
                              top_triad_acat=float(tri_acat[besti]),
                              lambda_AB=float(lam["AB"]), lambda_AD=float(lam["AD"]),
                              lambda_BD=float(lam["BD"]), h2_proxy=float(h2),
                              candidate_for_perm=cand))
        for i in range(G):
            grid_rows.append((t, kept[i][0], kept[i][1], kept[i][2],
                              float(pw["AB"][i]), float(pw["AD"][i]), float(pw["BD"][i]),
                              float(tri_acat[i])))
        print(f"  [{t}] n={n_t} minACAT={tri_acat.min():.2g} nsig={nsig} "
              f"top={kept[besti]} λ(AB/AD/BD)={lam['AB']:.2f}/{lam['AD']:.2f}/{lam['BD']:.2f} "
              f"h2={h2:.2f} {'CAND' if cand else ''}")

    # cross-grid BH-FDR over all (trait x triad) ACAT p
    grid = pd.DataFrame(grid_rows, columns=["trait", "gene_A", "gene_B", "gene_D",
                                            "p_AB", "p_AD", "p_BD", "p_acat"])
    grid["FDR_acat_global"] = _bh(grid["p_acat"].values)
    grid = grid.sort_values("p_acat")
    grid.head(200).to_csv(OUT / "wheat_multitrait_top200.tsv", sep="\t", index=False)
    pt = pd.DataFrame(per_trait).sort_values("min_triad_acat")
    pt.to_csv(OUT / "wheat_multitrait_per_trait.tsv", sep="\t", index=False)

    import json
    n_gw = int((grid["p_acat"] < 0.05 / G).sum())
    n_fdr = int((grid["FDR_acat_global"] < 0.10).sum())
    summary = dict(task="wheat_multitrait_scan", prereg="reports/multitrait_prereg.md",
                   n_traits_scanned=len(pt), G_triads=G, MIN_N=MIN_N,
                   lambda_BD_median=float(pt["lambda_BD"].median()),
                   lambda_BD_range=[float(pt["lambda_BD"].min()), float(pt["lambda_BD"].max())],
                   n_triadACAT_genomewide_bonferroni=n_gw,
                   n_triadACAT_BH_FDR10=n_fdr,
                   top_triad_per_trait=pt[["trait", "n", "min_triad_acat", "n_sig_triad",
                                           "top_triad", "lambda_BD", "h2_proxy",
                                           "candidate_for_perm"]].to_dict("records"),
                   note=("CALIBRATION+BREADTH primary (lambda_GC across traits); discovery=asymptotic; "
                         "candidates (triad ACAT<1e-4 or pairwise<1e-5) flagged for follow-up perm; "
                         "emergence chr1 B-D anchor scored separately. BH-FDR over ALL trait x triad "
                         "ACAT p (not per-trait minima)."))
    (OUT / "wheat_multitrait_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"\n  λ_BD median={summary['lambda_BD_median']:.3f} range={[round(x,2) for x in summary['lambda_BD_range']]}")
    print(f"  triad-ACAT genome-wide Bonferroni hits={n_gw} | BH-FDR<0.10={n_fdr} | candidates={int(pt['candidate_for_perm'].sum())}")
    print(f"  wrote {OUT}/wheat_multitrait_{{per_trait.tsv,top200.tsv,summary.json}}")


if __name__ == "__main__":
    main()
