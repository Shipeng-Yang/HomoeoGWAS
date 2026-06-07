#!/usr/bin/env python3
"""Cotton expand-pairing SECONDARY EXPLORATORY scan (pre-registered:
reports/cotton_expand_pairing_prereg.md; uses pre-RBH BLAST-homolog pairs).

Tests whether the strict 1:1 RBH restriction (372 curated pairs) missed homoeolog-pair signal, by
relaxing the pairing to ALL A-D BLAST-homolog gene pairs (diamond union.blast, e<=1e-5, both callable
in flank mode) = 9,177 pairs (24.7x). NOT the interval all-genes universe (~360k, rejected as regional
epistasis). All 13 fibre BLUE traits (ran-all-reported-all); flank mode. Bonferroni 0.05/G primary bar
with BH-FDR as the exploratory screen; the 2 anchors scored SEPARATELY as positive controls. Reuses
the production run_pair_scan engine and the cotton_multitrait_scan loaders.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/mnt/7302share/fast_ysp/U7_GWAS")
BLAST = Path("/mnt/nvme/cotton_hbau/mcscanx_full/union.blast")
GENES = ROOT / "results/phase7/bio_full/genes.tsv"
OUT = ROOT / "results/phase7/bio_full/multitrait"
EVAL_CUT = 1e-5
PERM_B = 2000
N_JOBS = 96

# reuse loaders/engine wiring from the curated multi-trait scan
_spec = importlib.util.spec_from_file_location("cmt", str(ROOT / "scripts/phase7/bio/cotton_multitrait_scan.py"))
_cmt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cmt)
SubgenomeData = _cmt.SubgenomeData
run_pair_scan = _cmt.run_pair_scan
TRAITS = _cmt.TRAITS
ANCHORS = _cmt.ANCHORS


def build_homolog_pairs():
    """All A-D BLAST-homolog pairs (e<=1e-5), both callable in flank mode, unordered unique."""
    g = pd.read_csv(GENES, sep="\t")
    call = set(g.loc[g["retained_flank"] == 1, "gene_id"])
    subm = dict(zip(g["gene_id"], g["sub"], strict=True))
    b = pd.read_csv(BLAST, sep="\t", header=None,
                    names=["q", "s", "pid", "alen", "mm", "go", "qs", "qe", "ss", "se", "e", "bit"])
    b = b[b["q"] != b["s"]]
    qs, ss, ee = b["q"].to_numpy(), b["s"].to_numpy(), b["e"].to_numpy()
    pairs = set()
    for q, s, e in zip(qs, ss, ee, strict=True):
        if e > EVAL_CUT or q not in call or s not in call:
            continue
        a, d = (q, s) if subm.get(q) == "A" and subm.get(s) == "D" else (
            (s, q) if subm.get(s) == "A" and subm.get(q) == "D" else (None, None))
        if a is not None:
            pairs.add((a, d))
    return sorted(pairs)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("=== cotton EXPAND-PAIRING exploratory (BLAST-homolog A-D pairs, flank, 13 traits) ===")
    pairs = build_homolog_pairs()
    print(f"  expanded universe: {len(pairs)} A-D BLAST-homolog callable pairs (vs 372 curated)")

    X, samples, bedA = _cmt._load_bed(_cmt.GENO["A"])
    XD, samplesD, bedD = _cmt._load_bed(_cmt.GENO["D"])
    assert samples == samplesD
    bed = {"A": (X, bedA), "D": (XD, bedD)}
    ph = pd.read_csv(_cmt.PHENO, sep="\t").set_index("sample")
    gsnp = {s: _cmt._gene_snp(_cmt.NPZ["flank"]) for s in ("A", "D")}
    subdata = {s: SubgenomeData(X=bed[s][0], gene_snp=gsnp[s], samples=samples, chunk=bed[s][1])
               for s in ("A", "D")}

    rows, new_hits = [], []
    for trait in TRAITS:
        valid = [s for s in samples if s in ph.index and pd.notna(ph.loc[s, trait])]
        sidx = np.array([samples.index(s) for s in valid])
        y = np.array([float(ph.loc[s, trait]) for s in valid])
        r = run_pair_scan(subdata, pairs, y, sidx, cap=150, transform="INT", perm_B=PERM_B,
                          n_jobs=N_JOBS, pair_subs=("A", "D"), grm_method="compute_grm_maf")
        top = r.top[0] if r.top else dict(pair=("", ""), p=float("nan"))
        is_anchor = trait in ANCHORS.values()
        rows.append(dict(trait=trait, mode="flank", n=r.n, G=r.G, pair_acat=r.pair_acat,
                         pair_acat_emp=r.pair_acat_emp, min_p=r.min_p, lambda_gc_obs=r.lambda_gc_obs,
                         lambda_gc_perm_median=r.lambda_gc_perm_median,
                         bonferroni_alpha=r.bonferroni_alpha, n_sig=r.n_sig,
                         top_pair=f"{top['pair'][0]}|{top['pair'][1]}", top_p=top["p"],
                         known_or_new=("anchor" if is_anchor else "new")))
        # collect NEW per-pair hits passing Bonferroni (anchors excluded from discovery)
        if not is_anchor:
            for h in r.top:
                if h["p"] < r.bonferroni_alpha:
                    new_hits.append(dict(trait=trait, pair=f"{h['pair'][0]}|{h['pair'][1]}", p=h["p"]))
        print(f"  {trait:28s} n={r.n} G={r.G} minP={r.min_p:.2g} bonf={r.bonferroni_alpha:.2g} "
              f"nsig={r.n_sig} λ={r.lambda_gc_obs:.2f} top={top['pair']} p={top['p']:.2g} "
              f"[{'ANCHOR' if is_anchor else 'new'}]")

    df = pd.DataFrame(rows)
    new = df[df["known_or_new"] == "new"].copy()
    df["FDR_acat_omnibus_new"] = np.nan
    if len(new):
        df.loc[new.index, "FDR_acat_omnibus_new"] = _cmt._bh(new["pair_acat_emp"].fillna(1.0).values)
    df = df.sort_values("min_p")
    df.to_csv(OUT / "cotton_expand_pairing.tsv", sep="\t", index=False)

    G = len(pairs)
    summary = dict(task="cotton_expand_pairing", prereg="reports/cotton_expand_pairing_prereg.md",
                   tier="secondary_exploratory", mode="flank", universe="A-D BLAST-homolog (e<=1e-5)",
                   G_pairs=G, vs_curated_372=round(G / 372, 1), perm_B=PERM_B,
                   bonferroni_alpha=0.05 / G,
                   lambda_gc_median=float(df["lambda_gc_obs"].median()),
                   lambda_gc_range=[float(df["lambda_gc_obs"].min()), float(df["lambda_gc_obs"].max())],
                   n_new_traits_with_genomewide_pair=int((new["n_sig"] > 0).sum()) if len(new) else 0,
                   n_new_scans_acat_FDR10=int((df["FDR_acat_omnibus_new"] < 0.10).sum()),
                   new_pair_hits_bonferroni=new_hits,
                   anchors=df[df["known_or_new"] == "anchor"][
                       ["trait", "min_p", "n_sig", "top_pair", "top_p"]].to_dict("records"),
                   note=("SECONDARY EXPLORATORY relaxing strict 1:1 RBH -> all A-D BLAST-homolog pairs. "
                         "Anchors scored separately as positive controls (not in new-discovery "
                         "denominator). BH-FDR over new-scan ACAT-omnibus emp p is the exploratory "
                         "screen; per-pair genome-wide = min_p < 0.05/G. ran-all-reported-all."))
    (OUT / "cotton_expand_pairing.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"\n  λ_GC median={summary['lambda_gc_median']:.3f} | new per-pair Bonferroni hits="
          f"{len(new_hits)} | new scans FDR<0.10={summary['n_new_scans_acat_FDR10']}")
    print(f"  wrote {OUT}/cotton_expand_pairing.{{tsv,json}}")


if __name__ == "__main__":
    main()
