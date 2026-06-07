#!/usr/bin/env python3
"""Multi-trait expansion (cotton): per-trait homoeolog-pair interaction scan across all 13 BLUE
traits x {body, flank}, pre-registered in reports/multitrait_prereg.md.

Primary claim = CALIBRATION + BREADTH (does the method stay calibrated across 13 traits?), with new
hits as a pre-registered discovery catalogue. Anchors (fibre_length=Hit1, length_uniformity=Hit2) are
scored but flagged 'anchor', separate from new discoveries. Per (trait x mode): ACAT omnibus,
empirical y-shuffle calibration, lambda_GC (obs + perm), Bonferroni genome-wide bar, top hit; then
BH-FDR across the grid (new discoveries only); h2 proxy = sigma_sub/(sigma_sub+e) from the null LMM.
Genotype is loaded ONCE and reused across all trait x mode scans.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from homoeogwas.interact import SubgenomeData, run_pair_scan

ROOT = Path("/mnt/7302share/fast_ysp/U7_GWAS")
GENO = {"A": ROOT / "data/processed/cotton/A/all", "D": ROOT / "data/processed/cotton/D/all"}
NPZ = {"body": ROOT / "results/phase7/bio_full/snp_to_gene_body.npz",
       "flank": ROOT / "results/phase7/bio_full/snp_to_gene_flank2000bp.npz"}
PAIRS = {"body": ROOT / "results/phase7/bio_full/pairs_body_same.tsv",
         "flank": ROOT / "results/phase7/bio_full/pairs_flank_same.tsv"}
PHENO = ROOT / "data/processed/cotton/pheno_m3_4_blue.tsv"
OUT = ROOT / "results/phase7/bio_full/multitrait"
PERM_B = 2000
N_JOBS = 96
ANCHORS = {"fibre_length": "fiber_length_BLUE", "length_uniformity": "length_uniformity_BLUE"}
TRAITS = ["fiber_length_BLUE", "fiber_strength_BLUE", "micronaire_BLUE", "elongation_BLUE",
          "length_uniformity_BLUE", "maturity_BLUE", "spinning_consistency_BLUE", "boll_weight_BLUE",
          "lint_percentage_BLUE", "seed_index_BLUE", "lint_index_BLUE", "fiber_weight_per_boll_BLUE",
          "fiber_density_BLUE"]


def _load_bed(prefix):
    from homoeogwas.io import load_bed_hardcall
    bed = load_bed_hardcall(str(prefix))
    return np.asarray(bed.dosage, dtype=np.float64), [str(s) for s in np.asarray(bed.samples)], bed


def _gene_snp(npz_path):
    z = np.load(npz_path, allow_pickle=True)
    gids = z["gene_ids"].tolist()
    si = z["snp_idx"]
    return {g: np.asarray(si[i], int) for i, g in enumerate(gids)}


def _load_pairs(path):
    df = pd.read_csv(path, sep="\t")
    return [(r.gene_A, r.gene_D) for r in df[["gene_A", "gene_D"]].itertuples(index=False)]


def _bh(pvals):
    p = np.asarray(pvals, float)
    n = p.size
    order = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for k in range(n - 1, -1, -1):
        i = order[k]
        prev = min(prev, p[i] * n / (k + 1))
        q[i] = prev
    return q


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("=== cotton multi-trait expansion (13 traits x body/flank) ===")
    # genotype + GRM inputs loaded ONCE per subgenome
    X, samples, bedA = _load_bed(GENO["A"])
    XD, samplesD, bedD = _load_bed(GENO["D"])
    assert samples == samplesD, "A/D sample order mismatch"
    bed = {"A": (X, bedA), "D": (XD, bedD)}
    ph = pd.read_csv(PHENO, sep="\t").set_index("sample")

    rows = []
    for mode in ("body", "flank"):
        gsnp = {s: _gene_snp(NPZ[mode]) for s in ("A", "D")}
        subdata = {s: SubgenomeData(X=bed[s][0], gene_snp=gsnp[s], samples=samples, chunk=bed[s][1])
                   for s in ("A", "D")}
        pairs = _load_pairs(PAIRS[mode])
        for trait in TRAITS:
            valid = [s for s in samples if s in ph.index and pd.notna(ph.loc[s, trait])]
            sidx = np.array([samples.index(s) for s in valid])
            y = np.array([float(ph.loc[s, trait]) for s in valid])
            r = run_pair_scan(subdata, pairs, y, sidx, cap=150, transform="INT", perm_B=PERM_B,
                              n_jobs=N_JOBS, pair_subs=("A", "D"), grm_method="compute_grm_maf")
            top = r.top[0] if r.top else dict(pair=("", ""), p=float("nan"))
            sig = r.sigma_hat
            h2 = (sig.get("A", 0) + sig.get("D", 0)) / max(
                sig.get("A", 0) + sig.get("D", 0) + sig.get("e", 1e-9), 1e-9)
            rows.append(dict(
                species="cotton", trait=trait, mode=mode, n=r.n, G=r.G,
                pair_acat=r.pair_acat, pair_acat_emp=r.pair_acat_emp, min_p=r.min_p,
                lambda_gc_obs=r.lambda_gc_obs, lambda_gc_perm_median=r.lambda_gc_perm_median,
                bonferroni_alpha=r.bonferroni_alpha, n_sig=r.n_sig,
                top_pair=f"{top['pair'][0]}|{top['pair'][1]}", top_p=top["p"],
                h2_proxy=float(h2),
                known_or_new=("anchor" if trait in ANCHORS.values() else "new")))
            print(f"  [{mode}] {trait:28s} n={r.n} G={r.G} ACAT={r.pair_acat:.3g} "
                  f"emp={r.pair_acat_emp} minP={r.min_p:.2g} λ={r.lambda_gc_obs:.2f} "
                  f"nsig={r.n_sig} top={top['pair']} p={top['p']:.2g} [{'anchor' if trait in ANCHORS.values() else 'new'}]")

    df = pd.DataFrame(rows)
    # BH-FDR across NEW scans on the per-scan ACAT OMNIBUS empirical p (the calibrated scan-level
    # statistic). NB: do NOT BH the raw min_p -- min_p is the extreme over G pairs within a scan, not a
    # calibrated per-scan p; per-pair genome-wide significance is min_p < bonferroni_alpha (=> n_sig).
    new = df[df["known_or_new"] == "new"].copy()
    df["FDR_acat_omnibus_new"] = np.nan
    if len(new):
        q = _bh(new["pair_acat_emp"].fillna(1.0).values)
        df.loc[new.index, "FDR_acat_omnibus_new"] = q
    # effective number of independent traits (trait-correlation eigenvalues, Galwey)
    tmat = ph[TRAITS].dropna()
    C = np.corrcoef(tmat.values.T)
    ev = np.linalg.eigvalsh(C)
    ev = ev[ev > 0]
    meff = float((ev.sum() ** 2) / (ev ** 2).sum())   # Galwey effective number of tests

    df = df.sort_values("min_p")
    df.to_csv(OUT / "cotton_multitrait_scan.tsv", sep="\t", index=False)
    summary = dict(
        task="cotton_multitrait_scan", prereg="reports/multitrait_prereg.md", n_traits=len(TRAITS),
        modes=["body", "flank"], perm_B=PERM_B, effective_n_traits_Galwey=meff,
        lambda_gc_median=float(df["lambda_gc_obs"].median()),
        lambda_gc_range=[float(df["lambda_gc_obs"].min()), float(df["lambda_gc_obs"].max())],
        n_scans=len(df),
        n_new_pairs_genomewide_bonferroni=int((df[df["known_or_new"] == "new"]["n_sig"] > 0).sum()),
        n_new_scans_acat_omnibus_FDR10=int((df["FDR_acat_omnibus_new"] < 0.10).sum()),
        new_hits=df[(df["known_or_new"] == "new") & ((df["n_sig"] > 0) | (df["FDR_acat_omnibus_new"] < 0.10))]
            [["trait", "mode", "top_pair", "top_p", "n_sig", "pair_acat_emp", "FDR_acat_omnibus_new"]]
            .to_dict("records"),
        anchors=df[df["known_or_new"] == "anchor"]
            [["trait", "mode", "top_pair", "top_p", "n_sig", "pair_acat_emp"]].to_dict("records"),
        note=("CALIBRATION+BREADTH primary: lambda_GC across all 13x2 scans. New hits = pre-registered "
              "catalogue with BH-FDR (anchors scored separately as positive controls). 13 fibre traits "
              "are correlated -> effective_n_traits_Galwey reported; a pair hitting multiple correlated "
              "traits = one fibre-quality locus, not independent discoveries."))
    (OUT / "cotton_multitrait_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"\n  λ_GC median={summary['lambda_gc_median']:.3f} range={summary['lambda_gc_range']} "
          f"| effective_n_traits={meff:.1f}/13")
    print(f"  NEW genome-wide-Bonferroni pairs={summary['n_new_pairs_genomewide_bonferroni']} | "
          f"NEW ACAT-omnibus FDR<0.10 scans={summary['n_new_scans_acat_omnibus_FDR10']} | "
          f"anchors reproduced={sum(a['n_sig']>0 for a in summary['anchors'])}/{len(summary['anchors'])}")
    print(f"  wrote {OUT}/cotton_multitrait_{{scan.tsv,summary.json}}")


if __name__ == "__main__":
    main()
