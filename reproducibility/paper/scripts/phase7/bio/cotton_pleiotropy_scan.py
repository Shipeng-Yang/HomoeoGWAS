#!/usr/bin/env python3
"""Cotton PLEIOTROPY (ACAT-across-traits) scan — the pre-registered multi-trait mode (multitrait_prereg
§ "two analysis modes"). Per A-D pair, combine the per-trait whitened interaction p-values across a
FROZEN trait set with ACAT into ONE pleiotropy p (multiplicity = G pairs, not GxT). Higher power than
single-trait scans when one homoeolog interaction affects several correlated fibre traits. Frozen sets:
fibre-quality-5 (pre-declared cluster) and all-13. Both pair modes (body, flank). This settles whether
the single-trait '0 new hits' missed a pair that is consistently sub-threshold across the cluster.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from homoeogwas.interact import FrozenTraitSet, SubgenomeData, run_multitrait_pair_scan

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
FIBRE5 = ["fiber_length_BLUE", "fiber_strength_BLUE", "micronaire_BLUE", "elongation_BLUE",
          "length_uniformity_BLUE"]
ALL13 = FIBRE5 + ["maturity_BLUE", "spinning_consistency_BLUE", "boll_weight_BLUE",
                  "lint_percentage_BLUE", "seed_index_BLUE", "lint_index_BLUE",
                  "fiber_weight_per_boll_BLUE", "fiber_density_BLUE"]
SETS = {"fibre_quality_5": FIBRE5, "all_13": ALL13}


def _load_bed(prefix):
    from homoeogwas.io import load_bed_hardcall
    bed = load_bed_hardcall(str(prefix))
    return np.asarray(bed.dosage, dtype=np.float64), [str(s) for s in np.asarray(bed.samples)], bed


def _gene_snp(npz_path):
    z = np.load(npz_path, allow_pickle=True)
    return {g: np.asarray(z["snp_idx"][i], int) for i, g in enumerate(z["gene_ids"].tolist())}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("=== cotton PLEIOTROPY (ACAT-across-traits) scan ===")
    X, samples, bedA = _load_bed(GENO["A"])
    XD, samplesD, bedD = _load_bed(GENO["D"])
    assert samples == samplesD
    bed = {"A": (X, bedA), "D": (XD, bedD)}
    ph = pd.read_csv(PHENO, sep="\t").set_index("sample")

    out = {}
    for mode in ("body", "flank"):
        gsnp = {s: _gene_snp(NPZ[mode]) for s in ("A", "D")}
        subdata = {s: SubgenomeData(X=bed[s][0], gene_snp=gsnp[s], samples=samples, chunk=bed[s][1])
                   for s in ("A", "D")}
        df = pd.read_csv(PAIRS[mode], sep="\t")
        pairs = [(r.gene_A, r.gene_D) for r in df[["gene_A", "gene_D"]].itertuples(index=False)]
        for setname, traits in SETS.items():
            ts = FrozenTraitSet.from_list(traits)
            valid = [s for s in samples if s in ph.index and bool(ph.loc[s, traits].notna().all())]
            sidx = np.array([samples.index(s) for s in valid])
            y_by = {t: np.array([float(ph.loc[s, t]) for s in valid]) for t in traits}
            r = run_multitrait_pair_scan(subdata, pairs, y_by, sidx, trait_set=ts, cap=150,
                                         transform="INT", perm_B=PERM_B, n_jobs=N_JOBS,
                                         pair_subs=("A", "D"), grm_method="compute_grm_maf")
            top = r["top"][0] if r["top"] else {}
            key = f"{mode}:{setname}"
            out[key] = dict(mode=mode, trait_set=setname, n_traits=len(traits), n=r["n"], G=r["G"],
                            pleio_acat_omnibus=r["pleio_acat_omnibus"],
                            pleio_acat_omnibus_emp=r["pleio_acat_omnibus_emp"],
                            min_p=r["min_p"], min_p_emp=r["min_p_emp"],
                            lambda_gc_obs=r["lambda_gc_obs"], n_sig=r["n_sig"],
                            top_pair=(f"{top.get('pair',['',''])[0]}|{top.get('pair',['',''])[1]}"
                                      if top else ""),
                            top_pleio_p=top.get("pleio_p"),
                            top_min_trait=top.get("audit_min_trait_name"),
                            sig=r["sig"])
            print(f"  [{mode}] {setname} ({len(traits)}tr): n={r['n']} G={r['G']} "
                  f"pleio_ACAT={r['pleio_acat_omnibus']:.3g} emp={r['pleio_acat_omnibus_emp']} "
                  f"minP={r['min_p']:.2g} minP_emp={r['min_p_emp']} nsig={r['n_sig']} "
                  f"top={top.get('pair')} pleio_p={top.get('pleio_p',float('nan')):.2g} "
                  f"(driven by {top.get('audit_min_trait_name')})")
            for h in r["sig"]:
                print(f"        PLEIO-HIT {h['pair']} pleio_p={h['pleio_p']:.3g}")

    (OUT / "cotton_pleiotropy_scan.json").write_text(json.dumps(out, indent=2, default=float))
    n_hits = sum(v["n_sig"] for v in out.values())
    print(f"\n  total pleiotropy genome-wide hits across all sets/modes: {n_hits}")
    print(f"  wrote {OUT}/cotton_pleiotropy_scan.json")


if __name__ == "__main__":
    main()
