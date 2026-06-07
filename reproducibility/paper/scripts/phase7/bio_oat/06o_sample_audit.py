#!/usr/bin/env python3
"""OAT QC: defend against silent sample/order misalignment (Codex TOP_RISK for the
unattended run). Re-derives the sample alignment across every layer and asserts they are
identical, hashing each so any drift is caught: merged.fam order, per-subgenome npz sample
order (A/C/D x body/flank), phenotype rows, per-trait nonmissing masks, and the
pheno-aligned trait vectors. Also asserts no duplicate/missing IDs and logs input-file
hashes. Writes oat_sample_audit.json (PASS/FAIL)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/mnt/7302share/fast_ysp/U7_GWAS")
OUT = ROOT / "results/phase7/bio_oat"
FAM = ROOT / "data/processed/avena_sativa_logical/merged.fam"
PHENO = ROOT / "data/processed/oat/pheno_clean.tsv"
SUBS = ("A", "C", "D")


def _sha(obj):
    return hashlib.sha256(repr(obj).encode()).hexdigest()[:16]


def _file_sha(path, nbytes=None):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        data = f.read(nbytes) if nbytes else f.read()
        h.update(data)
    return h.hexdigest()[:16]


def main():
    audit = dict(checks=[], asserts=[], hashes={})
    fails = []

    def check(name, cond, detail=""):
        audit["asserts"].append(dict(name=name, ok=bool(cond), detail=detail))
        if not cond:
            fails.append(f"{name}: {detail}")

    # 1. fam sample order
    fam_ids = [l.split()[1] for l in open(FAM)]
    check("fam_no_duplicates", len(fam_ids) == len(set(fam_ids)),
          f"{len(fam_ids)} ids, {len(set(fam_ids))} unique")
    audit["hashes"]["fam_order"] = _sha(fam_ids)
    audit["n_fam"] = len(fam_ids)

    # 2. per-subgenome npz sample order (must equal fam order, identical across all 6 npz)
    npz_orders = {}
    for s in SUBS:
        for mode in ("body", "flank"):
            z = np.load(OUT / f"snp_to_gene_{s}_{mode}.npz", allow_pickle=True)
            samp = [str(x) for x in z["samples"].tolist()]
            npz_orders[f"{s}_{mode}"] = _sha(samp)
            check(f"npz_{s}_{mode}_matches_fam", samp == fam_ids,
                  "order identical to fam" if samp == fam_ids else "ORDER DIFFERS from fam")
    audit["hashes"]["npz_sample_orders"] = npz_orders
    check("all_npz_same_order", len(set(npz_orders.values())) == 1,
          f"{len(set(npz_orders.values()))} distinct orders across 6 npz")

    # 3. phenotype: dup/missing, intersection with geno
    ph = pd.read_csv(PHENO, sep="\t")
    idcol = ph.columns[0]
    ph_ids = ph[idcol].astype(str).tolist()
    check("pheno_no_duplicates", len(ph_ids) == len(set(ph_ids)),
          f"{len(ph_ids)} ids, {len(set(ph_ids))} unique")
    inter = set(fam_ids) & set(ph_ids)
    check("full_geno_pheno_intersection", len(inter) == len(fam_ids),
          f"intersection {len(inter)} vs n_fam {len(fam_ids)}")
    missing_in_pheno = [g for g in fam_ids if g not in set(ph_ids)]
    check("no_geno_sample_missing_pheno", len(missing_in_pheno) == 0,
          f"{len(missing_in_pheno)} geno samples absent from pheno")

    # 4. per-trait nonmissing mask + pheno-aligned trait-vector hashes (reproduce deploy alignment)
    ph_idx = ph.set_index(idcol)
    traits = list(ph_idx.columns)
    trait_audit = {}
    for t in traits:
        valid = [s for s in fam_ids if s in ph_idx.index and pd.notna(ph_idx.loc[s, t])]
        yvec = [float(ph_idx.loc[s, t]) for s in valid]
        trait_audit[t] = dict(n=len(valid), mask_hash=_sha(valid),
                              yvec_hash=_sha([round(v, 6) for v in yvec]))
    audit["per_trait"] = trait_audit
    audit["n_traits"] = len(traits)

    # 5. input-file content hashes (so a swapped input is detectable)
    audit["hashes"]["input_files"] = {
        "merged.fam": _file_sha(FAM),
        "pheno_clean.tsv": _file_sha(PHENO),
        "homoeolog_pairs.tsv": _file_sha(OUT / "homoeolog_pairs.tsv"),
        "triad_attrition.json": _file_sha(OUT / "triad_attrition.json"),
    }

    audit["VERDICT"] = "PASS" if not fails else "FAIL"
    audit["failures"] = fails
    (OUT / "oat_sample_audit.json").write_text(json.dumps(audit, indent=2, default=float))

    print(f"OAT SAMPLE AUDIT: {audit['VERDICT']}")
    print(f"  n_fam={len(fam_ids)} n_traits={len(traits)} "
          f"geno-pheno intersection={len(inter)}/{len(fam_ids)}")
    print(f"  all 6 npz same sample order as fam: "
          f"{all(a['ok'] for a in audit['asserts'] if 'npz_' in a['name'])}")
    n_min = min((v['n'] for v in trait_audit.values()))
    n_max = max((v['n'] for v in trait_audit.values()))
    print(f"  per-trait N range: {n_min}-{n_max}")
    if fails:
        print("  FAILURES:")
        for f in fails:
            print(f"    - {f}")
    print(f"  wrote {OUT/'oat_sample_audit.json'}")


if __name__ == "__main__":
    main()
