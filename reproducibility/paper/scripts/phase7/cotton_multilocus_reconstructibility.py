"""Phenotype-blind feasibility: how many of the CottonGVD strong-candidate pairs can be rebuilt
in the hebau/NDM8 panel at full density?

This reuses the criteria of cotton_replication_map.py verbatim (reciprocal best hit,
subgenome-consistent, a same-subgenome paralog no closer than MARGIN bits, >=3 callable SNPs
within +/-2kb) and reads the SAME full-density genotypes. It reads NO validation phenotype and
writes to a separate file, so the pre-registered single-lead artefact is untouched.

Candidate set is defined by the discovery panel only: pairs with interaction p < 1e-3 in the
CottonGVD FibLen scan, plus the published FibElo lead. That is a discovery-side rule; no hebau
outcome is involved.
"""
import gzip
import importlib.util
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/mnt/7302share/fast_ysp/U7_GWAS")
SCRATCH = Path(__file__).resolve().parent
SD = ROOT / "reproducibility/paper/scripts/phase7"
spec = importlib.util.spec_from_file_location("crm", str(SD / "cotton_replication_map.py"))
crm = importlib.util.module_from_spec(spec)
sys.modules["crm"] = crm
spec.loader.exec_module(crm)

PERPAIR = ROOT / "results/phase7/aggregate_perpair_cotton_cottongvd_cap150.tsv"
THRESH = 1e-3
FIBELO_LEAD = ("Gh_A01G025100", "Gh_D01G022200")

d = pd.read_csv(PERPAIR, sep="\t").sort_values("p")
cand = [(r.gene_A, r.gene_D, float(r.p)) for r in d[d.p < THRESH].itertuples()]
pairs = {f"FibLen_{i+1}": (a, b) for i, (a, b, _) in enumerate(cand)}
pairs["FibElo_lead"] = FIBELO_LEAD
pvals = {f"FibLen_{i+1}": p for i, (_a, _b, p) in enumerate(cand)}
print(f"candidate pairs from discovery only (FibLen p<{THRESH:.0e}): {len(cand)}  +1 FibElo lead")

cri, ndm8 = crm._read_fasta(crm.CRI_PEP), crm._read_fasta(crm.NDM8_PEP)
print(f"CRI proteome {len(cri)} | NDM8 proteome {len(ndm8)}")
cri_by_gene = {}
for k, v in cri.items():
    g = k.split(".")[0]
    if g not in cri_by_gene or len(v) > len(cri_by_gene[g]):
        cri_by_gene[g] = v

genes = sorted({g for pr in pairs.values() for g in pr})
missing = [g for g in genes if g not in cri_by_gene]
if missing:
    print(f"WARNING absent from CRI proteome: {missing}")
    genes = [g for g in genes if g in cri_by_gene]

tmp = Path(tempfile.mkdtemp())
try:
    q = tmp / "cand.faa"
    q.write_text("".join(f">{g}\n{cri_by_gene[g]}\n" for g in genes))
    fwd = crm._blast(q, crm.NDM8_PEP, tmp, "fwd")
    back_ids = sorted({h["sseqid"] for hs in fwd.values() for h in hs[:3]})
    cri_fa = tmp / "cri.faa"
    cri_fa.write_text("".join(f">{g}\n{s}\n" for g, s in cri_by_gene.items()))
    bq = tmp / "back.faa"
    bq.write_text("".join(f">{i}\n{ndm8[i]}\n" for i in back_ids if i in ndm8))
    rev = crm._blast(bq, cri_fa, tmp, "rev")
finally:
    shutil.rmtree(tmp, ignore_errors=True)

bim = {}
for sub in ("A", "D"):
    p = ROOT / f"results/viz_preview/fig3_cotton_locus/rescue/{sub}_wgs.bim"
    b = np.loadtxt(p, dtype=str, usecols=(0, 3))
    bim[sub] = (b[:, 0], b[:, 1].astype(np.int64))
    print(f"  {sub}_wgs.bim (full density): {len(b)} variants")

span = {}
with gzip.open(Path("/mnt/nvme/cotton_hbau/cottongen/NDM8.gff3.gz"), "rt") as fh:
    for line in fh:
        if line.startswith("#"):
            continue
        f = line.split("\t")
        if len(f) < 9 or f[2] != "gene":
            continue
        m = re.search(r"(GhM_[AD]\d\dG\d+)", f[8])
        if m:
            span[m.group(1)] = (f[0], int(f[3]), int(f[4]))

res = {"candidate_rule": f"CottonGVD FibLen interaction p < {THRESH:.0e} (discovery only) plus the "
                        f"published FibElo lead; no hebau phenotype was read",
       "full_density_source": "results/viz_preview/fig3_cotton_locus/rescue/{A,D}_wgs.bim",
       "criteria": {"MARGIN_bits": crm.MARGIN, "MIN_SNP": crm.MIN_SNP, "window": "+/-2kb"},
       "pairs": {}}
for tag, (gA, gD) in pairs.items():
    entry = {"cri_pair": [gA, gD], "discovery_p": pvals.get(tag), "mapped": {},
             "testable": True, "reasons": []}
    for g in (gA, gD):
        hs = fwd.get(g, [])
        if not hs:
            entry["testable"] = False; entry["reasons"].append(f"{g}: no NDM8 hit"); continue
        best = hs[0]
        same_sub = [h for h in hs[1:] if crm._sub(h["sseqid"]) == crm._sub(best["sseqid"])]
        para = same_sub[0] if same_sub else None
        margin = best["bitscore"] - (para["bitscore"] if para else 0.0)
        ok_rbh = rev.get(best["sseqid"], [{}])[0].get("sseqid") == g
        ok_sub = crm._sub(best["sseqid"]) == crm._sub(g)
        ok_margin = margin >= crm.MARGIN
        n_snp = None
        ndm8_gene = best["sseqid"].split(".")[0]
        if ndm8_gene in span:
            c, s, e = span[ndm8_gene]
            sub = crm._sub(ndm8_gene)
            if sub in bim:
                ch, po = bim[sub]
                n_snp = int(((ch == c) & (po >= s - 2000) & (po <= e + 2000)).sum())
        entry["mapped"][g] = dict(ndm8=best["sseqid"], pident=best["pident"], reciprocal=ok_rbh,
                                  subgenome_consistent=ok_sub, margin=margin, n_snp_pm2kb=n_snp)
        for cond, why in ((ok_rbh, "not a reciprocal best hit"),
                          (ok_sub, "subgenome-inconsistent"),
                          (ok_margin, f"same-subgenome paralog within {crm.MARGIN} bits"),
                          (n_snp is not None and n_snp >= crm.MIN_SNP, f"<{crm.MIN_SNP} callable SNPs")):
            if not cond:
                entry["testable"] = False; entry["reasons"].append(f"{g}: {why}")
    res["pairs"][tag] = entry

n_ok = sum(1 for e in res["pairs"].values() if e["testable"])
res["summary"] = dict(n_candidates=len(pairs), n_testable=n_ok,
                      n_non_testable=len(pairs) - n_ok)
print(f"\n{'tag':<14}{'disc p':>10}  {'A gene':<16}{'D gene':<16}{'nSNP_A':>7}{'nSNP_D':>7}  verdict")
for tag, e in res["pairs"].items():
    a, dd = e["cri_pair"]
    na = e["mapped"].get(a, {}).get("n_snp_pm2kb")
    nd = e["mapped"].get(dd, {}).get("n_snp_pm2kb")
    p = e["discovery_p"]
    print(f"{tag:<14}{(f'{p:.1e}' if p else '-'):>10}  {a:<16}{dd:<16}{str(na):>7}{str(nd):>7}  "
          f"{'TESTABLE' if e['testable'] else 'NO: ' + '; '.join(e['reasons'])[:70]}")
print(f"\n=> {n_ok} of {len(pairs)} candidate pairs are reconstructible in hebau at full density")
(SCRATCH / "cotton_multilocus_reconstructibility.json").write_text(json.dumps(res, indent=2))
print("wrote cotton_multilocus_reconstructibility.json (no validation phenotype was read)")
