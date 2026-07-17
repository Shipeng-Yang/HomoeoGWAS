#!/usr/bin/env python3
"""Map the two CottonGVD lead pairs from Gh_CRI v1 onto the HBAU/NDM8 replication panel.

Step 1 of PREREGISTRATION_cotton_hebau_replication.md (git 3b675e8ca696, committed before this ran).
Deliberately reads NO interaction p-value: the replication scans already sit on disk, so the mapping
and the look-up are kept in separate scripts and separate commits.

Declares a pair NON-TESTABLE, rather than substituting anything, if the reciprocal best hit is
missing, is not subgenome-consistent, is not clearly separated from the next-best hit, or leaves
fewer than 3 callable SNPs. Substituting a paralog for an absent ortholog is what retired this
panel's own previous hits.
"""
from __future__ import annotations

import gzip
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path("/mnt/7302share/fast_ysp/U7_GWAS")
DIAMOND = "/mnt/7302share/miniconda3/envs/canu_shared/bin/diamond"   # 2.1.13; 2.2.x deadlocks here
CRI_PEP = ROOT / "data/reference/cotton/CRI_TM1_v1/TM_1.Chr_genome_all_transcripts_final_gene.change.gff.pep.gz"
NDM8_PEP = Path("/mnt/nvme/cotton_hbau/cottongen/NDM8.pep.fa")
OUT = ROOT / "results/phase7/cotton_replication"

LEADS = {
    "fibre_length": ("Gh_A05G118800", "Gh_D05G131200"),
    "fibre_elongation": ("Gh_A01G025100", "Gh_D01G022200"),
}
MARGIN = 5.0        # next-best hit must be this many bits behind, else a paralog is in contention
MIN_SNP = 3


def _read_fasta(path):
    op = gzip.open if str(path).endswith(".gz") else open
    seqs, name, buf = {}, None, []
    with op(path, "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                if name:
                    seqs[name] = "".join(buf)
                name, buf = line[1:].split()[0], []
            else:
                buf.append(line.strip())
    if name:
        seqs[name] = "".join(buf)
    # this CRI proteome marks the terminal stop with "." (79,692 of them, one per protein) instead of
    # the usual "*"; diamond rejects the character outright
    return {k: v.replace(".", "").replace("*", "") for k, v in seqs.items()}


def _sub(gene):
    m = re.search(r"[_]?([AD])\d\d", gene)
    return m.group(1) if m else "?"


def _blast(q_fa, db_fa, tmp, tag):
    db = tmp / f"{tag}.dmnd"
    subprocess.run([DIAMOND, "makedb", "--in", str(db_fa), "-d", str(db), "--quiet"], check=True)
    out = tmp / f"{tag}.tsv"
    subprocess.run([DIAMOND, "blastp", "-q", str(q_fa), "-d", str(db), "-o", str(out), "--quiet",
                    "--max-target-seqs", "20", "--outfmt", "6", "qseqid", "sseqid", "pident",
                    "length", "evalue", "bitscore", "qcovhsp"], check=True)
    hits = {}
    with open(out) as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            hits.setdefault(f[0], []).append(
                dict(sseqid=f[1], pident=float(f[2]), evalue=float(f[4]),
                     bitscore=float(f[5]), qcov=float(f[6])))
    return hits


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cri, ndm8 = _read_fasta(CRI_PEP), _read_fasta(NDM8_PEP)
    print(f"CRI proteome {len(cri)} | NDM8 proteome {len(ndm8)}")

    # CRI ids in the pep file may carry a transcript suffix; index by gene
    cri_by_gene = {}
    for k, v in cri.items():
        g = k.split(".")[0]
        if g not in cri_by_gene or len(v) > len(cri_by_gene[g]):
            cri_by_gene[g] = v

    genes = [g for pair in LEADS.values() for g in pair]
    missing = [g for g in genes if g not in cri_by_gene]
    if missing:
        raise SystemExit(f"lead gene(s) absent from the CRI proteome: {missing}")

    tmp = Path(tempfile.mkdtemp())
    try:
        q = tmp / "leads.faa"
        q.write_text("".join(f">{g}\n{cri_by_gene[g]}\n" for g in genes))
        fwd = _blast(q, NDM8_PEP, tmp, "fwd")

        # reciprocal leg: best NDM8 hits back against the whole CRI proteome
        back_ids = sorted({h["sseqid"] for hs in fwd.values() for h in hs[:3]})
        cri_fa = tmp / "cri.faa"
        cri_fa.write_text("".join(f">{g}\n{s}\n" for g, s in cri_by_gene.items()))
        bq = tmp / "back.faa"
        bq.write_text("".join(f">{i}\n{ndm8[i]}\n" for i in back_ids if i in ndm8))
        rev = _blast(bq, cri_fa, tmp, "rev")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # callable SNP counts on the replication panel's full-density genotypes
    sys.path.insert(0, str(ROOT / "src"))
    bim = {}
    for sub in ("A", "D"):
        p = ROOT / f"results/viz_preview/fig3_cotton_locus/rescue/{sub}_wgs.bim"
        b = np.loadtxt(p, dtype=str, usecols=(0, 3))
        bim[sub] = (b[:, 0], b[:, 1].astype(np.int64))
        print(f"  {sub}_wgs.bim: {len(b)} variants")

    ndm8_gff = Path("/mnt/nvme/cotton_hbau/cottongen/NDM8.gff3.gz")
    span = {}
    with gzip.open(ndm8_gff, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.split("\t")
            if len(f) < 9 or f[2] != "gene":
                continue
            m = re.search(r"(GhM_[AD]\d\dG\d+)", f[8])
            if m:
                span[m.group(1)] = (f[0], int(f[3]), int(f[4]))

    res = {"prereg": "reproducibility/paper/notes/PREREGISTRATION_cotton_hebau_replication.md",
           "prereg_blob": "3b675e8ca696", "pairs": {}}
    for trait, (gA, gD) in LEADS.items():
        entry = {"cri_pair": [gA, gD], "mapped": {}, "testable": True, "reasons": []}
        for g in (gA, gD):
            hs = fwd.get(g, [])
            if not hs:
                entry["testable"] = False
                entry["reasons"].append(f"{g}: no NDM8 hit")
                continue
            best = hs[0]
            nxt = hs[1] if len(hs) > 1 else None
            margin = best["bitscore"] - (nxt["bitscore"] if nxt else 0.0)
            rbh_back = rev.get(best["sseqid"], [{}])[0].get("sseqid")
            ok_rbh = rbh_back == g
            ok_sub = _sub(best["sseqid"]) == _sub(g)
            ok_margin = margin >= MARGIN
            n_snp = None
            if best["sseqid"] in span:
                c, s, e = span[best["sseqid"]]
                sub = _sub(best["sseqid"])
                if sub in bim:
                    ch, po = bim[sub]
                    n_snp = int(((ch == c) & (po >= s - 2000) & (po <= e + 2000)).sum())
            entry["mapped"][g] = dict(ndm8=best["sseqid"], pident=best["pident"], qcov=best["qcov"],
                                      bitscore=best["bitscore"],
                                      next_best=(nxt["sseqid"] if nxt else None),
                                      next_bitscore=(nxt["bitscore"] if nxt else None),
                                      margin=margin, reciprocal=ok_rbh,
                                      subgenome_consistent=ok_sub, n_snp_pm2kb=n_snp)
            for cond, why in ((ok_rbh, "not a reciprocal best hit"),
                              (ok_sub, "subgenome-inconsistent"),
                              (ok_margin, f"next-best within {MARGIN} bits (paralog in contention)"),
                              (n_snp is not None and n_snp >= MIN_SNP, f"<{MIN_SNP} callable SNPs")):
                if not cond:
                    entry["testable"] = False
                    entry["reasons"].append(f"{g}: {why}")
        res["pairs"][trait] = entry

    (OUT / "lead_mapping.json").write_text(json.dumps(res, indent=2))
    print(f"\n{'trait':<18}{'CRI gene':<16}{'NDM8':<18}{'pid':>6}{'qcov':>6}{'margin':>8}{'RBH':>5}{'sub':>5}{'SNP':>6}")
    for trait, e in res["pairs"].items():
        for g, m in e["mapped"].items():
            print(f"{trait:<18}{g:<16}{m['ndm8']:<18}{m['pident']:>6.1f}{m['qcov']:>6.1f}"
                  f"{m['margin']:>8.1f}{str(m['reciprocal']):>5}{str(m['subgenome_consistent']):>5}"
                  f"{str(m['n_snp_pm2kb']):>6}")
        print(f"  -> {trait}: {'TESTABLE' if e['testable'] else 'NON-TESTABLE: ' + '; '.join(e['reasons'])}\n")
    print(f"wrote {OUT/'lead_mapping.json'}  (no p-value was read by this script)")


if __name__ == "__main__":
    main()
