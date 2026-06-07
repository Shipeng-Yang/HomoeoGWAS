#!/usr/bin/env python3
"""P1 deliverable: functional annotation of the wheat candidate homoeolog genes (chr1 B-D interaction
+ its A copy; chr5 A-D second positive + its B copy). Dual-planned (Codex PLAN_READY). Annotation
strategy (Codex fixes): protein domains (Pfam/InterPro) from Ensembl Plants as the wheat-native
annotation; BLASTp -> local Arabidopsis TAIR10 proteome for comparative orthology (the same pipeline
used for the cotton dossier, for cross-species symmetry). ALL six copies are annotated (interacting
pair + third homoeolog), chr5 (the expression-balanced contrast) included to avoid selective
annotation, and 'unknown protein' is reported honestly. BLAST rules: blastp vs TAIR10, best hit by
bitscore, report %identity / e-value / query-coverage; flagged low-confidence if id<30% or cov<50%.

No fabrication: descriptions come verbatim from TAIR10 FASTA headers + Ensembl protein features.
"""
from __future__ import annotations

import json
import subprocess
import time
import urllib.request
from pathlib import Path

ROOT = Path("/mnt/7302share/fast_ysp/U7_GWAS")
TAIR = ROOT / "data/reference/arabidopsis/TAIR10.pep.fa.gz"
WORK = Path("/tmp/wheat_annot")
OUT = ROOT / "results/phase7/bio_wheat/heb_causal_ladder/wheat_candidate_annotation.json"
REST = "https://rest.ensembl.org"

GENES = [
    ("chr1", "B", "TraesCS1B02G143800", "interacting (B-D)"),
    ("chr1", "D", "TraesCS1D02G128400", "interacting (B-D)"),
    ("chr1", "A", "TraesCS1A02G125400", "third copy"),
    ("chr5", "A", "TraesCS5A02G169500", "interacting (A-D)"),
    ("chr5", "D", "TraesCS5D02G173900", "interacting (A-D)"),
    ("chr5", "B", "TraesCS5B02G166300", "third copy"),
]


def _get(url, tries=4):
    # response format is set by the ?content-type= query param; do NOT send a conflicting
    # Content-Type request header (it makes the sequence endpoint return JSON instead of FASTA).
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "homoeogwas-annot"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read().decode()
        except Exception as e:  # noqa: BLE001
            if i == tries - 1:
                print(f"    REST fail {url}: {e}")
                return None
            time.sleep(2 + i)
    return None


def fetch_protein_and_domains():
    WORK.mkdir(parents=True, exist_ok=True)
    fasta = WORK / "wheat_cand.pep.fa"
    recs, domains = [], {}
    for _chrom, _copy, g, _role in GENES:
        tx = f"{g}.1"
        seq = _get(f"{REST}/sequence/id/{tx}?type=protein;multiple_sequences=1;content-type=text/x-fasta")
        if seq and seq.startswith(">"):
            body = "".join(seq.splitlines()[1:])
            recs.append(f">{g}\n{body}")
        # protein-domain features (Pfam/InterPro/etc) on the translation
        pf = _get(f"{REST}/overlap/translation/{tx}?content-type=application/json")
        doms = []
        if pf:
            try:
                for d in json.loads(pf):
                    typ = d.get("type", "")
                    desc = d.get("description") or d.get("interpro_description") or ""
                    pid = d.get("id", "")
                    if typ.lower() in ("pfam", "smart", "prosite_profiles", "tigrfam", "panther") and (desc or pid):
                        doms.append(f"{typ}:{pid}{(' ' + desc) if desc else ''}")
            except Exception:  # noqa: BLE001
                pass
        domains[g] = sorted(set(doms))[:6]
        print(f"  {g}: protein={'ok' if recs and recs[-1].startswith('>'+g) else 'MISSING'} "
              f"domains={len(domains[g])}")
        time.sleep(0.2)
    fasta.write_text("\n".join(recs) + "\n")
    return fasta, domains, len(recs)


def blast_to_tair(fasta):
    db = WORK / "tair10"
    pep = WORK / "tair10.pep.fa"
    subprocess.run(f"zcat {TAIR} > {pep}", shell=True, check=True)
    subprocess.run(["makeblastdb", "-in", str(pep), "-dbtype", "prot", "-out", str(db)],
                   capture_output=True, check=True)
    res = subprocess.run(["blastp", "-query", str(fasta), "-db", str(db), "-evalue", "1e-3",
                          "-max_target_seqs", "3", "-outfmt",
                          "6 qseqid sseqid pident length qlen slen evalue bitscore qcovs stitle"],
                         capture_output=True, text=True, check=True)
    # TAIR header description map (locus -> description) from the FASTA
    desc = {}
    import gzip
    with gzip.open(TAIR, "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                tid = line[1:].split()[0]
                d = line.split("description:", 1)[1].strip() if "description:" in line else ""
                sym = line.split("gene_symbol:", 1)[1].split(" description:")[0].strip() if "gene_symbol:" in line else ""
                desc[tid] = (sym, d)
    best = {}
    for ln in res.stdout.splitlines():
        p = ln.split("\t")
        q, s, pid, _l, _ql, _sl, ev, bit, qcov = p[0], p[1], float(p[2]), p[3], p[4], p[5], p[6], float(p[7]), p[8]
        if q not in best:  # first = best by bitscore
            sym, d = desc.get(s, ("", ""))
            best[q] = dict(at_locus=s, at_symbol=sym, at_desc=d, pident=pid, evalue=ev,
                           bitscore=bit, qcov=float(qcov),
                           low_conf=bool(pid < 30 or float(qcov) < 50))
    return best


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    print("=== wheat candidate gene annotation (Ensembl domains + BLASTp->TAIR10) ===")
    fasta, domains, n = fetch_protein_and_domains()
    print(f"  fetched {n}/6 proteins; running blastp vs TAIR10 ...")
    best = blast_to_tair(fasta)
    rows = []
    for chrom, copy, g, role in GENES:
        b = best.get(g, {})
        rows.append(dict(locus=chrom, copy=copy, gene=g, role=role,
                         pfam_interpro=domains.get(g, []),
                         arabidopsis_homolog=b.get("at_locus", "none"),
                         at_symbol=b.get("at_symbol", ""), at_description=b.get("at_desc", ""),
                         blast_pident=b.get("pident"), blast_evalue=b.get("evalue"),
                         blast_qcov=b.get("qcov"), blast_low_conf=b.get("low_conf"),
                         annotation_note=("no usable domain/ortholog (unknown protein)"
                                          if not domains.get(g) and not b.get("at_locus") else "")))
        print(f"  [{chrom} {copy} {g}] dom={domains.get(g)} -> AT {b.get('at_symbol','-')} "
              f"{b.get('at_desc','')[:50]} (id={b.get('pident')}% e={b.get('evalue')} cov={b.get('qcov')}%)")
    payload = dict(task="wheat_candidate_annotation",
                   annotation_type="sequence-homology (Pfam/PANTHER domains + BLASTp), NOT functional validation",
                   databases="Ensembl Plants REST (Triticum_aestivum IWGSC) + Arabidopsis TAIR10 proteome",
                   blast_rules="blastp evalue<1e-3, best hit by bitscore, COMPARATIVE HOMOLOG (not reciprocal-best / not phylogenetically verified ortholog); low_conf if pident<30%% or qcov<50%%",
                   method=("Ensembl Plants protein-feature domains (wheat-native Pfam/PANTHER) + blastp "
                           "vs local Arabidopsis TAIR10 = closest comparative homolog. All 6 homoeolog "
                           "copies incl. chr5 balanced contrast; 'unknown' reported honestly. "
                           "Sequence-homology annotation, NOT experimental functional validation. No fabrication."),
                   genes=rows)
    OUT.write_text(json.dumps(payload, indent=2, default=float))
    print(f"\n  wrote {OUT}")


if __name__ == "__main__":
    main()
