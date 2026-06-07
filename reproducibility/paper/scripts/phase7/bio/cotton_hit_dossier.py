#!/usr/bin/env python3
"""Task #3: cotton Hit2 (+ Hit1 caveated) candidate-gene DOSSIER.

Annotates the 4 cotton homoeolog-interaction hit genes (NDM8 assembly) with: NDM8 coordinates
(genes.tsv), closest Arabidopsis homolog via BLASTp of the NDM8 peptide vs TAIR10 (tiered confidence
tiered by confidence), and a PRE-DECLARED fibre-pathway cross-check. Claim ceiling = candidate-
gene plausibility + functional class, NOT causality/mechanism/expression. Hit2 is primary; Hit1 is a
caveated secondary (influence-sensitive, A06 TM-1 crosswalk failed). Expression (per-stage fibre TPM /
NDM8-native HEB) is BLOCKED: the 4 hit genes are absent from the only local NDM8 fibre file (a curated
3,089-gene 4-pattern subset) and no NDM8-native full fibre transcriptome was found — reported as a
limitation, NOT as biological absence.
"""
from __future__ import annotations

import gzip
import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path("/mnt/7302share/fast_ysp/U7_GWAS")
GENES = ROOT / "results/phase7/bio_full/genes.tsv"
PEP = ROOT / "results/phase7/bio_full/union_pep.fa"
TAIR_GZ = ROOT / "data/reference/arabidopsis/TAIR10.pep.fa.gz"
WORK = ROOT / "results/phase7/bio_full/dossier"
OUT_JSON = WORK / "cotton_hit_dossier.json"
BIN = Path.home() / ".local/share/mamba/envs/polygwas-cpu/bin"

HITS = {
    "Hit2_length_uniformity_PRIMARY": dict(
        A="GhM_A11G2420", D="GhM_D11G2742", trait="fibre length uniformity (BLUE)",
        p_int_INT=4.75e-5, p_int_raw=4.93e-6, robustness="PASS (robust); max|DFBETA|=8.6%; distributed SNP-pair",
        crosswalk="A11->GH_A11G2162 (100% RBH); D11->GH_D11G2535 (100%)", tier="PRIMARY"),
    "Hit1_fibre_length_SECONDARY": dict(
        A="GhM_A06G1605", D="GhM_D06G1557", trait="fibre length (BLUE)",
        p_int_INT=2.14e-4, p_int_raw=1.74e-4,
        robustness="PASS_WITH_CAVEAT; influence-sensitive max|DFBETA|=38%; near single-SNP-pair",
        crosswalk="A06 FAILS TM-1 RBH (63%, chrom-discordant); D06->GH_D06G1406 (100%)", tier="SECONDARY_CAVEATED"),
}

# PRE-DECLARED fibre-development pathway categories (locked BEFORE looking at BLAST results; anti
# cherry-picking). Match is by Arabidopsis gene_symbol / description keywords.
FIBRE_PATHWAYS = {
    "cellulose_synthesis": ["CESA", "CSL", "cellulose synthase", "COBRA", "KORRIGAN", "KOR"],
    "cell_wall_loosening": ["EXPANSIN", "EXP", "XTH", "xyloglucan", "pectin", "PME", "PMEI"],
    "fibre_TF": ["MYB", "bHLH", "HD-ZIP", "HOMEOBOX", "GL2", "WRKY", "GLABRA", "TTG"],
    "cytoskeleton": ["TUBULIN", "ACTIN", "kinesin", "MAP", "microtubule"],
    "lipid_wax_VLCFA": ["KCS", "CER", "wax", "fatty acid elong", "lipid transfer", "LTP"],
    "hormone_signalling": ["auxin", "ethylene", "ACO", "ACS", "gibberellin", "GA20ox", "brassinosteroid", "BRI"],
    "sugar_osmotic_transport": ["sucrose synth", "SUS", "SWEET", "aquaporin", "PIP", "TIP",
                                "potassium", "K+ transport", "KT", "invertase"],
    "ROS_redox": ["peroxidase", "RBOH", "oxidase", "glutathione", "thioredoxin"],
}


def _load_pep(path, gz=False):
    seqs, cur = {}, None
    op = gzip.open if gz else open
    with op(path, "rt") as f:
        for line in f:
            if line.startswith(">"):
                cur = line[1:].split()[0]
                seqs[cur] = []
            elif cur:
                seqs[cur].append(line.strip())
    return {k: "".join(v) for k, v in seqs.items()}


def _tair_meta():
    """ATxgxxxxx.N -> (gene_symbol, description) from the TAIR10 fasta headers."""
    meta = {}
    with gzip.open(TAIR_GZ, "rt") as f:
        for line in f:
            if not line.startswith(">"):
                continue
            tid = line[1:].split()[0]
            sym = ""
            desc = ""
            if "gene_symbol:" in line:
                sym = line.split("gene_symbol:")[1].split()[0]
            if "description:" in line:
                desc = line.split("description:")[1].split("[Source")[0].strip()
            meta[tid] = (sym, desc)
    return meta


def _classify(best, second):
    """Tiered homology confidence from one-way BLASTp (no full-proteome RBH available)."""
    if best is None:
        return "no_hit", "no TAIR10 hit at e<=1e-3"
    e, pid, qcov, scov, bit = best["evalue"], best["pident"], best["qcov"], best["scov"], best["bitscore"]
    sep = (bit / second["bitscore"]) if (second and second["bitscore"] > 0) else float("inf")
    ambiguous = sep < 1.10
    if e <= 1e-10 and qcov >= 0.70 and scov >= 0.60 and pid >= 35 and not ambiguous:
        return "ortholog_grade_oneway", (f"closest TAIR10 homolog, high-confidence one-way "
                                         f"(e={e:.1e}, pid={pid:.0f}%, qcov={qcov:.0%}, scov={scov:.0%}, "
                                         f"bit-sep={sep:.2f}x); RBH not run (full NDM8 proteome absent)")
    if e <= 1e-5 and qcov >= 0.50 and pid >= 25:
        return "family_level", (f"closest TAIR10 family member (e={e:.1e}, pid={pid:.0f}%, "
                                f"qcov={qcov:.0%}); {'paralog-ambiguous ' if ambiguous else ''}NOT a 1:1 ortholog claim")
    return "weak", f"weak hit (e={e:.1e}, pid={pid:.0f}%, qcov={qcov:.0%}); domain-level at most"


def _pathway(sym, desc):
    text = f"{sym} {desc}".upper()
    hits = [cat for cat, kws in FIBRE_PATHWAYS.items() if any(k.upper() in text for k in kws)]
    return hits


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    print("=== cotton hit dossier (Hit2 primary, Hit1 caveated) ===")
    genes = pd.read_csv(GENES, sep="\t").set_index("gene_id")
    pep = _load_pep(PEP)

    # query fasta = the 4 hit peptides
    qfa = WORK / "hit_pep.fa"
    with qfa.open("w") as fh:
        for h in HITS.values():
            for g in (h["A"], h["D"]):
                fh.write(f">{g}\n{pep[g]}\n")

    # TAIR10 blast db
    tair_fa = WORK / "TAIR10.pep.fa"
    if not tair_fa.exists():
        with gzip.open(TAIR_GZ, "rt") as fi, tair_fa.open("w") as fo:
            shutil.copyfileobj(fi, fo)
    if not (WORK / "TAIR10.pep.fa.pdb").exists():
        subprocess.run([str(BIN / "makeblastdb"), "-in", str(tair_fa), "-dbtype", "prot"],
                       check=True, capture_output=True)
    bo = WORK / "blast_hits.tsv"
    subprocess.run([str(BIN / "blastp"), "-query", str(qfa), "-db", str(tair_fa),
                    "-evalue", "1e-3", "-max_target_seqs", "5", "-num_threads", "8",
                    "-outfmt", "6 qseqid sseqid pident length qlen slen evalue bitscore", "-out", str(bo)],
                   check=True, capture_output=True)

    meta = _tair_meta()
    # parse blast: per query keep ranked hits
    perq = {}
    for line in bo.read_text().splitlines():
        q, s, pid, ln, qlen, slen, ev, bit = line.split("\t")
        rec = dict(sid=s, pident=float(pid), qcov=int(ln) / int(qlen), scov=int(ln) / int(slen),
                   evalue=float(ev), bitscore=float(bit))
        perq.setdefault(q, []).append(rec)
    for q in perq:
        perq[q].sort(key=lambda r: -r["bitscore"])

    dossier = {}
    for hit_name, h in HITS.items():
        cards = {}
        for sub, g in (("A", h["A"]), ("D", h["D"])):
            hits = perq.get(g, [])
            best = hits[0] if hits else None
            second = hits[1] if len(hits) > 1 else None
            tier, note = _classify(best, second)
            sym, desc = meta.get(best["sid"], ("", "")) if best else ("", "")
            paths = _pathway(sym, desc) if best else []
            gi = genes.loc[g]
            cards[sub] = dict(
                gene=g, ndm8_chrom=str(gi["chrom"]), ndm8_start=int(gi["start"]), ndm8_end=int(gi["end"]),
                ndm8_len=int(gi["end"] - gi["start"] + 1), n_snp_flank=int(gi["n_snp_flank"]),
                pep_len=len(pep[g]),
                tair_best=(best["sid"] if best else None), tair_symbol=sym, tair_desc=desc,
                homology_tier=tier, homology_note=note,
                fibre_pathway_match=paths)
            print(f"  [{h['tier']}] {g} ({sub}) -> {best['sid'] if best else 'NO HIT'} "
                  f"{sym} | {tier} | pathways={paths}")
        dossier[hit_name] = dict(meta=h, A_card=cards["A"], D_card=cards["D"])

    payload = dict(
        task="cotton_hit_dossier", assembly="HBAU_NDM8_ASM1899796v1_GCA_018997965.1",
        prereg_fibre_pathways=FIBRE_PATHWAYS,
        ortholog_method=("one-way BLASTp vs TAIR10 (Ensembl Plants release-58), tiered; RBH NOT run "
                         "(full NDM8 proteome not local; union_pep.fa is the 8,091-gene homoeolog-pair "
                         "subset). Pfam/InterPro pending (hmmer not installed) — functional class via "
                         "the TAIR homolog symbol/description."),
        expression_status=("BLOCKED: the 4 hit genes are ABSENT from the only local NDM8-native fibre "
                           "file (a curated 3,089-gene 4-pattern subset; Gh_==GhM_ NDM8, 3074/3089 "
                           "suffix match). No NDM8-native full fibre transcriptome found. This is a "
                           "data limitation, NOT biological absence — per-stage TPM / A-D HEB not "
                           "computable here. Optional future: cross-reference TM-1 fibre atlas via the "
                           "existing NDM8->TM-1 crosswalk (extra assembly hop, to be caveated)."),
        claim_ceiling=("candidate-gene plausibility + functional class + Hit2 prioritisation; NOT "
                       "causality / mechanism / fibre-stage activity / expression bias / validation."),
        hits=dossier)
    OUT_JSON.write_text(json.dumps(payload, indent=2, default=float))
    print(f"  wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
