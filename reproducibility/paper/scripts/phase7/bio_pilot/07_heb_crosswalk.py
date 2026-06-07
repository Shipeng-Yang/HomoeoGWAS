#!/usr/bin/env python
"""Phase 7 cotton HEB weak-corroboration: NDM8 hit genes -> TM-1 v2.1 crosswalk + Table 3 membership.

RBH (reciprocal-best-hit) via diamond blastp, subgenome/chromosome anchoring,
ambiguity flagging, then binary membership in Frontiers-2021 Table 3 (A-biased HEB list).
Framing: hit = weak expression-context annotation only; non-hit != refutation.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

import openpyxl

TM1_PEP = Path("/mnt/nvme/cotton_hbau/TM1_v2.1.pep.fa")          # GH_ ids, 72761, no isoform
NDM8_PEP = Path("/mnt/nvme/cotton_hbau/cottongen/NDM8.pep.fa")   # GhM_..N isoforms, 80124
TABLE3 = Path("/mnt/7302share/fast_ysp/U7_GWAS/data/raw/expression/cotton/HEB_special_pattern_Table3.xlsx")
OUTDIR = Path("/mnt/7302share/fast_ysp/U7_GWAS/results/phase7/bio_pilot")
WORK = OUTDIR / "heb_crosswalk"
WORK.mkdir(parents=True, exist_ok=True)
OUT_JSON = OUTDIR / "cotton_heb_tm1_crosswalk.json"

# hit homoeolog pairs (A copy, D copy)
HITS = {
    "fiber_length_hit1": ("GhM_A06G1605", "GhM_D06G1557"),
    "length_uniformity_hit2": ("GhM_A11G2420", "GhM_D11G2742"),
    "dosage_imbalance_Ihit": ("GhM_A05G3471", "GhM_D05G3365"),
}
QUERY_GENES = sorted({g for pair in HITS.values() for g in pair})

# thresholds (Codex fix: same-species upland cotton, orthologs should be strong)
EVALUE_MAX = 1e-20
QCOV_MIN = 70.0
SCOV_MIN = 70.0
PIDENT_MIN = 70.0
SEPARATION_MIN = 1.02  # top/second bitscore ratio for unambiguous best
THREADS = "32"
DIAMOND = "/home/yys05/.local/share/mamba/envs/polygwas-cpu/bin/diamond"

FIELDS = ["qseqid", "sseqid", "pident", "length", "evalue", "bitscore", "qcovhsp", "scovhsp"]


def read_fasta(path):
    seqs, name, buf = {}, None, []
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                if name is not None:
                    seqs[name] = "".join(buf)
                name = line[1:].strip().split()[0]
                buf = []
            else:
                buf.append(line.strip())
    if name is not None:
        seqs[name] = "".join(buf)
    return seqs


def gene_of(seqid):
    """Collapse protein/isoform id to gene id. GhM_A06G1605.1 -> GhM_A06G1605; GH_A01G0001 -> itself."""
    return re.sub(r"\.\d+$", "", seqid)


def write_fasta(seqs, path):
    with open(path, "w") as fh:
        for name, seq in seqs.items():
            fh.write(f">{name}\n{seq}\n")


def run(cmd):
    print("  $", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def makedb(fasta, dbpath):
    if not Path(str(dbpath) + ".dmnd").exists():
        run([DIAMOND, "makedb", "--in", str(fasta), "-d", str(dbpath), "--threads", THREADS,
             "--quiet"])


def blastp(query_fa, db, out_tsv, max_target=5):
    run([DIAMOND, "blastp", "-q", str(query_fa), "-d", str(db), "-o", str(out_tsv),
         "--outfmt", "6", *FIELDS, "--max-target-seqs", str(max_target),
         "--threads", THREADS, "--quiet", "--evalue", "1e-5"])


def parse_tsv(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) != len(FIELDS):
                continue
            rec = dict(zip(FIELDS, p))
            for k in ("pident", "evalue", "bitscore", "qcovhsp", "scovhsp"):
                rec[k] = float(rec[k])
            rec["length"] = int(rec["length"])
            rows.append(rec)
    return rows


def best_per_query_gene(rows):
    """Group rows by query gene, return ranked gene-level hits by bitscore (best HSP per target gene)."""
    by_q = {}
    for r in rows:
        qg = gene_of(r["qseqid"])
        tg = gene_of(r["sseqid"])
        d = by_q.setdefault(qg, {})
        if tg not in d or r["bitscore"] > d[tg]["bitscore"]:
            d[tg] = r
    ranked = {}
    for qg, d in by_q.items():
        ranked[qg] = sorted(d.values(), key=lambda x: -x["bitscore"])
    return ranked


def subgenome_chrom(gene):
    """GhM_A06G1605 / GH_A06G0001 -> ('A','06'); GhM_D11G.. -> ('D','11'). None if unparseable."""
    m = re.search(r"_([AD])(\d{2})G", gene)
    if m:
        return m.group(1), m.group(2)
    return None, None


def main():
    tm1 = read_fasta(TM1_PEP)
    ndm8 = read_fasta(NDM8_PEP)

    # build query.fa (the 6 hit gene proteins, single isoform .1 each)
    q_seqs = {}
    for g in QUERY_GENES:
        pid = f"{g}.1"
        if pid not in ndm8:
            sys.exit(f"FATAL: {pid} not in NDM8 PEP")
        q_seqs[pid] = ndm8[pid]
    write_fasta(q_seqs, WORK / "query.fa")

    # diamond dbs
    print("[1] makedb", flush=True)
    makedb(TM1_PEP, WORK / "tm1")
    makedb(NDM8_PEP, WORK / "ndm8")

    # forward: query (NDM8) vs TM-1
    print("[2] forward blastp query->TM1", flush=True)
    blastp(WORK / "query.fa", WORK / "tm1", WORK / "fwd.tsv")
    fwd = best_per_query_gene(parse_tsv(WORK / "fwd.tsv"))

    # reverse: forward-best TM-1 genes vs NDM8 full
    fwd_best_tm1 = {}
    for qg in QUERY_GENES:
        hits = fwd.get(qg, [])
        if hits:
            fwd_best_tm1[qg] = gene_of(hits[0]["sseqid"])
    rev_q = {}
    for tg in set(fwd_best_tm1.values()):
        if tg in tm1:
            rev_q[tg] = tm1[tg]
    write_fasta(rev_q, WORK / "rev_query.fa")
    print("[3] reverse blastp TM1-best->NDM8", flush=True)
    blastp(WORK / "rev_query.fa", WORK / "ndm8", WORK / "rev.tsv")
    rev = best_per_query_gene(parse_tsv(WORK / "rev.tsv"))

    # Table 3 A-biased lists (col A = 10 DPA, col D = 20 DPA), GH_ ids
    wb = openpyxl.load_workbook(TABLE3, read_only=True)
    ws = wb["S3"]
    set_10dpa, set_20dpa = set(), set()
    for row in ws.iter_rows(min_row=4, values_only=True):
        if row[0] and isinstance(row[0], str) and row[0].startswith("GH_"):
            set_10dpa.add(row[0].strip())
        if len(row) > 3 and row[3] and isinstance(row[3], str) and row[3].startswith("GH_"):
            set_20dpa.add(row[3].strip())
    table3_union = set_10dpa | set_20dpa

    # assemble per-gene crosswalk verdicts
    results = {}
    for qg in QUERY_GENES:
        sub_q, chr_q = subgenome_chrom(qg)
        hits = fwd.get(qg, [])
        rec = {"query_gene": qg, "query_subgenome": sub_q, "query_chrom": chr_q,
               "n_fwd_targets": len(hits)}
        if not hits:
            rec["status"] = "no_fwd_hit"
            results[qg] = rec
            continue
        top = hits[0]
        tg = gene_of(top["sseqid"])
        sub_t, chr_t = subgenome_chrom(tg)
        rec.update({
            "tm1_gene": tg, "pident": top["pident"], "evalue": top["evalue"],
            "bitscore": top["bitscore"], "qcov": top["qcovhsp"], "scov": top["scovhsp"],
            "tm1_subgenome": sub_t, "tm1_chrom": chr_t,
        })
        # separation
        if len(hits) > 1 and hits[1]["bitscore"] > 0:
            rec["bitscore_separation"] = round(top["bitscore"] / hits[1]["bitscore"], 4)
            rec["second_tm1_gene"] = gene_of(hits[1]["sseqid"])
        else:
            rec["bitscore_separation"] = None
        # RBH
        rev_hits = rev.get(tg, [])
        rec["rbh"] = bool(rev_hits and gene_of(rev_hits[0]["sseqid"]) == qg)
        rec["rev_best_ndm8"] = gene_of(rev_hits[0]["sseqid"]) if rev_hits else None
        # threshold pass
        rec["pass_thresholds"] = bool(
            top["evalue"] <= EVALUE_MAX and top["qcovhsp"] >= QCOV_MIN
            and top["scovhsp"] >= SCOV_MIN and top["pident"] >= PIDENT_MIN)
        # subgenome anchoring
        rec["subgenome_concordant"] = (sub_q == sub_t)
        rec["chrom_concordant"] = (sub_q == sub_t and chr_q == chr_t)
        # ambiguity
        ambig = []
        if rec["bitscore_separation"] is not None and rec["bitscore_separation"] < SEPARATION_MIN:
            ambig.append("low_bitscore_separation")
        if not rec["subgenome_concordant"]:
            ambig.append("homoeolog_subgenome_discordant")
        if not rec["rbh"]:
            ambig.append("not_reciprocal_best")
        rec["ambiguity_flags"] = ambig
        rec["crosswalk_confident"] = bool(
            rec["rbh"] and rec["pass_thresholds"] and rec["subgenome_concordant"]
            and not ("low_bitscore_separation" in ambig))
        # Table 3 membership (only meaningful for A-copy; D rows not in this A-biased table)
        rec["in_table3_10dpa"] = tg in set_10dpa
        rec["in_table3_20dpa"] = tg in set_20dpa
        rec["in_table3_any"] = tg in table3_union
        results[qg] = rec

    # supplementary D-anchored A-homoeolog check (robustness vs A-copy crosswalk ambiguity):
    # the D copy often hits its TM-1 D ortholog (best) AND the TM-1 A homoeolog (cross-subgenome 2nd),
    # giving an alternative, often cleaner, TM-1 A-side representative for the pair.
    d_anchored = {}
    for trait, (a, d) in HITS.items():
        hits = fwd.get(d, [])
        # best A-subgenome target among the D-copy's hits
        a_homoeolog = None
        for h in hits:
            tg = gene_of(h["sseqid"])
            sub_t, _ = subgenome_chrom(tg)
            if sub_t == "A":
                a_homoeolog = {"tm1_A_gene": tg, "pident": h["pident"], "qcov": h["qcovhsp"],
                               "scov": h["scovhsp"], "evalue": h["evalue"],
                               "in_table3_any": tg in table3_union,
                               "in_table3_10dpa": tg in set_10dpa,
                               "in_table3_20dpa": tg in set_20dpa}
                break
        d_anchored[trait] = {"d_copy": d, "inferred_tm1_A_homoeolog": a_homoeolog}

    payload = {
        "description": "cotton HEB weak corroboration: NDM8 hit genes -> TM-1 v2.1 RBH crosswalk + Table3 (A-biased HEB) membership",
        "d_anchored_A_homoeolog_check": d_anchored,
        "framing_redline": {
            "CONCLUSION": "Cotton HEB lookup yielded NO overlap with the Frontiers-2021 A-biased HEB list for any direct A-RBH or D-anchored inferred-A TM-1 representative; retained ONLY as an information-absence annotation, NOT used as supporting evidence for the main cotton statistical hits. wheat HEB remains the strong expression evidence.",
            "wording": "call this an exploratory cross-assembly background lookup, NOT 'HEB support'/'consistent with HEB'. Only claim 'no positive overlap observed', never 'negative evidence'.",
            "evidence_layering": "report three layers separately, do NOT treat D-anchored as equivalent to A-copy: (i) direct A-copy RBH, (ii) D-copy RBH, (iii) D-anchored INFERRED A homoeolog (cross-subgenome best hit of D copy, NOT a direct A-copy mapping).",
            "hit_would_mean": "weak expression-context annotation only: TM-1 ortholog appears in published HEB fiber At>Dt list; compatible with A-side expression bias for this homoeolog pair. NOT statistical validation / NOT eQTL / NOT GWAS replication / NOT proof A-copy is functional-dominant / A-biased expression != burden-product interaction.",
            "nonhit_means": "information absence, NOT refutation (Table3 is a special-pattern A-biased SUBSET, specific 10/20 DPA, different population TM-1/Hai7124, different assembly, binary list).",
            "A06G1605_note": "primary-hit A copy GhM_A06G1605 has NO clean TM-1 ortholog (best 63.4% paralog GH_A12G0414; D-copy cleanly identifies TM-1 pair GH_A06G1369|GH_D06G1406). This is an NDM8/TM-1 annotation discordance confined to the cross-assembly crosswalk layer; it does NOT bear on the NDM8 locus-level statistical association, which is defined entirely in NDM8 coordinates.",
        },
        "thresholds": {"evalue_max": EVALUE_MAX, "qcov_min": QCOV_MIN, "scov_min": SCOV_MIN,
                       "pident_min": PIDENT_MIN, "separation_min": SEPARATION_MIN},
        "table3_sizes": {"n_10dpa_A": len(set_10dpa), "n_20dpa_A": len(set_20dpa),
                         "n_union": len(table3_union)},
        "hit_pairs": HITS,
        "note_isoform": "each query gene has exactly one isoform (.1) in NDM8 PEP; TM-1 PEP has no isoforms -> representative-protein choice is unambiguous",
        "crosswalk": results,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2))
    print(f"\n[written] {OUT_JSON}")

    # console summary
    print("\n=== CROSSWALK SUMMARY ===")
    for trait, (a, d) in HITS.items():
        print(f"\n{trait}: {a} | {d}")
        for g in (a, d):
            r = results[g]
            if r.get("status") == "no_fwd_hit":
                print(f"  {g}: NO FORWARD HIT")
                continue
            t3 = "TABLE3-HIT" if r["in_table3_any"] else "table3-absent"
            t3d = f"(10DPA={r['in_table3_10dpa']},20DPA={r['in_table3_20dpa']})"
            print(f"  {g} -> {r['tm1_gene']}  pid={r['pident']:.1f} qcov={r['qcov']:.0f} "
                  f"scov={r['scov']:.0f} ev={r['evalue']:.0e} RBH={r['rbh']} "
                  f"sub_conc={r['subgenome_concordant']} chr_conc={r['chrom_concordant']} "
                  f"conf={r['crosswalk_confident']} sep={r['bitscore_separation']} "
                  f"flags={r['ambiguity_flags']}")
            print(f"      Table3: {t3} {t3d}")


if __name__ == "__main__":
    main()
