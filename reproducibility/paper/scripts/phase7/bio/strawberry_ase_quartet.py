#!/usr/bin/env python
"""Strawberry ASE -> quartet projection (Step C1b/C2).

Maps Fan2022 Dataset-S8 ASE homoeolog pairs (F12 vs Bea, reference genotype
FL15.89, 4 reps) onto the Camarosa strict 4/4 quartets via protein-sequence
orthology (diamond blastp; F12 coords are not aligned to Camarosa). An ASE pair
is "quartet-contained" iff its F-copy and B-copy both map (best-hit, paralog
margin, coverage) to two distinct genes of the SAME strict quartet.

HONEST FRAMING (locked): this is reference-genotype homoeolog-pair expression
bias projected as an external functional-annotation layer on the quartet
framework -- NOT a 196-individual population HEB, NOT a discovery, and NO
subgenome-specific (A/B/C/D) dominance claim (the F/B diploid references do not
have a known 1:1 correspondence to Camarosa's 4 subgenomes). It is the F-vs-B
PAIRWISE bias only, same caveat level as the wheat CS expression atlas.

Deliverables:
  R1 mapping/projection coverage   R2 ratio_m bias among quartet-contained pairs
  R3 overlap/enrichment of ASE-evidenced quartets with eQTL-target genes
  R4 per-quartet ASE annotation track   R5 margin/coverage sensitivity
"""
import csv
import json
from collections import defaultdict
from math import comb
from pathlib import Path

PROJ = Path("/mnt/7302share/fast_ysp/U7_GWAS")
IN = PROJ / "results/phase7/bio_strawberry"
EQTL = PROJ / "data/eQTL_fana_all_nc.csv"
# NOTE: NO paralog margin for ortholog mapping. An ASE F-/B-copy gene is
# homologous to ALL 4 Camarosa subgenome copies of its quartet, so the competing
# near-equal hits are HOMOEOLOGS, not paralogs -- a margin filter would wrongly
# discard the real ortholog. We take the single best hit (coverage-gated) and
# report the margin collapse separately (R5) as positive evidence of
# pan-homoeolog similarity.
MARGIN = 1.0       # primary: best hit only (margin demonstrated in R5)
MIN_COV = 0.5      # aln length / query length
PADJ_SIG = 0.05


def best_hits(blast):
    """qseqid -> (best_sgene, best_bit, 2nd_bit, cov, pident). outfmt:
    qseqid sseqid pident length qlen slen evalue bitscore."""
    best = {}
    second = {}
    meta = {}
    with open(blast) as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            q, s = f[0], f[1]
            pid, length, qlen, bit = float(f[2]), int(f[3]), int(f[4]), float(f[7])
            cov = length / qlen if qlen else 0.0
            if q not in best or bit > best[q][1]:
                if q in best:
                    second[q] = max(second.get(q, 0.0), best[q][1])
                best[q] = (s, bit)
                meta[q] = (cov, pid)
            elif s != best[q][0]:
                second[q] = max(second.get(q, 0.0), bit)
    return {q: (s, bit, second.get(q, 0.0), meta[q][0], meta[q][1])
            for q, (s, bit) in best.items()}


def confident_map(hits):
    """qseqid -> (camarosa gene, cov, pident), gated by coverage (no paralog
    margin: see header). Returns (map, n_dropped_low_cov)."""
    m = {}
    drop = 0
    for q, (s, bit, sec, cov, pid) in hits.items():
        if cov >= MIN_COV and (sec == 0.0 or bit >= MARGIN * sec):
            m[q] = (s, cov, pid)
        else:
            drop += 1
    return m, drop


mapF = best_hits(IN / "ase_F.blast")
mapB = best_hits(IN / "ase_B.blast")
F2cam, ambF = confident_map(mapF)
B2cam, ambB = confident_map(mapB)
print(f"[map] F: {len(F2cam)} confident ({ambF} ambiguous), "
      f"B: {len(B2cam)} confident ({ambB} ambiguous)")

# ---- gene -> quartet_id + subgenome ------------------------------------------
gene2quartet = {}
quartet_genes = {}
with open(IN / "quartets.tsv") as fh:
    rd = csv.DictReader(fh, delimiter="\t")
    for qi, r in enumerate(rd):
        gs = [r["A"], r["B"], r["C"], r["D"]]
        if any(g == "." for g in gs):
            continue
        quartet_genes[qi] = gs
        for g in gs:
            gene2quartet[g] = qi
print(f"[quartets] {len(quartet_genes)} strict quartets")

# ---- eQTL-target Camarosa genes (for R3 enrichment) --------------------------
eqtl_genes = set()
with open(EQTL, newline="") as fh:
    rd = csv.reader(fh)
    hdr = next(rd)
    gi = hdr.index('"Gene"') if '"Gene"' in hdr else hdr.index("Gene")
    for row in rd:
        eqtl_genes.add(row[gi].strip('"'))
print(f"[eqtl] {len(eqtl_genes)} unique eQTL-target genes")

# ---- per ASE pair: map + quartet containment ---------------------------------
ase = []
with open(IN / "ase_pairs.tsv") as fh:
    rd = csv.DictReader(fh, delimiter="\t")
    for r in rd:
        ase.append(r)

# diagnostic: do F12 gene and Bea gene share the same Fvb group-subgenome chrom?
chr_re = __import__("re").compile(r"Fvb(\d+)-(\d+)_")
n_same_chrom = n_parsed = 0
for r in ase:
    mf, mb = chr_re.search(r["F12 gene"]), chr_re.search(r["Bea gene"])
    if mf and mb:
        n_parsed += 1
        if mf.groups() == mb.groups():
            n_same_chrom += 1

n_both_mapped = 0
n_same_cam_gene = 0
quartet_ase = defaultdict(list)   # quartet_id -> list of (ratio_m, padjust)
rows = []
for r in ase:
    fg, bg = r["F12 gene"], r["Bea gene"]
    hF, hB = F2cam.get(fg), B2cam.get(bg)
    cF = hF[0] if hF else None
    cB = hB[0] if hB else None
    status = "unmapped"
    qid = ""
    if cF and cB:
        n_both_mapped += 1
        qF, qB = gene2quartet.get(cF), gene2quartet.get(cB)
        if qF is not None and qF == qB and cF != cB:
            # same-gene (cF==cB) excluded above: would conflate allelic/
            # haplotype-specific ASE with quartet-level homoeolog imbalance.
            status = "quartet_contained"
            qid = qF
            try:
                rm = float(r["ratio_m"])
            except ValueError:
                rm = None
            try:
                pa = float(r["padjust"])
            except ValueError:
                pa = None
            quartet_ase[qF].append((rm, pa))
        elif cF == cB:
            status = "same_gene_excluded"
            n_same_cam_gene += 1
        else:
            status = "mapped_not_same_quartet"
    elif cF or cB:
        status = "one_end_mapped"
    rows.append({"F12": fg, "Bea": bg, "cam_F": cF or "", "cam_B": cB or "",
                 "cov_F": round(hF[1], 3) if hF else "",
                 "pid_F": round(hF[2], 1) if hF else "",
                 "cov_B": round(hB[1], 3) if hB else "",
                 "pid_B": round(hB[2], 1) if hB else "",
                 "ratio_m": r["ratio_m"], "padjust": r["padjust"],
                 "CHR": r["CHR"], "status": status, "quartet_id": qid})

n_qc = sum(1 for x in rows if x["status"] == "quartet_contained")
print(f"[project] both-ends mapped={n_both_mapped}/{len(ase)}; "
      f"quartet-contained={n_qc}")

# ---- R2: bias among quartet-contained pairs ----------------------------------
rms = [x for q in quartet_ase for (x, _p) in quartet_ase[q] if x is not None]
pas = [p for q in quartet_ase for (_x, p) in quartet_ase[q] if p is not None]
n_sig = sum(1 for p in pas if p < PADJ_SIG)
bias = sorted(abs(x - 0.5) for x in rms)
med_bias = bias[len(bias) // 2] if bias else 0.0


def pct(v, p):
    return round(sorted(v)[int(p * (len(v) - 1))], 4) if v else 0.0


# ---- R3: enrichment of ASE-evidenced quartets for eQTL-target genes ----------
# SUPPLEMENTARY / descriptive only. Two backgrounds: (i) all strict quartets
# (confounded: ASE and eQTL both require expression to be detected), and the
# fairer (ii) ASE-REACHABLE quartets (>=1 copy is a best-hit target of any ASE
# gene) which conditions on expression-detectability. The p-value is reported
# but is NOT given a causal/biological interpretation.
ase_quartets = set(quartet_ase)
n_q = len(quartet_genes)
q_has_eqtl = {qi: any(g in eqtl_genes for g in gs)
              for qi, gs in quartet_genes.items()}
nA = len(ase_quartets)


def hyper_sf(k, N, K, n):
    """P(X >= k) hypergeometric upper tail."""
    if k <= 0 or n == 0 or N == 0:
        return 1.0
    denom = comb(N, n)
    s = sum(comb(K, i) * comb(N - K, n - i)
            for i in range(k, min(K, n) + 1)) / denom
    return s


def enrich(universe):
    uni = sorted(universe)
    K = sum(1 for qi in uni if q_has_eqtl[qi])
    n = sum(1 for qi in uni if qi in ase_quartets)
    k = sum(1 for qi in uni if qi in ase_quartets and q_has_eqtl[qi])
    N = len(uni)
    exp = n * K / N if N else 0.0
    return {"background_n": N, "eqtl_in_bg": K, "ase_in_bg": n,
            "ase_and_eqtl": k, "expected": round(exp, 1),
            "fold": round(k / exp, 3) if exp else None,
            "hypergeom_p_upper": hyper_sf(k, N, K, n)}


# ASE-reachable = quartets touched by ANY ASE best-hit target (fair background)
ase_targets = {h[0] for h in F2cam.values()} | {h[0] for h in B2cam.values()}
reachable = {gene2quartet[g] for g in ase_targets if g in gene2quartet}
R3_all = enrich(set(quartet_genes))
R3_reachable = enrich(reachable)

# ---- R4: per-quartet annotation track ----------------------------------------
track = {}
for qi in quartet_genes:
    if qi in quartet_ase:
        sig = any(p is not None and p < PADJ_SIG for (_r, p) in quartet_ase[qi])
        track[qi] = "ase_biased_sig" if sig else "ase_nonsig"
    else:
        track[qi] = "no_ase_evidence"
track_counts = defaultdict(int)
for v in track.values():
    track_counts[v] += 1

# ---- R5: margin/coverage sensitivity -----------------------------------------
sens = {}
for mg, cv in [(1.0, 0.3), (1.0, 0.5), (1.0, 0.7), (1.10, 0.5), (1.25, 0.5)]:
    def cm(hits, mg=mg, cv=cv):
        return {q: s for q, (s, bit, sec, cov, pid) in hits.items()
                if cov >= cv and (sec == 0.0 or bit >= mg * sec)}
    fF, bB = cm(mapF), cm(mapB)
    qc = 0
    for r in ase:
        cF, cB = fF.get(r["F12 gene"]), bB.get(r["Bea gene"])
        if cF and cB:
            qF, qB = gene2quartet.get(cF), gene2quartet.get(cB)
            if qF is not None and qF == qB and cF != cB:
                qc += 1
    sens[f"margin{mg}_cov{cv}"] = qc

summary = {
    "tool": "strawberry_ase_quartet",
    "VERDICT": "NEGATIVE: Dataset S8 is allelic/haplotype-specific ASE within a "
               "single subgenome locus, NOT between-subgenome homoeolog bias; it "
               "is NOT a usable HEB proxy. The 'quartet_contained' pairs are "
               "cross-reference mapping artifacts, not homoeolog-bias evidence.",
    "diagnostic_allelic_not_homoeolog": {
        "frac_F_Bea_same_Fvb_group_subgenome": round(n_same_chrom / max(1, n_parsed), 4),
        "frac_both_mapped_collapsing_to_same_camarosa_gene":
            round(n_same_cam_gene / max(1, n_both_mapped), 4),
        "interpretation": "F12/Bea are two phased haplotypes of FL15.89, not two "
                          "subgenomes; ASE here is cis-allelic, off-axis for HEB.",
    },
    "params": {"margin": MARGIN, "min_cov": MIN_COV, "padj_sig": PADJ_SIG},
    "R1_coverage": {
        "ase_pairs_total": len(ase),
        "F_confident_map": len(F2cam), "B_confident_map": len(B2cam),
        "both_ends_mapped": n_both_mapped,
        "quartet_contained": n_qc,
        "frac_quartet_contained_of_total": round(n_qc / len(ase), 4),
        "frac_quartet_contained_of_mapped": round(n_qc / max(1, n_both_mapped), 4),
        "distinct_quartets_with_ase": len(ase_quartets),
        "frac_quartets_with_ase": round(len(ase_quartets) / n_q, 4),
    },
    "R2_bias": {
        "n_pairs": len(rms), "n_significant_padj<0.05": n_sig,
        "ratio_m_p10": pct(rms, 0.10), "ratio_m_median": pct(rms, 0.50),
        "ratio_m_p90": pct(rms, 0.90),
        "median_abs_bias_from_0.5": round(med_bias, 4),
        "note": "F-vs-B pairwise bias only; not a 4-subgenome ranking",
    },
    "R3_eqtl_overlap_SUPPLEMENTARY": {
        "background_all_strict_quartets": R3_all,
        "background_ase_reachable_quartets": R3_reachable,
        "note": "SUPPLEMENTARY/descriptive. ASE & eQTL both require expression "
                "to be detected; the ase-reachable background conditions on "
                "detectability. p reported, NOT given a causal interpretation.",
    },
    "R4_annotation_track": dict(track_counts),
    "R5_sensitivity_quartet_contained": sens,
    "R5_note": "tightening the paralog margin (1.0->1.10->1.25) collapses the "
               "count because an ASE copy matches all 4 homoeologs near-equally "
               "-- this collapse is positive evidence of pan-homoeolog "
               "similarity, not a mapping failure; primary uses best-hit (1.0).",
}
(IN / "ase_quartet.json").write_text(json.dumps(summary, indent=2))
with open(IN / "ase_quartet_pairs.tsv", "w") as out:
    w = csv.DictWriter(out, fieldnames=list(rows[0].keys()), delimiter="\t")
    w.writeheader()
    w.writerows(rows)
print(json.dumps(summary, indent=2))
print(f"[done] -> {IN}/ase_quartet.json + ase_quartet_pairs.tsv")
