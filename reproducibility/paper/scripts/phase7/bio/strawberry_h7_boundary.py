#!/usr/bin/env python
"""Strawberry 8n H7 callable-pair boundary (Step 2).

Projects the octoploid (Fragaria x ananassa, Camarosa) array eQTL panel onto the
existing H7 feasibility framework WITHOUT recalibrating it (no circularity): the
0-70 / 70-300 / 300-1000 / 1000+ callable-pair bands and the min_snp>=3 reference
are taken as-is from scripts/phase7/h7_marker_density.py.

Definition matched to the H7 anchors:
  - in-gene SNP count = unique markers whose position falls INSIDE a gene span
    (gene-span containment only; nearest-gene assignment is NOT used, it would
    inflate per-gene callable counts in an 8n genome).
  - a homoeolog PAIR is callable iff BOTH copies carry >= min_snp in-gene SNPs.
  - G_callable = number of callable homoeolog pairs in scope (all 6 pairs per
    strict 4/4 quartet, genome-wide).

Marker source = eQTL_fana_all_nc.csv (1.69M marker x gene assoc rows; deduped to
unique (chr_id,pos) array sites). Quartets = strict 4/4 from
strawberry_quartet_assemble.py. Conclusion is feasibility/boundary, NOT discovery
(no dense WGS + phenotype on this panel).
"""
import csv
import gzip
import json
import re
from bisect import bisect_right
from collections import defaultdict
from itertools import combinations
from pathlib import Path

PROJ = Path("/mnt/7302share/fast_ysp/U7_GWAS")
IN = PROJ / "results/phase7/bio_strawberry"
GFF = PROJ / "data/raw/strawberry/camarosa/gff.gz"
MARKERS = PROJ / "data/eQTL_fana_all_nc.csv"
MIN_SNP_GRID = [1, 2, 3, 4, 5]
REF_MIN_SNP = 3
BANDS = [(0, 70, "underpowered/null-likely"), (70, 300, "borderline/risky"),
         (300, 1000, "discovery-feasible"), (1000, None, "strong-design")]


def classify(g):
    for lo, hi, lab in BANDS:
        if g >= lo and (hi is None or g < hi):
            return lab
    return "?"


CHR_RE = re.compile(r"^Fvb(\d+)-(\d+)$")
GID_RE = re.compile(r"^FxaC_(\d+)g")

# ---- gene spans per seqid + chr_id->seqid map (gene FxaC_<chrid>g on Fvb{g}-{s}) ----
gene_span = {}                       # gene_id -> (seqid, start, end)
spans_by_chr = defaultdict(list)     # seqid -> [(start, end, gene_id)]
chrid_to_seqid = {}                  # eQTL chr_id (1..28) -> Fvb seqid
with gzip.open(GFF, "rt") as fh:
    for line in fh:
        if line.startswith("#") or not line.strip():
            continue
        c = line.rstrip("\n").split("\t")
        if len(c) < 9 or c[2] != "gene" or not CHR_RE.match(c[0]):
            continue
        m = re.search(r"ID=([^;]+)", c[8])
        if not m:
            continue
        gid, s, e = m.group(1), int(c[3]), int(c[4])
        gene_span[gid] = (c[0], s, e)
        spans_by_chr[c[0]].append((s, e, gid))
        cm = GID_RE.match(gid)
        if cm:
            chrid_to_seqid[int(cm.group(1))] = c[0]

# sort spans + build start arrays for bisect containment lookup
starts = {}
for sq, lst in spans_by_chr.items():
    lst.sort()
    starts[sq] = [x[0] for x in lst]
print(f"[gff] {len(gene_span)} genes; chr_id map covers {len(chrid_to_seqid)}/28")

# ---- unique array markers (dedup once), then assign at flank 0 and +-1kb ------
markers = []  # (seqid, pos) for unique, mappable sites
seen = set()
n_rows = n_uniq = 0
with open(MARKERS, newline="") as fh:
    rd = csv.reader(fh)
    next(rd)  # header
    for row in rd:
        n_rows += 1
        chr_id, pos = row[0], row[2]
        key = (chr_id, pos)
        if key in seen:
            continue
        seen.add(key)
        n_uniq += 1
        sq = chrid_to_seqid.get(int(chr_id))
        if sq is not None:
            markers.append((sq, int(pos)))


def assign(flank):
    """marker -> gene by span (+/- flank) containment; returns (gene_snp, n_in)."""
    gs = defaultdict(int)
    n_in = 0
    for sq, p in markers:
        lst, st = spans_by_chr[sq], starts[sq]
        i = bisect_right(st, p) - 1  # rightmost gene whose start <= pos
        j = i                        # scan back for overlapping spans
        while j >= 0 and j >= i - 8:
            s, e, gid = lst[j]
            if s - flank <= p <= e + flank:
                gs[gid] += 1
                n_in += 1
                break
            j -= 1
    return gs, n_in


gene_snp, n_assigned = assign(0)              # primary: gene-span containment only
gene_snp_flank, n_assigned_flank = assign(1000)  # sensitivity: +-1 kb
print(f"[markers] {n_rows} assoc rows -> {n_uniq} unique sites; "
      f"{n_assigned} in gene span ({100*n_assigned/n_uniq:.1f}%); "
      f"{n_assigned_flank} within +-1kb ({100*n_assigned_flank/n_uniq:.1f}%)")

# ---- load strict quartets ----------------------------------------------------
quartets = []
with open(IN / "quartets.tsv") as fh:
    rd = csv.DictReader(fh, delimiter="\t")
    for r in rd:
        genes = [r["A"], r["B"], r["C"], r["D"]]
        if all(g != "." for g in genes):
            quartets.append((r["group"], genes))
print(f"[quartets] {len(quartets)} strict 4/4 with all 4 copies present")

# ---- callable pairs per min_snp ----------------------------------------------
quartet_genes = {g for _, gs in quartets for g in gs}


def callable_curve(gs_map):
    """G_callable(min_snp) over all 6 pairs of every strict quartet."""
    pm = [min(gs_map.get(a, 0), gs_map.get(b, 0))
          for _, gs in quartets for a, b in combinations(gs, 2)]
    return pm, {ms: int(sum(1 for m in pm if m >= ms)) for ms in MIN_SNP_GRID}


pair_minsnp, G_by_ms = callable_curve(gene_snp)
_, G_by_ms_flank = callable_curve(gene_snp_flank)   # +-1 kb sensitivity

covered = sum(1 for g in quartet_genes if gene_snp.get(g, 0) >= 1)
n_pairs = len(pair_minsnp)
G_ref = G_by_ms[REF_MIN_SNP]
snp_counts = [gene_snp.get(g, 0) for g in quartet_genes]

result = {
    "tool": "strawberry_h7_boundary",
    "species": "strawberry octoploid (Fragaria x ananassa, Camarosa)",
    "ploidy": "8n",
    "tech": "SNP-array (Axiom eQTL panel, 196 individuals)",
    "scope": "all 6 homoeolog pairs per strict 4/4 quartet, genome-wide",
    "framework_note": "bands + min_snp>=3 reference taken AS-IS from h7_marker_density.py; "
                      "not recalibrated on strawberry (no circularity)",
    "marker_assignment": "gene-span containment only (nearest-gene NOT used)",
    "n_assoc_rows": n_rows,
    "n_unique_markers": n_uniq,
    "n_markers_in_gene": n_assigned,
    "frac_markers_in_gene": round(n_assigned / n_uniq, 4),
    "n_strict_quartets": len(quartets),
    "n_quartet_genes": len(quartet_genes),
    "quartet_genes_with_ge1_marker": covered,
    "frac_quartet_genes_covered": round(covered / max(1, len(quartet_genes)), 4),
    "mean_in_gene_snps_per_quartet_gene": round(sum(snp_counts) / max(1, len(snp_counts)), 4),
    "n_homoeolog_pairs": n_pairs,
    "pair_min_snp_distribution": {
        "max": max(pair_minsnp) if pair_minsnp else 0,
        "ge1": int(sum(1 for m in pair_minsnp if m >= 1)),
        "ge2": int(sum(1 for m in pair_minsnp if m >= 2)),
        "ge3": int(sum(1 for m in pair_minsnp if m >= 3)),
    },
    "G_callable_by_min_snp": G_by_ms,
    "G_callable_by_min_snp_flank1kb": G_by_ms_flank,
    "n_markers_within_1kb": n_assigned_flank,
    "G_callable_ref": G_ref,
    "G_callable_ref_flank1kb": G_by_ms_flank[REF_MIN_SNP],
    "ref_min_snp": REF_MIN_SNP,
    "predicted_band": classify(G_ref),
    "predicted_band_flank1kb": classify(G_by_ms_flank[REF_MIN_SNP]),
    "anchors_for_context": {"wheat_6n_WGS": 1759, "cotton_4n_array": 372,
                            "oat_6n_GBS": 69, "rapeseed_4n_array": 14},
    "interpretation": None,  # filled below
}
band = result["predicted_band"]
result["interpretation"] = (
    f"At array density ({n_uniq} sites) on the octoploid genome, "
    f"G_callable={G_ref} homoeolog pairs (min_snp>=3) -> '{band}'. "
    "8n framework reaches this band empirically under array density; this is a "
    "feasibility boundary, NOT a discovery claim (no dense WGS + phenotype here)."
)
(IN / "h7_strawberry.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
print(f"[done] -> {IN}/h7_strawberry.json")
