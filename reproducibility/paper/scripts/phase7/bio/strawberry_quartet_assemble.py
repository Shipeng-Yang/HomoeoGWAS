#!/usr/bin/env python
"""Strawberry 8n quartet assembly (Step 1c).

Combine DIAMOND self-blast (best-per-subgenome with paralog margin) with MCScanX
synteny anchors to build high-confidence homoeolog quartets across the 4 Camarosa
subgenomes. A quartet edge is asserted ONLY when BOTH:
  (i)  reciprocal best hit per subgenome pair (a's best on subgenome Y is b AND
       b's best on subgenome X is a), each passing a paralog margin (best bitscore
       >= MARGIN x second-best on that subgenome), AND
  (ii) the pair is an MCScanX syntenic anchor (cross-subgenome, same group).
Union (blast-only OR synteny-only) is reported separately as a sensitivity tier.

Edges are restricted to the same homoeology group (digit of the MCScanX tag) and
different subgenome (letter). Position is used ONLY to confine candidates to the
same group; pairing itself is homology+synteny, never positional adjacency.

Connected components per group are classified:
  strict 4/4  : 4 genes, one per subgenome {A,B,C,D}
  partial 3/4 : 3 genes, 3 distinct subgenomes
  pair-only   : 2 genes, 2 subgenomes
  ambiguous   : a component with >1 gene from the same subgenome (paralog/tandem)
Main H7 analysis uses strict 4/4; partial reported as supplementary sensitivity.
"""
import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

PROJ = Path("/mnt/7302share/fast_ysp/U7_GWAS")
IN = Path(sys.argv[1]) if len(sys.argv) > 1 else PROJ / "results/phase7/bio_strawberry"
MARGIN = 1.10  # best bitscore must exceed MARGIN x second-best on a subgenome

# ---- gene -> (group, subletter) from the MCScanX gff (tag = <letter><group}) ---
meta = {}
for line in (IN / "camarosa.gff").read_text().splitlines():
    tag, gid, _s, _e = line.split("\t")
    meta[gid] = (int(tag[1:]), tag[0])  # (group, subgenome letter)
print(f"[meta] {len(meta)} genes")

# ---- DIAMOND blast: best + 2nd-best hit per (query, target-subgenome) ----------
# best[q][Y] = (gene, bitscore); second[q][Y] = bitscore
best = defaultdict(dict)
second = defaultdict(dict)
with open(IN / "union.blast") as fh:
    for line in fh:
        f = line.rstrip("\n").split("\t")
        q, s = f[0], f[1]
        if q == s or q not in meta or s not in meta:
            continue
        gq, xq = meta[q]
        gs, xs = meta[s]
        if gq != gs or xq == xs:  # same group, different subgenome only
            continue
        bit = float(f[11])
        cur = best[q].get(xs)
        if cur is None or bit > cur[1]:
            if cur is not None:
                second[q][xs] = max(second[q].get(xs, 0.0), cur[1])
            best[q][xs] = (s, bit)
        else:
            second[q][xs] = max(second[q].get(xs, 0.0), bit)


def margin_ok(q, y):
    b = best[q][y][1]
    sec = second[q].get(y, 0.0)
    return sec == 0.0 or b >= MARGIN * sec


# ---- MCScanX synteny anchors (cross-subgenome, same group) ---------------------
syn_pairs = set()
n_aln = n_aln_inter = 0
with open(IN / "union.collinearity") as fh:
    keep = False
    for line in fh:
        if line.startswith("## Alignment"):
            n_aln += 1
            # header tail: "...  AX&BY plus" -> recover the two tags
            toks = line.strip().split()
            pair = next((t for t in toks if "&" in t), "")
            t1, _, t2 = pair.partition("&")
            keep = (len(t1) >= 2 and len(t2) >= 2 and t1[1:] == t2[1:] and t1[0] != t2[0])
            if keep:
                n_aln_inter += 1
            continue
        if line.startswith("#") or not keep:
            continue
        parts = line.split("\t")
        if len(parts) >= 3:
            a, b = parts[1].strip(), parts[2].strip()
            if a in meta and b in meta:
                syn_pairs.add(frozenset((a, b)))
print(f"[synteny] {n_aln} alignments, {n_aln_inter} cross-subgenome same-group, "
      f"{len(syn_pairs)} anchor pairs")

# ---- build edges ---------------------------------------------------------------
# reciprocal best per subgenome pair, both margins ok
rbh_edges = set()
for q in best:
    gq, xq = meta[q]
    for y, (t, _bit) in best[q].items():
        # reciprocal: t's best back on subgenome xq is q
        if best.get(t, {}).get(xq, (None,))[0] != q:
            continue
        if not (margin_ok(q, y) and margin_ok(t, xq)):
            continue
        rbh_edges.add(frozenset((q, t)))

hc_edges = rbh_edges & syn_pairs          # primary: blast RBH INTERSECT synteny
union_edges = rbh_edges | syn_pairs       # sensitivity: union
print(f"[edges] RBH={len(rbh_edges)} synteny={len(syn_pairs)} "
      f"intersect(HC)={len(hc_edges)} union={len(union_edges)}")


def components_by_group(edges):
    """union-find over edges, keyed within group; returns list of node-sets."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(a)] = find(b)

    for e in edges:
        a, b = tuple(e)
        union(a, b)
    comps = defaultdict(set)
    for n in parent:
        comps[find(n)].add(n)
    return list(comps.values())


def classify(comps, edges):
    edge_set = set(edges)
    out = {"strict4": [], "partial3": [], "pair": [], "ambiguous": []}
    for comp in comps:
        subs = [meta[g][1] for g in comp]
        if len(set(subs)) != len(subs):
            out["ambiguous"].append(sorted(comp))
            continue
        nsub = len(set(subs))
        n_e = sum(1 for a, b in combinations(comp, 2) if frozenset((a, b)) in edge_set)
        rec = {"group": meta[next(iter(comp))][0],
               "genes": {meta[g][1]: g for g in comp},
               "n_edges": n_e, "completeness": round(n_e / (nsub * (nsub - 1) / 2), 3)}
        if nsub == 4:
            out["strict4"].append(rec)
        elif nsub == 3:
            out["partial3"].append(rec)
        elif nsub == 2:
            out["pair"].append(rec)
    return out


hc = classify(components_by_group(hc_edges), hc_edges)
un = classify(components_by_group(union_edges), union_edges)

# per-group strict quartet counts + edge completeness
by_group = defaultdict(int)
for r in hc["strict4"]:
    by_group[r["group"]] += 1
mean_compl = (sum(r["completeness"] for r in hc["strict4"]) / len(hc["strict4"])
              if hc["strict4"] else 0.0)
# edge-completeness histogram (n_edges out of 6) for strict 4/4 quartets
edge_hist = defaultdict(int)
for r in hc["strict4"]:
    edge_hist[r["n_edges"]] += 1
frac_full_clique = (sum(1 for r in hc["strict4"] if r["n_edges"] == 6)
                    / len(hc["strict4"]) if hc["strict4"] else 0.0)

summary = {
    "tool": "strawberry_quartet_assemble",
    "margin": MARGIN,
    "n_genes_placed": len(meta),
    "blast_rbh_edges": len(rbh_edges),
    "synteny_anchor_pairs": len(syn_pairs),
    "hc_edges_intersect": len(hc_edges),
    "union_edges": len(union_edges),
    "primary_intersect": {
        "strict4_quartets": len(hc["strict4"]),
        "partial3": len(hc["partial3"]),
        "pair_only": len(hc["pair"]),
        "ambiguous_components": len(hc["ambiguous"]),
        "strict4_per_group": dict(sorted(by_group.items())),
        "strict4_mean_edge_completeness": round(mean_compl, 3),
        "strict4_edge_count_histogram_of6": dict(sorted(edge_hist.items())),
        "strict4_frac_full_6edge_clique": round(frac_full_clique, 3),
    },
    "sensitivity_union": {
        "strict4_quartets": len(un["strict4"]),
        "partial3": len(un["partial3"]),
        "pair_only": len(un["pair"]),
        "ambiguous_components": len(un["ambiguous"]),
    },
    "definition": "edge = reciprocal-best-per-subgenome (paralog margin >=1.10x "
                  "2nd-best) INTERSECT MCScanX cross-subgenome synteny anchor; "
                  "quartet = connected component with 1 gene per subgenome.",
}
(IN / "quartet_summary.json").write_text(json.dumps(summary, indent=2))

# strict quartet table (primary)
with open(IN / "quartets.tsv", "w") as out:
    out.write("group\tA\tB\tC\tD\tn_edges\tcompleteness\n")
    for r in sorted(hc["strict4"], key=lambda x: (x["group"], -x["completeness"])):
        g = r["genes"]
        out.write(f"{r['group']}\t{g.get('A', '.')}\t{g.get('B', '.')}\t"
                  f"{g.get('C', '.')}\t{g.get('D', '.')}\t{r['n_edges']}\t"
                  f"{r['completeness']}\n")

print(json.dumps(summary, indent=2))
print(f"[done] -> {IN}/quartets.tsv  +  quartet_summary.json")
