#!/usr/bin/env python3
"""OAT Step 3: build de-novo 1:1:1 A-C-D homoeolog triads from MCScanX collinearity.

Design:
  Tier1 (PRIMARY) strict 1:1:1: gene a has a UNIQUE reciprocal C partner (A-C blocks),
    a UNIQUE reciprocal D partner (A-D blocks), and that (c,d) pair is itself a UNIQUE
    reciprocal C-D collinear pair. Each gene used in at most one triad. Ambiguous /
    multi-partner / tandem-duplicated genes are dropped (not forced).
  Tier2 (SENSITIVITY): any-two-of-three consistent, third missing but no conflict.
Writes triads_strict.tsv, triads_tier2.tsv, triad_attrition.json.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path("/mnt/7302share/fast_ysp/U7_GWAS")
COLL = Path("/mnt/nvme/oat_raw/mcscanx/oat.collinearity")
OUT = ROOT / "results/phase7/bio_oat"
CODE2SUB = {"aa": "A", "cc": "C", "dd": "D"}
_HDR = re.compile(r"^## Alignment \d+:.* N=\d+ (\w\w)\d+&(\w\w)\d+ ")


def _parse_inter_sub_pairs():
    """Return dict {('A','C'): {(a_gene,c_gene): best_evalue}, ...} for inter-sub pairs."""
    pairs = {("A", "C"): {}, ("A", "D"): {}, ("C", "D"): {}}
    cur = None  # (sub1, sub2) or None for intra/skip
    with open(COLL) as f:
        for line in f:
            if line.startswith("## Alignment"):
                m = _HDR.match(line)
                cur = None
                if m:
                    s1 = CODE2SUB.get(m.group(1)); s2 = CODE2SUB.get(m.group(2))
                    if s1 and s2 and s1 != s2:
                        cur = (s1, s2)
                continue
            if cur is None or line.startswith("#"):
                continue
            # anchor line: "  N-  M:\tG1\tG2\t evalue"
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            g1 = parts[1].strip(); g2 = parts[2].strip()
            try:
                ev = float(parts[3].strip()) if len(parts) > 3 and parts[3].strip() else 0.0
            except ValueError:
                ev = 0.0
            s1, s2 = cur
            key = (s1, s2) if (s1, s2) in pairs else (s2, s1)
            if (s1, s2) in pairs:
                a, b = g1, g2
            else:
                a, b = g2, g1
            d = pairs[key]
            if (a, b) not in d or ev < d[(a, b)]:
                d[(a, b)] = ev
    return pairs


def _reciprocal_unique(pair_dict):
    """From {(x,y): ev}, keep only x<->y that are mutually unique (x maps to exactly one
    y and vice versa). Returns dict x->y (and implicit y->x)."""
    x2y = defaultdict(set); y2x = defaultdict(set)
    for (x, y) in pair_dict:
        x2y[x].add(y); y2x[y].add(x)
    out = {}
    for x, ys in x2y.items():
        if len(ys) == 1:
            y = next(iter(ys))
            if len(y2x[y]) == 1:
                out[x] = y
    return out


def main():
    pairs = _parse_inter_sub_pairs()
    n_anchor = {k: len(v) for k, v in pairs.items()}
    AC = _reciprocal_unique(pairs[("A", "C")])  # a -> c
    AD = _reciprocal_unique(pairs[("A", "D")])  # a -> d
    CD = _reciprocal_unique(pairs[("C", "D")])  # c -> d
    print(f"  anchor pairs: AC={n_anchor[('A','C')]} AD={n_anchor[('A','D')]} CD={n_anchor[('C','D')]}")
    print(f"  reciprocal-unique: AC={len(AC)} AD={len(AD)} CD={len(CD)}")

    # strict 1:1:1: a has unique c (AC) and unique d (AD), and CD[c]==d
    strict = []
    used = {"A": set(), "C": set(), "D": set()}
    for a, c in AC.items():
        d = AD.get(a)
        if d is None:
            continue
        if CD.get(c) == d:
            if a in used["A"] or c in used["C"] or d in used["D"]:
                continue
            strict.append((a, c, d))
            used["A"].add(a); used["C"].add(c); used["D"].add(d)

    # tier2: any two of three consistent, third missing (no conflict), genes not in strict
    tier2 = []
    s_used = {s: set(used[s]) for s in ("A", "C", "D")}
    for a, c in AC.items():
        if a in s_used["A"] or c in s_used["C"]:
            continue
        d_ad = AD.get(a); d_cd = CD.get(c)
        # consistent if at most one of d_ad/d_cd defined, or they agree
        if d_ad and d_cd and d_ad != d_cd:
            continue  # conflict
        d = d_ad or d_cd
        if d and d in s_used["D"]:
            continue
        # require it's NOT already a full strict triad (third present & consistent -> would be strict)
        if d_ad and d_cd and d_ad == d_cd:
            continue  # this is strict-eligible, handled above
        tier2.append((a, c, d if d else ""))

    with (OUT / "triads_strict.tsv").open("w") as f:
        f.write("A\tC\tD\n")
        for a, c, d in strict:
            f.write(f"{a}\t{c}\t{d}\n")

    # PRIMARY UNIT (sparse-GBS pivot): reciprocal-unique homoeolog PAIRS per subgenome-pair.
    # Strict 3-way-retained triads are ~0 under GBS sparsity (reported in attrition);
    # pairwise testing needs only 2 of 3 homoeologs retained -> many more callable units,
    # each whitened by the full hexaploid LMM {K_A,K_C,K_D}. This is the cotton AADD
    # homoeolog-pair framework applied to the 3 subgenome-pairs of a hexaploid.
    with (OUT / "homoeolog_pairs.tsv").open("w") as f:
        f.write("sub_pair\tgene_x\tgene_y\n")
        for tag, d in (("AC", AC), ("AD", AD), ("CD", CD)):
            for x, y in d.items():
                f.write(f"{tag}\t{x}\t{y}\n")

    attr = dict(anchor_pairs={f"{a}-{b}": v for (a, b), v in n_anchor.items()},
                reciprocal_unique=dict(AC=len(AC), AD=len(AD), CD=len(CD)),
                strict_triads_all=len(strict),
                note=("strict 3-way-retained triads ~0 under sparse GBS -> PRIMARY unit is "
                      "reciprocal-unique homoeolog PAIRS (AC/AD/CD), each needs only 2 of 3 "
                      "homoeologs retained; whitened by full hexaploid LMM {K_A,K_C,K_D}."))
    (OUT / "triad_attrition.json").write_text(json.dumps(attr, indent=2))
    print(f"  STRICT 1:1:1 triads = {len(strict)} (reported only; ~0 callable under GBS)")
    print(f"  homoeolog PAIRS (PRIMARY): AC={len(AC)} AD={len(AD)} CD={len(CD)}")
    print(f"  wrote triads_strict.tsv / homoeolog_pairs.tsv / triad_attrition.json")


if __name__ == "__main__":
    main()
