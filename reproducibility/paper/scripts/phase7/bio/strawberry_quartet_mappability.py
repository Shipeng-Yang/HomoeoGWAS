#!/usr/bin/env python
"""Stage 0: per-quartet copy-specific k-mer identifiability.

For each strict 4/4 quartet, take the representative (longest) transcript of each
of the 4 Camarosa copies and compute, per copy, the fraction of its k-mers (k=31,
the salmon index k) that are UNIQUE to that copy (absent from the other 3 copies
of the same quartet). This is the homoeolog identifiability: high fraction => a
copy carries copy-distinguishing sequence that reads can be uniquely assigned to;
low fraction => the 4 copies are near-identical and per-copy expression is not
separable. Quartets are tiered high/medium/low and this gate is carried into all
downstream HEB conclusions (strong claims only on high-identifiability quartets).

Self-contained from transcripts.fa (gffread output) + quartets.tsv; no reads.
"""
import csv
import sys
from pathlib import Path

PROJ = Path("/mnt/7302share/fast_ysp/U7_GWAS")
TX = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/mnt/nvme/strawberry_quant/transcripts.fa")
IN = PROJ / "results/phase7/bio_strawberry"
K = 31
HIGH, MED = 0.30, 0.10   # mean copy-specific k-mer fraction tiers


def load_longest_tx(fa):
    """gene_id (strip .tN) -> longest transcript sequence."""
    best = {}
    cur, buf = None, []
    with open(fa) as fh:
        for line in fh:
            if line.startswith(">"):
                if cur is not None:
                    g = cur.rsplit(".t", 1)[0] if ".t" in cur else cur
                    s = "".join(buf).upper()
                    if g not in best or len(s) > len(best[g]):
                        best[g] = s
                cur = line[1:].split()[0]
                buf = []
            else:
                buf.append(line.strip())
        if cur is not None:
            g = cur.rsplit(".t", 1)[0] if ".t" in cur else cur
            s = "".join(buf).upper()
            if g not in best or len(s) > len(best[g]):
                best[g] = s
    return best


def kmers(seq):
    return {seq[i:i + K] for i in range(len(seq) - K + 1)} if len(seq) >= K else set()


tx = load_longest_tx(TX)
print(f"[tx] {len(tx)} genes with a transcript")

quartets = []
with open(IN / "quartets.tsv") as fh:
    for r in csv.DictReader(fh, delimiter="\t"):
        gs = [r["A"], r["B"], r["C"], r["D"]]
        if all(g != "." for g in gs):
            quartets.append((r["group"], gs))
print(f"[quartets] {len(quartets)} strict 4/4")

rows = []
tiers = {"high": 0, "medium": 0, "low": 0, "untranscribed": 0}
for grp, gs in quartets:
    km = [kmers(tx.get(g, "")) for g in gs]
    if any(len(k) == 0 for k in km):
        tiers["untranscribed"] += 1
        rows.append([grp, *gs, "", "", "", "", "", "untranscribed"])
        continue
    fr = []
    for i in range(4):
        others = km[0] | km[1] | km[2] | km[3]
        others = set().union(*[km[j] for j in range(4) if j != i])
        uniq = km[i] - others
        fr.append(len(uniq) / len(km[i]))
    mean_fr = sum(fr) / 4
    min_fr = min(fr)
    tier = "high" if mean_fr >= HIGH else "medium" if mean_fr >= MED else "low"
    tiers[tier] += 1
    rows.append([grp, *gs, *[f"{x:.3f}" for x in fr], f"{mean_fr:.3f}", f"{min_fr:.3f}", tier])

with open(IN / "quartet_mappability.tsv", "w") as out:
    w = csv.writer(out, delimiter="\t")
    w.writerow(["group", "A", "B", "C", "D", "csf_A", "csf_B", "csf_C", "csf_D",
                "mean_csf", "min_csf", "tier"])
    # note: untranscribed rows have fewer csf cols; pad
    for r in rows:
        if r[-1] == "untranscribed":
            w.writerow(r[:5] + ["", "", "", "", "", "", "untranscribed"])
        else:
            w.writerow(r)

print(f"[tiers] {tiers}")
print(f"[done] -> {IN}/quartet_mappability.tsv  (k={K}, high>={HIGH}, med>={MED})")
