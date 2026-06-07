#!/usr/bin/env python
"""Strawberry 8n quartet prep (Step 1a).

Camarosa octoploid genome: 28 chromosomes Fvb{g}-{s}, g=1..7 homoeology group,
s=1..4 subgenome. Builds two inputs for the homolog-quartet pipeline:

  1. rep_pep.fa   : one representative (longest) protein isoform per gene,
                    header = bare gene id (isoform .tN stripped) so DIAMOND
                    blast ids match the MCScanX gff gene ids.
  2. camarosa.gff : MCScanX 4-column gff (tag, gene, start, end). tag encodes
                    subgenome+group as <A|B|C|D><group>, e.g. Fvb3-2 -> 'B3'.
                    First char = subgenome (consistent across groups per
                    Edger2019 Camarosa assignment), digit = homoeology group.
                    A within-group cross-subgenome block then connects e.g.
                    B3 & A3 (same digit, different letter).

No homology is asserted here; this only normalises ids/coords. Quartets are
built downstream from DIAMOND best-per-subgenome INTERSECT MCScanX synteny.
"""
import gzip
import re
import sys
from pathlib import Path

PROJ = Path("/mnt/7302share/fast_ysp/U7_GWAS")
GFF = PROJ / "data/raw/strawberry/camarosa/gff.gz"
PEP = PROJ / "data/raw/strawberry/camarosa/proteins.fa.gz"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else PROJ / "results/phase7/bio_strawberry"
OUT.mkdir(parents=True, exist_ok=True)

SUB_LETTER = {1: "A", 2: "B", 3: "C", 4: "D"}
CHR_RE = re.compile(r"^Fvb(\d+)-(\d+)$")


def parse_attr(field, key):
    m = re.search(rf"{key}=([^;]+)", field)
    return m.group(1) if m else None


# ---- 1. gene coords + tag from GFF gene lines ------------------------------
genes = {}  # gene_id -> (tag, start, end)
with gzip.open(GFF, "rt") as fh:
    for line in fh:
        if line.startswith("#") or not line.strip():
            continue
        c = line.rstrip("\n").split("\t")
        if len(c) < 9 or c[2] != "gene":
            continue
        m = CHR_RE.match(c[0])
        if not m:
            continue
        group, sub = int(m.group(1)), int(m.group(2))
        gid = parse_attr(c[8], "ID")
        if gid is None:
            continue
        tag = f"{SUB_LETTER[sub]}{group}"
        genes[gid] = (tag, int(c[3]), int(c[4]))

with open(OUT / "camarosa.gff", "w") as out:
    for gid, (tag, s, e) in genes.items():
        out.write(f"{tag}\t{gid}\t{s}\t{e}\n")
print(f"[gff] {len(genes)} genes on 28 chromosomes -> camarosa.gff")

# ---- 2. longest isoform per gene from PEP ----------------------------------
seqs = {}  # header_id(with .tN) -> seq
cur, buf = None, []
with gzip.open(PEP, "rt") as fh:
    for line in fh:
        if line.startswith(">"):
            if cur is not None:
                seqs[cur] = "".join(buf)
            cur = line[1:].strip().split()[0]
            buf = []
        else:
            buf.append(line.strip())
    if cur is not None:
        seqs[cur] = "".join(buf)

best = {}  # gene_id -> (len, isoform_id, seq)
for iso, seq in seqs.items():
    gid = iso.rsplit(".t", 1)[0]
    seq = seq.replace(".", "").replace("*", "")  # Camarosa uses '.' as stop marker
    L = len(seq)
    if gid not in best or L > best[gid][0]:
        best[gid] = (L, seq)

# only keep genes that are also in the gff (placed on the 28 chromosomes)
n_written = 0
with open(OUT / "rep_pep.fa", "w") as out:
    for gid, (_L, seq) in best.items():
        if gid not in genes:
            continue
        out.write(f">{gid}\n")
        for i in range(0, len(seq), 60):
            out.write(seq[i:i + 60] + "\n")
        n_written += 1
print(f"[pep] {len(seqs)} isoforms -> {len(best)} genes -> {n_written} placed reps -> rep_pep.fa")
print(f"[done] outputs in {OUT}")
