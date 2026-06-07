#!/usr/bin/env python
"""Stage 3 reference-bias QC (runnable BEFORE real data).

Simulate EQUAL per-base coverage from the 4 copy transcripts of every strict
quartet (so the TRUE per-copy expression is identical = 0.25 each), quantify the
pooled reads with the SAME salmon index used for real data, and measure how far
each quartet's ESTIMATED proportions deviate from 0.25. A large deviation under a
known-balanced input is pure reference/mappability bias (one copy is more
"Camarosa-mappable" than its homoeologs). Such quartets are flagged so they are
demoted in the real population-HEB analysis (a biased quartet would otherwise
masquerade as real HEB).

Equal coverage C => reads ∝ transcript length => equal TPM truth. Paired-end,
read length 100, modest substitution error. Self-contained: writes FASTQ, calls
salmon, parses quant.sf. No panel data needed.
"""
import csv
import json
import subprocess
from pathlib import Path

PROJ = Path("/mnt/7302share/fast_ysp/U7_GWAS")
IN = PROJ / "results/phase7/bio_strawberry"
TX = Path("/mnt/nvme/strawberry_quant/transcripts.fa")
IDX = Path("/mnt/nvme/strawberry_quant/salmon_index")
WORK = Path("/mnt/nvme/strawberry_quant/insilico")
SALMON = Path.home() / ".local/share/mamba/envs/straw_quant/bin/salmon"
COV = 15            # per-base coverage per copy (equal => balanced truth)
RLEN = 100
FRAG = 250
ERR = 0.002        # per-base substitution error
SEED = 12345       # fixed (Math.random unavailable; reproducible LCG)
BIAS_THRESH = 0.10  # |max prop - 0.25| above this under equal truth = biased

WORK.mkdir(parents=True, exist_ok=True)
COMP = str.maketrans("ACGT", "TGCA")


def load_longest_tx(fa):
    best = {}
    cur, buf = None, []
    with open(fa) as fh:
        for line in fh:
            if line.startswith(">"):
                if cur is not None:
                    g = cur.rsplit(".t", 1)[0] if ".t" in cur else cur
                    s = "".join(buf).upper()
                    if g not in best or len(s) > len(best[g]):
                        best[g] = (cur, s)
                cur = line[1:].split()[0]
                buf = []
            else:
                buf.append(line.strip())
        if cur is not None:
            g = cur.rsplit(".t", 1)[0] if ".t" in cur else cur
            s = "".join(buf).upper()
            if g not in best or len(s) > len(best[g]):
                best[g] = (cur, s)
    return best  # gene -> (tx_id, seq)


# deterministic LCG (numpy/random fine, but keep reproducible & dependency-light)
_state = SEED


def rnd():
    global _state
    _state = (_state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
    return (_state >> 11) / (1 << 53)


def rc(s):
    return s.translate(COMP)[::-1]


def mutate(s):
    if ERR <= 0:
        return s
    out = list(s)
    for i in range(len(out)):
        if rnd() < ERR:
            out[i] = "ACGT"[int(rnd() * 4) % 4]
    return "".join(out)


tx = load_longest_tx(TX)
quartets = []
with open(IN / "quartets.tsv") as fh:
    for r in csv.DictReader(fh, delimiter="\t"):
        gs = [r["A"], r["B"], r["C"], r["D"]]
        if all(g != "." for g in gs):
            quartets.append((r["group"], gs))
print(f"[sim] {len(quartets)} quartets; equal coverage C={COV}")

# tx_id (as in index, e.g. FxaC..t1) -> gene, and quartet membership
txid2gene = {}
quartet_txids = {}
r1p = WORK / "sim_1.fastq"
r2p = WORK / "sim_2.fastq"
n_pairs = 0
with open(r1p, "w") as f1, open(r2p, "w") as f2:
    for qi, (_grp, gs) in enumerate(quartets):
        ids = []
        for g in gs:
            rec = tx.get(g)
            if rec is None:
                ids.append(None)
                continue
            txid, seq = rec
            ids.append(txid)
            txid2gene[txid] = g
            L = len(seq)
            if L < FRAG:
                continue
            npair = max(1, int(COV * L / (2 * RLEN)))
            for _ in range(npair):
                start = int(rnd() * (L - FRAG + 1))
                frag = seq[start:start + FRAG]
                read1 = mutate(frag[:RLEN])
                read2 = mutate(rc(frag[-RLEN:]))
                n_pairs += 1
                rid = f"q{qi}_{txid}_{n_pairs}"
                f1.write(f"@{rid}/1\n{read1}\n+\n{'I' * RLEN}\n")
                f2.write(f"@{rid}/2\n{read2}\n+\n{'I' * RLEN}\n")
        quartet_txids[qi] = ids
print(f"[sim] wrote {n_pairs} read pairs -> {r1p.name}/{r2p.name}")

# quantify with the SAME index
out = WORK / "salmon_out"
print("[sim] running salmon quant ...", flush=True)
subprocess.run([str(SALMON), "quant", "-i", str(IDX), "-l", "A",
                "-1", str(r1p), "-2", str(r2p), "-p", "32",
                "--validateMappings", "-o", str(out)], check=True,
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# parse quant.sf: txid -> TPM. TPM (length-normalised) is the right scale here:
# equal per-base coverage = equal TPM truth, so balanced TPM => 0.25 per copy.
# (NumReads would scale with effective length and mis-flag length differences as
# reference bias -- the homoeolog proportion is an expression-level share.)
numreads = {}
with open(out / "quant.sf") as fh:
    next(fh)
    for line in fh:
        c = line.rstrip("\n").split("\t")
        numreads[c[0]] = float(c[3])

# per-quartet estimated proportions vs balanced truth (0.25)
rows = []
biased = 0
evaluable = 0
for qi, (grp, _gs) in enumerate(quartets):
    ids = quartet_txids[qi]
    if any(i is None for i in ids):
        continue
    vals = [numreads.get(i, 0.0) for i in ids]   # TPM per copy
    tot = sum(vals)
    if tot < 1.0:  # essentially unquantified at this coverage
        continue
    evaluable += 1
    props = [v / tot for v in vals]
    maxdev = max(abs(p - 0.25) for p in props)
    is_biased = maxdev > BIAS_THRESH
    biased += is_biased
    rows.append([qi, grp, *[f"{p:.3f}" for p in props], f"{maxdev:.3f}",
                 "biased" if is_biased else "fair"])

with open(IN / "insilico_fairness.tsv", "w") as o:
    w = csv.writer(o, delimiter="\t")
    w.writerow(["quartet_idx", "group", "pA", "pB", "pC", "pD", "max_dev_from_0.25", "verdict"])
    w.writerows(rows)

summary = {
    "tool": "strawberry_insilico_fairness",
    "purpose": "reference-mapping bias QC under known-balanced (equal-coverage) input",
    "params": {"coverage": COV, "read_len": RLEN, "frag": FRAG, "err": ERR,
               "bias_threshold_maxdev": BIAS_THRESH},
    "n_read_pairs": n_pairs,
    "n_quartets_evaluable": evaluable,
    "n_biased": biased,
    "frac_biased": round(biased / max(1, evaluable), 4),
    "interpretation": "quartets flagged 'biased' deviate from 0.25 despite equal "
                      "true expression -> reference/mappability artifact; demote "
                      "them in real HEB (keep 'fair' for strong claims).",
}
(IN / "insilico_fairness.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
print(f"[done] -> {IN}/insilico_fairness.{{tsv,json}}")
