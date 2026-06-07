#!/usr/bin/env python3
"""Wheat DEFENSE: WGS burden-cap sensitivity for the two curated-triad hits.

The cap=150 SNP/gene de-dilution is one tunable. Codex: show the hits are not a single-
parameter accident AND that global calibration is not broken by the cap. Extract SNPs +
GRM + whitening ONCE (all cap-independent); only the burden capping changes -> sweep
cap in {50,150,300,100000(~uncapped)} and report, per cap: the chr1 B-D and chr5 A-D hit
interaction p, the carrier variant counts, and the global lambda_obs per pairwise (drift
check). INT primary, days_to_emergence.
"""
from __future__ import annotations

import bisect
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/mnt/7302share/fast_ysp/U7_GWAS")
sys.path.insert(0, str(ROOT / "scripts/phase7"))
from d3_perpair_interaction import _lambda_gc  # noqa: E402

_s4 = importlib.util.spec_from_file_location(
    "w4", str(ROOT / "scripts/phase7/bio_wheat/04w_wheat_genomewide.py"))
_w4 = importlib.util.module_from_spec(_s4); _s4.loader.exec_module(_w4)
_w2 = _w4._w2

OUT = ROOT / "results/phase7/bio_wheat"
CHROMS = _w4.CHROMS
SUB_BED = _w4.SUB_BED
WORK = _w4.WORK
PHENO = _w4.PHENO
HCTRIADS = _w4.HCTRIADS
PLINK2 = _w4.PLINK2

CAPS = [50, 150, 300, 100000]
TRAIT = "days_to_emerg"
MODE = "flank"
MIN_SNP = 3
# the two hits (v1.1 ids): (pairwise tag, A, B, D-relevant genes)
HIT_CHR1_BD = ("BD", "TraesCS1B02G143800", "TraesCS1D02G128400")  # B,D
HIT_CHR5_AD = ("AD", "TraesCS5A02G169500", "TraesCS5D02G173900")  # A,D


def _extract_raw(sub, genes, mode):
    """Like 04w but return (X, retained gene2snp, GRM, samples) — cap-independent."""
    WORK.mkdir(parents=True, exist_ok=True)
    rf = WORK / f"caps_range_{sub}_{mode}.txt"
    with open(rf, "w") as f:
        for gid, gff_chrom, gs, ge, fs, fe in genes:
            ws, we = (gs, ge) if mode == "body" else (fs, fe)
            f.write(f"chr{gff_chrom}\t{ws}\t{we}\t{gid}\n")
    outp = WORK / f"caps_{sub}_{mode}"
    cmd = [str(PLINK2), "--pfile", str(SUB_BED[sub]), "--extract", "range", str(rf),
           "--make-bed", "--out", str(outp), "--threads", "32", "--memory", "200000"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"plink2 {sub}: {r.stderr[-600:]}")
    from homoeogwas.io import load_bed_hardcall
    bed = load_bed_hardcall(outp)
    X = np.asarray(bed.dosage, dtype=np.float64)
    samples = list(np.asarray(bed.samples))
    pos = []; bchrom = []
    with open(f"{outp}.bim") as f:
        for line in f:
            pp = line.rstrip("\n").split("\t")
            bchrom.append(pp[0]); pos.append(int(pp[3]))
    pos = np.array(pos); bchrom = np.array(bchrom)
    gene2snp = {}
    for gff_chrom in CHROMS[sub]:
        sel = np.where(bchrom == f"chr{gff_chrom}")[0]
        if sel.size == 0:
            continue
        spos = pos[sel]; order = np.argsort(spos); spos_s = spos[order]; sel_s = sel[order]
        gc = [(g[0], (g[2] if mode == "body" else g[4]), (g[3] if mode == "body" else g[5]))
              for g in genes if g[1] == gff_chrom]
        gc.sort(key=lambda x: x[1]); starts = [g[1] for g in gc]
        for j, ps in enumerate(spos_s):
            k = bisect.bisect_right(starts, ps) - 1; steps = 0
            while k >= 0 and steps < 3000:
                gid, ws, we = gc[k]
                if ps > we and ps - ws > 2_000_000:
                    break
                if ws <= ps <= we:
                    gene2snp.setdefault(gid, []).append(int(sel_s[j]))
                k -= 1; steps += 1
    retained = {g: idx for g, idx in gene2snp.items() if len(idx) >= MIN_SNP}
    K = _w2._grm_from_X(X)
    return X, retained, K, samples


def main():
    t0 = time.time()
    rng = np.random.default_rng(7)
    print(f"=== wheat cap sensitivity (mode={MODE}, trait={TRAIT}, caps={CAPS}) ===")
    Xs = {}; retained = {}; grms = {}; samples_ref = None
    for s in ("A", "B", "D"):
        genes = _w4._parse_gff(s, 2000)
        X, ret, K, samples = _extract_raw(s, genes, MODE)
        Xs[s] = X; retained[s] = ret; grms[s] = K
        samples_ref = samples_ref or samples
        assert samples == samples_ref
        print(f"  {s}: {len(ret)} retained genes, X={X.shape} ({time.time()-t0:.0f}s)")
    common = samples_ref

    hc = pd.read_csv(HCTRIADS); hc = hc[hc["cardinality_abs"] == "1:1:1"]
    kept = []
    for _, r in hc.iterrows():
        a = _w4._v10_to_v11(str(r["A"])); b = _w4._v10_to_v11(str(r["B"])); d = _w4._v10_to_v11(str(r["D"]))
        if a in retained["A"] and b in retained["B"] and d in retained["D"]:
            kept.append((a, b, d))
    G = len(kept)
    gene_triads = [{"A": a, "B": b, "D": d} for a, b, d in kept]
    print(f"  curated triads (all retained): {G}")
    # hit triad indices
    idx_chr1 = next((i for i, (a, b, d) in enumerate(kept) if b == HIT_CHR1_BD[1] and d == HIT_CHR1_BD[2]), None)
    idx_chr5 = next((i for i, (a, b, d) in enumerate(kept) if a == HIT_CHR5_AD[1] and d == HIT_CHR5_AD[2]), None)
    print(f"  chr1 B-D hit triad idx={idx_chr1}; chr5 A-D hit triad idx={idx_chr5}")

    # phenotype align + GRM subset + whiten ONCE (cap-independent)
    ph = pd.read_csv(PHENO, sep="\t").set_index("sample")
    valid = [s for s in common if s in ph.index and pd.notna(ph.loc[s, TRAIT])]
    idx_in = np.array([common.index(s) for s in valid]); n_t = len(valid)
    Ks = {}
    for s in ("A", "B", "D"):
        Kt = grms[s][np.ix_(idx_in, idx_in)]; Ks[s] = Kt / (np.trace(Kt) / n_t)
    y = _w2._rank_int(np.array([float(ph.loc[s, TRAIT]) for s in valid]))
    Wh, cv = _w2._whiten3(Ks["A"], Ks["B"], Ks["D"], y, seed=42)
    print(f"  n={n_t} whitened once; sigma_hat A={cv.get('A',0):.3f} B={cv.get('B',0):.3f} "
          f"D={cv.get('D',0):.3f} e={cv.get('e',0):.3f} ({time.time()-t0:.0f}s)")

    sweep = {}
    for cap in CAPS:
        # build capped burdens for triad genes
        Bd = {}
        for i, s in enumerate(("A", "B", "D")):
            cols = [_w2._block_burden_capped(Xs[s], retained[s][g[i]], cap, np.random.default_rng(7))
                    for g in kept]
            Bd[s] = _w2._scols_safe(np.column_stack(cols))
        Bd = {s: _w2._scols_safe(Bd[s][idx_in]) for s in ("A", "B", "D")}
        rec = dict(cap=cap)
        for tag, (sx, sy) in [("AB", ("A", "B")), ("AD", ("A", "D")), ("BD", ("B", "D"))]:
            pv = _w2._pairwise_pvals(Wh, y, Bd[sx], Bd[sy])
            rec[f"lambda_obs_{tag}"] = float(_lambda_gc(pv))
            if tag == "BD" and idx_chr1 is not None:
                rec["chr1_BD_p"] = float(pv[idx_chr1])
            if tag == "AD" and idx_chr5 is not None:
                rec["chr5_AD_p"] = float(pv[idx_chr5])
        # carrier variant counts for the hit genes at this cap
        rec["chr1_varcount"] = {g: min(len(retained[s].get(g, [])), cap)
                                for s, g in [("B", HIT_CHR1_BD[1]), ("D", HIT_CHR1_BD[2])]}
        rec["chr5_varcount"] = {g: min(len(retained[s].get(g, [])), cap)
                                for s, g in [("A", HIT_CHR5_AD[1]), ("D", HIT_CHR5_AD[2])]}
        sweep[cap] = rec
        print(f"  cap={cap:6d}: chr1_BD p={rec.get('chr1_BD_p'):.3e} "
              f"chr5_AD p={rec.get('chr5_AD_p'):.3e} | "
              f"lambda AB/AD/BD={rec['lambda_obs_AB']:.2f}/{rec['lambda_obs_AD']:.2f}/{rec['lambda_obs_BD']:.2f}")

    payload = dict(analysis="wheat WGS burden-cap sensitivity (curated-triad hits)",
                   trait=TRAIT, mode=MODE, n=n_t, G_triads=G, caps=CAPS,
                   hit_chr1_BD=HIT_CHR1_BD, hit_chr5_AD=HIT_CHR5_AD,
                   framing=("INT primary. Extract/GRM/whitening computed once (cap-independent); "
                            "only burden capping varies. Stable hit p across caps + flat global "
                            "lambda => hit not a cap artifact and cap does not break calibration."),
                   sweep=sweep)
    fp = OUT / "wheat_cap_sensitivity.json"
    fp.write_text(json.dumps(payload, indent=2, default=float))
    print(f"  wrote {fp} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
