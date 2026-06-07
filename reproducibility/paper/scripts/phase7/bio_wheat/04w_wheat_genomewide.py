#!/usr/bin/env python3
"""Phase C wheat Step 4: GENOME-WIDE (all 21 chrom) curated-triad deployment.

Extends the chr1 pilot to all 7 homoeologous chromosome groups. Memory-safe:
processes one subgenome at a time (plink2-extract gene-region SNPs -> compute per-gene
capped burden + subgenome GRM -> release the big dosage matrix). Then pairs genes via
HCTriads curated 1:1:1 triads and runs the 3-pairwise hexaploid-whitened GLS + triad
ACAT + B-perm (reusing 02w machinery).
"""
from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import re
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/mnt/7302share/fast_ysp/U7_GWAS")
GFF = ROOT / "data/reference/wheat/annotation/Triticum_aestivum.IWGSC.57.gff3.gz"
PLINK2 = Path.home() / ".local/share/mamba/envs/polygwas-cpu/bin/plink2"
PHENO = ROOT / "data/processed/wheat/pheno_clean.tsv"
HCTRIADS = ROOT / "data/raw/expression/wheat/HCTriads.csv"
OUT = ROOT / "results/phase7/bio_wheat"
WORK = Path("/mnt/nvme/wheat_genomewide")
SUB_BED = {s: ROOT / f"data/processed/wheat/{s}/all" for s in ("A", "B", "D")}
_GENE_ID_RE = re.compile(r"gene_id=([^;]+)")
CHROMS = {s: [f"{i}{s}" for i in range(1, 8)] for s in ("A", "B", "D")}

_spec = importlib.util.spec_from_file_location(
    "wheat02", str(ROOT / "scripts/phase7/bio_wheat/02w_wheat_deploy.py"))
_w2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_w2)


def _v10_to_v11(gid):
    return gid.replace("01G", "02G", 1) if "01G" in gid else gid


def _parse_gff(sub, flank):
    """genes on the 7 chroms of subgenome `sub`: (gid, gff_chrom, start, end, fs, fe)."""
    chroms = set(CHROMS[sub])
    rows = []
    with gzip.open(GFF, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) < 9 or p[2] != "gene" or p[0] not in chroms or "biotype=protein_coding" not in p[8]:
                continue
            m = _GENE_ID_RE.search(p[8])
            if not m:
                continue
            gs, ge = int(p[3]), int(p[4])
            rows.append((m.group(1), p[0], gs, ge, max(1, gs - flank), ge + flank))
    return rows


def _extract_and_burden(sub, genes, mode, flank, cap, min_snp, rng):
    """plink2-extract gene-region SNPs for all 7 chroms of `sub`, return
    (gene2burden dict, GRM, samples)."""
    WORK.mkdir(parents=True, exist_ok=True)
    rf = WORK / f"range_{sub}_{mode}.txt"
    with open(rf, "w") as f:
        for gid, gff_chrom, gs, ge, fs, fe in genes:
            ws, we = (gs, ge) if mode == "body" else (fs, fe)
            f.write(f"chr{gff_chrom}\t{ws}\t{we}\t{gid}\n")
    outp = WORK / f"gw_{sub}_{mode}"
    cmd = [str(PLINK2), "--pfile", str(SUB_BED[sub]), "--extract", "range", str(rf),
           "--make-bed", "--out", str(outp), "--threads", "32", "--memory", "200000"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"plink2 {sub}: {r.stderr[-600:]}")
    from homoeogwas.io import load_bed_hardcall
    bed = load_bed_hardcall(outp)
    X = np.asarray(bed.dosage, dtype=np.float64)
    samples = list(np.asarray(bed.samples))
    # SNP-to-gene by position (per chrom)
    import bisect
    pos = []
    bchrom = []
    with open(f"{outp}.bim") as f:
        for line in f:
            pp = line.rstrip("\n").split("\t")
            bchrom.append(pp[0])
            pos.append(int(pp[3]))
    pos = np.array(pos)
    bchrom = np.array(bchrom)
    gene2snp = {}
    for gff_chrom in CHROMS[sub]:
        cn = f"chr{gff_chrom}"
        sel = np.where(bchrom == cn)[0]
        if sel.size == 0:
            continue
        spos = pos[sel]
        order = np.argsort(spos)
        spos_s = spos[order]
        sel_s = sel[order]
        gc = [(g[0], (g[2] if mode == "body" else g[4]), (g[3] if mode == "body" else g[5]))
              for g in genes if g[1] == gff_chrom]
        gc.sort(key=lambda x: x[1])
        starts = [g[1] for g in gc]
        for j, ps in enumerate(spos_s):
            k = bisect.bisect_right(starts, ps) - 1
            steps = 0
            while k >= 0 and steps < 3000:
                gid, ws, we = gc[k]
                if ps > we and ps - ws > 2_000_000:
                    break
                if ws <= ps <= we:
                    gene2snp.setdefault(gid, []).append(int(sel_s[j]))
                k -= 1
                steps += 1
    retained = {g: idx for g, idx in gene2snp.items() if len(idx) >= min_snp}
    g2b = {g: _w2._block_burden_capped(X, idx, cap, rng) for g, idx in retained.items()}
    n_snp = {g: int(len(idx)) for g, idx in retained.items()}    # callable SNP count per gene
    K = _w2._grm_from_X(X)
    del X
    return g2b, n_snp, K, samples


def _write_ranking(path, label, sink, kept, meta_kept, nsnp, glen):
    """FULL genome-wide per-triad ranking TSV (every callable 1:1:1 triad), ordered by ascending
    per-triad ACAT. Descriptive only — the inferential test is the pre-registered enrichment
    (reports/enrichment_prereg.md), NOT per-triad Bonferroni. gene_len comes solely from the GFF
    (no fabrication). Same p-values the scan already computed; just not discarded."""
    import csv

    from scipy import stats as _st
    tags = ("AB", "AD", "BD")
    pmat = np.column_stack([sink[t] for t in tags])              # (G, 3) aligned to kept order
    pmat = np.where(np.isfinite(pmat), pmat, 1.0)                # Fix4: sanitize before ranking
    # rank = 0-based ordinal after a STABLE ascending sort on per-triad ACAT; tie_group = first
    # ordinal of an exact-p tie (exact float equality), NOT a shared statistical rank.
    p_acat = np.array([_w2._acat([pmat[i, 0], pmat[i, 1], pmat[i, 2]]) for i in range(len(kept))])
    p_acat = np.where(np.isfinite(p_acat), p_acat, 1.0)
    nl = -np.log10(np.clip(p_acat, 1e-300, 1.0))
    min_pair = pmat.min(1)
    min_tag = np.array(tags)[pmat.argmin(1)]
    ns = {s: np.array([nsnp[s][k[i]] for k in kept]) for i, s in enumerate(("A", "B", "D"))}
    ns_sum = ns["A"] + ns["B"] + ns["D"]
    # decile bin of callable-SNP-count (descriptive; authoritative matching re-derived downstream)
    r_bin = _st.rankdata(ns_sum, method="average")
    cbin = np.clip(((r_bin - 0.5) / ns_sum.size * 10).astype(int), 0, 9)
    order = np.argsort(p_acat, kind="stable")
    rank_of = np.empty(len(kept), int)
    rank_of[order] = np.arange(len(kept))
    # tie groups on identical p_acat
    tie_of = np.empty(len(kept), int)
    tg, prev = 0, None
    for j, i in enumerate(order):
        if prev is None or p_acat[i] != prev:
            tg, prev = j, p_acat[i]
        tie_of[i] = tg
    bonf = 0.05 / len(kept)
    gl = {s: np.array([glen[s].get(g[i], -1) for g in kept]) for i, s in enumerate(("A", "B", "D"))}
    gl_sum = np.where((gl["A"] > 0) & (gl["B"] > 0) & (gl["D"] > 0), gl["A"] + gl["B"] + gl["D"], -1)
    head = ["rank", "gene_A", "gene_B", "gene_D", "p_AB", "p_AD", "p_BD", "p_acat", "neglog10p_acat",
            "min_pair_p", "min_pair_tag", "n_snp_A", "n_snp_B", "n_snp_D", "n_snp_triad",
            "callable_snp_decile", "gene_len_A", "gene_len_B", "gene_len_D", "gene_len_triad_sum",
            "group_id", "cardinality_abs", "synteny", "HC_LC", "tie_group", "bonferroni_sig_acat"]
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="") as fh:
        wtr = csv.writer(fh, delimiter="\t")
        wtr.writerow(head)
        for i in order:
            gid, card, syn, hclc = meta_kept[i]
            wtr.writerow([int(rank_of[i]), kept[i][0], kept[i][1], kept[i][2],
                          repr(float(pmat[i, 0])), repr(float(pmat[i, 1])), repr(float(pmat[i, 2])),
                          repr(float(p_acat[i])), repr(float(nl[i])), repr(float(min_pair[i])),
                          str(min_tag[i]), int(ns["A"][i]), int(ns["B"][i]), int(ns["D"][i]),
                          int(ns_sum[i]), int(cbin[i]),
                          (int(gl["A"][i]) if gl["A"][i] > 0 else "NA"),
                          (int(gl["B"][i]) if gl["B"][i] > 0 else "NA"),
                          (int(gl["D"][i]) if gl["D"][i] > 0 else "NA"),
                          (int(gl_sum[i]) if gl_sum[i] > 0 else "NA"),
                          gid, card, syn, hclc, int(tie_of[i]), int(p_acat[i] < bonf)])
    print(f"  [{label}] wrote ranking {p} ({len(kept)} triads)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["body", "flank"], default="flank")
    ap.add_argument("--trait", default="days_to_emerg")
    ap.add_argument("--cap", type=int, default=150)
    ap.add_argument("--flank-bp", type=int, default=2000)
    ap.add_argument("--min-snp", type=int, default=3)
    ap.add_argument("--perm-B", type=int, default=2000)
    ap.add_argument("--n-jobs", type=int, default=48)
    args = ap.parse_args()
    t0 = time.time()
    print(f"=== wheat GENOME-WIDE curated-triad deploy (mode={args.mode}, trait={args.trait}) ===")

    rng = np.random.default_rng(7)
    burdens, grms, nsnp, glen, samples_ref = {}, {}, {}, {}, None
    for s in ("A", "B", "D"):
        genes = _parse_gff(s, args.flank_bp)
        # gene body length (end - start + 1) from the GFF; the ONLY source of gene_len (no fabrication)
        glen[s] = {gid: int(ge - gs + 1) for gid, _gc, gs, ge, _fs, _fe in genes}
        g2b, n_snp, K, samples = _extract_and_burden(s, genes, args.mode, args.flank_bp,
                                                     args.cap, args.min_snp, rng)
        burdens[s] = g2b
        grms[s] = K
        nsnp[s] = n_snp
        if samples_ref is None:
            samples_ref = samples
        assert samples == samples_ref, f"sample mismatch {s}"
        print(f"  {s}: {len(g2b)} retained genes, GRM {K.shape} ({time.time()-t0:.1f}s)")
    common = samples_ref

    # curated 1:1:1 triads, all chroms, both genes retained
    hc = pd.read_csv(HCTRIADS)
    hc = hc[hc["cardinality_abs"] == "1:1:1"]
    kept = []
    meta_kept = []                          # HCTriads provenance per kept triad (aligned to kept)
    for _, r in hc.iterrows():
        a, b, d = _v10_to_v11(str(r["A"])), _v10_to_v11(str(r["B"])), _v10_to_v11(str(r["D"]))
        if a in burdens["A"] and b in burdens["B"] and d in burdens["D"]:
            kept.append((a, b, d))
            meta_kept.append((r.get("group_id", ""), str(r.get("cardinality_abs", "")),
                              str(r.get("synteny", "")), str(r.get("HC.LC", ""))))
    G = len(kept)
    print(f"  curated 1:1:1 triads with full genotype coverage: {G} ({time.time()-t0:.1f}s)")

    gene_triads = [{"A": a, "B": b, "D": d} for a, b, d in kept]
    Bdict_full = {s: _w2._scols_safe(np.column_stack([burdens[s][g[i]] for g in kept]))
                  for i, s in enumerate(("A", "B", "D"))}

    ph = pd.read_csv(PHENO, sep="\t").set_index("sample")
    valid = [s for s in common if s in ph.index and pd.notna(ph.loc[s, args.trait])]
    idx_in = np.array([common.index(s) for s in valid])
    n_t = len(valid)
    KA = grms["A"][np.ix_(idx_in, idx_in)]
    KB = grms["B"][np.ix_(idx_in, idx_in)]
    KD = grms["D"][np.ix_(idx_in, idx_in)]
    KA = KA / (np.trace(KA) / n_t)
    KB = KB / (np.trace(KB) / n_t)
    KD = KD / (np.trace(KD) / n_t)
    Bd = {s: _w2._scols_safe(Bdict_full[s][idx_in]) for s in ("A", "B", "D")}
    y_raw = np.array([float(ph.loc[s, args.trait]) for s in valid])
    print(f"  trait={args.trait} n={n_t} G={G} ({time.time()-t0:.1f}s)")

    sink_INT, sink_raw = {}, {}
    out_INT = _w2.deploy(_w2._rank_int(y_raw), "INT", KA, KB, KD, Bd, gene_triads, args.n_jobs,
                         args.perm_B, full_sink=sink_INT)
    out_raw = _w2.deploy(y_raw, "raw", KA, KB, KD, Bd, gene_triads, args.n_jobs, args.perm_B,
                         full_sink=sink_raw)
    for lab, o in [("INT", out_INT), ("raw", out_raw)]:
        print(f"  [{lab}] triad ACAT obs={o['triad_acat_obs']:.4g} emp={o['triad_acat_emp']:.4g}")
        for tag in ("AB", "AD", "BD"):
            pw = o["pairwise"][tag]
            pe = o["pairwise_emp"][tag]
            print(f"    {tag}: ACAT_obs={pw['acat']:.4g} emp={pe['acat_emp']:.4g} "
                  f"λ_perm={pe['lambda_gc_perm_median']:.3f} nsig(α/G={pw['bonferroni_alpha']:.1e})={pw['n_sig']}")
            for h in pw["top5"][:3]:
                if h["p"] < pw["bonferroni_alpha"]:
                    g = h["genes"]
                    print(f"        HIT p={h['p']:.3g} A={g['A']} B={g['B']} D={g['D']}")

    _write_ranking(OUT / f"wheat_GW_ranking_{args.mode}_INT.tsv", "INT", sink_INT, kept,
                   meta_kept, nsnp, glen)
    _write_ranking(OUT / f"wheat_GW_ranking_{args.mode}_raw.tsv", "raw", sink_raw, kept,
                   meta_kept, nsnp, glen)

    full = dict(panel="wheat_watkins", mode=args.mode, trait=args.trait, scope="genome_wide_21chrom",
                triad_source="HCTriads_curated_1to1to1", n_triad=G, perm_B=args.perm_B,
                INT_primary=out_INT, raw_sensitivity=out_raw)
    fp = OUT / f"deploy_wheat_GW_CURATED_{args.mode}.json"
    fp.write_text(json.dumps(full, indent=2, default=float))
    print(f"  wrote {fp} ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
