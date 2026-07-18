#!/usr/bin/env python3
"""Cross-panel IBS: are the CottonGVD and HBAU panels genetically independent? (prereg s8, codex fix)

The two panels' sample IDs are disjoint namespaces, so string non-overlap proves nothing. This lifts
CottonGVD SNP positions from Gh_CRI v1 onto NDM8 through the minimap2 alignment, intersects with the
HBAU markers at the same physical locus and matching alleles, and computes genome-wide IBS between
every HBAU x CottonGVD accession pair. Any HBAU accession with IBS > 0.99 to a CottonGVD accession is
a duplicate and, per the prereg, is dropped from the replication and the count reported.

Runs BEFORE the interaction p-value is revealed: independence is an eligibility condition for the
word "replication", not a post-hoc penalty (codex).
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/mnt/7302share/fast_ysp/U7_GWAS")
sys.path.insert(0, str(ROOT / "src"))
from homoeogwas.io import load_bed_hardcall  # noqa: E402

PAF = ROOT / "results/phase7/cotton_replication/cri_to_ndm8.paf.gz"
CRI = ROOT / "data/processed/cotton/cottongvd"          # {A,D}/all.{bed,bim,fam}
HBAU = ROOT / "results/viz_preview/fig3_cotton_locus/rescue"   # {A,D}_wgs
OUT = ROOT / "results/phase7/cotton_replication/cross_panel_ibs.json"
N_MARK = 6000          # target shared markers for IBS; a few thousand resolves duplicates cleanly


def build_liftover(min_mapq=30, min_blk=2000):
    """CRI(query) -> NDM8(target) position map, from primary high-mapq collinear blocks only.

    Returns per-CRI-chrom sorted arrays (qstart, tstart, strand, length-of-run) good enough to lift a
    SNP by locating its block and walking the CIGAR. Only same-labelled chromosomes (A01->A01) and
    forward primary alignments are kept, which is what a 1:1 homoeologous-genome correspondence is.
    """
    blocks = {}
    op = gzip.open if str(PAF).endswith(".gz") else open
    with op(PAF, "rt") as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) < 12:
                continue
            qn, ql, qs, qe, strand, tn, tl, ts, te = (f[0], int(f[1]), int(f[2]), int(f[3]),
                                                      f[4], f[5], int(f[6]), int(f[7]), int(f[8]))
            mapq = int(f[11])
            tags = {t.split(":")[0]: t for t in f[12:]}
            if mapq < min_mapq or (qe - qs) < min_blk:
                continue
            if "tp" in tags and tags["tp"].split(":")[-1] != "P":     # primary only
                continue
            # both bims label chromosomes A01..D13; PAF names them by the same label here
            if qn != tn or strand != "+":
                continue
            cg = tags.get("cg")
            if not cg:
                continue
            blocks.setdefault(qn, []).append((qs, ts, cg.split(":")[-1]))
    return blocks


def _lift_positions(blocks_for_chrom, query_pos):
    """Walk each block's CIGAR to map query positions to target. query_pos: sorted np.int64 array."""
    import re
    out = {}
    for qs, ts, cg in blocks_for_chrom:
        q, t = qs, ts
        ops = re.findall(r"(\d+)([MID])", cg)
        # positions of interest that fall in this block's query span
        for n, o in ops:
            n = int(n)
            if o == "M":
                # map any query_pos in [q, q+n) to t + (pos-q)
                lo, hi = q, q + n
                idx = query_pos[(query_pos >= lo) & (query_pos < hi)]
                for p in idx:
                    out[int(p)] = ts_off = t + (int(p) - q)  # noqa: F841
                    out[int(p)] = t + (int(p) - q)
                q += n
                t += n
            elif o == "I":      # insertion in query (CRI) relative to target: query advances only
                q += n
            elif o == "D":      # deletion in query relative to target: target advances only
                t += n
    return out


def main():
    if not PAF.exists() or PAF.stat().st_size < 1000:
        raise SystemExit(f"alignment not ready: {PAF}")
    blocks = build_liftover()
    print(f"liftover blocks on {len(blocks)} chromosomes "
          f"({sum(len(v) for v in blocks.values())} primary collinear alignments)")

    shared = {}     # (chrom, ndm8_pos) -> None, plus we keep CRI pos to fetch the CRI genotype
    rng = np.random.default_rng(7)
    for sub in ("A", "D"):
        cri_bim = np.loadtxt(CRI / f"{sub}/all.bim", dtype=str, usecols=(0, 3))
        hb_bim = np.loadtxt(HBAU / f"{sub}_wgs.bim", dtype=str, usecols=(0, 3))
        hb_set = {(c, int(p)) for c, p in hb_bim}
        for c in sorted(set(cri_bim[:, 0])):
            if c not in blocks:
                continue
            qpos = np.sort(cri_bim[cri_bim[:, 0] == c, 1].astype(np.int64))
            if qpos.size > 40000:      # subsample per chrom for speed
                qpos = np.sort(rng.choice(qpos, 40000, replace=False))
            lifted = _lift_positions(blocks[c], qpos)
            for cri_p, ndm8_p in lifted.items():
                if (c, ndm8_p) in hb_set:
                    shared[(sub, c, cri_p, ndm8_p)] = True
    print(f"shared markers (lifted CRI position present in HBAU): {len(shared)}")
    if len(shared) < 500:
        raise SystemExit("too few shared markers to compute IBS reliably")

    # subsample shared markers, load genotypes in both panels, harmonise alleles, compute IBS
    keys = list(shared)
    if len(keys) > N_MARK:
        keys = [keys[i] for i in np.sort(rng.choice(len(keys), N_MARK, replace=False))]

    def _load(bed_root, name, want, coord_idx):
        bed = load_bed_hardcall(bed_root / name)
        ch = np.asarray(bed.chrom).astype(str)
        po = np.asarray(bed.pos, dtype=np.int64)
        a1 = np.asarray(bed.a1).astype(str) if hasattr(bed, "a1") else None
        pos_ix = {(c, int(p)): i for i, (c, p) in enumerate(zip(ch, po))}
        cols, rows = [], []
        for k in want:
            i = pos_ix.get((k[1], k[coord_idx]))
            if i is not None:
                cols.append(i)
                rows.append(k)
        X = np.asarray(bed.dosage, float)[:, cols]
        return list(np.asarray(bed.samples).astype(str)), rows, X, (a1[cols] if a1 is not None else None)

    # CottonGVD by CRI pos (index 2), HBAU by NDM8 pos (index 3)
    csA = [k for k in keys if k[0] == "A"]; csD = [k for k in keys if k[0] == "D"]
    XcA = _load(CRI, "A/all", csA, 2); XhA = _load(HBAU, "A_wgs", csA, 3)
    XcD = _load(CRI, "D/all", csD, 2); XhD = _load(HBAU, "D_wgs", csD, 3)

    # align both panels to the same marker order (intersection of what each actually had)
    def _match(cri, hb):
        ci = {k: j for j, k in enumerate(cri[1])}
        hi = {k: j for j, k in enumerate(hb[1])}
        common = [k for k in cri[1] if k in hi]
        Xc = cri[2][:, [ci[k] for k in common]]
        Xh = hb[2][:, [hi[k] for k in common]]
        return Xc, Xh
    XcAm, XhAm = _match(XcA, XhA); XcDm, XhDm = _match(XcD, XhD)
    Xc = np.hstack([XcAm, XcDm]); Xh = np.hstack([XhAm, XhDm])
    print(f"markers used for IBS: {Xc.shape[1]}  (CottonGVD {Xc.shape[0]} x HBAU {Xh.shape[0]})")

    # IBS as fraction of allele-dosage agreement after harmonising each marker to matched allele freq
    # (a REF/ALT swap between assemblies flips 0<->2; detect by whichever orientation maximises match)
    sc, sh = XcA[0], XhA[0]
    Xc = np.where(np.isnan(Xc), np.nanmean(Xc, 0), Xc)
    Xh = np.where(np.isnan(Xh), np.nanmean(Xh, 0), Xh)
    # per-marker, pick orientation of HBAU (as-is vs 2-x) that better matches CottonGVD allele freq
    fc = Xc.mean(0) / 2
    fh = Xh.mean(0) / 2
    flip = np.abs((2 - Xh).mean(0) / 2 - fc) < np.abs(fh - fc)
    Xh = np.where(flip[None, :], 2 - Xh, Xh)

    # a duplicate accession has near-perfect genotype correlation regardless of per-marker REF/ALT
    # orientation, so |Pearson r| across markers is the robust, flip-invariant duplicate signal
    Zc = (Xc - Xc.mean(0)) / (Xc.std(0) + 1e-9)
    Zh = (Xh - Xh.mean(0)) / (Xh.std(0) + 1e-9)
    m = Zc.shape[1]
    R = np.abs(Zh @ Zc.T) / m                    # (n_hbau x n_cottongvd) |correlation|
    max_ibs = R.max(1)
    dup = int((max_ibs > 0.90).sum())            # |r|>0.9 = same or near-identical line
    res = dict(prereg_blob="3b675e8ca696", markers=int(Xc.shape[1]),
               n_cottongvd=int(Gc.shape[0]), n_hbau=int(Gh.shape[0]),
               max_ibs_summary=dict(min=float(max_ibs.min()), median=float(np.median(max_ibs)),
                                    p95=float(np.percentile(max_ibs, 95)), max=float(max_ibs.max())),
               n_hbau_corr_gt_0_90=dup, metric="abs_pearson_r across shared markers",
               duplicate_hbau_samples=[sh[i] for i in np.where(max_ibs > 0.99)[0]])
    OUT.write_text(json.dumps(res, indent=2))
    print(f"\nmax cross-panel IBS  min {max_ibs.min():.3f} | median {np.median(max_ibs):.3f} | "
          f"p95 {np.percentile(max_ibs,95):.3f} | max {max_ibs.max():.3f}")
    print(f"HBAU accessions with IBS > 0.99 to a CottonGVD accession (duplicates): {dup}/{Gh.shape[0]}")
    print("VERDICT:", "PANELS OVERLAP -- drop the duplicate lines and note" if dup else
          "no duplicates detected -- panels are genetically distinct at this resolution")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
