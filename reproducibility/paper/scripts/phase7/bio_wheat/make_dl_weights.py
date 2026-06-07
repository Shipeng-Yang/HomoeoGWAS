#!/usr/bin/env python3
"""Compute y-INDEPENDENT DL-prior pair/triad weights for homoeogwas `interact --weights`
(the DL analogue of make_wheat_heb_weights.py).

FIREWALL (critical): the weight uses ONLY the zero-shot, sequence-based DL columns
(`z_llr_abs_ensemble`, the z-scored absolute log-likelihood-ratio of the PlantCaduceus +
AgroNT ensemble). It MUST NOT use `fusion_score`, `z_gwas`, `p`, or `nlog10_p`, which fuse
the GWAS phenotype and would make the prior y-dependent (double-dipping).

Pipeline: variant z_llr_abs -> gene score (max over the gene's DL SNPs) ->
triad/pair weight (mean of available homoeolog gene scores) -> monotone weight
exp(gamma * zscore), normalized to mean 1.

HONEST CAVEAT (reported in the output): the available DL scores are restricted to
GWAS-candidate SNPs (`source=loco_p_max`), i.e. a y-DEPENDENT selection, and are sparse.
This script demonstrates the aggregation + firewall; a clean production y-independent DL
prior requires GENOME-WIDE DL scoring of all gene-region SNPs (future work).
"""
from __future__ import annotations

import bisect
import gzip
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/mnt/7302share/fast_ysp/U7_GWAS")
DL = ROOT / "results/phase3/m3_3_dl_prior/wheat_watkins/dl_prior_scores_ensemble.tsv.gz"
OUT = ROOT / "results/phase7/interact_validate/wheat_chr1"
GAMMA = 2.0
FLANK = 2000
SUBS = ("A", "B", "D")
# y-INDEPENDENT DL column ONLY; forbidden (y-dependent) columns listed for the audit note
DL_COL = "z_llr_abs_ensemble"
FORBIDDEN = ["fusion_score", "z_gwas", "p", "nlog10_p"]

_s = importlib.util.spec_from_file_location(
    "w4", str(ROOT / "scripts/phase7/bio_wheat/04w_wheat_genomewide.py"))
_w4 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(_w4)


def _load_dl_by_chrom():
    """chrom -> sorted [(pos, z_llr_abs_ensemble)], using ONLY the y-independent DL column."""
    by = {}
    with gzip.open(DL, "rt") as f:
        h = f.readline().rstrip("\n").split("\t")
        ci = {c: i for i, c in enumerate(h)}
        for line in f:
            p = line.rstrip("\n").split("\t")
            val = p[ci[DL_COL]]
            if val == "":
                continue
            by.setdefault(p[ci["chrom"]], []).append((int(p[ci["pos"]]), float(val)))
    return {c: sorted(v) for c, v in by.items()}


def _gene_scores(sub, dl_by_chrom):
    """gene_id -> max z_llr_abs over the gene's flank-window DL SNPs (genes with no DL SNP omitted)."""
    genes = [g for g in _w4._parse_gff(sub, FLANK) if g[1] == f"1{sub}"]  # (gid,chrom,gs,ge,fs,fe)
    snps = dl_by_chrom.get(f"chr1{sub}", [])
    pos = [x[0] for x in snps]
    val = [x[1] for x in snps]
    out = {}
    for gid, _chrom, _gs, _ge, fs, fe in genes:
        lo = bisect.bisect_left(pos, fs)
        hi = bisect.bisect_right(pos, fe)
        if hi > lo:
            out[gid] = float(np.max(val[lo:hi]))
    return out, len(genes)


def main():
    dl_by_chrom = _load_dl_by_chrom()
    gene_score = {}
    cov = {}
    for s in SUBS:
        gs, ngene = _gene_scores(s, dl_by_chrom)
        gene_score.update(gs)
        cov[s] = dict(n_genes=ngene, n_genes_with_dl=len(gs))
        print(f"  chr1{s}: {cov[s]['n_genes_with_dl']}/{ngene} genes with DL score")

    triads = pd.read_csv(OUT / "triads_chr1.tsv", sep="\t")
    rows = []
    n_full = 0
    for _, r in triads.iterrows():
        gs = [gene_score.get(r[f"gene_{s}"]) for s in SUBS]
        present = [x for x in gs if x is not None]
        if not present:
            continue
        if len(present) == 3:
            n_full += 1
        rows.append(dict(gene_A=r["gene_A"], gene_B=r["gene_B"], gene_D=r["gene_D"],
                         score=float(np.mean(present))))
    df = pd.DataFrame(rows)
    if len(df):
        z = (df["score"] - df["score"].mean()) / (df["score"].std() + 1e-12)
        w = np.exp(GAMMA * z)
        df["weight"] = w * (len(df) / w.sum())
        df[["gene_A", "gene_B", "gene_D", "weight"]].to_csv(
            OUT / "triads_chr1_dl_weights.tsv", sep="\t", index=False)

    audit = dict(dl_source=str(DL), dl_column_used=DL_COL, forbidden_columns_excluded=FORBIDDEN,
                 gamma=GAMMA, coverage=cov,
                 n_triads_total=int(len(triads)),
                 n_triads_any_dl=int(len(df)), n_triads_all3_dl=int(n_full),
                 caveat=("CANNOT be used as a clean y-independent prior (mechanistic demo only). "
                         "Two firewall layers: (1) score-level — z_llr_abs_ensemble is zero-shot "
                         "sequence LLR, fusion_score/z_gwas/p excluded (clean). (2) SELECTION-level "
                         "— the scored variant universe is GWAS-candidate-selected (source="
                         "loco_p_max), a y-DEPENDENT selection, so even pure DL scores aggregate "
                         "over a phenotype-filtered SNP set = selection leakage. Secondary biases: "
                         "max-over-flank amplifies candidate selection; missing pattern is non-"
                         "random; zscore/mean-1 normalized within the candidate-restricted set "
                         "inherits that bias; all-3-DL triads = 0 here. A clean DL prior requires "
                         "y-INDEPENDENT DL scoring over the FULL test space (all gene-flank SNPs), "
                         "not just GWAS candidates. The power simulation supports that an "
                         "informative prior discovers more; HEB is the current real y-independent "
                         "demo; DL is capability + audited firewall + documented limitation."))
    (OUT / "dl_weights_audit.json").write_text(json.dumps(audit, indent=2))
    print(f"  triads with any DL: {len(df)}/{len(triads)} (all-3 DL: {n_full})")
    if len(df):
        hit = df[df["gene_B"] == "TraesCS1B02G143800"]
        if len(hit):
            print(f"  chr1 B-D hit triad DL weight = {hit['weight'].iloc[0]:.3f}")
        print(f"  weight range [{df['weight'].min():.3f}, {df['weight'].max():.3f}]")
    print("  wrote triads_chr1_dl_weights.tsv + dl_weights_audit.json")
    print("  CAVEAT: GWAS-candidate-restricted + sparse DL -> mechanistic demo, not a clean prior")


if __name__ == "__main__":
    main()
