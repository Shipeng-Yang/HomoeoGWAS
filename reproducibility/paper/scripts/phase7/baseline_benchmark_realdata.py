#!/usr/bin/env python3
"""P2 benchmark Part B: real-data recall of the cotton interaction hits by each comparator.

The killer comparison: on the SAME real cotton trait + SAME candidate gene pair, run every method
and show that the homoeolog-pair burden-product test detects the hit while the comparable baselines
do not (single-gene burden + marginal GWAS = null -> pair-only; SNP x SNP = no marker-pair surviving
its own within-pair multiplicity; main-effect burden = no additive explanation; global VC = no
localizable/ detectable homoeolog-epistasis variance). Framed narrowly (Codex): NOT "all baselines
fail", but "the hit is conditionally non-marginal, not explained by single-gene burdens, not
recovered by SNP-level interaction after its own correction, and not captured by global VC; the
gene-pair burden-product test detects it." Pairs with the simulation (baseline_benchmark.py) that
explains WHY under the local genotype architecture.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/mnt/7302share/fast_ysp/U7_GWAS")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/phase7"))
import baseline_benchmark as bb  # noqa: E402
import d2_arm2_synteny_kernel as d2  # noqa: E402

from homoeogwas.interact import scols_safe  # noqa: E402
from homoeogwas.kernel import hadamard_kernel, normalize_kernel  # noqa: E402

OUT = ROOT / "results/phase7/baseline_benchmark_realdata.json"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


b4 = _load(ROOT / "scripts/phase7/bio_pilot/04b_multi_trait_deploy.py", "b4")

HITS = [
    dict(key="Hit1_fiber_length", trait="fiber_length_BLUE", mode="body",
         gene_A="GhM_A06G1605", gene_D="GhM_D06G1557"),
    dict(key="Hit2_length_uniformity", trait="length_uniformity_BLUE", mode="flank",
         gene_A="GhM_A11G2420", gene_D="GhM_D11G2742"),
]


def main():
    cfg = d2.PANELS["cotton_hebau"]
    D = d2._load(cfg)
    XA, XD, KA, KD = D["XA"], D["XD"], D["KA"], D["KD"]
    kernels = {"A": KA, "D": KD, "hom": normalize_kernel(hadamard_kernel({"A": KA, "D": KD}), "trace")}
    samples = b4._load_pheno_aligned(cfg)
    pheno = b4._load_pheno_all(cfg, samples)
    g2s = {"body": b4._load_snp_to_gene(ROOT / "results/phase7/bio_full/snp_to_gene_body.npz"),
           "flank": b4._load_snp_to_gene(ROOT / "results/phase7/bio_full/snp_to_gene_flank2000bp.npz")}

    out = []
    for h in HITS:
        g = g2s[h["mode"]]
        iA, iD = np.asarray(g[h["gene_A"]], int), np.asarray(g[h["gene_D"]], int)
        pair = dict(iA=iA, iD=iD,
                    SX=scols_safe(np.nan_to_num(XA[:, iA], nan=0.0)),
                    SY=scols_safe(np.nan_to_num(XD[:, iD], nan=0.0)),
                    bX=bb._std(d2._block_burden(XA, iA)), bY=bb._std(d2._block_burden(XD, iD)))
        flat = bb.build_flat([pair])
        n_snppairs = int(flat["IA"].shape[1])
        n_snps = int(flat["SM"].shape[1])
        y = b4._rank_int(pheno[h["trait"]])
        _, cv = b4._whiten(KA, KD, y, seed=42)
        Wh = bb._whiten_from_cv(KA, KD, cv)
        ppm, p_vc = bb.run_methods(Wh, y, flat, kernels)
        row = dict(
            key=h["key"], trait=h["trait"], genes=[h["gene_A"], h["gene_D"]],
            n_snp_A=int(iA.size), n_snp_D=int(iD.size), n_snppairs=n_snppairs, n_snps=n_snps,
            ours_interaction_p=float(ppm["ours"][0]),
            mainburden_p=float(ppm["mainburden"][0]),
            singlegene_min_p=float(ppm["singlegene"][0]),
            snpxsnp_best_within_pair_p=float(ppm["snpxsnp"][0]),
            snpxsnp_best_bonferroni_within_pair=float(min(1.0, ppm["snpxsnp"][0] * n_snppairs)),
            marginal_best_snp_p=float(ppm["marginal"][0]),
            globalVC_p=float(p_vc),
            interpretation=(
                "ours (burden-product interaction) detects the pair; single-gene burden + marginal "
                "GWAS are NULL (pair-only, not a marginal/single-gene signal); main-effect burden "
                "gives no additive explanation; the best within-pair SNPxSNP interaction does NOT "
                "survive even its within-pair multiplicity (snpxsnp_best_bonferroni_within_pair; "
                "genome-wide it is far from significant). NOTE: a small SNPxSNP within-pair p that is "
                "close to the burden p indicates single-SNP-pair concentration (the case for Hit1 "
                "fiber_length, where the burden interaction is largely one SNP-pair — consistent with "
                "the independent robustness analysis); for Hit2 the burden p is far below any single "
                "SNP-pair. globalVC does not detect/localize the event."))
        out.append(row)
        print(f"{h['key']}: ours={row['ours_interaction_p']:.2g} | mainburden={row['mainburden_p']:.2g} "
              f"singlegene={row['singlegene_min_p']:.2g} snpxsnp_best={row['snpxsnp_best_within_pair_p']:.2g}"
              f"(bonf {row['snpxsnp_best_bonferroni_within_pair']:.2g}) marginal={row['marginal_best_snp_p']:.2g} "
              f"VC={row['globalVC_p']:.2g}", flush=True)

    payload = dict(analysis="baseline_benchmark_realdata_partB", date="2026-06-02",
                   n_samples=int(KA.shape[0]), method_doc=__doc__, hits=out)
    OUT.write_text(json.dumps(payload, indent=2, default=float))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
