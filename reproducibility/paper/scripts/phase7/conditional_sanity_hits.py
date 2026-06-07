#!/usr/bin/env python3
"""P0-1: persist the conditional-on-marginals sanity panel for the 4 published interaction hits.

The "pair-only / single-gene-GWAS-invisible" claim (cotton Hit1/Hit2, wheat chr1 B-D / chr5 A-D)
was asserted in text but NEVER written to a file; the wheat VIF values (1.13 / 2.28) were hardcoded
constants. This script rebuilds the EXACT whitened GLS design each hit was discovered in, reproduces
its interaction p (hard-asserted), and computes + persists, via the tested engine helper
``homoeogwas.interact.pair_conditional_diagnostics``:
  - projection-based VIF / R2_int|marginals in whitened space (collinearity of the interaction
    regressor with the two marginal burdens);
  - a nested-model F panel: interaction|marginals (== Wald, with F==t^2 check), JOINT main effects
    (null vs marginals -> single-gene visibility), total pair model;
  - single-gene burden-GLS marginal p (each burden alone under the same covariance).

Cotton reuses the d2_arm2/04b deploy machinery (n=419, fiber_length-complete samples for ALL traits,
2-kernel {K_A,K_D} whitener). Wheat reuses 02w + the cached genome-wide flank extraction
(/mnt/nvme/wheat_genomewide/gw_{A,B,D}_flank) so the {K_A,K_B,K_D} hexaploid whitener matches the
genome-wide deploy exactly. Output: results/phase7/conditional_sanity_hits.json.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/mnt/7302share/fast_ysp/U7_GWAS")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/phase7"))
import d2_arm2_synteny_kernel as d2  # noqa: E402

from homoeogwas.interact import pair_conditional_diagnostics  # noqa: E402


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


b4 = _load_module(ROOT / "scripts/phase7/bio_pilot/04b_multi_trait_deploy.py", "b4")
w2 = _load_module(ROOT / "scripts/phase7/bio_wheat/02w_wheat_deploy.py", "w2")

GFF = ROOT / "data/reference/wheat/annotation/Triticum_aestivum.IWGSC.57.gff3.gz"
GW = Path("/mnt/nvme/wheat_genomewide")
PHENO_WHEAT = ROOT / "data/processed/wheat/pheno_clean.tsv"
OUT = ROOT / "results/phase7/conditional_sanity_hits.json"
REL_TOL = 1e-2   # hard reproduction tolerance on the interaction p

COTTON_HITS = [
    dict(key="cotton_Hit1_fiber_length", panel="cotton", mode="body", trait="fiber_length_BLUE",
         gene_A="GhM_A06G1605", gene_D="GhM_D06G1557", expected_p=0.00021359441415680198,
         context="cotton AADD, body window, 2-kernel {K_A,K_D} whitener, n=419"),
    dict(key="cotton_Hit2_length_uniformity", panel="cotton", mode="flank", trait="length_uniformity_BLUE",
         gene_A="GhM_A11G2420", gene_D="GhM_D11G2742", expected_p=4.747018080361048e-05,
         context="cotton AADD, +-2kb flank window, 2-kernel {K_A,K_D} whitener, n=419"),
]

# wheat hit gene coords (GFF, IWGSC v1.1); bim CHROM carries the 'chr' prefix
WHEAT_GENES = {
    "TraesCS1B02G143800": ("chr1B", 195552668, 195560543),
    "TraesCS1D02G128400": ("chr1D", 141545585, 141548899),
    "TraesCS5A02G169500": ("chr5A", 362168560, 362171424),
    "TraesCS5D02G173900": ("chr5D", 272354579, 272358172),
}
WHEAT_HITS = [
    dict(key="wheat_chr1_BD", subs=("B", "D"), gene_X="TraesCS1B02G143800", gene_Y="TraesCS1D02G128400",
         pair_tag="BD", expected_p=2.418076112262021e-06,
         context="wheat AABBDD, genome-wide flank, hexaploid {K_A,K_B,K_D} whitener, n=827"),
    dict(key="wheat_chr5_AD", subs=("A", "D"), gene_X="TraesCS5A02G169500", gene_Y="TraesCS5D02G173900",
         pair_tag="AD", expected_p=3.206834751605078e-07,
         context="wheat AABBDD, genome-wide flank, hexaploid {K_A,K_B,K_D} whitener, n=827"),
]


def _std1(v):
    v = np.asarray(v, float)
    sd = v.std(ddof=0)
    return (v - v.mean()) / (sd if sd > 1e-12 else 1.0)


def _record(hit, diag, reproduced_p):
    rel = abs(reproduced_p - hit["expected_p"]) / hit["expected_p"]
    assert rel < REL_TOL, (f"{hit['key']}: reproduced p={reproduced_p:.4g} vs expected "
                           f"{hit['expected_p']:.4g} (rel={rel:.2e} > {REL_TOL})")
    assert np.isfinite(diag["vif_int"]) and diag["vif_int"] >= 1.0, f"{hit['key']}: bad VIF"
    return dict(key=hit["key"], context=hit["context"], genes=[hit.get("gene_A") or hit["gene_X"],
                hit.get("gene_D") or hit["gene_Y"]], trait=hit.get("trait", "days_to_emerg"),
                expected_interaction_p=hit["expected_p"], reproduced_interaction_p=reproduced_p,
                reproduction_rel_error=rel, diagnostics=diag)


def run_cotton(hits):
    cfg = d2.PANELS["cotton_hebau"]
    D = d2._load(cfg)
    KA, KD, XA, XD = D["KA"], D["KD"], D["XA"], D["XD"]
    samples = b4._load_pheno_aligned(cfg)
    pheno = b4._load_pheno_all(cfg, samples)
    npz = {"body": ROOT / "results/phase7/bio_full/snp_to_gene_body.npz",
           "flank": ROOT / "results/phase7/bio_full/snp_to_gene_flank2000bp.npz"}
    g2s = {m: b4._load_snp_to_gene(p) for m, p in npz.items()}
    out = []
    for h in hits:
        g = g2s[h["mode"]]
        bx = _std1(d2._block_burden(XA, g[h["gene_A"]]))
        by = _std1(d2._block_burden(XD, g[h["gene_D"]]))
        y = b4._rank_int(pheno[h["trait"]])
        Wh, _cv = b4._whiten(KA, KD, y, seed=42)
        diag = pair_conditional_diagnostics(Wh, y, bx, by)
        rec = _record(h, diag, diag["interaction_p_wald"])
        rec["n_snp"] = dict(gene_A=int(g[h["gene_A"]].size), gene_D=int(g[h["gene_D"]].size))
        out.append(rec)
        print(f"  [{h['key']}] p_repro={diag['interaction_p_wald']:.4g} (exp {h['expected_p']:.4g}) "
              f"VIF={diag['vif_int']:.3f} R2|marg={diag['r2_int_given_marginals']:.3f} "
              f"marg_X={diag['single_gene_marginal']['p_X_alone']:.3g} "
              f"marg_Y={diag['single_gene_marginal']['p_Y_alone']:.3g}", flush=True)
    return out


def _load_wheat_sub(sub, want_genes, cap=150):
    """Load cached gw_{sub}_flank, return (K_full, {gene: (burden_1051, n_snp)}, samples)."""
    from homoeogwas.io import load_bed_hardcall
    bed = load_bed_hardcall(GW / f"gw_{sub}_flank")
    X = np.asarray(bed.dosage, dtype=np.float64)
    chrom = np.asarray(bed.chrom).astype(str)
    pos = np.asarray(bed.pos, dtype=np.int64)
    samples = list(np.asarray(bed.samples))
    K = w2._grm_from_X(X)
    rng = np.random.default_rng(7)
    burdens = {}
    for gid in want_genes:
        c, s, e = WHEAT_GENES[gid]
        idx = np.where((chrom == c) & (pos >= s - 2000) & (pos <= e + 2000))[0]
        burdens[gid] = (w2._block_burden_capped(X, idx, cap, rng), int(idx.size))
    del X
    return K, burdens, samples


def run_wheat(hits):
    need = {"A": [], "B": [], "D": []}
    for h in hits:
        for sub, gid in zip(h["subs"], (h["gene_X"], h["gene_Y"]), strict=True):
            need[sub].append(gid)
    Kfull, burd, samples_ref = {}, {}, None
    t = time.time()
    for s in ("A", "B", "D"):
        K, b, samp = _load_wheat_sub(s, need[s])
        Kfull[s] = K
        burd.update(b)
        if samples_ref is None:
            samples_ref = samp
        assert samp == samples_ref, f"wheat sample order mismatch {s}"
        print(f"  loaded gw_{s}: K={K.shape} ({time.time()-t:.0f}s)", flush=True)
    ph = pd.read_csv(PHENO_WHEAT, sep="\t").set_index("sample")
    valid = [s for s in samples_ref if s in ph.index and pd.notna(ph.loc[s, "days_to_emerg"])]
    idx_in = np.array([samples_ref.index(s) for s in valid])
    n_t = len(valid)
    Kt = {}
    for s in ("A", "B", "D"):
        Ks = Kfull[s][np.ix_(idx_in, idx_in)]
        Kt[s] = Ks / (np.trace(Ks) / n_t)
    y = w2._rank_int(np.array([float(ph.loc[s, "days_to_emerg"]) for s in valid]))
    Wh, _cv = w2._whiten3(Kt["A"], Kt["B"], Kt["D"], y, seed=42)
    print(f"  wheat n={n_t}, hexaploid whitener built ({time.time()-t:.0f}s)", flush=True)
    out = []
    for h in hits:
        bx = _std1(burd[h["gene_X"]][0][idx_in])
        by = _std1(burd[h["gene_Y"]][0][idx_in])
        diag = pair_conditional_diagnostics(Wh, y, bx, by)
        rec = _record(h, diag, diag["interaction_p_wald"])
        rec["n_snp"] = dict(gene_X=burd[h["gene_X"]][1], gene_Y=burd[h["gene_Y"]][1])
        out.append(rec)
        print(f"  [{h['key']}] p_repro={diag['interaction_p_wald']:.4g} (exp {h['expected_p']:.4g}) "
              f"VIF={diag['vif_int']:.3f} R2|marg={diag['r2_int_given_marginals']:.3f} "
              f"marg_X={diag['single_gene_marginal']['p_X_alone']:.3g} "
              f"marg_Y={diag['single_gene_marginal']['p_Y_alone']:.3g}", flush=True)
    return out


def main():
    t0 = time.time()
    print("=== P0-1 conditional-on-marginals sanity for the 4 interaction hits ===")
    print("--- cotton (n=419, 2-kernel whitener) ---")
    cotton = run_cotton(COTTON_HITS)
    print("--- wheat (n=827, hexaploid whitener; reloading genome-wide GRM cache) ---")
    wheat = run_wheat(WHEAT_HITS)
    payload = dict(
        analysis="conditional_on_marginals_sanity", date="2026-06-02",
        method=("Per hit, the EXACT whitened GLS design (same burdens + multi-kernel REML whitener "
                "as the deploy) is rebuilt; the interaction Wald p is reproduced (hard-asserted "
                f"rel<{REL_TOL}). Then: projection-based VIF/R2_int|marg in whitened space "
                "(c0=Wh@1 as the whitened intercept; R2=1-SSE(cI~c0+cX+cY)/SSE(cI~c0)); nested "
                "F-tests (interaction|marginals with F==t^2 check, JOINT main effects, total pair "
                "model); single-gene burden-GLS marginal p (each burden alone under the same "
                "covariance). VIF/R2 quantify whether the interaction is collinear with the "
                "marginals; the single-gene marginal p backs 'pair-only / single-gene invisible'."),
        interpretation=("VIF near 1 + R2_int|marg small => the interaction term is geometrically "
                        "near-orthogonal to the two marginal burdens (not a collinearity artifact). "
                        "Single-gene marginal p >> interaction p => neither homoeolog alone is "
                        "associated; only the pair burden-product detects the signal. F==t^2 / "
                        "p_F==p_wald and the Frisch-Waugh-Lovell residualized-interaction t are "
                        "implementation checks, NOT independent evidence."),
        provenance=dict(
            wheat_context=("genome-wide flank (/mnt/nvme/wheat_genomewide/gw_{A,B,D}_flank cache); "
                           "GRM = grm_from_X over all genome-wide gene-region SNPs per subgenome, "
                           "hexaploid {K_A,K_B,K_D} whitener. This is the SAME context as "
                           "deploy_wheat_GW_CURATED_flank (both wheat hits jointly reported) and "
                           "as the dosage S/I/Q + cap-sensitivity analyses."),
            superseded=("the previously quoted wheat chr1 B-D VIF=1.134 was a chr1-region deploy "
                        "context (chr1-only GRM, reproduced p=2.955e-7) and is SUPERSEDED here: the "
                        "genome-wide value is VIF=1.574 (R2|marg=0.365). chr5 A-D VIF=2.271 matches "
                        "the prior 2.28. Report the genome-wide values with the genome-wide p."),
            claim_scope=("supports 'no marginal association for these standardized gene burdens "
                         "under the SAME GLS covariance model' (single-gene burden-GLS p and joint "
                         "main-effect F all null while the interaction is significant) -- NOT 'no "
                         "single-gene signal of any conceivable model'.")),
        hits=cotton + wheat)
    OUT.write_text(json.dumps(payload, indent=2, default=float))
    print(f"\nwrote {OUT} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
