"""Which component drives the omniB p-values the candidates were selected on?

The candidate table (aggregate_perpair) holds omniB = ACAT(minor-burden, PC1xPC1, kernel-Hadamard),
but the pre-registered replication test is the burden-product GLS. If a candidate's omniB is carried
by the kernel term while its burden term is null, then by this project's own precedent (a rapeseed
hit driven solely by the kernel component was declared NOT_SUPPORTED) it is not a burden-product
candidate at all and must not be carried into a burden-product replication.

Discovery side only; no hebau phenotype is read.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/mnt/7302share/fast_ysp/U7_GWAS")
SCRATCH = Path(__file__).resolve().parent
SD = ROOT / "reproducibility/paper/scripts/phase7"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(SD))
import d2_arm2_synteny_kernel as d2
from homoeogwas.interact import (acat, block_burden_capped, gene_pc_scores,
                                 kernel_interaction_pvals, pairwise_pvals, rank_int)


def _lm(n, r):
    s = importlib.util.spec_from_file_location(n, str(SD / r)); m = importlib.util.module_from_spec(s)
    sys.modules[n] = m; s.loader.exec_module(m); return m


b4 = _lm("b4", "bio_pilot/04b_multi_trait_deploy.py")
CAP, N_PC, MAF = 150, 3, 0.01
LOCI = {
    "L1": ("Gh_A08G064900", "Gh_D08G060500", 6.0872968114522585e-06),
    "L2": ("Gh_A06G029400", "Gh_D06G029100", 2.09709034538208e-05),
    "L3": ("Gh_A09G109300", "Gh_D09G103000", 0.0001549825501095),
    "S1": ("Gh_A08G065900", "Gh_D08G061400", 1.3658265549776338e-05),
    "FibLen_lead": ("Gh_A05G118800", "Gh_D05G131200", 8.98e-06),
    "FibLen_1": ("Gh_A12G285200", "Gh_D12G279400", 5.06e-06),
}

cfg = dict(d2.PANELS["cotton_cottongvd"]); cfg["trait"] = "FibLen"
D = d2._load(cfg)
XA, XD = D["XA"], D["XD"]
y = rank_int(np.asarray(D["y"], float))
Wh, cv = b4._whiten(D["KA"], D["KD"], y, seed=42)
g2s = {}
for p in ("data/processed/cotton/cottongvd/interact/snp_to_gene_A.npz",
          "data/processed/cotton/cottongvd/interact/snp_to_gene_D.npz"):
    g2s.update(b4._load_snp_to_gene(ROOT / p))


def gate(idx, X):
    idx = np.asarray(idx, int)
    mu = np.nanmean(X[:, idx], axis=0) / 2.0
    return idx[np.minimum(mu, 1 - mu) >= MAF]


out = {}
print(f"{'tag':<12}{'selected omniB':>15}{'minor-burden':>14}{'PC1xPC1':>11}{'kernel':>11}"
      f"{'recomputed omniB':>18}  driver")
for tag, (gA, gD, sel_p) in LOCI.items():
    iA, iD = gate(g2s[gA], XA), gate(g2s[gD], XD)
    rng = np.random.default_rng(0)
    bA = block_burden_capped(XA, iA, CAP, rng, minor=True).reshape(-1, 1)
    bD = block_burden_capped(XD, iD, CAP, rng, minor=True).reshape(-1, 1)
    PA = gene_pc_scores(XA, iA, CAP, np.random.default_rng(0), N_PC)
    PD = gene_pc_scores(XD, iD, CAP, np.random.default_rng(0), N_PC)
    p_b = float(pairwise_pvals(Wh, y, bA, bD)[0])
    p_pc = float(pairwise_pvals(Wh, y, PA[:, :1], PD[:, :1])[0])
    p_k = float(kernel_interaction_pvals(Wh, y, [PA], [PD])[0])
    p_omni = float(acat(np.array([p_b, p_pc, p_k])))
    driver = min((("minor_burden", p_b), ("pc1", p_pc), ("kernel", p_k)), key=lambda x: x[1])[0]
    burden_null = p_b > 0.05
    out[tag] = dict(genes=[gA, gD], selected_omniB=sel_p, minor_burden=p_b, pc1=p_pc, kernel=p_k,
                    recomputed_omniB=p_omni, driver=driver, burden_component_null=burden_null,
                    n_snp=[int(iA.size), int(iD.size)])
    print(f"{tag:<12}{sel_p:>15.3e}{p_b:>14.3e}{p_pc:>11.3e}{p_k:>11.3e}{p_omni:>18.3e}  {driver}"
          + ("   <-- burden component NULL" if burden_null else ""))

json.dump(out, open(SCRATCH / "cotton_component_diag.json", "w"), indent=2, default=float)
print("\nwrote cotton_component_diag.json (no hebau phenotype read)")
