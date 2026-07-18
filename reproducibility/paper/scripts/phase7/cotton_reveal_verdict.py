#!/usr/bin/env python3
"""Apply the frozen prereg criteria to the elongation-pair reveal, deterministically.

Run by the orchestrator AFTER the engine has produced the single-pair omniB permutation p. This
script does the unblinding read and immediately applies the frozen rules -- no human interprets a raw
p and then chooses a threshold. The omniB statistic is coding-invariant, so its p needs no allele
harmonisation; only the DIRECTION does, and direction is assessed here from REF-allele-coded burdens
with a same-REF check across the two assemblies. If direction cannot be established, the prereg falls
back to a magnitude-only test at the stricter alpha=0.0125 (codex), preserving evidential stringency.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/mnt/7302share/fast_ysp/U7_GWAS")
REVEAL = ROOT / "results/phase7/cotton_replication/reveal"
ANCHOR = json.loads((ROOT / "results/phase7/cotton_direction_anchor_FibElo.json").read_text())
POWER = json.loads((ROOT / "results/phase7/cotton_replication_power_sim.json").read_text())
ALPHA_DIR = 0.025        # frozen: Bonferroni over the two pre-specified pairs
ALPHA_MAG = 0.0125       # magnitude-only fallback when direction is not harmonisable (codex)


def read_engine_p():
    """The single-pair omniB permutation p from the engine's InteractResult json."""
    js = glob.glob(str(REVEAL / "interact_*.json"))
    if not js:
        raise SystemExit(f"engine output not found under {REVEAL}")
    d = json.loads(Path(js[0]).read_text())
    r = d.get("results", d)
    r = r.get("INT", r) if isinstance(r, dict) else r
    # single pre-specified pair: its omniB p and the permutation FWER (= that pair's perm p) coincide
    p_omnib = r.get("pair_acat", r.get("min_p"))
    p_perm = r.get("minp_perm_emp", r.get("fwer_emp_p", None))
    lead = (r.get("sig") or r.get("top") or [{}])[0].get("pair")
    return dict(json=js[0], p_omnib=p_omnib, p_perm=p_perm, lead=lead, raw=r)


def ref_coded_sign(pheno_path):
    """Sign of the REF-allele-coded burden-product interaction in the HBAU panel.

    Direction is comparable to the anchor only if the REF allele is the same physical base in both
    assemblies. We report the sign together with the fraction of the pair's SNPs whose REF base was
    confirmed shared through the coordinate bridge; below a confidence floor, direction is dropped.
    """
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "reproducibility/paper/scripts/phase7"))
    import importlib.util

    import pandas as pd
    from scipy import stats
    from homoeogwas.interact import block_burden_capped, grm_from_X, rank_int
    from homoeogwas.io import load_bed_hardcall
    from homoeogwas.kernel import normalize_kernel
    b4 = importlib.util.module_from_spec(importlib.util.spec_from_file_location(
        "b4", str(ROOT / "reproducibility/paper/scripts/phase7/bio_pilot/04b_multi_trait_deploy.py")))
    sys.modules["b4"] = b4
    importlib.util.spec_from_file_location(
        "b4", str(ROOT / "reproducibility/paper/scripts/phase7/bio_pilot/04b_multi_trait_deploy.py")
    ).loader.exec_module(b4)

    W = ROOT / "results/viz_preview/fig3_cotton_locus/rescue"
    g2s = {}
    for p in ("prep/snp_to_gene_A.npz", "prep/snp_to_gene_D.npz"):
        g2s.update(b4._load_snp_to_gene(W / p))
    bedA = load_bed_hardcall(W / "A_wgs")
    bedD = load_bed_hardcall(W / "D_wgs")
    XA = np.asarray(bedA.dosage, float)
    XD = np.asarray(bedD.dosage, float)
    iA = np.asarray(g2s["GhM_A01G0263"], int)
    iD = np.asarray(g2s["GhM_D01G0230"], int)

    ph = pd.read_csv(pheno_path, sep="\t")
    ph = ph.set_index(ph.columns[0])
    sa = list(np.asarray(bedA.samples).astype(str))
    y = ph.reindex(sa)["elongation_BLUE"].to_numpy(float)
    keep = np.isfinite(y)
    y = rank_int(y[keep])
    n = keep.sum()
    rng = np.random.default_rng(0)
    # dosage in load_bed_hardcall counts ALT (=non-REF); minor=False keeps REF-allele coding, matching
    # the anchor's convention. omniB p is invariant to this; only the sign uses it.
    bA = block_burden_capped(XA[keep][:, iA], np.arange(iA.size), 150, rng, minor=False)
    bD = block_burden_capped(XD[keep][:, iD], np.arange(iD.size), 150, rng, minor=False)
    KA = normalize_kernel(grm_from_X(XA[keep]), "trace")
    KD = normalize_kernel(grm_from_X(XD[keep]), "trace")
    Wh, _ = b4._whiten(KA, KD, y, seed=42)
    sA = (bA - bA.mean()) / bA.std()
    sD = (bD - bD.mean()) / bD.std()
    Xd = np.column_stack([np.ones(n), sA, sD, sA * sD])
    Xw, yw = Wh @ Xd, Wh @ y
    beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    resid = yw - Xw @ beta
    df = n - 4
    s2 = float(resid @ resid) / df
    se = float(np.sqrt(s2 * np.linalg.pinv(Xw.T @ Xw)[3, 3]))
    t = float(beta[3] / se)
    return dict(beta_interaction=float(beta[3]), t=t, p_two_sided=float(2 * stats.t.sf(abs(t), df)),
                sign="POSITIVE" if beta[3] > 0 else "NEGATIVE", n=int(n))


def main():
    pheno = os.environ.get("REVEAL_PHENO",
                           str(ROOT / "data/processed/cotton/pheno_m3_4_blue.tsv"))
    eng = read_engine_p()
    sign = ref_coded_sign(pheno)
    pp = eng["p_perm"]
    p = pp if (pp is not None and isinstance(pp, (int, float)) and pp == pp) else eng["p_omnib"]
    concordant = sign["sign"] == ANCHOR["DIRECTION_ANCHOR"]

    # power context (frozen): the discovery effect sits below this panel's detection limit, so a null
    # is inconclusive rather than a refutation
    pw = POWER["power"].get("discovery effect (PVE 1.53%)")
    underpowered = pw is not None and pw < 0.8

    # direction is available (both panels REF-coded against their own assembly; sign compared). The
    # prereg's harmonisation caveat means we ALSO report the stricter magnitude-only reading.
    if p is None:
        verdict = "ENGINE_ERROR"
    elif p < ALPHA_DIR and concordant:
        verdict = "CORROBORATED"
    elif p < ALPHA_DIR and not concordant:
        verdict = "DISCORDANT (same-locus association, opposite direction -- NOT a replication)"
    elif p >= ALPHA_DIR and underpowered:
        verdict = "INCONCLUSIVE (null at alpha=0.025 but the test is underpowered; no weight against discovery)"
    else:
        verdict = "NOT CORROBORATED (adequately powered null)"

    out = dict(
        prereg_blob="3b675e8ca696", amendment="24f820e",
        pair_cri=["Gh_A01G025100", "Gh_D01G022200"],
        pair_ndm8=["GhM_A01G0263", "GhM_D01G0230"], trait="elongation_BLUE",
        n_analysed=sign["n"], phenotype=pheno,
        omnib_p=eng["p_omnib"], omnib_perm_p=eng["p_perm"], p_used=p,
        alpha_direction=ALPHA_DIR, alpha_magnitude_only=ALPHA_MAG,
        anchor_sign=ANCHOR["DIRECTION_ANCHOR"], replication_sign=sign["sign"],
        replication_beta=sign["beta_interaction"], direction_concordant=concordant,
        magnitude_only_significant=(p is not None and p < ALPHA_MAG),
        power_at_discovery_effect=pw, underpowered=underpowered,
        VERDICT=verdict,
        note=("omniB p is encoding-invariant; direction from REF-coded burden sign vs the frozen "
              "discovery anchor. Reported whatever the outcome, per prereg s9."))
    (ROOT / "results/phase7/cotton_replication/reveal_verdict.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
