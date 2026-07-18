#!/usr/bin/env python3
"""Frozen expression-product test of the cotton elongation homoeolog pair (prereg §5).

Runs AFTER salmon quantification of the 20 DPA samples. Applies exactly the model frozen in
PREREGISTRATION_cotton_expression_product.md (git c885124, committed before the RNA-seq was
downloaded):

    FE ~ 1 + ExprA + ExprD + ExprA:ExprD + (expression PCs 1..5)

Success (CORROBORATED) = the interaction coefficient is significant AND neither single-copy main
effect is, i.e. the molecular signal is pair-only, mirroring the genetic result. Reported whatever
the outcome (§8).
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

ROOT = "/mnt/7302share/fast_ysp/U7_GWAS"
QUANT = "/mnt/lldata/cotton_multiomic_376/salmon_quant"
LEAD = {"FE": ("Ghir_A01G002400", "Ghir_D01G002310"),
        "FL": ("Ghir_A05G013600", "Ghir_D05G013340")}
N_PC = 5
ALPHA = 0.05          # primary FE pair; family-of-2 Bonferroni 0.025 also reported (§6)


def _int(x):
    x = np.asarray(x, float)
    r = stats.rankdata(x)
    return stats.norm.ppf((r - 0.5) / len(r))


def load_matrix():
    """Full gene-level TPM matrix (accession x gene) from all quant.sf, summing isoforms to gene."""
    accs, per = [], {}
    for d in sorted(glob.glob(QUANT + "/S*")):
        sf = d + "/quant.sf"
        if not os.path.exists(sf):
            continue
        acc = os.path.basename(d)
        t = pd.read_csv(sf, sep="\t", usecols=["Name", "TPM"])
        t["gene"] = t["Name"].str.split(".").str[0]
        g = t.groupby("gene")["TPM"].sum()
        per[acc] = g
        accs.append(acc)
    M = pd.DataFrame(per).T          # accession x gene
    print(f"expression matrix: {M.shape[0]} accessions x {M.shape[1]} genes")
    return M


def main():
    M = load_matrix()
    # expression PCs from moderately-expressed genes (log1p), to absorb batch/structure (§5)
    expr = M.loc[:, (M > 1).mean(axis=0) >= 0.5]
    L = np.log1p(expr.to_numpy())
    L = (L - L.mean(0)) / (L.std(0) + 1e-9)
    U, S, Vt = np.linalg.svd(L - L.mean(0), full_matrices=False)
    PC = U[:, :N_PC] * S[:N_PC]
    pc = pd.DataFrame(PC, index=M.index, columns=[f"PC{i+1}" for i in range(N_PC)])

    # phenotype: SI Table 13 (FE, FL) keyed by S### accession
    si = pd.read_excel(f"{ROOT}/results/phase7/cotton_multiomic/MOESM4.xlsx",
                       sheet_name="Supp Table 13", header=1)
    si = si.rename(columns={si.columns[0]: "accession"}).set_index("accession")

    res = {}
    for trait, (gA, gD) in LEAD.items():
        pcol = "FE" if trait == "FE" else "FL"
        df = pc.copy()
        df["exprA"] = _int(M[gA].reindex(df.index).fillna(0).to_numpy())
        df["exprD"] = _int(M[gD].reindex(df.index).fillna(0).to_numpy())
        df[pcol] = si[pcol].reindex(df.index)
        df = df.dropna(subset=[pcol])
        y = _int(df[pcol].to_numpy())
        n = len(df)
        X = np.column_stack([np.ones(n), df["exprA"], df["exprD"],
                             (df["exprA"] * df["exprD"]).to_numpy(),
                             df[[f"PC{i+1}" for i in range(N_PC)]].to_numpy()])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        dof = n - X.shape[1]
        s2 = float(resid @ resid) / dof
        cov = s2 * np.linalg.pinv(X.T @ X)
        se = np.sqrt(np.diag(cov))
        t = beta / se
        p = 2 * stats.t.sf(np.abs(t), dof)
        # columns: 0 intercept, 1 exprA(main), 2 exprD(main), 3 interaction
        entry = dict(n=int(n), gene_A=gA, gene_D=gD,
                     beta_interaction=float(beta[3]), p_interaction=float(p[3]),
                     p_mainA=float(p[1]), p_mainD=float(p[2]),
                     beta_mainA=float(beta[1]), beta_mainD=float(beta[2]))
        pair_only = p[3] < ALPHA and p[1] >= ALPHA and p[2] >= ALPHA
        if p[3] < ALPHA and pair_only:
            entry["verdict"] = "CORROBORATED (pair-only: interaction sig, single copies not)"
        elif p[3] < ALPHA:
            entry["verdict"] = "INTERACTION+MAIN (interaction sig but a single copy also sig)"
        elif p[1] < ALPHA or p[2] < ALPHA:
            entry["verdict"] = "MARGINAL-ONLY (single-copy main effect, no pair signal)"
        else:
            entry["verdict"] = "NOT CORROBORATED (no interaction; power-limited -> treat as INCONCLUSIVE)"
        res[trait] = entry

    import json
    out = dict(prereg_blob="c885124", stage="20DPA", n_pc=N_PC, alpha=ALPHA,
               alpha_family2=0.025, results=res)
    open(f"{ROOT}/results/phase7/cotton_multiomic/expr_product_verdict.json", "w").write(
        json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
