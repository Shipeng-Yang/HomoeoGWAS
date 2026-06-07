#!/usr/bin/env python3
"""Phase 2 M2.4.1 — Horvath2020 A+C multi-kernel REML acceptance.

Fits y = X β + u_A + u_C + e on plant_height (default trait).
Consumes M2.1 GRM artifact; trace-normalizes each kernel before REML.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from homoeogwas import fit_multi_reml, normalize_kernel  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trait", default="plant_height")
    args = ap.parse_args()

    grm_npz = ROOT / "results/phase2/m2_1/horvath2020/grm_A_C.npz"
    pheno_tsv = ROOT / "data/processed/rapeseed/horvath/pheno_clean.tsv"
    out_dir = ROOT / "results/phase2/m2_4/horvath2020"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not grm_npz.exists():
        sys.exit(f"ERR: M2.1 GRM artifact missing: {grm_npz}")
    if not pheno_tsv.exists():
        sys.exit(f"ERR: pheno_clean.tsv missing: {pheno_tsv}")

    print(f"loading {grm_npz}")
    npz = np.load(grm_npz, allow_pickle=True)
    G_A = np.asarray(npz["G_A"], dtype=np.float64)
    G_C = np.asarray(npz["G_C"], dtype=np.float64)
    samples = list(np.asarray(npz["samples"]))
    n_kernel = len(samples)
    print(f"  n_samples_kernel={n_kernel}  trace(G_A)={np.trace(G_A):.2f}  trace(G_C)={np.trace(G_C):.2f}")

    # Trace-normalize each kernel to n
    K_A = normalize_kernel(G_A, mode="trace")
    K_C = normalize_kernel(G_C, mode="trace")
    print(f"  after trace-normalize: trace(K_A)={np.trace(K_A):.2f}, trace(K_C)={np.trace(K_C):.2f}")

    # pheno
    pheno = pd.read_csv(pheno_tsv, sep="\t").set_index("sample")
    if args.trait not in pheno.columns:
        sys.exit(f"ERR: trait {args.trait!r} not in pheno; have {list(pheno.columns)}")

    overlap = [s for s in samples if s in pheno.index and not pd.isna(pheno.loc[s, args.trait])]
    n = len(overlap)
    if n < 50:
        sys.exit(f"ERR: too few samples {n} after inner-join")
    idx = {s: i for i, s in enumerate(samples)}
    sel = np.array([idx[s] for s in overlap], dtype=np.int64)
    K_A_sub = K_A[np.ix_(sel, sel)]
    K_C_sub = K_C[np.ix_(sel, sel)]
    y = pheno.loc[overlap, args.trait].astype(np.float64).to_numpy()
    X = np.ones((n, 1))
    print(f"  inner-join n={n} non-NA {args.trait}; var(y)={np.var(y, ddof=1):.3f}")

    print("\nfitting multi-kernel REML: {A, C} ...")
    res = fit_multi_reml(y, X, {"A": K_A_sub, "C": K_C_sub})
    print(f"  optimizer: {res.optimizer_status} ({res.optimizer_message})  n_iter={res.n_iter}")
    print(f"  log_lik = {res.log_lik:.4f}")
    print(f"  σ²: {res.sigma2}")
    print(f"  PVE: { {k: f'{v:.4f}' for k, v in res.pve.items()} }")
    print(f"  boundary components: {res.boundary_components}")
    print(f"  kernel corr(A,C)={res.kernel_corr['A']['C']:.4f}  "
          f"design cond={res.kernel_design_cond:.2e}")

    # acceptance assertions
    assert all(np.isfinite(v) and v >= 0 for v in res.sigma2.values()), "σ² invalid"
    pve_sum = sum(res.pve.values())
    assert abs(pve_sum - 1.0) < 1e-9, f"PVE sum = {pve_sum}, expected 1.0"

    # Outputs
    rows = []
    for k, v in res.sigma2.items():
        rows.append({
            "component": k,
            "sigma2": float(v),
            "pve": float(res.pve[k]),
            "is_boundary": k in res.boundary_components,
        })
    df = pd.DataFrame(rows)
    out_tsv = out_dir / f"multi_reml_A_C_{args.trait}.tsv"
    df.to_csv(out_tsv, sep="\t", index=False)
    print(f"\nwrote {out_tsv}")

    info = {
        "trait": args.trait,
        "n": n,
        "p": 1,
        "var_y": float(np.var(y, ddof=1)),
        "log_lik": res.log_lik,
        "optimizer_status": res.optimizer_status,
        "optimizer_message": res.optimizer_message,
        "n_iter": res.n_iter,
        "sigma2": res.sigma2,
        "pve": res.pve,
        "beta": res.beta.tolist(),
        "boundary_components": res.boundary_components,
        "kernel_corr": res.kernel_corr,
        "kernel_design_cond": res.kernel_design_cond,
        "kernel_names": res.kernel_names,
        "input_grm_npz": str(grm_npz),
    }
    out_json = out_dir / f"multi_reml_A_C_{args.trait}.json"
    with open(out_json, "w") as f:
        json.dump(info, f, indent=2)
    print(f"wrote {out_json}")

    # Bar plot of PVE
    fig, ax = plt.subplots(figsize=(5, 4))
    keys = ["A", "C", "e"]
    pves = [res.pve[k] for k in keys]
    bars = ax.bar(keys, pves, color=["#1f77b4", "#ff7f0e", "#999999"])
    for b, v in zip(bars, pves, strict=True):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}",
                ha="center", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("PVE")
    ax.set_title(f"Multi-kernel REML PVE — Horvath2020 {args.trait} (n={n})")
    fig.tight_layout()
    out_png = out_dir / f"pve_bar_A_C_{args.trait}.png"
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"wrote {out_png}")

    # 2-line summary file
    summary = out_dir / f"summary_{args.trait}.txt"
    with open(summary, "w") as f:
        f.write(f"trait={args.trait} n={n} log_lik={res.log_lik:.4f}\n")
        f.write(f"PVE: A={res.pve['A']:.4f} C={res.pve['C']:.4f} e={res.pve['e']:.4f}\n")
        f.write(f"kernel_corr(A,C)={res.kernel_corr['A']['C']:.4f}\n")
        f.write(f"boundary={res.boundary_components}\n")

    print("\n✅ M2.4.1 acceptance PASS")


if __name__ == "__main__":
    main()
