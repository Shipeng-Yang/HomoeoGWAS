#!/usr/bin/env python3
"""Phase 2 M2.3 — Horvath2020 single-kernel REML acceptance.

Fits y = X β + u + e for plant_height (or another trait) with
  - K_hom_AC (homoeolog Hadamard kernel)
  - K_sum_AC (additive sum kernel)
and reports σ_g², σ_e², h², log-lik, eigen condition.

Reuses M2.2 kernel artifact; no recompute.
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

from homoeogwas import fit_reml  # noqa: E402
from homoeogwas.lmm import _neg_log_reml  # noqa: E402  (after sys.path + mpl backend)


def _project_eigenbasis(K, y, X):
    K_sym = 0.5 * (K + K.T)
    lam, U = np.linalg.eigh(K_sym)
    lam = np.clip(lam, 0.0, None)
    return lam, U.T @ y, U.T @ X


def _plot_reml_profile(K, y, X, log_delta_grid, log_delta_hat, log_lik_hat, out_path, title):
    lam, y_t, X_t = _project_eigenbasis(K, y, X)
    n, p = X.shape[0], X.shape[1]
    profile = np.array([-_neg_log_reml(ld, lam, y_t, X_t, n, p) for ld in log_delta_grid])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(log_delta_grid, profile, lw=1.5)
    ax.axvline(log_delta_hat, color="red", lw=0.8, ls="--",
               label=f"argmin log_δ={log_delta_hat:.3f}")
    ax.scatter([log_delta_hat], [log_lik_hat], color="red", s=30, zorder=5)
    ax.set_xlabel("log(σ_e²/σ_g²)")
    ax.set_ylabel("REML log-lik (up to const)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trait", default="plant_height")
    ap.add_argument("--backend", default="auto", choices=["cpu", "gpu", "auto"])
    ap.add_argument("--cpu-check", action="store_true",
                    help="If GPU available, also run CPU and compare")
    args = ap.parse_args()

    k_npz = ROOT / "results/phase2/m2_2/horvath2020/K_hom_AC.npz"
    pheno_tsv = ROOT / "data/processed/rapeseed/horvath/pheno_clean.tsv"
    out_dir = ROOT / "results/phase2/m2_3/horvath2020"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not k_npz.exists():
        sys.exit(f"ERR: M2.2 kernel artifact missing: {k_npz}")
    if not pheno_tsv.exists():
        sys.exit(f"ERR: pheno_clean.tsv missing: {pheno_tsv}")

    print(f"loading {k_npz}")
    # samples is an object array from M2.1; need allow_pickle until that artifact
    # is regenerated with str dtype. Numeric kernels are safe with allow_pickle=False
    # but mixed dtypes in the same .npz force the file-level flag.
    kernels = np.load(k_npz, allow_pickle=True)
    K_hom = np.asarray(kernels["K_hom_AC"], dtype=np.float64)
    K_sum = np.asarray(kernels["K_sum_AC"], dtype=np.float64)
    samples = np.asarray(kernels["samples"])
    print(f"  n_samples in kernel = {len(samples)}")

    pheno = pd.read_csv(pheno_tsv, sep="\t")
    if args.trait not in pheno.columns:
        sys.exit(f"ERR: trait '{args.trait}' not in pheno columns: {pheno.columns.tolist()}")

    # Inner-join (case-insensitive on lower; pheno sample IDs already lower)
    pheno = pheno.set_index("sample")
    kernel_iids = [str(s) for s in samples]
    overlap_iids = [s for s in kernel_iids if s in pheno.index and not pd.isna(pheno.loc[s, args.trait])]
    n_in = len(overlap_iids)
    print(f"  inner-join {len(kernel_iids)} kernel samples × {pheno[args.trait].notna().sum()} non-NA pheno = {n_in}")
    if n_in < 50:
        sys.exit(f"ERR: too few overlapping samples ({n_in}) for stable REML")

    # Index lookup in kernel
    kernel_idx = {s: i for i, s in enumerate(kernel_iids)}
    sel = np.array([kernel_idx[s] for s in overlap_iids], dtype=np.int64)

    K_hom_sub = K_hom[np.ix_(sel, sel)]
    K_sum_sub = K_sum[np.ix_(sel, sel)]
    y = pheno.loc[overlap_iids, args.trait].astype(np.float64).to_numpy()
    X = np.ones((n_in, 1))  # intercept only

    print(f"\nfitting REML for trait={args.trait}, n={n_in}, p=1, backend={args.backend} ...")
    fits = {}
    for name, K in [("K_hom_AC", K_hom_sub), ("K_sum_AC", K_sum_sub)]:
        print(f"\n--- fit {name} ---")
        res = fit_reml(y, X, K, backend=args.backend)
        print(f"  backend_used={res.backend_used}")
        print(f"  log_lik={res.log_lik:.4f}  log_δ={res.log_delta:.4f}")
        print(f"  σ_g²={res.sigma_g2:.4f}  σ_e²={res.sigma_e2:.4f}  h²={res.h2:.4f}")
        print(f"  min_eig(K)={res.min_eig:.4g}  n_eig_clipped={res.n_eig_clipped}")
        # 0 ≤ h² ≤ 1 acceptance
        assert 0.0 <= res.h2 <= 1.0, f"{name} h²={res.h2} out of [0,1]"
        assert res.sigma_e2 >= 0, f"{name} σ_e² negative"
        fits[name] = res

        # Optional CPU check
        if args.cpu_check and res.backend_used == "gpu":
            res_cpu = fit_reml(y, X, K, backend="cpu")
            d_ll = abs(res_cpu.log_lik - res.log_lik)
            d_h2 = abs(res_cpu.h2 - res.h2)
            print(f"  CPU check: |Δlog_lik|={d_ll:.4g} |Δh²|={d_h2:.4g}")
            assert d_ll < 1e-3 and d_h2 < 1e-3, "CPU/GPU divergence"

        # Profile plot
        grid = np.linspace(res.log_delta_bounds[0], res.log_delta_bounds[1], 80)
        _plot_reml_profile(K, y, X, grid, res.log_delta, res.log_lik,
                           out_dir / f"reml_profile_{name}.png",
                           f"{name} REML profile ({args.trait})")

    # Save .tsv (one row per kernel)
    rows = []
    for name, r in fits.items():
        rows.append({
            "kernel": name, "trait": args.trait, "n": r.n, "p": r.p,
            "backend_used": r.backend_used, "log_lik": r.log_lik,
            "sigma_g2": r.sigma_g2, "sigma_e2": r.sigma_e2, "h2": r.h2,
            "log_delta": r.log_delta, "min_eig": r.min_eig,
            "n_eig_clipped": r.n_eig_clipped,
            "optimizer_status": r.optimizer_status,
        })
    df = pd.DataFrame(rows)
    fit_tsv = out_dir / "reml_fit.tsv"
    df.to_csv(fit_tsv, sep="\t", index=False)
    print(f"\nwrote {fit_tsv}")

    # Save info json
    info = {
        "trait": args.trait,
        "n_overlap_samples": n_in,
        "input_kernel_npz": str(k_npz),
        "input_pheno_tsv": str(pheno_tsv),
        "fits": {name: r.to_dict() for name, r in fits.items()},
    }
    with open(out_dir / "reml_info.json", "w") as f:
        json.dump(info, f, indent=2, default=str)
    print(f"wrote {out_dir / 'reml_info.json'}")
    print(f"\n✅ M2.3 acceptance PASS for trait={args.trait}")


if __name__ == "__main__":
    main()
