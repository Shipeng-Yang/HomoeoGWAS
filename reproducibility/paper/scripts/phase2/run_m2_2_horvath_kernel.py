#!/usr/bin/env python3
"""Phase 2 M2.2 — Horvath2020 A+C homoeolog Hadamard kernel acceptance.

Consumes M2.1 GRM artifact (no recompute) → builds K_hom + K_sum + heatmap.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/mnt/7302share/fast_ysp/U7_GWAS")
sys.path.insert(0, str(ROOT / "src"))

from homoeogwas import hadamard_kernel, normalize_kernel, sum_kernel  # noqa: E402


def main() -> None:
    in_npz = ROOT / "results/phase2/m2_1/horvath2020/grm_A_C.npz"
    if not in_npz.exists():
        sys.exit(f"ERR: M2.1 artifact missing: {in_npz}\n"
                 f"   Run scripts/phase2/run_m2_1_horvath_grm.py first (M2.1).")

    out_dir = ROOT / "results/phase2/m2_2/horvath2020"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading {in_npz}")
    npz = np.load(in_npz, allow_pickle=True)
    G_A = np.asarray(npz["G_A"], dtype=np.float64)
    G_C = np.asarray(npz["G_C"], dtype=np.float64)
    samples = np.asarray(npz["samples"])
    n = G_A.shape[0]
    print(f"  n_samples={n}  G_A trace={np.trace(G_A):.4f}  G_C trace={np.trace(G_C):.4f}")

    print("computing Hadamard kernel K_hom = G_A ⊙ G_C ...")
    K_hom_raw = hadamard_kernel({"A": G_A, "C": G_C})
    K_hom = normalize_kernel(K_hom_raw, mode="trace")
    eig_hom_min = float(np.linalg.eigvalsh(K_hom).min())

    print("computing additive sum kernel K_sum = G_A + G_C ...")
    K_sum_raw = sum_kernel({"A": G_A, "C": G_C})
    K_sum = normalize_kernel(K_sum_raw, mode="trace")
    eig_sum_min = float(np.linalg.eigvalsh(K_sum).min())

    info = {
        "n_samples": int(n),
        "input_npz": str(in_npz),
        "trace_G_A": float(np.trace(G_A)),
        "trace_G_C": float(np.trace(G_C)),
        "trace_K_hom_raw": float(np.trace(K_hom_raw)),
        "trace_K_hom_AC": float(np.trace(K_hom)),
        "trace_K_sum_AC": float(np.trace(K_sum)),
        "min_eig_K_hom_AC": eig_hom_min,
        "min_eig_K_sum_AC": eig_sum_min,
    }
    print("\n=== kernel_info ===")
    print(json.dumps(info, indent=2))

    # save npz + json
    out_npz = out_dir / "K_hom_AC.npz"
    np.savez_compressed(out_npz,
                        K_hom_AC=K_hom, K_hom_AC_raw=K_hom_raw,
                        K_sum_AC=K_sum, samples=samples)
    print(f"\nwrote {out_npz}")

    with open(out_dir / "kernel_info.json", "w") as f:
        json.dump(info, f, indent=2)
    print(f"wrote {out_dir / 'kernel_info.json'}")

    # heatmap (4-panel)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    panels = [("G_A", G_A), ("G_C", G_C),
              ("K_sum_AC (norm trace)", K_sum),
              ("K_hom_AC (norm trace)", K_hom)]
    for ax, (title, M) in zip(axes, panels, strict=True):
        # 共享色阶但裁极值
        vmin = float(np.percentile(M, 1))
        vmax = float(np.percentile(M, 99))
        im = ax.imshow(M, cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(f"{title}\n(trace={np.trace(M):.2f}, n={n})", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Horvath2020 rapeseed A/C subgenome kernels (M2.2)", fontsize=12)
    fig.tight_layout()
    heatmap = out_dir / "kernel_heatmap_AC.png"
    fig.savefig(heatmap, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"wrote {heatmap}")

    # final acceptance assertions
    assert abs(np.trace(K_hom) - n) < 1e-6, f"K_hom trace != n: {np.trace(K_hom)}"
    assert abs(np.trace(K_sum) - n) < 1e-6, f"K_sum trace != n: {np.trace(K_sum)}"
    assert eig_hom_min > -1e-6, f"K_hom not PSD: min eig={eig_hom_min}"
    assert eig_sum_min > -1e-6, f"K_sum not PSD: min eig={eig_sum_min}"
    print("\n✅ M2.2 acceptance PASS")


if __name__ == "__main__":
    main()
