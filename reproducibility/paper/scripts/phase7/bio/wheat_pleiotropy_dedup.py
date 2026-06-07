#!/usr/bin/env python3
"""Codex PASS_WITH_FIX recompute: emergence === heading_date (days_to_emerg bit-identical to
Hd_dto_days, verified max_abs_diff=0). The original pleiotropy PHENOLOGY cluster double-counted the
SAME phenology phenotype (both emergence and heading_date). Fold them: PHENOLOGY keeps heading_date
only; emergence dropped (it is the anchor and is the identical record). Recompute all clusters fresh
from observed per-trait triad-ACAT vectors so no evidence is double-weighted. Cheap: observed scans
are ~16 s/trait; the build (plink2 x3) is the only slow part. No new traits/triads selected.
"""
from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import importlib.util  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path("/mnt/7302share/fast_ysp/U7_GWAS")
OUT = ROOT / "results/phase7/bio_wheat/multitrait"
_spec = importlib.util.spec_from_file_location("mtp", str(ROOT / "scripts/phase7/bio/wheat_multitrait_perm.py"))
M = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(M)

# DEDUPLICATED clusters: emergence is folded into heading_date (identical record); it is NOT an
# independent phenology trait. The 14 multitrait columns are already unique (heading_date once).
CLUSTERS = {
    "PHENOLOGY": ["heading_date", "plant_height", "flag_leaf_sen", "stem_sen"],
    "GRAIN_SIZE": ["grain_surfarea", "grain_width", "grain_length", "grain_diameter", "grain_weight_1000"],
    "GRAIN_QUALITY": ["grain_hardness_pct", "grain_moisture", "grain_protein_NIR", "grain_starch_NIR", "grain_fibre_NIR"],
}


def main():
    grms, Bfull, kept, common = M._build()
    G = len(kept)
    mt = pd.read_csv(M.PHENO_MT, sep="\t").set_index("sample")
    traits14 = list(mt.columns)

    tri = {}
    for t in traits14:
        ser = mt[t]
        valid = [s for s in common if s in ser.index and pd.notna(ser.loc[s])]
        if len(valid) < M.MIN_N:
            continue
        idx = np.array([common.index(s) for s in valid])
        _pw, tvec, *_ = M._observed(np.array([float(ser.loc[s]) for s in valid]), idx, grms, Bfull, common)
        tri[t] = tvec
        print(f"  [obs {t}] minACAT={tvec.min():.3g}")

    alpha = 0.05 / (G * 3)  # 3 primary frozen clusters
    out = dict(task="wheat_pleiotropy_dedup", note="emergence===heading_date folded; PHENOLOGY uses "
               "heading_date only (no double-count). alpha=0.05/(G*3), 3 primary clusters; ALL_14=sensitivity.",
               G=G, alpha_primary=alpha, clusters=[])
    for cname, members in {**CLUSTERS, "ALL_14": traits14}.items():
        present = [m for m in members if m in tri]
        stk = np.column_stack([tri[m] for m in present])
        ctri = np.array([M._w2._acat(list(stk[i])) for i in range(G)])
        lead = int(np.argmin(ctri))
        is_main = cname in CLUSTERS
        rec = dict(cluster=cname, n_traits=len(present), traits=present,
                   role=("primary" if is_main else "sensitivity"),
                   min_cluster_acat=float(ctri.min()), lead_triad="|".join(kept[lead]),
                   alpha=(alpha if is_main else None),
                   passes=bool(is_main and ctri.min() < alpha),
                   single_trait_floor=float(min(tri[m].min() for m in present)))
        out["clusters"].append(rec)
        print(f"  [{cname}] n={len(present)} min_cluster={ctri.min():.3g} "
              f"single_best={rec['single_trait_floor']:.3g} "
              f"{'PASS' if rec['passes'] else 'no-pass'} (alpha={alpha:.2e})")

    (OUT / "wheat_pleiotropy_dedup.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"\n  wrote {OUT}/wheat_pleiotropy_dedup.json")


if __name__ == "__main__":
    main()
