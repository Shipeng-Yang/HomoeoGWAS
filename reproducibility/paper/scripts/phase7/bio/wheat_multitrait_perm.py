#!/usr/bin/env python3
"""Wheat landrace multi-trait DISCOVER-MORE permutation follow-up (pre-registered, LOCKED 2026-06-05:
reports/multitrait_prereg.md + memory multitrait_expansion). Consumes the observed-only scan
(wheat_multitrait_scan.py) and adds the locked permutation/pleiotropy layer. Design is frozen — this
script does NOT select traits/triads on their permutation result.

Steps (LOCKED):
(1) ANCHOR  : days_to_emergence through the SAME pipeline -> chr1 B-D pairwise must reproduce ~2.4e-6.
(2) TARGETED: per-triad y-shuffle empirical p for the lead triad of EVERY candidate trait
              (min triad-ACAT < 1e-4) -> calibrates the asymptotic p. Reuses deploy() with a G=1
              Bdict subset (deploy.triad_acat_emp == that triad's ACAT empirical p; dry-run verified).
    GENOME-WIDE minP perm (FWER) ONLY for emergence + heading_date + grain_width: each perm re-whitens
              the shuffled y, recomputes all G pairwise p, takes the genome-wide MIN per-triad ACAT;
              empirical FWER p = (1+#{perm_min <= obs_min})/(B+1). min pairwise p reported as secondary.
(3) PLEIOTROPY (frozen clusters, declared before looking): PHENOLOGY / GRAIN_SIZE / GRAIN_QUALITY,
              per-triad ACAT-across-traits of the per-trait triad-ACATs; min over G vs 0.05/(G*3).
              ALL_14 omnibus = SENSITIVITY only (not in the 3-cluster Bonferroni family).
(4) PER-ENV : emergence (CFLN06/10/14/20) + heading_date/grain_width per sheet -> SENSITIVITY only,
              observed p, no discovery claim.

HONESTY: permutation empirical p has a finite-resolution floor 1/(B+1); it
confirms the asymptotic p sits in the extreme tail / the genome-wide FWER-corrected p, it does NOT
re-estimate a 1e-6 asymptotic p. The genome-wide minP perm targets the FWER-adjusted p (~0.01-0.03 for
the Bonferroni hits), which B_gw resolves.
"""
from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import importlib.util  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

ROOT = Path("/mnt/7302share/fast_ysp/U7_GWAS")
LAND = ROOT / "data/raw/wheat/phenotype/Natural_Populations/Watkins_Collection_WGIN_WISP_DFW_watseq_phenotype_data_JIC.xlsx"
PHENO_CLEAN = ROOT / "data/processed/wheat/pheno_clean.tsv"
PHENO_MT = ROOT / "data/processed/wheat/pheno_landrace_multitrait.tsv"
HCTRIADS = ROOT / "data/raw/expression/wheat/HCTriads.csv"
OUT = ROOT / "results/phase7/bio_wheat/multitrait"
SHEETS = {"CFLN06": "WGIN_Watkins_JIC_CFLN06", "CFLN10": "WISP_Watkins_JIC_CFLN10",
          "CFLN14": "DFW_Watkins_JI_CFLN14", "CFLN20": "DFW_Watkins_JI_CFLN20"}
TRAIT_PAT = {"heading_date": r"Hd_dto", "grain_width": r"^GWid"}  # for per-env sensitivity ETL

MIN_N = 500
N_JOBS = 200
B_TARGETED = 5000          # per-triad empirical p (G=1, cheap)
B_GW = 2000               # genome-wide minP FWER (expensive); resolves FWER p ~0.01-0.03 (SE~0.003)
CLUSTERS = {
    "PHENOLOGY": ["emergence", "heading_date", "plant_height", "flag_leaf_sen", "stem_sen"],
    "GRAIN_SIZE": ["grain_surfarea", "grain_width", "grain_length", "grain_diameter", "grain_weight_1000"],
    "GRAIN_QUALITY": ["grain_hardness_pct", "grain_moisture", "grain_protein_NIR", "grain_starch_NIR", "grain_fibre_NIR"],
}
GW_TRAITS = ["emergence", "heading_date", "grain_width"]  # locked: only these get full minP perm

_s4 = importlib.util.spec_from_file_location("w4", str(ROOT / "scripts/phase7/bio_wheat/04w_wheat_genomewide.py"))
_w4 = importlib.util.module_from_spec(_s4)
_s4.loader.exec_module(_w4)
_w2 = _w4._w2
PAIRS = [("A", "B"), ("A", "D"), ("B", "D")]
TAGS = ["AB", "AD", "BD"]


def _gw_rep(KA, KB, KD, Bdict, y_shuf, seed):
    """One genome-wide permutation replicate: re-whiten shuffled y, all-G pairwise p,
    return genome-wide MIN per-triad ACAT and MIN pairwise p. Module-level for joblib pickling."""
    try:
        Wh, _ = _w2._whiten3(KA, KB, KD, y_shuf, seed=seed)
        pw = {f"{x}{z}": _w2._pairwise_pvals(Wh, y_shuf, Bdict[x], Bdict[z]) for x, z in PAIRS}
        G = pw["AB"].size
        tri = np.array([_w2._acat([pw["AB"][i], pw["AD"][i], pw["BD"][i]]) for i in range(G)])
        return dict(ok=True, min_tri=float(np.nanmin(tri)),
                    min_pair=float(min(np.nanmin(p) for p in pw.values())))
    except Exception:  # noqa: BLE001
        return dict(ok=False)


def _build():
    """ONE genotype extraction + per-gene capped burdens + subgenome GRMs + curated 1:1:1 triads."""
    rng = np.random.default_rng(7)
    burdens, grms, samples_ref = {}, {}, None
    for s in ("A", "B", "D"):
        genes = _w4._parse_gff(s, 2000)
        g2b, _nsnp, K, samp = _w4._extract_and_burden(s, genes, "flank", 2000, 150, 3, rng)
        burdens[s], grms[s] = g2b, K
        if samples_ref is None:
            samples_ref = samp
        assert samp == samples_ref
        print(f"  {s}: {len(g2b)} retained genes")
    hc = pd.read_csv(HCTRIADS)
    hc = hc[hc["cardinality_abs"] == "1:1:1"]
    kept = []
    for _, r in hc.iterrows():
        a, b, d = (_w4._v10_to_v11(str(r["A"])), _w4._v10_to_v11(str(r["B"])), _w4._v10_to_v11(str(r["D"])))
        if a in burdens["A"] and b in burdens["B"] and d in burdens["D"]:
            kept.append((a, b, d))
    G = len(kept)
    print(f"  curated 1:1:1 triads with full coverage: {G}")
    Bfull = {s: _w2._scols_safe(np.column_stack([burdens[s][k[i]] for k in kept]))
             for i, s in enumerate(("A", "B", "D"))}
    return grms, Bfull, kept, samples_ref


def _observed(y_raw, idx, grms, Bfull, common):
    """Observed per-trait scan; returns (pw dict of G-vec, tri G-vec, KA,KB,KD, Bd, y_int, n)."""
    n_t = idx.size
    KA = grms["A"][np.ix_(idx, idx)]
    KB = grms["B"][np.ix_(idx, idx)]
    KD = grms["D"][np.ix_(idx, idx)]
    KA, KB, KD = (K / (np.trace(K) / n_t) for K in (KA, KB, KD))
    Bd = {s: _w2._scols_safe(Bfull[s][idx]) for s in ("A", "B", "D")}
    y = _w2._rank_int(y_raw)
    Wh, _cv = _w2._whiten3(KA, KB, KD, y, seed=42)
    pw = {f"{x}{z}": _w2._pairwise_pvals(Wh, y, Bd[x], Bd[z]) for x, z in PAIRS}
    G = pw["AB"].size
    tri = np.array([_w2._acat([pw["AB"][i], pw["AD"][i], pw["BD"][i]]) for i in range(G)])
    return pw, tri, KA, KB, KD, Bd, y, n_t


def _per_env_etl(geno):
    """Per-sheet (env) WATDE-aligned values for heading_date & grain_width (sensitivity only)."""
    xl = pd.ExcelFile(LAND)
    out = {t: {} for t in TRAIT_PAT}  # trait -> {env -> {WATDE: mean}}
    for env, sh in SHEETS.items():
        df = xl.parse(sh)
        sc = next((c for c in df.columns if "StoreCode" in str(c)), None)
        if sc is None:
            continue
        codes = df[sc].astype(str).str.strip()
        for t, pat in TRAIT_PAT.items():
            cols = [c for c in df.columns if re.search(pat, str(c)) and "Rep" not in str(c)
                    and "date_ymd" not in str(c) and not str(c).startswith(("Min.", "Max."))]
            if not cols:
                continue
            acc = {}
            for c in cols:
                vals = pd.to_numeric(df[c], errors="coerce")
                for code, v in zip(codes, vals, strict=True):
                    if code in geno and np.isfinite(v):
                        acc.setdefault(code, []).append(float(v))
            out[t][env] = {k: float(np.mean(vs)) for k, vs in acc.items()}
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print("=== wheat multi-trait DISCOVER-MORE permutation follow-up (LOCKED plan) ===")
    grms, Bfull, kept, common = _build()
    G = len(kept)

    # ---- phenotypes: 14 cross-env (multitrait tsv) + emergence (pheno_clean) ----
    mt = pd.read_csv(PHENO_MT, sep="\t").set_index("sample")
    pc = pd.read_csv(PHENO_CLEAN, sep="\t").set_index("sample")
    traits14 = [c for c in mt.columns]
    pheno = {t: mt[t] for t in traits14}
    pheno["emergence"] = pc["days_to_emerg"]

    def idx_for(series):
        valid = [s for s in common if s in series.index and pd.notna(series.loc[s])]
        return np.array([common.index(s) for s in valid]), valid

    report = dict(task="wheat_multitrait_perm", prereg="reports/multitrait_prereg.md",
                  G=G, B_targeted=B_TARGETED, B_gw=B_GW, n_jobs=N_JOBS)

    # ---- observed scan for all needed traits (14 + emergence); store tri vectors for pleiotropy ----
    obs = {}
    for t in [*traits14, "emergence"]:
        idx, valid = idx_for(pheno[t])
        if idx.size < MIN_N:
            print(f"  [{t}] SKIP n={idx.size}")
            continue
        pw, tri, KA, KB, KD, Bd, y, n_t = _observed(np.array([float(pheno[t].loc[s]) for s in valid]),
                                                    idx, grms, Bfull, common)
        obs[t] = dict(pw=pw, tri=tri, KA=KA, KB=KB, KD=KD, Bd=Bd, y=y, n=n_t, idx=idx, valid=valid)
        lead = int(np.argmin(tri))
        print(f"  [obs {t}] n={n_t} minACAT={tri.min():.3g} lead={kept[lead]} ({time.time()-t0:.0f}s)")

    # ---- (1) ANCHOR: emergence chr1 B-D ----
    e = obs["emergence"]
    chr1_idx = [i for i, k in enumerate(kept) if k[0].startswith("TraesCS1A")]
    bd_chr1 = [(i, e["pw"]["BD"][i]) for i in chr1_idx]
    bd_chr1.sort(key=lambda x: x[1])
    best_i, best_bd = bd_chr1[0]
    report["anchor_emergence"] = dict(
        n=e["n"], min_triad_acat=float(e["tri"].min()), lead_triad="|".join(kept[int(np.argmin(e["tri"]))]),
        chr1_best_BD_pairwise_p=float(best_bd), chr1_best_BD_triad="|".join(kept[best_i]),
        expected_chr1_BD="~2.4e-6")
    print(f"  ANCHOR emergence: chr1 best B-D pairwise p={best_bd:.3g} triad={kept[best_i]} "
          f"(expect ~2.4e-6); min triad ACAT={e['tri'].min():.3g}")

    # ---- dry-run: verify deploy(G=1).triad_acat_emp degenerates to the lead-triad ACAT ----
    h = obs["heading_date"]
    lead_h = int(np.argmin(h["tri"]))
    Bd1 = {s: h["Bd"][s][:, [lead_h]] for s in ("A", "B", "D")}
    dry = _w2.deploy(h["y"], "dry", h["KA"], h["KB"], h["KD"], Bd1, [{"A": kept[lead_h][0], "B": kept[lead_h][1], "D": kept[lead_h][2]}], 8, 10)
    assert abs(dry["triad_acat_obs"] - h["tri"][lead_h]) < 1e-9, \
        f"deploy(G=1) obs {dry['triad_acat_obs']} != observed {h['tri'][lead_h]}"
    print(f"  DRY-RUN ok: deploy(G=1) obs={dry['triad_acat_obs']:.3g} == scan {h['tri'][lead_h]:.3g}")

    # ---- (2a) TARGETED per-triad empirical p for every candidate (min triad-ACAT<1e-4) ----
    candidates = sorted([t for t in obs if obs[t]["tri"].min() < 1e-4], key=lambda t: obs[t]["tri"].min())
    print(f"  candidates (min ACAT<1e-4): {candidates}")
    targeted = []
    for t in candidates:
        o = obs[t]
        lead = int(np.argmin(o["tri"]))
        Bd1 = {s: o["Bd"][s][:, [lead]] for s in ("A", "B", "D")}
        gt = [{"A": kept[lead][0], "B": kept[lead][1], "D": kept[lead][2]}]
        d = _w2.deploy(o["y"], t, o["KA"], o["KB"], o["KD"], Bd1, gt, N_JOBS, B_TARGETED)
        targeted.append(dict(trait=t, lead_triad="|".join(kept[lead]), n=o["n"],
                             obs_triad_acat=float(o["tri"][lead]), emp_p=float(d["triad_acat_emp"]),
                             emp_floor=1.0 / (B_TARGETED + 1)))
        print(f"  [targeted {t}] obs={o['tri'][lead]:.3g} emp_p={d['triad_acat_emp']:.3g} "
              f"(floor {1.0/(B_TARGETED+1):.1e}) ({time.time()-t0:.0f}s)")
    report["targeted_per_triad"] = targeted

    # ---- (2b) GENOME-WIDE minP FWER perm: emergence + heading_date + grain_width ----
    gw = []
    for t in GW_TRAITS:
        o = obs[t]
        obs_min_tri = float(o["tri"].min())
        obs_min_pair = float(min(p.min() for p in o["pw"].values()))
        rng = np.random.default_rng(2027)
        perms = [rng.permutation(o["n"]) for _ in range(B_GW)]
        res = Parallel(n_jobs=N_JOBS)(
            delayed(_gw_rep)(o["KA"], o["KB"], o["KD"], o["Bd"], o["y"][perms[i]], 900000 + i)
            for i in range(B_GW))
        res = [r for r in res if r.get("ok")]
        Bok = len(res)
        mt_perm = np.array([r["min_tri"] for r in res])
        mp_perm = np.array([r["min_pair"] for r in res])
        emp_tri = (1 + int((mt_perm <= obs_min_tri).sum())) / (Bok + 1)
        emp_pair = (1 + int((mp_perm <= obs_min_pair).sum())) / (Bok + 1)
        gw.append(dict(trait=t, n=o["n"], B_ok=Bok, obs_min_triad_acat=obs_min_tri,
                       fwer_emp_p_triad=float(emp_tri), obs_min_pairwise_p=obs_min_pair,
                       fwer_emp_p_pairwise=float(emp_pair),
                       lead_triad="|".join(kept[int(np.argmin(o["tri"]))]),
                       asym_bonferroni_p=float(min(obs_min_tri * G, 1.0))))
        print(f"  [GW {t}] obs_min_tri={obs_min_tri:.3g} FWER_emp_p={emp_tri:.4g} "
              f"(pairwise {emp_pair:.4g}) Bok={Bok} asym_bonf={min(obs_min_tri*G,1.0):.3g} "
              f"({time.time()-t0:.0f}s)")
    report["genome_wide_minP"] = gw

    # ---- (3) PLEIOTROPY frozen clusters ----
    pleio = []
    alpha_cluster = 0.05 / (G * 3)
    for cname, members in {**CLUSTERS, "ALL_14": traits14}.items():
        present = [m for m in members if m in obs]
        if len(present) < 2:
            continue
        stk = np.column_stack([obs[m]["tri"] for m in present])  # (G, n_traits)
        ctri = np.array([_w2._acat(list(stk[i])) for i in range(G)])
        lead = int(np.argmin(ctri))
        is_main = cname in CLUSTERS
        pleio.append(dict(cluster=cname, traits=present, role=("primary" if is_main else "sensitivity"),
                          min_cluster_acat=float(ctri.min()), lead_triad="|".join(kept[lead]),
                          alpha=alpha_cluster if is_main else None,
                          passes=bool(is_main and ctri.min() < alpha_cluster)))
        print(f"  [pleio {cname}] min={ctri.min():.3g} lead={kept[lead]} "
              f"{'PASS' if (is_main and ctri.min()<alpha_cluster) else ''} (alpha={alpha_cluster:.2e})")
    report["pleiotropy"] = pleio

    # ---- (4) PER-ENV sensitivity: emergence (CFLN cols) + heading/grain_width (per sheet) ----
    per_env = []
    for env in ("CFLN06", "CFLN10", "CFLN14", "CFLN20"):
        col = f"days_to_emerg_{env}"
        if col not in pc.columns:
            continue
        idx, valid = idx_for(pc[col])
        if idx.size < MIN_N:
            per_env.append(dict(trait="emergence", env=env, n=int(idx.size), skipped=True))
            continue
        pw, tri, *_ = _observed(np.array([float(pc[col].loc[s]) for s in valid]), idx, grms, Bfull, common)
        lead = int(np.argmin(tri))
        per_env.append(dict(trait="emergence", env=env, n=idx.size, min_triad_acat=float(tri.min()),
                            lead_triad="|".join(kept[lead])))
        print(f"  [per-env emergence {env}] n={idx.size} minACAT={tri.min():.3g} lead={kept[lead]}")
    geno = list(pc.index)
    pe_etl = _per_env_etl(set(geno))
    for t in ("heading_date", "grain_width"):
        for env, dvals in pe_etl.get(t, {}).items():
            ser = pd.Series(dvals)
            idx, valid = idx_for(ser)
            if idx.size < MIN_N:
                per_env.append(dict(trait=t, env=env, n=int(idx.size), skipped=True))
                continue
            pw, tri, *_ = _observed(np.array([float(ser.loc[s]) for s in valid]), idx, grms, Bfull, common)
            lead = int(np.argmin(tri))
            per_env.append(dict(trait=t, env=env, n=idx.size, min_triad_acat=float(tri.min()),
                                lead_triad="|".join(kept[lead])))
            print(f"  [per-env {t} {env}] n={idx.size} minACAT={tri.min():.3g} lead={kept[lead]}")
    report["per_env_sensitivity"] = per_env

    (OUT / "wheat_multitrait_perm.json").write_text(json.dumps(report, indent=2, default=float))
    print(f"\n  wrote {OUT}/wheat_multitrait_perm.json ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
