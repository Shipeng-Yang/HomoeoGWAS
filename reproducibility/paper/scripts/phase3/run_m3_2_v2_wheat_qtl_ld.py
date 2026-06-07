#!/usr/bin/env python3
"""Phase 3 M3.2-v2 — wheat QTL recovery with LD fine-mapping + per-env validation.

Builds on M3.2-v1 (`run_m3_2_wheat_qtl.py`):
  v1 = discovery pass (85 physical-clumped candidates, no LD)
  v2 = the **acceptance pass** for charter §3.3 v1.0:
       per-candidate LD validation (plink2 --r2-unphased on full scan BED)
       + per-env targeted regression (CFLN06/10/14/20)
       + candidate tiering into HIGH_CONF / SUPPLEMENTARY / QUARANTINED.

Charter §3.3 v1.0 硬指标:
  ≥1 independent novel locus passing 5e-8 + LD-fine-mapping + per-env consistency.

Inputs:
  data/reference/wheat/known_qtl_wheat.tsv           (13 QTLs + trait_relevance)
  data/processed/wheat_bed/{A,B,D}/all.bed           (full scan BED for LD)
  data/processed/wheat/pheno_clean.tsv               (4 env sub-trait columns)
  results/phase3/m3_2_qtl/wheat_watkins/             (v1 outputs)

Outputs in same dir (v1 outputs preserved):
  novel_loci_tiered.tsv          HIGH_CONF / SUPPLEMENTARY / QUARANTINED
  known_qtl_hits_v2.tsv          v1 known_qtl_hits + trait_relevance
  ld_validation.tsv              pairwise r² (cand vs known + cand vs cand)
  per_env_validation.tsv         per-env β / p for each top candidate
  m3_2_v2_summary_<trait>.json   acceptance + provenance

The script is *idempotent*: rerunning skips already-cached plink2 LD outputs.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

# Thresholds (Codex review reconciliation)
P_GENOME_WIDE = 5.0e-8
P_SUGGESTIVE = 5.0e-5
R2_QUARANTINE = 0.20          # max r² to known sentinel above which we flag
R2_MERGE = 0.50               # merge candidates with pairwise r² >= 0.5
LD_WINDOW_KB = 30000          # plink2 --ld-window-kb (30 Mb covers within-chrom)
LD_WINDOW_R2 = 0.05           # plink2 --ld-window-r2 floor
PER_ENV_P_NOMINAL = 0.05
PER_ENV_MIN_AGREEING_ENVS = 2  # ≥2/4 env with nominal + same direction
EMMAX_NOVEL_P = 1.0e-5         # novel must have EMMAX concordant at p<1e-5
KNOWN_SENTINEL_FLANK_BP = 100_000   # use SNPs within ±100kb of QTL as sentinels
KNOWN_SENTINEL_PER_QTL = 5     # max 5 sentinel SNPs per QTL
TOP_CAND_FOR_LD = 100          # cap LD/per-env work to top-N (by p)
PLINK2_MEMORY_MB = 8000
PLINK2_THREADS = 8


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serialisable: {type(o)}")


def _which_plink2(arg_path: str | None) -> str:
    if arg_path:
        p = Path(arg_path)
        if p.exists():
            return str(p)
    p = shutil.which("plink2") or \
        "/home/yys05/.local/share/mamba/envs/polygwas-cpu/bin/plink2"
    if not Path(p).exists():
        sys.exit(f"ERR: plink2 not found (looked at PATH and {p})")
    return p


# =====================================================================
# Inputs
# =====================================================================


def load_known_qtls(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    needed = {"qtl_name", "qtl_family", "chrom", "start", "end", "qtl_pos",
              "window_bp", "qtl_class", "trait_relevance_emerg"}
    missing = needed - set(df.columns)
    if missing:
        sys.exit(f"ERR: known_qtl_wheat.tsv missing columns: {missing}")
    df["subgenome"] = df["chrom"].str[-1]
    for c in ("start", "end", "qtl_pos", "window_bp"):
        df[c] = pd.to_numeric(df[c]).astype(np.int64)
    return df


def load_v1_outputs(v1_dir: Path) -> dict[str, pd.DataFrame]:
    needed = ["new_locus_candidates.tsv", "known_qtl_hits.tsv",
              "all_significant_leads.tsv"]
    for n in needed:
        p = v1_dir / n
        if not p.exists():
            sys.exit(f"ERR: M3.2-v1 output missing: {p}")
    return {
        "novel": pd.read_csv(v1_dir / "new_locus_candidates.tsv", sep="\t"),
        "known_hits": pd.read_csv(v1_dir / "known_qtl_hits.tsv", sep="\t"),
        "all_leads": pd.read_csv(v1_dir / "all_significant_leads.tsv", sep="\t"),
    }


# =====================================================================
# Sentinel SNPs near known QTLs
# =====================================================================


def pick_known_sentinels(
    qtls: pd.DataFrame, full_bed_root: Path,
    flank_bp: int = KNOWN_SENTINEL_FLANK_BP,
    per_qtl: int = KNOWN_SENTINEL_PER_QTL,
) -> dict[str, list[str]]:
    """For each QTL, pick up-to-`per_qtl` real SNP IDs near the gene as LD sentinels.

    Reads each subgenome's .bim once and indexes by chrom; the BIM rows are
    sorted by chrom+pos so a range filter is cheap.
    """
    sentinels: dict[str, list[str]] = {}
    for sg in ("A", "B", "D"):
        sg_qtls = qtls[qtls["subgenome"] == sg]
        if len(sg_qtls) == 0:
            continue
        bim_path = full_bed_root / sg / "all.bim"
        bim = pd.read_csv(bim_path, sep="\t", header=None,
                          names=["chrom", "snp_id", "cm", "pos", "a1", "a2"],
                          usecols=["chrom", "snp_id", "pos"])
        bim["chrom"] = bim["chrom"].astype(str)
        for _, q in sg_qtls.iterrows():
            mask = ((bim["chrom"] == q["chrom"])
                    & (bim["pos"] >= q["qtl_pos"] - flank_bp)
                    & (bim["pos"] <= q["qtl_pos"] + flank_bp))
            window = bim.loc[mask]
            if len(window) == 0:
                sentinels[q["qtl_name"]] = []
                print(f"  WARN: no SNP in ±{flank_bp//1000}kb of {q['qtl_name']} "
                      f"({q['chrom']}:{q['qtl_pos']})")
                continue
            # pick `per_qtl` evenly spaced
            step = max(1, len(window) // per_qtl)
            picks = window.iloc[::step].head(per_qtl)["snp_id"].astype(str).tolist()
            sentinels[q["qtl_name"]] = picks
    return sentinels


# =====================================================================
# Per-chrom plink2 LD (cached)
# =====================================================================


def run_ld_per_chrom(
    chrom: str, subgenome: str,
    snp_list: list[str], full_bed_root: Path, out_dir: Path,
    plink2_bin: str,
) -> pd.DataFrame | None:
    """plink2 --r2-unphased on `snp_list` for `chrom` on `subgenome`'s full BED.

    Output cached at {out_dir}/ld_{chrom}.vcor; re-uses iff the cached SNP
    list (sha256) matches the requested one. Returns parsed pairs DataFrame
    (cols: snp_a, snp_b, r2) or None if fewer than 2 SNPs in list.
    """
    import hashlib
    if len(snp_list) < 2:
        return None
    bfile = full_bed_root / subgenome / "all"
    list_payload = "\n".join(sorted(snp_list)) + "\n"
    list_sha = hashlib.sha256(list_payload.encode()).hexdigest()
    list_path = out_dir / f"snps_{chrom}.list"
    list_path.write_text(list_payload)
    sha_path = out_dir / f"snps_{chrom}.sha256"
    out_prefix = out_dir / f"ld_{chrom}"
    vcor = Path(str(out_prefix) + ".vcor")
    cache_valid = (vcor.exists() and vcor.stat().st_size > 0 and
                   sha_path.exists() and sha_path.read_text().strip() == list_sha)
    if cache_valid:
        # use cached
        pass
    else:
        if vcor.exists():
            print(f"    LD cache {chrom}: SNP-list sha changed, rebuilding")
        cmd = [plink2_bin,
               "--bfile", str(bfile),
               "--chr", chrom,
               "--extract", str(list_path),
               "--r2-unphased",
               "--ld-window-kb", str(LD_WINDOW_KB),
               "--ld-window-r2", str(LD_WINDOW_R2),
               "--ld-window", "1000000",          # max variant-pair count
               "--memory", str(PLINK2_MEMORY_MB),
               "--threads", str(PLINK2_THREADS),
               "--out", str(out_prefix)]
        try:
            log = subprocess.run(cmd, check=True, capture_output=True, text=True)
            tail = log.stdout.strip().splitlines()[-3:] if log.stdout else []
            print(f"    plink2 LD {chrom}: " + " | ".join(tail))
            sha_path.write_text(list_sha)
        except subprocess.CalledProcessError as e:
            print(f"  ERR plink2 LD {chrom}: stderr={e.stderr[-400:] if e.stderr else 'none'}")
            return None
    if not vcor.exists() or vcor.stat().st_size == 0:
        print(f"  no .vcor produced for {chrom} — probably no pair in window")
        return None
    # parse .vcor. Header is "#CHROM_A POS_A ID_A CHROM_B POS_B ID_B UNPHASED_R2"
    df = pd.read_csv(vcor, sep="\t")
    # find ID columns and r² column
    def _col(prefer: list[str]):
        for p in prefer:
            for c in df.columns:
                if c.upper() == p:
                    return c
        return None
    id_a = _col(["ID_A"]) or _col(["#ID_A", "VARIANT_A"])
    id_b = _col(["ID_B"]) or _col(["VARIANT_B"])
    r2 = _col(["UNPHASED_R2", "R2"])
    if id_a is None or id_b is None or r2 is None:
        print(f"  WARN: unrecognised .vcor columns for {chrom}: {list(df.columns)}")
        return None
    out = pd.DataFrame({
        "snp_a": df[id_a].astype(str),
        "snp_b": df[id_b].astype(str),
        "r2": df[r2].astype(np.float64),
        "chrom": chrom,
    })
    return out


# =====================================================================
# Per-env regression
# =====================================================================


def load_lead_dosages(
    leads: pd.DataFrame, full_bed_root: Path,
) -> tuple[pd.DataFrame, list[str]]:
    """Extract lead-SNP dosages from full BED.

    Returns (dosage_df, sample_ids). dosage_df has shape (n_samples, n_leads)
    with columns = lead snp_id. Uses bed_reader; missing as NaN.
    """
    from bed_reader import open_bed
    parts: list[pd.DataFrame] = []
    common_samples: list[str] | None = None
    for sg in ("A", "B", "D"):
        sg_leads = leads[leads["subgenome"] == sg]
        if len(sg_leads) == 0:
            continue
        snp_set = set(sg_leads["snp_id"].astype(str).tolist())
        bed_prefix = full_bed_root / sg / "all"
        with open_bed(str(bed_prefix.with_suffix(".bed"))) as bed:
            samples = np.asarray(bed.iid, dtype=object)
            sid = np.asarray(bed.sid, dtype=object)
            mask = np.isin(sid, list(snp_set))
            idx = np.where(mask)[0]
            if len(idx) == 0:
                continue
            dos = bed.read(index=(slice(None), idx), dtype="float32")
            cols = sid[idx]
            sub_df = pd.DataFrame(dos, columns=cols)
            sub_df.insert(0, "_sample", [str(s) for s in samples])
            parts.append(sub_df)
            if common_samples is None:
                common_samples = [str(s) for s in samples]
    if not parts:
        return pd.DataFrame(), []
    # Merge on _sample (intersect)
    merged = parts[0]
    for p in parts[1:]:
        merged = merged.merge(p, on="_sample", how="inner")
    samples_out = merged["_sample"].tolist()
    dosage_out = merged.drop(columns="_sample")
    return dosage_out, samples_out


def per_env_regression(
    leads: pd.DataFrame, dosage_df: pd.DataFrame, sample_ids: list[str],
    pheno_path: Path, env_cols: list[str],
) -> pd.DataFrame:
    """OLS y_env ~ dosage_lead (no covariates), for each (lead, env).

    Records β, SE, p, and per-env sample count. Direction concordance is
    computed downstream (lead's β sign vs overall scan β sign).
    """
    from scipy import stats as scstats
    pheno = pd.read_csv(pheno_path, sep="\t").set_index("sample")
    if not all(c in pheno.columns for c in env_cols):
        sys.exit(f"ERR: pheno_clean.tsv missing env cols {env_cols}")
    rows: list[dict] = []
    sample_ids_arr = np.asarray(sample_ids, dtype=object)
    # restrict pheno to samples we have dosages for
    pheno_sub = pheno.reindex([str(s) for s in sample_ids_arr])
    for _, lead in leads.iterrows():
        snp = str(lead["snp_id"])
        if snp not in dosage_df.columns:
            continue
        g = dosage_df[snp].to_numpy(dtype=np.float64)
        # mean-impute missing
        finite = np.isfinite(g)
        if finite.sum() < 20:
            continue
        g[~finite] = g[finite].mean()
        for env in env_cols:
            y = pheno_sub[env].to_numpy(dtype=np.float64)
            mask = np.isfinite(y)
            if mask.sum() < 20:
                continue
            yi = y[mask]
            gi = g[mask]
            try:
                slope, intercept, rval, pval, stderr = scstats.linregress(gi, yi)
            except Exception:                              # noqa: BLE001
                continue
            rows.append({
                "lead_snp": snp,
                "chrom": lead["chrom"],
                "pos": int(lead["pos"]),
                "subgenome": lead["subgenome"],
                "env": env,
                "n": int(mask.sum()),
                "beta_env": float(slope),
                "se_env": float(stderr),
                "p_env": float(pval),
                "r_env": float(rval),
            })
    return pd.DataFrame(rows)


def aggregate_per_env(per_env: pd.DataFrame, leads: pd.DataFrame
                       ) -> pd.DataFrame:
    """Aggregate per-env rows to a single row per lead.

    Returns: n_env_nominal, n_env_same_direction, min_p_env, max_p_env, etc.
    Direction concordance: per_env.beta_env sign == sign of leads.beta column.
    """
    out_rows: list[dict] = []
    leads_idx = leads.set_index("snp_id")
    for snp, sub in per_env.groupby("lead_snp"):
        if snp not in leads_idx.index:
            continue
        scan_beta = float(leads_idx.loc[snp, "beta"]) if "beta" in leads_idx.columns else None
        same_dir = (np.sign(sub["beta_env"]) == np.sign(scan_beta)).sum() if scan_beta is not None else 0
        out_rows.append({
            "lead_snp": snp,
            "n_envs_tested": int(len(sub)),
            "n_envs_nominal": int((sub["p_env"] < PER_ENV_P_NOMINAL).sum()),
            "n_envs_same_direction": int(same_dir),
            "min_p_env": float(sub["p_env"].min()),
            "max_p_env": float(sub["p_env"].max()),
            "median_p_env": float(sub["p_env"].median()),
        })
    return pd.DataFrame(out_rows)


# =====================================================================
# Tiering
# =====================================================================


def tier_candidates(
    novel: pd.DataFrame, ld_summary: pd.DataFrame, env_summary: pd.DataFrame,
    ld_pairs: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Assign HIGH_CONFIDENCE / SUPPLEMENTARY / QUARANTINED tier per lead, plus
    seed-vs-seed clumping into ``independent_locus_id`` and a stricter
    ``strict_high_confidence`` subset.

    Tier rules (Codex v2 review):
      - QUARANTINED:   max_r2_to_known_sentinel >= R2_QUARANTINE
      - HIGH_CONFIDENCE: p<5e-8 AND r2_to_known<R2_QUARANTINE AND
          n_envs_nominal >= PER_ENV_MIN_AGREEING_ENVS AND
          n_envs_same_direction >= PER_ENV_MIN_AGREEING_ENVS AND
          emmax_concordant_p1e5
      - SUPPLEMENTARY: any other p<5e-8

    strict_high_confidence (paper-main subset):
        HIGH_CONFIDENCE AND n_envs_nominal >= 3 AND n_envs_same_direction >= 3

    Seed-vs-seed clumping (Codex v2 review #3):
      - within HIGH_CONFIDENCE, greedy by p ascending
      - lead_of_locus = True for first SNP in each clump
      - independent_locus_id = "locus_<chrom>_<rank>" assigned to all clump members
      - redundant_with = lead-of-locus SNP id (NaN if itself is the lead)
      - r2_to_locus_lead = pairwise r² to the locus lead
    """
    out = novel.copy()
    # v1 calls the column ``snp_id``; LD/env summaries use ``lead_snp``
    if "lead_snp" not in out.columns and "snp_id" in out.columns:
        out = out.rename(columns={"snp_id": "lead_snp"})
    out = out.merge(
        ld_summary[["lead_snp", "max_r2_to_known_sentinel",
                    "nearest_known_sentinel", "best_seed_pair_r2"]],
        on="lead_snp", how="left", suffixes=("", "_ld"))
    out = out.merge(
        env_summary, left_on="lead_snp", right_on="lead_snp",
        how="left", suffixes=("", "_env"))
    out["max_r2_to_known_sentinel"] = out["max_r2_to_known_sentinel"].fillna(0.0)
    out["n_envs_nominal"] = out["n_envs_nominal"].fillna(0).astype(int)
    out["n_envs_same_direction"] = out["n_envs_same_direction"].fillna(0).astype(int)

    # Codex v2 review #6: name the column explicitly so it doesn't get
    # confused with the v1 known-recovery 5e-5 concordance.
    out["emmax_concordant_p1e5"] = out["emmax_p"].fillna(1.0) < EMMAX_NOVEL_P

    def _tier(row) -> str:
        if row["max_r2_to_known_sentinel"] >= R2_QUARANTINE:
            return "QUARANTINED"
        if row["p"] >= P_GENOME_WIDE:
            return "SUPPLEMENTARY"
        if not row["emmax_concordant_p1e5"]:
            return "SUPPLEMENTARY"
        if (row["n_envs_nominal"] >= PER_ENV_MIN_AGREEING_ENVS
                and row["n_envs_same_direction"] >= PER_ENV_MIN_AGREEING_ENVS):
            return "HIGH_CONFIDENCE"
        return "SUPPLEMENTARY"

    out["tier"] = out.apply(_tier, axis=1)
    out["strict_high_confidence"] = (
        (out["tier"] == "HIGH_CONFIDENCE")
        & (out["n_envs_nominal"] >= 3)
        & (out["n_envs_same_direction"] >= 3)
    )

    # ------- seed-vs-seed clumping inside HIGH_CONFIDENCE per chrom ----
    out["independent_locus_id"] = ""
    out["lead_of_locus"] = False
    out["redundant_with"] = ""
    out["r2_to_locus_lead"] = np.nan
    if ld_pairs is not None and len(ld_pairs) > 0:
        # build undirected lookup {(a,b)->r2}
        ld_lookup: dict[tuple[str, str], float] = {}
        for _, r in ld_pairs.iterrows():
            a, b = str(r["snp_a"]), str(r["snp_b"])
            r2 = float(r["r2"])
            prev = ld_lookup.get((a, b), 0.0)
            if r2 > prev:
                ld_lookup[(a, b)] = r2
                ld_lookup[(b, a)] = r2
        hc_mask = out["tier"] == "HIGH_CONFIDENCE"
        for chrom, sub in out.loc[hc_mask].groupby("chrom"):
            ordered = sub.sort_values("p").index.tolist()
            assigned = set()
            rank = 0
            for i in ordered:
                if i in assigned:
                    continue
                rank += 1
                lead_snp = str(out.at[i, "lead_snp"])
                locus_id = f"locus_{chrom}_{rank:02d}"
                out.at[i, "independent_locus_id"] = locus_id
                out.at[i, "lead_of_locus"] = True
                out.at[i, "redundant_with"] = ""
                out.at[i, "r2_to_locus_lead"] = 1.0
                assigned.add(i)
                # absorb later HC SNPs on same chrom with r² >= R2_MERGE
                for j in ordered:
                    if j in assigned:
                        continue
                    other_snp = str(out.at[j, "lead_snp"])
                    r2 = ld_lookup.get((lead_snp, other_snp), 0.0)
                    if r2 >= R2_MERGE:
                        out.at[j, "independent_locus_id"] = locus_id
                        out.at[j, "redundant_with"] = lead_snp
                        out.at[j, "r2_to_locus_lead"] = float(r2)
                        assigned.add(j)
    return out.sort_values(
        ["tier", "p"], key=lambda s: s if s.name == "p"
        else s.map({"HIGH_CONFIDENCE": 0, "SUPPLEMENTARY": 1, "QUARANTINED": 2})
    ).reset_index(drop=True)


# =====================================================================
# Main
# =====================================================================


def main():
    ap = argparse.ArgumentParser(description="M3.2-v2 LD + per-env validation")
    ap.add_argument("--trait", default="days_to_emerg")
    ap.add_argument("--v1-dir", default=str(
        ROOT / "results/phase3/m3_2_qtl/wheat_watkins"))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--known-qtl",
                    default=str(ROOT / "data/reference/wheat/known_qtl_wheat.tsv"))
    ap.add_argument("--full-bed-root",
                    default=str(ROOT / "data/processed/wheat_bed"))
    ap.add_argument("--pheno",
                    default=str(ROOT / "data/processed/wheat/pheno_clean.tsv"))
    ap.add_argument("--env-cols",
                    default="days_to_emerg_CFLN06,days_to_emerg_CFLN10,"
                            "days_to_emerg_CFLN14,days_to_emerg_CFLN20")
    ap.add_argument("--plink2", default=None)
    ap.add_argument("--top-cand", type=int, default=TOP_CAND_FOR_LD)
    ap.add_argument("--ld-cache-dir", default=None,
                    help="dir for plink2 LD outputs (default <out>/ld_cache)")
    args = ap.parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.v1_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ld_dir = Path(args.ld_cache_dir) if args.ld_cache_dir else out_dir / "ld_cache"
    ld_dir.mkdir(parents=True, exist_ok=True)
    plink2_bin = _which_plink2(args.plink2)
    t0 = time.time()

    print(f"=== M3.2-v2 wheat QTL recovery (LD + per-env) — trait={args.trait} ===")
    print(f"  plink2 = {plink2_bin}")

    # 1. Inputs
    print("[1] loading inputs")
    qtls = load_known_qtls(Path(args.known_qtl))
    v1 = load_v1_outputs(Path(args.v1_dir))
    novel = v1["novel"][v1["novel"]["is_novel_candidate"].eq(True)].copy()
    novel = novel.sort_values("p").head(args.top_cand).reset_index(drop=True)
    print(f"  known QTL: {len(qtls)}, v1 novel candidates: {len(v1['novel'])}, "
          f"top-{args.top_cand}: {len(novel)}")

    # 2. Augment known_qtl_hits with trait_relevance
    print("[2] augment known_qtl_hits with trait_relevance")
    qkey = qtls.set_index("qtl_name")["trait_relevance_emerg"].to_dict()
    known_v2 = v1["known_hits"].copy()
    known_v2["trait_relevance_emerg"] = known_v2["qtl_name"].map(qkey)
    known_v2_path = out_dir / "known_qtl_hits_v2.tsv"
    known_v2.to_csv(known_v2_path, sep="\t", index=False)
    print(f"  wrote {known_v2_path}")

    # 3. Sentinel SNPs
    print("[3] picking known-QTL sentinel SNPs from full BED")
    sentinels = pick_known_sentinels(
        qtls, Path(args.full_bed_root))
    n_total_sent = sum(len(v) for v in sentinels.values())
    print(f"  picked {n_total_sent} sentinel SNPs across {len(sentinels)} QTLs")

    # 4. LD per chrom — combined candidate-leads + known-sentinels
    print(f"[4] plink2 --r2-unphased LD per chrom (cached at {ld_dir})")
    ld_pairs_all: list[pd.DataFrame] = []
    sentinel_to_qtl: dict[str, str] = {
        s: q for q, ss in sentinels.items() for s in ss}
    for chrom in sorted(set(novel["chrom"]).union({q["chrom"] for _, q in qtls.iterrows()})):
        chrom_str = str(chrom)
        sg = chrom_str[-1]                       # A/B/D
        cand_snps = novel.loc[novel["chrom"] == chrom_str, "snp_id"].astype(str).tolist()
        # sentinel SNPs for any QTL on this chrom
        chrom_qtls = qtls[qtls["chrom"] == chrom_str]
        sent_snps: list[str] = []
        for _, q in chrom_qtls.iterrows():
            sent_snps.extend(sentinels.get(q["qtl_name"], []))
        snp_list = sorted(set(cand_snps) | set(sent_snps))
        if len(snp_list) < 2:
            continue
        pairs = run_ld_per_chrom(chrom_str, sg, snp_list,
                                  Path(args.full_bed_root), ld_dir, plink2_bin)
        if pairs is not None and len(pairs) > 0:
            ld_pairs_all.append(pairs)
    ld_pairs = (pd.concat(ld_pairs_all, ignore_index=True) if ld_pairs_all
                else pd.DataFrame(columns=["snp_a", "snp_b", "r2", "chrom"]))
    ld_pairs_path = out_dir / "ld_pairs.tsv"
    ld_pairs.to_csv(ld_pairs_path, sep="\t", index=False)
    print(f"  parsed {len(ld_pairs)} LD pairs (r²≥{LD_WINDOW_R2}); wrote {ld_pairs_path}")

    # 5. LD summary per lead
    print("[5] LD summary per candidate lead")
    sentinel_set = set(sentinel_to_qtl)
    cand_set = set(novel["snp_id"].astype(str))
    summary_rows: list[dict] = []
    for _, lead in novel.iterrows():
        snp = str(lead["snp_id"])
        # bi-directional pairs
        same_a = ld_pairs[ld_pairs["snp_a"] == snp]
        same_b = ld_pairs[ld_pairs["snp_b"] == snp].rename(
            columns={"snp_a": "snp_b", "snp_b": "snp_a"})
        all_p = pd.concat([same_a, same_b], ignore_index=True)
        # r² to known sentinels
        sent_pairs = all_p[all_p["snp_b"].isin(sentinel_set)]
        if len(sent_pairs):
            best = sent_pairs.loc[sent_pairs["r2"].idxmax()]
            max_r2_known = float(best["r2"])
            nearest_sent = str(best["snp_b"])
            nearest_qtl = sentinel_to_qtl.get(nearest_sent, "")
        else:
            max_r2_known = 0.0
            nearest_sent = ""
            nearest_qtl = ""
        # r² to other candidates on same chrom
        cand_pairs = all_p[all_p["snp_b"].isin(cand_set - {snp})]
        best_seed_pair_r2 = float(cand_pairs["r2"].max()) if len(cand_pairs) else 0.0
        summary_rows.append({
            "lead_snp": snp,
            "chrom": lead["chrom"],
            "max_r2_to_known_sentinel": max_r2_known,
            "nearest_known_sentinel": nearest_sent,
            "nearest_known_qtl_via_ld": nearest_qtl,
            "best_seed_pair_r2": best_seed_pair_r2,
            "n_ld_partners_r2_ge05": int(((all_p["r2"] >= 0.5)).sum()),
        })
    ld_summary = pd.DataFrame(summary_rows)
    ld_summary_path = out_dir / "ld_validation.tsv"
    ld_summary.to_csv(ld_summary_path, sep="\t", index=False)
    print(f"  wrote {ld_summary_path}")

    # 6. Per-env regression
    print("[6] extracting lead-SNP dosages from full BED + per-env OLS")
    env_cols = [c.strip() for c in args.env_cols.split(",") if c.strip()]
    dos, samples = load_lead_dosages(novel, Path(args.full_bed_root))
    if len(dos) == 0:
        print("  ERR: could not extract any lead dosages")
        per_env = pd.DataFrame()
        env_summary = pd.DataFrame()
    else:
        print(f"  loaded {dos.shape[1]} lead dosages × {len(samples)} samples")
        per_env = per_env_regression(novel, dos, samples,
                                       Path(args.pheno), env_cols)
        per_env_path = out_dir / "per_env_validation.tsv"
        per_env.to_csv(per_env_path, sep="\t", index=False)
        print(f"  {len(per_env)} per-(lead,env) rows; wrote {per_env_path}")
        env_summary = aggregate_per_env(per_env, novel)
        env_summary_path = out_dir / "per_env_summary.tsv"
        env_summary.to_csv(env_summary_path, sep="\t", index=False)
        print(f"  wrote {env_summary_path}")

    # 7. Tiering + seed-vs-seed clumping + STRICT_HC subset
    print("[7] tiering candidates (+ seed-vs-seed locus clumping)")
    tiered = tier_candidates(novel, ld_summary, env_summary, ld_pairs=ld_pairs)
    tiered_path = out_dir / "novel_loci_tiered.tsv"
    tiered.to_csv(tiered_path, sep="\t", index=False)
    tier_counts = tiered["tier"].value_counts().to_dict()
    n_independent_hc_loci = int(
        tiered.loc[tiered["tier"] == "HIGH_CONFIDENCE", "lead_of_locus"].sum())
    n_strict_hc = int(tiered["strict_high_confidence"].sum())
    print(f"  tier counts: {tier_counts}")
    print(f"  HIGH_CONFIDENCE leads → {n_independent_hc_loci} independent loci "
          f"(r²<{R2_MERGE} merge)")
    print(f"  STRICT_HIGH_CONFIDENCE (≥3/4 env nominal + same direction): {n_strict_hc}")
    print(f"  wrote {tiered_path}")

    # 8. Acceptance gates
    print("[8] acceptance gates")
    acceptance = []
    def check(name, ok, msg=""):
        acceptance.append({"check": name, "passed": bool(ok), "message": msg})
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {msg}" if msg else ""))
    n_hc = int((tiered["tier"] == "HIGH_CONFIDENCE").sum())
    n_supp = int((tiered["tier"] == "SUPPLEMENTARY").sum())
    n_quar = int((tiered["tier"] == "QUARANTINED").sum())
    check("at_least_1_HIGH_CONFIDENCE_novel",
          n_hc >= 1,
          f"{n_hc} HIGH_CONFIDENCE / {n_supp} SUPPLEMENTARY / {n_quar} QUARANTINED")
    check("at_least_1_independent_HIGH_CONF_locus",
          n_independent_hc_loci >= 1,
          f"{n_independent_hc_loci} independent HC loci after seed-vs-seed clump")
    check("ld_pairs_computed",
          len(ld_pairs) > 0,
          f"{len(ld_pairs)} LD pairs computed")
    check("per_env_validation_completed",
          len(per_env) > 0,
          f"{len(per_env)} per-(lead,env) regressions")
    # chr5B-vs-Vrn-B1 LD assertion (Codex v3 fix #5): if chr5B has any v1
    # novel candidates AND a Vrn-B1 sentinel exists in ld_pairs, the LD
    # filter MUST have quarantined ≥1 of them. Vacuously PASS if either
    # precondition is absent.
    chr5b_quarantined = int(
        ((tiered["chrom"] == "chr5B") & (tiered["tier"] == "QUARANTINED")).sum())
    chr5b_cand_n = int((tiered["chrom"] == "chr5B").sum())
    vrnb1_sentinel_in_pairs = bool(
        len(ld_pairs) > 0 and
        ld_pairs["chrom"].astype(str).eq("chr5B").any())
    chr5b_gate_active = chr5b_cand_n > 0 and vrnb1_sentinel_in_pairs
    check("chr5B_VrnB1_LD_filter_active",
          (not chr5b_gate_active) or chr5b_quarantined >= 1,
          f"{chr5b_quarantined}/{chr5b_cand_n} chr5B leads quarantined "
          f"(precondition={'active' if chr5b_gate_active else 'vacuous'})")
    all_passed = all(c["passed"] for c in acceptance)

    runtime = time.time() - t0
    summary = {
        "script": "run_m3_2_v2_wheat_qtl_ld.py", "milestone": "M3.2-v2",
        "version": "v2 (LD validation + per-env regression + tiering)",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "trait": args.trait, "runtime_sec": round(runtime, 1),
        "thresholds": {
            "p_genome_wide": P_GENOME_WIDE,
            "r2_quarantine": R2_QUARANTINE,
            "r2_merge": R2_MERGE,
            "per_env_p_nominal": PER_ENV_P_NOMINAL,
            "per_env_min_agreeing_envs": PER_ENV_MIN_AGREEING_ENVS,
            "emmax_novel_p": EMMAX_NOVEL_P,
            "ld_window_kb": LD_WINDOW_KB,
            "ld_window_r2": LD_WINDOW_R2,
        },
        "v1_novel_count": int(len(v1["novel"])),
        "top_cand_for_ld": int(args.top_cand),
        "tier_counts": tier_counts,
        "n_independent_high_conf_loci": int(n_independent_hc_loci),
        "n_strict_high_confidence": int(n_strict_hc),
        "n_known_qtl_sentinels": int(n_total_sent),
        "n_ld_pairs": int(len(ld_pairs)),
        "n_per_env_rows": int(len(per_env)),
        "env_cols": env_cols,
        "outputs": {
            "known_qtl_hits_v2": str(known_v2_path),
            "ld_pairs": str(ld_pairs_path),
            "ld_validation": str(ld_summary_path),
            "per_env_validation": str(out_dir / "per_env_validation.tsv")
                                   if len(per_env) else None,
            "per_env_summary": str(out_dir / "per_env_summary.tsv")
                                if len(env_summary) else None,
            "novel_loci_tiered": str(tiered_path),
        },
        "acceptance": acceptance,
        "acceptance_all_passed": all_passed,
    }
    summary_path = out_dir / f"m3_2_v2_summary_{args.trait}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=_json_default)
    print(f"\nwrote {summary_path}")
    print(f"acceptance: {sum(c['passed'] for c in acceptance)}/{len(acceptance)} "
          f"(runtime {runtime:.0f}s)")
    if all_passed:
        print("✅ M3.2-v2 acceptance PASS")
        return 0
    failed = [c['check'] for c in acceptance if not c['passed']]
    print(f"❌ M3.2-v2 acceptance FAIL — {failed}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
