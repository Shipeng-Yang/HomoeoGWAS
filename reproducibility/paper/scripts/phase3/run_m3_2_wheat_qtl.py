#!/usr/bin/env python3
"""Phase 3 M3.2 — wheat Watkins known QTL recovery + new locus candidates.

Cross-references the M3.1-v2 LOCO sumstats against canonical wheat
flowering/emergence QTLs (Ppd-1, Vrn-1, Vrn-3/FT, ELF3/Eps, Rht-1) and
extracts independent novel candidates by physical clumping. Charter §3.3
v1.0 requires ≥1 independent new locus surviving the genome-wide threshold.

Inputs:
  data/reference/wheat/known_qtl_wheat.tsv            (13 anchor QTLs, GFF3 IDs)
  data/reference/wheat/iwgsc_refseqv1.1_genes_2017July06.zip (GFF3 for nearest-gene)
  results/phase3/m3_1_loco_v2/wheat_watkins/scan_loco_{A,B,D}.tsv.gz
  results/phase2/m2_5_v2/wheat_watkins/scan_{A,B,D}.tsv.gz   (EMMAX baseline)

Outputs in results/phase3/m3_2_qtl/wheat_watkins/:
  known_qtl_hits.tsv            per-QTL lead SNP + window stats (LOCO+EMMAX)
  new_locus_candidates.tsv      independent novel loci passing acceptance
  m3_2_summary_<trait>.json     acceptance + provenance
  manhattan_<trait>_qtl.png     LOCO Manhattan with QTL/novel annotations

Implementation note: M3.2 v1 does *physical* clumping only (1 Mb window). LD
r²-based fine-mapping (plink2 --r2 on the full scan BED) is M3.2-v2;
acceptance gates are written so the v1 physical-only result is auditable
and the LD step adds confidence without changing the locus call.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
_CHI2_1_MEDIAN = 0.4549364231195724
WHEAT_CHROM_ORDER = [f"chr{i}{s}" for i in range(1, 8) for s in ("A", "B", "D")]

# Thresholds (from M3.2 plan, reconciled with Codex dual-plan)
P_GENOME_WIDE = 5.0e-8                # novel candidate / significant seed
P_SUGGESTIVE = 5.0e-5                 # known QTL recovery floor
KNOWN_WINDOW_BP = 500_000             # ±500 kb around each known QTL
NOVEL_EXCLUSION_BP = 1_000_000        # novel must be >1 Mb from any known QTL
CLUMP_KB = 1_000_000                  # physical clumping window (1 Mb)


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


# =====================================================================
# Known QTL table
# =====================================================================

def load_known_qtls(path: Path) -> pd.DataFrame:
    """Load known_qtl_wheat.tsv; validate chrom names + integer coords."""
    df = pd.read_csv(path, sep="\t")
    needed = {"qtl_name", "qtl_family", "gene_symbol", "iwgsc_gene_id",
              "chrom", "start", "end", "qtl_pos", "window_bp", "qtl_class"}
    missing = needed - set(df.columns)
    if missing:
        sys.exit(f"ERR: known_qtl_wheat.tsv missing columns: {missing}")
    if df["qtl_name"].duplicated().any():
        sys.exit(f"ERR: duplicate qtl_name in {path}")
    if not df["chrom"].astype(str).str.match(r"^chr[1-7][ABD]$").all():
        sys.exit(f"ERR: chrom column not all chr[1-7][ABD] in {path}")
    for c in ("start", "end", "qtl_pos", "window_bp"):
        df[c] = pd.to_numeric(df[c], errors="raise").astype(np.int64)
    df["subgenome"] = df["chrom"].str[-1]   # A/B/D from chrXA/B/D
    return df


# =====================================================================
# Sumstats streaming
# =====================================================================

def stream_sumstats_for_chrom(path: Path, chrom: str, chunksize: int = 1_000_000):
    """Yield DataFrame chunks of `path` (gzip tsv) filtered to `chrom`.

    All-chunk-filtered approach: load only chrom column first, build the
    indices on the file, then stream. Simpler: just read by chunk and filter
    in-memory — slower IO but no two-pass.
    """
    for chunk in pd.read_csv(
            path, sep="\t",
            usecols=["snp_id", "subgenome", "chrom", "pos",
                     "beta", "se", "chi2", "p"],
            chunksize=chunksize, low_memory=False):
        m = chunk["chrom"].astype(str) == chrom
        if m.any():
            yield chunk.loc[m].copy()


def scan_known_windows_loco(
    loco_paths: dict[str, Path], qtls: pd.DataFrame,
) -> pd.DataFrame:
    """For each QTL, find the lead SNP within ±window_bp from LOCO sumstats.

    Streams the appropriate subgenome's sumstats once per QTL (chrom-filtered).
    Records min-p / max-chi2 SNP + p<5e-8 and p<5e-5 counts in the window.
    """
    rows = []
    cache: dict[str, pd.DataFrame] = {}     # chrom -> all qtl-matched rows
    qtls_by_sub: dict[str, pd.DataFrame] = {sg: g for sg, g in qtls.groupby("subgenome")}
    for sg, sg_qtls in qtls_by_sub.items():
        path = loco_paths[sg]
        unique_chroms = sg_qtls["chrom"].unique()
        for chrom in unique_chroms:
            qrows = sg_qtls[sg_qtls["chrom"] == chrom].sort_values("qtl_pos")
            min_pos = int(qrows["qtl_pos"].min() - qrows["window_bp"].max())
            max_pos = int(qrows["qtl_pos"].max() + qrows["window_bp"].max())
            # collect all SNPs on this chrom within the union range
            buf: list[pd.DataFrame] = []
            for ch in stream_sumstats_for_chrom(path, chrom):
                inrange = (ch["pos"] >= min_pos) & (ch["pos"] <= max_pos)
                if inrange.any():
                    buf.append(ch.loc[inrange, ["snp_id","chrom","pos",
                                                "beta","se","chi2","p"]].copy())
            chrom_df = (pd.concat(buf, ignore_index=True) if buf
                        else pd.DataFrame(columns=["snp_id","chrom","pos",
                                                    "beta","se","chi2","p"]))
            cache[chrom] = chrom_df
            for _, qrow in qrows.iterrows():
                qp = int(qrow["qtl_pos"])
                wbp = int(qrow["window_bp"])
                wlo, whi = qp - wbp, qp + wbp
                w = chrom_df[(chrom_df["pos"] >= wlo) & (chrom_df["pos"] <= whi)]
                if len(w) == 0:
                    rows.append({
                        "qtl_name": qrow["qtl_name"],
                        "qtl_family": qrow["qtl_family"],
                        "gene_symbol": qrow["gene_symbol"],
                        "iwgsc_gene_id": qrow["iwgsc_gene_id"],
                        "chrom": chrom, "qtl_start": int(qrow["start"]),
                        "qtl_end": int(qrow["end"]), "qtl_pos": qp,
                        "window_start": wlo, "window_end": whi,
                        "subgenome": sg,
                        "lead_snp": None, "lead_pos": None,
                        "lead_dist_to_qtl": None,
                        "lead_beta": None, "lead_se": None,
                        "lead_chi2": None, "lead_p": None,
                        "n_snps_in_window": 0,
                        "n_snps_in_window_p5e8": 0,
                        "n_snps_in_window_p5e5": 0,
                        "recovered_p5e5": False,
                        "recovered_p5e8": False,
                    })
                    continue
                # min-p SNP in window
                lead = w.loc[w["chi2"].idxmax()]
                rows.append({
                    "qtl_name": qrow["qtl_name"],
                    "qtl_family": qrow["qtl_family"],
                    "gene_symbol": qrow["gene_symbol"],
                    "iwgsc_gene_id": qrow["iwgsc_gene_id"],
                    "chrom": chrom, "qtl_start": int(qrow["start"]),
                    "qtl_end": int(qrow["end"]), "qtl_pos": qp,
                    "window_start": wlo, "window_end": whi,
                    "subgenome": sg,
                    "lead_snp": str(lead["snp_id"]),
                    "lead_pos": int(lead["pos"]),
                    "lead_dist_to_qtl": int(lead["pos"] - qp),
                    "lead_beta": float(lead["beta"]),
                    "lead_se": float(lead["se"]),
                    "lead_chi2": float(lead["chi2"]),
                    "lead_p": float(lead["p"]),
                    "n_snps_in_window": int(len(w)),
                    "n_snps_in_window_p5e8": int((w["p"] < P_GENOME_WIDE).sum()),
                    "n_snps_in_window_p5e5": int((w["p"] < P_SUGGESTIVE).sum()),
                    "recovered_p5e5": bool(lead["p"] < P_SUGGESTIVE),
                    "recovered_p5e8": bool(lead["p"] < P_GENOME_WIDE),
                })
    return pd.DataFrame(rows).sort_values(["chrom", "qtl_pos"]).reset_index(drop=True)


def collect_significant_seeds(
    loco_paths: dict[str, Path], p_thresh: float = P_GENOME_WIDE,
) -> pd.DataFrame:
    """One full pass over LOCO sumstats; return all SNPs with p < p_thresh."""
    parts: list[pd.DataFrame] = []
    for _sg, path in loco_paths.items():
        for chunk in pd.read_csv(
                path, sep="\t",
                usecols=["snp_id","subgenome","chrom","pos","beta","se","chi2","p"],
                chunksize=1_000_000, low_memory=False):
            m = chunk["p"] < p_thresh
            if m.any():
                parts.append(chunk.loc[m].copy())
    if not parts:
        return pd.DataFrame(columns=["snp_id","subgenome","chrom","pos",
                                       "beta","se","chi2","p"])
    return pd.concat(parts, ignore_index=True).sort_values(
        ["chrom","pos"]).reset_index(drop=True)


def physical_clump_seeds(
    seeds: pd.DataFrame, clump_kb: int = CLUMP_KB,
) -> pd.DataFrame:
    """Greedy physical clumping: lead = lowest-p SNP; remove all within
    ``clump_kb`` on same chrom; repeat until empty. Returns lead SNPs only.

    This is a *physical* clump (no LD). LD-based fine-mapping is M3.2-v2.
    """
    if len(seeds) == 0:
        return seeds.assign(clump_size=0, clump_min_pos=None, clump_max_pos=None)
    leads = []
    remaining = seeds.copy().sort_values("p").reset_index(drop=True)
    while len(remaining) > 0:
        lead = remaining.iloc[0]
        lead_chrom = lead["chrom"]
        lead_pos = int(lead["pos"])
        same_chrom_mask = (remaining["chrom"] == lead_chrom)
        within = same_chrom_mask & (
            (remaining["pos"] >= lead_pos - clump_kb) &
            (remaining["pos"] <= lead_pos + clump_kb)
        )
        clump = remaining.loc[within]
        leads.append({
            **lead.to_dict(),
            "clump_size": int(len(clump)),
            "clump_min_pos": int(clump["pos"].min()),
            "clump_max_pos": int(clump["pos"].max()),
            "clump_kb": int(clump_kb // 1000),
        })
        remaining = remaining.loc[~within].reset_index(drop=True)
    return pd.DataFrame(leads).sort_values(
        ["chrom","pos"]).reset_index(drop=True)


def annotate_emmax(
    leads: pd.DataFrame, emmax_paths: dict[str, Path],
) -> pd.DataFrame:
    """Look up EMMAX (M2.5-v2) p / chi2 for each LOCO lead SNP."""
    if len(leads) == 0:
        leads["emmax_chi2"] = []
        leads["emmax_p"] = []
        leads["loco_emmax_concordant_p1e5"] = []
        return leads
    needs_by_sg: dict[str, set[str]] = {}
    for _, row in leads.iterrows():
        needs_by_sg.setdefault(row["subgenome"], set()).add(str(row["snp_id"]))
    out_chi2: dict[str, float] = {}
    out_p: dict[str, float] = {}
    for sg, snp_set in needs_by_sg.items():
        path = emmax_paths[sg]
        for chunk in pd.read_csv(
                path, sep="\t",
                usecols=["snp_id","chi2","p"],
                chunksize=1_000_000, low_memory=False):
            mask = chunk["snp_id"].astype(str).isin(snp_set)
            if mask.any():
                for _, r in chunk.loc[mask, ["snp_id","chi2","p"]].iterrows():
                    out_chi2[str(r["snp_id"])] = float(r["chi2"])
                    out_p[str(r["snp_id"])] = float(r["p"])
    leads = leads.copy()
    leads["emmax_chi2"] = leads["snp_id"].astype(str).map(out_chi2)
    leads["emmax_p"] = leads["snp_id"].astype(str).map(out_p)
    leads["loco_emmax_concordant_p1e5"] = (
        leads["emmax_p"].fillna(1.0) < 1e-5
    )
    return leads


def annotate_known_qtl_distance(
    leads: pd.DataFrame, qtls: pd.DataFrame, exclusion_bp: int = NOVEL_EXCLUSION_BP,
) -> pd.DataFrame:
    """For each lead, find nearest known QTL on same chrom + distance."""
    if len(leads) == 0:
        leads["nearest_known_qtl"] = []
        leads["dist_to_nearest_known_bp"] = []
        leads["in_known_window_1mb"] = []
        return leads
    out_nearest: list[str | None] = []
    out_dist: list[int | None] = []
    out_in: list[bool] = []
    for _, row in leads.iterrows():
        chr_qtls = qtls[qtls["chrom"] == row["chrom"]]
        if len(chr_qtls) == 0:
            out_nearest.append(None)
            out_dist.append(None)
            out_in.append(False)
            continue
        dists = (chr_qtls["qtl_pos"] - int(row["pos"])).abs()
        min_idx = dists.idxmin()
        out_nearest.append(str(chr_qtls.loc[min_idx, "qtl_name"]))
        out_dist.append(int(dists[min_idx]))
        out_in.append(bool(dists[min_idx] <= exclusion_bp))
    leads = leads.copy()
    leads["nearest_known_qtl"] = out_nearest
    leads["dist_to_nearest_known_bp"] = out_dist
    leads["in_known_window_1mb"] = out_in
    return leads


def annotate_nearest_gene(
    leads: pd.DataFrame, gff3_path: Path,
) -> pd.DataFrame:
    """Nearest HC gene from IWGSC v1.1 (any subgenome, same chrom; ±500 kb)."""
    if len(leads) == 0:
        leads["nearest_gene"] = []
        leads["nearest_gene_dist_bp"] = []
        return leads
    # Build a per-chrom sorted gene index on demand (only chroms we need)
    needed_chroms = set(leads["chrom"].astype(str))
    gene_index: dict[str, np.ndarray] = {}     # chrom -> (n,3) [start,end,id-idx]
    gene_ids_by_chrom: dict[str, list[str]] = {}
    with open(gff3_path) as fh:
        for line in fh:
            if not line or line.startswith("#"):
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 9 or cols[2] != "gene":
                continue
            chrom = cols[0]
            if chrom not in needed_chroms:
                continue
            try:
                start = int(cols[3])
                end = int(cols[4])
            except ValueError:
                continue
            # extract ID=...;
            gid = ""
            for kv in cols[8].split(";"):
                if kv.startswith("ID="):
                    gid = kv[3:]
                    break
            gene_index.setdefault(chrom, []).append((start, end))
            gene_ids_by_chrom.setdefault(chrom, []).append(gid)
    # convert to ndarray + sort
    chrom_arr: dict[str, tuple[np.ndarray,np.ndarray,list[str]]] = {}
    for c, gs in gene_index.items():
        ords = np.argsort([g[0] for g in gs])
        starts = np.array([gs[i][0] for i in ords], dtype=np.int64)
        ends = np.array([gs[i][1] for i in ords], dtype=np.int64)
        ids = [gene_ids_by_chrom[c][i] for i in ords]
        chrom_arr[c] = (starts, ends, ids)
    nearest_gene: list[str | None] = []
    nearest_dist: list[int | None] = []
    for _, row in leads.iterrows():
        ch = row["chrom"]
        pos = int(row["pos"])
        if ch not in chrom_arr:
            nearest_gene.append(None)
            nearest_dist.append(None)
            continue
        starts, ends, ids = chrom_arr[ch]
        # overlap?
        ov = (starts <= pos) & (ends >= pos)
        if ov.any():
            i = int(np.argmax(ov))
            nearest_gene.append(ids[i])
            nearest_dist.append(0)
            continue
        # distance to nearest gene start or end
        before_dist = pos - ends     # >0 means SNP is downstream of gene
        before_dist[before_dist < 0] = 10**12
        after_dist = starts - pos    # >0 means SNP is upstream of gene
        after_dist[after_dist < 0] = 10**12
        bi = int(np.argmin(before_dist))
        ai = int(np.argmin(after_dist))
        if before_dist[bi] <= after_dist[ai]:
            nearest_gene.append(ids[bi])
            nearest_dist.append(int(before_dist[bi]))
        else:
            nearest_gene.append(ids[ai])
            nearest_dist.append(int(after_dist[ai]))
    leads = leads.copy()
    leads["nearest_gene"] = nearest_gene
    leads["nearest_gene_dist_bp"] = nearest_dist
    return leads


def filter_novel_candidates(leads: pd.DataFrame) -> pd.DataFrame:
    """Apply novel-locus criteria (M3.2 v1, physical-only)."""
    if len(leads) == 0:
        return leads.assign(is_novel_candidate=[], rank=[])
    novel = leads.copy()
    novel["is_novel_candidate"] = (
        (novel["p"] < P_GENOME_WIDE) &
        (~novel["in_known_window_1mb"]) &
        (novel["loco_emmax_concordant_p1e5"])
    )
    # rank by chi2 within candidates
    novel = novel.sort_values("p").reset_index(drop=True)
    novel["rank"] = np.arange(1, len(novel) + 1)
    return novel


# =====================================================================
# Manhattan with QTL annotations
# =====================================================================

def make_manhattan_qtl(
    loco_paths: dict[str, Path], qtls: pd.DataFrame,
    novel_leads: pd.DataFrame, out_path: Path, trait: str,
    plot_p_max: float = 1e-3, thin: int = 50,
):
    """Manhattan + QTL/novel annotation. Thinned for rendering."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    parts: list[pd.DataFrame] = []
    for _sg, path in loco_paths.items():
        for chunk in pd.read_csv(
                path, sep="\t",
                usecols=["chrom","pos","chi2","p"],
                chunksize=1_000_000, low_memory=False):
            sig = chunk[chunk["p"] < plot_p_max]
            bg = chunk.iloc[::thin]
            parts.append(pd.concat([sig, bg]).drop_duplicates())
    plot = pd.concat(parts, ignore_index=True)
    chrom = plot["chrom"].astype(str).to_numpy()
    present = [c for c in WHEAT_CHROM_ORDER if c in set(chrom)]
    rank = {c: i for i, c in enumerate(present)}
    mask = np.array([c in rank for c in chrom])
    chrom = chrom[mask]
    pos = plot["pos"].to_numpy(dtype=np.int64)[mask]
    p_arr = plot["p"].to_numpy(dtype=np.float64)[mask]
    logp = -np.log10(np.clip(p_arr, 1e-300, 1.0))
    span = int(pos.max()) + 1 if pos.size else 1
    order = np.argsort(np.array([rank[c] for c in chrom], dtype=np.int64) * span + pos)
    chrom = chrom[order]
    logp = logp[order]
    pos = pos[order]
    x = np.arange(chrom.size)
    fig, ax = plt.subplots(figsize=(14, 4.5))
    cols = ["#1f77b4" if rank[c] % 2 == 0 else "#9ecae1" for c in chrom]
    ax.scatter(x, logp, s=2, c=cols)
    # known QTL vertical lines
    for _, q in qtls.iterrows():
        c = q["chrom"]
        if c not in rank:
            continue
        idx = np.where(chrom == c)[0]
        if idx.size == 0:
            continue
        rel = (pos[idx] - int(q["qtl_pos"])).astype(np.int64)
        nearest = idx[np.argmin(np.abs(rel))]
        ax.axvline(x[nearest], color="red", lw=0.6, alpha=0.55, ls=":")
        ax.text(x[nearest], 0.5, str(q["qtl_name"]),
                color="red", fontsize=5, rotation=90, va="bottom", ha="right")
    # novel candidates
    for _, n in novel_leads[novel_leads["is_novel_candidate"]].iterrows():
        c = n["chrom"]
        if c not in rank:
            continue
        idx = np.where(chrom == c)[0]
        if idx.size == 0:
            continue
        rel = (pos[idx] - int(n["pos"])).astype(np.int64)
        nearest = idx[np.argmin(np.abs(rel))]
        ax.scatter([x[nearest]], [logp[nearest]], s=40,
                   facecolors="none", edgecolors="darkorange", linewidths=1.5)
        ax.text(x[nearest], logp[nearest] + 0.25, f"★{n['snp_id']}",
                fontsize=5, color="darkorange", ha="center")
    ax.axhline(-np.log10(P_GENOME_WIDE), ls="--", color="red", lw=0.8)
    ax.axhline(-np.log10(P_SUGGESTIVE), ls=":", color="grey", lw=0.5)
    ticks, labels = [], []
    for c in present:
        idx = np.where(chrom == c)[0]
        if idx.size:
            ticks.append(float(idx.mean()))
            labels.append(c.replace("chr", ""))
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_ylabel("-log10(p)")
    ax.set_title(f"M3.2 wheat LOCO Manhattan — {trait}  "
                  "(red dashed line = 5e-8; QTL ticks = known; "
                  "orange circles = novel candidates)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# Main
# =====================================================================

def main():
    ap = argparse.ArgumentParser(description="M3.2 wheat known-QTL recovery")
    ap.add_argument("--trait", default="days_to_emerg")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument(
        "--known-qtl",
        default=str(ROOT / "data/reference/wheat/known_qtl_wheat.tsv"))
    ap.add_argument("--gff3-zip",
                    default=str(ROOT / "data/reference/wheat/"
                                "iwgsc_refseqv1.1_genes_2017July06.zip"),
                    help="(optional) IWGSC v1.1 HC GFF3 zip for nearest-gene")
    ap.add_argument("--gff3-extracted",
                    default="/tmp/gff3/iwgsc_refseqv1.1_genes_2017July06/"
                            "IWGSC_v1.1_HC_20170706.gff3",
                    help="(faster) pre-extracted GFF3 path")
    args = ap.parse_args()
    out_dir = (Path(args.out_dir) if args.out_dir
               else ROOT / "results/phase3/m3_2_qtl/wheat_watkins")
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print(f"=== M3.2 wheat known-QTL recovery — trait={args.trait} ===")
    qtls = load_known_qtls(Path(args.known_qtl))
    print(f"[1] loaded {len(qtls)} known QTLs: {qtls['qtl_family'].value_counts().to_dict()}")

    loco_paths = {
        "A": ROOT / "results/phase3/m3_1_loco_v2/wheat_watkins/scan_loco_A.tsv.gz",
        "B": ROOT / "results/phase3/m3_1_loco_v2/wheat_watkins/scan_loco_B.tsv.gz",
        "D": ROOT / "results/phase3/m3_1_loco_v2/wheat_watkins/scan_loco_D.tsv.gz",
    }
    emmax_paths = {
        "A": ROOT / "results/phase2/m2_5_v2/wheat_watkins/scan_A.tsv.gz",
        "B": ROOT / "results/phase2/m2_5_v2/wheat_watkins/scan_B.tsv.gz",
        "D": ROOT / "results/phase2/m2_5_v2/wheat_watkins/scan_D.tsv.gz",
    }
    for _sg, p in loco_paths.items():
        if not p.exists():
            sys.exit(f"ERR: missing LOCO sumstats: {p}")
    for _sg, p in emmax_paths.items():
        if not p.exists():
            sys.exit(f"ERR: missing EMMAX baseline sumstats: {p}")

    print(f"[2] known-QTL window scan (LOCO, ±{KNOWN_WINDOW_BP//1000} kb)")
    known_hits = scan_known_windows_loco(loco_paths, qtls)
    known_hits = annotate_emmax(known_hits.rename(columns={
        "lead_snp": "snp_id", "lead_pos": "pos",
        "lead_chi2": "chi2", "lead_p": "p",
        "lead_beta": "beta", "lead_se": "se",
    }), emmax_paths).rename(columns={
        "snp_id": "lead_snp", "pos": "lead_pos",
        "chi2": "lead_chi2", "p": "lead_p",
        "beta": "lead_beta", "se": "lead_se",
    })
    n_recovered_p5e5 = int(known_hits["recovered_p5e5"].fillna(False).sum())
    n_recovered_p5e8 = int(known_hits["recovered_p5e8"].fillna(False).sum())
    print(f"  known recovery: {n_recovered_p5e5}/{len(known_hits)} at p<5e-5, "
          f"{n_recovered_p5e8} at p<5e-8")

    print(f"[3] genome-wide significant seeds (p<{P_GENOME_WIDE:.0e})")
    seeds = collect_significant_seeds(loco_paths, p_thresh=P_GENOME_WIDE)
    print(f"  collected {len(seeds)} significant seeds")

    print(f"[4] physical clumping (±{CLUMP_KB//1000} kb)")
    leads = physical_clump_seeds(seeds, clump_kb=CLUMP_KB)
    print(f"  {len(leads)} independent leads after physical clumping")

    print("[5] EMMAX baseline + known-QTL distance + nearest gene annotation")
    leads = annotate_emmax(leads, emmax_paths)
    leads = annotate_known_qtl_distance(leads, qtls, exclusion_bp=NOVEL_EXCLUSION_BP)
    gff3 = Path(args.gff3_extracted)
    if gff3.exists():
        leads = annotate_nearest_gene(leads, gff3)
    else:
        print(f"  WARN: GFF3 not at {gff3}, skipping nearest-gene "
              "(unzip the iwgsc_refseqv1.1_genes_*.zip to /tmp/gff3/...)")
        leads["nearest_gene"] = None
        leads["nearest_gene_dist_bp"] = None

    novel = filter_novel_candidates(leads)
    n_novel = int(novel["is_novel_candidate"].sum())
    print(f"  novel candidates: {n_novel} pass "
          f"(p<5e-8, dist>{NOVEL_EXCLUSION_BP//1000}kb to known, EMMAX p<1e-5)")

    # lambda_GC sanity check using significant + small thinned background
    print("[6] lambda_GC sanity check")
    chi2_parts = []
    for _sg, p in loco_paths.items():
        for chunk in pd.read_csv(p, sep="\t", usecols=["chi2"],
                                 chunksize=2_000_000, low_memory=False):
            chi2_parts.append(chunk["chi2"].to_numpy(dtype=np.float64))
    all_chi2 = np.concatenate(chi2_parts)
    lam_gc = float(np.median(all_chi2) / _CHI2_1_MEDIAN)
    print(f"  λ_GC (recomputed) = {lam_gc:.4f}")

    print("[7] writing outputs")
    out_hits = out_dir / "known_qtl_hits.tsv"
    known_hits.to_csv(out_hits, sep="\t", index=False)
    out_novel = out_dir / "new_locus_candidates.tsv"
    novel.to_csv(out_novel, sep="\t", index=False)
    out_leads_all = out_dir / "all_significant_leads.tsv"
    leads.to_csv(out_leads_all, sep="\t", index=False)
    print(f"  wrote {out_hits}")
    print(f"  wrote {out_novel}")
    print(f"  wrote {out_leads_all}")

    print("[8] Manhattan with QTL annotations")
    manhattan_path = out_dir / f"manhattan_{args.trait}_qtl.png"
    make_manhattan_qtl(loco_paths, qtls, novel, manhattan_path, args.trait)
    print(f"  wrote {manhattan_path}")

    # acceptance
    runtime = time.time() - t0
    acceptance = []
    def check(name, ok, msg=""):
        acceptance.append({"check": name, "passed": bool(ok), "message": msg})
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {msg}" if msg else ""))
    print("\n[9] acceptance gates")
    check("at_least_3_known_p5e5_recovered",
          n_recovered_p5e5 >= 3,
          f"{n_recovered_p5e5}/{len(known_hits)} known QTL recovered at p<5e-5")
    check("at_least_1_novel_candidate",
          n_novel >= 1,
          f"{n_novel} novel candidates pass full gate")
    check("lambda_gc_in_range",
          0.9 <= lam_gc <= 1.15,
          f"λ_GC={lam_gc:.4f} (expect ≈ 1.085 from M3.1-v2)")
    all_passed = all(c["passed"] for c in acceptance)

    summary = {
        "script": "run_m3_2_wheat_qtl.py", "milestone": "M3.2",
        "version": "v1 (physical clumping; LD r² fine-mapping is v2)",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "trait": args.trait, "runtime_sec": round(runtime, 1),
        "thresholds": {
            "p_genome_wide": P_GENOME_WIDE, "p_suggestive": P_SUGGESTIVE,
            "known_window_bp": KNOWN_WINDOW_BP,
            "novel_exclusion_bp": NOVEL_EXCLUSION_BP,
            "clump_kb": CLUMP_KB},
        "known_qtl_count": int(len(qtls)),
        "known_recovery": {
            "n_qtl": int(len(known_hits)),
            "n_recovered_p5e5": n_recovered_p5e5,
            "n_recovered_p5e8": n_recovered_p5e8,
        },
        "significant_seeds": int(len(seeds)),
        "independent_leads_physical": int(len(leads)),
        "novel_candidates": int(n_novel),
        "lambda_gc_recomputed": lam_gc,
        "outputs": {
            "known_qtl_hits": str(out_hits),
            "new_locus_candidates": str(out_novel),
            "all_significant_leads": str(out_leads_all),
            "manhattan": str(manhattan_path),
        },
        "acceptance": acceptance,
        "acceptance_all_passed": all_passed,
        "follow_up": "M3.2-v2: plink2 --r2 LD fine-mapping on full BED, "
                     "per-env targeted validation (CFLN06/10/14/20)",
    }
    summary_path = out_dir / f"m3_2_summary_{args.trait}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=_json_default)
    print(f"\nwrote {summary_path}")
    n_pass = sum(c["passed"] for c in acceptance)
    print(f"acceptance: {n_pass}/{len(acceptance)} gates passed  "
          f"(runtime {runtime:.0f}s)")
    if all_passed:
        print("✅ M3.2 wheat QTL recovery (v1) acceptance PASS")
        return 0
    failed = [c['check'] for c in acceptance if not c['passed']]
    print(f"❌ M3.2 acceptance FAIL — {failed}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
