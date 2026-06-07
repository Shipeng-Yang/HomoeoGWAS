#!/usr/bin/env python3
"""M3.4 v2 — cotton known QTL coords: NAU v1.1 (Gh_*) → NDM8 (GhM_*) via BLASTP.

Strategy: SAME-SUBGENOME BEST HIT (homoeolog A/D pairs in polyploid cotton can
score nearly identically across A↔D; global best by bitscore frequently selects
the wrong subgenome). Force best hit to come from the query's own subgenome;
fall back to global best (flagged) only when no same-sub hit exists.

Tier scheme:
  PASS          : same-sub + pident>=90 + qcov>=80
  PARTIAL       : same-sub + pident>=90 + qcov<80  (kept; gene fragment / partial CDS in NDM8)
  LOW_PIDENT    : same-sub + pident<90            (kept; flagged)
  CROSS_SUB     : no same-sub hit; global fallback (kept; flagged)
  DROPPED       : query missing from NAU pep (no BLAST possible)

All non-DROPPED rows are emitted with NDM8 coords. Acceptance gate: ≥8 of 10
anchors get a coord (any non-DROPPED tier).
"""
import gzip
import shutil
import subprocess
import time
from pathlib import Path

import pandas as pd

ROOT = Path("/mnt/7302share/fast_ysp/U7_GWAS")
WORK = Path("/mnt/nvme/cotton_hbau/cottongen")
NAU_PEP_GZ  = WORK / "NAU_v1.1.pep.fa.gz"
NDM8_PEP_GZ = WORK / "NDM8.pep.gz"
NDM8_GFF_GZ = WORK / "NDM8.gff3.gz"
KNOWN_TSV   = ROOT / "data/reference/cotton/known_qtl_cotton.tsv"
BACKUP_TSV  = ROOT / "data/reference/cotton/known_qtl_cotton.previous_NAU.tsv"
AUDIT_TSV   = ROOT / "data/reference/cotton/m3_4_v2_blastp_audit.tsv"
CHROM_MAP   = ROOT / "data/reference/cotton/chrom_map_hebau_to_ncbi.tsv"

BLAST_BIN_DIR = Path.home() / ".local/share/mamba/envs/polygwas-cpu/bin"
BLASTP   = BLAST_BIN_DIR / "blastp"
MAKEBDB  = BLAST_BIN_DIR / "makeblastdb"


def load_anchors():
    # Prefer pristine backup if exists (it's the original NAU literature version);
    # else read current TSV.
    src = BACKUP_TSV if BACKUP_TSV.exists() else KNOWN_TSV
    print(f"Loading anchors from {src.name}")
    return pd.read_csv(src, sep="\t")


def extract_query_pep(anchors):
    keep = set(anchors["gene_id"])
    out = WORK / "anchors_query.pep.fa"
    found = set()
    with gzip.open(NAU_PEP_GZ, "rt") as fin, out.open("w") as fout:
        keep_now = False
        for line in fin:
            if line.startswith(">"):
                gid = line[1:].strip().split()[0]
                keep_now = gid in keep
                if keep_now:
                    found.add(gid)
            if keep_now:
                fout.write(line)
    missing = keep - found
    return out, found, missing


def build_ndm8_blast_db():
    db_pep = WORK / "NDM8.pep.fa"
    if not db_pep.exists():
        with gzip.open(NDM8_PEP_GZ, "rt") as fin, db_pep.open("w") as fout:
            shutil.copyfileobj(fin, fout)
    db_prefix = WORK / "NDM8_blastdb"
    if not Path(str(db_prefix) + ".phr").exists():
        subprocess.run([str(MAKEBDB), "-in", str(db_pep), "-dbtype", "prot",
                        "-out", str(db_prefix), "-title", "NDM8_AD1_pep"],
                       check=True, capture_output=True)
    return db_prefix


def run_blastp(query, db_prefix):
    out_tsv = WORK / "blastp_anchors_vs_NDM8.tsv"
    cmd = [str(BLASTP), "-query", str(query), "-db", str(db_prefix),
           "-outfmt", "6 qseqid sseqid pident length qcovs evalue bitscore",
           "-evalue", "1e-30",
           "-max_target_seqs", "20",
           "-num_threads", "16",
           "-out", str(out_tsv)]
    subprocess.run(cmd, check=True)
    cols = ["qseqid","sseqid","pident","length","qcovs","evalue","bitscore"]
    return pd.read_csv(out_tsv, sep="\t", names=cols)


def query_subgenome(qid: str) -> str:
    # Gh_A12G2570 → 'A'
    try:
        return qid.split("_")[1][0]
    except Exception:
        return "?"


def target_subgenome(sseqid: str) -> str:
    # GhM_A12G2570.1 → 'A'; GhM_Scaf085G241.1 → 'S' (scaffold; unanchored)
    body = sseqid.replace("GhM_", "").split(".")[0]
    return body[0] if body else "?"


def pick_same_subgenome_best(blast_df: pd.DataFrame) -> pd.DataFrame:
    """Per query: best (highest bitscore) hit restricted to same subgenome;
    if none, fall back to global best (flagged)."""
    df = blast_df.copy()
    df["target_sub"] = df["sseqid"].map(target_subgenome)
    rows = []
    for q, sub in df.groupby("qseqid"):
        qsub = query_subgenome(q)
        same = sub[sub["target_sub"] == qsub].sort_values(
            ["bitscore","evalue"], ascending=[False, True])
        if len(same):
            r = same.iloc[0].to_dict()
            r["chosen_strategy"] = "same_subgenome_best"
        else:
            r = sub.sort_values(["bitscore","evalue"], ascending=[False, True]).iloc[0].to_dict()
            r["chosen_strategy"] = "cross_subgenome_fallback"
        rows.append(r)
    return pd.DataFrame(rows)


def parse_ndm8_gff_for_genes(gene_ids):
    """Return DF chrom/start/end/strand keyed by gene_id (GhM_*G*)."""
    keep = set(gene_ids)
    rows = []
    with gzip.open(NDM8_GFF_GZ, "rt") as fin:
        for line in fin:
            if line.startswith("#") or not line.strip():
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9 or f[2] != "gene":
                continue
            attrs = dict(kv.split("=",1) for kv in f[8].split(";") if "=" in kv)
            gid = attrs.get("ID","")
            if gid in keep:
                rows.append({"gene_id_ndm8": gid, "chrom": f[0],
                             "start": int(f[3]), "end": int(f[4]), "strand": f[6]})
    return pd.DataFrame(rows)


def main():
    anchors = load_anchors()
    print(f"Loaded {len(anchors)} anchors from source TSV")
    query_pep, found, missing = extract_query_pep(anchors)
    print(f"Extracted {len(found)} query peps; missing: {sorted(missing)}")

    db_prefix = build_ndm8_blast_db()
    print(f"NDM8 BLAST db ready at {db_prefix}")
    t0 = time.time()
    blast_df = run_blastp(query_pep, db_prefix)
    print(f"BLASTP done in {time.time()-t0:.1f}s, {len(blast_df)} HSPs")

    best = pick_same_subgenome_best(blast_df)
    best["ndm8_gene_id"] = best["sseqid"].str.replace(r"\.\d+$","",regex=True)
    print(f"\nBest hits (same-sub preferred): {len(best)}")

    gff_df = parse_ndm8_gff_for_genes(best["ndm8_gene_id"].tolist())
    print(f"GFF coords resolved: {len(gff_df)}/{len(best)}")
    merged = best.merge(gff_df, left_on="ndm8_gene_id", right_on="gene_id_ndm8", how="left")
    merged["query_subgenome"]  = merged["qseqid"].map(query_subgenome)
    merged["target_subgenome"] = merged["chrom"].str[0]
    merged["subgenome_consistent"] = merged["query_subgenome"] == merged["target_subgenome"]

    def tier(r):
        if r["chosen_strategy"] == "cross_subgenome_fallback":
            return "CROSS_SUB"
        if r["pident"] < 90:
            return "LOW_PIDENT"
        if r["qcovs"] < 80:
            return "PARTIAL"
        return "PASS"
    merged["liftover_tier"] = merged.apply(tier, axis=1)

    audit_cols = ["qseqid","sseqid","ndm8_gene_id","pident","length","qcovs",
                  "evalue","bitscore","chrom","start","end","strand",
                  "query_subgenome","target_subgenome",
                  "subgenome_consistent","chosen_strategy","liftover_tier"]
    audit = merged[audit_cols].copy()
    audit.to_csv(AUDIT_TSV, sep="\t", index=False)
    print(f"\nAudit written: {AUDIT_TSV}")
    print(audit.to_string(index=False))

    anchors_indexed = anchors.set_index("gene_id")
    out_rows = []
    for _, r in merged.iterrows():
        orig = anchors_indexed.loc[r["qseqid"]]
        midpoint = (int(r["start"]) + int(r["end"])) // 2
        notes = (f"NDM8 ortholog={r['ndm8_gene_id']}; pident={r['pident']:.1f} "
                 f"qcov={r['qcovs']:.0f} length={int(r['length'])} "
                 f"bitscore={r['bitscore']:.0f} strategy={r['chosen_strategy']} "
                 f"tier={r['liftover_tier']}")
        out_rows.append({
            "qtl_name": orig["qtl_name"], "qtl_family": orig["qtl_family"],
            "gene_symbol": orig["gene_symbol"], "gene_id": r["qseqid"],
            "chrom": r["chrom"], "start": int(r["start"]), "end": int(r["end"]),
            "qtl_pos": midpoint, "strand": r["strand"],
            "coord_basis": "NDM8_HEAU_official_GFF3_via_BLASTP_same_sub",
            "source_primary": "NDM8_HEAU_ASM1899796v1",
            "source_secondary": f"NAU_pep_{r['qseqid']}->NDM8_pep_{r['ndm8_gene_id']}",
            "window_bp": 500000, "qtl_class": orig["qtl_class"],
            "trait_relevance_fibre": orig["trait_relevance_fibre"],
            "notes": notes,
        })

    # Append rows for anchors absent from NAU pep (only Gh_A11G3340 expected)
    for gid in missing:
        if gid in anchors_indexed.index:
            orig = anchors_indexed.loc[gid]
            out_rows.append({
                "qtl_name": orig["qtl_name"], "qtl_family": orig["qtl_family"],
                "gene_symbol": orig["gene_symbol"], "gene_id": gid,
                "chrom": "NA", "start": -1, "end": -1, "qtl_pos": -1, "strand": ".",
                "coord_basis": "DROPPED",
                "source_primary": "NDM8_HEAU_ASM1899796v1",
                "source_secondary": "MISSING_FROM_NAU_v1.1_pep",
                "window_bp": 500000, "qtl_class": orig["qtl_class"],
                "trait_relevance_fibre": orig["trait_relevance_fibre"],
                "notes": "literature gene_id not present in NAU NBI v1.1 pep (A11 max=Gh_A11G3285); excluded from M3.2 known recovery",
            })

    out_df = pd.DataFrame(out_rows)
    if not BACKUP_TSV.exists():
        shutil.copy2(KNOWN_TSV, BACKUP_TSV)
    out_df.to_csv(KNOWN_TSV, sep="\t", index=False)
    print(f"\nNew known_qtl_cotton.tsv written: {len(out_df)} rows")
    tier_counts = out_df["coord_basis"].value_counts().to_dict()
    print(f"  by coord_basis: {tier_counts}")
    tier_audit = audit["liftover_tier"].value_counts().to_dict()
    print(f"  by tier (audit): {tier_audit}")

    coord_n = (out_df["coord_basis"] != "DROPPED").sum()
    print(f"\nAcceptance: coord_n={coord_n}/10 (gate ≥8)")
    assert coord_n >= 8, f"Only {coord_n} coords, gate failed"
    print("ACCEPTANCE_OK")


if __name__ == "__main__":
    main()
