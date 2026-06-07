#!/usr/bin/env python
"""Phase 5e — build the RiceVarMap2 diploid panel (genotype subset + pheno +
anchors + chrom_map) from the local regubreed RiceVarMap2 resources.

DIPLOID generalisation panel (supplementary): validates that the DL-prior
re-ranking transfers beyond allopolyploids. Ploidy/subgenome machinery is
INACTIVE here (n_sub=1); see memory dl_prior_positioning.

Inputs (local, regubreed):
  rice4k.{bed,bim,fam}  RiceVarMap2 4726 acc x 17.4M SNP (IRGSP-1.0, chrom 1..12)
  phenos.csv            529 acc x 10 traits (id_name C001.. matches FAM IID)
Outputs (U7_GWAS):
  data/processed/rice/ALL/all.{bed,bim,fam}   529 x ~4.69M SNP (MAF>=.05, geno<=.1)
  data/processed/rice/pheno_clean.tsv         FID IID + 4 traits (-9 -> NA)
  data/reference/rice/known_qtl_rice_<trait>.tsv  cloned-gene anchors (IRGSP GFF coords)
  data/reference/rice/chrom_map_rice.tsv      identity 1..12, subgenome ALL

Reference FASTA + GFF (Ensembl release-57, chrom "1".."12"):
  /mnt/nvme/rice_ref/Oryza_sativa.IRGSP-1.0.dna.toplevel.fa
  /mnt/nvme/rice_ref/Oryza_sativa.IRGSP-1.0.57.gff3.gz

Run (geno subset uses plink2; do it once):
  plink2 --bfile <rice4k> --keep keep_529.txt --maf 0.05 --geno 0.1 \
         --set-all-var-ids '@:#' --rm-dup force-first --make-bed \
         --out data/processed/rice/ALL/all
  python scripts/phase5e/prepare_rice_panel.py   # pheno + anchors + chrom_map
"""
import gzip
import re
from pathlib import Path

import numpy as np
import pandas as pd

REGU = Path("/mnt/7302share/fast_ysp/tools/regubreed/data/external/ricevarmap2")
ROOT = Path("/mnt/7302share/fast_ysp/U7_GWAS")
GFF = Path("/mnt/nvme/rice_ref/Oryza_sativa.IRGSP-1.0.57.gff3.gz")

TRAITS = ["Heading_date", "Grain_length", "Grain_width", "Plant_height"]

# Canonical cloned rice genes per trait (RAP-DB Os locus IDs). Coordinates are
# resolved from the Ensembl IRGSP-1.0 GFF3 (gene_id=Os...g).
ANCHORS = {
    "Heading_date": [
        ("Hd1", "Os06g0275000"), ("Hd3a", "Os06g0157700"), ("RFT1", "Os06g0157500"),
        ("Ghd7", "Os07g0261200"), ("Ghd7.1", "Os07g0695100"), ("Hd5", "Os08g0174500"),
        ("Hd6", "Os03g0762000"), ("Ehd1", "Os10g0463400"), ("OsMADS50", "Os03g0122600"),
        ("OsMADS51", "Os01g0922800"), ("DTH2", "Os02g0724000"), ("Hd16", "Os03g0793500"),
    ],
    "Grain_length": [
        ("GS3", "Os03g0407400"), ("GW7", "Os07g0603300"), ("GL3.1", "Os03g0646900"),
        ("TGW6", "Os06g0623700"), ("GW6a", "Os06g0650300"),
    ],
    "Grain_width": [
        ("GW2", "Os02g0244100"), ("GW5", "Os05g0187500"), ("GS5", "Os05g0158500"),
        ("TGW2", "Os02g0763000"), ("GW8", "Os08g0531600"),
    ],
    "Plant_height": [
        ("sd1", "Os01g0883800"), ("d18", "Os01g0177400"), ("OsGA20ox1", "Os03g0856700"),
    ],
}
WINDOW_BP = 500_000


def build_pheno():
    ph = pd.read_csv(REGU / "phenos.csv")
    fam = pd.read_csv(REGU / "rice4k.fam", sep=r"\s+", header=None,
                      names=["FID", "IID", "x1", "x2", "x3", "x4"])
    iid2fid = dict(zip(fam["IID"].astype(str), fam["FID"].astype(str), strict=True))
    ph["IID"] = ph["id_name"].astype(str)          # id_name (C001..) matches FAM IID
    ph = ph[ph["IID"].isin(iid2fid)].copy()
    ph["FID"] = ph["IID"].map(iid2fid)
    for c in TRAITS:
        ph[c] = ph[c].replace(-9, np.nan)           # -9 = missing sentinel
    out = ph[["FID", "IID"] + TRAITS]
    (ROOT / "data/processed/rice").mkdir(parents=True, exist_ok=True)
    ph[["FID", "IID"]].to_csv(ROOT / "data/processed/rice/keep_529.txt",
                              sep="\t", index=False, header=False)
    out.to_csv(ROOT / "data/processed/rice/pheno_clean.tsv",
               sep="\t", index=False, na_rep="NA")
    print(f"pheno: {len(out)} acc x {len(TRAITS)} traits")


def parse_gff_coords():
    coords = {}
    with gzip.open(GFF, "rt") as f:
        for ln in f:
            if ln.startswith("#"):
                continue
            c = ln.rstrip("\n").split("\t")
            if len(c) < 9 or c[2] != "gene":
                continue
            m = re.search(r"gene_id=([^;]+)", c[8])
            if m:
                coords[m.group(1)] = (c[0], int(c[3]), int(c[4]), c[6])
    return coords


def build_anchors():
    coords = parse_gff_coords()
    hard = {"GW5": ("5", 5371949, 5374705, "+")}     # Os05g0187500 fallback
    refdir = ROOT / "data/reference/rice"
    refdir.mkdir(parents=True, exist_ok=True)
    for trait, genes in ANCHORS.items():
        rows = []
        for sym, osid in genes:
            if osid in coords:
                ch, st, en, sd = coords[osid]
            elif sym in hard:
                ch, st, en, sd = hard[sym]
            else:
                print(f"  WARN miss {sym} ({osid})")
                continue
            if ch not in [str(i) for i in range(1, 13)]:
                continue
            rows.append(dict(
                qtl_name=sym, qtl_family=sym, gene_symbol=sym, gene_id=osid,
                chrom=ch, start=st, end=en, qtl_pos=(st + en) // 2, strand=sd,
                coord_basis="IRGSP-1.0_Ensembl_GFF3_RAP", source_primary="funRiceGenes",
                source_secondary="RAP-DB_IRGSP1.0", window_bp=WINDOW_BP,
                qtl_class=trait, trait_relevance=trait, notes=f"cloned {trait} gene"))
        pd.DataFrame(rows).to_csv(
            refdir / f"known_qtl_rice_{trait}.tsv", sep="\t", index=False)
        print(f"{trait}: {len(rows)} anchors")
    # chrom_map: identity 1..12, single subgenome ALL (diploid)
    pd.DataFrame({"panel_chrom": [str(i) for i in range(1, 13)],
                  "fasta_chrom": [str(i) for i in range(1, 13)],
                  "subgenome": ["ALL"] * 12}).to_csv(
        refdir / "chrom_map_rice.tsv", sep="\t", index=False)
    print("chrom_map: identity 1..12, subgenome ALL")


if __name__ == "__main__":
    build_pheno()
    build_anchors()
