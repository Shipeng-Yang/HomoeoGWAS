"""Fill chrom_map_oat.tsv fasta_chrom column with Ensembl 1A/1C/1D names.

Phase 5d Step 2 helper. After Ensembl Plants OT3098 v2 fasta is downloaded
(with chromosome headers 1A..7A / 1C..7C / 1D..7D), the panel VCF chrom
"1A_0" / "1A_1" both map to fasta chrom "1A" (i.e. segments collapse to
logical chrom for fasta lookup).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

CHROM_MAP_PATH = Path("/mnt/7302share/fast_ysp/U7_GWAS/data/reference/oat/chrom_map_oat.tsv")

cm = pd.read_csv(CHROM_MAP_PATH, sep="\t")
# Ensembl OT3098 v2 uses standard 1A/1C/1D names (per Kamal 2022 + GrainGenes
# canonical nomenclature). Segment label _0 / _1 are panel-only; fasta carries
# the whole logical chromosome.
cm["fasta_chrom"] = cm["logical_chrom"]    # 1A_0 / 1A_1 → 1A; UN → UN
# UN entries: fasta has no exact "UN" chrom — Ensembl puts them under
# nonchromosomal contigs in dna.nonchromosomal.fa.gz. Set to empty so
# downstream (DL prior fasta lookup) skips UN SNPs explicitly.
cm.loc[cm["panel_chrom"] == "UN", "fasta_chrom"] = ""

cm.to_csv(CHROM_MAP_PATH, sep="\t", index=False)
print(f"[chrom_map] updated {CHROM_MAP_PATH}")
print(cm.to_string(index=False))
print()
print("[verify] unique fasta_chrom values:", sorted(cm["fasta_chrom"].unique()))
print("[verify] panel_chrom count (expect 42 + UN = 43):", len(cm))
