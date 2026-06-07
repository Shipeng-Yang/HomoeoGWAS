# Phase 3 — Rapeseed ZS11 reference (manual browser download)

> Run this once. It is the only manual piece of Phase 3 — the 19 VCFs and the
> phenotype table are handled by `phase3_rapeseed.sh`.

## Why this is manual

1. **It must be ZS11, not Darmor-bzh.** The BnIR panel files are
   `bna2311_ZS11_SNP.*` — every SNP position/allele is called against the
   **ZS11** assembly with BnIR's `BnaA01g…`/`BnaC09g…` gene IDs. Using any
   other B. napus assembly (Darmor-bzh v4.1/v2.0/v10, Da-Ae, …) means every
   coordinate and gene annotation downstream is silently wrong. No liftover —
   take ZS11 directly.
2. **NCBI is blocked from mainland China**, and NCBI's `GCF_000686985.2` is
   Darmor-bzh anyway (wrong assembly). So we pull ZS11 from BnIR's own
   genome page — a China-hosted server (`yanglab.hzau.edu.cn`), the same host
   the VCFs come from, so coordinates are guaranteed to match.
3. BnIR's genome download is behind its web UI (not a stable static path we
   can script), so it's a browser click.

Target directory on the NAS:

```
/volume2/U7_GWAS/data/reference/rapeseed/
```

---

## Option A (recommended) — China-hosted direct download pages

`https://yanglab.hzau.edu.cn/BnIR/genome_data` is a JS single-page app and
often will not open. Use one of these instead (all China-hosted, all the
same Sun et al. 2017 HZAU ZS11):

1. **NGDC GWH** (National Genomics Data Center — fastest in China, has
   explicit Download buttons for FASTA + GFF):
   - **<https://ngdc.cncb.ac.cn/gwh/Assembly/9669/show>**
   - Accession `GWHANRE00000000`, BioProject `PRJCA002883`.
2. **BnPIR download page** (HZAU; a plain file listing, not a JS app):
   - **<http://cbi.hzau.edu.cn/cgi-bin/rape/download_ext>**
   - Grab the `ZS11` rows: `ZS11.genome.fa` + `ZS11.annotation.gff3`
     (also `ZS11.protein.pep` / `.cds` optionally).
3. **CNGBdb** (BGI China National GeneBank — backup):
   - **<https://db.cngb.org/search/?q=Brassica%20napus%20ZS11>**

Download from ONE of the above (prefer 1 or 2), then put the files in
`/volume2/U7_GWAS/data/reference/rapeseed/` (DSM File Station upload, or
`scp` from the machine that fetched them).

---

## Option B (fallback, only if A is unreachable) — ENA mirror

EBI/ENA (`ftp.ebi.ac.uk`) **is reachable from China** — it is the same host
the wheat Watkins VCFs downloaded from. The ZS11 assembly is on ENA as
**`GCA_001746955.1`** (ASM174695v1, *B. napus* cv. ZS11, Sun et al. 2017).

```bash
ROOT=/volume2/U7_GWAS
ARIA="-Z -x 16 -s 16 -j 1 -c --auto-file-renaming=false --allow-overwrite=false"

# Confirm the assembly name says "ZS11" before trusting it:
#   https://www.ebi.ac.uk/ena/browser/view/GCA_001746955.1
aria2c $ARIA -d $ROOT/data/reference/rapeseed \
  "https://ftp.ebi.ac.uk/pub/databases/ena/assembly/GCA_001/GCA_001746/GCA_001746955.1.fasta.gz"
```

**Caveat:** the ENA GenBank copy may use different sequence names (e.g.
`CMxxxxxx` accessions) and carries no BnIR gene model. If you go this route
you may need a chromosome-name map (`A01↔CMxxxxxx`) before the VCFs line up.
Option A avoids this entirely — prefer it whenever BnIR/BnPIR is reachable.

---

## Do NOT use

| Assembly | Why not |
|---|---|
| `GCF_000686985.2` Bra_napus_v2.0 (NCBI) | Darmor-bzh, **not ZS11** — wrong coordinates; also NCBI blocked from China |
| `GCA_905183035.1` darmor_bzh_v10 | Darmor-bzh long-read — wrong coordinates for these VCFs |
| `GCF_020379485.1` Da-Ae | Different cultivar |

---

## Verify after download

```bash
cd /volume2/U7_GWAS/data/reference/rapeseed
ls -lh

# FASTA headers should be the 19 B. napus chromosomes (A01..A10, C01..C09):
zcat *.fa*.gz | grep '^>' | head -25

# Chromosome IDs in the FASTA must match the ##contig / CHROM field of:
zcat /volume2/U7_GWAS/data/raw/rapeseed/vcf/bna2311_ZS11_SNP.A01.vcf.gz \
  | grep -m1 -v '^##' | cut -f1
# ^ this CHROM token must appear as a FASTA header above. If it does not,
#   you downloaded the wrong assembly — go back to Option A.
```

When the CHROM tokens match the FASTA headers, Phase 3 is complete and
`scripts/preprocess/subset_napus_991.sh` (TBW) can proceed.
