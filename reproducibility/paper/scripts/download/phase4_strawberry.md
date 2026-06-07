# Phase 4 — Strawberry (manual download instructions)

> **Priority**: P2 (deferred to v2 — see `docs/01_project_plan.md`). Not needed for v1 main paper. Read this when v2 stress-test approaches (~M9).

Unlike the other three species, **none of the canonical strawberry data sources offer an open, scriptable HTTP/FTP path**. Each requires either a license click-through or NCBI account/Aspera. This file walks you through the manual steps and lists confirmed alternatives that ARE auto-downloadable.

---

## A. Reference genome — choose ONE of three options

### Option A1 (recommended) — Royal Royce FaRR1 v1.0 (manual, GDR)

The reference used by the Hardigan/Feldmann UC Davis program and most recent strawberry GWAS work.

1. Visit **<https://www.rosaceae.org/Analysis/12335030>** in a browser (the page is behind a soft auth that blocks `wget`/`aria2c`).
2. Accept the **Ft. Lauderdale Agreement** click-through.
3. Download files into `/volume2/U7_GWAS/data/reference/strawberry/`:
   - `Fragaria_x_ananassa_Royal_Royce_Genome_v1.0.fasta.gz` (~800 MB)
   - `Fragaria_x_ananassa_Royal_Royce_Genome_v1.0.gene.gff3.gz`
   - `Fragaria_x_ananassa_Royal_Royce_Genome_v1.0.gene.cds.fasta.gz`
4. After download, verify:
   ```bash
   cd /volume2/U7_GWAS/data/reference/strawberry
   zcat Fragaria_x_ananassa_Royal_Royce_Genome_v1.0.fasta.gz | head -1
   # Expect a fasta header line; no error
   ```

### Option A2 (auto-downloadable) — Florida Brilliance FaFM1 (NCBI)

University of Florida 2024 haplotype-resolved assembly. Fully open on NCBI. Less commonly used as a canonical reference but methodologically equivalent.

```bash
ROOT=/volume2/U7_GWAS
mkdir -p $ROOT/data/reference/strawberry
ARIA="-x 16 -s 16 -c --auto-file-renaming=false --allow-overwrite=false"

# Haplotype 1
aria2c $ARIA -d $ROOT/data/reference/strawberry \
  "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/049/309/125/GCA_049309125.1_FaFM1_hap1/GCA_049309125.1_FaFM1_hap1_genomic.fna.gz" \
  "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/049/309/125/GCA_049309125.1_FaFM1_hap1/GCA_049309125.1_FaFM1_hap1_genomic.gff.gz"

# Haplotype 2 (optional)
aria2c $ARIA -d $ROOT/data/reference/strawberry \
  "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/049/308/895/GCA_049308895.1_FaFM1_hap2/GCA_049308895.1_FaFM1_hap2_genomic.fna.gz"
```

Use this if Option A1 is blocked or you want a fully scripted pipeline.

### Option A3 (Chinese alternative) — Yuexin haplotype-resolved (Zhejiang U, NCBI)

```bash
aria2c $ARIA -d $ROOT/data/reference/strawberry \
  "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/045/516/675/GCA_045516675.1_fana_Yuexin_hap1/GCA_045516675.1_fana_Yuexin_hap1_genomic.fna.gz"
```

---

## B. GWAS panel data — choose ONE

### Option B1 (recommended) — Hardigan 2020/2021 octoploid WGS panel (NCBI SRA)

145 octoploid Fragaria accessions (99 *F.×ananassa* + 24 *F. chiloensis* + 22 *F. virginiana*).

- **BioProject**: PRJNA578384
- **Access**: raw fastq via NCBI SRA — TB-scale, requires variant calling
- **Recommended tool**: `sra-toolkit` + `fasterq-dump`, or NCBI's `prefetch`

Manual download recipe:

```bash
# Install sra-toolkit first (Synology Package Center → search "SRA" or install via conda)
ROOT=/volume2/U7_GWAS
mkdir -p $ROOT/data/raw/strawberry/{sra,phenotype}

# Step 1: pull metadata
esearch -db sra -query PRJNA578384 | efetch -format runinfo \
  > $ROOT/data/raw/strawberry/PRJNA578384_runinfo.csv

# Step 2: pull all SRA runs (TB scale!)
cut -d, -f1 $ROOT/data/raw/strawberry/PRJNA578384_runinfo.csv | tail -n +2 \
  > $ROOT/data/raw/strawberry/sra_accessions.txt

while read acc; do
  prefetch -O $ROOT/data/raw/strawberry/sra "$acc"
  fasterq-dump -O $ROOT/data/raw/strawberry/sra/"$acc" \
               --split-files --threads 8 "$acc"
done < $ROOT/data/raw/strawberry/sra_accessions.txt
```

This will take **days** and consume **~3 TB**. Only do this if you really need strawberry for v2.

### Option B2 (lighter) — Prohaska 2024 European-centered diversity panel (SNP array)

223 accessions, 50K FanaSNP array → ~50 MB SNP genotypes. Trade-off: SNP array (no SVs, no rare variants), but immediately usable.

- Paper: <https://academic.oup.com/hr/article/11/7/uhae137/7672963>
- bioRxiv: <https://www.biorxiv.org/content/10.1101/2024.03.04.583203v1>
- **Data**: check the paper's Data Availability section for the figshare DOI / supplementary file. Download manually into `data/raw/strawberry/snp_array/`.

### Option B3 — bioRxiv Hardigan/Feldmann 2024 (565 accessions, in preparation)

- bioRxiv 2024.03.11.584394 — *"Genome-wide association studies in a diverse strawberry collection unveil loci controlling agronomic and fruit quality traits"*
- 565 accessions, 121 traits, 95 QTL
- As of 2026-05, deposited data location not consistently indexed. Check the bioRxiv supplement and Hardigan UC Davis lab page:
  - <https://strawberry.ucdavis.edu/publications>

---

## C. Phenotype data

- **For Option B1 (Hardigan 145)**: phenotypes are partially embedded in the Hardigan 2020 *Plant Cell* paper supplements and on UC Davis breeding program pages. Pull manually.
- **For Option B2 (Prohaska 223)**: phenotype table is supplementary to the *Horticulture Research* 2024 paper — download via the journal's supplement link.

---

## D. Recommended path for our v2 workflow

When v2 strawberry work starts (~month 9 of project plan):

1. **Reference**: try Option A1 (FaRR1) first; if GDR registration is friction, fall back to Option A2 (FaFM1) which is one `aria2c` away.
2. **Panel**: Option B2 (Prohaska 223 SNP array) for first pass — gets the pipeline working in days, not weeks. Then escalate to Option B1 if needed for SV detection.
3. **Skip raw WGS recall unless we genuinely need SV-level resolution** — at TB scale + days of compute, it's the wrong investment for what should be a v2 *stress test*, not the v2 main result.

---

## E. Why no auto-script for Phase 4?

| Resource | Auto-downloadable? | Reason |
|---|---|---|
| FaRR1 ref (GDR) | ❌ | HTTP 403 to non-browser UA; Ft. Lauderdale click-through |
| FaFM1 ref (NCBI) | ✅ | Could be scripted — see Option A2 |
| Hardigan PRJNA578384 fastq | ⚠ | Requires sra-toolkit; TB-scale; out of W2 scope |
| Prohaska SNP array | ⚠ | Journal supplement, not a stable URL |

The auto-scriptable subset (FaFM1) does not justify a separate `phase4_strawberry.sh` — once you commit to v2, copy the Option A2 snippet inline.

---

## F. Checklist when v2 starts

- [ ] Decide reference: FaRR1 (manual) or FaFM1 (auto)
- [ ] Decide panel: Prohaska 223 (fast) or Hardigan 145 (deep but slow)
- [ ] Allocate disk: 50 GB for B2 path, ~3 TB for B1 path
- [ ] Get bcftools + sra-toolkit installed (`mamba install -c bioconda bcftools sra-tools`)
- [ ] Re-read `docs/04_method.md` §3 for octoploid dosage handling
