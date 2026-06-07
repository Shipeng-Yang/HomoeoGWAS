#!/bin/bash
# scripts/phase5d/reheader_oat_vcf.sh — OPTIONAL Phase 5d robustness helper
#
# WHY
# ===
# Raw OLD panel VCF (nsgc.gdv.gln.cohort.clean.maf01maxmiss75diploid.beagle_sorted_filt.gt.vcf)
# has NO ##contig header lines (bcftools complains
# "Contig '1A_0' is not defined in the header"). bcftools view -r and plink2
# --vcf both tolerate this — Step 3 split ran fine in 43 s without reheader.
#
# However, strict downstream tools fail on missing ##contig defs:
#   - pysam.VariantFile (when iterating with .fetch(chrom, ...))
#   - GATK ValidateVariants
#   - Snakefile CI sanity stages that call `bcftools view -h | grep '##contig'`
#   - some VCF→GDS converters (e.g. SeqArray)
#
# This script generates a bgzipped + csi-indexed copy of the raw VCF with the
# 42 segment + UN ##contig header lines injected, derived from observed
# max(POS) per chrom in the panel data itself (NCBI/Ensembl assembly headers
# use different naming; bcftools reheader --fai needs a name+length file that
# matches the VCF CHROM column, so we build it from the VCF itself).
#
# OUTPUT
# ======
# /mnt/nvme/oat_raw/figshare/oat_old.reheadered.vcf.gz   (size ~50 MB)
# /mnt/nvme/oat_raw/figshare/oat_old.reheadered.vcf.gz.csi
#
# NOT WIRED INTO PIPELINE by default — Step 3 split uses the original .vcf.gz
# directly. Run this only when you need the strict-header copy for downstream
# strict validators.
#
# USAGE
# =====
#   bash scripts/phase5d/reheader_oat_vcf.sh
#   bash scripts/phase5d/reheader_oat_vcf.sh --force         # overwrite existing
#   bash scripts/phase5d/reheader_oat_vcf.sh --out <path>    # custom output

set -uo pipefail
export PATH=/home/yys05/.local/share/mamba/envs/polygwas-cpu/bin:$PATH

# --- defaults ---
SRC_VCF=/mnt/nvme/oat_raw/figshare/nsgc.gdv.gln.cohort.clean.maf01maxmiss75diploid.beagle_sorted_filt.gt.vcf.gz
OUT_VCF=/mnt/nvme/oat_raw/figshare/oat_old.reheadered.vcf.gz
FORCE=0
WORKDIR=/tmp/reheader_oat_$$

# --- arg parse ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --src)   SRC_VCF=$2; shift 2 ;;
        --out)   OUT_VCF=$2; shift 2 ;;
        --force) FORCE=1;    shift   ;;
        -h|--help)
            grep '^# ' "$0" | sed 's/^# //' | head -50
            exit 0 ;;
        *) echo "ERROR: unknown arg $1" >&2; exit 2 ;;
    esac
done

# --- preflight ---
if [[ ! -s "$SRC_VCF" ]]; then
    echo "ERROR: source VCF missing or empty: $SRC_VCF" >&2
    echo "       (Step 3 must have produced the bgzipped .vcf.gz first.)" >&2
    exit 2
fi
if [[ -s "$OUT_VCF" && $FORCE -eq 0 ]]; then
    echo "[skip] output exists: $OUT_VCF  (pass --force to overwrite)"
    exit 0
fi
mkdir -p "$WORKDIR"
trap 'rm -rf "$WORKDIR"' EXIT

echo "=== Phase 5d optional reheader $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "[src] $SRC_VCF"
echo "[out] $OUT_VCF"

# --- 1) derive max(POS) per chrom from observed data ---
echo "[step 1/4] scanning chrom + max(POS) from VCF..."
FAI=$WORKDIR/contigs.fai
# bcftools query is fast on the bgzipped + indexed VCF; awk tracks per-chrom max
bcftools query -f '%CHROM\t%POS\n' --threads 8 "$SRC_VCF" \
  | awk -v OFS='\t' '{ if ($2 > m[$1]) m[$1] = $2 }
                     END { for (c in m) print c, m[c] }' \
  | LC_ALL=C sort -V \
  > "$FAI"
n_chroms=$(wc -l < "$FAI")
echo "[step 1/4] DONE — $n_chroms contigs (expect 43 = 42 segments + UN):"
head -5 "$FAI"; echo "..."; tail -3 "$FAI"

# --- 2) reheader using --fai ---
echo "[step 2/4] bcftools reheader --fai..."
TMP_VCF=$WORKDIR/reheaded.vcf.gz
bcftools reheader --fai "$FAI" --threads 8 "$SRC_VCF" -o "$TMP_VCF"
echo "[step 2/4] DONE size=$(stat -c%s "$TMP_VCF") B"

# --- 3) move into place + csi index ---
echo "[step 3/4] moving + csi-indexing output..."
mkdir -p "$(dirname "$OUT_VCF")"
mv "$TMP_VCF" "$OUT_VCF"
bcftools index --csi --force --threads 8 "$OUT_VCF"
echo "[step 3/4] DONE: $OUT_VCF + .csi"

# --- 4) verify ##contig lines now present ---
echo "[step 4/4] verifying ##contig defs..."
n_contig_hdr=$(bcftools view -h "$OUT_VCF" 2>/dev/null | grep -c '^##contig')
echo "  ##contig lines in header: $n_contig_hdr (expect 43)"
echo "  first 3 ##contig defs:"
bcftools view -h "$OUT_VCF" 2>/dev/null | grep '^##contig' | head -3
if [[ $n_contig_hdr -ne $n_chroms ]]; then
    echo "WARNING: ##contig count $n_contig_hdr != observed chrom count $n_chroms"
    exit 3
fi

echo "=== reheader FINISHED $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "Output ready: $OUT_VCF"
echo
echo "To use this VCF in homoeogwas split, point species YAML geno.vcf at it"
echo "(but the original .vcf.gz already works for the Step 3 split — this"
echo "reheader copy is only required for strict-header downstream tools)."
