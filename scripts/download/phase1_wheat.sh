#!/usr/bin/env bash
# =============================================================================
# U7_GWAS — Phase 1: WHEAT (hexaploid AABBDD)
# Downloads:
#   - IWGSC RefSeq v2.1 (Chinese Spring) reference  ~4.3 GB
#   - Watkins 1,051-accession SNP VCFs (ENA mirror) ~430 GB across 21 chromosomes
#   - Watkins phenotype data (Earlham webdav)       <50 MB
# Total: ~435 GB; recommend ≥500 GB free.
# Target: Synology NAS at /volume2/U7_GWAS
# Usage:   bash phase1_wheat.sh
# Resumable: aria2c -c on every file; safe to Ctrl-C and re-run.
# tmux/screen STRONGLY recommended — the panel VCFs may take many hours.
# URLs verified 2026-05-13.
# =============================================================================
set -euo pipefail
export LC_ALL=C

ROOT="/volume2/U7_GWAS"
ARIA_OPTS=(-Z -x 16 -s 16 -j 1 -c
           --auto-file-renaming=false
           --allow-overwrite=false
           --console-log-level=warn
           --summary-interval=60
           --retry-wait=10
           --max-tries=5)
# -Z = force-sequential: treat each URL as a SEPARATE file (not mirrors of one).
#      Critical fix; without it, aria2c only downloads the first URL.

# ---------- Tool checks ----------
command -v aria2c >/dev/null || { echo "ERROR: aria2c not installed"; exit 1; }
command -v wget   >/dev/null || { echo "ERROR: wget not installed"; exit 1; }

# ---------- Disk check (portable) ----------
FREE_GB=$(df -k "$ROOT" 2>/dev/null | tail -1 | awk '{printf "%d", $4/1024/1024}')
NEED_GB=500
if [ "${FREE_GB:-0}" -lt "$NEED_GB" ]; then
  echo "WARNING: $ROOT has ${FREE_GB:-unknown} GB free, recommended ≥${NEED_GB} GB."
  printf "Continue anyway? [y/N] "
  read -r confirm
  [ "$confirm" = "y" ] || { echo "Aborted."; exit 1; }
fi

# ---------- Directories ----------
mkdir -p "$ROOT"/data/reference/wheat
mkdir -p "$ROOT"/data/raw/wheat/{vcf,phenotype}
mkdir -p "$ROOT"/logs

LOG="$ROOT/logs/phase1_wheat_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "============================================================"
echo " U7_GWAS Phase 1 — WHEAT — $(date)"
echo " ROOT=$ROOT, free=${FREE_GB} GB, LOG=$LOG"
echo "============================================================"

# =============================================================================
# Step 1 — IWGSC RefSeq v2.1 (Chinese Spring) — ~4.3 GB
# =============================================================================
echo
echo "===== [1/3] Reference genome (NCBI IWGSC RefSeq v2.1, GCF_018294505.1) ====="

aria2c "${ARIA_OPTS[@]}" -d "$ROOT/data/reference/wheat" \
  "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/018/294/505/GCF_018294505.1_IWGSC_CS_RefSeq_v2.1/GCF_018294505.1_IWGSC_CS_RefSeq_v2.1_genomic.fna.gz" \
  "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/018/294/505/GCF_018294505.1_IWGSC_CS_RefSeq_v2.1/GCF_018294505.1_IWGSC_CS_RefSeq_v2.1_genomic.gff.gz" \
  "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/018/294/505/GCF_018294505.1_IWGSC_CS_RefSeq_v2.1/md5checksums.txt"

# =============================================================================
# Step 2 — Watkins panel VCFs (ENA mirror, PRJEB71453 / ERZ22157275) — ~430 GB
# =============================================================================
echo
echo "===== [2/3] Watkins 1,051-accession SNP VCFs (ENA mirror) ====="
echo "  21 chromosomes, ~20 GB each, ~430 GB total"
echo "  HOURS-DAYS depending on link speed. Ctrl-C is safe."

ENA="https://ftp.sra.ebi.ac.uk/vol1/analysis/ERZ221/ERZ22157275"
SFX="SNP.Missing-unphasing.ID.ann.finalSID.allele2_retain.hard_retain.InbreedingCoeff_retain.missing_retain.maf_retain.vcf.gz"

URLS=()
for chr in 1A 1B 1D 2A 2B 2D 3A 3B 3D 4A 4B 4D 5A 5B 5D 6A 6B 6D 7A 7B 7D; do
  URLS+=("$ENA/chr${chr}.${SFX}")
done
aria2c "${ARIA_OPTS[@]}" -d "$ROOT/data/raw/wheat/vcf" "${URLS[@]}"

# =============================================================================
# Step 3 — Watkins phenotype (Earlham webdav)
# Python stdlib crawler — Synology busybox wget lacks -r / --no-check-certificate.
# =============================================================================
echo
echo "===== [3/3] Watkins phenotype data (Earlham webdav) ====="

PHENO_DIR="$ROOT/data/raw/wheat/phenotype"
mkdir -p "$PHENO_DIR"

PY=""
for cand in python3 python; do
  if command -v "$cand" >/dev/null 2>&1; then PY="$cand"; break; fi
done

if [ -z "$PY" ]; then
  echo "  WARNING: no python found — phenotype mirror skipped."
  echo "  Download manually from:"
  echo "  https://opendata.earlham.ac.uk/wheat/under_license/toronto/WatSeq_2023-09-15_landrace_modern_Variation_Data/Watseq_phenotype_data/"
else
  "$PY" - "$PHENO_DIR" <<'PYEOF'
import re, os, sys, urllib.request, urllib.parse

base = "https://opendata.earlham.ac.uk/wheat/under_license/toronto/WatSeq_2023-09-15_landrace_modern_Variation_Data/Watseq_phenotype_data/"
target = sys.argv[1]
os.makedirs(target, exist_ok=True)

EXT_RE = re.compile(r'\.(csv|tsv|xlsx|txt|pdf|md|json|readme)$', re.IGNORECASE)

def crawl(url, depth=0):
    if depth > 6:
        return
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        html = urllib.request.urlopen(req, timeout=60).read().decode('utf-8', errors='replace')
    except Exception as e:
        print(f"  skip {url}: {e}")
        return
    for href in re.findall(r'href="([^"#?]+)"', html):
        if href in ('.', '..', '/') or href.startswith('javascript:'):
            continue
        absurl = urllib.parse.urljoin(url, href)
        # stay within the phenotype tree
        if not absurl.startswith(base):
            continue
        if absurl.endswith('/'):
            crawl(absurl, depth + 1)
        elif EXT_RE.search(absurl):
            rel = absurl[len(base):]
            outpath = os.path.join(target, rel)
            os.makedirs(os.path.dirname(outpath) or target, exist_ok=True)
            if os.path.exists(outpath) and os.path.getsize(outpath) > 0:
                continue
            print(f"  ↓ {rel}")
            try:
                urllib.request.urlretrieve(absurl, outpath)
            except Exception as e:
                print(f"    failed: {e}")

crawl(base)
print("  phenotype mirror done")
PYEOF
fi

# =============================================================================
# Summary
# =============================================================================
echo
echo "============================================================"
echo " Phase 1 (wheat) complete — $(date)"
echo "============================================================"
du -sh "$ROOT"/data/reference/wheat "$ROOT"/data/raw/wheat/* 2>/dev/null
echo
echo "Sanity check:"
echo "  bcftools view -h $ROOT/data/raw/wheat/vcf/chr1A.${SFX} | tail -5"
echo "  ls $ROOT/data/raw/wheat/phenotype/"
echo
echo "Next:"
echo "  - Run phase2_cotton.sh / phase3_rapeseed.sh in parallel if disk permits"
echo "  - scripts/preprocess/wheat_to_dosage.sh (TBW)"
