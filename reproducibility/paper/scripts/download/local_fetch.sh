#!/usr/bin/env bash
# =============================================================================
# U7_GWAS — LOCAL fetch (this GPU box, NOT the Synology NAS)
#   ROOT = /mnt/7302share/fast_ysp/U7_GWAS   (NFS 192.168.50.192, 8.5T free)
#   aria2c from conda env `u7dl`. No screen — run backgrounded by the agent.
#   This box reaches NCBI + ENA + NGDC + hf-mirror (verified 2026-05-17),
#   so references come straight from NCBI (no China-mirror juggling needed).
# Usage:  bash local_fetch.sh {wheat|rapeseed|cotton|models}
# Resumable: aria2c -c; safe to re-run.
# =============================================================================
set -euo pipefail
export LC_ALL=C

ROOT=/mnt/7302share/fast_ysp/U7_GWAS
ARIA=/home/yysu7/miniconda3/envs/u7dl/bin/aria2c
OPTS=(-Z -x 16 -s 16 -c --auto-file-renaming=false --allow-overwrite=false
      --console-log-level=warn --summary-interval=60 --retry-wait=10
      --max-tries=5 --file-allocation=none --conditional-get=true)
# -Z : treat each URL as a separate file (not mirrors of one).

TS=$(date +%Y%m%d_%H%M%S)
mkdir -p "$ROOT"/logs

# ---------------------------------------------------------------------------
do_wheat() {
  local L="$ROOT/logs/local_wheat_$TS.log"; exec > >(tee -a "$L") 2>&1
  echo "== WHEAT $(date) =="

  # The Watkins VCFs are called against IWGSC RefSeq **v1.0** (chr-prefixed
  # names: chr1A..chr7D, chrUn) — confirmed from the WatSeq datapackage.json.
  # NOT v2.1. Take v1.0 pseudomolecules from URGI (chr names match the VCF
  # exactly; NCBI GCA_900519105.1 uses GenBank accessions and would NOT match).
  echo "-- [1/3] IWGSC RefSeq v1.0 assembly + v1.1 gene annotation (URGI) --"
  "$ARIA" "${OPTS[@]}" -j 2 -d "$ROOT/data/reference/wheat" \
    "https://urgi.versailles.inrae.fr/download/iwgsc/IWGSC_RefSeq_Assemblies/v1.0/iwgsc_refseqv1.0_all_chromosomes.zip" \
    "https://urgi.versailles.inrae.fr/download/iwgsc/IWGSC_RefSeq_Annotations/v1.1/iwgsc_refseqv1.1_genes_2017July06.zip" \
    "https://urgi.versailles.inrae.fr/download/iwgsc/IWGSC_RefSeq_Annotations/v1.1/iwgsc_refseqv1.1_genes_2017July06.zip.md5.txt"

  echo "-- [2/3] Watkins 21-chromosome VCFs (ENA, ~430 GB) --"
  # NOTE: huge international files; in practice fetched via 115 cloud and
  # dropped into data/raw/wheat/vcf/ (see scripts/download/wheat_urls.txt,
  # Earlham mirror). This step is the fallback if downloading on-box.
  local ENA="https://ftp.sra.ebi.ac.uk/vol1/analysis/ERZ221/ERZ22157275"
  local SFX="SNP.Missing-unphasing.ID.ann.finalSID.allele2_retain.hard_retain.InbreedingCoeff_retain.missing_retain.maf_retain.vcf.gz"
  local U=()
  for c in 1A 1B 1D 2A 2B 2D 3A 3B 3D 4A 4B 4D 5A 5B 5D 6A 6B 6D 7A 7B 7D; do
    U+=("$ENA/chr${c}.${SFX}")
  done
  "$ARIA" "${OPTS[@]}" -j 1 -d "$ROOT/data/raw/wheat/vcf" "${U[@]}"

  echo "-- [3/3] Watkins phenotype (Earlham webdav, python crawler) --"
  python3 - "$ROOT/data/raw/wheat/phenotype" <<'PY'
import re, os, sys, urllib.request, urllib.parse
base = "https://opendata.earlham.ac.uk/wheat/under_license/toronto/WatSeq_2023-09-15_landrace_modern_Variation_Data/Watseq_phenotype_data/"
target = sys.argv[1]; os.makedirs(target, exist_ok=True)
EXT = re.compile(r'\.(csv|tsv|xlsx|txt|pdf|md|json|readme)$', re.I)
def crawl(url, d=0):
    if d > 6: return
    try:
        rq = urllib.request.Request(url, headers={'User-Agent':'Mozilla/5.0'})
        html = urllib.request.urlopen(rq, timeout=60).read().decode('utf-8','replace')
    except Exception as e:
        print("  skip", url, e); return
    for h in re.findall(r'href="([^"#?]+)"', html):
        if h in ('.','..','/') or h.startswith('javascript:'): continue
        a = urllib.parse.urljoin(url, h)
        if not a.startswith(base): continue
        if a.endswith('/'): crawl(a, d+1)
        elif EXT.search(a):
            rel = a[len(base):]; op = os.path.join(target, rel)
            os.makedirs(os.path.dirname(op) or target, exist_ok=True)
            if os.path.exists(op) and os.path.getsize(op) > 0: continue
            print("  down", rel)
            try: urllib.request.urlretrieve(a, op)
            except Exception as e: print("   fail", e)
crawl(base); print("  phenotype done")
PY
  echo "== WHEAT complete $(date) =="
  du -sh "$ROOT"/data/reference/wheat "$ROOT"/data/raw/wheat/* 2>/dev/null
}

# ---------------------------------------------------------------------------
do_rapeseed() {
  local L="$ROOT/logs/local_rapeseed_$TS.log"; exec > >(tee -a "$L") 2>&1
  echo "== RAPESEED $(date) =="

  # NOTE: BnIR ZS11 2,311 panel (yanglab.hzau.edu.cn) is offline everywhere.
  # Only obtainable genotyped+self-consistent rapeseed data is the NGDC
  # 403-accession SNP set, which is Darmor-bzh v2.0 anchored (NC_0277xx.2).
  # So we pair it with the matching NCBI Darmor-bzh v2.0 reference.
  echo "-- [1/3] Darmor-bzh v2.0 reference (NCBI GCF_000686985.2; matches NC_0277xx.2) --"
  "$ARIA" "${OPTS[@]}" -j 1 -d "$ROOT/data/reference/rapeseed" \
    "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/686/985/GCF_000686985.2_Bra_napus_v2.0/GCF_000686985.2_Bra_napus_v2.0_genomic.fna.gz" \
    "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/686/985/GCF_000686985.2_Bra_napus_v2.0/GCF_000686985.2_Bra_napus_v2.0_genomic.gff.gz" \
    "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/686/985/GCF_000686985.2_Bra_napus_v2.0/md5checksums.txt"

  echo "-- [2/3] NGDC 403-accession genotyped SNP VCF (~26 GB, Darmor v2.0) --"
  "$ARIA" "${OPTS[@]}" -j 1 -d "$ROOT/data/raw/rapeseed/vcf" \
    "https://download.cncb.ac.cn/GVM/Brassica_napus/SNP/detailed_vcf/all_snp.vcf.gz" \
    "https://download.cncb.ac.cn/GVM/Brassica_napus/SNP/detailed_vcf/all_snp.vcf.gz.tbi"

  echo "-- [3/3] Wu 2019 phenotype (Springer Supp 5; sample-match TBD) --"
  "$ARIA" "${OPTS[@]}" -j 1 -d "$ROOT/data/raw/rapeseed/phenotype" \
    "https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-019-09134-9/MediaObjects/41467_2019_9134_MOESM5_ESM.xlsx"

  echo "== RAPESEED complete $(date) =="
  du -sh "$ROOT"/data/raw/rapeseed/* "$ROOT"/data/reference/rapeseed 2>/dev/null
}

# ---------------------------------------------------------------------------
do_cotton() {
  local L="$ROOT/logs/local_cotton_$TS.log"; exec > >(tee -a "$L") 2>&1
  echo "== COTTON $(date) =="

  echo "-- [1/3] TM-1 NAU v1.1 reference (NCBI RefSeq, works from this box) --"
  "$ARIA" "${OPTS[@]}" -j 1 -d "$ROOT/data/reference/cotton" \
    "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/987/745/GCF_000987745.1_ASM98774v1/GCF_000987745.1_ASM98774v1_genomic.fna.gz" \
    "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/987/745/GCF_000987745.1_ASM98774v1/GCF_000987745.1_ASM98774v1_genomic.gff.gz" \
    "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/987/745/GCF_000987745.1_ASM98774v1/md5checksums.txt"

  echo "-- [2/3] 1,081-accession SNP zip (Hebei Ag U, http) --"
  "$ARIA" "${OPTS[@]}" -j 1 -d "$ROOT/data/raw/cotton/snp" \
    "http://cotton.hebau.edu.cn/wenjian/genotype-data.zip"

  echo "-- [3/3] phenotype + gene-expression zips --"
  "$ARIA" "${OPTS[@]}" -j 2 -d "$ROOT/data/raw/cotton/phenotype" \
    "http://cotton.hebau.edu.cn/wenjian/phenotypic-data.zip" \
    "http://cotton.hebau.edu.cn/wenjian/Gene-expression-data.zip"

  echo "== COTTON complete $(date) =="
  du -sh "$ROOT"/data/reference/cotton "$ROOT"/data/raw/cotton/* 2>/dev/null
}

# ---------------------------------------------------------------------------
do_models() {
  local L="$ROOT/logs/local_models_$TS.log"; exec > >(tee -a "$L") 2>&1
  echo "== MODELS + TOOLS $(date) =="
  : "${HF_ENDPOINT:=https://hf-mirror.com}"; export HF_ENDPOINT
  python3 -c "import huggingface_hub" 2>/dev/null || pip install --user -q -U huggingface_hub
  for s in \
    "kuleshov-group/PlantCaduceus_l32|$ROOT/data/reference/models/plantcaduceus" \
    "InstaDeepAI/agro-nucleotide-transformer-1b|$ROOT/data/reference/models/agront"; do
    repo="${s%%|*}"; dir="${s##*|}"; echo "  HF: $repo"
    python3 - <<PY
from huggingface_hub import snapshot_download
snapshot_download(repo_id="$repo", local_dir="$dir", max_workers=8)
PY
  done
  if command -v git >/dev/null 2>&1; then
    for r in zhangyixing3/NodeGWAS jendelman/GWASpoly BorgwardtLab/networkGWAS \
             rgcgithub/regenie saigegit/SAIGE eblerjana/pangenie \
             xihaoli/STAAR PMBio/deeprvat; do
      n="${r##*/}"; t="$ROOT/benchmarks/$n"
      [ -d "$t/.git" ] && { echo "  skip $n"; continue; }
      mkdir -p "$ROOT/benchmarks"; echo "  git: $n"
      git clone --depth 1 "https://github.com/$r.git" "$t" || echo "   clone failed $n"
    done
  else
    echo "  git missing — skip tool repos"
  fi
  echo "== MODELS complete $(date) =="
}

case "${1:-}" in
  wheat)    do_wheat ;;
  rapeseed) do_rapeseed ;;
  cotton)   do_cotton ;;
  models)   do_models ;;
  *) echo "usage: bash local_fetch.sh {wheat|rapeseed|cotton|models}"; exit 1 ;;
esac
