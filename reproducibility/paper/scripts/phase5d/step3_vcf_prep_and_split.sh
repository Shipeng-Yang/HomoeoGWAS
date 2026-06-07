#!/bin/bash
# Phase 5d Step 3 — VCF prep + homoeogwas split
# Sequence:
#   1) bgzip raw VCF (was plain .vcf)
#   2) bcftools index --csi (oat chrom 1-7 D each ~500-700 Mb, > 512 Mb tbi limit)
#   3) homoeogwas split --species-yaml configs/species/avena_sativa.yaml
#       (calls bcftools view -r per subgenome + plink2 --vcf --make-pgen)
# Expected runtime: 30-60 min wall-clock (1.18 GB VCF, IO-bound).

set -uo pipefail
export PATH=/home/yys05/.local/share/mamba/envs/polygwas-cpu/bin:$PATH

PROJ=/mnt/7302share/fast_ysp/U7_GWAS
RAW_VCF=/mnt/nvme/oat_raw/figshare/nsgc.gdv.gln.cohort.clean.maf01maxmiss75diploid.beagle_sorted_filt.gt.vcf
BGZ_VCF=/mnt/nvme/oat_raw/figshare/nsgc.gdv.gln.cohort.clean.maf01maxmiss75diploid.beagle_sorted_filt.gt.vcf.gz
LOG=$PROJ/scripts/phase5d/step3_vcf_prep_and_split.log
exec > >(tee -a "$LOG") 2>&1
echo "=== Phase 5d Step 3 started $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

# --- 1) bgzip ---
if [[ -s "$BGZ_VCF" ]]; then
    echo "[bgzip] EXISTS skip: $BGZ_VCF ($(stat -c%s "$BGZ_VCF") B)"
else
    echo "[bgzip] compressing raw VCF..."
    bgzip -c --threads 8 "$RAW_VCF" > "$BGZ_VCF"
    echo "[bgzip] DONE: $(stat -c%s "$BGZ_VCF") B"
fi

# --- 2) csi index ---
if [[ -s "${BGZ_VCF}.csi" ]]; then
    echo "[index] EXISTS skip: ${BGZ_VCF}.csi"
else
    echo "[index] building csi (oat chrom > 512 Mb forces csi)..."
    bcftools index --csi --threads 8 "$BGZ_VCF"
    echo "[index] DONE: ${BGZ_VCF}.csi"
fi

# --- 2.5) Update species YAML to point at .vcf.gz (was .vcf) ---
SPECIES_YAML=$PROJ/configs/species/avena_sativa.yaml
if grep -q "beagle_sorted_filt.gt.vcf$" "$SPECIES_YAML"; then
    sed -i 's|beagle_sorted_filt.gt.vcf$|beagle_sorted_filt.gt.vcf.gz|' "$SPECIES_YAML"
    echo "[yaml] updated species YAML geno.vcf path → .vcf.gz"
fi

# --- 3) Verify species YAML validate now ---
echo "[validate] running species YAML validator post-update..."
python <<PY
import sys; sys.path.insert(0, "$PROJ/src")
from homoeogwas.species_config import load_species_config
from pathlib import Path
cfg = load_species_config("$SPECIES_YAML", project_root=Path("$PROJ"))
rep = cfg.__dict__["_validation_report"]
print(f"   n_chroms_in_chrom_map={rep.n_chroms_in_chrom_map}, n_known_qtl={rep.n_known_qtl}, n_samples={rep.n_samples_in_pheno}")
for w in rep.warnings: print(f"   warn: {w}")
PY

# --- 4) homoeogwas split ---
echo "[split] running homoeogwas split (ACD × 14 segments each, ~30-60 min)..."
homoeogwas split \
    --species-yaml "$SPECIES_YAML" \
    --threads 8 \
    --keep-vcf

echo "=== Phase 5d Step 3 FINISHED $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

# --- 5) Post-split sanity ---
echo "[post] per-sub artifacts:"
for sub in A C D; do
    pgen=$PROJ/data/processed/avena_sativa/$sub/all.pgen
    bim=$PROJ/data/processed/avena_sativa/$sub/all.bim
    vcf=$PROJ/data/processed/avena_sativa/$sub/all.subset.vcf.gz
    if [[ -s "$pgen" ]]; then
        n_snp=$(wc -l < "${PROJ}/data/processed/avena_sativa/$sub/all.pvar" 2>/dev/null || echo "??")
        echo "  sub $sub: pgen=$(stat -c%s "$pgen") B, n_snp(pvar)=$n_snp"
    else
        echo "  sub $sub: MISSING pgen"
    fi
done

df -h /mnt/7302share /mnt/nvme | tail -2
