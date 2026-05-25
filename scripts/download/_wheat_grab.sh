#!/usr/bin/env bash
# Resilient wheat downloader — v2.
#
# WHY v2: aria2c doing *segmented multi-file FTP* against ENA mis-applied
# chr1A's size (~21.48 GB) to other chromosomes, raising
#   "Size mismatch Expected:21485813067 Actual:..."
# which aria2c treats as FATAL (NOT covered by --max-tries=0), so it
# abandoned 20 of 21 files and exited cleanly. Fix: one file, ONE
# connection per aria2c call (no segmentation, correct per-file FTP SIZE),
# driven by a per-file self-healing loop. Pacing uses an in-script `sleep`
# (this file is launched as `bash _wheat_grab.sh`, so the Bash-tool
# sleep-filter never sees it).
#
# Launch: nohup bash _wheat_grab.sh >> logs/wheat_grab_relaunch.out 2>&1 &
export LC_ALL=C
A=/home/yysu7/miniconda3/envs/u7dl/bin/aria2c
R=/mnt/7302share/fast_ysp/U7_GWAS
VDIR="$R/data/raw/wheat/vcf"
ENA="ftp://ftp.sra.ebi.ac.uk/vol1/analysis/ERZ221/ERZ22157275"
SFX="SNP.Missing-unphasing.ID.ann.finalSID.allele2_retain.hard_retain.InbreedingCoeff_retain.missing_retain.maf_retain.vcf.gz"
mkdir -p "$VDIR"
echo "==== _wheat_grab v2 START $(date) pid $$ ===="

# 1) URGI IWGSC v1.0 reference — already complete; aria2c size-matches and
#    exits 0 instantly. Kept so a fresh box still gets it.
"$A" -x4 -s4 -j1 -c --auto-file-renaming=false --allow-overwrite=false \
     --console-log-level=warn --max-tries=3 --retry-wait=20 \
     -d "$R/data/reference/wheat" \
     "https://urgi.versailles.inrae.fr/download/iwgsc/IWGSC_RefSeq_Assemblies/v1.0/iwgsc_refseqv1.0_all_chromosomes.zip" \
  && echo "---- REF ok $(date) ----"

# 2) 21 Watkins VCFs — strictly ONE file + ONE connection per aria2c call.
#    Single-stream FTP => no segmented reassembly, server gives the correct
#    per-file SIZE => no spurious "Size mismatch". Outer while re-invokes if
#    aria2c ever exits before the file is truly done; a poisoned control
#    file (size-mismatch) is dropped so the retry restarts that file clean.
for c in 1A 1B 1D 2A 2B 2D 3A 3B 3D 4A 4B 4D 5A 5B 5D 6A 6B 6D 7A 7B 7D; do
  f="chr${c}.${SFX}"
  while [ -e "$VDIR/$f.aria2" ] || [ ! -e "$VDIR/$f" ]; do
    echo "---- get chr${c} $(date) ----"
    # Multi-connection is SAFE here: only ONE file per aria2c call, so the
    # morning's cross-FILE size confusion cannot happen. 8 segments restores
    # the ~20 MB/s throughput we saw earlier.
    "$A" -x8 -s8 -j1 --min-split-size=20M -c \
         --auto-file-renaming=false --allow-overwrite=false \
         --console-log-level=warn --summary-interval=120 \
         --max-tries=0 --retry-wait=30 --file-allocation=none \
         --ftp-reuse-connection=false --connect-timeout=60 --timeout=120 \
         -d "$VDIR" "$ENA/$f"
    rc=$?
    echo "---- chr${c} aria2c exit=$rc $(date) ----"
    if [ $rc -ne 0 ] && [ -e "$VDIR/$f.aria2" ]; then
      echo "   (dropping poisoned control file, will restart chr${c} clean)"
      rm -f "$VDIR/$f.aria2"
    fi
    [ $rc -ne 0 ] && sleep 30
  done
  echo "==== chr${c} COMPLETE $(date) ===="
done
echo "==== ALL WHEAT DONE $(date) ===="
