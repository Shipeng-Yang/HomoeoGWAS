#!/usr/bin/env bash
# Auxiliary downloader: DL-prior models (direct HuggingFace, no mirror —
# box confirmed to reach foreign sites), 8 baseline GWAS tool repos, and
# the verified-real WISP/DFW Watkins field-phenotype xlsx files.
# Launch: nohup bash _aux_grab.sh >> logs/aux_grab.out 2>&1 &
export LC_ALL=C
R=/mnt/7302share/fast_ysp/U7_GWAS
A=/home/yysu7/miniconda3/envs/u7dl/bin/aria2c
unset HF_ENDPOINT   # direct huggingface.co
echo "==== _aux_grab START $(date) pid $$ ===="

# 1) Wheat WISP/DFW phenotype xlsx (small, links verified live 2026-05-18)
WP="$R/data/raw/wheat/phenotype"; mkdir -p "$WP"
WB="https://wisplandracepillar.jic.ac.uk"
PH="$WB/results/WISP_WGIN_Watkins_JIC_2006_2010_2011_Field_Data.xlsx
$WB/resources/JIC_Wheat_Pre-Breeding_WISP_Watkins_Whole_Collection.xlsx
$WB/results/JIC_Wheat_Pre-Breeding_WISP_Watkins_Core_Collection.xlsx"
for y in 2011 2012 2013 2014 2015 2016 2017; do
  PH="$PH
$WB/results/DFW_phenotype_data_CF_${y}.xlsx"
done
echo "---- [1/3] wheat WISP phenotype xlsx ----"
"$A" -Z -x4 -s4 -j4 -c --auto-file-renaming=false --allow-overwrite=false \
     --console-log-level=warn --max-tries=5 --retry-wait=10 \
     -d "$WP" $PH
echo "---- phenotype step exit=$? ----"

# 2) 8 baseline GWAS tool repos (shallow clones)
echo "---- [2/3] baseline tool repos ----"
mkdir -p "$R/benchmarks"
for r in zhangyixing3/NodeGWAS jendelman/GWASpoly BorgwardtLab/networkGWAS \
         rgcgithub/regenie saigegit/SAIGE eblerjana/pangenie \
         xihaoli/STAAR PMBio/deeprvat; do
  n="${r##*/}"; t="$R/benchmarks/$n"
  if [ -d "$t/.git" ]; then echo "  skip $n (exists)"; continue; fi
  echo "  git clone $n"
  git clone --depth 1 "https://github.com/$r.git" "$t" || echo "  !! clone failed $n"
done

# 3) DL-prior models via direct HuggingFace
echo "---- [3/3] HF models (direct) ----"
python3 -c "import huggingface_hub" 2>/dev/null || pip install --user -q -U huggingface_hub
for s in \
  "kuleshov-group/PlantCaduceus_l32|$R/data/reference/models/plantcaduceus" \
  "InstaDeepAI/agro-nucleotide-transformer-1b|$R/data/reference/models/agront"; do
  repo="${s%%|*}"; dir="${s##*|}"; mkdir -p "$dir"; echo "  HF: $repo"
  python3 - "$repo" "$dir" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download(repo_id=sys.argv[1], local_dir=sys.argv[2],
                  max_workers=8, resume_download=True)
PY
done
echo "==== _aux_grab DONE $(date) ===="
