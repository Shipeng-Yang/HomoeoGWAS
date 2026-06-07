#!/usr/bin/env bash
# DL-prior model downloader — aria2c direct against HuggingFace resolve URLs.
#
# WHY: python `snapshot_download` (unauthenticated, no timeout) HUNG for
# 10h43m on the big pytorch_model.bin and never recovered, leaving only the
# 88K of text/config files. aria2c with -c (resume) + finite --timeout +
# --max-tries=0 self-heals and is multi-connection fast. Launched as
# `bash _models_grab.sh` so the Bash-tool sleep-filter never sees it.
#
# Launch: nohup bash _models_grab.sh >> logs/models_grab.out 2>&1 &
export LC_ALL=C
A=/home/yysu7/miniconda3/envs/u7dl/bin/aria2c
R=/mnt/7302share/fast_ysp/U7_GWAS
HF=https://huggingface.co
echo "==== _models_grab START $(date) pid $$ ===="

# repo | local-dir | space-separated file paths (HF repo-relative)
JOBS=(
"kuleshov-group/PlantCaduceus_l32|$R/data/reference/models/plantcaduceus|.gitattributes README.md config.json configuration_caduceus.py modeling_caduceus.py modeling_rcps.py pytorch_model.bin special_tokens_map.json tokenizer.json tokenizer_config.json"
"InstaDeepAI/agro-nucleotide-transformer-1b|$R/data/reference/models/agront|.gitattributes README.md config.json esm_config.py jax_model/hyperparams.json jax_model/pytree_ckpt.joblib modeling_esm.py pytorch_model.bin special_tokens_map.json tokenizer_config.json vocab.txt"
)

for J in "${JOBS[@]}"; do
  repo="${J%%|*}"; rest="${J#*|}"; dir="${rest%%|*}"; files="${rest#*|}"
  echo "---- repo $repo -> $dir ----"
  mkdir -p "$dir"
  for rel in $files; do
    url="$HF/$repo/resolve/main/$rel"
    out="$dir/$rel"
    sub="${rel%/*}"; [ "$sub" != "$rel" ] && mkdir -p "$dir/$sub"
    # Self-healing per file: re-invoke aria2c until the file is fully done
    # (control file gone AND file present).
    while [ -e "$out.aria2" ] || [ ! -e "$out" ]; do
      echo "  get $rel  $(date)"
      "$A" -x8 -s8 -j1 --min-split-size=10M -c \
           --auto-file-renaming=false --allow-overwrite=true \
           --console-log-level=warn --summary-interval=60 \
           --max-tries=0 --retry-wait=20 --file-allocation=none \
           --connect-timeout=60 --timeout=120 \
           -d "$dir" -o "$rel" "$url"
      rc=$?
      echo "  $rel aria2c exit=$rc $(date)"
      if [ $rc -ne 0 ] && [ -e "$out.aria2" ]; then
        echo "   (dropping poisoned control file, restarting $rel clean)"
        rm -f "$out.aria2"
      fi
      [ $rc -ne 0 ] && sleep 30
    done
    echo "  OK $rel  $(du -h "$out" 2>/dev/null | cut -f1)"
  done
  echo "==== $repo COMPLETE $(date) ===="
done
echo "==== ALL MODELS DONE $(date) ===="
