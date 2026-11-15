#!/usr/bin/env bash
# Runs every GPU stage of the progress-report experiments on Colab, in order.
# Each stage logs to results/logs/<stage>.log and is skipped if its output already exists,
# so the script can simply be re-run after a disconnect.
#   bash colab/run_all.sh            # all stages
#   bash colab/run_all.sh retell     # only from the Retell stage on
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results/logs
FROM="${1:-all}"
started=0; [ "$FROM" = "all" ] && started=1

stage() {  # stage <name> <done-file> <command...>
  local name="$1" done="$2"; shift 2
  [ "$name" = "$FROM" ] && started=1
  [ $started -eq 1 ] || return 0
  if [ -n "$done" ] && [ -e "$done" ]; then echo "[skip] $name ($done exists)"; return 0; fi
  echo "[run ] $name  $(date +%H:%M:%S)"
  ( "$@" ) > "results/logs/$name.log" 2>&1 || { echo "[FAIL] $name -- tail of log:"; tail -30 "results/logs/$name.log"; exit 1; }
  tail -3 "results/logs/$name.log"
}

P=data/cache/pipeline
stage download  data/flickr8k/data             python scripts/00_download.py
stage stats     ""                              python scripts/01_data_stats.py
stage blip      data/cache/blip/train.f16.npy  python scripts/02_cache_features.py --splits validation train --what blip
stage owl_val   data/cache/owl/validation.npz  python scripts/02_cache_features.py --splits validation --what owl --bs 16
stage owl_B     data/cache/owl/train.npz       python scripts/02_cache_features.py --splits train --what owl --only B --bs 16
stage finetune  data/cache/ckpt/decoder_A.pt   python scripts/03_finetune_decoder.py --half A --epochs 3
stage drafts_zero $P/zero_validation.json      python scripts/04_draft_look_check.py --split validation --decoder zero --attn raw --region-rule v1
stage drafts_val  $P/A_validation.json         python scripts/04_draft_look_check.py --split validation --decoder A --attn raw --region-rule v1
stage drafts_B    $P/A_B.json                  python scripts/04_draft_look_check.py --split B --decoder A --attn raw --region-rule v1
stage retell    results/retell_b0plus.json     python scripts/06_train_retell.py --mode b0plus --attn raw
stage ours      results/retell_ours.json       python scripts/06_train_retell.py --mode ours --rho 0.15 --attn raw
stage b3        results/retell_b3.json         python scripts/06_train_retell.py --mode b3 --rho 0.15 --attn raw
stage b2        results/retell_b2.json         python scripts/06_train_retell.py --mode b2 --attn raw
stage ours_rho0 results/retell_ours_rho0.json  python scripts/06_train_retell.py --mode ours --rho 0 --tag rho0 --attn raw
stage metrics   results/full_metrics.json      python scripts/colab_full_metrics.py

# Small artefacts to bring back (no images, no checkpoints, no feature caches).
zip -qr lcr_results.zip results report/figures report/generated $P/*.json $P/*.maps.npz
ls -lh lcr_results.zip
echo "ALL DONE $(date +%H:%M:%S)"
