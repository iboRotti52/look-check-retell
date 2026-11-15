#!/usr/bin/env bash
# Check v3 signal extraction on Colab (docs/CHECK_V3_PROTOKOL.md §1). A T4 is enough (BLIP-base inference).
#   bash colab/run_check3.sh
# Resumable: scripts 31/32 skip rows already in their output. Writes /content/lcr_check3.zip.
# cons uses --bs 16: --bs 64 ran out of memory on a T4 next to the VLM job (2026-11-14; retry was run_check3_cons.sh).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results/logs
P=data/cache/pipeline
[ -d data/flickr8k/data ] || python scripts/00_download.py > results/logs/c3_download.log 2>&1
[ -e data/cache/blip/validation.f16.npy ] && [ -e data/cache/blip/train.f16.npy ] || \
  python scripts/02_cache_features.py --splits validation train --what blip > results/logs/c3_blip.log 2>&1
echo "features ready $(date +%H:%M:%S)"
( for s in A_validation_look4 A_B_look4; do python scripts/31_check_vlm_signals.py --stem $s; done ) > results/logs/c3_vlm.log 2>&1 &
VLM=$!
for s in A_validation_look4 A_B_look4; do python scripts/32_check_consistency.py --stem $s --bs 16 --threads 2; done > results/logs/c3_cons.log 2>&1
wait $VLM
grep -h "^done" results/logs/c3_vlm.log results/logs/c3_cons.log
rm -f /content/lcr_check3.zip
zip -qj /content/lcr_check3.zip $P/A_*_look4_vlm.json $P/A_*_look4_cons.json results/logs/c3_*.log
ls -lh /content/lcr_check3.zip; echo "CHECK3 DONE $(date +%H:%M:%S)"
