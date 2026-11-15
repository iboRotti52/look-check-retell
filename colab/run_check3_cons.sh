#!/usr/bin/env bash
# Retry of the cons stage after OOM at --bs 64 on a T4 (VLM job shares the GPU); then zip once VLM is done.
cd /content/lcr
for s in A_validation_look4 A_B_look4; do python scripts/32_check_consistency.py --stem $s --bs 16 --threads 2; done > results/logs/c3_cons2.log 2>&1
while pgrep -f 31_check_vlm >/dev/null; do sleep 20; done
P=data/cache/pipeline
rm -f /content/lcr_check3.zip
zip -qj /content/lcr_check3.zip $P/A_*_look4_vlm.json $P/A_*_look4_cons.json results/logs/c3_*.log
echo "CHECK3 ZIPPED $(date +%H:%M:%S)" >> results/logs/c3_cons2.log
