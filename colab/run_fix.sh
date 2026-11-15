#!/usr/bin/env bash
# Development round 2 (after colab/run_sink.sh): make Retell actually use its evidence.
# Diagnosis of round 1: gates stayed ~0.002, training verdicts were 31% contradicted / 60% uncertain,
# and false evidence gave identical outputs. Two causes are addressed here:
#   (1) noisy verdicts  -> interim thresholds calibrated on half B against the reference proxy (script 11),
#                          separately for attention regions (LCR) and detector regions (B3);
#   (2) redundant input -> patch dropout during training (--img-drop), so the evidence carries
#                          information the decoder cannot read off the (degraded) image.
# Every run is selected and reported on validation only. Runs go in parallel on one A100.
set -euo pipefail
cd "$(dirname "$0")/.."
# needs the sink-attention records of half B and validation (written by run_sink.sh before ours_sink starts)
until [ -e results/logs/ours_sink.log ] || [ -e results/retell_ours_sink.json ]; do sleep 20; done
P=data/cache/pipeline
L=results/logs
python scripts/11_proxy_thresholds.py --attn sink --records att > $L/calib_att_sink.log 2>&1; tail -2 $L/calib_att_sink.log
python scripts/11_proxy_thresholds.py --attn raw  --records det > $L/calib_det.log 2>&1;      tail -2 $L/calib_det.log
TA=results/thresholds_proxy_att_sink.json
TD=results/thresholds_proxy_det.json

job() {  # job <name> <env...> -- <args...>
  local name="$1"; shift
  [ -e "results/retell_${name}_sink.json" ] && { echo "[skip] $name"; return 0; }
  echo "[run ] $name $(date +%H:%M:%S)"
  env "$@" > "$L/$name.log" 2>&1 &
}
job ours_thr    LCR_THRESHOLDS=$TA python scripts/06_train_retell.py --mode ours   --attn sink --tag thr
job ours_fix    LCR_THRESHOLDS=$TA python scripts/06_train_retell.py --mode ours   --attn sink --tag fix --img-drop 0.5
job b0plus_fix  LCR_THRESHOLDS=$TA python scripts/06_train_retell.py --mode b0plus --attn sink --tag fix --img-drop 0.5
job b3_fix      LCR_THRESHOLDS=$TD python scripts/06_train_retell.py --mode b3     --attn sink --tag fix --img-drop 0.5
job ours_fixr30 LCR_THRESHOLDS=$TA python scripts/06_train_retell.py --mode ours   --attn sink --tag fixr30 --img-drop 0.5 --rho 0.3
job b2_fix      LCR_THRESHOLDS=$TD python scripts/06_train_retell.py --mode b2     --attn sink --tag fix --img-drop 0.5
wait
for n in ours_thr ours_fix b0plus_fix b3_fix ours_fixr30 b2_fix; do echo "== $n"; grep -E "^(real|masked|false|sensitivity)" $L/$n.log | cut -c1-200 || tail -5 $L/$n.log; done
until grep -q "SINK STAGE DONE" run_sink.log; do sleep 20; done   # final metrics/zip must include every run
rm -f results/full_metrics.json; python scripts/colab_full_metrics.py > $L/metrics_fix.log 2>&1 || tail -5 $L/metrics_fix.log
rm -f /content/lcr_results.zip
zip -qr /content/lcr_results.zip results report/figures report/generated $P/*.json $P/*.maps.npz
ls -lh /content/lcr_results.zip; echo "FIX STAGE DONE $(date +%H:%M:%S)"
