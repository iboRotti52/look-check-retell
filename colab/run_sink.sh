#!/usr/bin/env bash
# Second stage: rerun Look/Check and the attention-dependent Retell systems with
# sink-corrected attention maps. Run after colab/run_all.sh has finished.
set -euo pipefail
cd "$(dirname "$0")/.."
run() { local name="$1"; shift; echo "[run ] $name $(date +%H:%M:%S)"; ( "$@" ) > "results/logs/$name.log" 2>&1 || { echo "[FAIL] $name"; tail -30 "results/logs/$name.log"; exit 1; }; tail -2 "results/logs/$name.log"; }
P=data/cache/pipeline
[ -e $P/A_validation_sink.json ] || run drafts_val_sink python scripts/04_draft_look_check.py --split validation --decoder A --attn sink --region-rule v1
[ -e $P/A_B_sink.json ]          || run drafts_B_sink   python scripts/04_draft_look_check.py --split B --decoder A --attn sink --region-rule v1
[ -e results/retell_ours_sink.json ] || run ours_sink python scripts/06_train_retell.py --mode ours --rho 0.15 --attn sink
[ -e results/attention_variants.json ] || run attn_variants python scripts/09_attention_variants.py --n 1000 --decoder A
rm -f results/full_metrics.json; run metrics python scripts/colab_full_metrics.py
rm -f /content/lcr_results.zip
zip -qr /content/lcr_results.zip results report/figures report/generated $P/*.json $P/*.maps.npz
ls -lh /content/lcr_results.zip; echo "SINK STAGE DONE $(date +%H:%M:%S)"
