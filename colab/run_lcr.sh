#!/usr/bin/env bash
# Look/Check verification only. Retell is explicitly deferred by the user.
# Uses existing caches; no training or new GPU model execution is needed for this check.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results/logs
python scripts/39_verify_lcr.py > results/logs/lcr_contract.log 2>&1
echo 'Look/Check contracts verified. No Retell training, validation, or test evaluation was run.'
