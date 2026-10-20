#!/usr/bin/env bash
# Trains one setting (candidate scoring, unseen words, learning curve, ablation) with seeds 42, 43, 44 for the given
# models, using each model's selected learning rate. For cand_main the best seed's checkpoint is kept in
# checkpoints/{model}_cand. Resumable: finished runs are skipped.
# usage: bash scripts/run_setting.sh SETTING MODEL [MODEL ...]     e.g. bash scripts/run_setting.sh lc2 banglabert
set -euo pipefail
source "$(dirname "$0")/env.sh"
setting=$1
shift
run_one() {
  local model=$1
  echo "=== $model $setting $(date -Is) ==="
  for seed in 42 43 44; do
    python -m src.train --config "configs/$model.yaml" --seed "$seed" --setting "$setting"
  done
  if [[ $setting == cand_main ]]; then
    python -m scripts.keep_best "$model" cand_main "checkpoints/${model}_cand"
  fi
  python -m scripts.aggregate
  echo "=== $model $setting done $(date -Is) ==="
}

for model in "$@"; do
  run_one "$model" 2>&1 | tee -a "logs/$model.log"
done
