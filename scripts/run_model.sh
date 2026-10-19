#!/usr/bin/env bash
# Full per-model cycle: lr search -> 3 seeds -> keep best checkpoint -> latency -> demo -> requirements -> aggregate.
# usage: bash scripts/run_model.sh MODEL [all|train|measure]   (resumable: finished runs are skipped)
#   train   = lr search, seeds, keep best checkpoint;  measure = latency, demo, requirements, aggregation
set -euo pipefail
source "$(dirname "$0")/env.sh"
model=$1
phase=${2:-all}
config=configs/$model.yaml
exec > >(tee -a "logs/$model.log") 2>&1
echo "=== $model $phase $(date -Is) ==="

if [[ $phase == all || $phase == train ]]; then
  for lr in $(python -c "import sys, yaml; print(*yaml.safe_load(open(sys.argv[1]))['lr_grid'])" "$config"); do
    python -m src.train --config "$config" --lr-search "$lr"
  done
  python -m scripts.select_lr "$model"
  for seed in 42 43 44; do
    python -m src.train --config "$config" --seed "$seed"
  done
  python -m scripts.keep_best "$model" main "checkpoints/$model"
fi
if [[ $phase == all || $phase == measure ]]; then
  python -m src.benchmark_latency --checkpoint "checkpoints/$model" --out "runs/$model/latency.json"
  python -m src.predict --checkpoint "checkpoints/$model" --file data/latency_paragraph.txt > "runs/$model/phoneticize_demo.txt"
  python -m scripts.write_requirements "$model"
  python -m scripts.aggregate
fi
echo "=== $model $phase done $(date -Is) ==="
