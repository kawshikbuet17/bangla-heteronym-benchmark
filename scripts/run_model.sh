#!/usr/bin/env bash
# Full per-model cycle: lr search -> 3 seeds -> keep best checkpoint -> latency -> demo -> requirements -> aggregate.
# usage: bash scripts/run_model.sh MODEL   (resumable: finished runs are skipped)
set -euo pipefail
source "$(dirname "$0")/env.sh"
model=$1
config=configs/$model.yaml
exec > >(tee -a "logs/$model.log") 2>&1
echo "=== $model $(date -Is) ==="

for lr in $(python -c "import sys, yaml; print(*yaml.safe_load(open(sys.argv[1]))['lr_grid'])" "$config"); do
  python -m src.train --config "$config" --lr-search "$lr"
done
python -m scripts.select_lr "$model"
for seed in 42 43 44; do
  python -m src.train --config "$config" --seed "$seed"
done
python -m scripts.keep_best "$model" main "checkpoints/$model"
python -m src.benchmark_latency --checkpoint "checkpoints/$model" --out "runs/$model/latency.json"
python -m src.predict --checkpoint "checkpoints/$model" --file data/latency_paragraph.txt > "runs/$model/phoneticize_demo.txt"
python -m scripts.write_requirements "$model"
python -m scripts.aggregate
echo "=== $model done $(date -Is) ==="
