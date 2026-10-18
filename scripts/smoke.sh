#!/usr/bin/env bash
# ~20 steps per model on a tiny subset (one seed) to catch errors, OOM and NaN before the real runs.
# usage: bash scripts/smoke.sh [MODEL ...]   (default: all configs)
set -uo pipefail
source "$(dirname "$0")/env.sh"
models=${*:-$(ls configs | sed 's/\.yaml$//')}
for model in $models; do
  rm -rf "logs/smoke/$model"
  python -m src.train --config "configs/$model.yaml" --smoke > "logs/smoke_$model.log" 2>&1 \
    && echo "OK   $model: $(grep '^done' "logs/smoke_$model.log")" \
    || echo "FAIL $model: $(tail -1 "logs/smoke_$model.log")"
done
