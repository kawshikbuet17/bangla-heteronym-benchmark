#!/usr/bin/env bash
# ~20 steps per model on a tiny subset (one seed) to catch errors, OOM and NaN before the real runs.
# usage: bash scripts/smoke.sh [MODEL[:SETTING] ...]   (default: every config, main setting)
set -uo pipefail
source "$(dirname "$0")/env.sh"
targets=${*:-$(ls configs | sed 's/\.yaml$//')}
for target in $targets; do
  model=${target%%:*}; setting=main; [[ $target == *:* ]] && setting=${target#*:}
  log="logs/smoke_${model}_${setting}.log"
  rm -rf "logs/smoke/$model/$setting"
  python -m src.train --config "configs/$model.yaml" --setting "$setting" --smoke > "$log" 2>&1 \
    && echo "OK   $model:$setting $(grep '^done' "$log" | sed 's/^done [^:]*: //')" \
    || echo "FAIL $model:$setting $(tail -1 "$log")"
done
