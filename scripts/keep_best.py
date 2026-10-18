"""Keeps the checkpoint of the best seed (by dev accuracy) of one model/setting in `dest` and deletes the others.

usage: python -m scripts.keep_best MODEL SETTING DEST   e.g. banglabert main checkpoints/banglabert
"""
import json
import shutil
import sys
from pathlib import Path

from src.train import CHECKPOINT_RUNS, RUNS, dir_size_mb

model, setting, dest = sys.argv[1], sys.argv[2], Path(sys.argv[3])
runs = sorted(Path(RUNS, model, setting).glob("seed*/metrics.json"))
if not runs:
    raise SystemExit(f"no finished runs in {RUNS / model / setting}")
dev = {p.parent.name: json.loads(p.read_text())["dev"]["accuracy"] for p in runs}
best = max(sorted(dev), key=lambda seed: dev[seed])
source = CHECKPOINT_RUNS / model / setting / best
if source.exists():
    if dest.exists():
        shutil.rmtree(dest)
    shutil.move(str(source), dest)
elif not dest.exists():
    raise SystemExit(f"neither {source} nor {dest} exists")
shutil.rmtree(CHECKPOINT_RUNS / model / setting, ignore_errors=True)
if setting == "main":
    shutil.rmtree(CHECKPOINT_RUNS / model / "lr_search", ignore_errors=True)
if CHECKPOINT_RUNS.joinpath(model).exists() and not any(CHECKPOINT_RUNS.joinpath(model).iterdir()):
    CHECKPOINT_RUNS.joinpath(model).rmdir()
kept = {"seed": int(best.removeprefix("seed")), "dev_accuracy_by_seed": dev, "checkpoint": str(dest), "size_mb": dir_size_mb(dest)}
Path(RUNS, model, setting, "kept_checkpoint.json").write_text(json.dumps(kept, indent=2))
print(json.dumps(kept, indent=2))
