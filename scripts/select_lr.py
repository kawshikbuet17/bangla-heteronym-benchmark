"""Picks the best learning rate of a model by dev accuracy (ties: dev macro-F1, then the smaller lr)."""
import json
import sys
from pathlib import Path

from src.train import load_config

model = sys.argv[1]
grid = load_config(f"configs/{model}.yaml")["lr_grid"]
search = {}
for lr in grid:
    path = Path("runs", model, "lr_search", f"lr{lr:g}", "seed42", "metrics.json")
    if not path.exists():
        raise SystemExit(f"missing lr-search run: {path}")
    dev = json.loads(path.read_text())["dev"]
    search[f"{lr:g}"] = {"dev_accuracy": dev["accuracy"], "dev_macro_f1": dev["macro_f1"]}
best = max(grid, key=lambda lr: (search[f"{lr:g}"]["dev_accuracy"], search[f"{lr:g}"]["dev_macro_f1"], -lr))
result = {"lr": best, "criterion": "dev accuracy, then dev macro-F1, then smaller lr", "search": search}
Path("runs", model, "best_lr.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
