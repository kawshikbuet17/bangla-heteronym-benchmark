"""Re-scores the saved unseen-split predictions after the ব্যাংককে label fix (src.data.READING_SWAPS).

On the main split the swap was consistent between training and test, so those results stand. On the unseen split
ব্যাংককে is a held-out test word, so its gold readings were wrong. This rewrites `gold` and `correct` for that
word in predictions.csv and recomputes the test metrics in metrics.json with src.evaluate.summarize (the model
predictions themselves are unchanged). Idempotent: it compares with the fixed labels in data/splits/unseen_test.csv.

    python -m scripts.rescore_label_fix && python -m scripts.aggregate
"""
import json
from pathlib import Path

import pandas as pd

from src.data import READING_SWAPS, read_csv
from src.evaluate import summarize

TEST = read_csv("data/splits/unseen_test.csv")


def rescore(run: Path) -> None:
    predictions = pd.read_csv(run / "predictions.csv", keep_default_na=False)
    gold = dict(zip(TEST["sentence"], TEST["pronunciation"]))
    fixed = predictions["spelling"].isin(READING_SWAPS)
    predictions.loc[fixed, "gold"] = predictions.loc[fixed, "sentence"].map(gold)
    correct = predictions["prediction"] == predictions["gold"]
    if "raw_output" in predictions:  # seq2seq: a misaligned output counts as wrong (src.predict.from_output)
        aligned = [len(o.split()) == len(s.split()) for o, s in zip(predictions["raw_output"], predictions["sentence"])]
        correct &= pd.Series(aligned, index=predictions.index)
    predictions.loc[fixed, "correct"] = correct[fixed]
    predictions["correct"] = predictions["correct"].astype(str) == "True"
    predictions.to_csv(run / "predictions.csv", index=False)
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    metrics["unseen_test"].update(summarize(predictions))
    metrics["unseen_test"]["label_fix"] = "ব্যাংককে gold readings corrected and rescored on 2026-10-22 (scripts/rescore_label_fix.py)"
    (run / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{run}: {metrics['unseen_test']['accuracy']:.4f}")


def main() -> None:
    for run in sorted(Path("runs").glob("*/*unseen/seed*")):
        rescore(run)


if __name__ == "__main__":
    main()
