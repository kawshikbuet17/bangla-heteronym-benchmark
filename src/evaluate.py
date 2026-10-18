import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.data import label_rows, load_words, read_csv
from src.predict import Predictor, roundtrip


def summarize(predictions: pd.DataFrame) -> dict:
    gold = predictions["gold"]
    return {
        "n": len(predictions),
        "accuracy": float(predictions["correct"].mean()),
        "macro_f1": float(f1_score(gold, predictions["prediction"], labels=sorted(set(gold)), average="macro", zero_division=0)),
        "accuracy_by_readings": {str(k): float(v) for k, v in predictions.groupby("n_readings")["correct"].mean().items()},
        "accuracy_by_word": {k: float(v) for k, v in predictions.groupby("spelling")["correct"].mean().items()},
    }


def other_word_changed(output: str, reference: str, idx: int) -> bool:
    out, ref = output.split(), reference.split()
    return len(out) != len(ref) or any(o != r for i, (o, r) in enumerate(zip(out, ref)) if i != idx)


def evaluate(predictor: Predictor, df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """df must be labeled with `label_rows`."""
    items = list(zip(df["sentence"], df["start"], df["end"], df["spelling"]))
    preds = predictor.predict_items(items)
    out = df[["spelling", "sentence", "n_readings"]].copy()
    out["gold"] = df["pronunciation"].values
    out["prediction"] = [p.pronunciation for p in preds]
    out["confidence"] = [round(p.confidence, 6) for p in preds]
    out["correct"] = [p.aligned and p.pronunciation == g for p, g in zip(preds, out["gold"])]
    metrics = summarize(out)
    if predictor.kind == "seq2seq":
        out["raw_output"] = [p.output for p in preds]
        metrics["avg_generated_tokens"] = float(np.mean([p.generated_tokens for p in preds]))
        if predictor.hconfig["output"] == "sentence":
            references = [roundtrip(predictor.tokenizer, predictor.norm(s)) for s in df["sentence_phonetic"]]
            changed = [other_word_changed(p.output, r, i) for p, r, i in zip(preds, references, df["token_idx"])]
            metrics["other_word_change_rate"] = float(np.mean(changed))
    return metrics, out


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved checkpoint on any labeled CSV (e.g. data/real_test.csv).")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True, help="metrics json; predictions are written next to it as .csv")
    args = parser.parse_args()
    metrics, predictions = evaluate(Predictor.load(args.checkpoint), label_rows(read_csv(args.data), load_words()))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    predictions.to_csv(out.with_suffix(".csv"), index=False, encoding="utf-8")
    print(f"{args.data}: accuracy {metrics['accuracy']:.4f}, macro-F1 {metrics['macro_f1']:.4f} (n={metrics['n']})")


if __name__ == "__main__":
    main()
