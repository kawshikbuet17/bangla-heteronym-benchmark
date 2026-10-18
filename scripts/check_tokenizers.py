"""Logs, per model tokenizer, whether the ZWNJ survives (round trip and token ids) and how exact the round trip
is on the dataset. For seq2seq it also checks that references scored as predictions give 100% accuracy."""
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

from src.data import ZWNJ, label_rows, load_dataset, load_words
from src.evaluate import other_word_changed
from src.models import get_normalizer, keeps_zwnj
from src.predict import Predictor, roundtrip
from src.train import load_config

SAMPLE = "আশ্‌লো"

df = label_rows(load_dataset(), load_words())
words = load_words()
rows = []
for path in sorted(Path("configs").glob("*.yaml")):
    cfg = load_config(str(path))
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"], **cfg.get("tokenizer_kwargs", {}))
    seq2seq = cfg["model_type"] == "seq2seq"
    strip = seq2seq and not keeps_zwnj(tokenizer)
    norm = get_normalizer(cfg["normalize"], strip)
    texts = [get_normalizer(cfg["normalize"])(s) for s in df["sentence_phonetic"]]
    ids = tokenizer(texts)["input_ids"]
    row = {
        "model": cfg["model_name"],
        "tokenizer": type(tokenizer).__name__,
        "zwnj_kept_in_round_trip": keeps_zwnj(tokenizer),
        "zwnj_changes_token_ids": tokenizer(SAMPLE)["input_ids"] != tokenizer(SAMPLE.replace(ZWNJ, ""))["input_ids"],
        "phonetic_sentences_exact_round_trip": round(sum(tokenizer.decode(i, skip_special_tokens=True) == t for i, t in zip(ids, texts)) / len(texts), 4),
        "unk_tokens_per_sentence": round(sum(i.count(tokenizer.unk_token_id) for i in ids) / len(ids), 4) if tokenizer.unk_token_id is not None else 0.0,
        "strip_zwnj_for_seq2seq": strip,
    }
    if seq2seq:
        predictor = Predictor.__new__(Predictor)
        predictor.tokenizer, predictor.norm, predictor.candidates, predictor.hconfig = tokenizer, norm, words, {"output": "sentence"}
        predictor.canon = {w: [predictor.canonical(c) for c in cands] for w, cands in words.items()}
        refs = [roundtrip(tokenizer, norm(s)) for s in df["sentence_phonetic"]]
        preds = [predictor.from_output(r, item) for r, item in zip(refs, zip(df["sentence"], df["start"], df["end"], df["spelling"]))]
        row["reference_as_prediction_accuracy"] = sum(p.aligned and p.pronunciation == g for p, g in zip(preds, df["pronunciation"])) / len(df)
        row["reference_other_word_changes"] = sum(other_word_changed(r, r, i) for r, i in zip(refs, df["token_idx"]))
        row["colliding_candidates"] = sum(len(set(c)) != len(c) for c in predictor.canon.values())
    rows.append(row)
    print(row, flush=True)
Path("results").mkdir(exist_ok=True)
pd.DataFrame(rows).to_csv("results/tokenizer_zwnj.csv", index=False)
