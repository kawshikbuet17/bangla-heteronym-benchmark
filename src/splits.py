import argparse
import json

import numpy as np
import pandas as pd

from src.data import (
    DATASET_REVISION, LATENCY_PARAGRAPH, LATENCY_SENTENCES, REAL_TEST, SPLITS_DIR, ZWNJ,
    find_occurrences, label_rows, load_dataset, load_words, read_csv, word_pattern,
)

SEED = 42
N_TRAIN, N_DEV, N_TEST = 14, 3, 3
LEARNING_CURVE = (2, 5, 10, 14)
UNSEEN_FRACTION = 0.2
COLUMNS = ["spelling", "pronunciation", "sentence", "sentence_phonetic"]


def save(df: pd.DataFrame, name: str) -> None:
    df[COLUMNS].to_csv(SPLITS_DIR / f"{name}.csv", index=False, encoding="utf-8")


def create() -> None:
    if (SPLITS_DIR / "train.csv").exists():
        raise SystemExit(f"{SPLITS_DIR} already exists; splits are created once and never regenerated")
    SPLITS_DIR.mkdir(parents=True)
    df = load_dataset()
    words = load_words()

    shuffled = df.sample(frac=1, random_state=SEED)
    rank = shuffled.groupby(["spelling", "pronunciation"]).cumcount()
    train = shuffled[rank < N_TRAIN]
    dev = shuffled[(rank >= N_TRAIN) & (rank < N_TRAIN + N_DEV)]
    test = shuffled[rank >= N_TRAIN + N_DEV]
    for name, part in (("train", train), ("dev", dev), ("test", test)):
        save(part.sort_index(), name)

    # Nested learning-curve subsets: one fixed order per pronunciation, take the first k.
    lc_rank = train.sample(frac=1, random_state=SEED).groupby(["spelling", "pronunciation"]).cumcount()
    for k in LEARNING_CURVE[:-1]:
        save(train.loc[lc_rank[lc_rank < k].index].sort_index(), f"train_lc{k}")

    rng = np.random.default_rng(SEED)
    held = []
    for n in sorted({len(c) for c in words.values()}):
        stratum = sorted(w for w, c in words.items() if len(c) == n)
        k = max(1, round(UNSEEN_FRACTION * len(stratum)))
        held += list(rng.choice(stratum, size=k, replace=False))
    is_held = df["spelling"].isin(held)
    save(train[~train["spelling"].isin(held)].sort_index(), "unseen_train")
    save(dev[~dev["spelling"].isin(held)].sort_index(), "unseen_dev")
    save(df[is_held], "unseen_test")

    latency = test.sort_index().sample(n=200, random_state=SEED)
    latency[COLUMNS].to_csv(LATENCY_SENTENCES, index=False, encoding="utf-8")
    three = test[test["spelling"].map(lambda w: len(words[w]) == 3)].sample(n=2, random_state=SEED)
    rest = test[~test["spelling"].isin(three["spelling"])].drop_duplicates("spelling").sample(n=6, random_state=SEED)
    sentences = list(pd.concat([three, rest]).sample(frac=1, random_state=SEED)["sentence"])
    LATENCY_PARAGRAPH.write_text(" ".join(sentences[:4]) + "\n" + " ".join(sentences[4:]) + "\n", encoding="utf-8")

    meta = {"dataset_revision": DATASET_REVISION, "seed": SEED, "unseen_words": sorted(held)}
    (SPLITS_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def keys(df: pd.DataFrame) -> set[tuple[str, str]]:
    return set(zip(df["spelling"], df["sentence"]))


def check() -> dict:
    df, words = load_dataset(), load_words()
    assert len(df) == 7360, len(df)
    assert df["spelling"].nunique() == 183 and len(words) == 183
    assert df["pronunciation"].nunique() == 368 == sum(len(c) for c in words.values())
    assert (df.groupby(["spelling", "pronunciation"]).size() == 20).all()
    assert sorted(w for w, c in words.items() if len(c) == 3) == sorted(["কর", "খেল"])
    assert all(p in words[w] for w, p in zip(df["spelling"], df["pronunciation"]))
    assert df["pronunciation"].str.contains(ZWNJ).sum() > 0
    assert sum(ZWNJ in p for c in words.values() for p in c) > 0
    labeled = label_rows(df, words)
    rebuilt = [s[:a] + p + s[b:] for s, a, b, p in zip(labeled["sentence"], labeled["start"], labeled["end"], labeled["pronunciation"])]
    assert rebuilt == list(df["sentence_phonetic"]), "sentence_phonetic must equal sentence with the target replaced"

    split = {name: label_rows(read_csv(SPLITS_DIR / f"{name}.csv"), words) for name in
             ["train", "dev", "test", "train_lc2", "train_lc5", "train_lc10", "unseen_train", "unseen_dev", "unseen_test"]}
    for name, n in (("train", N_TRAIN), ("dev", N_DEV), ("test", N_TEST)):
        assert (split[name].groupby(["spelling", "pronunciation"]).size() == n).all(), name
        assert split[name]["pronunciation"].nunique() == 368, name
    tr, dv, te = keys(split["train"]), keys(split["dev"]), keys(split["test"])
    assert not (tr & dv or tr & te or dv & te), "main splits overlap"
    assert tr | dv | te == keys(df)
    previous = set()
    for k in LEARNING_CURVE[:-1]:
        part = split[f"train_lc{k}"]
        assert (part.groupby(["spelling", "pronunciation"]).size() == k).all()
        assert previous <= keys(part) <= tr, f"learning-curve subsets must be nested (lc{k})"
        previous = keys(part)
    held = set(split["unseen_test"]["spelling"])
    seen = set(split["unseen_train"]["spelling"]) | set(split["unseen_dev"]["spelling"])
    assert not held & seen, "unseen-word split leaks test words into train/dev"
    assert held | seen == set(words)
    assert len(split["unseen_test"]) == 20 * sum(len(words[w]) for w in held)
    assert {len(words[w]) for w in held} == {2, 3}

    latency = read_csv(LATENCY_SENTENCES)
    assert len(latency) == 200 and keys(latency) <= te
    paragraph = LATENCY_PARAGRAPH.read_text(encoding="utf-8")
    report = {
        "rows": len(df), "words": len(words), "pronunciations": int(df["pronunciation"].nunique()),
        "pronunciations_with_zwnj": int(sum(ZWNJ in p for c in words.values() for p in c)),
        "sizes": {name: len(part) for name, part in split.items()},
        "unseen_words": len(held), "unseen_word_fraction": round(len(held) / len(words), 3),
        "sentence_texts_shared_between_train_and_test": len(set(split["train"]["sentence"]) & set(split["test"]["sentence"])),
        "latency_sentences": len(latency),
        "latency_paragraph_heteronyms": len(find_occurrences(paragraph, word_pattern(list(words)))),
    }
    if REAL_TEST.exists():
        report["real_test"] = len(label_rows(read_csv(REAL_TEST), words))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["create", "check"])
    args = parser.parse_args()
    if args.command == "create":
        create()
    print(json.dumps(check(), ensure_ascii=False, indent=2))
