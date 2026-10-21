import re
import string
from collections.abc import Callable
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

DATASET_REPO = "kawshikbuet17/bangla-heteronyms"
DATASET_REVISION = "2c3d86b67571e15ba3520d3d7b11930fb8f34e3a"
DATASET_FILE = "bangla_heteronym_dataset.csv"
WORDS_FILE = "bangla_heteronym_words.csv"
RAW_DIR = Path("data/raw")
SPLITS_DIR = Path("data/splits")
REAL_TEST = Path("data/real_test.csv")
LATENCY_SENTENCES = Path("data/latency_sentences.csv")
LATENCY_PARAGRAPH = Path("data/latency_paragraph.txt")

# Pronunciations use hasanta + ZWNJ (U+09CD U+200C); the ZWNJ must never be stripped from data.
ZWNJ = "‌"
HASANTA = "্"

# Word boundaries: whitespace, Bengali danda/double danda, ASCII punctuation, plus the curly quotes and
# dashes that occur in the dataset. `\b` cannot be used: it splits Bengali words at vowel signs.
BOUNDARY_CHARS = "।॥" + string.punctuation + "“”‘’—–"
_B = re.escape(BOUNDARY_CHARS)
_STRIP = BOUNDARY_CHARS + " \t\n"

Marker = tuple[str, str] | None
Normalizer = Callable[[str], str]


# Local corrections to dataset revision DATASET_REVISION (owner, 2026-10-22): word -> the two readings whose
# sentences are swapped. ব্যাংককে "the bank" (object) is ব্যাংক্‌কে and "in/to Bangkok" is ব্যাংকোকে; the revision
# has them the other way round in all 40 sentences. load_dataset() applies the fix; the committed split files and
# latency inputs are fixed already. Checkpoints trained before the fix map their output with `reading_fixes` in
# heteronym_config.json (scripts/patch_checkpoints.py).
READING_SWAPS = {"ব্যাংককে": ("ব্যাংক্‌কে", "ব্যাংকোকে")}


def fix_readings(df: pd.DataFrame) -> pd.DataFrame:
    """Applies READING_SWAPS to rows with spelling / pronunciation / sentence / sentence_phonetic columns."""
    df = df.copy()
    for word, (a, b) in READING_SWAPS.items():
        rows = df["spelling"] == word
        swapped = df.loc[rows, "pronunciation"].map({a: b, b: a})
        for i in df.index[rows]:
            start, end, _ = find_occurrences(df.at[i, "sentence"], word_pattern([word]))[0]
            df.at[i, "sentence_phonetic"] = df.at[i, "sentence"][:start] + swapped[i] + df.at[i, "sentence"][end:]
        df.loc[rows, "pronunciation"] = swapped
    return df


def download() -> None:
    for name in (DATASET_FILE, WORDS_FILE):
        hf_hub_download(DATASET_REPO, name, repo_type="dataset", revision=DATASET_REVISION, local_dir=RAW_DIR)


def read_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False)


def load_dataset() -> pd.DataFrame:
    """The downloaded revision with READING_SWAPS applied (the file on disk stays the original revision)."""
    return fix_readings(read_csv(RAW_DIR / DATASET_FILE))


def load_words() -> dict[str, list[str]]:
    df = read_csv(RAW_DIR / WORDS_FILE)
    cols = ["pronunciation 1", "pronunciation 2", "pronunciation 3"]
    return {row["spelling"]: [row[c] for c in cols if row[c]] for _, row in df.iterrows()}


def word_pattern(words: list[str]) -> re.Pattern:
    alternation = "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True))
    return re.compile(rf"(?<![^\s{_B}])(?:{alternation})(?![^\s{_B}])")


def find_occurrences(text: str, pattern: re.Pattern) -> list[tuple[int, int, str]]:
    return [(m.start(), m.end(), m.group()) for m in pattern.finditer(text)]


def strip_punct(token: str) -> str:
    return token.strip(_STRIP)


def token_index(sentence: str, start: int) -> int:
    before = sentence[:start]
    n = len(before.split())
    return n - 1 if before and not before[-1].isspace() else n


def split_pieces(token: str) -> list[str]:
    return re.split(rf"[{_B}]", token)


def piece_index(sentence: str, start: int) -> int:
    """Index of the target inside its whitespace token split on punctuation (e.g. 1 for সাদা-কালো)."""
    token_prefix = re.split(r"\s", sentence[:start])[-1]
    return len(split_pieces(token_prefix)) - 1


def build_input(sentence: str, start: int, end: int, marker: Marker, norm: Normalizer) -> tuple[str, int, int, int]:
    """Marks one occurrence. Returns the text, the anchor char (opening marker, or the target if there is no
    marker) and the target's char span in the returned text."""
    left, target, right = norm(sentence[:start]), norm(sentence[start:end]), norm(sentence[end:])
    opening, closing = (marker[0] + " ", " " + marker[1]) if marker else ("", "")
    target_start = len(left) + len(opening)
    return left + opening + target + closing + right, len(left), target_start, target_start + len(target)


def label_rows(df: pd.DataFrame, words: dict[str, list[str]]) -> pd.DataFrame:
    """Adds the target span, whitespace token index, label and number of readings to dataset rows."""
    df = df.copy()
    spans = [find_occurrences(s, word_pattern([w])) for s, w in zip(df["sentence"], df["spelling"])]
    if any(len(s) != 1 for s in spans):
        bad = [s for s, sp in zip(df["sentence"], spans) if len(sp) != 1]
        raise ValueError(f"target must occur exactly once as a whole word: {bad[:3]}")
    df["start"] = [s[0][0] for s in spans]
    df["end"] = [s[0][1] for s in spans]
    df["token_idx"] = [token_index(s, st) for s, st in zip(df["sentence"], df["start"])]
    df["label"] = [words[w].index(p) for w, p in zip(df["spelling"], df["pronunciation"])]
    df["n_readings"] = [len(words[w]) for w in df["spelling"]]
    return df


def load_split(name: str) -> pd.DataFrame:
    return label_rows(read_csv(SPLITS_DIR / f"{name}.csv"), load_words())


if __name__ == "__main__":
    download()
    print(f"downloaded {DATASET_REPO}@{DATASET_REVISION} to {RAW_DIR}")
