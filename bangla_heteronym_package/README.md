# bangla_heteronym_package

Reads Bangla heteronyms in context: `কর` becomes `কর্‌` (tax), `করো` (do, now) or `কোরো` (do, later), depending on
the sentence. Covers the 183 heteronyms of the
[bangla-heteronyms](https://huggingface.co/datasets/kawshikbuet17/bangla-heteronyms) dataset. Copy this folder into
your project; it needs the packages in `requirements.txt` and a trained checkpoint folder from the benchmark.

## Setup

```bash
pip install -r bangla_heteronym_package/requirements.txt                 # use the PyTorch build for your CUDA
cp bangla_heteronym_package/.env.example bangla_heteronym_package/.env   # optional: set the model path there
```

Recommended checkpoint: `checkpoints/banglabert` (99.46% test accuracy, ~12.5 ms per sentence on a T4 GPU, ~33 ms on 4
CPU threads). `checkpoints/banglabert_small` is the fastest on a CPU (98.16%, 13 ms), `checkpoints/banglabert_large_cand`
the most accurate (99.73%, 23.8 ms GPU / 139 ms CPU).

## Use

```python
import bangla_heteronym_package as bhp

reader = bhp.initialize("/abs/path/checkpoints/banglabert")   # loads once (singleton); bhp.get_reader() elsewhere

reader.phoneticize("আগে কর কাজটা, তারপর আবার কর।")                    # 'আগে কোরো কাজটা, তারপর আবার করো।'
reader.predict("সময়মতো কর না দিলে জরিমানা গুনতে হবে।", "কর")           # 'কর্‌'
reader.predict_marked("সরকার কর বাড়িয়েছে, তাই হিসাবটা আবার [কর]।")      # 'করো'
reader.phoneticize_marked("সরকার কর বাড়িয়েছে, তাই হিসাবটা আবার [কর]।")  # 'সরকার কর বাড়িয়েছে, তাই হিসাবটা আবার করো।'
```

| Method | Input | Output |
|---|---|---|
| `phoneticize(text)` | any text | the text with every heteronym replaced by its reading |
| `predict(sentence, word)` | a sentence and a heteronym in it | the reading of its first occurrence |
| `predict_marked(text)` | text with a heteronym marked: `[কর]` | the reading of the marked word |
| `phoneticize_marked(text)` | text with one or more heteronyms marked | the text with only the marked words replaced, marks removed |
| `*_batch(list)` | a list of the above | a list of the above, all through the model together |
| `heteronyms` | | dict: each heteronym and its readings |

`demo.py` calls every method and shows the model input and output under each call:
`python -m bangla_heteronym_package.demo /abs/path/checkpoints/banglabert`.

## What the model receives and returns

The model always receives **one sentence with the target word marked by `@`** and returns **the reading of that word**.
The methods only choose which words to send (all heteronyms, the given word, or the marked words); your `[ ]` marks
never reach the model. Afterwards the reading is returned, or put back into your text.

| Model type | The model receives | The model returns | We take |
|---|---|---|---|
| Encoder (BanglaBERT etc., recommended) | the word + the marked sentence: `কর` + `সরকার কর বাড়িয়েছে, তাই তুমি এখনই হিসাবটা আবার @ কর @।` | one probability per reading: `কর্‌ 0.0003, করো 0.9699, কোরো 0.0298` | the highest: **করো** |
| Candidate scorer (`*_cand`) | the marked sentence once per reading: `... আবার @ কর @।` + `কর্‌` / `করো` / `কোরো` | one probability per pair: `0.0001, 0.9995, 0.0005` | the highest: **করো** |
| Seq2seq (BanglaT5, mT5) | `phonetic: সরকার কর বাড়িয়েছে, তাই তুমি এখনই হিসাবটা আবার @ কর @।` | the sentence rewritten: `সরকার কর বাড়িয়েছে, তাই তুমি এখনই হিসাবটা আবার করো।` | the word at the marked position: **করো** |

Before the call, the sentence is cut out of your text (split at `।`, `?`, `!`, newlines), normalized for the BanglaBERT
and BanglaT5 checkpoints, and the T5 models drop the ZWNJ (their output word is mapped back to the reading with it).

Each method with the real model input and output of the three model types (`# ->` is the result with
`checkpoints/banglabert`; the models do not always agree):

```python
# Text in, corrected text out
print(reader.phoneticize("আগে কর কাজটা, তারপর আবার কর। বিকেলে মাঠে বল নিয়ে যাব।"))
#   encoder (banglabert): in = word + marked sentence, out = probability per reading
#     in:  কর + আগে @ কর @ কাজটা, তারপর আবার কর।
#     out: কর্‌ 0.2683, করো 0.0227, কোরো 0.7090 -> কোরো
#     in:  কর + আগে কর কাজটা, তারপর আবার @ কর @।
#     out: কর্‌ 0.0013, করো 0.7762, কোরো 0.2225 -> করো
#     in:  বল + বিকেলে মাঠে @ বল @ নিয়ে যাব।
#     out: বল্‌ 0.9998, বলো 0.0002 -> বল্‌
#   candidate (banglabert_cand): in = one row per reading (marked sentence + reading), out = probability per row
#     in:  আগে @ কর @ কাজটা, তারপর আবার কর। + each of কর্‌ / করো / কোরো
#     out: কর্‌ 0.0555, করো 0.9440, কোরো 0.0004 -> করো
#     in:  আগে কর কাজটা, তারপর আবার @ কর @। + each of কর্‌ / করো / কোরো
#     out: কর্‌ 0.0084, করো 0.9821, কোরো 0.0095 -> করো
#     in:  বিকেলে মাঠে @ বল @ নিয়ে যাব। + each of বল্‌ / বলো
#     out: বল্‌ 1.0000, বলো 0.0000 -> বল্‌
#   seq2seq (banglat5): in = phonetic: marked sentence, out = generated sentence (word at the marked position)
#     in:  phonetic: আগে @ কর @ কাজটা, তারপর আবার কর।
#     out: আগে করো কাজটা, তারপর আবার করো। -> করো
#     in:  phonetic: আগে কর কাজটা, তারপর আবার @ কর @।
#     out: আগে করো কাজটা, তারপর আবার করো। -> করো
#     in:  phonetic: বিকেলে মাঠে @ বল @ নিয়ে যাব।
#     out: বিকেলে মাঠে বল্ নিয়ে যাব। -> বল্‌
# -> আগে কোরো কাজটা, তারপর আবার করো। বিকেলে মাঠে বল্‌ নিয়ে যাব।

# Sentence + word in, only the word's reading out (first occurrence of the word)
print(reader.predict("সময়মতো কর না দিলে জরিমানা গুনতে হবে।", "কর"))
#   encoder (banglabert): in = word + marked sentence, out = probability per reading
#     in:  কর + সময়মতো @ কর @ না দিলে জরিমানা গুনতে হবে।
#     out: কর্‌ 0.9868, করো 0.0002, কোরো 0.0130 -> কর্‌
#   candidate (banglabert_cand): in = one row per reading (marked sentence + reading), out = probability per row
#     in:  সময়মতো @ কর @ না দিলে জরিমানা গুনতে হবে। + each of কর্‌ / করো / কোরো
#     out: কর্‌ 1.0000, করো 0.0000, কোরো 0.0000 -> কর্‌
#   seq2seq (banglat5): in = phonetic: marked sentence, out = generated sentence (word at the marked position)
#     in:  phonetic: সময়মতো @ কর @ না দিলে জরিমানা গুনতে হবে।
#     out: সময়মতো কর্ না দিলে জরিমানা গুনতে হবে। -> কর্‌
# -> কর্‌

# Marked word in, only its reading out (use it when the word occurs more than once)
print(reader.predict_marked("সরকার কর বাড়িয়েছে, তাই হিসাবটা আবার [কর]।"))
#   encoder (banglabert): in = word + marked sentence, out = probability per reading
#     in:  কর + সরকার কর বাড়িয়েছে, তাই হিসাবটা আবার @ কর @।
#     out: কর্‌ 0.0071, করো 0.6238, কোরো 0.3691 -> করো
#   candidate (banglabert_cand): in = one row per reading (marked sentence + reading), out = probability per row
#     in:  সরকার কর বাড়িয়েছে, তাই হিসাবটা আবার @ কর @। + each of কর্‌ / করো / কোরো
#     out: কর্‌ 0.9955, করো 0.0026, কোরো 0.0019 -> কর্‌
#   seq2seq (banglat5): in = phonetic: marked sentence, out = generated sentence (word at the marked position)
#     in:  phonetic: সরকার কর বাড়িয়েছে, তাই হিসাবটা আবার @ কর @।
#     out: সরকার কর্ বাড়িয়েছে, তাই হিসাবটা আবার করো। -> করো
# -> করো

# Marked text in, whole text out: only the marked words are replaced, the marks are removed
print(reader.phoneticize_marked("সরকার কর বাড়িয়েছে, তাই হিসাবটা আবার [কর]।"))
#   encoder (banglabert): in = word + marked sentence, out = probability per reading
#     in:  কর + সরকার কর বাড়িয়েছে, তাই হিসাবটা আবার @ কর @।
#     out: কর্‌ 0.0071, করো 0.6238, কোরো 0.3691 -> করো
#   candidate (banglabert_cand): in = one row per reading (marked sentence + reading), out = probability per row
#     in:  সরকার কর বাড়িয়েছে, তাই হিসাবটা আবার @ কর @। + each of কর্‌ / করো / কোরো
#     out: কর্‌ 0.9955, করো 0.0026, কোরো 0.0019 -> কর্‌
#   seq2seq (banglat5): in = phonetic: marked sentence, out = generated sentence (word at the marked position)
#     in:  phonetic: সরকার কর বাড়িয়েছে, তাই হিসাবটা আবার @ কর @।
#     out: সরকার কর্ বাড়িয়েছে, তাই হিসাবটা আবার করো। -> করো
# -> সরকার কর বাড়িয়েছে, তাই হিসাবটা আবার করো।
```

The `*_batch` versions send the same rows, for all inputs, through the model together.

## Batch inference

All checkpoints support batch inference (64 words per model call): the 12 models BanglaBERT-small, BanglaBERT,
Bangla-BERT-base, MuRIL-base, XLM-R-base, IndicBERTv2-MLM, mDeBERTa-v3-base, BanglaBERT-large, XLM-R-large (encoders),
BanglaT5, mT5-small, mT5-base (seq2seq), and the candidate scorers `banglabert_cand`, `banglabert_large_cand`,
`xlm_roberta_large_cand`. GPU throughput (Tesla T4, sentences per second):

| Model | Batch 1 | Batch 32 |
|---|---|---|
| BanglaBERT-small / BanglaBERT / BanglaBERT-large | 77 / 79 / 43 | 1488 / 1500 / 820 |
| Bangla-BERT-base / MuRIL-base / IndicBERTv2-MLM | 89 / 96 / 89 | 2068 / 1976 / 1972 |
| XLM-R-base / XLM-R-large / mDeBERTa-v3-base | 94 / 55 / 38 | 1845 / 668 / 235 |
| BanglaT5 / mT5-small / mT5-base | 4 / 3 / 2 | 67 / 42 / 24 |
| candidate: BanglaBERT / BanglaBERT-large / XLM-R-large | 71 / 41 / 49 | 1001 / 327 / 306 |

A checkpoint without batch support (`"batch": false` in its `heteronym_config.json`, or an unknown model type) gives
`UserWarning: ... does not support batch inference; the *_batch methods will process their inputs one at a time`, and
the batch methods then run one input at a time. `reader.supports_batch` tells which case applies.

## Settings

| Variable | Meaning |
|---|---|
| `BANGLA_HETERONYM_MODEL` | checkpoint folder, used when `initialize()` gets no path |
| `BANGLA_HETERONYM_DEVICE` | `cuda`, `cuda:1` or `cpu` (default: cuda if available) |

Read from the environment, else from `.env` in this folder; arguments to `initialize()` win. `initialize()` again
returns the loaded reader; call `bhp.reset()` first to load another model.

## Input rules

- Plain Unicode Bangla. Mark a word only to choose a specific occurrence: `[কর]`, no spaces inside the marks
  (other marks: `marks=("<<", ">>")`). Do not pass marked text to `phoneticize`; use `phoneticize_marked`.
- A word that is not one of the 183 heteronyms, or not in the text, raises `ValueError`. `phoneticize` leaves all
  other words unchanged.
- Readings with a visible hasanta keep the ZWNJ (`কর্‌` = কর + U+09CD + U+200C); keep it for the TTS front end.

## Files

`interface.py` interface (`HeteronymReader`), `reader.py` implementation (`HeteronymModelReader`), `modeling.py`
model classes and checkpoint loading, `text.py` text helpers, `__init__.py` `initialize` / `get_reader` / `reset`,
`demo.py` demo, `.env.example` settings.
