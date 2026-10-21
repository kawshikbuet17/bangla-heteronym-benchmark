# bangla_heteronym_package

Reads Bangla heteronyms in context: `কর` becomes `কর্‌` (tax), `করো` (do, now) or `কোরো` (do, later) depending on the
sentence. 183 heteronyms. Copy this folder into your project; it needs the packages in `requirements.txt` and a
trained checkpoint folder from the benchmark (`checkpoints/banglabert` is recommended: 99.46% test accuracy, ~12.5 ms
per sentence on a T4 GPU, ~33 ms on 4 CPU threads).

```bash
pip install -r bangla_heteronym_package/requirements.txt
cp bangla_heteronym_package/.env.example bangla_heteronym_package/.env    # optional: set the model path there
```

## Use

```python
import bangla_heteronym_package as bhp

reader = bhp.initialize("/abs/path/checkpoints/banglabert")   # once (singleton); initialize() alone uses .env / env vars
reader = bhp.get_reader()                                     # the same reader anywhere else

reader.phoneticize("আগে কর কাজটা, তারপর আবার কর।")             # 'আগে কোরো কাজটা, তারপর আবার করো।'
reader.predict("সময়মতো কর না দিলে জরিমানা গুনতে হবে।", "কর")     # 'কর্‌'
reader.predict_marked("সরকার কর বাড়িয়েছে, তাই হিসাবটা আবার [কর]।")  # 'করো'
reader.phoneticize_marked("সরকার কর বাড়িয়েছে, তাই হিসাবটা আবার [কর]।")  # 'সরকার কর বাড়িয়েছে, তাই হিসাবটা আবার করো।'

reader.phoneticize_batch([text1, text2])                      # lists in, lists out, one model call per 64 words
reader.predict_batch([(sentence1, "কর"), (sentence2, "বল")])
reader.predict_marked_batch([marked1, marked2])
reader.phoneticize_marked_batch([marked1, marked2])
```

`demo.py` calls every method and shows its output: `python -m bangla_heteronym_package.demo /abs/path/checkpoints/banglabert`.

| Method | Input | Output |
|---|---|---|
| `phoneticize(text)` | any text | the text with every heteronym replaced by its reading |
| `predict(sentence, word)` | a sentence and a heteronym in it | the reading of its first occurrence |
| `predict_marked(text, marks=("[", "]"))` | text with a heteronym marked, `[কর]` | the reading of the marked word |
| `phoneticize_marked(text, marks=("[", "]"))` | text with one or more heteronyms marked | the text with only the marked words replaced and the marks removed |
| `*_batch(list)` | a list of the above | a list of the above |
| `heteronyms` | | dict: each heteronym and its readings |

## Batch inference

Every benchmark model supports batch inference: encoder classifiers, candidate scorers and seq2seq models all run on
padded batches (64 heteronym occurrences per model call in the `*_batch` methods). Measured GPU throughput on a Tesla
T4 (full path, 200 test sentences):

| Model | Checkpoint | Type | Batch inference | Sentences/s, batch 1 | Sentences/s, batch 32 |
|---|---|---|---|---|---|
| BanglaBERT-small | `checkpoints/banglabert_small` | encoder classifier | yes | 77 | 1488 |
| BanglaBERT | `checkpoints/banglabert` | encoder classifier | yes | 79 | 1500 |
| Bangla-BERT-base | `checkpoints/bangla_bert_base` | encoder classifier | yes | 89 | 2068 |
| MuRIL-base | `checkpoints/muril_base` | encoder classifier | yes | 96 | 1976 |
| XLM-R-base | `checkpoints/xlm_roberta_base` | encoder classifier | yes | 94 | 1845 |
| IndicBERTv2-MLM | `checkpoints/indicbertv2_mlm` | encoder classifier | yes | 89 | 1972 |
| mDeBERTa-v3-base | `checkpoints/mdeberta_v3_base` | encoder classifier | yes | 38 | 235 |
| BanglaBERT-large | `checkpoints/banglabert_large` | encoder classifier | yes | 43 | 820 |
| XLM-R-large | `checkpoints/xlm_roberta_large` | encoder classifier | yes | 55 | 668 |
| BanglaT5 | `checkpoints/banglat5` | seq2seq | yes | 4 | 67 |
| mT5-small | `checkpoints/mt5_small` | seq2seq | yes | 3 | 42 |
| mT5-base | `checkpoints/mt5_base` | seq2seq | yes | 2 | 24 |
| BanglaBERT | `checkpoints/banglabert_cand` | candidate scorer | yes | 71 | 1001 |
| BanglaBERT-large | `checkpoints/banglabert_large_cand` | candidate scorer | yes | 41 | 327 |
| XLM-R-large | `checkpoints/xlm_roberta_large_cand` | candidate scorer | yes | 49 | 306 |
| BanglaT5 target-only | `checkpoints/_runs/banglat5/abl_output_target_keep/seed42` | seq2seq | yes | 12 | 205 |

If a checkpoint does not support batching (a model type not listed above, or `"batch": false` in its
`heteronym_config.json`), loading it prints a warning

```
UserWarning: <checkpoint> does not support batch inference; the *_batch methods will process their inputs one at a time
```

and the `*_batch` methods still work, one input at a time. `reader.supports_batch` tells which case applies.

## Settings

| Variable | Meaning |
|---|---|
| `BANGLA_HETERONYM_MODEL` | checkpoint folder, used when `initialize()` gets no path |
| `BANGLA_HETERONYM_DEVICE` | `cuda`, `cuda:1` or `cpu` (default: cuda if available) |

They are read from the environment, else from `.env` in this folder. A path or device given to `initialize()` wins.
`initialize()` again returns the loaded reader; to load another model, call `bhp.reset()` first.

## Input rules

- Plain Unicode Bangla. Mark a word when you want only that occurrence read: `[কর]`, no spaces inside the marks.
  `predict_marked` reads the first marked word; `phoneticize_marked` replaces every marked word.
- Do not pass marked text to `phoneticize`: it reads every heteronym and keeps the marks as punctuation.
- A word that is not one of the 183 heteronyms, or not in the text, raises `ValueError` (in `predict` and
  `predict_marked`); `phoneticize` leaves all other words unchanged.
- Paragraphs are fine: each heteronym is read from its own sentence (split at `।`, `?`, `!` and newlines).
- Readings with a visible hasanta keep the ZWNJ (`কর্‌` = কর + U+09CD + U+200C); keep it for the TTS front end.

## Files

`interface.py` the interface (`HeteronymReader`), `reader.py` its implementation (`HeteronymModelReader`),
`modeling.py` the model classes and checkpoint loading, `text.py` text helpers, `__init__.py` `initialize` /
`get_reader` / `reset`, `demo.py` the demo.
