# bangla_heteronym_maker

Reads Bangla heteronyms in context: `কর` becomes `কর্‌` (tax), `করো` (do, now) or `কোরো` (do, later) depending on the
sentence. Covers the 183 heteronyms of the
[bangla-heteronyms](https://huggingface.co/datasets/kawshikbuet17/bangla-heteronyms) dataset. Self-contained: copy
this folder into your project; it needs only the packages in `requirements.txt` and a trained checkpoint folder.

## Setup

```bash
pip install -r bangla_heteronym_maker/requirements.txt   # use the PyTorch build that matches your CUDA
```

A checkpoint is a folder from the benchmark's `checkpoints/` (it holds the weights, the tokenizer and
`heteronym_config.json` with the word list), for example:

| Checkpoint | Test accuracy | p50 per sentence (T4 GPU / 4 CPU threads) |
|---|---|---|
| `banglabert` (recommended) | 99.46% | 12.5 ms / 33 ms |
| `banglabert_small` (fastest on CPU) | 98.16% | 12.6 ms / 13 ms |
| `banglabert_large_cand` (most accurate) | 99.73% | 23.8 ms / 139 ms |

## Use

```python
import bangla_heteronym_maker as bhm

bhm.initialize("/abs/path/checkpoints/banglabert")   # load once (singleton); or set BANGLA_HETERONYM_MODEL and call initialize()
reader = bhm.get_reader()                            # the same instance anywhere in the program

# 1. Whole text in, corrected text out (every heteronym replaced)
reader.phoneticize("আগে কর কাজটা, তারপর আবার কর। বিকেলে মাঠে বল নিয়ে যাব।")
# 'আগে কোরো কাজটা, তারপর আবার করো। বিকেলে মাঠে বল্‌ নিয়ে যাব।'

# 2. Sentence + word in, only the word's reading out (first occurrence of the word)
reader.predict("সময়মতো কর না দিলে জরিমানা গুনতে হবে।", "কর")          # 'কর্‌'

# 3. Marked text in, only the marked word's reading out (use when the word occurs more than once)
reader.predict_marked("সরকার কর বাড়িয়েছে, তাই হিসাবটা আবার [কর]।")    # 'করো'
reader.predict_marked("তুমি <<বল>> নিয়ে এসো।", marks=("<<", ">>"))     # other marks if [ ] can occur

# 4. Every occurrence with position and confidence
reader.readings("বিকেলে মাঠে বল নিয়ে খেলব, কাল আবার দেখা হবে।")
# [Reading(word='বল', start=12, end=14, reading='বল্‌', confidence=0.9997),
#  Reading(word='কাল', start=27, end=30, reading='কাল্‌', confidence=0.9995)]
```

Batch versions take and return lists. All occurrences of all inputs go through the model together, `batch_size`
(default 64, set in `initialize`) per forward pass, so they are much faster than calling the single methods in a loop:

```python
reader.phoneticize_batch([text1, text2, ...])                      # -> [str, ...]
reader.readings_batch([text1, text2, ...])                         # -> [[Reading, ...], ...]
reader.predict_batch([(sentence1, "কর"), (sentence2, "বল"), ...])   # -> [str, ...]
reader.predict_marked_batch(["... [কর] ...", "... [বল] ..."])        # -> [str, ...]
```

## Demo

`demo.py` calls every method once and prints the results (initialization, all single and batch methods, the errors,
batch vs loop timing, `reset`). From the folder that contains `bangla_heteronym_maker/`:

```bash
python -m bangla_heteronym_maker.demo /abs/path/checkpoints/banglabert            # GPU if available
python -m bangla_heteronym_maker.demo /abs/path/checkpoints/banglabert --device cpu
```

On a Tesla T4, 64 texts take 68 ms with `phoneticize_batch` and 677 ms with a loop of `phoneticize` calls (same output).

## Input rules

- Normal Unicode Bangla text. No marking is needed for `phoneticize`, `readings` and `predict`.
- Mark only one word per text, directly around it: `[কর]`, not `[ কর ]`.
- `predict` / `predict_marked` raise `ValueError` if the word is not one of the known heteronyms
  (`reader.heteronyms` lists them with their readings) or does not occur in the text. `phoneticize` leaves every
  other word unchanged.
- Paragraphs are fine: text is split into sentences at `।`, `?`, `!` and newlines, and each heteronym is read from its
  own sentence.
- Readings with a visible hasanta keep the ZWNJ (`কর্‌` = `কর` + U+09CD + U+200C); keep it for the TTS front end.

## Other options

- `initialize(path, device="cpu")` or `device="cuda:1"`; `precision="fp32"` to force fp32 on a GPU.
- `initialize` again with the same path returns the loaded reader; a different path raises `RuntimeError` until
  `bhm.reset()` is called. `bhm.is_initialized()` tells whether a model is loaded.
- `HeteronymReader` (in `interface.py`) is the abstract interface; `HeteronymModelReader` (in `reader.py`) implements
  it. Type hints in your code can use `HeteronymReader`.
- The reader can be shared by threads (model calls are serialized by a lock); one batch call is faster than many
  small calls from several threads.
