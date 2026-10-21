# Bangla heteronym disambiguation benchmark

A Bangla heteronym is a word with one spelling and several pronunciations. Given a sentence and a target word in it,
a model picks the pronunciation that is meant. This repo fine-tunes and compares 12 pretrained models (encoders as
classifiers or candidate scorers, and T5-style seq2seq models) on the same fixed splits, for accuracy and latency,
and ships a copyable inference package.

```
input : তুমি গ্রাফ পেপারে সরলরেখাটা আঁক।      target: আঁক   readings: আঁক্‌ | আঁকো
output: তুমি গ্রাফ পেপারে সরলরেখাটা আঁকো।
```

Dataset: [kawshikbuet17/bangla-heteronyms](https://huggingface.co/datasets/kawshikbuet17/bangla-heteronyms) (7,360
sentences, 183 words, 368 pronunciations, 20 sentences each). Readings use hasanta + ZWNJ (`্‌`, U+09CD U+200C); the
ZWNJ is never removed from the data. In the pinned revision the two readings of ব্যাংককে are swapped; this repo
corrects them (see Notes). Also see [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) (every run, problem and decision),
[PROGRESS.md](PROGRESS.md) and [results/](results/).

## 1. Setup

```bash
conda create -n hetero python=3.11 -y && conda activate hetero
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126   # CUDA 12.6 build
pip install -r requirements.txt               # everything, or only one model: requirements/banglabert.txt
```

`scripts/env.sh` pins the GPU (`CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1` on the original server; override
it elsewhere). All scripts source it.

## 2. Data and splits

```bash
python -m src.data                   # download the CSVs (pinned dataset commit) to data/raw/
python -m src.splits check           # data and split checks
python -m pytest tests               # tests
```

The splits in `data/splits/` were made once (seed 42) and must never be regenerated.

| Split | Content |
|---|---|
| `train` / `dev` / `test` | 14 / 3 / 3 sentences per pronunciation (5,152 / 1,104 / 1,104) |
| `unseen_train` / `unseen_dev` / `unseen_test` | 37 words held out completely (their 1,500 sentences form `unseen_test`) |
| `train_lc2` / `train_lc5` / `train_lc10` | nested subsets with 2 / 5 / 10 sentences per pronunciation (learning curves; 14 = `train`) |

One row of `data/raw/bangla_heteronym_dataset.csv` (the split files have the same columns):

```csv
spelling,pronunciation,sentence,sentence_phonetic
কর,কর্‌,এ বছর থেকে মোবাইল ফোনের ওপর কর বাড়ছে।,এ বছর থেকে মোবাইল ফোনের ওপর কর্‌ বাড়ছে।
```

`data/raw/bangla_heteronym_words.csv` lists each word's readings (`কর,কর্‌,করো,কোরো`); their order is the label index.

## 3. Approaches

The target is marked as `@ word @` so the model knows which occurrence is meant. Training, testing and inference build
the same input. Example: `হাত-মুখ ধুয়ে তুমি খাওয়া শুরু কর।`, target কর, gold করো.

| `model_type` | Model input | Model output |
|---|---|---|
| `encoder` (classifier) | `[CLS] কর [SEP] হাত - মুখ ধুয়ে তুমি খাওয়া শুরু @ কর @ । [SEP]` | one score per reading (from the opening `@`); highest = করো |
| `candidate` (scorer) | one row per reading: `[CLS] ... শুরু @ কর @ । [SEP] করো [SEP]` (and কর্, কোরো) | one score per row; highest = করো; also works for unseen words |
| `seq2seq` | `phonetic: হাত-মুখ ধুয়ে তুমি খাওয়া শুরু @ কর @।` | the sentence rewritten, `হাত-মুখ ধুয়ে তুমি খাওয়া শুরু করো।`; the word at the target's position |

## 4. Experiments

Every experiment is a **setting** (`SETTINGS` in `src/train.py`), trained with seeds 42, 43, 44 at the model's selected
learning rate (learning curves: seed 42 only) and reported as mean ± std. Output: `runs/{model}/{setting}/seed{seed}/`.

| Setting | Question | Models | Results |
|---|---|---|---|
| `lr_search` | best learning rate (one seed-42 run per value in `lr_grid`, judged on dev) | all | `results/lr_search.csv` |
| `main` | the benchmark: accuracy, macro-F1, per-word accuracy | all 12 | `results/summary.md`, `per_word.csv` |
| `cand_main`, `cand_unseen` | candidate scoring on the main split and on unseen words | 3 best encoders | `results/summary.md` |
| `unseen` | seq2seq on unseen words | 3 seq2seq | `results/summary.md` |
| `lc2`, `lc5`, `lc10` | learning curve: how many sentences per pronunciation are needed | all 12 | `results/learning_curve.csv` |
| `abl_marker_special`, `abl_marker_none` | `<h> word </h>` or no marker instead of `@ word @` | BanglaBERT-large | `results/ablations.md` |
| `abl_head_mean`, `abl_head_cls` | mean of the target's tokens or `[CLS]` instead of the opening marker | BanglaBERT-large | `results/ablations.md` |
| `abl_output_target(_keep)` | seq2seq writes only the reading instead of the sentence | BanglaT5 | `results/ablations.md` |

ByT5-small was planned as a 13th model and dropped (about 50 GPU hours on the T4); see EXPERIMENT_LOG.md.

## 5. Run

```bash
bash scripts/run_model.sh banglabert                  # full cycle: lr search, 3 seeds, keep best, latency, aggregate
bash scripts/run_setting.sh cand_main banglabert      # one setting, seeds 42-44
SEEDS=42 bash scripts/run_setting.sh lc2 banglabert   # learning curves: seed 42 only
python -m scripts.aggregate                           # results/ tables and plots
```

Single steps:

```bash
python -m src.train --config configs/banglabert.yaml --lr-search 3e-5   # one lr-search run
python -m scripts.select_lr banglabert                                  # -> runs/banglabert/best_lr.json
python -m src.train --config configs/banglabert.yaml --seed 43 [--setting cand_main]
python -m scripts.keep_best banglabert main checkpoints/banglabert
python -m src.evaluate --checkpoint checkpoints/banglabert --data data/real_test.csv --out runs/banglabert/real_test.json
python -m src.benchmark_latency --checkpoint checkpoints/banglabert --out runs/banglabert/latency.json
bash scripts/smoke.sh                                                   # 20-step smoke test of every model
```

A run whose `metrics.json` exists is skipped, so an interrupted command can be re-run. Recipe (same for all models,
`configs/{model}.yaml`): AdamW, weight decay 0.01, 10% warm-up then linear decay, clipping 1.0, max length 128, at most
10 epochs (encoders) or 15 (seq2seq), early stopping (patience 3) on dev accuracy, best epoch kept.

Each run writes `config.json` (config, package versions, GPU, git commit), `metrics.json` (train time, epochs, memory,
params; dev and test accuracy, macro-F1, per-word accuracy), `predictions.csv` (every test sentence with gold,
prediction, confidence) and `train_log.csv`. Latency (`runs/{model}/latency*.json`, `results/latency.csv`) is the full
path (normalizer, tokenizer, model, decoding) on the T4 and on a CPU with 4 and 1 threads: p50/p95/p99 at batch 1 (200
sentences after 20 warm-up calls) and throughput at batch 8 and 32.

## 6. Use a trained model

Kept checkpoints (not committed): `checkpoints/{model}` is the best `main` seed of each model,
`checkpoints/{model}_cand` the best candidate scorer.

```bash
python -m src.predict --checkpoint checkpoints/banglabert --text "তুমি একটু অপেক্ষা কর, আমি আসছি। বিকেলে তুমি পার্কে খেল।"
# তুমি একটু অপেক্ষা করো, আমি আসছি। বিকেলে তুমি পার্কে খ্যালো।   (+ one line per word with its confidence, and the time)
```

**In another project**, copy the self-contained package `bangla_heteronym_package/` (see its
[README](bangla_heteronym_package/README.md) and `demo.py`):

```python
import bangla_heteronym_package as bhp
reader = bhp.initialize("/abs/path/checkpoints/banglabert")              # once (singleton)
reader.phoneticize("আগে কর কাজটা, তারপর আবার কর।")                       # 'আগে কোরো কাজটা, তারপর আবার করো।'
reader.predict("সময়মতো কর না দিলে জরিমানা গুনতে হবে।", "কর")              # 'কর্‌'
reader.predict_marked("সরকার কর বাড়িয়েছে, তাই হিসাবটা আবার [কর]।")         # 'করো'
reader.phoneticize_marked("সরকার কর বাড়িয়েছে, তাই হিসাবটা আবার [কর]।")     # 'সরকার কর বাড়িয়েছে, তাই হিসাবটা আবার করো।'
reader.phoneticize_batch([text1, text2])                                  # every method has a *_batch version
```

`src.predict.Predictor` (used by the benchmark) has the same methods (`Predictor.load(path)`), plus
`phoneticize_detailed` (position and confidence per word) and `predict_items` (character offsets).

**What the model receives and returns.** Every method sends the model one sentence with the target word marked by
`@` and gets back the reading of that word; the methods only choose which words to send, and your `[ ]` marks never
reach the model. With `checkpoints/banglabert` for `... তাই তুমি এখনই হিসাবটা আবার [কর]।`:

| Model type | The model receives | The model returns | We take |
|---|---|---|---|
| Encoder | `কর` + `সরকার কর বাড়িয়েছে, তাই তুমি এখনই হিসাবটা আবার @ কর @।` | `কর্‌ 0.0003, করো 0.9699, কোরো 0.0298` | the highest: **করো** |
| Candidate scorer | the marked sentence + `কর্‌` / `করো` / `কোরো` (one row each) | `0.0001, 0.9995, 0.0005` | the highest: **করো** |
| Seq2seq | `phonetic: সরকার কর বাড়িয়েছে, তাই তুমি এখনই হিসাবটা আবার @ কর @।` | `সরকার কর বাড়িয়েছে, তাই তুমি এখনই হিসাবটা আবার করো।` | the word at `@`: **করো** |

The package README and `demo.py` show this for every method and model type.

**Batch inference** is supported by all checkpoints: BanglaBERT-small, BanglaBERT, Bangla-BERT-base, MuRIL-base,
XLM-R-base, IndicBERTv2-MLM, mDeBERTa-v3-base, BanglaBERT-large, XLM-R-large, BanglaT5, mT5-small, mT5-base and the
three candidate scorers. The `*_batch` methods send up to 64 words per model call (about 10 times faster than a loop
on the GPU); a checkpoint without batch support would give a warning and run one input at a time.

**Which checkpoint:** `banglabert` (99.46%, 12.5 ms on the T4, 33 ms on 4 CPU threads) in general,
`banglabert_small` on a CPU (98.16%, 13 ms), `banglabert_large_cand` for the best accuracy (99.73%, 23.8 / 139 ms).

## 7. Results

Main split test set (1,104 sentences), mean ± std over 3 seeds; latency p50 per sentence at batch 1. Full tables in
[results/summary.md](results/summary.md), [results/latency.csv](results/latency.csv),
[results/ablations.md](results/ablations.md) and [results/learning_curve.csv](results/learning_curve.csv); plots in
[results/plots/](results/plots/) (accuracy against size and latency, learning curves, hardest words).

<!-- results-table -->
| # | model | size | precision | best lr | test acc (%) | macro-F1 (%) | train time / seed (min) | GPU p50 (ms) | CPU p50 (ms, 4 thr) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | banglabert_small | 13.7M | fp16 | 0.0001 | 98.16 ± 0.14 | 98.09 ± 0.13 | 2.5 | 12.6 | 13.0 |
| 2 | banglabert | 110.0M | fp16 | 2e-05 | 99.46 ± 0.16 | 99.40 ± 0.16 | 3.7 | 12.5 | 32.8 |
| 3 | bangla_bert_base | 164.4M | fp16 | 2e-05 | 98.01 ± 0.39 | 97.93 ± 0.41 | 4.4 | 10.6 | 30.5 |
| 4 | muril_base | 237.6M | fp16 | 5e-05 | 95.89 ± 1.04 | 95.54 ± 1.13 | 5.7 | 10.3 | 39.8 |
| 5 | banglat5 | 247.6M | fp32 | 0.0005 | 98.37 ± 0.18 | 98.37 ± 0.12 | 31.4 | 236.8 | 323.3 |
| 6 | xlm_roberta_base | 278.0M | fp16 | 2e-05 | 97.37 ± 0.39 | 97.27 ± 0.36 | 6.5 | 10.3 | 37.5 |
| 7 | indicbertv2_mlm | 278.0M | fp16 | 5e-05 | 98.34 ± 0.21 | 98.29 ± 0.21 | 6.2 | 11.3 | 31.4 |
| 8 | mdeberta_v3_base | 278.2M | fp32 | 3e-05 | 97.61 ± 0.14 | 97.48 ± 0.10 | 21.2 | 24.9 | 118.2 |
| 9 | mt5_small | 300.2M | fp32 | 0.0003 | 93.36 ± 0.21 | 92.99 ± 0.18 | 57.3 | 307.6 | 923.6 |
| 10 | banglabert_large | 335.6M | fp16 | 2e-05 | 99.18 ± 0.16 | 99.14 ± 0.16 | 11.7 | 22.8 | 107.8 |
| 11 | xlm_roberta_large | 559.9M | fp16 | 2e-05 | 98.43 ± 0.19 | 98.34 ± 0.14 | 23.5 | 17.8 | 126.4 |
| 12 | mt5_base | 582.4M | fp32 | 0.0003 | 96.11 ± 0.16 | 95.99 ± 0.24 | 124.6 | 436.2 | 1758.6 |
<!-- /results-table -->

The table between the `results-table` markers is rewritten by `scripts.aggregate`; don't edit it by hand.

### Further experiments

Accuracy in %, mean ± std over 3 seeds; GPU p50 in ms.

**Candidate scoring and unseen words** (`cand_main`, `cand_unseen`, `unseen`). The unseen test set has 1,500
sentences of 37 words that never occur in training or dev.

| model | approach | main test | unseen test | GPU p50 |
|---|---|---|---|---|
| banglabert_large | classifier | 99.18 ± 0.16 | n/a | 22.8 |
| banglabert_large | candidate | 99.73 ± 0.09 | 83.11 ± 2.38 | 23.8 |
| banglabert | classifier | 99.46 ± 0.16 | n/a | 12.5 |
| banglabert | candidate | 99.37 ± 0.16 | 82.31 ± 0.92 | 13.6 |
| xlm_roberta_large | classifier | 98.43 ± 0.19 | n/a | 17.8 |
| xlm_roberta_large | candidate | 83.30 ± 23.82 | 61.96 ± 21.28 | 19.9 |
| banglat5 | seq2seq | 98.37 ± 0.18 | 55.09 ± 1.85 | 236.8 |
| mt5_small | seq2seq | 93.36 ± 0.21 | 51.69 ± 0.63 | 307.6 |
| mt5_base | seq2seq | 96.11 ± 0.16 | 58.07 ± 0.88 | 436.2 |

- XLM-R-large failed to train as a candidate scorer in 3 of 6 runs (loss stuck at chance level, about 50% accuracy);
  they are kept in the mean. Its converged runs reach 97.0/97.1% (main) and 86.5% (unseen).
- On unseen words the seq2seq models produce one of the word's readings in only 68-74% of the sentences (otherwise an
  invented pronunciation); the candidate scorer always picks one of the readings, so it generalizes much better.

**Learning curve** (seed 42 only; 14 is the seed-42 main run), test accuracy by sentences per pronunciation:

| model | 2 | 5 | 10 | 14 |
|---|---|---|---|---|
| banglabert_small | 91.85 | 93.57 | 96.83 | 98.28 |
| banglabert | 91.94 | 97.19 | 98.73 | 99.28 |
| bangla_bert_base | 88.50 | 93.93 | 97.01 | 98.28 |
| muril_base | 88.86 | 93.39 | 94.38 | 96.92 |
| banglat5 | 83.97 | 96.65 | 98.73 | 98.55 |
| xlm_roberta_base | 87.59 | 93.75 | 97.92 | 97.55 |
| indicbertv2_mlm | 91.58 | 96.47 | 97.83 | 98.46 |
| mdeberta_v3_base | 92.39 | 95.65 | 97.83 | 97.46 |
| mt5_small | 52.72 | 81.70 | 92.66 | 93.48 |
| banglabert_large | 93.57 | 97.74 | 98.73 | 99.09 |
| xlm_roberta_large | 90.94 | 95.65 | 97.64 | 98.28 |
| mt5_base | 80.98 | 92.03 | 95.47 | 96.20 |

The biggest gain is from 2 to 5 sentences per pronunciation. At 10, every encoder except MuRIL is within 1.5 points
of its result at 14; mT5-small needs the most data.

**Ablations** (BanglaBERT-large and BanglaT5, 3 seeds; [results/ablations.md](results/ablations.md)): no marker
99.67 ± 0.10 and mean pooling of the target tokens 99.52 ± 0.14 beat the default `@ word @` marker with the
opening-marker head (99.18 ± 0.16); `<h>` tokens 99.34 ± 0.14; `[CLS]` 99.09 ± 0.31. BanglaT5 writing only the
target's pronunciation: 98.28 ± 0.47 (full sentence 98.37 ± 0.18) at 86.1 ms GPU p50 instead of 236.8 ms.

## 8. Notes

- **ব্যাংককে label fix (2026-10-22).** In dataset revision `2c3d86b` the 20 Bangkok sentences of ব্যাংককে are
  labelled ব্যাংক্‌কে and the 20 bank sentences ব্যাংকোকে; it is the other way round. `src.data.READING_SWAPS`
  corrects them in `load_dataset()`, the committed splits and latency inputs are corrected, and all checkpoints, which
  were trained before the fix, map their output with `reading_fixes` in `heteronym_config.json`
  (`scripts/patch_checkpoints.py`). Main-split results are unchanged (the swap was the same in training and test);
  the unseen split, where ব্যাংককে is a test word, was rescored from the saved predictions
  (`scripts/rescore_label_fix.py`).
- **ZWNJ after hasanta.** No tokenizer here round-trips it (`results/tokenizer_zwnj.csv`). WordPiece models
  (BanglaBERT, bangla-bert-base, MuRIL) silently drop it; SentencePiece models give different ids but cannot decode
  it back, and T5 tokenizers decode it as a space, which splits a word in two. So the data keeps it, seq2seq models
  remove it from their targets (`strip_zwnj`), and a seq2seq output is compared to the candidates after the same
  normalizer + tokenizer round trip, then mapped back to the original candidate (with its ZWNJ).
- **bangla-bert-base lowercasing.** `sagorsarker/bangla-bert-base` has no tokenizer config, so the default
  lowercasing strips Bangla vowel signs. Its config sets `tokenizer_kwargs: {do_lower_case: false, strip_accents: false}`.
- **Word boundaries.** Python's `\b` does not work for Bangla (vowel signs are not word characters), so heteronyms are
  found with an explicit boundary set (`BOUNDARY_CHARS` in `src/data.py`: `।`, `॥`, ASCII punctuation, curly quotes,
  dashes, whitespace).
- **Normalizer.** BanglaBERT (all sizes) and BanglaT5 inputs go through the csebuetnlp normalizer (pinned git commit
  in `requirements.txt`) before tokenization, as their authors recommend.
- **Automatic fallbacks** (`src/train.py`, recorded in `config.json` under `adjustments` and in EXPERIMENT_LOG):
  on CUDA OOM the batch is halved and gradient accumulation doubled (same effective batch), then gradient
  checkpointing; on a non-finite fp16 loss the run restarts in fp32. mT5, mDeBERTa and BanglaT5 are set to fp32 in their
  configs, because T5 and DeBERTa-v3 models are known to be unstable in fp16.
- **Seq2seq memory at evaluation.** Generation scores are collected with a small logits processor
  (`GreedyLogProbs`) instead of `output_scores`, which kept the whole vocabulary distribution per step and ran out of
  memory for mT5.
- **Shared GPU (original server).** Two idle Triton inference services also live on the T4. They were stopped while
  the three largest models and all further experiments trained (to free memory) and were running, idle, during every
  latency measurement, so all latency numbers share the same conditions.
- **mT5 epoch limit.** Both mT5 models were still improving at the 15-epoch limit; their numbers are a lower bound
  under this shared recipe.
- **Reproducibility.** Splits, configs and seeds are fixed; `config.json` records package versions and the git commit
  of every run. Results are reproducible up to GPU non-determinism (expect differences within the seed std).

## 9. Adding a model or data

- **New model:** copy a config in `configs/` with the same `model_type`, set `model_name`, `model_id`, `precision`,
  `batch_size` and `lr_grid`; run `bash scripts/smoke.sh <model>` first, then `bash scripts/run_model.sh <model>`, and
  add it to `MODELS` in `scripts/aggregate.py` (table order). `scripts.write_requirements` writes its requirements file.
- **New heteronyms or sentences:** the dataset is pinned to one commit in `src/data.py`. A new dataset version needs
  new splits (a new split seed or folder, never overwriting the old ones) and re-running every model, otherwise the
  numbers are not comparable. The learning curve tells roughly how many sentences per pronunciation to write.

## 10. Repo layout

```
configs/        one YAML per model
requirements/   pinned packages per model
src/            data, splits, models, train, evaluate, predict, benchmark_latency, env
scripts/        run_model.sh, run_setting.sh, smoke.sh, env.sh, select_lr, keep_best, aggregate,
                write_requirements, check_tokenizers
bangla_heteronym_package/  copyable inference package (interface, implementation, singleton, demo, README, .env.example)
tests/          phoneticize, package and data tests
data/splits/    the fixed splits;  data/latency_*: latency inputs
runs/           per-run config/metrics/predictions/train logs, latency*.json, phoneticize demos
checkpoints/    kept best models (not committed)
logs/           training logs (not committed)
results/        aggregated tables and plots
```

## 11. Prepared by

**Kawshik Kumar Paul**  
Senior Software Engineer | Researcher  
Department of Computer Science and Engineering (CSE)  
Bangladesh University of Engineering and Technology (BUET)  
Email: kawshikbuet17@gmail.com  
