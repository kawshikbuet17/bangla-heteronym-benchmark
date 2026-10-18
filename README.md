# Bangla heteronym disambiguation benchmark

## 1. What this is

A Bangla heteronym is a word with one spelling and several pronunciations. Given a sentence and one target word
in it, the model picks which pronunciation is meant. Several encoders (classification, candidate scoring) and
seq2seq models are fine-tuned and compared on the same splits, for accuracy and latency.

```
input : তুমি গ্রাফ পেপারে সরলরেখাটা আঁক।      target: আঁক   candidates: আঁক্‌ | আঁকো
output: তুমি গ্রাফ পেপারে সরলরেখাটা আঁকো।
```

Dataset: [kawshikbuet17/bangla-heteronyms](https://huggingface.co/datasets/kawshikbuet17/bangla-heteronyms)
(public, 7,360 sentences, 183 words, 368 pronunciations). Pronunciations use hasanta + ZWNJ (`্‌`, U+09CD U+200C);
the ZWNJ is never removed from the data.

## 2. Setup

```bash
conda create -n hetero python=3.11 -y && conda activate hetero
# PyTorch 2.7.1 built for CUDA 12.6 (works with NVIDIA driver >= 560)
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt                 # everything
# or only what one model needs:
pip install -r requirements/banglabert.txt
```

No Hugging Face login is needed. On the original server the T4 is GPU index 1; `scripts/env.sh` pins it with
`CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1` (override `CUDA_VISIBLE_DEVICES` elsewhere).

## 3. Data

```bash
python -m src.data                     # download the CSVs (pinned dataset commit) to data/raw/
python -m src.splits check             # data + split checks (the splits in data/splits/ are committed)
python -m scripts.check_tokenizers     # ZWNJ / round-trip report per tokenizer -> results/tokenizer_zwnj.csv
python -m pytest tests                 # phoneticize + data tests
```

`python -m src.splits create` made the splits once (seed 42) and refuses to overwrite them. Splits: main
(14/3/3 sentences per pronunciation), unseen-word (37 held-out words, test only), learning-curve subsets
(2/5/10/14 per pronunciation, nested). Put a real-text test set in `data/real_test.csv` (same columns) and every
later run also reports it; for existing checkpoints use `src.evaluate` (below).

## 4. Run experiments

Full cycle for one model (lr search, 3 seeds, keep best checkpoint, latency, demo, requirements, aggregation),
best inside tmux; logs go to `logs/{model}.log`:

```bash
bash scripts/run_model.sh banglabert_small
```

Single steps (run from the repo root inside the env, with the T4 pinned: `source scripts/env.sh`):

```bash
python -m src.train --config configs/banglabert.yaml --lr-search 3e-5      # one lr-search run (seed 42, dev only)
python -m scripts.select_lr banglabert                                     # -> runs/banglabert/best_lr.json
python -m src.train --config configs/banglabert.yaml --seed 43             # one seed with the best lr
python -m src.train --config configs/banglabert.yaml --seed 42 --setting cand_main   # other settings, see SETTINGS in src/train.py
python -m scripts.keep_best banglabert main checkpoints/banglabert         # keep best seed's checkpoint
python -m src.evaluate --checkpoint checkpoints/banglabert --data data/real_test.csv --out runs/banglabert/real_test.json
python -m src.benchmark_latency --checkpoint checkpoints/banglabert --out runs/banglabert/latency.json
python -m scripts.aggregate                                                # results/ tables and plots
bash scripts/smoke.sh                                                      # 20-step smoke test of every model
```

Resuming: every run whose `runs/{model}/{setting}/seed{seed}/metrics.json` exists is skipped, so re-running
the same command continues where it stopped. Training curves: `tensorboard --logdir runs`.

## 5. Use a trained model

```bash
python -m src.predict --checkpoint checkpoints/banglabert --text "তুমি একটু অপেক্ষা কর, আমি আসছি। বিকেলে পার্কে খেল।"
```

prints the phonetic text, one line per heteronym (word, position, prediction, confidence) and the time in ms.

```python
from src.predict import Predictor

predictor = Predictor.load("checkpoints/banglabert")      # any kept checkpoint folder: encoder, candidate or seq2seq
text = "তুমি একটু অপেক্ষা কর, আমি আসছি। কর কাজটা, তারপর আবার কর!\nবিকেলে তুমি পার্কে খেল।"
print(predictor.phoneticize(text))                        # every heteronym replaced, text otherwise unchanged
print(predictor.predict("তুমি গ্রাফ পেপারে সরলরেখাটা আঁক।", "আঁক"))
```

`phoneticize` splits the text on `।`, `?`, `!` and newlines, finds every whole-word occurrence of a known
heteronym, marks one occurrence at a time, predicts all of them in one batch and rebuilds the exact text.

## 6. Results

Full tables: [results/summary.md](results/summary.md), [results/latency.csv](results/latency.csv),
[EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) (setup, problems, decisions, observations).

<!-- results-table -->
| # | model | size | precision | best lr | test acc (%) | macro-F1 (%) | train time / seed (min) | GPU p50 (ms) | CPU p50 (ms, 4 thr) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | banglabert_small | 13.7M | fp16 | 0.0001 | 98.16 ± 0.14 | 98.09 ± 0.13 | 2.5 | 12.6 | 13.0 |
| 2 | banglabert | 110.0M | fp16 | 2e-05 | 99.46 ± 0.16 | 99.40 ± 0.16 | 3.7 | 12.5 | 32.8 |
| 3 | bangla_bert_base | 164.4M | fp16 | 2e-05 | 98.01 ± 0.39 | 97.93 ± 0.41 | 4.4 | 10.6 | 30.5 |
| 4 | muril_base | 237.6M | fp16 | 5e-05 | 95.89 ± 1.04 | 95.54 ± 1.13 | 5.7 | 10.3 | 39.8 |
| 5 | banglat5 | 247.6M | fp32 | 0.0005 | 98.37 ± 0.18 | 98.37 ± 0.12 | 31.4 | 235.6 | 317.4 |
| 6 | xlm_roberta_base | 278.0M | fp16 | 2e-05 | 97.37 ± 0.39 | 97.27 ± 0.36 | 6.5 | 10.3 | 37.5 |
<!-- /results-table -->

## 7. Repo layout

```
configs/        one YAML per model
requirements/   pinned packages per model
src/            data, splits, models, train, evaluate, predict, benchmark_latency, env
scripts/        run_model.sh, smoke.sh, select_lr, keep_best, aggregate, write_requirements, check_tokenizers
tests/          phoneticize and data tests
data/splits/    the fixed splits;  data/latency_*: latency inputs
runs/           per-run config/metrics/predictions/train logs, latency.json, phoneticize demos
checkpoints/    kept best models (not committed)
results/        aggregated tables and plots
```

## 8. Prepared by

**Kawshik Kumar Paul**  
Senior Software Engineer | Researcher  
Department of Computer Science and Engineering (CSE)  
Bangladesh University of Engineering and Technology (BUET)  
Email: kawshikbuet17@gmail.com  
