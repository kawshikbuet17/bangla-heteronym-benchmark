# Bangla heteronym disambiguation benchmark

## 1. What this is

A Bangla heteronym is a word with one spelling and several pronunciations. Given a sentence and one target word
in it, the model picks which pronunciation is meant. 12 pretrained models (encoders used as classifiers or
candidate scorers, and T5-style seq2seq models) are fine-tuned and compared on the same fixed splits, for accuracy
and latency.

```
input : তুমি গ্রাফ পেপারে সরলরেখাটা আঁক।      target: আঁক   candidates: আঁক্‌ | আঁকো
output: তুমি গ্রাফ পেপারে সরলরেখাটা আঁকো।
```

Dataset: [kawshikbuet17/bangla-heteronyms](https://huggingface.co/datasets/kawshikbuet17/bangla-heteronyms)
(public, 7,360 sentences, 183 words, 368 pronunciations, 20 sentences per pronunciation). Pronunciations use
hasanta + ZWNJ (`্‌`, U+09CD U+200C); the ZWNJ is never removed from the data.

Where to look first:

| File | What it holds |
|---|---|
| this README | how to run everything, what every experiment means, things to know |
| [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) | dated log of every run, problem, fix and decision, plus observations for the paper |
| [PROGRESS.md](PROGRESS.md) | the step plan and which steps are done |
| [results/](results/) | aggregated tables and plots (regenerated from `runs/` by `scripts.aggregate`) |

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
`CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1` (override `CUDA_VISIBLE_DEVICES` elsewhere). All scripts
source `scripts/env.sh`; run single Python steps after `source scripts/env.sh`.

## 3. Data and splits

```bash
python -m src.data                     # download the CSVs (pinned dataset commit) to data/raw/
python -m src.splits check             # data + split checks (the splits in data/splits/ are committed)
python -m scripts.check_tokenizers     # ZWNJ / round-trip report per tokenizer -> results/tokenizer_zwnj.csv
python -m pytest tests                 # phoneticize + data tests
```

`python -m src.splits create` made the splits once (seed 42) and refuses to overwrite them. Never regenerate them:
every result in this repo depends on these exact files.

| Split file(s) | Content | Used by |
|---|---|---|
| `train` / `dev` / `test` | 14 / 3 / 3 sentences per pronunciation (5,152 / 1,104 / 1,104) | everything ("main") |
| `unseen_train` / `unseen_dev` / `unseen_test` | 37 words (20% of the words, stratified by number of pronunciations) are held out completely; their sentences form `unseen_test`; train/dev are the main ones without those words | `unseen`, `cand_unseen` |
| `train_lc2` / `train_lc5` / `train_lc10` | the first 2 / 5 / 10 training sentences per pronunciation (736 / 1,840 / 3,680) | learning curves |

The learning-curve subsets are nested (`lc2` ⊂ `lc5` ⊂ `lc10` ⊂ `train`), and they use the normal dev and test sets,
so the points are comparable. "14" on a learning curve is simply the main run.

A real-text test set can be added as `data/real_test.csv` (same columns); every later run also reports it, and for
existing checkpoints use `src.evaluate` (section 6).

### Sample rows

`data/raw/bangla_heteronym_dataset.csv` has one row per sentence; `sentence_phonetic` is the sentence with only the
target replaced by its pronunciation (here the three readings of কর):

```csv
spelling,pronunciation,sentence,sentence_phonetic
কর,কর্‌,এ বছর থেকে মোবাইল ফোনের ওপর কর বাড়ছে।,এ বছর থেকে মোবাইল ফোনের ওপর কর্‌ বাড়ছে।
কর,করো,"তুমি এখনই খাতাটা খুলে আজকের হোমওয়ার্কের অঙ্কগুলো শেষ কর, তারপর যত খুশি টিভি দেখবে।","তুমি এখনই খাতাটা খুলে আজকের হোমওয়ার্কের অঙ্কগুলো শেষ করো, তারপর যত খুশি টিভি দেখবে।"
কর,কোরো,"পরীক্ষার হলে তুমি প্রশ্ন ভালোভাবে না পড়ে উত্তর লেখা শুরু কর না, তাতে ভুল বেশি হয়।","পরীক্ষার হলে তুমি প্রশ্ন ভালোভাবে না পড়ে উত্তর লেখা শুরু কোরো না, তাতে ভুল বেশি হয়।"
```

`data/raw/bangla_heteronym_words.csv` lists each word's readings; their order defines the label index (0, 1, 2):

```csv
spelling,pronunciation 1,pronunciation 2,pronunciation 3
আঁক,আঁক্‌,আঁকো,
আসর,আশোর,আসোর,
```

The split files (`data/splits/*.csv`) have the same four columns. `src.data.load_split` adds the target's character
span, its whitespace-token index, the label and the number of readings. One training row:

```python
{'spelling': 'কর', 'pronunciation': 'করো',
 'sentence': 'হাত-মুখ ধুয়ে তুমি খাওয়া শুরু কর।', 'sentence_phonetic': 'হাত-মুখ ধুয়ে তুমি খাওয়া শুরু করো।',
 'start': 31, 'end': 33, 'token_idx': 5, 'label': 1, 'n_readings': 3}
```

`label` 1 is the second reading of কর in the words file (কর্‌, **করো**, কোরো). Readings ending in `্` are stored as hasanta +
ZWNJ (U+200C); keep it when copying them.

## 4. Approaches

The target word is marked in the input as `@ word @` (the marker), so the model knows which occurrence is meant.

| `model_type` | How it predicts | Confidence |
|---|---|---|
| `encoder` | classifier: the hidden state of the opening marker token goes to a linear layer with one output per reading slot; slots the word does not have are masked out | softmax probability |
| `candidate` | candidate scoring: one (marked sentence, candidate pronunciation) pair per candidate, one score each, softmax over the word's candidates; works for words never seen in training | softmax probability |
| `seq2seq` | input `phonetic: <marked sentence>`, the model writes the whole sentence back with the target replaced by its pronunciation (greedy decoding); the generated word is mapped back to a candidate, and an output that matches no candidate counts as wrong | exp(mean token log-prob) |

All models are wrapped by `src.predict.Predictor`, so a checkpoint of any type is used the same way (section 8).

### What a model sees and produces (training, dev and test)

Training, dev evaluation, testing and inference build the input in exactly the same way (`src/models.py`). For the
training row above (`হাত-মুখ ধুয়ে তুমি খাওয়া শুরু কর।`, target কর, gold করো):

**`encoder` (BanglaBERT shown, tokens as the tokenizer prints them).** Input is the pair (word, marked sentence); the
head reads the opening `@` (token 10); the output is one score per reading slot, masked to the word's readings;
training target = the label:

```text
[CLS] কর [SEP] হাত - মুখ ধুয়ে তুমি খাওয়া শুরু @ কর @ । [SEP]
candidates: ['কর্‌', 'করো', 'কোরো']   cand_mask: [True, True, True]   label: 1   ->  prediction করো
```

**`candidate`.** One pair (marked sentence, candidate) per reading, one score each, softmax over them; training target
= the index of the gold candidate. (BanglaBERT's WordPiece tokenizer drops the ZWNJ, so `কর্‌` shows as `কর্`.)

```text
[CLS] হাত - মুখ ধুয়ে তুমি খাওয়া শুরু @ কর @ । [SEP] কর্ [SEP]
[CLS] হাত - মুখ ধুয়ে তুমি খাওয়া শুরু @ কর @ । [SEP] করো [SEP]
[CLS] হাত - মুখ ধুয়ে তুমি খাওয়া শুরু @ কর @ । [SEP] কোরো [SEP]
```

**`seq2seq` (BanglaT5).** Input and training target are text (the target is `sentence_phonetic` after the same
normalizer); at test time the generated word at the target's position is compared with the gold reading:

```text
input : phonetic: হাত-মুখ ধুয়ে তুমি খাওয়া শুরু @ কর @।
target: হাত-মুখ ধুয়ে তুমি খাওয়া শুরু করো।
```

**Test outputs.** Every run writes `predictions.csv` (one row per test sentence) and `metrics.json`. Encoder
(`runs/banglabert/main/seed42/predictions.csv`):

```csv
spelling,sentence,n_readings,gold,prediction,confidence,correct
আঁক,খাতায় দশটা আঁক শেষ না হলে খেলতে যাওয়া বন্ধ।,2,আঁক্‌,আঁক্‌,0.999884,True
আঁক,দাদু এখনও মুখে মুখে যেকোনো আঁক মিলিয়ে দেন।,2,আঁক্‌,আঁক্‌,0.999879,True
```

Seq2seq runs add the generated sentence (`raw_output`; the ZWNJ is removed inside the T5 pipeline, the prediction is
mapped back to the canonical reading):

```csv
spelling,sentence,n_readings,gold,prediction,confidence,correct,raw_output
আঁক,খাতায় দশটা আঁক শেষ না হলে খেলতে যাওয়া বন্ধ।,2,আঁক্‌,আঁক্‌,0.999992,True,খাতায় দশটা আঁক্ শেষ না হলে খেলতে যাওয়া বন্ধ।
```

Candidate scoring on the unseen split (`runs/banglabert_large/cand_unseen/seed42/predictions.csv`; কর never
occurs in its training data):

```csv
spelling,sentence,n_readings,gold,prediction,confidence,correct
কর,এ বছর থেকে মোবাইল ফোনের ওপর কর বাড়ছে।,3,কর্‌,কর্‌,0.862083,True
```

`metrics.json` (shortened):

```json
{
 "train": {"train_time_s": 248.8, "epochs_trained": 10.0, "best_epoch": 7, "steps": 1610,
           "peak_gpu_memory_train_mb": 2114, "params": 110029059, "checkpoint_size_mb": 421.3},
 "dev":   {"n": 1104, "accuracy": 0.9918, "macro_f1": 0.9916, "accuracy_by_readings": {"2": 0.9926, "3": 0.9444},
           "accuracy_by_word": {"আঁক": 1.0, "আসর": 1.0, "আসল": 0.8333, "...": "..."}},
 "test":  {"n": 1104, "accuracy": "...", "macro_f1": "...", "accuracy_by_readings": {}, "accuracy_by_word": {}}
}
```

Unseen-split runs name their test block `unseen_test` instead of `test`.

## 5. Experiments: what each one means

Every experiment is a **setting** (`SETTINGS` in `src/train.py`) trained with 3 seeds (42, 43, 44) at the model's
selected learning rate, except the learning curves, which use seed 42 only. Results are reported as mean ± std over
the seeds. Output: `runs/{model}/{setting}/seed{seed}/`.

| Setting | Question it answers | Models | Results file |
|---|---|---|---|
| `lr_search` | which learning rate is best? one seed-42 run per lr in the config's `lr_grid`, judged on **dev** only; `scripts.select_lr` writes `runs/{model}/best_lr.json` | all | `results/lr_search.csv` |
| `main` | the benchmark itself: accuracy, macro-F1, per-word accuracy on the main test set | all 12 | `results/summary.md`, `per_word.csv` |
| `unseen` | can a model handle a heteronym it never saw in training? (only seq2seq and candidate models can, because a classifier has no slots for unknown words) | seq2seq models | `results/summary.md` |
| `cand_main`, `cand_unseen` | candidate scoring on the main and unseen splits; is it as accurate as the classifier, and does it generalize to unseen words? | 3 best encoders | `results/summary.md`, `latency.csv` |
| `lc2`, `lc5`, `lc10` | **learning curve**: how many labelled sentences per pronunciation are needed? Seed 42 only | all 12 | `results/learning_curve.csv`, `plots/learning_curves.png` |
| `abl_marker_special` | marker `<h> word </h>` (new tokens) instead of `@ word @` | BanglaBERT-large | `results/ablations.md` |
| `abl_marker_none` | no marker at all (the head reads the target word's first token) | BanglaBERT-large | `results/ablations.md` |
| `abl_head_mean`, `abl_head_cls` | pool the mean of the target's tokens, or `[CLS]`, instead of the opening marker token | BanglaBERT-large | `results/ablations.md` |
| `abl_output_target`, `abl_output_target_keep` | seq2seq writes only the target's pronunciation instead of the whole sentence (`_keep` is one extra seed whose checkpoint is kept to measure its latency, `runs/banglat5/latency_target.json`) | BanglaT5 | `results/ablations.md`, `latency.csv` |

How to read the learning curve: plot test accuracy against 2 / 5 / 10 / 14 sentences per pronunciation. Where the
curve flattens (the difference to the next point is within the seed spread of the main runs, about 0.1 to 0.4 points
for most models), more data stops helping; that point is the data budget per pronunciation when adding new
heteronyms. A curve still rising at 14 means more data would likely help.

Why only seed 42 for the learning curves (owner, 2026-10-20): the drops from 14 to 2 sentences are 5 to 15 points,
far larger than the spread between seeds (about 0.6 to 1.3 points at lc2, measured with 3 seeds on 4 models before
the change), so one seed shows the shape, and it saves about 10 h of GPU time. The point at 14 is the seed-42 main
run, so the whole curve uses the same seed (`LC_SEED` in `scripts/aggregate.py`). The extra lc2 seeds were deleted.

Why some experiments use only a few models: ablations explain *why* the best setup works, so one strong model is
enough (the owner decided on 2026-10-20 not to extend them; the no-marker gain is reported as one-model evidence); candidate scoring and the unseen split concern the models one would deploy (or the only ones that can handle
unseen words). Running every setting for all 12 models would multiply the GPU time without changing the conclusions.

ByT5-small was planned as a 13th model and dropped (2026-10-20): one run took ~7 h on the T4, and its remaining runs
would have taken ~50 h. See EXPERIMENT_LOG.md.

## 6. Run experiments

Full cycle for one model (lr search, 3 seeds, keep best checkpoint, latency, demo, requirements, aggregation),
best inside tmux; logs go to `logs/{model}.log`:

```bash
bash scripts/run_model.sh banglabert_small            # or: ... banglabert_small train | measure
bash scripts/run_setting.sh cand_main banglabert      # one setting, seeds 42-44, for the listed models
SEEDS=42 bash scripts/run_setting.sh lc2 banglabert   # learning curves: seed 42 only
```

`train` runs the lr search, the 3 seeds and keeps the best checkpoint; `measure` runs the latency benchmark,
the phoneticize demo, the requirements file and the aggregation (the default runs both). `run_setting.sh` needs the
model's `main` cycle first (it uses `best_lr.json`); for `cand_main` it also keeps the best seed in
`checkpoints/{model}_cand`.

Single steps:

```bash
python -m src.train --config configs/banglabert.yaml --lr-search 3e-5      # one lr-search run (seed 42, dev only)
python -m scripts.select_lr banglabert                                     # -> runs/banglabert/best_lr.json
python -m src.train --config configs/banglabert.yaml --seed 43             # one seed with the best lr
python -m src.train --config configs/banglabert.yaml --seed 42 --setting cand_main   # any setting from section 5
python -m scripts.keep_best banglabert main checkpoints/banglabert         # keep best seed's checkpoint
python -m src.evaluate --checkpoint checkpoints/banglabert --data data/real_test.csv --out runs/banglabert/real_test.json
python -m src.benchmark_latency --checkpoint checkpoints/banglabert --out runs/banglabert/latency.json
python -m scripts.aggregate                                                # results/ tables and plots
bash scripts/smoke.sh                                                      # 20-step smoke test of every model
```

Resuming: every run whose `runs/{model}/{setting}/seed{seed}/metrics.json` exists is skipped, so re-running the same
command continues where it stopped. Training curves: `tensorboard --logdir runs`.

Training recipe (same for every model, set in `configs/{model}.yaml`): AdamW, weight decay 0.01, 10% linear warm-up
then linear decay, gradient clipping 1.0, max length 128, at most 10 epochs (encoders) or 15 (seq2seq), early
stopping with patience 3 on dev accuracy, the best epoch's weights are kept.

## 7. What is measured

Per run, `runs/{model}/{setting}/seed{seed}/`:

| File | Content |
|---|---|
| `config.json` | the full config actually used, package versions, GPU, git commit, and any automatic `adjustments` |
| `metrics.json` | `train` (time, epochs, best epoch, steps, peak GPU memory, params, checkpoint size) and `dev` / `test` (accuracy, macro-F1, accuracy by word and by number of readings; seq2seq also `other_word_change_rate`, the share of outputs that change a word other than the target, and `avg_generated_tokens`) |
| `predictions.csv` | every test sentence with gold, prediction and confidence |
| `train_log.csv`, `tb/` | loss, learning rate, grad norm and dev accuracy per step / epoch (`tb/` is not committed) |

**Latency** (`runs/{model}/latency.json`, `latency_cand.json`, `latency_target.json`; table `results/latency.csv`)
is the time to answer one request through the full path: normalizer, tokenizer, model, decoding and mapping back
to a pronunciation. For seq2seq it is therefore mostly output generation time. Measured on the T4 GPU and on CPU
with 4 threads and 1 thread: batch size 1 (20 warm-up calls, 200 timed calls; mean, **p50**, p95, p99), forward
pass only, throughput at batch 8 and 32, the phoneticize demo paragraph, and peak GPU memory. Tables report p50 at
batch size 1.

## 8. Use a trained model

```bash
python -m src.predict --checkpoint checkpoints/banglabert --text "তুমি একটু অপেক্ষা কর, আমি আসছি। বিকেলে তুমি পার্কে খেল।"
```

prints the phonetic text, one line per heteronym (word, character position, prediction, confidence) and the time:

```text
তুমি একটু অপেক্ষা করো, আমি আসছি। বিকেলে তুমি পার্কে খ্যালো।

word	position	prediction	confidence
কর	18	করো	0.9464
খেল	51	খ্যালো	0.9790

2 heteronyms, 34.4 ms on cpu (checkpoints/banglabert)
```

Add `--device cpu` or `--device cuda` to choose the device, or `--file text.txt` instead of `--text`. Any kept
checkpoint works the same way; `checkpoints/banglat5` gives the same text here (confidences 0.9997 and 1.0000) in
284 ms on the CPU.

```python
from src.predict import Predictor

predictor = Predictor.load("checkpoints/banglabert")      # any kept checkpoint folder: encoder, candidate or seq2seq
text = "তুমি একটু অপেক্ষা কর, আমি আসছি। কর কাজটা, তারপর আবার কর!\nবিকেলে তুমি পার্কে খেল।"
print(predictor.phoneticize(text))                        # every heteronym replaced, text otherwise unchanged
# তুমি একটু অপেক্ষা করো, আমি আসছি। কোরো কাজটা, তারপর আবার করো!
# বিকেলে তুমি পার্কে খ্যালো।
# (the second কর is read as the "later" imperative কোরো; করো would be more natural here, one of the
#  hard same-slot cases described in the paper)
print(predictor.predict("তুমি গ্রাফ পেপারে সরলরেখাটা আঁক।", "আঁক"))   # আঁকো
print(predictor.predict("সময়মতো কর না দিলে জরিমানা গুনতে হবে।", "কর"))  # কর্‌

phonetic, details = predictor.phoneticize_detailed("বিকেলে মাঠে বল নিয়ে খেলব, কাল আবার দেখা হবে।")
# phonetic: বিকেলে মাঠে বল্‌ নিয়ে খেলব, কাল্‌ আবার দেখা হবে।
# details:  [{'word': 'বল', 'position': 12, 'prediction': 'বল্‌', 'confidence': 0.9997},
#            {'word': 'কাল', 'position': 27, 'prediction': 'কাল্‌', 'confidence': 0.9995}]
```

A longer input, the 8-heteronym paragraph used for the latency test (`data/latency_paragraph.txt`), with
`checkpoints/banglabert_small` (all 8 correct):

```text
in : বিকেলে তুমি পার্কে খেল, সন্ধ্যায় পড়তে বসবে। রাস্তা থেকে আনা বিড়ালছানাটা প্রথমে খুব আঁচড়াত, এখন একদম পোষ মেনে আমার কোলেই ঘুমায়। তুমি আমার জন্য একটু অপেক্ষা কর, আমি আসছি। চারদিকে রটে গেছে, বাঘ এসেছে।
     খালি পেটে ওষুধ খেলে গ্যাস হয়। চেয়ারে বসে ঘুমিয়ে পড়লেন দাদু। তুমি গ্রাফ পেপারে সরলরেখাটা আঁক। প্রেমিকের গিটার শুনে মেয়েটা পটল।
out: বিকেলে তুমি পার্কে খ্যালো, সন্ধ্যায় পড়তে বসবে। রাস্তা থেকে আনা বিড়ালছানাটা প্রথমে খুব আঁচড়াত, এখন একদম পোশ্‌ মেনে আমার কোলেই ঘুমায়। তুমি আমার জন্য একটু অপেক্ষা করো, আমি আসছি। চারদিকে রোটে গেছে, বাঘ এসেছে।
     খালি পেটে ওষুধ খেলে গ্যাস হয়। চেয়ারে বোশে ঘুমিয়ে পড়লেন দাদু। তুমি গ্রাফ পেপারে সরলরেখাটা আঁকো। প্রেমিকের গিটার শুনে মেয়েটা পট্‌লো।
```

Words that are not in the heteronym list (here আঁচড়াত, খেলব) are never changed.

### Using it from another project

There is no package to install: put this repo on the Python path and load a kept checkpoint once (loading takes a few
seconds; keep the `Predictor` and reuse it).

```python
import sys
sys.path.insert(0, "/path/to/bangla-heteronym-benchmark")
from src.predict import Predictor

p = Predictor.load("/path/to/bangla-heteronym-benchmark/checkpoints/banglabert")   # device="cpu" or "cuda" (default: cuda if available)
```

Three ways to call it:

| You have | Call | You get |
|---|---|---|
| any text, nothing marked | `p.phoneticize(text)` | the same text with **every** known heteronym replaced by its reading |
| a sentence and the word, the word occurs once | `p.predict(sentence, "কর")` | only the reading of that word (its first occurrence) |
| a sentence with the target marked | `p.predict_marked("... [কর] ...")` | only the reading of the marked occurrence |

```python
p.phoneticize("আগে কর কাজটা, তারপর আবার কর। বিকেলে মাঠে বল নিয়ে যাব।")
# 'আগে কোরো কাজটা, তারপর আবার করো। বিকেলে মাঠে বল্‌ নিয়ে যাব।'

p.predict("সময়মতো কর না দিলে জরিমানা গুনতে হবে।", "কর")             # 'কর্‌'
p.predict_marked("সরকার কর বাড়িয়েছে, তাই হিসাবটা আবার [কর]।")       # 'করো'  (the second কর; predict() would read the first)
p.predict_marked("তুমি <<বল>> নিয়ে এসো।", marks=("<<", ">>"))        # other marks if [ ] can occur in your text
```

Input rules:

- Pass normal Bangla text (Unicode, as typed). No marking is needed for `phoneticize` and `predict`.
- Mark only when a word occurs more than once and you want a specific occurrence: put the mark directly around the
  word, without spaces (`[কর]`, not `[ কর ]`), and mark one word per call.
- The marked word must be one of the 183 heteronyms (`data/raw/bangla_heteronym_words.csv`); otherwise a
  `ValueError` is raised. `phoneticize` simply leaves other words unchanged.
- You can pass a whole paragraph: sentences are split at `।`, `?`, `!` and newlines, and each heteronym is read from
  its own sentence (the model was trained on single sentences).
- Outputs keep the ZWNJ of readings such as `কর্‌` (hasanta + U+200C); keep it if the text goes to a TTS front end.
- `p.phoneticize_detailed(text)` also returns, per heteronym, its character position, reading and confidence, and
  `p.predict_items([(sentence, start, end, word)])` takes character offsets if you locate the words yourself.

Which checkpoint: `checkpoints/banglabert` is the most accurate classifier (99.46%, ~12.5 ms on the T4, ~33 ms on 4 CPU
threads); `checkpoints/banglabert_small` is the fastest on a CPU (98.16%, 13 ms); `checkpoints/banglabert_large_cand`
(candidate scoring) is the most accurate (99.73%) but about twice as slow. All are used the same way.

`phoneticize` splits the text on `।`, `?`, `!` and newlines, finds every whole-word occurrence of a known
heteronym, marks one occurrence at a time, predicts all of them in one batch and rebuilds the exact text.

Kept checkpoints (not committed, `checkpoints/` is git-ignored): `checkpoints/{model}` is the best `main` seed of
each model, `checkpoints/{model}_cand` the best `cand_main` seed. All other runs' checkpoints are deleted after
evaluation (`checkpoints/_runs/` holds them while a run is in progress).

## 9. Results

Full tables: [results/summary.md](results/summary.md), [results/latency.csv](results/latency.csv),
[results/ablations.md](results/ablations.md), [results/learning_curve.csv](results/learning_curve.csv),
plots in [results/plots/](results/plots/), and [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) for observations.

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

The table between the `results-table` markers (here and in EXPERIMENT_LOG.md) is rewritten by `scripts.aggregate`;
don't edit it by hand.

### Further experiments

Accuracy in %, mean ± std over seeds 42/43/44; GPU p50 in ms (Tesla T4, batch 1, full path).

**Candidate scoring and unseen words** (`cand_main`, `cand_unseen`, `unseen`). The unseen test set has 1,500
sentences of 37 words that never occur in training or dev.

| model | approach | main test | unseen test | GPU p50 |
|---|---|---|---|---|
| banglabert_large | classifier | 99.18 ± 0.16 | n/a | 22.8 |
| banglabert_large | candidate | 99.73 ± 0.09 | 83.24 ± 2.55 | 23.8 |
| banglabert | classifier | 99.46 ± 0.16 | n/a | 12.5 |
| banglabert | candidate | 99.37 ± 0.16 | 81.02 ± 1.22 | 13.6 |
| xlm_roberta_large | classifier | 98.43 ± 0.19 | n/a | 17.8 |
| xlm_roberta_large | candidate | 83.30 ± 23.82 | 62.09 ± 21.18 | 19.9 |
| banglat5 | seq2seq | 98.37 ± 0.18 | 55.09 ± 1.85 | 236.8 |
| mt5_small | seq2seq | 93.36 ± 0.21 | 51.69 ± 0.63 | 307.6 |
| mt5_base | seq2seq | 96.11 ± 0.16 | 58.07 ± 0.88 | 436.2 |

- XLM-R-large candidate runs collapsed in 3 of 6 runs (main seed 44: 55.8%; unseen seeds 42 and 43: about 50%):
  the training loss stayed at the two-candidate chance level until early stopping. They are kept in the mean
  (owner's decision, 2026-10-20); the converged seeds reach 97.0/97.1% (main) and 86.5% (unseen).
- On unseen words the seq2seq models write one of the word's readings in only 68-74% of the sentences (otherwise
  an invented pronunciation); when they do, it is the right one 76-81% of the time. The candidate scorer always
  picks one of the readings, which is why it generalizes much better.

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

Going from 2 to 5 sentences per pronunciation gives the largest gain for every model except BanglaBERT-small
(which gains more from 5 to 10). At 10, every encoder except MuRIL is within 1.5 points of its result at 14, and
three models (BanglaT5, XLM-R-base, mDeBERTa) are slightly higher at 10 than at 14, which is within single-seed
noise. mT5-small needs the most data (52.7% at 2).

**Ablations** (BanglaBERT-large and BanglaT5, 3 seeds; [results/ablations.md](results/ablations.md)): no marker
99.67 ± 0.10 and mean pooling of the target tokens 99.52 ± 0.14 beat the default `@ word @` marker with the
opening-marker head (99.18 ± 0.16); `<h>` tokens 99.34 ± 0.14; `[CLS]` 99.09 ± 0.31. BanglaT5 writing only the
target's pronunciation: 98.28 ± 0.47 (full sentence 98.37 ± 0.18) at 86.1 ms GPU p50 instead of 236.8 ms.

## 10. Things to know (pitfalls found while building this)

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
- **Normalizer.** Inputs go through the csebuetnlp normalizer (pinned git commit in `requirements.txt`) before
  tokenization, as the BanglaBERT/BanglaT5 authors recommend.
- **Automatic fallbacks** (`src/train.py`, recorded in `config.json` under `adjustments` and in EXPERIMENT_LOG):
  on CUDA OOM the batch is halved and gradient accumulation doubled (same effective batch), then gradient
  checkpointing; on a non-finite fp16 loss the run restarts in fp32. mT5, mDeBERTa and BanglaT5 are set to fp32 in their
  configs, because T5 and DeBERTa-v3 models are known to be unstable in fp16.
- **Seq2seq memory at evaluation.** Generation scores are collected with a small logits processor
  (`GreedyLogProbs`) instead of `output_scores`, which kept the whole vocabulary distribution per step and ran out of
  memory for mT5.
- **Shared GPU (original server).** Two idle Triton inference services also live on the T4. They are stopped while
  the largest models train (to free memory) and are running, idle, during every latency measurement, so all latency
  numbers share the same conditions.
- **mT5 epoch limit.** Both mT5 models were still improving at the 15-epoch limit; their numbers are a lower bound
  under this shared recipe.
- **Reproducibility.** Splits, configs and seeds are fixed; `config.json` records package versions and the git commit
  of every run. Results are reproducible up to GPU non-determinism (expect differences within the seed std).

## 11. Adding a model or data

- **New model:** copy a config in `configs/` with the same `model_type`, set `model_name`, `model_id`, `precision`,
  `batch_size` and `lr_grid`; run `bash scripts/smoke.sh <model>` first, then `bash scripts/run_model.sh <model>`, and
  add it to `MODELS` in `scripts/aggregate.py` (table order). `scripts.write_requirements` writes its requirements file.
- **New heteronyms or sentences:** the dataset is pinned to one commit in `src/data.py`. A new dataset version needs
  new splits (a new split seed or folder, never overwriting the old ones) and re-running every model, otherwise the
  numbers are not comparable. The learning curve tells roughly how many sentences per pronunciation to write.

## 12. Repo layout

```
configs/        one YAML per model
requirements/   pinned packages per model
src/            data, splits, models, train, evaluate, predict, benchmark_latency, env
scripts/        run_model.sh, run_setting.sh, smoke.sh, env.sh, select_lr, keep_best, aggregate,
                write_requirements, check_tokenizers
tests/          phoneticize and data tests
data/splits/    the fixed splits;  data/latency_*: latency inputs
runs/           per-run config/metrics/predictions/train logs, latency*.json, phoneticize demos
checkpoints/    kept best models (not committed)
logs/           training logs (not committed)
results/        aggregated tables and plots
```

## 13. Prepared by

**Kawshik Kumar Paul**  
Senior Software Engineer | Researcher  
Department of Computer Science and Engineering (CSE)  
Bangladesh University of Engineering and Technology (BUET)  
Email: kawshikbuet17@gmail.com  
