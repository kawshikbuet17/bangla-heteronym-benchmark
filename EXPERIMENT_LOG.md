# Experiment log: Bangla heteronym disambiguation benchmark

## Current results

Test = main split test set (3 sentences per pronunciation, 1,104 sentences), mean ± std over seeds 42/43/44.
Latency = batch size 1, full path (normalizer + tokenizer + model + decoding), p50.

<!-- results-table -->
| # | model | size | precision | best lr | test acc (%) | macro-F1 (%) | train time / seed (min) | GPU p50 (ms) | CPU p50 (ms, 4 thr) | checkpoint | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | banglabert_small | 13.7M | fp16 | 0.0001 | 98.16 ± 0.14 | 98.09 ± 0.13 | 2.5 | 12.6 | 13.0 | 54 MB | done |
| 2 | banglabert | ~110M |  |  |  |  |  |  |  |  | pending |
| 3 | bangla_bert_base | ~165M |  |  |  |  |  |  |  |  | pending |
| 4 | muril_base | ~237M |  |  |  |  |  |  |  |  | pending |
| 5 | banglat5 | ~250M |  |  |  |  |  |  |  |  | pending |
| 6 | xlm_roberta_base | ~278M |  |  |  |  |  |  |  |  | pending |
| 7 | indicbertv2_mlm | ~278M |  |  |  |  |  |  |  |  | pending |
| 8 | mdeberta_v3_base | ~280M |  |  |  |  |  |  |  |  | pending |
| 9 | mt5_small | ~300M |  |  |  |  |  |  |  |  | pending |
| 10 | byt5_small | ~300M |  |  |  |  |  |  |  |  | pending |
| 11 | banglabert_large | ~335M |  |  |  |  |  |  |  |  | pending |
| 12 | xlm_roberta_large | ~560M |  |  |  |  |  |  |  |  | pending |
| 13 | mt5_base | ~580M |  |  |  |  |  |  |  |  | pending |

Disk used by `checkpoints/`: 54M.
<!-- /results-table -->

## Observations for the paper

- Tokenizers and the ZWNJ (`results/tokenizer_zwnj.csv`): only ByT5 round-trips the ZWNJ. WordPiece tokenizers
  (BanglaBERT small/base/large, bangla-bert-base, MuRIL) drop it before tokenization, so those encoders never see
  it (token ids identical with and without it). The sentencepiece models (XLM-R, mDeBERTa, mT5, BanglaT5) and
  IndicBERTv2 produce different ids with a ZWNJ but cannot decode it back; T5 tokenizers decode it as a space.
- `sagorsarker/bangla-bert-base` ships without a `tokenizer_config.json`, so the default lowercasing strips Bangla
  vowel signs (তুমি -> তমি, হয়েছিল -> হযেছিল) although the vocab contains them. We load it with
  `do_lower_case=False, strip_accents=False`. Anyone using the hub default gets a crippled tokenizer.

## Entries (newest first)

### 2026-10-18 17:28: banglabert_small (model 1), full cycle

**Run.** `bash scripts/run_model.sh banglabert_small` (config `configs/banglabert_small.yaml`: encoder classifier,
`@` marker head, normalizer on, fp16, 32 × 1, AdamW wd 0.01, 10% warmup + linear decay, clip 1.0, max len 128,
≤10 epochs, patience 3). 17:12 -> 17:27 (15 min for the whole cycle).

**lr search (seed 42, dev accuracy).** 3e-5: 97.28 · 5e-5: 98.19 · **1e-4: 98.55** (selected). The best lr is the
top of the planned grid {3e-5, 5e-5, 1e-4}; a larger lr might be slightly better for this 13.7M model, but the
grid is kept as planned for comparability.

**Result (lr 1e-4, seeds 42/43/44).** Test accuracy **98.16 ± 0.14** (98.28 / 98.01 / 98.19), macro-F1
98.09 ± 0.13; 2-reading words 98.4%, 3-reading words (কর, খেল) 83.3%. Dev 98.55 / 98.28 / 98.37; seed 42 kept
(best dev). Each seed ran all 10 epochs (best epoch 8-9, no early stop), 2.5 min per seed, peak training memory
0.3 GB. Checkpoint 53.7 MB (fp32 weights).

**Latency (kept checkpoint, 200 sentences, 20 warm-up).**

| device | precision | p50 | p95 | p99 | forward p50 | throughput | paragraph (8 heteronyms) |
|---|---|---|---|---|---|---|---|
| T4 GPU | fp16 | 12.6 ms | 15.3 ms | 15.7 ms | 11.2 ms | 569 /s (b8), 1488 /s (b32) | 14.5 ms (1.8 ms/heteronym) |
| CPU, 4 threads | fp32 | 13.0 ms | 15.6 ms | 16.9 ms | 11.4 ms | 259 /s (b8) | 34.5 ms (4.3 ms/heteronym) |
| CPU, 1 thread | fp32 | 17.4 ms | 19.4 ms | 20.5 ms | 16.5 ms | - | 80.1 ms (10.0 ms/heteronym) |

Peak GPU memory at inference: 40 MB.

**phoneticize demo** (`runs/banglabert_small/phoneticize_demo.txt`): all 8 heteronyms of the paragraph correct
(খেল->খ্যালো, পোষ->পোশ্‌, কর->করো, রটে->রোটে, খেলে->খেলে, বসে->বোশে, আঁক->আঁকো, পটল->পট্‌লো), ZWNJ kept; lowest
confidence on the 3-reading কর (0.66).

**Observations.** At batch 1, the GPU is barely faster than 4 CPU threads for this small model: the time goes into
Python/HF overhead and launching ~hundreds of small kernels from a 2.3 GHz Xeon, not into compute (forward 11.2 ms
GPU vs 11.4 ms CPU). The GPU advantage shows up only with batching (1488 vs 259 sentences/s) and in
`phoneticize`, which batches all heteronyms of a text. Hardest words: আসর and কর (66.7%), কাল and সিম (77.8%).

**Problems.** None (no NaN, no OOM).

**Next.** Wait for review; then banglabert (model 2).

### 2026-10-18 17:15: Stage 2, smoke test

**Run.** `bash scripts/smoke.sh <all 13 models> banglabert_small:{cand_main,abl_marker_special,abl_marker_none,abl_head_mean,abl_head_cls} mt5_small:abl_output_target`:
20 optimizer steps on a tiny train subset (20 × effective batch), 64 dev sentences, then checkpoint save,
reload from the folder and evaluation. Logs in `logs/smoke_*.log`, results in `logs/smoke/` (not committed).

**Result.** All 13 models and all 6 extra settings (candidate scoring, special-token / no marker, mean / [CLS]
head, target-only seq2seq output) run end to end. No NaN in any fp16 run. Peak training memory (allocated /
reserved) and time for 20 steps:

| model | params | peak alloc | reserved | 20 steps |
|---|---|---|---|---|
| banglabert_small | 13.7M | 0.3 GB | 0.3 GB | 3.0 s |
| banglabert | 110.0M | 2.1 GB | 2.4 GB | 3.6 s |
| bangla_bert_base | 164.4M | 3.2 GB | 3.5 GB | 4.2 s |
| muril_base | 237.6M | 4.6 GB | 4.8 GB | 5.1 s |
| banglat5 | 247.6M | 4.8 GB | 5.6 GB | 14.2 s |
| xlm_roberta_base | 278.0M | 5.3 GB | 5.6 GB | 5.3 s |
| indicbertv2_mlm | 278.0M | 5.4 GB | 5.5 GB | 5.2 s |
| mdeberta_v3_base | 278.2M | 5.9 GB | 6.7 GB | 11.8 s |
| mt5_small | 300.2M | 6.4 GB | 10.1 GB | 10.7 s |
| byt5_small (4×4) | 299.6M | 7.6 GB | 9.2 GB | 86.4 s |
| banglabert_large | 335.6M | 6.4 GB | 6.8 GB | 11.5 s |
| xlm_roberta_large | 559.9M | 10.7 GB | 11.0 GB | 17.8 s |
| mt5_base | 582.4M | 11.2 GB | 12.6 GB | 21.3 s |

**Problems and fixes.**
- byt5_small OOM at 8 × 2 with ~10 GB free; the automatic fallback ran it at 4 × 4 (same effective batch 16).
  Config changed to 4 × 4.
- xlm_roberta_large and mt5_base OOM even at batch 1 + gradient checkpointing: AdamW with fp32 master weights
  needs ~9 GB for 560-580M parameters before activations, and only ~10.1 GB of the T4 is free (two triton
  docker containers, `fast_whisper_triton_test` and `dialect_translation_triton`, hold ~4.5 GB). With the
  owner's permission the two containers were stopped from 17:08:55 to 17:10:53 (restarted with `docker start`,
  both healthy again at ~17:12): both models then run at the planned 8 × 4 / 8 × 2 settings. Plan: stop the
  containers again only for these two models' runs, restart right after, log the downtime.
- mt5_small reserves 10.1 GB for 6.4 GB allocated (fragmentation); `scripts/env.sh` now sets
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

**Cost notes.** ByT5 is slow: byte-level sequences (up to ~300 bytes in, ~290 out) make one epoch ~23 min plus
dev generation; worst case (15 epochs) is many hours per run. Will estimate before running it.

**Next.** Stage 3: banglabert_small full cycle (estimated ~25 min).

### 2026-10-18 16:30: Stage 1, setup

**Environment.** Shared server: GPU 0 is an RTX 4090 used by vLLM, GPU 1 is the Tesla T4 (driver 560.35.05,
CUDA 12.6). The T4 is pinned with `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1` (`scripts/env.sh`).
tritonserver processes already hold ~4.5 GB of the T4, so about **10.8 GB** of its 15 GB is free for us, and
latency numbers may see some interference from those processes (idle during setup). CPU: Intel Xeon E5-2650 v3
@ 2.30GHz (40 logical cores). Conda env `hetero` (Python 3.11.17), PyTorch 2.7.1+cu126, transformers 4.57.6,
normalizer @ d405944.

**Data.** `kawshikbuet17/bangla-heteronyms` at commit `2c3d86b67571e15ba3520d3d7b11930fb8f34e3a`.
Checks passed (`python -m src.splits check`): 7,360 rows, 183 words, 368 pronunciations, 20 sentences each,
125 pronunciations contain a ZWNJ, every target occurs exactly once as a whole word, `sentence_phonetic` equals
the sentence with the target replaced, splits are disjoint and cover the dataset.

**Splits (seed 42, `data/splits/`).** train 5,152 / dev 1,104 / test 1,104 (14/3/3 per pronunciation).
Unseen-word split: 37 of 183 words held out (20.2%, stratified: 36 of 181 two-reading words + 1 of the 2
three-reading words, কর), unseen_test = all 1,500 sentences of those words, unseen_train/dev = main train/dev of
the other 146 words (4,102 / 879). Learning-curve subsets 2/5/10 per pronunciation (nested; 14 = full train).
Latency inputs: 200 test sentences (`data/latency_sentences.csv`) and an 8-heteronym, 2-line paragraph of test
sentences (`data/latency_paragraph.txt`, also used for the phoneticize demo).

**Changes from the plan and why.**
- Word boundaries also include `“ ” ‘ ’ — –`: the dataset has curly quotes and em dashes next to targets.
- Unseen-word split holds out max(1, round(20%)) words per stratum so the unseen test has a 3-reading word.
- Seq2seq tokenizers that cannot round-trip the ZWNJ (BanglaT5, mT5) get the ZWNJ removed from their inputs and
  targets (model side only; data files untouched). Without this the T5 tokenizers turn a mid-word ZWNJ into a
  space (আশ্‌লো -> "আশ্ লো"), which breaks the word alignment. Predictions and references are compared after the
  same normalizer + tokenizer round trip; a predicted target is mapped back to its candidate, which restores the
  ZWNJ in the output. Check: scoring the references as predictions gives 100% accuracy and 0 other-word changes
  for BanglaT5, mT5 and ByT5, and no word's candidates collide after canonicalization.
- Seq2seq target extraction uses the punctuation-split piece of the target token (e.g. সাদা-**কালো**).
- The seed-42 main run reuses the lr-search run with the best lr (identical config) instead of retraining it.
- lr-search runs are evaluated on dev only; test is evaluated once per final run, with the best checkpoint.
- The best seed (whose checkpoint is kept) is chosen by dev accuracy, not test accuracy.
- MuRIL uses a WordPiece vocab (no sentencepiece model), so `requirements/muril_base.txt` will not list
  sentencepiece; sentencepiece + protobuf are listed for models whose hub repo ships a `.model` file.
- Training writes only the best weights (kept in memory during training, written once at the end): no
  per-epoch checkpoints on disk.
- OOM and fp16-NaN fallbacks are automatic in `src/train.py` (halve batch / double accumulation, then gradient
  checkpointing; fp16 NaN -> fp32) and recorded in each run's `config.json` under `adjustments`.

**Tests.** `python -m pytest tests`: 10 passed, 1 skipped (real-checkpoint test, needs a trained model).

**TensorBoard.** `tensorboard --logdir runs --bind_all --port 6006` is running: http://119.148.4.20:6006/

**Next.** Stage 2: smoke test of all 13 models on the T4.
