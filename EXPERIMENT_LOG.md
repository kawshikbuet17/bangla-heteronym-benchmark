# Experiment log: Bangla heteronym disambiguation benchmark

## Current results

Test = main split test set (3 sentences per pronunciation, 1,104 sentences), mean ± std over seeds 42/43/44.
Latency = batch size 1, full path (normalizer + tokenizer + model + decoding), p50.

<!-- results-table -->
| # | model | size | precision | best lr | test acc (%) | macro-F1 (%) | train time / seed (min) | GPU p50 (ms) | CPU p50 (ms, 4 thr) | checkpoint | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | banglabert_small | 13.7M | fp16 | 0.0001 | 98.16 ± 0.14 | 98.09 ± 0.13 | 2.5 | 12.6 | 13.0 | 54 MB | done |
| 2 | banglabert | 110.0M | fp16 | 2e-05 | 99.46 ± 0.16 | 99.40 ± 0.16 | 3.7 | 12.5 | 32.8 | 421 MB | done |
| 3 | bangla_bert_base | 164.4M | fp16 | 2e-05 | 98.01 ± 0.39 | 97.93 ± 0.41 | 4.4 | 10.6 | 30.5 | 633 MB | done |
| 4 | muril_base | 237.6M | fp16 | 5e-05 | 95.89 ± 1.04 | 95.54 ± 1.13 | 5.7 | 10.3 | 39.8 | 915 MB | done |
| 5 | banglat5 | 247.6M | fp32 | 0.0005 | 98.37 ± 0.18 | 98.37 ± 0.12 | 31.4 | 235.6 | 317.4 | 948 MB | done |
| 6 | xlm_roberta_base | 278.0M | fp16 | 2e-05 | 97.37 ± 0.39 | 97.27 ± 0.36 | 6.5 | 10.3 | 37.5 | 1082 MB | done |
| 7 | indicbertv2_mlm | ~278M |  |  |  |  |  |  |  |  | pending |
| 8 | mdeberta_v3_base | ~280M |  |  |  |  |  |  |  |  | pending |
| 9 | mt5_small | ~300M |  |  |  |  |  |  |  |  | pending |
| 10 | byt5_small | ~300M |  |  |  |  |  |  |  |  | pending |
| 11 | banglabert_large | ~335M |  |  |  |  |  |  |  |  | pending |
| 12 | xlm_roberta_large | ~560M |  |  |  |  |  |  |  |  | pending |
| 13 | mt5_base | ~580M |  |  |  |  |  |  |  |  | pending |

Disk used by `checkpoints/`: 4.0G.
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

### 2026-10-19 03:02: xlm_roberta_base (model 6), full cycle

**Run.** Queue, `bash scripts/run_model.sh xlm_roberta_base` (encoder classifier, no normalizer, fp16, 32 × 1).
02:17 -> 03:02 (44 min).

**lr search (dev).** 1e-5: 97.64 · **2e-5: 98.01** · 3e-5: 97.74 · 5e-5: 97.46.

**Result (lr 2e-5).** Test accuracy **97.37 ± 0.39** (97.55 / 96.92 / 97.64), macro-F1 97.27 ± 0.36;
2-reading 97.5%, 3-reading 90.7%. Dev 98.01 / 97.37 / 97.28; seed 42 kept. Best epochs 8 / 6 / 8, 6.5 min per
seed, peak training memory 5.3 GB, checkpoint 1.08 GB (250k-token vocab: 192M of the 278M parameters are
embeddings).

**Latency.**

| device | precision | p50 | p95 | p99 | forward p50 | throughput | paragraph (8 heteronyms) |
|---|---|---|---|---|---|---|---|
| T4 GPU | fp16 | 10.3 ms | 11.9 ms | 14.6 ms | 9.5 ms | 699 /s (b8), 1845 /s (b32) | 11.7 ms (1.5 ms/heteronym) |
| CPU, 4 threads | fp32 | 37.5 ms | 53.9 ms | 65.6 ms | 36.1 ms | 38 /s (b8) | 278.2 ms (34.8 ms/heteronym) |
| CPU, 1 thread | fp32 | 95.9 ms | 125.2 ms | 140.7 ms | 97.2 ms | - | 912.4 ms (114.1 ms/heteronym) |

Peak GPU memory at inference: 562 MB.

**phoneticize demo.** All 8 correct (কর 0.94, others ≥ 0.996).

**Observations.** Multilingual XLM-R base sits between MuRIL (95.89) and bangla-bert-base (98.01), 2.1 points
below BanglaBERT: so far every Bangla-only model beats every multilingual one of similar size. Most of its
parameters are the 250k-token embedding table, so its compute per token (and GPU latency) equals that of the
other 12-layer base models. Hardest: আসলে (50.0%), চরে / দেব (72.2%).

**Problems.** None.

**Next.** indicbertv2_mlm (started 03:02).

### 2026-10-19 02:18: banglat5 (model 5), full cycle

**Run.** Queue, `bash scripts/run_model.sh banglat5` (seq2seq, input `phonetic: <sentence with @ target @>`,
output the full phonetic sentence, greedy decoding, normalizer on, ZWNJ removed model-side (`strip_zwnj`), fp32,
16 × 1, ≤15 epochs, patience 3 on dev target accuracy). 23:03 -> 02:17 (3 h 14 min).

**lr search (dev).** 1e-4: 97.92 · 3e-4: 98.55 · **5e-4: 98.91** (top of the planned seq2seq grid).

**Result (lr 5e-4).** Test target accuracy **98.37 ± 0.18** (98.55 / 98.19 / 98.37), macro-F1 98.37 ± 0.12;
2-reading 98.5%, 3-reading 88.9%. Other-word change rate 0.57% on average (0.09 / 1.00 / 0.63%): almost never
alters words other than the target. Dev 98.91 / 98.28 / 98.55; seed 42 kept. Best epochs 11 / 7 / 7 (stopped
after 14 / 10 / 10), 28-39 min per seed, peak training memory 4.7 GB, checkpoint 948 MB.

**Latency.** Greedy decoding of the whole sentence, 10.5 generated tokens on average.

| device | precision | p50 | p95 | p99 | forward (generate) p50 | throughput | paragraph (8 heteronyms) |
|---|---|---|---|---|---|---|---|
| T4 GPU | fp32 | 235.6 ms | 415.7 ms | 493.4 ms | 233.4 ms | 21 /s (b8), 66 /s (b32) | 536.4 ms (67.0 ms/heteronym) |
| CPU, 4 threads | fp32 | 317.4 ms | 543.3 ms | 634.1 ms | 316.4 ms | 10 /s (b8) | 1173.5 ms (146.7 ms/heteronym) |
| CPU, 1 thread | fp32 | 688.1 ms | 1256.1 ms | 1434.7 ms | 731.1 ms | - | 3246.2 ms (405.8 ms/heteronym) |

Peak GPU memory at inference: 1.9 GB.

**phoneticize demo.** All 8 correct, confidences ≥ 0.997.

**Observations.** The seq2seq formulation is competitive (98.37, between banglabert_small and banglabert) and
needs no candidate list at inference, but it is ~19× slower than the encoders on the T4 at batch 1 (235.6 vs
~12 ms): each output token is a decoder step, and fp32 is required (T5 NaNs in fp16). The latency is also more
variable (p95 1.8× p50) because it grows with sentence length. Training a seed costs ~30 min vs ~4 min for a
base encoder. Hardest words: কর (77.8%), চষে (77.8%), গলে / কাল / ঘটে / চটে (83.3%).

**Problems.** None (no NaN in fp32, no OOM).

**Next.** xlm_roberta_base (started 02:17).

### 2026-10-18 23:04: muril_base (model 4), full cycle

**Run.** Queue, `bash scripts/run_model.sh muril_base` (encoder classifier, no normalizer, fp16, 32 × 1).
22:24 -> 23:03 (39 min).

**lr search (dev).** 1e-5: 94.47 · 2e-5: 96.11 · 3e-5: 95.56 · **5e-5: 97.10**.

**Result (lr 5e-5).** Test accuracy **95.89 ± 1.04** (96.92 / 95.92 / 94.84), macro-F1 95.54 ± 1.13;
2-reading 96.3%, 3-reading 68.5%. Dev 97.10 / 95.74 / 96.56; seed 42 kept. Best epochs 8 / 8 / 7 (no early stop),
5.7 min per seed, peak training memory 4.5 GB, checkpoint 915 MB (197k-token vocab).

**Latency.**

| device | precision | p50 | p95 | p99 | forward p50 | throughput | paragraph (8 heteronyms) |
|---|---|---|---|---|---|---|---|
| T4 GPU | fp16 | 10.3 ms | 11.0 ms | 11.7 ms | 9.9 ms | 689 /s (b8), 1976 /s (b32) | 11.7 ms (1.5 ms/heteronym) |
| CPU, 4 threads | fp32 | 39.8 ms | 50.6 ms | 57.6 ms | 37.7 ms | 42 /s (b8) | 219.5 ms (27.4 ms/heteronym) |
| CPU, 1 thread | fp32 | 104.5 ms | 132.9 ms | 146.8 ms | 103.5 ms | - | 770.0 ms (96.2 ms/heteronym) |

Peak GPU memory at inference: 479 MB.

**phoneticize demo.** All 8 correct, but with flat confidences (0.96 for 6 words, 0.74 for খেল and কর).

**Observations.** Weakest model so far and the least stable across seeds (±1.04). The best lr is the top of the
planned base grid and the confidences saturate around 0.96, which suggests MuRIL is under-trained with this
budget (10 epochs, ≤ 5e-5) rather than unable to learn the task; kept as planned for comparability and worth a
sentence in the paper. Multilingual (17 Indian languages) MuRIL is 3.6 points below Bangla-only BanglaBERT of
similar size. 3-reading words are much harder for it (68.5%); hardest: করাত, কাল, কর (66.7%).

**Problems.** None at run time.

**Next.** banglat5 (started 23:03).

### 2026-10-18 22:25: bangla_bert_base (model 3), full cycle

**Run.** Queue, `bash scripts/run_model.sh bangla_bert_base` (encoder classifier, no normalizer, fp16, 32 × 1,
tokenizer loaded with `do_lower_case=False, strip_accents=False`, see Observations). 21:55 -> 22:24 (29 min).

**lr search (dev).** 1e-5: 97.64 · **2e-5: 98.01** · 3e-5: 97.74 · 5e-5: 97.64.

**Result (lr 2e-5).** Test accuracy **98.01 ± 0.39** (98.28 / 98.19 / 97.55), macro-F1 97.93 ± 0.41;
2-reading 98.1%, 3-reading 94.4%. Dev 98.01 / 97.92 / 98.19; seed 44 kept. Best epochs 6 / 8 / 8; 4.4 min per
seed, peak training memory 3.1 GB, checkpoint 633 MB (102k-token vocab).

**Latency.**

| device | precision | p50 | p95 | p99 | forward p50 | throughput | paragraph (8 heteronyms) |
|---|---|---|---|---|---|---|---|
| T4 GPU | fp16 | 10.6 ms | 12.8 ms | 13.9 ms | 9.4 ms | 707 /s (b8), 2068 /s (b32) | 11.5 ms (1.4 ms/heteronym) |
| CPU, 4 threads | fp32 | 30.5 ms | 36.6 ms | 37.7 ms | 30.1 ms | 59 /s (b8) | 167.2 ms (20.9 ms/heteronym) |
| CPU, 1 thread | fp32 | 82.8 ms | 99.5 ms | 103.4 ms | 82.1 ms | - | 563.1 ms (70.4 ms/heteronym) |

Peak GPU memory at inference: 337 MB.

**phoneticize demo.** All 8 correct (কর 0.94, others ≥ 0.9998).

**Observations.** Same BERT-base size class as banglabert but 1.45 points lower (98.01 vs 99.46) and with
3× the seed variance; BanglaBERT's ELECTRA pre-training (and its normalizer) looks clearly better for this task.
Its errors are spread over 2-reading words (কাল 66.7%, আসলে / চরে / সিম 72.2%) rather than the 3-reading ones.
Plain BERT is the fastest base-size model on the T4 so far (10.6 ms p50, 2068 /s at batch 32).

**Problems.** None at run time; the lowercasing bug of the hub tokenizer was fixed before training (stage 1).

**Next.** muril_base (started 22:24).

### 2026-10-18 21:56: banglabert (model 2), full cycle

**Run.** `bash scripts/run_model.sh banglabert` (encoder classifier, normalizer on, fp16, 32 × 1, same recipe as
model 1). 21:30 -> 21:55 (25 min). From here on the remaining models run back to back from a tmux queue
(`hetero-queue`, `logs/queue.log`), as agreed.

**lr search (seed 42, dev accuracy).** 1e-5: 98.73 · **2e-5: 99.18** · 3e-5: 98.91 · 5e-5: 99.00.

**Result (lr 2e-5).** Test accuracy **99.46 ± 0.16** (99.28 / 99.55 / 99.55), macro-F1 99.40 ± 0.16;
2-reading words 99.6%, 3-reading words 87.0%. Dev 99.18 / 99.46 / 99.28; seed 43 kept (best dev). Best epochs
7 / 5 / 9; seed 43 stopped early after epoch 8. 3.7 min per seed, peak training memory 2.1 GB, checkpoint
421 MB.

**Latency.**

| device | precision | p50 | p95 | p99 | forward p50 | throughput | paragraph (8 heteronyms) |
|---|---|---|---|---|---|---|---|
| T4 GPU | fp16 | 12.5 ms | 13.5 ms | 14.7 ms | 11.8 ms | 539 /s (b8), 1500 /s (b32) | 15.3 ms (1.9 ms/heteronym) |
| CPU, 4 threads | fp32 | 32.8 ms | 38.5 ms | 40.8 ms | 31.2 ms | 56 /s (b8) | 176.8 ms (22.1 ms/heteronym) |
| CPU, 1 thread | fp32 | 84.5 ms | 103.4 ms | 119.4 ms | 82.7 ms | - | 599.0 ms (74.9 ms/heteronym) |

Peak GPU memory at inference: 233 MB.

**phoneticize demo.** All 8 heteronyms correct, same output as banglabert_small, higher confidence on কর (0.90).

**Observations.** 8× the parameters of banglabert_small cut the test error from 1.84% to 0.54%. GPU batch-1
latency is the same as for the small model (12.5 vs 12.6 ms): at batch 1 on the T4 both are bound by per-layer
launch overhead (both have 12 layers), not compute; on CPU the cost scales with size (32.8 vs 13.0 ms with 4
threads). Hardest words: কর (74.1%, 3 readings), মজে (83.3%), বেচে and ধমকে (88.9%).

**Problems.** None.

**Next.** bangla_bert_base (queue started 21:55).

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
