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
| 5 | banglat5 | 247.6M | fp32 | 0.0005 | 98.37 ± 0.18 | 98.37 ± 0.12 | 31.4 | 236.8 | 323.3 | 948 MB | done |
| 6 | xlm_roberta_base | 278.0M | fp16 | 2e-05 | 97.37 ± 0.39 | 97.27 ± 0.36 | 6.5 | 10.3 | 37.5 | 1082 MB | done |
| 7 | indicbertv2_mlm | 278.0M | fp16 | 5e-05 | 98.34 ± 0.21 | 98.29 ± 0.21 | 6.2 | 11.3 | 31.4 | 1068 MB | done |
| 8 | mdeberta_v3_base | 278.2M | fp32 | 3e-05 | 97.61 ± 0.14 | 97.48 ± 0.10 | 21.2 | 24.9 | 118.2 | 1081 MB | done |
| 9 | mt5_small | 300.2M | fp32 | 0.0003 | 93.36 ± 0.21 | 92.99 ± 0.18 | 57.3 | 307.6 | 923.6 | 1165 MB | done |
| 10 | banglabert_large | 335.6M | fp16 | 2e-05 | 99.18 ± 0.16 | 99.14 ± 0.16 | 11.7 | 22.8 | 107.8 | 1282 MB | done |
| 11 | xlm_roberta_large | 559.9M | fp16 | 2e-05 | 98.43 ± 0.19 | 98.34 ± 0.14 | 23.5 | 17.8 | 126.4 | 2157 MB | done |
| 12 | mt5_base | 582.4M | fp32 | 0.0003 | 96.11 ± 0.16 | 95.99 ± 0.24 | 124.6 | 436.2 | 1758.6 | 2242 MB | done |

Disk used by `checkpoints/`: 13G.
<!-- /results-table -->

## Observations for the paper

- Tokenizers and the ZWNJ (`results/tokenizer_zwnj.csv`): none of the 12 benchmarked tokenizers round-trips the ZWNJ. WordPiece tokenizers
  (BanglaBERT small/base/large, bangla-bert-base, MuRIL) drop it before tokenization, so those encoders never see
  it (token ids identical with and without it). The sentencepiece models (XLM-R, mDeBERTa, mT5, BanglaT5) and
  IndicBERTv2 produce different ids with a ZWNJ but cannot decode it back; T5 tokenizers decode it as a space.
- `sagorsarker/bangla-bert-base` ships without a `tokenizer_config.json`, so the default lowercasing strips Bangla
  vowel signs (তুমি -> তমি, হয়েছিল -> হযেছিল) although the vocab contains them. We load it with
  `do_lower_case=False, strip_accents=False`. Anyone using the hub default gets a crippled tokenizer.

## Entries (newest first)

### 2026-10-20 18:40: ablations on banglabert_large (stage 8, encoder part)

**Run.** `bash scripts/run_setting.sh <setting> banglabert_large` for the four encoder ablations, 3 seeds each, lr 2e-5,
triton containers stopped; 14:41 -> 17:15 (about 38 min per setting).

**Result (main test set, accuracy / macro-F1 in %, mean ± std over seeds 42/43/44).**

| setting | variant | accuracy | macro-F1 |
|---|---|---|---|
| main | `@ target @`, head on the opening `@` | 99.18 ± 0.16 | 99.14 ± 0.16 |
| abl_marker_special | new tokens `<h> target </h>` | 99.34 ± 0.14 | 99.29 ± 0.14 |
| abl_marker_none | no marker, head on the target's first token | **99.67 ± 0.10** | 99.66 ± 0.11 |
| abl_head_mean | `@` marker, mean of the target's tokens | 99.52 ± 0.14 | 99.50 ± 0.14 |
| abl_head_cls | `@` marker, `[CLS]` token | 99.09 ± 0.31 | 99.03 ± 0.37 |

**Observations.** Reading the target's own tokens (no marker, or the mean of the target tokens) works better than
reading the `@` marker; without markers BanglaBERT-large reaches 99.67%, the best result of the benchmark so far.
`[CLS]` is the weakest and least stable head. The differences are 1-5 test sentences out of 1,104, so they are
small; the no-marker gain (0.49 points, about 3 std) is the clearest.

**banglat5, target-only output** (abl_output_target, 17:15 -> 18:32): 98.28 ± 0.47 accuracy, 98.13 ± 0.57 macro-F1,
against 98.37 ± 0.18 / 98.37 ± 0.12 for the full-sentence output. Generating only the reading is as accurate within
noise but less stable across seeds; it produces much shorter outputs, so its latency gain should be measured if it is
used in practice. Ablation checkpoints are deleted after evaluation, so one more seed-42 run
(`abl_output_target_keep`, same settings, checkpoint kept) is queued right after stages 5-8; its
latency goes to `runs/banglat5/latency_target.json` (variant `target-only` in `results/latency.csv`).

**Next.** Learning curves (started 18:32), candidate scoring, unseen-word split.

### 2026-10-20 19:38: learning curves with seed 42 only

The owner chose one seed for all learning curves. lc2 had already finished with 3 seeds for banglabert_large,
banglabert and xlm_roberta_large and 2 seeds for banglat5 (seed 44 was running and was stopped); the seed 43/44 runs
were deleted (never committed). Seed spread at lc2 was 0.6 to 1.3 points against drops of 5 to 15 points from 14 to
2 sentences, so one seed shows the shape. The point at 14 is now the seed-42 main run (`LC_SEED`), not the 3-seed
mean. `scripts/run_setting.sh` takes `SEEDS=42`. queue3/queue4 were replaced by queue5 (tmux `hetero-queue5`):
lc5/lc10 seed 42 for the four started models, cand_main, cand_unseen, unseen seq2seq, the target-only BanglaT5
checkpoint, lc2/lc5/lc10 seed 42 for the other eight models (mT5-base last), then the latency runs. Estimated end:
~19:00-20:00 on 2026-10-21. Seed-42 lc2 so far (test accuracy, 2 sentences per reading vs 14): BanglaBERT-large 93.57
vs 99.09, BanglaBERT 91.94 vs 99.28, XLM-R-large 90.94 vs 98.28, BanglaT5 83.97 vs 98.55.

### 2026-10-20 19:34: no more ablations

The owner decided against extending the ablations to more models (estimated ~8 h at one seed, ~22 h at three
seeds). The five ablations already finished (BanglaBERT-large: two markers, two heads; BanglaT5: target-only output,
three seeds each) are the final ablation set. The no-marker gain (+0.48) is reported as tested on one model only.
Only the target-only latency run (`abl_output_target_keep`) remains for stage 8.

### 2026-10-20 19:05: learning curves extended to all 12 models

At the owner's request, lc2/lc5/lc10 (3 seeds each) also run for banglabert_small, bangla_bert_base, muril_base,
xlm_roberta_base, indicbertv2_mlm, mdeberta_v3_base and mt5_small after stages 5-8 and the target-only latency run,
then mt5_base last (it alone is ~8 h). Estimated ~7 h for the seven models plus ~8.4 h for mt5_base. Finished runs
and `results/` are committed after every queue step. Order (owner): seed by seed, i.e. seed 42 (lc2, lc5, lc10) for all seven
models, then seed 43, then seed 44, so every model has a first curve after ~2.3 h; then mt5_base, also seed by seed
(~2.8 h per seed).

### 2026-10-20 18:55: ByT5-small dropped from the benchmark

The owner dropped byt5_small: the remaining ByT5 work (2 more learning rates, 3 seeds, the unseen-word split and
latency) was estimated at ~50 h of T4 time, about two thirds of everything left (one lr-search run took 6.9 h). The
benchmark now has 12 models. Removed: `configs/byt5_small.yaml`, its finished lr 1e-4 run (dev ~93%) and its
checkpoints; the queue ends after stages 5-8 and the target-only BanglaT5 latency run. Earlier log entries about
ByT5 (smoke test, memory, the queue order changes) are kept as history.

### 2026-10-20 14:20: stages 5-8 scheduled before the rest of ByT5

At the owner's request (all non-ByT5 results should be in the paper first), the queue pauses byt5_small after its
running lr-search run (lr 1e-4) is saved and runs stages 5-8 first, with `scripts/run_setting.sh` (3 seeds per
setting, each model's selected lr, triton containers stopped while training):
- Ablations on the best encoder by mean dev accuracy over seeds, **banglabert_large** (99.46 vs 99.31 for
  banglabert): marker `<h> </h>` and none, head mean and [CLS]; and on the best seq2seq model, **banglat5**:
  target-only output.
- Learning curves (2/5/10 sentences per reading; 14 is the main run) for the three best encoders by mean dev
  accuracy (banglabert_large, banglabert, xlm_roberta_large) and banglat5.
- Candidate scoring for the same three encoders on the main split (best seed kept in `checkpoints/{model}_cand`,
  latency in `runs/{model}/latency_cand.json`) and on the unseen-word split.
- Unseen-word split for banglat5, mt5_small and mt5_base.
Then the rest of ByT5 (4 runs, latency) and ByT5 on the unseen-word split. Estimated ~28 h for stages 5-8.

### 2026-10-20 07:47: mt5_base (model 13, optional), full cycle

**Run.** Queue (`train` 2026-10-19 20:21 -> 2026-10-20 06:49 with the triton containers stopped; `measure`
06:50-07:45 with them running idle). Seq2seq, full-sentence output, no normalizer, ZWNJ removed model-side,
fp32, 8 × 2 (planned setting; needs the full T4).

**lr search (dev).** 1e-4: 96.20 · **3e-4: 96.56** · 5e-4: 95.47.

**Result (lr 3e-4).** Test target accuracy **96.11 ± 0.16** (96.20 / 96.20 / 95.92), macro-F1 95.99 ± 0.24;
2-reading 96.2%, 3-reading 87.0%. Other-word change rate 0% in all three seeds. Dev 96.56 / 96.38 / 96.92; seed
44 kept. Best epochs 15 / 14 / 13, all runs used the full 15 epochs, ~2 h 05 min per seed, peak training memory
11.1 GB, checkpoint 2.24 GB.

**Latency.** 18.3 generated tokens on average (same mT5 tokenizer as mt5_small).

| device | precision | p50 | p95 | p99 | forward (generate) p50 | throughput | paragraph (8 heteronyms) |
|---|---|---|---|---|---|---|---|
| T4 GPU | fp32 | 436.2 ms | 858.1 ms | 1029.8 ms | 448.6 ms | 8 /s (b8), 24 /s (b32) | 1487.6 ms (185.9 ms/heteronym) |
| CPU, 4 threads | fp32 | 1758.6 ms | 3481.5 ms | 4195.2 ms | 1681.6 ms | 2 /s (b8) | 5389.1 ms (673.6 ms/heteronym) |
| CPU, 1 thread | fp32 | 2815.2 ms | 5590.2 ms | 6839.5 ms | 2808.7 ms | - | 15174.9 ms (1896.9 ms/heteronym) |

Peak GPU memory at inference: 2.9 GB. Generation costs ~24.5 ms per output token on the T4 (448.6 / 18.3).

**phoneticize demo.** All 8 correct (কর 0.994).

**Observations.** Doubling mT5 (300M -> 582M) gains 2.75 points (93.36 -> 96.11) but stays 2.3 below the
248M BanglaT5 (98.37), at 1.8× BanglaT5's GPU latency and 5.4× its CPU latency: the Bangla-specific T5 is better
and cheaper. Like mt5_small it was still improving at the 15-epoch cap (best epochs 13-15), so both mT5 models
are under-trained under the planned budget. It never changed a non-target word. Slowest model of the benchmark.
Hardest: কাল (61.1%), আসর / মেলা (66.7%).

**Containers.** Stopped 2026-10-19 20:21:31 -> 2026-10-20 06:50 for training (10 h 29 min), running for the
latency measurement, stopped again 07:45:52 for byt5_small.

**Problems.** None.

**Next.** byt5_small (last model; started 07:45, 8 × 2 on the full T4).

### 2026-10-19 20:22: xlm_roberta_large (model 12), full cycle

**Run.** Queue (`train` 18:11-20:14 with the triton containers stopped; `measure` 20:14-20:21 with them running
idle). Encoder classifier, no normalizer, fp16, 8 × 4 (planned setting; needs the full T4).

**lr search (dev).** 5e-6: 98.10 · 1e-5: 98.46 · **2e-5: 98.82** (top of the planned large grid).

**Result (lr 2e-5).** Test accuracy **98.43 ± 0.19** (98.28 / 98.37 / 98.64), macro-F1 98.34 ± 0.14;
2-reading 98.5%, 3-reading 94.4%. Dev 98.82 / 98.91 / 98.73; seed 43 kept. Best epochs 9 / 10 / 6, 22-24 min per
seed, peak training memory **10.7 GB** (would not fit next to the containers), checkpoint 2.16 GB.

**Latency.**

| device | precision | p50 | p95 | p99 | forward p50 | throughput | paragraph (8 heteronyms) |
|---|---|---|---|---|---|---|---|
| T4 GPU | fp16 | 17.8 ms | 19.1 ms | 21.5 ms | 17.2 ms | 404 /s (b8), 668 /s (b32) | 20.3 ms (2.5 ms/heteronym) |
| CPU, 4 threads | fp32 | 126.4 ms | 155.5 ms | 179.4 ms | 135.9 ms | 11 /s (b8) | 959.8 ms (120.0 ms/heteronym) |
| CPU, 1 thread | fp32 | 380.8 ms | 501.8 ms | 580.9 ms | 378.1 ms | - | 3317.9 ms (414.7 ms/heteronym) |

Peak GPU memory at inference: 1.1 GB.

**phoneticize demo.** All 8 correct, confidences ≥ 0.998.

**Observations.** Best multilingual model (98.43, above IndicBERTv2's 98.34 by a margin inside one std) and
+1.06 over XLM-R base, but still 1.03 below the 110M Bangla-only BanglaBERT at 5× its parameters, 1.4× its GPU
and ~4× its CPU latency. The best lr is again the top of the planned grid. Hardest: আসর (61.1%), কাল (77.8%).

**Containers.** Stopped 18:11:36-20:14:50 for training (2 h 3 min), running for the latency measurement,
stopped again 20:21:31 for mt5_base.

**Problems.** None.

**Next.** mt5_base (seq2seq, fp32, 8 × 2; started 20:21, up to ~9 h).

### 2026-10-19 18:12: banglabert_large (model 11), full cycle

**Run.** Queue (`run_model.sh banglabert_large train`, then `measure`; encoder classifier, normalizer on, fp16,
16 × 2). Training on the T4 with the triton containers stopped: 15:55-16:18 (lr 5e-6 run), paused for office
hours, 17:15-18:05; latency with the containers running idle, 18:06-18:11.

**lr search (dev).** 5e-6: 99.37 · 1e-5: 99.46 · **2e-5: 99.46** (tie on accuracy; 2e-5 wins on dev macro-F1).

**Result (lr 2e-5).** Test accuracy **99.18 ± 0.16** (99.09 / 99.09 / 99.37), macro-F1 99.14 ± 0.16;
2-reading 99.3%, 3-reading 92.6%. Dev 99.46 / 99.37 / 99.55; seed 44 kept. Best epochs 4 / 5 / 8 (early stop after
7 / 8 / 10), 10-14 min per seed, peak training memory 6.4 GB, checkpoint 1.28 GB.

**Latency.**

| device | precision | p50 | p95 | p99 | forward p50 | throughput | paragraph (8 heteronyms) |
|---|---|---|---|---|---|---|---|
| T4 GPU | fp16 | 22.8 ms | 26.0 ms | 28.6 ms | 21.9 ms | 309 /s (b8), 820 /s (b32) | 25.8 ms (3.2 ms/heteronym) |
| CPU, 4 threads | fp32 | 107.8 ms | 134.7 ms | 155.2 ms | 99.9 ms | 16 /s (b8) | 626.1 ms (78.3 ms/heteronym) |
| CPU, 1 thread | fp32 | 301.8 ms | 362.6 ms | 391.6 ms | 301.0 ms | - | 2089.4 ms (261.2 ms/heteronym) |

Peak GPU memory at inference: 668 MB.

**phoneticize demo.** All 8 correct, confidences ≥ 0.9985.

**Observations.** 3× the parameters of BanglaBERT base and no accuracy gain (99.18 ± 0.16 vs 99.46 ± 0.16;
within ~2 std, ~3 more errors per 1,104 sentences), at 1.8× the GPU and 3.3× the CPU latency: on this dataset
BanglaBERT base is the better trade-off; with 14 training sentences per pronunciation the large model may be
data-limited. GPU batch-1 latency now reflects depth (24 layers: 22.8 ms vs ~12 ms for 12 layers). Hardest:
ব্যাংককে and বেচে (83.3%), কর (85.2%).

**Problems.** None (the office-hours pause only cost the restarted lr 1e-5 run).

**Next.** xlm_roberta_large (started 18:11, containers stopped).

### 2026-10-19 16:19: training paused for office hours, resumes 17:15

At the owner's request the queue was paused at 16:18 and the two triton containers were started again
(healthy at 16:19) so the services are available until the end of office hours. banglabert_large's lr 5e-6
run had finished (dev 99.37%, kept); the lr 1e-5 run (8 min in) was stopped and will be retrained (stopping
rather than freezing the process keeps the recorded training times correct). At 17:15 a scheduled job
(`hetero-resume`) restarts the queue, which stops the containers while training and starts them again for each
latency measurement and when the queue finishes.

### 2026-10-19 15:55: triton containers stopped while training the remaining models

With the owner's permission, the two triton containers on the T4 (`fast_whisper_triton_test`,
`dialect_translation_triton`) are now stopped while the remaining models train, and started again (and allowed
to become healthy and idle) before each latency measurement, so the latency conditions match models 1-9
(containers loaded but idle). `scripts/run_model.sh` gained a phase argument for this (`train` = lr search,
seeds, keep best; `measure` = latency, demo, requirements, aggregation). The queue was restarted at 15:55;
banglabert_large's first lr-search run (started 15:47) was restarted from scratch.

byt5_small goes back to the planned 8 × 2 (it only fell back to 4 × 4 because of the containers' memory);
the OOM fallback still covers a shared GPU. Training-time numbers (memory, minutes per seed) of the remaining
models are therefore measured on a T4 with no other processes; accuracy is unaffected.

### 2026-10-19 15:45: byt5_small moved to the end of the queue (change from the plan's order)

ByT5 is slow on the T4: ~25 min per epoch (~18 min training at 4 × 4 on byte-level sequences of ~300 bytes,
~7 min of greedy dev generation), so its cycle (3 lr-search runs + 2 seeds, ≤15 epochs each) takes ~20-30 h.
At the owner's request it runs after the remaining models instead of in table order. Its running lr-search run
(lr 1e-4; dev 91.5% at epoch 7 of 15, still improving) was stopped at 15:47 and will be retrained from scratch
(runs are not resumable mid-training); the queue now runs banglabert_large, xlm_roberta_large, mt5_base, then
the full byt5_small cycle. Results are unaffected; ~3.3 h of GPU time was discarded.

### 2026-10-19 12:24: mt5_small (model 9), full cycle

**Run.** Queue, `bash scripts/run_model.sh mt5_small` (seq2seq, full-sentence output, no normalizer, ZWNJ
removed model-side, fp32, 16 × 1, ≤15 epochs). Retrained from scratch after the 06:57 failure, with the fixed
confidence code. 07:14 -> 12:23 (5 h 9 min).

**lr search (dev).** 1e-4: 83.70 · **3e-4: 94.11** · 5e-4: 93.57.

**Result (lr 3e-4).** Test target accuracy **93.36 ± 0.21** (93.48 / 93.12 / 93.48), macro-F1 92.99 ± 0.18;
2-reading 93.5%, 3-reading 85.2%. Other-word change rate 0.03% (0 / 0.09 / 0%). Dev 94.11 / 93.57 / 94.75; seed
44 kept. Best epochs 15 / 14 / 13, all runs used the full 15 epochs, 57 min per seed, peak training memory
6.4 GB, checkpoint 1.16 GB.

**Latency.** 18.3 generated tokens on average (vs 10.5 for BanglaT5: mT5's 250k multilingual vocabulary splits
Bangla into more pieces).

| device | precision | p50 | p95 | p99 | forward (generate) p50 | throughput | paragraph (8 heteronyms) |
|---|---|---|---|---|---|---|---|
| T4 GPU | fp32 | 307.6 ms | 617.1 ms | 722.4 ms | 304.7 ms | 14 /s (b8), 42 /s (b32) | 811.9 ms (101.5 ms/heteronym) |
| CPU, 4 threads | fp32 | 923.6 ms | 1857.6 ms | 2144.3 ms | 923.7 ms | 4 /s (b8) | 2755.3 ms (344.4 ms/heteronym) |
| CPU, 1 thread | fp32 | 1609.4 ms | 3220.6 ms | 3862.1 ms | 1607.3 ms | - | 8288.9 ms (1036.1 ms/heteronym) |

Peak GPU memory at inference: 1.5 GB.

**phoneticize demo.** 7 of 8 correct: কর -> কোরো (gold করো), confidence 0.99 (a confident error).

**Observations.** Lowest accuracy so far, 5 points below BanglaT5 of the same architecture family and similar
size: the Bangla-specific T5 (vocabulary and pre-training) matters a lot for seq2seq. All three seeds were still
improving at the 15-epoch cap (best epochs 13-15), so mT5-small is under-trained under the planned budget; the
gap to BanglaT5 is partly convergence speed. It almost never changes non-target words (0.03%), so its errors are
on the target itself. It is also the slowest model so far (GPU 307.6 ms, CPU-4 924 ms), because outputs are
~1.75× longer in tokens than BanglaT5's. Hardest: আসর (33.3%), কাল (50.0%), দেব (55.6%).

**Problems.** The confidence OOM described in the 07:00 entry (fixed before this run). None in this run.

**Next.** byt5_small (4 × 4, fp32; started 12:23).

### 2026-10-19 07:00: mt5_small failed at evaluation (OOM in confidence scoring); fixed, queue restarted

**What happened.** The first mt5_small lr-search run (lr 1e-4) trained all 15 epochs (06:00-06:57, best dev
83.7%), then failed in the post-training dev evaluation: `torch.OutOfMemoryError` in
`compute_transition_scores` (`src/predict.py`). Seq2seq confidence was computed from `generate(...,
output_scores=True)`, which keeps steps × batch × vocab logits and then log-softmaxes them all at once: harmless
for BanglaT5 (32k vocab) but ~2.4 GB per batch of 64 for mT5's 250k vocab, on top of ~8.4 GB in use. The queue
then started byt5_small; it was stopped after a few minutes so that models keep the table order.

**Fix.** `GreedyLogProbs`, a small logits processor that records each greedy step's log-probability
(max logit − logsumexp) during generation; no full-vocabulary scores are kept. Same confidence by definition
(geometric-mean token probability). Verified: re-scoring the kept banglat5 checkpoint gives identical predictions
and accuracy (98.55%) and confidences equal to the stored ones within 1e-6; the leftover mt5_small checkpoint
now evaluates dev at batch 64 with a 1.5 GB peak (83.70%). The "incorrect regex pattern" warning that transformers
prints when loading the saved mT5 tokenizer is a false alarm: identical token ids to `google/mt5-small` on all
8,464 dataset texts.

**Consequences.** banglat5's latency was measured with the heavier `output_scores` path, so its latency
benchmark and demo are re-run with the fixed code (first item of the restarted queue) to keep all models
comparable; its accuracy numbers are unchanged. The failed mt5_small run is retrained from scratch (no train
statistics were saved for it).

**Next.** Queue: banglat5 latency re-run, mt5_small, byt5_small, banglabert_large, xlm_roberta_large, mt5_base.

### 2026-10-19 06:01: mdeberta_v3_base (model 8), full cycle

**Run.** Queue, `bash scripts/run_model.sh mdeberta_v3_base` (encoder classifier, no normalizer, **fp32** as
planned (mDeBERTa gives NaN losses in fp16), 16 × 2). 03:44 -> 06:00 (2 h 17 min).

**lr search (dev).** 1e-5: 97.83 · 2e-5: 98.55 · **3e-5: 98.73** · 5e-5: 98.73 (tie on accuracy; 3e-5 wins on
dev macro-F1).

**Result (lr 3e-5).** Test accuracy **97.61 ± 0.14** (97.46 / 97.64 / 97.74), macro-F1 97.48 ± 0.10;
2-reading 97.8%, 3-reading 87.0%. Dev 98.73 / 98.19 / 98.46; seed 42 kept. Best epochs 6 / 8 / 10, 21 min per
seed (5× a fp16 base encoder), peak training memory 6.0 GB, checkpoint 1.08 GB.

**Latency.**

| device | precision | p50 | p95 | p99 | forward p50 | throughput | paragraph (8 heteronyms) |
|---|---|---|---|---|---|---|---|
| T4 GPU | fp32 | 24.9 ms | 31.0 ms | 31.6 ms | 24.1 ms | 220 /s (b8), 235 /s (b32) | 41.0 ms (5.1 ms/heteronym) |
| CPU, 4 threads | fp32 | 118.2 ms | 137.1 ms | 177.8 ms | 112.9 ms | 23 /s (b8) | 455.2 ms (56.9 ms/heteronym) |
| CPU, 1 thread | fp32 | 316.5 ms | 380.3 ms | 404.3 ms | 315.5 ms | - | 1327.6 ms (166.0 ms/heteronym) |

Peak GPU memory at inference: 1.3 GB.

**phoneticize demo.** 7 of 8 correct: the first demo error of any model, কর -> কোরো (gold করো) with
confidence 0.56 (the 3-reading word).

**Observations.** The cost of fp32 on the T4 is clear here: 2.4× the GPU batch-1 latency of the fp16 base
encoders (24.9 vs ~10-12 ms) and ~8× lower batch-32 throughput (235 vs ~1,900 /s): batching barely helps
because fp32 on Turing has no tensor-core path and is compute-bound already at batch 8. On CPU (all fp32) it is
also the slowest base encoder (118 ms, 4 threads), from DeBERTa's disentangled attention. Accuracy is mid-pack
for the multilingual models (above XLM-R base and MuRIL, below IndicBERTv2), with the lowest seed variance so far
(±0.14). Hardest: সিম (66.7%), বার / মজে (72.2%), কর (74.1%).

**Problems.** None (fp32 as planned; no OOM).

**Next.** mt5_small (seq2seq, fp32; started 06:00).

### 2026-10-19 03:44: indicbertv2_mlm (model 7), full cycle

**Run.** Queue, `bash scripts/run_model.sh indicbertv2_mlm` (encoder classifier, no normalizer, fp16, 32 × 1).
03:02 -> 03:44 (42 min).

**lr search (dev).** 1e-5: 97.74 · 2e-5: 98.19 · 3e-5: 98.55 · **5e-5: 98.64** (top of the planned base grid).

**Result (lr 5e-5).** Test accuracy **98.34 ± 0.21** (98.46 / 98.10 / 98.46), macro-F1 98.29 ± 0.21;
2-reading 98.3%, 3-reading **100%** in all three seeds. Dev 98.64 / 98.55 / 98.64; seed 42 kept (tie with 44,
lower seed). Best epochs 8 / 7 / 8, 6.2 min per seed, peak training memory 5.3 GB, checkpoint 1.07 GB.

**Latency.**

| device | precision | p50 | p95 | p99 | forward p50 | throughput | paragraph (8 heteronyms) |
|---|---|---|---|---|---|---|---|
| T4 GPU | fp16 | 11.3 ms | 12.9 ms | 13.5 ms | 9.3 ms | 646 /s (b8), 1972 /s (b32) | 11.2 ms (1.4 ms/heteronym) |
| CPU, 4 threads | fp32 | 31.4 ms | 38.7 ms | 40.8 ms | 30.9 ms | 50 /s (b8) | 196.3 ms (24.5 ms/heteronym) |
| CPU, 1 thread | fp32 | 82.0 ms | 98.8 ms | 111.7 ms | 81.3 ms | - | 655.7 ms (82.0 ms/heteronym) |

Peak GPU memory at inference: 555 MB.

**phoneticize demo.** All 8 correct (কর 0.996, others ≥ 0.999).

**Observations.** Best multilingual model so far and the only one with every 3-reading test sentence right.
It beats the Bangla-only bangla-bert-base (98.01), so the earlier "Bangla-only beats multilingual" pattern only
holds for BanglaBERT (99.46) vs. the multilingual models; IndicBERTv2 (trained on IndicCorp v2, 24 Indic
languages) is close to BanglaBERT-small/BanglaT5. Its tokenizer is also the only encoder tokenizer whose ids
change with a ZWNJ (results/tokenizer_zwnj.csv). Hardest: কাল (72.2%), আসর (77.8%).

**Problems.** None.

**Next.** mdeberta_v3_base (fp32, 16 × 2; started 03:44).

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

**Latency.** Greedy decoding of the whole sentence, 10.5 generated tokens on average. Re-measured on
2026-10-19 07:14 with the fixed confidence code (see the mt5_small entry); the first measurement was within noise
of these numbers (GPU p50 235.6 vs 236.8 ms, CPU-4 317.4 vs 323.3 ms) but used 1.9 GB peak GPU memory.

| device | precision | p50 | p95 | p99 | forward (generate) p50 | throughput | paragraph (8 heteronyms) |
|---|---|---|---|---|---|---|---|
| T4 GPU | fp32 | 236.8 ms | 410.6 ms | 481.1 ms | 235.5 ms | 21 /s (b8), 67 /s (b32) | 532.1 ms (66.5 ms/heteronym) |
| CPU, 4 threads | fp32 | 323.3 ms | 555.1 ms | 645.1 ms | 320.5 ms | 9 /s (b8) | 1261.8 ms (157.7 ms/heteronym) |
| CPU, 1 thread | fp32 | 745.9 ms | 1286.5 ms | 1488.5 ms | 744.2 ms | - | 3242.8 ms (405.4 ms/heteronym) |

Peak GPU memory at inference: 1.2 GB.

**phoneticize demo.** All 8 correct, confidences ≥ 0.997.

**Observations.** The seq2seq formulation is competitive (98.37, between banglabert_small and banglabert) and
needs no candidate list at inference, but it is ~20× slower than the encoders on the T4 at batch 1 (236.8 vs
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
