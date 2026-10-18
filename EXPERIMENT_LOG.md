# Experiment log: Bangla heteronym disambiguation benchmark

## Current results

Test = main split test set (3 sentences per pronunciation, 1,104 sentences), mean ± std over seeds 42/43/44.
Latency = batch size 1, full path (normalizer + tokenizer + model + decoding), p50.

| # | model | size | precision | best lr | test acc | macro-F1 | train time | GPU p50 (ms) | CPU p50 (ms, 4 thr) | checkpoint | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | banglabert_small | 13.7M | fp16 | | | | | | | | pending |
| 2 | banglabert | ~110M | fp16 | | | | | | | | pending |
| 3 | bangla_bert_base | ~165M | fp16 | | | | | | | | pending |
| 4 | muril_base | ~237M | fp16 | | | | | | | | pending |
| 5 | banglat5 | 247.6M | fp32 | | | | | | | | pending |
| 6 | xlm_roberta_base | ~278M | fp16 | | | | | | | | pending |
| 7 | indicbertv2_mlm | ~278M | fp16 | | | | | | | | pending |
| 8 | mdeberta_v3_base | ~280M | fp32 | | | | | | | | pending |
| 9 | mt5_small | ~300M | fp32 | | | | | | | | pending |
| 10 | byt5_small | ~300M | fp32 | | | | | | | | pending |
| 11 | banglabert_large | ~335M | fp16 | | | | | | | | pending |
| 12 | xlm_roberta_large | ~560M | fp16 | | | | | | | | pending |
| 13 | mt5_base | ~580M | fp32 | | | | | | | | pending (optional) |

Disk used by `checkpoints/`: 0 (nothing kept yet).

## Observations for the paper

- Tokenizers and the ZWNJ (`results/tokenizer_zwnj.csv`): only ByT5 round-trips the ZWNJ. WordPiece tokenizers
  (BanglaBERT small/base/large, bangla-bert-base, MuRIL) drop it before tokenization, so those encoders never see
  it (token ids identical with and without it). The sentencepiece models (XLM-R, mDeBERTa, mT5, BanglaT5) and
  IndicBERTv2 produce different ids with a ZWNJ but cannot decode it back; T5 tokenizers decode it as a space.
- `sagorsarker/bangla-bert-base` ships without a `tokenizer_config.json`, so the default lowercasing strips Bangla
  vowel signs (তুমি -> তমি, হয়েছিল -> হযেছিল) although the vocab contains them. We load it with
  `do_lower_case=False, strip_accents=False`. Anyone using the hub default gets a crippled tokenizer.

## Entries (newest first)

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
