# Progress (source of truth for resuming)

Spec: PROMPTS.md. GPU: T4 = `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1`. Env: conda `hetero`.

1. [x] Setup: conda env, requirements.txt, git init, data download, splits, data checks, latency inputs, code, tests. Commit.
2. [x] Smoke test all 13 models (~20 steps, tiny subset). Fix + log. Commit.
3. [x] banglabert_small (done, awaiting user OK) full cycle -> report, WAIT for user OK.
4. [ ] Remaining models one by one (one commit each):
   - [ ] banglabert  - [ ] bangla-bert-base  - [ ] muril  - [ ] banglat5  - [ ] xlm-roberta-base
   - [ ] IndicBERTv2  - [ ] mdeberta  - [ ] mt5-small  - [ ] byt5-small  - [ ] banglabert_large
   - [ ] xlm-roberta-large  - [ ] mt5-base (optional)
5. [ ] Candidate scoring: best 3 encoders, main + unseen split, latency. Commit.
6. [ ] Unseen-word split: seq2seq x 3 seeds. Commit.
7. [ ] Learning curve 2/5/10/14: best 3 encoders + best seq2seq x 3 seeds. Commit.
8. [ ] Ablations. Commit.
9. [ ] Final aggregation + summary table. Commit.
   Note: xlm_roberta_large + mt5_base need triton containers on T4 stopped (user allowed); restart after.
