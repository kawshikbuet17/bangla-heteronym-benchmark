# Progress (source of truth for resuming)

Spec: PROMPTS.md. GPU: T4 = `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1`. Env: conda `hetero`.

1. [x] Setup: conda env, requirements.txt, git init, data download, splits, data checks, latency inputs, code, tests. Commit.
2. [x] Smoke test all 13 models (ByT5 later dropped, 12 remain) (~20 steps, tiny subset). Fix + log. Commit.
3. [x] banglabert_small (user OK 2026-10-18: continue through all models, report after each) full cycle -> report, WAIT for user OK.
4. [x] Remaining models one by one (one commit each):
   - [x] banglabert  - [x] bangla-bert-base  - [x] muril  - [x] banglat5  - [x] xlm-roberta-base
   - [x] IndicBERTv2  - [x] mdeberta  - [x] mt5-small  - [-] byt5-small (dropped by owner 2026-10-20 18:55)  - [x] banglabert_large
   - [x] xlm-roberta-large  - [x] mt5-base (optional)
5. [x] Candidate scoring: best 3 encoders, main + unseen split, latency. Commit.
   cand_main done. XLM-R-large seed 44 collapsed (55.8%); report 3-seed mean + note, no extra mean (owner, 2026-10-20). README note at stage 9.
6. [x] Unseen-word split: seq2seq x 3 seeds. Commit.
7. [x] Learning curve 2/5/10/14: all 12 models, seed 42 only (owner, 2026-10-20 19:38; extra lc2 seeds deleted). Commit.
   Order (queue5, tmux hetero-queue5): lc5/lc10 for the 4 started models, stages 5/6, target-only ckpt, then the other 8 (mT5-base last).
   Results are committed automatically after every finished queue step (scratchpad/autocommit.sh).
8. [x] Ablations. Commit.
   Final set (owner, 2026-10-20 19:34, no more ablations): banglabert_large x4 + banglat5 target-only, 3 seeds, done.
   Target-only latency done (queue5, 2026-10-21 17:16).
9. [x] Final aggregation + summary table. Commit. queue5 finished 2026-10-21 17:16 (no failures), README results updated.
   Note: xlm_roberta_large + mt5_base need triton containers on T4 stopped (user allowed); restart after.
   Queue: tmux session hetero-queue (script in session scratchpad), log logs/queue.log; per model: log entry + commit.
   Order change (user, 2026-10-19 15:47): byt5_small stopped, runs after banglabert_large, xlm_roberta_large, mt5_base.
   2026-10-19 15:55: queue2 (scratchpad/queue2.sh): triton containers stopped while training, started for each 'measure' phase.
   2026-10-19 16:18: paused for office hours (services back up); hetero-resume restarts queue2 at 17:15.
10. [ ] Paper (user request 2026-10-20): bangla-heteronym-benchmark-paper/ with ACL preprint template (XeLaTeX, bundled Bengali font),
    tables generated from results/ by scripts/paper_tables.py, stages 5-8 as \PH placeholders; refill after. ByT5 removed from the paper.
    2026-10-21 17:25: all \PH marks filled with the final results (further.tex, conclusion, limitations); compiles. Not committed (gitignored, owner decides).
