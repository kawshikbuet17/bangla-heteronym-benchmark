# Summary

Accuracy and macro-F1 in %, mean ± std over seeds. Latency: batch 1, full path, p50 ms (GPU: Tesla T4 at training precision; CPU: fp32, 4 / 1 threads).

## main / test

| model | type | params | precision | batch | lr | seeds | accuracy | macro-F1 | acc 2 / 3 readings | other-word change | train time (min) | epochs | GPU p50 | CPU4 p50 | CPU1 p50 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| banglabert_small | encoder | 13.7M | fp16 | 32x1 | 0.0001 | 3 | 98.16 ± 0.14 | 98.09 ± 0.13 | 98.4 / 83.3 |  | 2.5 | 10.0 | 12.6 | 13.0 | 17.4 |
| banglabert | encoder | 110.0M | fp16 | 32x1 | 2e-05 | 3 | 99.46 ± 0.16 | 99.40 ± 0.16 | 99.7 / 87.0 |  | 3.7 | 9.3 | 12.5 | 32.8 | 84.5 |
