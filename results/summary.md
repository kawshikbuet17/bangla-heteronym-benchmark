# Summary

Accuracy and macro-F1 in %, mean ± std over seeds. Latency: batch 1, full path, p50 ms (GPU: Tesla T4 at training precision; CPU: fp32, 4 / 1 threads).

## main / test

| model | type | params | precision | batch | lr | seeds | accuracy | macro-F1 | acc 2 / 3 readings | other-word change | train time (min) | epochs | GPU p50 | CPU4 p50 | CPU1 p50 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| banglabert_small | encoder | 13.7M | fp16 | 32x1 | 0.0001 | 3 | 98.16 ± 0.14 | 98.09 ± 0.13 | 98.4 / 83.3 |  | 2.5 | 10.0 | 12.6 | 13.0 | 17.4 |
| banglabert | encoder | 110.0M | fp16 | 32x1 | 2e-05 | 3 | 99.46 ± 0.16 | 99.40 ± 0.16 | 99.7 / 87.0 |  | 3.7 | 9.3 | 12.5 | 32.8 | 84.5 |
| bangla_bert_base | encoder | 164.4M | fp16 | 32x1 | 2e-05 | 3 | 98.01 ± 0.39 | 97.93 ± 0.41 | 98.1 / 94.4 |  | 4.4 | 9.7 | 10.6 | 30.5 | 82.8 |
| muril_base | encoder | 237.6M | fp16 | 32x1 | 5e-05 | 3 | 95.89 ± 1.04 | 95.54 ± 1.13 | 96.3 / 68.5 |  | 5.7 | 10.0 | 10.3 | 39.8 | 104.5 |
| banglat5 | seq2seq | 247.6M | fp32 | 16x1 | 0.0005 | 3 | 98.37 ± 0.18 | 98.37 ± 0.12 | 98.5 / 88.9 | 0.6 | 31.4 | 11.3 | 235.6 | 317.4 | 688.1 |
| xlm_roberta_base | encoder | 278.0M | fp16 | 32x1 | 2e-05 | 3 | 97.37 ± 0.39 | 97.27 ± 0.36 | 97.5 / 90.7 |  | 6.5 | 9.7 | 10.3 | 37.5 | 95.9 |
| indicbertv2_mlm | encoder | 278.0M | fp16 | 32x1 | 5e-05 | 3 | 98.34 ± 0.21 | 98.29 ± 0.21 | 98.3 / 100.0 |  | 6.2 | 10.0 | 11.3 | 31.4 | 82.0 |
| mdeberta_v3_base | encoder | 278.2M | fp32 | 16x2 | 3e-05 | 3 | 97.61 ± 0.14 | 97.48 ± 0.10 | 97.8 / 87.0 |  | 21.2 | 9.7 | 24.9 | 118.2 | 316.5 |
