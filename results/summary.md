# Summary

Accuracy and macro-F1 in %, mean ± std over seeds. Latency: batch 1, full path, p50 ms (GPU: Tesla T4 at training precision; CPU: fp32, 4 / 1 threads).

## abl_head_cls / test

| model | type | params | precision | batch | lr | seeds | accuracy | macro-F1 | acc 2 / 3 readings | other-word change | train time (min) | epochs | GPU p50 | CPU4 p50 | CPU1 p50 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| banglabert_large | encoder | 335.6M | fp16 | 16x2 | 2e-05 | 3 | 99.09 ± 0.31 | 99.03 ± 0.37 | 99.2 / 94.4 |  | 12.2 | 9.0 | 22.8 | 107.8 | 301.8 |

## abl_head_mean / test

| model | type | params | precision | batch | lr | seeds | accuracy | macro-F1 | acc 2 / 3 readings | other-word change | train time (min) | epochs | GPU p50 | CPU4 p50 | CPU1 p50 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| banglabert_large | encoder | 335.6M | fp16 | 16x2 | 2e-05 | 3 | 99.52 ± 0.14 | 99.50 ± 0.14 | 99.5 / 98.1 |  | 12.4 | 9.0 | 22.8 | 107.8 | 301.8 |

## abl_marker_none / test

| model | type | params | precision | batch | lr | seeds | accuracy | macro-F1 | acc 2 / 3 readings | other-word change | train time (min) | epochs | GPU p50 | CPU4 p50 | CPU1 p50 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| banglabert_large | encoder | 335.6M | fp16 | 16x2 | 2e-05 | 3 | 99.67 ± 0.10 | 99.66 ± 0.11 | 99.7 / 100.0 |  | 12.7 | 9.3 | 22.8 | 107.8 | 301.8 |

## abl_marker_special / test

| model | type | params | precision | batch | lr | seeds | accuracy | macro-F1 | acc 2 / 3 readings | other-word change | train time (min) | epochs | GPU p50 | CPU4 p50 | CPU1 p50 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| banglabert_large | encoder | 335.6M | fp16 | 16x2 | 2e-05 | 3 | 99.34 ± 0.14 | 99.29 ± 0.14 | 99.4 / 94.4 |  | 12.5 | 9.0 | 22.8 | 107.8 | 301.8 |

## abl_output_target / test

| model | type | params | precision | batch | lr | seeds | accuracy | macro-F1 | acc 2 / 3 readings | other-word change | train time (min) | epochs | GPU p50 | CPU4 p50 | CPU1 p50 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| banglat5 | seq2seq | 247.6M | fp32 | 16x1 | 0.0005 | 3 | 98.28 ± 0.47 | 98.13 ± 0.57 | 98.4 / 88.9 |  | 25.1 | 12.0 |  |  |  |

## lc2 / test

| model | type | params | precision | batch | lr | seeds | accuracy | macro-F1 | acc 2 / 3 readings | other-word change | train time (min) | epochs | GPU p50 | CPU4 p50 | CPU1 p50 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| banglabert | encoder | 110.0M | fp16 | 32x1 | 2e-05 | 1 | 91.94 | 90.97 | 92.2 / 77.8 |  | 0.6 | 10.0 | 12.5 | 32.8 | 84.5 |
| banglat5 | seq2seq | 247.6M | fp32 | 16x1 | 0.0005 | 1 | 83.97 | 83.41 | 84.1 / 77.8 | 3.6 | 15.5 | 15.0 | 236.8 | 323.3 | 745.9 |
| banglabert_large | encoder | 335.6M | fp16 | 16x2 | 2e-05 | 1 | 93.57 | 92.93 | 93.6 / 88.9 |  | 2.3 | 10.0 | 22.8 | 107.8 | 301.8 |
| xlm_roberta_large | encoder | 559.9M | fp16 | 8x4 | 2e-05 | 1 | 90.94 | 89.67 | 91.3 / 66.7 |  | 3.8 | 10.0 | 17.8 | 126.4 | 380.8 |

## main / test

| model | type | params | precision | batch | lr | seeds | accuracy | macro-F1 | acc 2 / 3 readings | other-word change | train time (min) | epochs | GPU p50 | CPU4 p50 | CPU1 p50 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| banglabert_small | encoder | 13.7M | fp16 | 32x1 | 0.0001 | 3 | 98.16 ± 0.14 | 98.09 ± 0.13 | 98.4 / 83.3 |  | 2.5 | 10.0 | 12.6 | 13.0 | 17.4 |
| banglabert | encoder | 110.0M | fp16 | 32x1 | 2e-05 | 3 | 99.46 ± 0.16 | 99.40 ± 0.16 | 99.7 / 87.0 |  | 3.7 | 9.3 | 12.5 | 32.8 | 84.5 |
| bangla_bert_base | encoder | 164.4M | fp16 | 32x1 | 2e-05 | 3 | 98.01 ± 0.39 | 97.93 ± 0.41 | 98.1 / 94.4 |  | 4.4 | 9.7 | 10.6 | 30.5 | 82.8 |
| muril_base | encoder | 237.6M | fp16 | 32x1 | 5e-05 | 3 | 95.89 ± 1.04 | 95.54 ± 1.13 | 96.3 / 68.5 |  | 5.7 | 10.0 | 10.3 | 39.8 | 104.5 |
| banglat5 | seq2seq | 247.6M | fp32 | 16x1 | 0.0005 | 3 | 98.37 ± 0.18 | 98.37 ± 0.12 | 98.5 / 88.9 | 0.6 | 31.4 | 11.3 | 236.8 | 323.3 | 745.9 |
| xlm_roberta_base | encoder | 278.0M | fp16 | 32x1 | 2e-05 | 3 | 97.37 ± 0.39 | 97.27 ± 0.36 | 97.5 / 90.7 |  | 6.5 | 9.7 | 10.3 | 37.5 | 95.9 |
| indicbertv2_mlm | encoder | 278.0M | fp16 | 32x1 | 5e-05 | 3 | 98.34 ± 0.21 | 98.29 ± 0.21 | 98.3 / 100.0 |  | 6.2 | 10.0 | 11.3 | 31.4 | 82.0 |
| mdeberta_v3_base | encoder | 278.2M | fp32 | 16x2 | 3e-05 | 3 | 97.61 ± 0.14 | 97.48 ± 0.10 | 97.8 / 87.0 |  | 21.2 | 9.7 | 24.9 | 118.2 | 316.5 |
| mt5_small | seq2seq | 300.2M | fp32 | 16x1 | 0.0003 | 3 | 93.36 ± 0.21 | 92.99 ± 0.18 | 93.5 / 85.2 | 0.0 | 57.3 | 15.0 | 307.6 | 923.6 | 1609.4 |
| banglabert_large | encoder | 335.6M | fp16 | 16x2 | 2e-05 | 3 | 99.18 ± 0.16 | 99.14 ± 0.16 | 99.3 / 92.6 |  | 11.7 | 8.3 | 22.8 | 107.8 | 301.8 |
| xlm_roberta_large | encoder | 559.9M | fp16 | 8x4 | 2e-05 | 3 | 98.43 ± 0.19 | 98.34 ± 0.14 | 98.5 / 94.4 |  | 23.5 | 9.7 | 17.8 | 126.4 | 380.8 |
| mt5_base | seq2seq | 582.4M | fp32 | 8x2 | 0.0003 | 3 | 96.11 ± 0.16 | 95.99 ± 0.24 | 96.3 / 87.0 | 0.0 | 124.6 | 15.0 | 436.2 | 1758.6 | 2815.2 |
