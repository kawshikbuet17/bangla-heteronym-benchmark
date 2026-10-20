# Ablations

Main split test set, accuracy / macro-F1 in %, mean ± std over seeds.

## marker: banglabert_large

| variant | setting | seeds | accuracy | macro-F1 |
|---|---|---|---|---|
| @ target @ | main | 3 | 99.18 ± 0.16 | 99.14 ± 0.16 |
| <h> target </h> (new tokens) | abl_marker_special | 3 | 99.34 ± 0.14 | 99.29 ± 0.14 |
| no marker | abl_marker_none | 3 | 99.67 ± 0.10 | 99.66 ± 0.11 |

## encoder head: banglabert_large

| variant | setting | seeds | accuracy | macro-F1 |
|---|---|---|---|---|
| opening marker token | main | 3 | 99.18 ± 0.16 | 99.14 ± 0.16 |
| mean of target tokens | abl_head_mean | 3 | 99.52 ± 0.14 | 99.50 ± 0.14 |
| [CLS] | abl_head_cls | 3 | 99.09 ± 0.31 | 99.03 ± 0.37 |

## seq2seq output: banglat5

| variant | setting | seeds | accuracy | macro-F1 |
|---|---|---|---|---|
| full sentence | main | 3 | 98.37 ± 0.18 | 98.37 ± 0.12 |
| target pronunciation only | abl_output_target | 3 | 98.28 ± 0.47 | 98.13 ± 0.57 |
