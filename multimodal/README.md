# Multi-modal

## Brief Introduction

This folder contains codes of multi-modal prediction in the paper. Related figures: Figure 6, 8, 9. The processing structure is demonstrated in Figure 7(c).

## Usage

**(Please make sure the data set under `data/` is ready before running the following scripts)**

Command:

```
python run-multimodal.py {successive/important/longest} {GRANULARITY} [TIME_THRES or N_LOCATION] [MODULES]
```

For example, if you want to calculate next important target (granularity = 25, time_thres = 5 min), using pre-trained logits and broadcast data, run:

```
python run-multimodal.py important 25 5 logit broadcast
```

All possible commands are listed in `commands.sh`.



The results will be appended to `result.txt`.




