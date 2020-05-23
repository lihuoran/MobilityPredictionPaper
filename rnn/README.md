# RNN 

## Brief Introduction

This folder contains codes of the LSTM / vanilla RNN model mentioned in the paper (Figure 3(b)). You can find reproducable results of the LSTM / vanilla RNN models in the paper (for example, numbers in FIgure 4 and Figure 5, Figure 10, and Figure 11).

## Usage

**(Please make sure the data set under `data/` is ready before running the following scripts)**

The LSTM model is implemented in `rnn_model.py`

For next successive target, run:

```shell
python rnn-next-successive-location.py --granularity {GRANULARITY}
```

For next important target:

```shell
python rnn-next-important-location.py --granularity {GRANULARITY} --time_thres {TIME_THRESHOLD}
```

For next longest target:

```shell
python rnn-next-longest-location.py --granularity {GRANULARITY} --n_location {N_LOCATION}
```

The default RNN cell is LSTM cell. You could change `run_model.py` if you want to use vanilla cell.

The results may not be exactly equal to the numbers in the paper due to randomness.



After the training finishes, there will be the following generated files in the folder:

- `reslog.txt`: training log
- `logits-train/validation/test`: the pre-trained logits of the three parts of the data, respectively (check Figure 7(b) or Figure 7(c)). These logits could be used for downstream training tasks (i.e., multi-modal learning).
