# Markov 

## Brief Introduction

This folder contains codes of the Markov model / processing procedure in the paper. You can find reproducable results of the Markov models in the paper (for example, numbers in FIgure 4 and Figure 5).

## Usage

**(Please make sure the data set under `data/` is ready before running the following scripts)**

For next successive target, run:

```shell
python general-markov-next-successive.py {GRANULARITY}
```

For next important target:

```shell
python general-markov-next-important.py {GRANULARITY} {TIME_THRESHOLD}
```

For next longest target:

```shell
python general-markov-next-longest.py {GRANULARITY} {N_LOCATION}
```



Following is one of the example (you may not get the same number since we provide only a sampled dataset):

```
$ python general-markov-next-important.py 25 5
# Correct: 385
# Total: 1973
# Accuracy: 0.1951343132285859
```


