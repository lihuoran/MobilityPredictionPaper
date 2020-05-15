import numpy as np
import os
import sys
from sklearn.ensemble import RandomForestClassifier as rf

from scipy.stats import pearsonr

def get(fn):
    print('Getting data:', fn)
    return np.loadtxt(fn, delimiter='\t')

def read(tag):
    target = sys.argv[1] # important, longest, successive
    k = int(sys.argv[2]) # 5, 10, 25, 50, 75, 100
    if target != 'successive':
        thres = int(sys.argv[3])
        feature_set = set(sys.argv[4:])
    else:
        feature_set = set(sys.argv[3:])

    if target == 'important':
        logit_folder = {5: 'logits-part1/', 10: 'logits-part2/', 2: 'logits-part3/'}[thres]
    elif target == 'longest':
        logit_folder = {5: 'logits-part1/', 3: 'logits-part2/', 10: 'logits-part3/'}[thres]
    else:
        logit_folder = 'logits-part1/'
    logit_folder = os.path.join('../data/pretrained_logits', logit_folder)

    X = []
    if 'logit' in feature_set:
        X.append(get(os.path.join(logit_folder, '{}-{:03d}-{}'.format(target, k, tag))))
    if 'app' in feature_set:    
        X.append(get('../data/context_data/app-usage/app-usage-{}'.format(tag)))
    if 'time' in feature_set:
        X.append(get('../data/context_data/time/time-{}'.format(tag)))
    if 't2' in feature_set:
        X.append(get('../data/context_data/t2/t2-{}'.format(tag)))
    if 'broadcast' in feature_set:
        X.append(get('../data/context_data/broadcast/broadcast-{}'.format(tag)))
    X = np.concatenate(X, axis=1)
    
    if target == 'successive':
        y = get('../data/context_data/y/{}-{:03d}-{}'.format(target, k, tag))
    else:
        y = get('../data/context_data/y/{}-{}-{:03d}-{}'.format(target, thres, k, tag))

    X = [X[i] for i in range(len(y)) if y[i] >= 0]
    y = [y[i] for i in range(len(y)) if y[i] >= 0]
    X = np.array(X)
    y = np.array(y)
    return (X, y)

def main():
    onehot = lambda val, size: [1 if i == val else 0 for i in range(size)]
    correct = lambda t, p: sum(x == y for x, y in zip(t, p))

    np.random.seed(20180109)

    select = None
    (X_train, y_train) = read('train')
    (X_valid, y_valid) = read('validation')
    (X_test, y_test) = read('test')
    print('train:', X_train.shape, y_train.shape)
    print('test:', X_test.shape, y_test.shape)
    print('valid:', X_valid.shape, y_valid.shape)

    model_name, model = 'rf', rf(n_jobs=-1, n_estimators=500)

    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    total = len(pred)
    acc_test = correct(pred, y_test) / total

    pred = model.predict(X_valid)
    total = len(pred)
    acc_valid = correct(pred, y_valid) / total

    pred = model.predict(X_train)
    total = len(pred)
    acc_train = correct(pred, y_train) / total

    with open('result.txt', 'a') as fout:
        fout.write(' '.join(sys.argv[1:]) + '\ttrain: {:.4f}\ttest: {:.4f}\tvalid: {:.4f}\n'.format(
            acc_train, acc_test, acc_valid))

if __name__ == '__main__':
    main()

