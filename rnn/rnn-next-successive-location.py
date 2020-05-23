import numpy as np
import rnn_model
import argparse
import datetime

class Logger():
    def __init__(self, path):
        self._f = open(path, 'a')

    def print(self, line):
        print(line)
        self._f.write(line + '\n')
        self._f.flush()

    def close(self):
        self._f.close()

def batch(x, y, size):
    n = len(x)
    size = min(size, n)
    idx = np.random.choice(list(range(n)), size, replace=False)
    retx = np.array([x[i] for i in idx])
    rety = np.array([y[i] for i in idx])
    return (retx, rety)

def next_different_loc(l1, t1, l2, t2):
    assert(len(l1) == len(t1))
    assert(len(l2) == len(t2))

    for e in l2:
        if e != l1[-1]:
            return e
    return None

def onehot(e, n):
    return [1 if i == e else 0 for i in range(n)]

def read_data(tag, size, seqlimit=None):
    print('{}\treading data: {}'.format(datetime.datetime.now(), tag))
    x = []
    y = []
    fin = open('../data/trajectory_data/{}-{:03d}'.format(tag, size), 'r')
    for line in fin:
        (userid, stime, etime, l1, t1, l2, t2) = line.strip().split('\t')
        l1 = [int(e) - 1 for e in l1.split(',')]
        l2 = [int(e) - 1 for e in l2.split(',')]
        t1 = [int(e) for e in t1.split(',')]
        t2 = [int(e) for e in t2.split(',')]
        assert(len(l1) == len(t1))
        assert(len(l2) == len(t2))

        nextloc = next_different_loc(l1, t1, l2, t2)
        if nextloc == None:
            continue

        # shrink
        new_l1 = []
        new_t1 = []
        for i in range(len(l1)):
            if i == 0 or l1[i] != l1[i - 1]:
                new_l1.append(l1[i])
                new_t1.append(t1[i])
        l1 = new_l1
        t1 = new_t1

        if seqlimit == None:
            curlimit = len(l1)
        else:
            curlimit = min(seqlimit, len(l1))

        x.append([onehot(e, size) for e in l1[-curlimit:]])
        y.append(onehot(nextloc, size))

        hour = (t1[-1] // 1000 // 60 // 60) % 24
        day = (t1[-1] // 1000 // 60 // 60 // 24) % 7
    fin.close()

    x = np.array(x)
    y = np.array(y)
    return (x, y)

def getxy(tag, size, seqlimit=None):
    x = []
    y = []
    fin = open('../data/trajectory_data/{}-{:03d}'.format(tag, size), 'r')
    for line in fin:
        (userid, stime, etime, l1, t1, l2, t2) = line.strip().split('\t')
        l1 = [int(e) - 1 for e in l1.split(',')]
        l2 = [int(e) - 1 for e in l2.split(',')]
        t1 = [int(e) for e in t1.split(',')]
        t2 = [int(e) for e in t2.split(',')]
        assert(len(l1) == len(t1))
        assert(len(l2) == len(t2))

        # shrink
        new_l1 = []
        new_t1 = []
        for i in range(len(l1)):
            if i == 0 or l1[i] != l1[i - 1]:
                new_l1.append(l1[i])
                new_t1.append(t1[i])
        l1 = new_l1
        t1 = new_t1

        if seqlimit == None:
            curlimit = len(l1)
        else:
            curlimit = min(seqlimit, len(l1))

        x.append([onehot(e, size) for e in l1[-curlimit:]])
        y.append(onehot(0, size))
    fin.close()
    x = np.array(x)
    y = np.array(y)
    return (x, y)

def main(args):
    logout = Logger('reslog.txt')

    # read data
    seqlimit = 100
    train_x, train_y = read_data('train', args.granularity, seqlimit)
    valid_x, valid_y = read_data('validation', args.granularity, seqlimit)
    test_x, test_y = read_data('test', args.granularity, seqlimit)

    logout.print('shape of train data:\t{} {}'.format(train_x.shape, train_y.shape))
    logout.print('shape of valid data:\t{} {}'.format(valid_x.shape, valid_y.shape))
    logout.print('shape of test data:\t{} {}'.format(test_x.shape, test_y.shape))

    # read data
    max_seq_length = np.max([
        np.max([len(e) for e in train_x]),
        np.max([len(e) for e in valid_x]),
        np.max([len(e) for e in test_x])
        ])
    min_seq_length = np.min([
        np.min([len(e) for e in train_x]),
        np.min([len(e) for e in valid_x]),
        np.min([len(e) for e in test_x])
        ])

    logout.print('max_seq_length:\t{}'.format(max_seq_length))
    logout.print('min_seq_length:\t{}'.format(min_seq_length))
    batch_size = 256
    logout.print('batch_size:\t{}'.format(batch_size))

    model = rnn_model.RNN(
        max_seq_length=max_seq_length,
        input_dim=len(train_x[0][0]),
        output_n_vocab=args.granularity,
        num_hidden_layers=1,
        num_hidden_units=256,
        input_embedding=512,
        forget_bias=0.03,
        learning_rate=0.05
        )
    logout.print('{}'.format(model))
    model.create_model()

    last_valid_acc = None
    show_step = 1
    stop_step = 100
    stop_cnt = 0
    best_valid_acc = -1.0
    best_test_acc = -1.0
    best_result = None
    i = 0
    while True:
        batch_x, batch_y = batch(train_x, train_y, size=256)
        model.train_step(batch_x, batch_y)
        if i % show_step == 0:
            train_acc = model.get_accuracy(train_x, train_y)
            valid_acc = model.get_accuracy(valid_x, valid_y)
            test_acc = model.get_accuracy(test_x, test_y)

            log = '{}\tStep: {:04d}, '.format(datetime.datetime.now(), i)
            log += 'train/test/valid acc = {:.4f}/{:.4f}/{:.4f}'.format(
                train_acc, test_acc, valid_acc
                )

            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                best_test_acc = test_acc
                best_result = model.get_prediction(test_x, test_y)

                stop_cnt = 0
                log += '*'
            else:
                stop_cnt += show_step
                log += '\t{}'.format(stop_cnt)

            logout.print(log)
        if stop_cnt >= stop_step:
            break

        i += 1

    logout.print('final result: {:.4f}'.format(best_test_acc))
    logout.close()

    for tag in ['train', 'test', 'validation']:
        print('saving: ' + tag)
        x, y = getxy(tag, args.granularity, seqlimit)
        np.savetxt(
            'logits-{}'.format(tag),
            model.get_logits(x, y),
            delimiter='\t'
            )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--granularity',
        type=int,
        choices=[5, 10, 25, 50, 75, 100],
        help='total number of locations used in the experiment')
    args = parser.parse_args()

    main(args)

