import numpy as np
import collections
import sys
import os

def read(tag, group):
    ifolder = '../data/trajectory_data'

    data = []
    fin = open(os.path.join(ifolder, '{}-{:03d}'.format(tag, group)), 'r')
    for line in fin:
    	(userid, _, _, l1, t1, l2, t2) = line.strip().split('\t')
    	l1 = [int(e) - 1 for e in l1.split(',')]
    	l2 = [int(e) - 1 for e in l2.split(',')]
    	t1 = [int(e) for e in t1.split(',')]
    	t2 = [int(e) for e in t2.split(',')]
    	data.append((userid, l1, t1, l2, t2))
    fin.close()

    return data

def markov(seqs):
    transit = [[0 for _ in range(int(sys.argv[1]))] for _ in range(int(sys.argv[1]))]
    for seq in seqs:
    	for i in range(len(seq) - 1):
    		if seq[i] == seq[i + 1]:
    			continue
    		transit[seq[i]][seq[i + 1]] += 1
    return transit

def next_longest_loc(k, l1, t1, l2, t2): # List[int]
    assert(len(l1) == len(t1))
    assert(len(l2) == len(t2))

    l = [] # l for location
    d = [] # d for duration
    cur = None
    start = None
    for i in range(len(l2)):
    	if cur is None:
    		cur = l2[i]
    		start = t2[i]
    	elif l2[i] != cur:
    		l.append(cur)
    		d.append(t2[i] - start)
    		cur = l2[i]
    		start = t2[i]
    if cur != None:
    	l.append(cur)
    	d.append(t2[-1] - start)

    d = [d[i] for i in range(len(l)) if l[i] != l1[-1]]
    l = [l[i] for i in range(len(l)) if l[i] != l1[-1]]
    if len(l) < k:
    	return None, None

    l = l[:k]
    d = d[:k]
    idx = np.argmax(d)
    return l[idx], d[idx]

def main():
    assert(len(sys.argv) == 3)

    train = read('train', int(sys.argv[1]))
    valid = read('validation', int(sys.argv[1]))
    test = read('test', int(sys.argv[1]))
    k = int(sys.argv[2])

    model = markov([e[1] + e[3] for e in train + valid])
    
    y = []
    p = []
    for (userid, l1, t1, l2, t2) in test:
    	truth, duration = next_longest_loc(k, l1, t1, l2, t2)
    	if truth is not None:
    		y.append(truth)
    		p.append(np.argmax(model[l1[-1]]))
    cor = sum(x == y for (x, y) in zip(y, p))
    print('# Correct:', cor)
    print('# Total:', len(y))
    print('# Accuracy:', cor / len(y))

if __name__ == '__main__':
    main()
