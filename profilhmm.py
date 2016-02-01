from collections import Counter

import numpy as np

acid = 'CGAU'

def read(rfile='../Daten/LSU_train.fasta'):
    msa = []
    with open(rfile) as train_data:
        for line in train_data.readlines():
            if '>' in line:
                continue
            msa.append(np.array(list(line.strip())))
    return np.array(msa)

def count(MSA, buckets='CGAU-'):
    for column in MSA.T:
        bucket = {a : 0 for a in buckets}
        for a in column:
            bucket[a] += 1
        yield bucket

msa = read()
n, length = msa.shape

counts = [count for count in count(msa)]
matches = [count['-'] < n//2 for count in counts]
matches_c = Counter(matches)


M = ['M%d' % j for j in range(matches_c[True])]
D = ['M%d' % j for j in range(matches_c[True])]
I = [-1] + ['M%d' % j for j in range(matches_c[True])]
states = [M, D, I]

a = {}

for state_grp in states:
    for k, state in enumerate(state_grp):
        a[state] = {'M%d' % k: Counter(msa.T[matches][k])}