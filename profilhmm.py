from collections import Counter
from numpy import array
import numpy as np


def read(rfile='../Daten/LSU_train.fasta'):
    MSA = []
    with open(rfile) as train_data:
        for line in train_data.readlines():
            if '>' in line:
                continue
            MSA.append(np.array(list(line.strip())))
    return np.array(MSA)


def transition(alignment1, alignment2):
    transition = {'M->M' : 1, 'M->D' : 1, 'D->M' : 1, 'D->D' : 1}
    for a1, a2 in zip(alignment1, alignment2):
        if   a1 != '-' and a2 != '-':
            transition['M->M'] += 1
        elif a1 != '-' and a2 == '-':
            transition['M->D'] += 1
        elif a1 == '-' and a2 != '-':
            transition['D->M'] += 1
        else:
            transition['D->D'] += 1
    return transition


def find_next_match(index):
    while match_states[index] == False and index < n - 1:
        index += 1
    return index


def current_states(alignment):
    D = Counter(alignment)['-']
    M = len(alignment) - D
    return M, D


# read drom fasta data and set some basic values
MSA = read('small.txt')
alignments, length = MSA.shape
match_states = [Counter(column)['-'] < alignments//2 for column in MSA.T]

# will be the number of match states
n = Counter(match_states)[True]

M = ['M%d' % k for k in range(1, n + 1)]
D = ['D%d' % k for k in range(1, n + 1)]
I = ['I%d' % k for k in range(1, n + 1)]

# Initialize the transition propabilities
a = {}
for k in range(n):
    a[M[k]] = a[D[k]] = a[I[k]] = {}

# current state and start state == 1. Do not think about it!
k = 0

for i in range(length - 1):
    if match_states[i] == match_states[i + 1] == True:
        trans  = transition(MSA.T[i], MSA.T[i+1])
        M_states, D_states = current_states(MSA.T[i])
        a[M[k]][M[k + 1]] = trans['M->M'] / (M_states + 3)
        a[M[k]][D[k + 1]] = trans['M->D'] / (M_states + 3)
        a[M[k]][I[k]]     =             1 / (M_states + 3)
        a[D[k]][M[k + 1]] = trans['D->M'] / (D_states + 3)
        a[D[k]][D[k + 1]] = trans['D->D'] / (D_states + 3)
        a[D[k]][I[k]]     =             1 / (D_states + 3)
        # these cases are not possible so they get equal distribution
        a[I[k]][M[k + 1]] = 1 / 3
        a[I[k]][D[k + 1]] = 1 / 3
        a[I[k]][I[k]]     = 1 / 3
        k += 1
    elif match_states[i] == True and match_states[i+1] == False:
        next__step = transition(MSA.T[i], MSA.T[i+1])
        next_match = transition(MSA.T[i], MSA.T[find_next_match(i+1)])
        M_states, D_states = current_states(MSA.T[i])
        # sequences with insertions do not jump directly to next match
        a[M[k]][M[k + 1]] = (next_match['M->M'] - next__step['M->M'] + 1) / (M_states + 3)
        a[M[k]][D[k + 1]] =  next_match['M->D'] / (M_states + 3)
        a[M[k]][I[k]]     =  next__step['M->M'] / (M_states + 3)
        # sequences with insertions do not jump directly to next match
        a[D[k]][M[k + 1]] = (next_match['D->M'] - next__step['D->M'] + 1) / (D_states + 3)
        a[D[k]][D[k + 1]] =  next_match['D->D'] / (D_states + 3)
        a[D[k]][I[k]]     =  next__step['D->M'] / (D_states + 3)
        # sequences that have to return from a insertion to a matching state
        a[I[k]][M[k + 1]] =  next__step['M->M'] / (M_states + 3)
        a[I[k]][D[k + 1]] = (1 - a[I[k]][M[k + 1]]) / 2
        a[I[k]][I[k]]     = (1 - a[I[k]][M[k + 1]]) / 2
        k += 1

k = 0
e = {state : 0 for state in M}
for alignment, match in zip(MSA.T, match_states):
    if match:
        acid_count = Counter(alignment)
        acid_total = alignment.size - acid_count['-']
        e[M[k]] = {a : acid_count[a] / acid_total for a in set(acid_count) - {'-'}}
        k += 1

# e2 = {state : {acid : 1 / len(acids) for acid in acids}}

for key in M:
    print(key, e[key], sum(e[key].values()))

for key in I:
    print(key, a[key], sum(a[key].values()))
