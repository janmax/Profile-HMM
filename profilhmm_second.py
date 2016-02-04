from collections import Counter
from numpy import array
from math import log
from tictoc import *
import numpy as np
np.set_printoptions(linewidth=100, precision=4)


def read(rfile='../Daten/LSU_train.fasta'):
    MSA = []
    with open(rfile) as train_data:
        for line in train_data.readlines():
            if line.startswith('>'):
                continue
            MSA.append(np.array(list(line.strip())))
    return np.array(MSA)


def equal_parts(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i+n]


def boolify(array2d):
    boolarray = np.ndarray(shape=array2d.shape)
    boolarray = array2d != '-'
    return boolarray

# read drom fasta data and set some basic values
MSA = read()
# MSA = read('small.txt')
MSAbool = boolify(MSA)
MSAchar = [char for char in np.unique(MSA) if char != '-']

alignment_number, alignment_length = MSA.shape
match_states = [Counter(column)['-'] < alignment_number//2 for column in MSA.T]
n = Counter(match_states)[True]

emissions_from_M = {char: np.zeros(n) for char in MSAchar}
emissions_from_I = {char: np.zeros(n+1) for char in MSAchar}

match_index = 0
for alignment, match in zip(MSA.T, match_states):
    char_count = Counter(alignment)
    if match:
        for char in MSAchar:
            emissions_from_M[char][match_index] = char_count[char]
        match_index += 1
    if not match:
        for char in MSAchar:
            emissions_from_I[char][match_index] += char_count[char]

transmission_list = [
    'M->M', 'M->D', 'M->I', 'I->M', 'I->I', 'I->D', 'D->M', 'D->D', 'D->I']
transmissions = {t: np.zeros(n+1) for t in transmission_list}

first_row_char_count = Counter(MSAbool.T[0])
if match_states[0]:
    transmissions['M->M'][0] = first_row_char_count[True]
    transmissions['M->D'][0] = first_row_char_count[False]
    transmissions['M->I'][0] = 0
else:
    transmissions['I->M'][0] = first_row_char_count[True]
    transmissions['I->I'][0] = first_row_char_count[False]

match_index = 1
for i, (alignment, alignment_next, match, match_next) in \
        enumerate(
        zip(MSAbool.T[:-1], MSAbool.T[1:],
            match_states[:-1], match_states[1:])):
    transmission_count = Counter(zip(alignment, alignment_next))
    if match and match_next:
        for booltrans, transm in zip([(True, True), (True, False), (False, True), (False, False)], ['M->M', 'M->D', 'D->M', 'D->D']):
            transmissions[transm][match_index] = transmission_count[booltrans]
        match_index += 1
    if match and not match_next:
        next_match = match_states.index(True, i)
        empty_til_next = [not any(rest) for rest in MSAbool[:, i+2:next_match]]
        transmission_count = Counter(zip(alignment_next, empty_til_next))
        for booltrans, transm in zip([(True, True), (True, False), (False, True), (False, False)], ['M->M', 'M->I', 'D->M', 'D->I']):
            transmissions[transm][match_index] += transmission_count[booltrans]
    if not match and not match_next:
        for booltrans, transm in zip([(True, True), (True, False)], ['I->I', 'I->M']):
            transmissions[transm][match_index] += transmission_count[booltrans]
    if not match and match_next:
        for booltrans, transm in zip([(True, True), (True, False)], ['I->M', 'I->D']):
            transmissions[transm][match_index] += transmission_count[booltrans]
        transmissions['M->M'][match_index] = alignment_number - \
            transmissions['M->I'][match_index]
        match_index += 1
    if match_index == n:
        if i + 2 == alignment_length:
            transmissions['M->M'][match_index] = alignment_number
        else:
            empty_til_end = [not any(rest) for rest in MSAbool[:, i+2:]]
            transmission_count = Counter(zip(alignment_next, empty_til_end))
            for booltrans, transm in zip([(True, True), (True, False), (False, True), (False, False)], ['M->M', 'M->I', 'D->M', 'D->I']):
                transmissions[transm][match_index] = transmission_count[booltrans]
            transmissions['I->M'][match_index] = alignment_number - \
                transmissions['M->M'][match_index]
        break

# add one for laplaces rule
for t in transmission_list:
    transmissions[t] += np.ones(n+1)

for t1, t2, t3 in equal_parts(transmission_list, 3):
    abs_occurence = transmissions[t1] + transmissions[t2] + transmissions[t3]
    transmissions[t1] /= abs_occurence
    transmissions[t2] /= abs_occurence
    transmissions[t3] /= abs_occurence


for c in MSAchar:
    emissions_from_M[c] += np.ones(n)
    emissions_from_I[c] += np.ones(n+1)

M_sum = np.sum(list(emissions_from_M.values()), axis=0)
I_sum = np.sum(list(emissions_from_I.values()), axis=0)

# print(np.arange(n))
for c in MSAchar:
    emissions_from_M[c] /= M_sum
    emissions_from_I[c] /= I_sum

    # print(c, emissions_from_M[c])
    # print(c, emissions_from_I[c])

for transm in transmission_list:
    if transmissions[transm][transmissions[transm] < 0].size:
        print(transmissions[transm][transmissions[transm] < 0])
        print(transm)


def viterbi(x, e_M, e_I, a, L, q):
    x = np.array(list(x))
    N = x.size

    V_M = np.zeros((N, L+1))
    V_I = np.zeros((N, L+1))
    V_D = np.zeros((N, L+1))

    for i in range(N):
        for j in range(1, L+1):
            # print(i,j)
            # print(a['M->M'][j-1])
            V_M[i, j] = log(e_M[x[i]][j-1] / q) + \
                max(V_M[i-1][j-1] + log(a['M->M'][j-1]),
                    V_I[i-1][j-1] + log(a['I->M'][j-1]),
                    V_D[i-1][j-1] + log(a['D->M'][j-1]))

            V_I[i, j] = log(e_I[x[i]][j-1] / q) + \
                max(V_M[i-1][j] + log(a['M->I'][j]),
                    V_I[i-1][j] + log(a['I->I'][j]),
                    V_D[i-1][j] + log(a['D->I'][j]))

            V_D[i, j] = max(V_M[i][j-1] + log(a['M->D'][j-1]),
                            V_I[i][j-1] + log(a['I->D'][j-1]),
                            V_D[i][j-1] + log(a['D->D'][j-1]))

    return V_M[N-1, L]

with open('../Daten/LSU_full_test.fasta') as full:
    for line in full.readlines():
        if line.startswith('>'):
            continue
        print(viterbi(line, emissions_from_M, emissions_from_I,
                      transmissions, n, 1/len(char_count)))
