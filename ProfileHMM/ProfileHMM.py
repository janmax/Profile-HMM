# external Libraries
try:
    from numba import jit
    import numpy as np
except ImportError:
    print('[Error] Seems you do not have numpy or numba. Too bad - Exiting...')

# python modules
from collections import Counter
from math import log

# package imports
from .argument_parser import parseme
from .tictoc import *

class HMM:
    """docstring for HMM"""
    def __init__(self, MSA):

        # Getting relevant properties, borders, sizes, etc.
        self.MSA = MSA
        self.MSAbool = self.boolify(MSA)
        self.MSAchar = [char for char in np.unique(MSA) if char != '-']
        self.alignment_number, self.alignment_length = MSA.shape
        self.match_states = self.calc_match_states()
        self.n = Counter(self.match_states)[True]

        # Calculating/guessing the probabilities
        self.transmissions = self.calc_transmissions()
        self.emissions_from_M, self.emissions_from_I = self.calc_emissons()

    def viterbi(self, testdata):
        char_to_int = {c: i for i, c in enumerate(self.MSAchar)}
        testdata    = np.array([char_to_int[c] for c in testdata.strip()])

        return self._viterbi(
            testdata,
            self.emissions_from_M,
            self.emissions_from_I,
            self.transmissions,
            self.n,
            1/len(self.MSAchar)
        )

    @staticmethod
    @jit(nopython=True)
    def _viterbi(x, e_M, e_I, a, L, q):
        """ Straight forward implementation of the Viterbi algorithm as described
        in Durbin et. al. [109] Figure 5.1.
        Everything here is based on 2D arrays and therefore maybe a bit cryptic,
        but otherwise it would not be possible to compile the part of the
        implementation with numba and llvmlite. Performance gain: 100x """
        N = x.size

        V_M = np.zeros((N, L+1))
        V_I = np.zeros((N, L+1))
        V_D = np.zeros((N, L+1))

        for i in range(N):
            for j in range(1, L+1):
                V_M[i, j] = log(e_M[x[i]][j-1] / q) + \
                    max(V_M[i-1][j-1] + log(a[0][j-1]), # M->M
                        V_I[i-1][j-1] + log(a[1][j-1]), # M->D
                        V_D[i-1][j-1] + log(a[2][j-1])) # M->I

                V_I[i, j] = log(e_I[x[i]][j-1] / q) + \
                    max(V_M[i-1][j] + log(a[3][j]), # I->M
                        V_I[i-1][j] + log(a[4][j]), # I->I
                        V_D[i-1][j] + log(a[5][j])) # I->D

                V_D[i, j] = max(V_M[i][j-1] + log(a[6][j-1]), # D->M
                                V_I[i][j-1] + log(a[7][j-1]), # D->D
                                V_D[i][j-1] + log(a[8][j-1])) # D->I

        return max(V_M[N-1, L], V_I[N-1, L], V_D[N-1, L])

    def score(self, viterbi_scores):
        return self._score(np.array(viterbi_scores))

    @staticmethod
    @jit(nopython=True)
    def _score(viterbi_scores, threshold=0.8):
        return viterbi_scores > np.amin(viterbi_scores) + \
            abs(np.amax(viterbi_scores) - np.amin(viterbi_scores)) * threshold

    def calc_match_states(self):
        # Applying the 50% rule
        return [Counter(column)['-'] < self.alignment_number//2 for column in self.MSA.T]

    def equal_parts(self, l, n):
        # Yield successive n-sized chunks from l.
        for i in range(0, len(l), n):
            yield l[i:i+n]

    @staticmethod
    @jit
    def boolify(array2d):
        boolarray = np.ndarray(shape=array2d.shape)
        boolarray = array2d != '-'
        return boolarray

    def calc_emissons(self):
        emissions_from_M = {char: np.zeros(self.n) for char in self.MSAchar}
        emissions_from_I = {char: np.zeros(self.n+1) for char in self.MSAchar}

        # No magic here. Just two counter: one for insert states one for
        # match states. Here my implementation is a bit different from Durbin's
        match_index = 0
        for alignment, match in zip(self.MSA.T, self.match_states):
            char_count = Counter(alignment)
            if match:
                for char in self.MSAchar:
                    emissions_from_M[char][match_index] = char_count[char]
                match_index += 1
            if not match:
                for char in self.MSAchar:
                    emissions_from_I[char][match_index] += char_count[char]

        # add one for Laplaces rule
        for c in self.MSAchar:
            emissions_from_M[c] += np.ones(self.n)
            emissions_from_I[c] += np.ones(self.n+1)

        # Total count per column
        M_sum = np.sum(list(emissions_from_M.values()), axis=0)
        I_sum = np.sum(list(emissions_from_I.values()), axis=0)

        # Calculate probabilities
        for c in self.MSAchar:
            emissions_from_M[c] /= M_sum
            emissions_from_I[c] /= I_sum

        # return 2D arrays for performance
        return \
            np.vstack(emissions_from_M[c] for c in self.MSAchar), \
            np.vstack(emissions_from_I[c] for c in self.MSAchar)

    def calc_transmissions(self):

        # these are all the transmissions we want to observe
        transmission_list = [
            'M->M', 'M->D', 'M->I', 'I->M', 'I->I', 'I->D', 'D->M', 'D->D', 'D->I']
        transmissions = {t: np.zeros(self.n+1) for t in transmission_list}

        # setting the start state
        first_row_char_count = Counter(self.MSAbool.T[0])
        if self.match_states[0]:
            transmissions['M->M'][0] = first_row_char_count[True]
            transmissions['M->D'][0] = first_row_char_count[False]
            transmissions['M->I'][0] = 0
        else:
            transmissions['I->M'][0] = first_row_char_count[True]
            transmissions['I->I'][0] = first_row_char_count[False]

        # iterate over the MSA and count state transmissions. Maybe a bit
        # confusing since there is an index as well as iterators.
        match_index = 1
        for i, (alignment, alignment_next, match, match_next) in \
                enumerate(
                zip(self.MSAbool.T[:-1],    self.MSAbool.T[1:],
                    self.match_states[:-1], self.match_states[1:])):
            transmission_count = Counter(zip(alignment, alignment_next))

            # we go from match to match. nothing spectacular here
            if match and match_next:
                for booltrans, transm in zip(
                    [(True, True), (True, False), (False, True), (False, False)],
                    ['M->M',        'M->D',        'D->M',        'D->D']):
                    transmissions[transm][match_index] = transmission_count[booltrans]
                match_index += 1

            # we enter an insert state. count how many states go directly to the
            # next match state
            if match and not match_next:
                next_match = self.match_states.index(True, i)
                empty_til_next = [not any(rest)for rest in self.MSAbool[:, i+2:next_match]]
                transmission_count = Counter(zip(alignment_next, empty_til_next))
                for booltrans, transm in zip(
                    [(True, True), (True, False), (False, True), (False, False)],
                    ['M->M',        'M->I',        'D->M',        'D->I']):
                    transmissions[transm][match_index] += transmission_count[booltrans]

            # just counting the likelihood of staying in an insert state or
            # exiting it
            if not match and not match_next:
                for booltrans, transm in zip(
                    [(True, True), (True, False)],
                    ['I->I',        'I->M']):
                    transmissions[transm][match_index] += transmission_count[booltrans]

            # returning from insert state to match state don't forget to
            # increase the match index
            if not match and match_next:
                for booltrans, transm in zip(
                    [(True, True), (True, False)],
                    ['I->M',        'I->D']):
                    transmissions[transm][match_index] += transmission_count[booltrans]
                match_index += 1

            # this is the end. Count what happens till the end an finish up.
            if match_index == self.n:
                if i + 2 == self.alignment_length:
                    transmissions['M->M'][match_index] = self.alignment_number
                else:
                    empty_til_end = [not any(rest) for rest in self.MSAbool[:, i+2:]]
                    transmission_count = Counter(zip(alignment_next, empty_til_end))
                    for booltrans, transm in zip(
                        [(True, True), (True, False), (False, True), (False, False)],
                        ['M->M',        'M->I',        'D->M',        'D->I']):
                        transmissions[transm][
                            match_index] = transmission_count[booltrans]
                    transmissions['I->M'][match_index] = self.alignment_number - \
                        transmissions['M->M'][match_index]
                break

        # add one for Laplaces rule
        for t in transmission_list:
            transmissions[t] += np.ones(self.n+1)

        # calculate probabilities based on occurrences
        for t1, t2, t3 in self.equal_parts(transmission_list, 3):
            abs_occurrence = transmissions[t1] + \
                transmissions[t2] + transmissions[t3]
            transmissions[t1] /= abs_occurrence
            transmissions[t2] /= abs_occurrence
            transmissions[t3] /= abs_occurrence

        # return everything as a 2D array for performance
        return np.vstack(transmissions[t] for t in transmission_list)
