from math import log

def viterbi(obs, states, start_p, trans_p, emit_p):
    V = [start_p]
    path = {state: [state] for state in states}

    # Run Viterbi for t > 0
    for t in range(1, len(obs)):
        V.append({})
        newpath = {}

        for y in states:
            (prob, state) = max(
                (V[t-1][y0] * trans_p[y0][y] * emit_p[y][obs[t]], y0) for y0 in states)
            V[t][y] = prob
            newpath[y] = path[state] + [y]

        # don't need to remember the old paths
        path = newpath

    # return the most likely sequence over the given time frame (backtrace)
    n = len(obs) - 1

    (prob, state) = max((V[n][y], y) for y in states)

    return (prob, path[state])


def main():

    # read the generated rolls
    with open('../Daten/Casino.txt') as casinof:
        rolls, die, check = casinof.read().split('\n')

    # create the HHM
    states = ('Fair', 'Loaded')

    observations = (i for i in range(1, 7))

    start_probability = {'Fair': 0.95, 'Loaded': 0.05}

    transition_probability = {
        'Fair':   {'Fair': 0.95, 'Loaded': 0.05},
        'Loaded': {'Fair': 0.1,  'Loaded': 0.9}
    }

    emission_probability = {
        'Fair':   {str(i): 1/6 for i in range(1, 7)},
        'Loaded': {**{str(i): 0.1 for i in range(1, 6)}, **{'6': 0.5}}
    }

    # calculate the viterbi path
    _, path = viterbi(
        rolls,
        states,
        start_probability,
        transition_probability,
        emission_probability
    )

    # print the the output
    last = 0
    LINE_LENGTH = 50
    for i in range(LINE_LENGTH, len(path), LINE_LENGTH):
        print('rolls:', rolls[last:i])
        print('dice: ', die[last:i])
        print('viter:', ''.join(s[0] for s in path)[last:i])
        print('check:', check[last:i])
        print()
        last = i
    else:
        print('rolls:', rolls[last:])
        print('dice: ', die[last:])
        print('viter:', ''.join(s[0] for s in path)[last:])
        print('check:', check[last:])

main()
