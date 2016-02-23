from ProfileHMM.ProfileHMM import *
from multiprocessing import Pool, TimeoutError
from sys import exit

np.set_printoptions(linewidth=14000000, precision=4, threshold=1000000)


def _plot(y):
    import matplotlib.pyplot as plt
    for data, c, marker in zip(y, 'ACGU', 'xo.v'):
        data = data[:100]
        plt.plot(np.arange(len(data)), data, label=c, marker=marker, ls='None')

    plt.xlabel('Match state number')
    plt.ylabel('Propabilities')
    plt.legend(loc='upper right')
    # plt.yscale('log')

    plt.show()


def read(rfile):
    MSA = []
    with rfile as train_data:
        for line in train_data.readlines():
            if line.startswith('>'):
                continue
            MSA.append(np.array(list(line.strip())))
    return np.array(MSA)


def testdata_iter(testdata):
    for line in testdata.readlines():
        if line.startswith('>'):
            continue
        yield line
    testdata.seek(0)

traindata, testdata, out = parseme()

# read drom fasta data and set some basic values
print("[Info] Profile HMM started. Reading train data...")
MSA = read(traindata)

print("[Info] Creating HMM with probabilities based on given MSA...")
HMM_MSA = HMM(MSA)

print("[Info] HMM ready. ")
if not testdata:
    print(
        "[Info] No testdata provided. Just printing probabilities to:", out.name)
    print(HMM_MSA.emissions_from_M, file=out)
    print("[Info] We are done. Bye.")
    exit()

print("[Info] Start scoring test data...")


index = []
scores = []
seq_start = [line[:30] for line in testdata_iter(testdata)]
num_lines = sum(1 for line in testdata) // 2 - 1
testdata.seek(0)

with Pool() as pool:
    for i, score in enumerate(
            pool.imap(HMM_MSA.viterbi, testdata_iter(testdata))):
        print('[{} of {}] in progress..'.format(i, num_lines), end='\r')
        index.append(i)
        scores.append(score)
    print()

print('[Info] Scoring completed')

positiv = np.array(HMM_MSA.score(scores), dtype=int)

for i, start, score, positiv in zip(index, seq_start, scores, positiv):
    print("{}\t{}\t{}\t{}".format(i, start, score, positiv), file=out)

print("[Info] We are done. Bye.")
