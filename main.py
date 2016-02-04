from ProfileHMM.ProfileHMM import *

def read(rfile):
    MSA = []
    with rfile as train_data:
        for line in train_data.readlines():
            if line.startswith('>'):
                continue
            MSA.append(np.array(list(line.strip())))
    return np.array(MSA)

traindata, testdata = parseme()

# read drom fasta data and set some basic values
print("[Info] Profile HMM started. Reading train data...")
MSA = read(traindata)

print("[Info] Creating HMM with probabilitys based on given MSA...")
HMM_MSA = HMM(MSA)

print("[Info] HMM ready. Start scoring test data...")
with testdata as full:
    for i, line in enumerate(full.readlines()):
        if line.startswith('>'):
            continue
        print("[{}]\t{}\t{}".format(i // 2 + 1, line[:30], HMM_MSA.viterbi(line)))
print("[Info] We are done. Bye.")
