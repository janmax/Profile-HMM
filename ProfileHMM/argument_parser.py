import argparse

def parseme():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train-data',
        help='A MSA in FASTA format to train a Hidden Markov Model',
        type=argparse.FileType('r'),
        required=True,
        metavar='FILE')
    parser.add_argument(
        '--test-data',
        help='Squence data in FASTA format with for which the HHM Model has to calculate a score',
        type=argparse.FileType('r'),
        required=True,
        metavar="FILE")

    args = parser.parse_args()
    return args.train_data, args.test_data
