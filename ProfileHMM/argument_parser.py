import argparse
import sys

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
        help='Squence data in FASTA format with for which the \
        HHM Model has to calculate a score. If no data is provieded only the \
        emission_probabilities of the HMM are printed.',
        type=argparse.FileType('r'),
        metavar="FILE")
    parser.add_argument(
        '--out',
        help='Path to output File. Only raw data is printed (Default: stdout).',
        type=argparse.FileType('w'),
        default=sys.stdout,
        metavar='FILE')

    args = parser.parse_args()
    return args.train_data, args.test_data, args.out
