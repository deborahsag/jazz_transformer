import argparse
import pickle
import random
import sys
from copy import deepcopy

parser = argparse.ArgumentParser()

parser.add_argument('-pitch_change', type=int, help="number of tones to be transposed up and down the scale")
parser.add_argument('-out_dir', default='data', help="output file directory")

args = parser.parse_args()

sys.path.append('./transformer_xl/')
sys.path.append('./src/')

from build_vocab import Vocab


def pitch_augmentation_random(training_seqs, word2event, event2word, pitchaug_range=3):
    pitchaug_range = [x for x in range(-pitchaug_range, pitchaug_range + 1)]
    augmented_seqs = []
    for sequence in training_seqs:
        seq = deepcopy(sequence)
        pitch_change = random.choice(pitchaug_range)
        for i, ev in enumerate(seq):
            #  event_id = 21 -> Note-On_21 : the lowest pitch on piano
            if 'Note-On' in word2event[ev] and ev >= 21:
                seq[i] += pitch_change
            if 'Chord-Tone' in word2event[ev]:
                seq[i] += pitch_change
                # prevent pitch shift out of range
                if seq[i] > event2word['Chord-Tone_B']:
                    seq[i] -= 12
                elif seq[i] < event2word['Chord-Tone_C']:
                    seq[i] += 12
            if 'Chord-Slash' in word2event[ev]:
                seq[i] += pitch_change
                # prevent pitch shift out of range
                if seq[i] > event2word['Chord-Slash_B']:
                    seq[i] -= 12
                elif seq[i] < event2word['Chord-Slash_C']:
                    seq[i] += 12
        augmented_seqs.append(seq)
    return augmented_seqs


def pitch_augmentation_fixed(training_seqs, word2event, event2word, pitchaug_range=3):
    pitchaug_range = [-pitchaug_range, pitchaug_range]
    augmented_seqs = []
    for sequence in training_seqs:
        seq = deepcopy(sequence)
        for pitch_change in pitchaug_range:
            for i, ev in enumerate(seq):
                #  event_id = 21 -> Note-On_21 : the lowest pitch on piano
                if 'Note-On' in word2event[ev] and ev >= 21:
                    seq[i] += pitch_change
                if 'Chord-Tone' in word2event[ev]:
                    seq[i] += pitch_change
                    # prevent pitch shift out of range
                    if seq[i] > event2word['Chord-Tone_B']:
                        seq[i] -= 12
                    elif seq[i] < event2word['Chord-Tone_C']:
                        seq[i] += 12
                if 'Chord-Slash' in word2event[ev]:
                    seq[i] += pitch_change
                    # prevent pitch shift out of range
                    if seq[i] > event2word['Chord-Slash_B']:
                        seq[i] -= 12
                    elif seq[i] < event2word['Chord-Slash_C']:
                        seq[i] += 12
            augmented_seqs.append(seq)
    return augmented_seqs


def main():
    # load dictionary
    # generated from build_vocab.py
    vocab = pickle.load(open('pickles/remi_wstruct_vocab.pkl', 'rb'))
    event2word, word2event = vocab.event2idx, vocab.idx2event

    # load train data
    # training_seqs_final.pkl : all songs' remi format
    training_data_file = "data/training_seqs_struct_new_final.pkl"
    training_seqs = pickle.load(open(training_data_file, 'rb'))

    pitch_range = args.pitch_change

    augmented_seqs = pitch_augmentation_fixed(training_seqs, word2event, event2word, pitch_range)

    outfile = f'data/augmented_{pitch_range}.pkl'
    pickle.dump(augmented_seqs, open(outfile, 'wb'))


if __name__ == '__main__':
    main()
