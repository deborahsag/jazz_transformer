from glob import glob
import numpy as np
import pandas as pd
import regex as re
import os, sys, pickle , argparse
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

sys.path.append('./src/')
from midi_decoder import convert_events_to_midi
from build_vocab import Vocab
from chord_processor import ChordProcessor

sys.path.append('./transformer_xl/')
from model_aug import TransformerXL

# which gpu to use
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser()
parser.add_argument('--model', help="model name for inference (default: the downloaded ckpt via ``download_model.sh``)", default="ckpt/jazz-trsfmr-B-loss0.25")
parser.add_argument('output_midi', help="the output midi directory")
parser.add_argument('--temp', help="softmax sampling temperature (default: 1.2)", type=float, default=1.2)
parser.add_argument('--n_bars', help="# bars to generate (default: 32)", type=int, default=32)
parser.add_argument('--n_samples', help="# samples to generate (default: 1)", type=int, default=1)
parser.add_argument('--struct_csv', help="(optional) directory for csv output for generated structure-related events", default=False)
args = parser.parse_args()


def seq_to_csv(seq, word2event, out_csv):
    placeholder = np.empty( (len(seq), 2) )
    df_out = pd.DataFrame(placeholder, columns=['EVENT', 'ENCODING'])

    for i, ev in enumerate(seq):
        df_out.loc[i] = [word2event[ev], int(ev)]

    df_out.to_csv(out_csv, encoding='utf-8', index=False)

    return


def get_checkpoint(model_dir):
    path = os.path.join(model_dir, "checkpoint")
    ckpt_file = open(path, 'r')
    line = ckpt_file.readline()
    ckpt_file.close()
    ckpt = re.findall('"(.*)"', line)
    return os.path.join(model_dir, ckpt[0])

if __name__ == '__main__':
    # load dictionary
    vocab = pickle.load(open('pickles/remi_wstruct_vocab.pkl', 'rb'))
    event2word, word2event = vocab.event2idx, vocab.idx2event
    out_midi_dir = args.output_midi
    out_csv_dir = args.struct_csv

    if not out_midi_dir == "":
        if not os.path.exists(out_midi_dir):
            os.makedirs(out_midi_dir)

    checkpoint = get_checkpoint(args.model)

    # declare model
    model = TransformerXL(
        event2word=event2word,
        word2event=word2event,
        checkpoint=checkpoint,
        is_training=False
    )

    for i in range(args.n_samples):
        # inference
        # recommended temperature = 1.2
        word_seq = model.inference(
            n_bars=args.n_bars,
            strategies=['temperature', 'nucleus'],
            params={'t': args.temp, 'p': 0.9},
            use_structure=True
        )

        events = [ word2event[w] for w in word_seq ]
        print ("First 20 events: {}".format(events[:20]))
        chord_processor = pickle.load(open('pickles/chord_processor.pkl', 'rb'))

        if out_csv_dir:
            print('struct csv will be written to:', out_csv_dir)

        midi_name = "jazz-sample-" + str(i) + ".midi"
        out_midi_file = os.path.join(out_midi_dir, midi_name)
        try:
            if out_csv_dir:
                csv_name = "jazz-sample-" + str(i) + ".csv"
                out_struct_csv_file = os.path.join(out_csv_dir, csv_name)
                convert_events_to_midi(events, out_midi_file, chord_processor, use_structure=True, output_struct_csv=out_struct_csv_file)
            else:
                convert_events_to_midi(events, out_midi_file, chord_processor)
            event_file = out_midi_file.replace(os.path.splitext(out_midi_file)[-1], '.csv')
            print ('generated event sequence will be written to:', event_file)
            seq_to_csv(word_seq, word2event, event_file)
        except Exception as e:
            print ('error occurred when converting to', out_midi_file)
            print (e)

    # close
    model.close()