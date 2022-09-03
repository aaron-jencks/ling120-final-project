import pathlib
import argparse
import warnings

import numpy as np

from src.models.dataset import find_largest_waveform_size, AudioFileWindowDataset
from src.utils.tsv import TSVEntry, append_tsv
from src.utils.mp_util import round_robin_map


def parse_waveform_segment(tup):
    cols, fn, wfm, start, stop = tup
    segment = wfm[start:stop]
    fft = np.fft.rfft(segment).tolist()
    entry_data = [fn, start, stop, fft, np.sqrt(sum(map(lambda x: x * x, fft)))]
    return TSVEntry(cols, list(map(str, entry_data)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='KNN Parser for the dataset to analyze phones')
    parser.add_argument('tsv_file', type=pathlib.Path, help='The TSV file to train from')
    parser.add_argument('clip_dir', type=pathlib.Path, help='The location of the .wav files')
    parser.add_argument('--wave_size', type=int, default=-1,
                        help='The output size of the waveform, for if you\'ve ran this before.')
    parser.add_argument('--window_size', type=int, default=20000,
                        help='The output size of the window to use for the fft.')
    parser.add_argument('--output', type=pathlib.Path, default=pathlib.Path('./knn.tsv'),
                        help='The file that you would like to save the KNN in')

    args = parser.parse_args()
    tsv_columns = [
        'filename',
        'start',
        'stop',
        'fft',
        'distance'
    ]

    if args.wave_size < 0:
        max_output_size = find_largest_waveform_size(args.phoneme_dir)
    else:
        max_output_size = args.wave_size

    dataset = AudioFileWindowDataset(args.tsv_file, args.clip_dir, max_output_size, False)

    warnings.filterwarnings('ignore')

    args.output.touch()

    for waveform, sr, fname in dataset:
        print(f'parsing {fname} with {len(waveform)} values and sample rate of {sr}')
        windows = [(tsv_columns, fname, waveform, window, window + args.window_size)
                   for window in range(len(waveform) - args.window_size)]
        # ffts = map(parse_waveform_segment, tqdm(windows, desc='generating ffts'))
        tsv_entries = round_robin_map(windows, parse_waveform_segment)
        append_tsv(args.output, tsv_entries)
