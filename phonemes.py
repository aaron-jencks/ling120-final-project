import pathlib
from typing import List
import argparse

import matplotlib.pyplot as plt
import librosa
import soundfile as sf
from tqdm import tqdm

from tsv import TSVEntry, read_tsv
from mp_util import round_robin_map


class RecordingSample:
    def __init__(self, e: TSVEntry, loc: pathlib.Path):
        self.entry = e
        self.location = loc / e['path']
        self.output_location = loc / (e['path'].split('.')[0] + '.wav')


def rec_trim_empty_space(v: RecordingSample):
    w, sr = librosa.load(v.location, mono=True)
    wi = 0
    for wi, wt in enumerate(w):
        if abs(wt) > 0.01:
            break
    we = -1
    for we, wt in enumerate(reversed(w)):
        if abs(wt) > 0.01:
            break
    if we <= 0:
        w = w[wi:]
    else:
        w = w[wi:-we]
    sf.write(v.output_location, w, sr)
    return True


def trim_empty_space(values: List[TSVEntry], recording_locations: pathlib.Path):
    round_robin_map(list(map(lambda x: RecordingSample(x, recording_locations),
                             tqdm(values, desc='Creating output file locations'))), rec_trim_empty_space,
                    1024, 'Trimming empty space')
    # values = list(map(lambda x: RecordingSample(x, recording_locations),
    #                   tqdm(values, desc='Creating output file locations')))
    # waves = []
    # for v in tqdm(values, desc='Reading in mp3s'):
    #     waves.append(librosa.load(v.location, mono=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Processes the phonemes for the audio files')
    parser.add_argument('tsv_file', type=pathlib.Path, help='The file containing the list of sample file locations')
    parser.add_argument('clips_dir', type=pathlib.Path, help='The directory containing the sample files')
    parser.add_argument('--trim', action='store_true')

    args = parser.parse_args()
    records = read_tsv(args.tsv_file)
    if args.trim:
        trim_empty_space(records, args.clips_dir)
