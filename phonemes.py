import pathlib
from typing import List
import argparse
import os
import warnings

import librosa
import soundfile as sf
from tqdm import tqdm
import pyfoal
import numpy as np

from src.utils.tsv import TSVEntry, read_tsv
from src.utils.mp_util import round_robin_map
from src.utils.sound_util import seconds_to_index


warnings.filterwarnings('ignore', 'PySoundFile failed. Trying audioread instead.')


class Phoneme:
    def __init__(self, name: str, start: float, stop: float, sample_rate: int, file_length: int):
        self.name = name
        self.start = start
        self.stop = stop
        self.sample_rate = sample_rate
        self.file_length = file_length

    @property
    def start_index(self) -> int:
        i = seconds_to_index(self.start, self.sample_rate)
        if i >= self.file_length:
            i -= 1
        return i

    @property
    def stop_index(self) -> int:
        i = seconds_to_index(self.stop, self.sample_rate)
        if i >= self.file_length:
            i -= 1
        return i


class RecordingSample:
    def __init__(self, e: TSVEntry, loc: pathlib.Path):
        self.entry = e
        self.directory = loc
        self.base_filename = e['path'].split('.')[0]
        self.location = loc / e['path']
        self.output_location = loc / (self.base_filename + '.wav')
        self.alignment_location = loc / (self.base_filename + '.json')
        self.text_location = loc / (self.base_filename + '_text.txt')


def rec_trim_empty_space(v: RecordingSample):
    w, sr = librosa.load(v.location if not v.output_location.exists() else v.output_location, mono=True)
    wi = 0
    ringbuffer = []
    for wi, wt in enumerate(w):
        if abs(wt) > 0.01:
            if len(ringbuffer) >= 100:
                if np.mean(ringbuffer) > -0.01:
                    wi -= len(ringbuffer)
                    break
        if len(ringbuffer) < 500:
            ringbuffer.append(wt)
        else:
            ringbuffer.pop(0)
            ringbuffer.append(wt)

    ringbuffer = []
    we = -1
    for we, wt in enumerate(reversed(w)):
        if abs(wt) > 0.01:
            if len(ringbuffer) >= 100:
                if np.mean(ringbuffer) > -0.01:
                    we -= len(ringbuffer)
                    break
        if len(ringbuffer) < 500:
            ringbuffer.append(wt)
        else:
            ringbuffer.pop(0)
            ringbuffer.append(wt)

    if we <= 0:
        w = w[wi:]
    else:
        w = w[wi:-we]
    sf.write(v.output_location, w, sr)
    return True


def rec_create_forced_alignment(v):
    v, w, sr = v
    alignment = pyfoal.align(v.entry['sentence'], w, sr)
    alignment.save_json(v.alignment_location)
    phonemes = [Phoneme(p.phoneme, p.start(), p.end(), sr, len(w)) for p in alignment.phonemes()]
    pdir = v.directory / 'phonemes'

    if not pdir.exists():
        os.mkdir(pdir)

    for pi, p in enumerate(phonemes):
        psubdir = pdir / p.name
        if not psubdir.exists():
            os.mkdir(psubdir)

        sf.write(psubdir / (v.base_filename + '_{}.wav'.format(pi)), w[p.start_index:p.stop_index], sr)

    return True


def trim_empty_space(values: List[TSVEntry], recording_locations: pathlib.Path):
    samples = list(map(lambda x: RecordingSample(x, recording_locations),
                       tqdm(values, desc='Creating output file locations')))
    round_robin_map(samples, rec_trim_empty_space,
                    1024, 'Trimming empty space')


def create_phoneme_alignment(values: List[TSVEntry], recording_location: pathlib.Path):
    samples = list(map(lambda x: RecordingSample(x, recording_location),
                       tqdm(values, desc='Creating output file locations')))
    for start in tqdm(range(0, len(samples), 1024), desc='Aligning phonemes'):
        subset = samples[start:start + 1024]
        wsr = list(map(lambda x: librosa.load(x.location, mono=True), tqdm(subset, desc='Loading waveforms')))
        zipped = [(s, w[0], w[1]) for s, w in zip(subset, wsr)]
        round_robin_map(zipped, rec_create_forced_alignment, 10, 'Aligning chunk')


def create_text_files(values: List[TSVEntry], recording_location: pathlib.Path):
    for v in tqdm(values, desc='Generating text files for pyfoal'):
        fname = v['path'].split('.')[0]
        with open(recording_location / (fname + '_text.txt'), 'w+') as fp:
            fp.write(v['sentence'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Processes the phonemes for the audio files')
    parser.add_argument('tsv_file', type=pathlib.Path, help='The file containing the list of sample file locations')
    parser.add_argument('clips_dir', type=pathlib.Path, help='The directory containing the sample files')
    parser.add_argument('--trim', action='store_true')
    parser.add_argument('--text', action='store_true')

    args = parser.parse_args()
    records = read_tsv(args.tsv_file)
    if args.trim:
        trim_empty_space(records, args.clips_dir)

    if args.text:
        create_text_files(records, args.clips_dir)

    create_phoneme_alignment(records, args.clips_dir)
