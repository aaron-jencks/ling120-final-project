import pathlib
from typing import List

import matplotlib.pyplot as plt
import librosa
import soundfile as sf
from tqdm import tqdm

from tsv import TSVEntry


def trim_empty_space(values: List[TSVEntry], recording_locations: pathlib.Path):
    for v in tqdm(values, desc='Trimming empty space'):
        w, sr = librosa.load(recording_locations / v['path'], mono=True)
        fltr = [wt > 0.1 for wt in w]  # TODO figure out what the threshold should be
        sf.write(recording_locations / v['path'], w[fltr], sr)
