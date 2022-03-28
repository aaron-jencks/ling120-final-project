import json
import pathlib

from sklearn import preprocessing
import numpy as np
import torch
from torch.utils.data import Dataset
import os
import librosa
from tqdm import tqdm

import settings
from utils.tsv import read_tsv

# logger = set_logger('running_data_boosting_classifier', use_tb_logger=True)
device = 'cuda' if torch.cuda.is_available() and settings.enable_gpu else 'cpu'


def find_largest_waveform_size(phone_dir: pathlib.Path) -> int:
    sizes = []
    for root, dirs, files in tqdm(os.walk(phone_dir), desc='Finding largest waveform size'):
        for f in files:
            if f.split('.')[1] == 'wav':
                w, _ = librosa.load(pathlib.Path(root) / f, mono=True)
                sizes.append(len(w))
    return max(sizes)


class TSVAudioDataset(Dataset):
    def __init__(self, tsv_file: pathlib.Path, clip_dir: pathlib.Path, phoneme_dir: pathlib.Path, output_size: int):
        self.tsv = read_tsv(tsv_file)
        self.clips = clip_dir
        self.phonemes = phoneme_dir
        self.padding_size = output_size

        phones = []
        for root, dirs, files in os.walk(self.phonemes):
            for d in dirs:
                phones.append(d)
            break

        self.enc = preprocessing.LabelEncoder().fit(phones)
        self.space = self.enc.transform(['sp'])[0]

        self.indices = []
        for e in self.tsv:
            phonemes = []
            fname = e['path'].split('.')[0]
            with open(self.clips / (fname + '.json'), 'r') as fp:
                alignment = json.load(fp)
            for w in alignment['words']:
                for name, _, _ in w['phonemes']:
                    phonemes.append(name)
            self.indices.append(len(phonemes))

    def __len__(self):
        return sum(self.indices)


class AudioDataset(TSVAudioDataset):
    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()
        if isinstance(item, int):
            item = [item]

        result = []
        waveforms = []
        for it in item:
            for i, j in enumerate(self.indices):
                if it < j:
                    e = self.tsv[i]
                    phonemes = []
                    fname = e['path'].split('.')[0]
                    with open(self.clips / (fname + '.json'), 'r') as fp:
                        alignment = json.load(fp)
                    for w in alignment['words']:
                        for name, _, _ in w['phonemes']:
                            phonemes.append(name)

                    w, sr = librosa.load(self.phonemes / phonemes[it] / (fname + '_{}.wav'.format(it)), mono=True)
                    if len(w) < self.padding_size:
                        diff = self.padding_size - len(w)
                        w = np.pad(w, (0, diff), 'constant', constant_values=(0, 0))
                    waveforms.append(w)

                    phonemes = self.enc.transform(phonemes)
                    if it == 0:
                        result.append([self.space, phonemes[it], phonemes[it + 1]])
                    elif it == j - 1:
                        result.append([phonemes[it - 1], phonemes[it], self.space])
                    else:
                        result.append([phonemes[it - 1], phonemes[it], phonemes[it + 1]])
                    break
                it -= j

        result = np.squeeze(np.array(result).astype('double'))
        waveforms = np.squeeze(np.array(waveforms, 'double'))
        sample = {'phonemes': torch.from_numpy(result).to(device),
                  'waveforms': torch.from_numpy(waveforms).to(device)}

        return sample['phonemes'], sample['waveforms']


class AudioEncoderDataset(TSVAudioDataset):
    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()
        if isinstance(item, int):
            item = [item]

        waveforms = []
        for it in item:
            for i, j in enumerate(self.indices):
                if it < j:
                    e = self.tsv[i]
                    phonemes = []
                    fname = e['path'].split('.')[0]
                    with open(self.clips / (fname + '.json'), 'r') as fp:
                        alignment = json.load(fp)
                    for w in alignment['words']:
                        for name, _, _ in w['phonemes']:
                            phonemes.append(name)

                    w, sr = librosa.load(self.phonemes / phonemes[it] / (fname + '_{}.wav'.format(it)), mono=True)
                    if len(w) < self.padding_size:
                        diff = self.padding_size - len(w)
                        w = np.pad(w, (0, diff), 'constant', constant_values=(0, 0))
                    waveforms.append(w)
                    break
                it -= j

        waveforms = np.squeeze(np.array(waveforms, 'double'))

        return torch.from_numpy(waveforms).to(device), torch.from_numpy(waveforms).to(device)
