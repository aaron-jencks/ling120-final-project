import json
import os
import pathlib
import argparse
import warnings
from typing import Tuple, List

from sklearn import preprocessing
from tqdm import tqdm

from src.utils.mp_util import round_robin_map
from src.utils.tsv import read_tsv, TSVEntry, write_tsv

warnings.filterwarnings('ignore', 'PySoundFile failed. Trying audioread instead.')


def collect_phonemes_and_filenames(v: Tuple[str, pathlib.Path, pathlib.Path, preprocessing.LabelEncoder]) -> \
        List[Tuple[str, int, int, pathlib.Path]]:
    fname, clip_name, phoneme_dir, enc = v
    phonemes = []
    try:
        with open(clip_name, 'r') as fp:
            alignment = json.load(fp)

        for w in alignment['words']:
            for it, (name, _, _) in enumerate(w['phonemes']):
                phonemes.append((fname, it, enc.transform([name])[0],
                                 phoneme_dir / name / (fname + '_{}.wav'.format(it))))
    except json.JSONDecodeError:
        return []
    except OSError:
        return []

    return phonemes


def generate_tsv_file(tsv_file_in: pathlib.Path, tsv_file_out: pathlib.Path,
                      clip_dir: pathlib.Path, phoneme_dir: pathlib.Path):
    tsv = read_tsv(tsv_file_in)

    phones = []
    for root, dirs, files in os.walk(phoneme_dir):
        for d in dirs:
            phones.append(d)
        break

    enc = preprocessing.LabelEncoder().fit(phones)
    space = enc.transform(['sp'])[0]

    clips = []
    for e in tqdm(tsv, desc='Generating indices'):
        fname = e['path'].split('.')[0]
        clip_name = clip_dir / (fname + '.json')
        if clip_name.exists():
            clips.append((fname, clip_name, phoneme_dir, enc))

    # indices = round_robin_map(clips, collect_phonemes_and_filenames, tqdm_label='Generating phoneme indices')
    indices = []
    for c in tqdm(clips, desc='Generating phoneme indices'):
        indices.append(collect_phonemes_and_filenames(c))

    fmap = {}

    for index in tqdm(indices, desc='Parsing mp result'):
        for fname, it, pv, pf in index:
            if fname not in fmap:
                fmap[fname] = []
            fmap[fname].append((it, pv, pf))

    indices = []
    for fname in tqdm(fmap, desc='Generating tsv'):
        fmap[fname].sort(key=lambda x: x[0])

        phonemes = fmap[fname]
        for it, (_, p, d) in enumerate(phonemes[1:]):
            _, pp, pd = phonemes[it]
            _, ppp, _ = phonemes[it + 1]

            if it == 0:
                trigram = (space, pp, p)
            elif it == len(phonemes[1:]) - 1:
                trigram = (pp, p, space)
            else:
                trigram = (ppp, pp, p)

            if pd.exists():
                ent = TSVEntry(['previous', 'current', 'next', 'directory'], [*trigram, str(pd)])
                indices.append(ent)

    write_tsv(tsv_file_out, indices)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Processes the phonemes for the audio files')
    parser.add_argument('tsv_file_in', type=pathlib.Path, help='The file containing the list of sample file locations')
    parser.add_argument('tsv_file_out', type=pathlib.Path, help='The file to store the new TSV file in')
    parser.add_argument('clips_dir', type=pathlib.Path, help='The directory containing the sample files')
    parser.add_argument('phonemes_dir', type=pathlib.Path, help='The directory containing the sample phoneme files')

    args = parser.parse_args()
    generate_tsv_file(args.tsv_file_in, args.tsv_file_out, args.clips_dir, args.phonemes_dir)
