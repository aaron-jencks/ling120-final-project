import argparse
import pathlib

import torch
import numpy as np
import sounddevice as sd
from tqdm import tqdm

from models.phone_model import GeneralPerceptron

device = 'cuda' if torch.cuda.is_available() else 'cpu'
sd.default.samplerate = 22050  # Default for librosa


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model Trainer for Phoneme Generation')
    parser.add_argument('--model_location', type=pathlib.Path, default=pathlib.Path('phoneme_model.sav'),
                        help='The file where your model is located')

    args = parser.parse_args()

    model_dict = torch.load(args.model_location, map_location=device)
    model: GeneralPerceptron = model_dict['model']
    model.eval()

    encoder = model_dict['encoder']
    space = encoder.transform(['sp'])[0]

    print('Running on {}'.format(device))
    while True:
        print('Try: DH AH0 sp M UW1 N sp IH1 Z sp R IY0 S P AA1 N S AH0 B AH0 L sp F AO1 R sp T AY1 D Z sp AA1 N sp '
              'ER1 TH sp')
        s = input('Enter something to say: ')
        phonemes = encoder.transform(s.split())

        trigrams = [[space, phonemes[0], phonemes[1]]]
        for p in range(len(phonemes) - 2):
            trigrams.append(phonemes[p:p + 3])
        trigrams.append([phonemes[-2], phonemes[-1], space])

        waveforms = model(torch.from_numpy(np.squeeze(np.array(trigrams))).float().to(device)).cpu().detach()
        for w in tqdm(waveforms, desc='Playing phonemes'):
            sd.play(w, blocking=True)
