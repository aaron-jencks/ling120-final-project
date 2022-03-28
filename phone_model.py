import json
import pathlib

from sklearn import preprocessing, utils
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List
import argparse
import os
import librosa

from utils.tsv import read_tsv

# logger = set_logger('running_data_boosting_classifier', use_tb_logger=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class AudioDataset(Dataset):
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


class GeneralPerceptron(torch.nn.Module):
    def __init__(self, n_param: int, n_out: int = 1,
                 n_layers: int = 1, layer_contents: List[int] = None, dropout: bool = True,
                 activation_function: torch.nn.Module = torch.nn.Sigmoid):
        super().__init__()
        self.n_param = n_param
        self.n_out = n_out
        self.n_layers = n_layers
        self.layer_contents = ([self.n_param] * self.n_layers) if layer_contents is None else layer_contents
        self.layers = torch.nn.ModuleList()
        self.activation = activation_function
        self.initialized_activation = self.activation()
        self.initialize_layers()
        self.dropout = torch.nn.Dropout(0.5)
        self.do_dropout = dropout

    def initialize_layers(self):
        self.layers = torch.nn.ModuleList()
        self.layers.append(torch.nn.Linear(self.n_param, self.layer_contents[0]))
        for i in range(self.n_layers - 1):
            self.layers.append(torch.nn.Linear(self.layer_contents[i],
                                               self.layer_contents[i + 1]))
        self.layers.append(torch.nn.Linear(self.layer_contents[-1], self.n_out))
        self.initialized_activation = self.activation()

    def forward(self, x):
        current = x
        for layer in self.layers[:-1]:
            current = layer(current)
            if self.do_dropout:
                current = self.dropout(current)
            # print(current.shape)
            current = self.initialized_activation(current)
        current = self.layers[-1](current)
        # print(current.shape)
        # current = self.initialized_activation(current)
        # print(current.shape)
        return current


def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(tqdm(dataloader)):
        # Compute prediction and loss
        pred = model(X.float())
        loss = loss_fn(pred, y.float())

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    #         if batch % 100 == 0:
    #             loss, current = loss.item(), batch * len(X)
    #             print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    return loss.item()


def test_loop(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in tqdm(dataloader):
            pred = model(X.float())
            #             print(pred.shape)
            #             print(pred)
            test_loss += loss_fn(pred, y.float()).item()

    test_loss /= num_batches
    print(f"Test Error: \nAvg loss: {test_loss:>8f} \n")
    return test_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model Trainer for Phoneme Generation')
    parser.add_argument('tsv_file', type=pathlib.Path, help='The TSV file to train from')
    parser.add_argument('clip_dir', type=pathlib.Path, help='The location of the .wav files')
    parser.add_argument('phoneme_dir', type=pathlib.Path,
                        help='The location of the subdirectories of the phoneme clips')
    parser.add_argument('--layer_count', type=int, default=5, help='Number of layers to use in the MLP')
    parser.add_argument('--layer_size', type=int, default=100, help='Number of neurons in the layers to use in the MLP')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='The learning rate')
    parser.add_argument('--batch_size', type=int, default=256, help='The batch size for the data loader')
    parser.add_argument('--epochs', type=int, default=10, help='The number of epochs for training')
    parser.add_argument('--wave_size', type=int, default=-1,
                        help='The output size of the waveform, for if you\'ve ran this before.')
    parser.add_argument('--output', type=pathlib.Path, default=pathlib.Path('./phoneme_model.sav'),
                        help='The file that you would like to save your model in')

    args = parser.parse_args()
    print('Running on {}'.format(device))

    # Determine the largest waveform size
    if args.wave_size < 0:
        sizes = []
        for root, dirs, files in tqdm(os.walk(args.phoneme_dir), desc='Finding largest waveform size'):
            for f in files:
                if f.split('.')[1] == 'wav':
                    w, _ = librosa.load(pathlib.Path(root) / f, mono=True)
                    sizes.append(len(w))
        max_output_size = max(sizes)
    else:
        max_output_size = args.wave_size

    dataset = AudioDataset(args.tsv_file, args.clip_dir, args.phoneme_dir, max_output_size)

    # Version 1 (No Gradient Boosting)
    model = GeneralPerceptron(3, max_output_size, args.layer_count, [args.layer_size] * args.layer_count, True).to(device)
    # Version 2 (Gradient Boosting)
    # model = GradientBoostingClassifier(model, 10, cuda=torch.cuda.is_available())
    # model.set_optimizer('SGD', lr=0.0001)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)

    loss = train_loop(torch.utils.data.DataLoader(dataset, batch_size=args.batch_size), model, criterion, optimizer)
    i = 0
    while loss > 0.01 and i < args.epochs:
        print('{}x{}: Training iteration {}, Loss {}\n'.format(args.layer_count, args.layer_size, i, loss))
        loss = train_loop(torch.utils.data.DataLoader(dataset, batch_size=args.batch_size), model, criterion, optimizer)
        print('Training Error: {}'.format(loss))
        i += 1

    print(test_loop(torch.utils.data.DataLoader(dataset, batch_size=args.batch_size), model, criterion))

    torch.save({'model': model, 'encoder': dataset.enc}, args.output)
