import pathlib

import torch
from torch.utils.data import Dataset
import argparse

from src.models.dataset import AudioPhonemeDataset, find_largest_waveform_size
from src.models.generic_model import GeneralPerceptron, train_loop, test_loop


device = 'cuda' if torch.cuda.is_available() else 'cpu'


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
    parser.add_argument('--output', type=pathlib.Path, default=pathlib.Path('../../phoneme_model.sav'),
                        help='The file that you would like to save your model in')

    args = parser.parse_args()
    print('Running on {}'.format(device))

    # Determine the largest waveform size
    if args.wave_size < 0:
        max_output_size = find_largest_waveform_size(args.phoneme_dir)
    else:
        max_output_size = args.wave_size

    dataset = AudioPhonemeDataset(args.tsv_file, args.clip_dir, args.phoneme_dir, max_output_size)

    # Version 1 (No Gradient Boosting)
    model = GeneralPerceptron(3, max_output_size, args.layer_count,
                              [args.layer_size] * args.layer_count, True).to(device)
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
