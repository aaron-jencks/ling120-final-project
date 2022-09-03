import pathlib
import argparse

from sklearn.cluster import KMeans
from tqdm import tqdm

from src.utils.tsv import read_tsv


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='KNN display for the dataset to analyze phones')
    parser.add_argument('tsv_file', type=pathlib.Path, help='The TSV file to train from')
    parser.add_argument('--num_clusters', type=int, default=100)

    args = parser.parse_args()
    tsv_columns = [
        'filename',
        'start',
        'stop',
        'fft',
        'distance'
    ]

    data = read_tsv(args.tsv_file)

    X = [x['distance'] for x in tqdm(data, desc='Getting fft data')]

    model = KMeans(args.num_clusters).fit(X)
