import pathlib
import argparse

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

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
        'fft'
    ]

    data = read_tsv(args.tsv_file)

    X = [list(map(float, x['fft'].split(','))) for x in tqdm(data, desc='Getting fft data')]

    pca = PCA(2).fit(X)

    labels = [l / 100 for l in KMeans(args.num_clusters).fit_predict(X)]
    cmap = plt.get_cmap('plasma')

    Xred = pca.transform(X)
    xs = [x[0] for x in Xred]
    ys = [y[1] for y in Xred]
    zs = [np.linalg.norm(fft) for fft in X]

    fig = plt.figure()
    ax = fig.add_subplot(2, 2, 1, projection='3d')
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)

    ax.scatter(xs, ys, zs, c=labels, cmap=cmap, s=0.1)
    ax2.hist(labels, 100)

    for fi, fft in tqdm(enumerate(X), desc='Generating overlay plot'):
        ax3.plot(range(len(fft)), fft, s=0.1)

    fig.show()
