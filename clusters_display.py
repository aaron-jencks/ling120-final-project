import pathlib
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='KNN display for the dataset to analyze phones')
    parser.add_argument('tsv_file', type=pathlib.Path, help='The TSV file to train from')

    args = parser.parse_args()
    tsv_columns = [
        'filename',
        'start',
        'stop',
        'fft',
        'distance'
    ]
