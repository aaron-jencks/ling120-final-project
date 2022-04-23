import pathlib
import argparse
import warnings

from src.models.dataset import generate_tsv_file


warnings.filterwarnings('ignore', 'PySoundFile failed. Trying audioread instead.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Processes the phonemes for the audio files')
    parser.add_argument('tsv_file_in', type=pathlib.Path, help='The file containing the list of sample file locations')
    parser.add_argument('tsv_file_out', type=pathlib.Path, help='The file to store the new TSV file in')
    parser.add_argument('clips_dir', type=pathlib.Path, help='The directory containing the sample files')
    parser.add_argument('phonemes_dir', type=pathlib.Path, help='The directory containing the sample phoneme files')

    args = parser.parse_args()
    generate_tsv_file(args.tsv_file_in, args.tsv_file_out, args.clips_dir, args.phonemes_dir)
