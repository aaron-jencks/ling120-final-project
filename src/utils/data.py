import pathlib

from src.utils.sound_util import seconds_to_index
from src.utils.tsv import TSVEntry


class Phoneme:
    def __init__(self, name: str, start: float, stop: float, sample_rate: int, file_length: int):
        self.name = name
        self.start = start
        self.stop = stop
        self.sample_rate = sample_rate
        self.file_length = file_length

    @property
    def start_index(self) -> int:
        i = seconds_to_index(self.start, self.sample_rate)
        if i >= self.file_length:
            i -= 1
        return i

    @property
    def stop_index(self) -> int:
        i = seconds_to_index(self.stop, self.sample_rate)
        if i >= self.file_length:
            i -= 1
        return i


class RecordingSample:
    def __init__(self, e: TSVEntry, loc: pathlib.Path):
        self.entry = e
        self.directory = loc
        self.base_filename = e['path'].split('.')[0]
        self.location = loc / e['path']
        self.output_location = loc / (self.base_filename + '.wav')
        self.alignment_location = loc / (self.base_filename + '.json')