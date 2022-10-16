import pathlib
import re
from typing import List


class DictEntry:
    def __init__(self, term: str, phonemes: List[str]):
        self.term = term
        self.phonemes = phonemes

    def __repr__(self):
        return ' '.join([self.term, *self.phonemes])

    def phoneme_display(self) -> str:
        return ' '.join([*self.phonemes])


def parse_dict(dfile: pathlib) -> List[DictEntry]:
    entries = []
    with open(dfile, 'r') as fp:
        lines = fp.readlines()
        for line in lines:
            bits = re.split(r'\s+', line)
            entries.append(DictEntry(bits[0], bits[1:]))
    return entries


if __name__ == '__main__':
    from src.utils.config import get_settings

    settings = get_settings()
    for entry in parse_dict(settings['word_dict']):
        print(entry)
