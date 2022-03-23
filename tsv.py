import pathlib
from tqdm import tqdm
from typing import List, Optional, Dict


class TSVEntry:
    def __init__(self, columns: List[str], values: List[str]):
        assert len(columns) == len(values)
        self.values: Dict[str, str] = {}
        for h, v in zip(columns, values):
            assert h not in self.values
            self.values[h] = v

    def to_tsv(self) -> str:
        return '\t'.join(self.values.values())

    def __contains__(self, item):
        return item in self.values

    def __getitem__(self, item: str) -> Optional[str]:
        if item in self.values:
            return self.values[item]
        return None

    def __setitem__(self, key: str, value: str):
        self.values[key] = value


def read_tsv(f: pathlib.Path) -> List[TSVEntry]:
    with open(f, 'rb') as fp:
        data = f.read_text('utf8')
    lines = data.splitlines(keepends=False)
    result = []
    columns: List[str] = []
    first = True
    for l in tqdm(lines, desc="Parsing TSV lines"):
        values = l.split('\t')
        if first:
            first = False
            columns = values
        else:
            result.append(TSVEntry(columns, values))
    return result


def write_tsv(f: pathlib.Path, values: List[TSVEntry]):
    with open(f, 'wb') as fp:
        data = '\n'.join(map(TSVEntry.to_tsv, values)).encode('utf8')
        f.write_bytes(data)
