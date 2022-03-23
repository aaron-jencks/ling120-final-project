import argparse
import pathlib
import random
import sys
from typing import List

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QCheckBox, QLabel, QPushButton, QApplication
from tqdm import tqdm
import eng_to_ipa as ipa
import multiprocessing as mp
from multiprocessing import Queue
import os
from queue import Full, Empty

from tsv import read_tsv, write_tsv, TSVEntry


class SentenceWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QHBoxLayout()
        self.checkbox = QCheckBox()
        self.eng = QLabel()
        self.ipa = QLabel()
        self.layout.addWidget(self.checkbox)
        qvl = QWidget()
        vl = QVBoxLayout()
        vl.addWidget(self.eng)
        vl.addWidget(self.ipa)
        qvl.setLayout(vl)
        self.layout.addWidget(qvl)
        self.setLayout(self.layout)
        self.index = -1

    def is_wrong(self) -> bool:
        return self.checkbox.isChecked()

    def set_classification(self, eid: int, eng: str, tipa: str):
        self.index = eid
        self.eng.setText(eng)
        self.ipa.setText(tipa)
        self.checkbox.setChecked(False)


class MainWindow(QWidget):
    def __init__(self, entries: List[TSVEntry], filename: pathlib.Path):
        super().__init__()
        self.display_count = 10
        self.layout = QVBoxLayout()
        self.sentences = []
        for _ in range(self.display_count):
            q = SentenceWidget()
            self.sentences.append(q)
            self.layout.addWidget(q)
        self.submit_button = QPushButton("Finished")
        self.submit_button.clicked.connect(self.double_check)
        self.progress_status = QLabel()
        self.layout.addWidget(self.submit_button)
        self.layout.addWidget(self.progress_status)
        self.entries = list(enumerate(entries[:]))
        self.dataset = entries[:]
        self.filename = filename
        self.setLayout(self.layout)
        self.generate_next_set()

    def save_dataset(self):
        write_tsv(self.filename, self.dataset)

    def generate_next_set(self):
        if self.display_count > len(self.entries):
            entry_set = self.entries[:]
            self.entries.clear()
        else:
            entry_set = random.sample(self.entries, k=self.display_count)
            for e in entry_set:
                self.entries.remove(e)

        for e, d in zip(entry_set, self.sentences):
            index, entry = e
            d.set_classification(index, entry['sentence'], entry['ipa'])

        self.progress_status.setText('{} entires remain'.format(len(self.entries)))

    def double_check(self):
        for s in self.sentences:
            if s.is_wrong():
                s.ipa.setText(input('What is the ipa for "{}"? '.format(s.eng.text())))
            self.dataset[s.index]['ipa'] = s.ipa.text()
        self.save_dataset()
        self.generate_next_set()


def insert_tsv_ipa(pin: Queue, pout: Queue):
    while True:
        pkt = pin.get()
        if pkt:
            if isinstance(pkt, TSVEntry):
                pkt['ipa'] = ipa.convert(pkt['sentence'])
                pout.put(pkt)
        else:
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Converts english to IPA and updates the tsv file accordingly')
    parser.add_argument('tsv_file', type=pathlib.Path, help='The file to insert ipa into')
    parser.add_argument('--batch_size', type=int, help='The number of entries to queue up on the processor',
                        default=256, required=False)

    args = parser.parse_args()

    values = read_tsv(args.tsv_file)

    if not all(map(lambda x: 'ipa' in x, tqdm(values, desc="Checking for IPA column"))):
        result = []
        ccount = os.cpu_count()

        procout = Queue(ccount * args.batch_size)

        print('Creating processors', file=sys.stderr)
        processors = []
        pipes = []
        for _ in range(ccount):
            i = Queue(args.batch_size)
            processors.append(mp.Process(target=insert_tsv_ipa, args=(i, procout)))
            pipes.append(i)
            processors[-1].start()

        ecount = len(values)
        new_values = []
        pbar = tqdm(total=len(values), desc="Converting sentences to IPA")
        keep_going = True
        while keep_going:
            try:
                while True:
                    resp = procout.get_nowait()
                    pbar.update(1)
                    new_values.append(resp)
                    if len(new_values) == ecount:
                        keep_going = False
                        break
            except Empty:
                pass

            for proc in range(len(processors)):
                for _ in range(args.batch_size):
                    if len(values) > 0:
                        v = values.pop(-1)
                        try:
                            pipes[proc].put_nowait(v)
                        except Full:
                            values.append(v)
                            break
                    else:
                        break

        for proc in pipes:
            proc.put(False)
            proc.close()

        for proc in processors:
            proc.join()

        write_tsv(args.tsv_file, new_values)
        values = new_values

    app = QApplication(sys.argv)
    v = MainWindow(values, args.tsv_file)
    v.show()
    app.exec()

