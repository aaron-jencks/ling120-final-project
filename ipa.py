import argparse
import pathlib
import sys
from tqdm import tqdm
import eng_to_ipa as ipa
import multiprocessing as mp
from multiprocessing import Queue
import os
from queue import Full, Empty

from tsv import read_tsv, write_tsv, TSVEntry


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

