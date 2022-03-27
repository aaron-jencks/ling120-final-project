import multiprocessing as mp
from multiprocessing import Queue
from queue import Full, Empty
import os
import sys
from typing import List, Callable

from tqdm import tqdm


def process_processor(qin: Queue, qout: Queue, func: Callable):
    while True:
        pkt = qin.get()
        if pkt:
            qout.put(func(pkt))
        else:
            break


def round_robin_map(values: List, mapping_function: Callable, batch_size: int = 256, tqdm_label: str = '') -> List:
    ccount = os.cpu_count()

    procout = Queue(ccount * batch_size)

    print('Creating processors', file=sys.stderr)
    processors = []
    pipes = []
    for _ in range(ccount):
        i = Queue(batch_size)
        processors.append(mp.Process(target=process_processor, args=(i, procout, mapping_function)))
        pipes.append(i)
        processors[-1].start()

    ecount = len(values)
    new_values = []
    pbar = tqdm(total=len(values), desc=tqdm_label)
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
            for _ in range(batch_size):
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

    return new_values
