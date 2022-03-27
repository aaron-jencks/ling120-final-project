import numpy as np


def seconds_to_index(t: float, sr: int) -> int:
    return int(np.floor(t * sr))
