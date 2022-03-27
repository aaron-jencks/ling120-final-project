import numpy as np


def seconds_to_index(t: float, sr: int) -> int:
    return np.floor(t * sr)
