from typing import List

import pyfoal
import numpy as np
import sounddevice as sd

from src.utils.data import Phoneme


def record_sample(seconds: int = 3, sample_rate: int = 44100) -> np.ndarray:
    myrecording = sd.rec(int(seconds * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()
    return myrecording


def create_forced_alignment(data: List[float], phrase: str, sr: int = 44100):
    try:
        alignment = pyfoal.align(phrase, data, sr)
        phonemes = [Phoneme(p.phoneme, p.start(), p.end(), sr, len(data)) for p in alignment.phonemes()]
        return phonemes
    except RuntimeError as e:
        print(e)
    except IndexError as e:
        print(e)
    except ValueError as e:
        print(e)