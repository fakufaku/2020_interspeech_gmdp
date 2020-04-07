import numpy as np
from scipy.io import wavfile


def load_audio(filename):
    fs, audio = wavfile.read(filename)
    if audio.dtype == np.int16:
        audio = audio.astype(np.float64) / 2 ** 15
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float64) / 2 ** 31

    return fs, audio


def save_audio(filename, fs, audio):
    audio = (audio * 2 ** 15).astype(np.int16)
    wavfile.write(filename, fs, audio)
