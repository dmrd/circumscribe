import os
import scipy.io.wavfile as wav
from collections import defaultdict


def load_data(path):
    """ Given path to data directory, returns dictionary of letter: [examples] """
    examples = defaultdict(lambda: list())
    for dirname, _, files in os.walk(path):
        if not(os.path.samefile(dirname, path)):
            label = os.path.split(dirname)[1]
            for f in files:
                print(f)
                _, clip = wav.read(os.path.join(dirname, f))
                examples[label].append(clip)
    return examples
