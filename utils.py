import os
import random
import numpy as np
import scipy.io.wavfile as wav
from collections import defaultdict


def load_data(path, pad=False):
    """ Given path to data directory, returns dictionary of letter: [examples] """
    examples = defaultdict(lambda: list())
    max_l = 0
    for dirname, _, files in os.walk(path):
        if not(os.path.samefile(dirname, path)):
            label = os.path.split(dirname)[1]
            for f in files:
                _, clip = wav.read(os.path.join(dirname, f))
                examples[label].append(pcm2float(clip))
                max_l = max(len(clip), max_l)
    if pad:
        for k in examples.values():
            for i, clip in enumerate(k):
                k[i] = np.pad(clip, (0, max_l - clip.size), mode='constant',
                              constant_values=(0, 0))
    return examples


# https://github.com/mgeier/python-audio/blob/master/audio-files/utility.py
def pcm2float(sig, dtype=np.float64):
    """
    Convert PCM signal to floating point with a range from -1 to 1.

    Use dtype=np.float32 for single precision.

    Parameters
    ----------
    sig : array_like
        Input array, must have (signed) integral type.
    dtype : data-type, optional
        Desired (floating point) data type.

    Returns
    -------
    ndarray
        normalized floating point data.

    See Also
    --------
    dtype

    """
    # TODO: allow unsigned (e.g. 8-bit) data

    sig = np.asarray(sig)  # make sure it's a NumPy array
    assert sig.dtype.kind == 'i', "'sig' must be an array of signed integers!"
    dtype = np.dtype(dtype)  # allow string input (e.g. 'f')

    # Note that 'min' has a greater (by 1) absolute value than 'max'!
    # Therefore, we use 'min' here to avoid clipping.
    return sig.astype(dtype) / dtype.type(-np.iinfo(sig.dtype).min)


def prob_position(clf, X, Y):
    probs = clf.predict_proba(X)
    results = []
    for label, prob in zip(Y, probs):
        ordered = reversed(sorted(zip(prob, clf.classifier.classes_)))
        found = False
        for i, (p, l) in enumerate(ordered):
            if l == label:
                results.append(i)
                found = True
        if found is False:
            result.append(None)
    return np.array(results)
