import scipy as sp
import numpy as np
import cv2
import matplotlib.pylab as plt
from random import random
# from scikits.talkbox.features import mfcc
# Reference: https://github.com/cournape/talkbox/blob/master/scikits/talkbox/features/mfcc.py


def fft(s):
    return sp.fft(s)

def flatten(m):
    return np.reshape(m, np.prod(m.shape))

def random_patches(img, count, radius=5):
    for i in xrange(count):
        patch = random_patch(img, radius=radius)
        shape = patch.shape
        yield flatten(patch)

def get_slices(img, count, width=5):
    rows, cols = img.shape
    for i in xrange(count):
        progress = i / (count - 1.0)
        start = int((cols - width) * progress)
        my_slice = img[:, start:start+width]  # TODO(Bieber): Weird all 75s (0s in Pxx)
        yield flatten(my_slice)


def random_patch(img, radius=5):
    r = random()
    rows, cols = img.shape
    c = int(r * (cols - 2 * radius - 2)) + radius + 1
    return img[:, c-radius:c+radius]


def as_img(data):
    data = np.log(data)
    img = np.zeros(data.shape, np.uint8)
    img[:] = 10 * (data + 20)  # make all data positive then scale...
    return img


def generate_windows(audio_samples, patches):
    windows = []
    for example in audio_samples:
        (Pxx, freqs, bins, im) = plt.specgram(example)
        # print np.min(Pxx)
        # print np.max(Pxx)
        # print np.mean(Pxx)
        # print np.median(Pxx)
        # print
        Pxx[Pxx < 1e-8] = 1e-8
        # plt.show()
        img = as_img(Pxx)
        plt.clf()  # specgram plots, so clear the plot
        windows.extend(get_slices(img, patches))

    return np.array(windows)
