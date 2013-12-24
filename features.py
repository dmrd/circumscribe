import scipy as sp
from scikits.talkbox.features import mfcc
# Reference: https://github.com/cournape/talkbox/blob/master/scikits/talkbox/features/mfcc.py


def fft(s):
    return sp.fft(s)
