from time import time
import matplotlib.pylab as plt
import numpy as np

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.cross_validation import StratifiedKFold
from sklearn import svm
from classifier import SoundClassifier

import cv2

import utils
import features

USE_PCA = False
NUM_CHARS = 2
PER_CLASS = 40
PATCH_TYPES = 120
N_PATCHES = 100
SEED = 12
N_FOLDS = 2

np.random.seed(SEED)

data_dict = utils.load_data('data')

# Flatten out dictionary to example (X) and label (Y) arrays
X = []
Y = []
for key in data_dict.keys()[:NUM_CHARS]:
    for example in data_dict[key][:PER_CLASS]:
        X.append(example)
        Y.append(key)
X = np.array(X)
Y = np.array(Y)

for i, (train, test) in StratifiedKFold(Y, n_folds=N_FOLDS):
    print("Running fold {}".format(i))
    clf = SoundClassifier(patch_types=PATCH_TYPES,
                          n_patches=N_PATCHES,
                          PCA=USE_PCA,
                          verbose=True)

    clf.fit(X[train], Y[train])

    print "Testing on training data......", clf.score(X[train], Y[train])
    for example, label in zip(X[train], Y[train]):
        print("{} : {}".format(label, clf.predict(example)))

    print "Testing on new data...", clf.score(X[test], Y[test])
    for example, label in zip(X[test], Y[test]):
        print("{} : {}".format(label, clf.predict(example)))

