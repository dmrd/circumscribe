from time import time
import matplotlib.pylab as plt
import numpy as np

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.cross_validation import StratifiedKFold
from sklearn import svm

import cv2

import utils
import features

USE_PCA = False
NUM_CHARS = 2
PER_CLASS = 40
PATCH_TYPES = 50
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

for train, test in StratifiedKFold(Y, n_folds=N_FOLDS):
    print("Creating windows...")
    windows = features.generate_windows(X[train], N_PATCHES)
    windows2 = features.generate_windows2(X[train], N_PATCHES)

    print("Clustering patches...")
    patch_clusterer = KMeans(init='k-means++', n_clusters=PATCH_TYPES, n_init=3)
    patch_clusterer.fit(windows)
    patch_clusterer2 = KMeans(init='k-means++', n_clusters=PATCH_TYPES, n_init=3)
    patch_clusterer2.fit(windows2)

    print("Calculating features for each example...")
    # Map every sample to its features
    histograms = []
    for example in X:
        (Pxx, freqs, bins, im) = plt.specgram(example)
        img = features.as_img(Pxx)
        patches = features.get_slices(img, N_PATCHES)
        patches2 = features.get_slices2(img, N_PATCHES)
        patch_counts = [0] * PATCH_TYPES
        patch_counts2 = [0] * PATCH_TYPES
        for patch in patches:
            patch_type = patch_clusterer.predict(patch)[0]
            patch_counts[patch_type] += 1
        for patch in patches2:
            patch_type2 = patch_clusterer2.predict(patch)[0]
            patch_counts2[patch_type2] += 1
        # print patch_counts
        histograms.append(patch_counts + patch_counts2)

    histograms = np.array(histograms)

    if USE_PCA:
        print("Finding principal components...")
        pca = PCA(n_components=10).fit(histograms[train])
        histograms = pca.transform(histograms)

    # Train!
    print("Training...")
    clf = svm.SVC()
    clf.fit(histograms[train], Y[train])

    print "Testing on training data......", clf.score(histograms[train],Y[train])
    for example, label in zip(histograms[train], Y[train]):
        print("{} : {}".format(label, clf.predict(example)))

    print "Testing on new data...", clf.score(histograms[test],Y[test])
    for example, label in zip(histograms[test], Y[test]):
        print("{} : {}".format(label, clf.predict(example)))

