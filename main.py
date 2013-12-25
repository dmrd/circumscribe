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

NUM_CHARS = 2
PER_CLASS = 10
PATCH_TYPES = 100
N_PATCHES = 50
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

    print("Clustering patches...")
    patch_clusterer = KMeans(init='k-means++', n_clusters=PATCH_TYPES, n_init=5)
    patch_clusterer.fit(windows)

    print("Calculating features for each example...")
    # Map every sample to its features
    descriptors = []
    for example in X:
        (Pxx, freqs, bins, im) = plt.specgram(example)
        img = features.as_img(Pxx)
        patches = features.get_slices(img, N_PATCHES)
        patch_counts = [0] * PATCH_TYPES
        for patch in patches:
            patch_type = patch_clusterer.predict(patch)[0]
            patch_counts[patch_type] += 1
        descriptors.append(patch_counts)

    descriptors = np.array(descriptors)

    # Train!
    print("Training...")
    clf = svm.SVC()
    clf.fit(descriptors[train], Y[train])

    print("Testing on training data......")
    for example, label in zip(descriptors[train], Y[train]):
        print("{} : {}".format(label, clf.predict(example)))

    print("Testing on new data...")
    for example, label in zip(descriptors[test], Y[test]):
        print("{} : {}".format(label, clf.predict(example)))

# data = np.array(data)
# print data
# print data.shape
# reduced_data = PCA(n_components=2).fit_transform(data)
# print reduced_data

# n_groups = 2
# print data.shape
# n_samples, n_features = data.shape

# estimator = KMeans(init='k-means++', n_clusters=n_groups, n_init=10)
# estimator.fit(data)
# print estimator.labels_

# # n_samples, n_features = data.shape
