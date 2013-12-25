import utils
import features

from time import time
import numpy as np
import matplotlib.pylab as plt

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn import svm

import cv2

NUM_CHARS = 4
PATCH_TYPES = 10

np.random.seed(12)

data_dict = utils.load_data('data')
data = []

SIFT = cv2.SIFT(nfeatures=20)

windows = []
for key in data_dict.keys()[0:NUM_CHARS]:
    for example in data_dict[key][0:5]:
        (Pxx, freqs, bins, im) = plt.specgram(example)
        img = features.as_img(Pxx)
        plt.clf() # specgram plots, so clear
        windows.extend(features.random_patches(img, 1000))

windows = np.array(windows)
patch_clusterer = KMeans(init='k-means++', n_clusters=PATCH_TYPES, n_init=5)
patch_clusterer.fit(windows)

# Train!
X = []
Y = []
for key in data_dict.keys()[0:NUM_CHARS]:
    for example in data_dict[key][0:10]:  # examples per class to train
        (Pxx, freqs, bins, im) = plt.specgram(example)
        patches = features.random_patches(img, 2000)
        patch_counts = [0] * PATCH_TYPES
        for patch in patches:
            patch_type = patch_clusterer.predict(patch)[0]
            patch_counts[patch_type] += 1
        X.append(patch_counts)
        Y.append(key)

clf = svm.SVC()
clf.fit(X,Y)


# Test!
for key in data_dict.keys()[0:NUM_CHARS]:
    for example in data_dict[key][0:10]:
        (Pxx, freqs, bins, im) = plt.specgram(example)
        patches = features.random_patches(img, 2000)
        patch_counts = [0] * PATCH_TYPES
        for patch in patches:
            patch_type = patch_clusterer.predict(patch)[0]
            patch_counts[patch_type] += 1
        prediction = clf.predict(patch_counts)
        print key, prediction

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


