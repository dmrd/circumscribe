import features
import numpy as np
import matplotlib.pylab as plt
from sklearn import svm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator


class SoundClassifier(BaseEstimator):
    def __init__(self, patch_types=120, n_patches=100, PCA=False,
                 verbose=False, classifier=None):
        self.patch_types = patch_types
        self.n_patches = n_patches
        self.PCA = PCA
        self.classifier = classifier or svm.SVC()
        self.verbose = verbose

    def _create_histogram(self, example):
        (Pxx, freqs, bins, im) = plt.specgram(example)
        img = features.as_img(Pxx)
        patches = features.get_slices(img, self.n_patches)
        patch_counts = [0] * self.patch_types
        for patch in patches:
            patch_type = self.patch_clusterer.predict(patch)[0]
            patch_counts[patch_type] += 1
        return patch_counts

    def fit(self, X, Y):
        if self.verbose:
            print("Creating windows...")
        windows = features.generate_windows(X, self.n_patches)

        if self.verbose:
            print("Clustering patches...")
        self.patch_clusterer = KMeans(init='k-means++', n_clusters=self.patch_types, n_init=3)
        self.patch_clusterer.fit(windows)

        if self.verbose:
            print("Creating histograms...")
        histograms = []
        for example in X:
            histograms.append(self._create_histogram(example))
        histograms = np.array(histograms)

        if self.PCA:
            if self.verbose:
                print("Finding principal components...")
            pca = PCA(n_components=10).fit(histograms)
            histograms = pca.transform(histograms)

        if self.verbose:
            print("Training classifier...")
        self.classifier.fit(histograms, Y)
        return

    def predict(self, X):
        histogram = self._create_histogram(X)
        return self.classifier.predict(histogram)

    def score(self, X, Y):
        score = 0.0
        for example, label in zip(X, Y):
            pred = self.predict(example)
            score += (pred == label)
        return score / len(Y)

