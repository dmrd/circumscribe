import features
import numpy as np
import matplotlib.pylab as plt
from sklearn import svm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, ClassifierMixin


class SoundClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, patch_types=120, n_patches=100, use_pca=False,
                 verbose=False, classifier=None, patch_clusterer=None):
        """
        n_patches: number of patches to generate from a single sample
        patch_types: Number of clusters for the k-means
        """
        self.patch_types = patch_types
        self.n_patches = n_patches
        self.use_pca = use_pca
        self.classifier = classifier or svm.SVC(probability=True)
        self.patch_clusterer = patch_clusterer or None
        self.verbose = verbose

    def _create_histogram(self, example):
        (Pxx, freqs, bins, im) = plt.specgram(example)
        plt.clf()  # specgram plots so we clear
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
        if self.patch_clusterer is None:
            self.patch_clusterer = KMeans(init='k-means++', n_clusters=self.patch_types, n_init=3)
        self.patch_clusterer.fit(windows)

        if self.verbose:
            print("Creating histograms...")
        histograms = []
        for example in X:
            histograms.append(self._create_histogram(example))
        histograms = np.array(histograms)

        if self.use_pca:
            if self.verbose:
                print("Finding principal components...")
            pca = PCA(n_components=10).fit(histograms)
            histograms = pca.transform(histograms)

        if self.verbose:
            print("Training classifier...")
        self.classifier.fit(histograms, Y)
        return

    def predict(self, X):
        if X.dtype != object:  # dtype of numpy arrays
            raise Exception("Predict takes an array of items to classify, not a single item")
        histograms = []
        for example in X:
            histograms.append(self._create_histogram(example))
        histograms = np.array(histograms)
        return self.classifier.predict(histograms)

    def predict_proba(self, X):
        if X.dtype != object:  # dtype of numpy arrays
            raise Exception("Predict takes an array of items to classify, not a single item")
        histograms = []
        for example in X:
            histograms.append(self._create_histogram(example))
        histograms = np.array(histograms)
        return self.classifier.predict_proba(histograms)
