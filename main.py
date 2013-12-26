import numpy as np
import matplotlib.pylab as plt

from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import confusion_matrix
from classifier import SoundClassifier

import utils

USE_PCA = False
NUM_CHARS = 3
PER_CLASS = 40
PATCH_TYPES = 120
N_PATCHES = 100
SEED = 12
N_FOLDS = 2

np.random.seed(SEED)

data_dict = utils.load_data('data')

cm_predictions = []
cm_labels = []

# Flatten out dictionary to example (X) and label (Y) arrays
X = []
Y = []
for key in data_dict.keys()[:NUM_CHARS]:
    for example in data_dict[key][:PER_CLASS]:
        X.append(example)
        Y.append(key)
X = np.array(X)
Y = np.array(Y)

for i, (train, test) in enumerate(StratifiedKFold(Y, n_folds=N_FOLDS)):
    print("Running fold {}".format(i + 1))
    clf = SoundClassifier(patch_types=PATCH_TYPES,
                          n_patches=N_PATCHES,
                          use_pca=USE_PCA,
                          verbose=True)

    clf.fit(X[train], Y[train])

    print "Testing on training data......", clf.score(X[train], Y[train])
    print "Testing on new data...", clf.score(X[test], Y[test])
    test_predictions = clf.predict(X[test])

    # Store results for computing confusion matrix after all folds
    cm_predictions.append(test_predictions)
    cm_labels.append(Y[test])

# Compute confusion matrix
cm = confusion_matrix(np.concatenate(cm_labels), np.concatenate(cm_predictions))

# Show confusion matrix in a separate window
plt.matshow(cm)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
