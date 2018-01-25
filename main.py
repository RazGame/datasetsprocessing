import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from sklearn.svm import SVC

from util import loader, feature_extractor


def show_nodule(nodule):
    plt.imshow(nodule.pixels, cmap=cm.Greys_r)
    plt.show()


if __name__ == '__main__':
    NODULES_DAMP_PATH = 'nodules.dump'
    FEATURES_DAMP_PATH = 'features.dump'
    USE_DAMPED_NODULES = True
    USE_DAMPED_FEATURES = True

    nodules = None

    if USE_DAMPED_NODULES and os.path.isfile(NODULES_DAMP_PATH):
        nodules = loader.restore_nodules(NODULES_DAMP_PATH)
    else:
        lidc_nodules = loader.load_nodules('lidc-data/', 'LIDC')
        nsclc_nodules = loader.load_nodules('nsclc-data/', 'NSCLC')

        nodules = np.append(lidc_nodules, nsclc_nodules)

        loader.dump_nodules(NODULES_DAMP_PATH, nodules)

    features = None

    if USE_DAMPED_FEATURES and os.path.isfile(FEATURES_DAMP_PATH):
        features = feature_extractor.restore_features(FEATURES_DAMP_PATH)
    else:
        features = feature_extractor.get_features(nodules)

        feature_extractor.dump_features(FEATURES_DAMP_PATH, features)

    X = feature_extractor.features_to_matrix(features)
    y = np.random.choice(a=[False, True], size=len(features))  # TODO

    svm = SVC()
    svm.fit(X, y)

    print('score', svm.score(X, y))
