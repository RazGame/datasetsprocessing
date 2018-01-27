import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import MinMaxScaler

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
        print 'lens:', len(lidc_nodules), len(nsclc_nodules), len(nodules)

        loader.dump_nodules(NODULES_DAMP_PATH, nodules)

    print len(nodules)

    features = None

    if USE_DAMPED_FEATURES and os.path.isfile(FEATURES_DAMP_PATH):
        features = feature_extractor.restore_features(FEATURES_DAMP_PATH)
    else:
        features = feature_extractor.get_features(nodules)

        feature_extractor.dump_features(FEATURES_DAMP_PATH, features)

    print(len(features))

    X = feature_extractor.features_to_matrix(features)
    y = feature_extractor.features_to_results(features)

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5, random_state=0, shuffle=True)

    svm = SVC(kernel='poly', max_iter=40000, probability=True)
    svm.fit(X_train, y_train)

    y_score = svm.decision_function(X_test)

    fp, tp, _ = roc_curve(y_test.ravel(), y_score.ravel())
    auc = auc(fp, tp)

    plt.figure()
    plt.plot(fp, tp, color='darkorange',
             lw=2, label='ROC curve (area = %0.2f)' % auc)
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.01])
    plt.legend(loc="lower right")
    plt.show()

    print('score', svm.score(X_test, y_test))
