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


def load_dataset(dataset_path, dataset_type, use_dump=False):
    basename = os.path.basename(os.path.normpath(dataset_path))
    dump_path = 'nodules-' + basename + '.dump'

    if use_dump and os.path.isfile(dump_path):
        print 'Load nodules from file ' + dump_path
        nodules = loader.restore_nodules(dump_path)
    else:
        print 'Load nodules from dataset ' + dataset_path
        nodules = loader.load_nodules(dataset_path, dataset_type)
        loader.dump_nodules(dump_path, nodules)

    print 'Nodules loaded: ' + str(len(nodules))

    return nodules


def get_features(nodules, use_dump=False, dump_name='default'):
    dump_path = 'features-' + dump_name + '.dump'

    if use_dump and os.path.isfile(dump_path):
        print 'Load features from file ' + dump_path
        features = feature_extractor.restore_features(dump_path)
    else:
        print 'Load features from nodules'
        features = feature_extractor.get_features(lidc_nodules)

        feature_extractor.dump_features(dump_path, features)

    print 'Features loaded: ' + str(len(features))

    return features


if __name__ == '__main__':
    lidc_nodules = load_dataset('lidc-data', 'LIDC', use_dump=True)
    nsclc_nodules = load_dataset('nsclc-data', 'NSCLC', use_dump=True)

    all_nodules = lidc_nodules + nsclc_nodules

    all_features = get_features(all_nodules, use_dump=True, dump_name='all')

    X = feature_extractor.features_to_matrix(all_features)
    y = feature_extractor.features_to_results(all_features)

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
