import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.utils import shuffle
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
        features = feature_extractor.get_features(nodules)

        feature_extractor.dump_features(dump_path, features)

    print 'Features loaded: ' + str(len(features))

    return np.array(features)


def plot_roc_curves(vals):
    colors = ['k', 'b', 'g', 'r', 'c', 'm', 'y', 'w']
    plt.figure()

    i = 0
    for (fp, tp, a, name) in vals:
        label = name + '(area = ' + str(a) + ')'
        plt.plot(fp, tp, color=colors[i], lw=1, label=label)
        i += 1

    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.01])

    plt.legend(loc="lower right")
    plt.show()


if __name__ == '__main__':
    known_nodules = load_dataset('lidc-data', 'LIDC', use_dump=True)
    extra_nodules = load_dataset('nsclc-data', 'NSCLC', use_dump=True)

    known_nodules = shuffle(known_nodules, random_state=1)

    test_size = len(known_nodules) // 2

    test_nodules = known_nodules[:test_size]
    known_nodules = known_nodules[test_size:]

    known_features = get_features(known_nodules, use_dump=True, dump_name='known')[:80]
    extra_features = get_features(extra_nodules, use_dump=True, dump_name='extra')
    test_features = get_features(test_nodules, use_dump=True, dump_name='test')

    X_test = feature_extractor.features_to_matrix(test_features)
    y_test = feature_extractor.features_to_results(test_features)

    X_orig = feature_extractor.features_to_matrix(known_features)
    y_orig = feature_extractor.features_to_results(known_features)

    scaler = MinMaxScaler()
    X_orig = scaler.fit_transform(X_orig)
    X_test_norm_orig = scaler.transform(X_test)

    svm = SVC(kernel='poly', max_iter=10000, probability=True)
    svm.fit(X_orig, y_orig)

    y_score = svm.decision_function(X_test_norm_orig)

    fpo, tpo, _ = roc_curve(y_test.ravel(), y_score.ravel())
    ao = auc(fpo, tpo)

    vals = [(fpo, tpo, ao, 'ROC orig')]

    mask = np.array(range(len(extra_features)))
    print y_test

    for i in range(5):
        new_mask = np.random.choice(mask, len(mask) - 3, replace=False)

        chosen_extra_features = extra_features[new_mask]
        all_features = np.append(known_features, chosen_extra_features)

        X_train = feature_extractor.features_to_matrix(all_features)
        y_train = feature_extractor.features_to_results(all_features)

        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test_norm = scaler.transform(X_test)

        svm = SVC(kernel='poly', max_iter=10000, probability=True)
        svm.fit(X_train, y_train)

        y_score = svm.decision_function(X_test_norm)
        #print 5
        fp, tp, _ = roc_curve(y_test.ravel(), y_score.ravel())
        #print y_test
        a = auc(fp, tp)
        #print 6
        vals.append((fp, tp, a, 'ROC #' + str(i)))

        print('score', svm.score(X_test, y_test))

    plot_roc_curves(vals)
