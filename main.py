import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from sklearn.svm import SVC
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

    it = 0
    for (fpv, tpv, av, name) in vals:
        label = name + '(area = ' + str(av) + ')'
        plt.plot(fpv, tpv, lw=1, label=label)
        it += 1

    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.01])

    plt.legend(loc="lower right")
    plt.show()


def add_noise(features, step):
    for i in range(0, len(features), step):
        features[i].conclusion = not features[i].conclusion


def run_svm(features_train, features_test):
    x_train, y_train = feature_extractor.transform_features(features_train)
    x_test, y_test = feature_extractor.transform_features(features_test)

    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    svm = SVC(kernel='poly', max_iter=-1, probability=True)
    svm.fit(x_train, y_train)

    y_score = svm.decision_function(x_test)

    fpv, tpv, _ = roc_curve(y_test, y_score)
    av = auc(fpv, tpv)

    return fpv, tpv, av


if __name__ == '__main__':
    lidc_nodules = load_dataset('lidc-data', 'LIDC', use_dump=True)
    # nsclc_nodules = load_dataset('nsclc-data', 'NSCLC', use_dump=True)

    lidc_features = get_features(lidc_nodules, use_dump=True, dump_name='lidc')

    lidc_features = shuffle(lidc_features[:600], random_state=1)  #

    part_size = len(lidc_features) // 3

    known_features = lidc_features[:part_size]
    extra_features = lidc_features[part_size:part_size * 2]
    test_features = lidc_features[part_size * 2:]

    add_noise(extra_features, 2)

    fpo, tpo, ao = run_svm(known_features, test_features)

    vals = [(fpo, tpo, ao, 'ROC orig')]

    mask = np.array(range(len(extra_features)))

    for i in range(5):
        new_mask = np.random.choice(mask, len(mask) - 3, replace=False)

        chosen_extra_features = extra_features[new_mask]
        all_features = np.append(known_features, chosen_extra_features)

        fp, tp, a = run_svm(all_features, test_features)

        vals.append((fp, tp, a, 'ROC #' + str(i)))
        print 'step'

    plot_roc_curves(vals)
