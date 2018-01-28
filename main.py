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

    plt.xlabel('FP rate')
    plt.ylabel('TP rate')

    plt.legend(loc="lower right")
    plt.show()


def add_noise(features, step):
    for i in range(0, len(features), step):
        print i
        features[i].conclusion = not features[i].conclusion


def calculate_impact(a_train, a_ext, f_train, f_ext, f_test, ind):
    print ind
    f_train_t = np.append(f_train, f_ext[ind])
    f_ext_t = np.delete(f_ext, ind)

    _, _, a1 = run_svm(f_train_t, f_test)
    _, _, a2 = run_svm(f_ext_t, f_test)

    d0 = a_train - a_ext
    dt = a1 - a2

    return abs(d0 - dt)


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


def main():
    lidc_nodules = load_dataset('lidc-data', 'LIDC', use_dump=True)
    # nsclc_nodules = load_dataset('nsclc-data', 'NSCLC', use_dump=True)

    lidc_features = get_features(lidc_nodules, use_dump=True, dump_name='lidc')

    lidc_features = shuffle(lidc_features[:1200], random_state=1)

    part_size = len(lidc_features) // 3

    f_train = lidc_features[:part_size]
    f_extra = lidc_features[part_size:part_size * 2]
    f_test = lidc_features[part_size * 2:]

    add_noise(f_extra, 10)

    fpo, tpo, ao = run_svm(f_train, f_test)
    vals = [(fpo, tpo, ao, 'ROC train')]

    fpe, tpe, ae = run_svm(f_extra, f_test)
    vals.append((fpe, tpe, ae, 'ROC extra'))

    impacts = []

    for i in range(len(f_extra)):
        imp = calculate_impact(ao, ae, f_train, f_extra, f_test, i)
        impacts.append(imp)

    impacts = impacts/np.linalg.norm(impacts)

    bad_inds = []

    for i, imp in enumerate(impacts):
        if imp > 0.1:
            bad_inds.append(i)

    print bad_inds

    f_filtered = np.delete(f_extra, bad_inds)

    f_train_and_filtered = np.append(f_train, f_filtered)

    fpf, tpf, af = run_svm(f_train_and_filtered, f_test)
    vals.append((fpf, tpf, af, 'ROC train + filtered extra'))

    plot_roc_curves(vals)


if __name__ == '__main__':
    main()
