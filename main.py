import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from sklearn.svm import SVC

from util import loader, feature_extractor


def show_nodule(nodule):
    plt.imshow(nodule.pixels, cmap=cm.Greys_r)
    plt.show()


if __name__ == '__main__':
    lidc_data = loader.load_nodules('lidc-data/', 'LIDC')
    nsclc_data = loader.load_nodules('nsclc-data/', 'NSCLC')

    lidc_features = feature_extractor.get_features(lidc_data[:10])
    nsclc_features = feature_extractor.get_features(nsclc_data)

    print("LIDC:")
    for f in lidc_features:
        print (f)

    print("NSCLC:")
    for f in nsclc_features:
        print (f)

    X = feature_extractor.features_to_matrix(lidc_features)
    y = np.random.choice(a=[False, True], size=len(lidc_features))  # TODO

    svm = SVC()
    svm.fit(X, y)

    print('score', svm.score(X, y))
