#import matplotlib.pyplot as plt
#import matplotlib.cm as cm
#import skimage.feature as skimg
#import numpy as np
#from sklearn.metrics.cluster import entropy

from util import loader, featureExtractor

if __name__ == '__main__':
    lidc_data = loader.load_nodules('lidc-data/', 'LIDC')
    nsclc_data = loader.load_nodules('nsclc-data/', 'NSCLC')

    lidc_features = featureExtractor.get_features(lidc_data[:5])
    nsclc_features = featureExtractor.get_features(nsclc_data)

    print(lidc_features)
    print(nsclc_features)

    print("LIDC:")
    for f in lidc_features:
        print (f)

    print("NSCLC:")
    for f in nsclc_features:
        print (f)

    # plt.imshow(i, cmap = cm.Greys_r)
    # plt.show()
