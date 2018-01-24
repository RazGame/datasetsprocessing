#import matplotlib.pyplot as plt
#import matplotlib.cm as cm
#import skimage.feature as skimg
#import numpy as np
#from sklearn.metrics.cluster import entropy

from util import loader, featureExtractor

if __name__ == '__main__':
    lidc_data = loader.load_nodules("lidc-data/", "LIDC")

    lidc_features = featureExtractor.get_features(lidc_data[:5])

    print(lidc_features)

    # plt.imshow(i, cmap = cm.Greys_r)
    # plt.show()
