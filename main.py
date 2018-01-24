#from medpy.io import load
#import matplotlib.pyplot as plt
#import matplotlib.cm as cm
#import skimage.feature as skimg
#import numpy as np
#from sklearn.metrics.cluster import entropy

import util as util

if __name__ == '__main__':
    file_to_process = '208.xml'

    util.load_nodules("lidc-data/", "LIDC")

    # i, h = load("000208.dcm")
    # print(i.shape, i.dtype)
    # # plt.imshow(i, cmap = cm.Greys_r)
    # # plt.show()
    # maxi = np.max(i)
    # print(maxi)
    # # x = i[157:189, 181:213]
    # x = i[185:235, 140:190]
    # # x = i[195:225,150:180]
    # # x = i[205:215,160:170]
    # # print (x)
    # minx = np.min(x)
    # maxx = np.max(x)
    # print(minx, maxx)
    # # plt.imshow(x, cmap = cm.Greys_r)
    # # plt.show()
    # g = skimg.greycomatrix(x, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], maxx + 1)
    # contrast = skimg.greycoprops(g, 'contrast')
    # dissimilarity = skimg.greycoprops(g, 'dissimilarity')
    # homogeneity = skimg.greycoprops(g, 'homogeneity')
    # energy = skimg.greycoprops(g, 'energy')
    # correlation = skimg.greycoprops(g, 'correlation')
    # ASM = skimg.greycoprops(g, 'ASM')
    # ent = entropy(x)
    # print(contrast, dissimilarity, homogeneity, energy, correlation, ASM)
    # print(ent)