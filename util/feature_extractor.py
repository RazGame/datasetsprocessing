import skimage.feature as skimg
import numpy as np
from sklearn.metrics.cluster import entropy
import pickle
import gzip
import os
import medpy.io as medpy
from skimage import io
from lxml import etree
import scipy.misc
import warnings


class Features:
    def __init__(self, nodule):
        self.max_coord = 0
        self.conclusion = False
        self.features = dict()
        self.nodule = nodule


def get_features(nodules):
    features = []

    i = 0

    for nodule in nodules:
        nodule_feature = Features(nodule)

        nodule_feature.max_coord = np.max(nodule.pixels)
        nodule_feature.conclusion = nodule.conclusion

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            grey_comatrix = skimg.greycomatrix(nodule.pixels, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                                               nodule_feature.max_coord + 1)

        nodule_feature.features['contrast'] = skimg.greycoprops(grey_comatrix, 'contrast').flatten().astype(float)
        nodule_feature.features['dissimilarity'] = skimg.greycoprops(grey_comatrix, 'dissimilarity').flatten().astype(
            float)
        nodule_feature.features['homogeneity'] = skimg.greycoprops(grey_comatrix, 'homogeneity').flatten()
        nodule_feature.features['energy'] = skimg.greycoprops(grey_comatrix, 'energy').flatten()
        nodule_feature.features['correlation'] = skimg.greycoprops(grey_comatrix, 'correlation').flatten()
        nodule_feature.features['ASM'] = skimg.greycoprops(grey_comatrix, 'ASM').flatten().astype(float)
        nodule_feature.features['entropy'] = entropy(nodule.pixels)

        features.append(nodule_feature)

        i += 1
        print 'progress =', i, '/', len(nodules), nodule.source_id

    return np.array(features)


def transform_features(features):
    features_matrix = np.array([])
    conclusions_vector = np.array([])

    for feature in features:
        for value in feature.features.values():
            features_matrix = np.append(features_matrix, value)

        conclusions_vector = np.append(conclusions_vector, feature.conclusion)

    features_matrix = features_matrix.reshape((len(features), -1))

    return features_matrix, conclusions_vector


def dump_features(file_path, features):
    f = gzip.open(file_path, "wb")

    pickle.dump(features, f)

    f.close()


def restore_features(file_path):
    f = gzip.open(file_path, "rb")

    features = pickle.load(f)

    f.close()

    return features


def save_as_dataset(features, path):
    if not os.path.exists(path):
        os.makedirs(path)

    nodules_node = etree.Element('nodules')

    for i, feature in enumerate(features):
        file_path = os.path.join(path, str(i) + '.tif')
        file_pathp = os.path.join(path, str(i) + '.png')
        nodule = feature.nodule

        io.imsave(file_path, nodule.pixels)
        scipy.misc.toimage(nodule.pixels, high=255, cmin=0.0, cmax=3000.0).save(file_pathp)

        nodule_node = etree.SubElement(nodules_node, 'nodule')
        etree.SubElement(nodule_node, 'Id').text = str(nodule.source_id)
        etree.SubElement(nodule_node, 'SourcePath').text = str(nodule.source_path)
        etree.SubElement(nodule_node, 'SourceX').text = str(nodule.source_x)
        etree.SubElement(nodule_node, 'SourceY').text = str(nodule.source_y)
        etree.SubElement(nodule_node, 'Conclusion').text = str(nodule.conclusion)
        etree.SubElement(nodule_node, 'ImageName').text = file_path
        etree.SubElement(nodule_node, 'ImageSize').text = str(nodule.size)

    annotation_file = open(os.path.join(path, 'annotation.xml'), 'w')
    annotation_file.write(etree.tostring(nodules_node, pretty_print=True))
    annotation_file.close()
