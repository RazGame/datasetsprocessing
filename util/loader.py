import os as os
import numpy as np
import medpy.io as medpy
import lxml.etree as lxml
from decimal import *


SUPPORTED_DATASETS = ["LIDC"]


class Nodule:
    def __init__(self):
        # width or height of image
        size = 0

        # source image ID, path and slice
        source_id = ""
        source_path = ""
        source_slice = ""

        # coordinates from source image
        source_x = 0
        source_y = 0

        # pixels of image
        pixels = np.array([])

        # is malignant?
        malignant = False


def load_nodules(dataset_path, dataset_type, debug=False):
    full_path = os.path.realpath(dataset_path)

    nodules = []
    if dataset_type == 'LIDC':
        nodules = load_lidc(full_path, debug)
    else:
        raise Exception('Dataset_type is wrong')

    return nodules


def get_all_files(path, ending=""):
    paths = []

    for root, dirs, files in os.walk(path):
        for file_path in files:
            if file_path.endswith(ending):
                full_path = os.path.join(root, file_path)
                paths.append(full_path)

    return paths


class LidcImage:
    def __init__(self):
        id = ""
        fullpath = ""
        slice_location = Decimal(0)
        pixels = np.array([])


def load_lidc(dataset_path, debug):
    img_paths = get_all_files(dataset_path, '.dcm')
    ann_paths = get_all_files(dataset_path, '.xml')

    lids_images = []

    for path in img_paths:
        image = LidcImage()

        pixels, header = medpy.load(path)

        image.pixels = pixels
        image.id = header.data_element("SOPInstanceUID").value
        image.fullpath = path
        image.slice_location = Decimal(header.data_element("SliceLocation").value)

        lids_images.append(image)

    nodules = []


    for path in ann_paths:
        tree = lxml.parse(path)
        root_node = tree.getroot()


        for roi_node in root_node.iter('roi'):
            ann_id = roi_node.find('imageSOP_UID').text
            ann_slice = None
            ann_x = int(roi_node.find('edgeMap').find('xCoord').text)
            ann_y = int(roi_node.find('edgeMap').find('yCoord').text)
            ann_malignant = roi_node.find('inclusion').text

            slice_node = roi_node.find('imageZposition')
            if slice_node is not None:
                ann_slice = Decimal(slice_node.text)

            img = None
            for i in lids_images:
                if i.id == ann_id and (ann_slice is None or i.slice_location == ann_slice):
                    img = i

            if img is None:
                if debug:
                    print("Image for nodule " + ann_id + " not found.", ann_slice)
            else:
                nodule = Nodule()

                s = 32
                nodule.size = s * 2
                nodule.pixels = img.pixels[ann_y - s: ann_y + s, ann_x - s: ann_x + s]
                nodule.source_id = ann_id
                nodule.source_slice = ann_slice
                nodule.source_x = ann_x
                nodule.source_y = ann_y
                nodule.malignant = ann_malignant
                nodule.source_path = img.fullpath

                nodules.append(nodule)

    return nodules
