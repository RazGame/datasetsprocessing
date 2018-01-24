import os as os
import numpy as np
import medpy.io as medpy
import lxml.etree as lxml
from decimal import *


SUPPORTED_DATASETS = ['LIDC', 'NSCLC']


class Nodule:
    def __init__(self):
        # width or height of image
        self.size = 0

        # source image ID, path and slice
        self.source_id = ""
        self.source_path = ""
        self.source_slice = ""

        # coordinates from source image
        self.source_x = 0
        self.source_y = 0

        # pixels of image
        self.pixels = np.array([])

        # is malignant?
        self.malignant = False


def load_nodules(dataset_path, dataset_type, debug=False):
    """ Load dataset (which has type DATASET_TYPE) from DATASET_PATH recursively.

     DATASET_PATH - path to dataset.
     DATASET_TYPE - type of dataset. Look SUPPORTED_DATASETS for available dataset types.
     DEBUF - activates debug messages.

     Returns list of Nodules.
     """
    full_path = os.path.realpath(dataset_path)

    nodules = []

    if dataset_type == 'LIDC':
        nodules = load_lidc(full_path, debug)
    elif dataset_type == 'NSCLC':
        nodules = load_nsclc(full_path, debug)
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


def crop_image(pixels, center_x, center_y, image_size):
    s = int(image_size / 2)
    return pixels[center_x - s: center_x + s, center_y - s: center_y + s]


class DicomImage:
    def __init__(self):
        self.id = ""
        self.fullpath = ""
        self.slice_location = Decimal(0)
        self.pixels = np.array([])


def load_dicom_image(path):
    image = DicomImage()

    pixels, header = medpy.load(path)

    image.pixels = pixels
    image.id = header.data_element("SOPInstanceUID").value
    image.fullpath = path
    image.slice_location = Decimal(header.data_element("SliceLocation").value)

    return image


def load_lidc(dataset_path, debug):
    img_paths = get_all_files(dataset_path, '.dcm')
    ann_paths = get_all_files(dataset_path, '.xml')

    lids_images = []

    for path in img_paths:
        lids_images.append(load_dicom_image(path))
    nodules = []

    for path in ann_paths:
        tree = lxml.parse(path)
        root_node = tree.getroot()

        for roi_node in root_node.iter('roi'):
            ann_id = roi_node.find('imageSOP_UID').text
            ann_slice = None
            ann_x = int(roi_node.find('edgeMap').find('xCoord').text)
            ann_y = int(roi_node.find('edgeMap').find('yCoord').text)
            slice_node = roi_node.find('imageZposition')
            if slice_node is not None:
                ann_slice = Decimal(slice_node.text)

            img = None
            for i in lids_images:
                if i.id == ann_id and (ann_slice is None or i.slice_location == ann_slice):
                    img = i

            if img is None:
                if debug:
                    print("Image for nodule " + ann_id + " not found.")
            else:
                nodule = Nodule()

                nodule.size = 64
                nodule.pixels = crop_image(img.pixels, ann_x, ann_y, nodule.size)
                nodule.source_id = ann_id
                nodule.source_slice = ann_slice
                nodule.source_x = ann_x
                nodule.source_y = ann_y
                nodule.source_path = img.fullpath

                nodules.append(nodule)

    return nodules

def load_nsclc(dataset_path, debug):    img_paths = get_all_files(dataset_path, '.dcm')
    ann_paths = get_all_files(dataset_path, '.xml')

    lids_images = []

    for path in img_paths:
        lids_images.append(load_dicom_image(path))

    nodules = []

    for path in ann_paths:
        tree = lxml.parse(path)
        root_node = tree.getroot()

        uid_node = root_node.find('.//{gme://caCORE.caCORE/4.4/edu.northwestern.radiology.AIM}sopInstanceUid')
        x_node = root_node.find('.//{gme://caCORE.caCORE/4.4/edu.northwestern.radiology.AIM}x')
        y_node = root_node.find('.//{gme://caCORE.caCORE/4.4/edu.northwestern.radiology.AIM}y')

        ann_id = uid_node.attrib['root']
        ann_x = int(float(x_node.attrib['value']))
        ann_y = int(float(y_node.attrib['value']))

        img = None
        for i in lids_images:
            if i.id == ann_id:
                img = i

        if img is None:
            if debug:
                print("Image for nodule " + ann_id + " not found.")
        else:
            nodule = Nodule()

            nodule.size = 64
            nodule.pixels = crop_image(img.pixels, ann_x, ann_y, nodule.size)
            nodule.source_id = ann_id
            nodule.source_slice = img.slice_location
            nodule.source_x = ann_x
            nodule.source_y = ann_y
            nodule.source_path = img.fullpath

            nodules.append(nodule)

    return nodules