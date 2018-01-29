import os as os
import numpy as np
import medpy.io as medpy
import lxml.etree as lxml
import pandas as pd
import gzip
import pickle
from decimal import *

SUPPORTED_DATASETS = ['LIDC', 'NSCLC', 'SPIE']


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
        self.conclusion = False


def load_nodules(dataset_path, dataset_type, image_size=32, debug=False):
    """ Load dataset (which has type DATASET_TYPE) from DATASET_PATH recursively.

     DATASET_PATH - path to dataset.
     DATASET_TYPE - type of dataset. Look SUPPORTED_DATASETS for available dataset types.
     IMAGE_SIZE - image size of nodule.
     DEBUG - activates debug messages.

     Returns list of Nodules.
    """

    full_path = os.path.realpath(dataset_path)

    if dataset_type == 'LIDC':
        return load_lidc(full_path, image_size, debug)
    elif dataset_type == 'NSCLC':
        return load_nsclc(full_path, image_size, debug)
    elif dataset_type == 'SPIE':
        return load_spie(full_path, image_size, debug)
    else:
        raise Exception('Dataset_type is wrong')


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
    fix_pixels = np.vectorize(lambda p: max(p, 0.0))
    return fix_pixels(pixels[center_x - s: center_x + s, center_y - s: center_y + s])


class DicomImage:
    def __init__(self):
        self.id = ""
        self.fullpath = ""
        self.slice_location = Decimal(0)
        self.instance_number = Decimal(0)
        self.pixels = np.array([])


def load_dicom_image(path):
    image = DicomImage()

    pixels, header = medpy.load(path)

    image.pixels = pixels
    image.id = header.data_element("SOPInstanceUID").value
    image.fullpath = path
    image.slice_location = Decimal(header.get("SliceLocation", 0))
    image.instance_number = Decimal(header.get("InstanceNumber", 0))


    return image


def load_lidc(dataset_path, image_size, debug):
    patient_results = load_lidc_conclusions(os.path.join(dataset_path, 'annotation.xls'))

    subpaths = [os.path.join(dataset_path, p) for p in os.listdir(dataset_path)]
    subdirs = [p for p in subpaths if os.path.isdir(p)]

    nodules = []

    for subdir in subdirs:
        conclusion = patient_results[os.path.basename(subdir)]

        img_paths = get_all_files(subdir, '.dcm')
        ann_paths = get_all_files(subdir, '.xml')

        lids_images = []

        for path in img_paths:
            lids_images.append(load_dicom_image(path))

        for path in ann_paths:
            tree = lxml.parse(path)
            root_node = tree.getroot()

            for roi_node in root_node.iter('{*}roi'):
                if roi_node.find('{*}imageSOP_UID') is None:
                    print(path)

                ann_id = roi_node.find('{*}imageSOP_UID').text
                ann_slice = None

                n = 0
                x = 0.0
                y = 0.0

                for coord_node in roi_node.iter('{*}edgeMap'):
                    n += 1
                    x += float(coord_node.find('{*}xCoord').text)
                    y += float(coord_node.find('{*}yCoord').text)

                ann_x = int(x / n)
                ann_y = int(y / n)

                slice_node = roi_node.find('{*}imageZposition')
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

                    nodule.size = image_size
                    nodule.pixels = crop_image(img.pixels, ann_x, ann_y, nodule.size)
                    nodule.source_id = ann_id
                    nodule.source_slice = ann_slice
                    nodule.source_x = ann_x
                    nodule.source_y = ann_y
                    nodule.source_path = img.fullpath
                    nodule.conclusion = conclusion

                    nodules.append(nodule)

    return nodules


def load_nsclc(dataset_path, image_size, debug):
    img_paths = get_all_files(dataset_path, '.dcm')
    ann_paths = get_all_files(dataset_path, '.xlsx')

    lids_images = []

    for path in img_paths:
        lids_images.append(load_dicom_image(path))

    nodules = []

    for path in ann_paths:
        tree = lxml.parse(path)
        root_node = tree.getroot()

        uid_node = root_node.find('.//{*}sopInstanceUid')
        x_node = root_node.find('.//{*}x')
        y_node = root_node.find('.//{*}y')

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

            nodule.size = image_size
            nodule.pixels = crop_image(img.pixels, ann_x, ann_y, nodule.size)
            nodule.source_id = ann_id
            nodule.source_slice = img.slice_location
            nodule.source_x = ann_x
            nodule.source_y = ann_y
            nodule.source_path = img.fullpath
            nodule.conclusion = True

            nodules.append(nodule)

    return nodules


def load_spie(dataset_path, image_size, debug):
    ann_path = os.path.join(dataset_path, 'spie_annotation.xlsx')

    f = open(ann_path, "rb")
    spie_annotation = pd.read_excel(f)
    f.close()

    nodules = []

    for i, row in spie_annotation.iterrows():
        patient_name = row['Scan Number']

        if debug:
            print 'Loading ', patient_name

        ann_instance_number = row['Nodule Center Image']

        ann_x = int(row['Nodule Center x,y Position*'])
        ann_y = int(row['Nodule Center x,y Position*'])

        conclusion = row['Final Diagnosis']
        if conclusion == 'Benign nodule':
            ann_conclusion = False
        else:
            ann_conclusion = True

        subdir = os.path.join(dataset_path, patient_name)

        img_paths = get_all_files(subdir, '.dcm')

        spie_images = []

        for path in img_paths:
            spie_images.append(load_dicom_image(path))

        img = None
        for i in spie_images:
            if ann_instance_number == i.instance_number:
                img = i

        nodule = Nodule()

        nodule.size = image_size
        nodule.pixels = crop_image(img.pixels, ann_x, ann_y, nodule.size)
        nodule.source_id = img.id
        nodule.source_path = img.fullpath
        nodule.source_slice = ann_instance_number
        nodule.source_x = ann_x
        nodule.source_y = ann_y
        nodule.conclusion = ann_conclusion

        nodules.append(nodule)

    return nodules


def dump_nodules(file_path, features):
    f = gzip.open(file_path, "wb")

    pickle.dump(features, f)

    f.close()


def restore_nodules(file_path):
    f = gzip.open(file_path, "rb")

    nodules = pickle.load(f)

    f.close()

    return nodules


def load_lidc_conclusions(path):
    f = open(path, "rb")
    df = pd.read_excel(f)
    f.close()

    patient_ids = np.array(df['TCIA Patient ID'], dtype=str)
    conclusions = np.array(df['Conclusion'])

    return dict(zip(patient_ids, conclusions))
