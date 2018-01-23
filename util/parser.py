import numpy as np
from medpy.io import load
try:
    from lxml import etree
    print('Lidc parser running with lxml.etree')
except ImportError:
    print('vse ploxo')


class LidcParser:
    def __init__(self):
        self.dir = 'lidc-data/'

    def get_data(self, filename):
        pathToXml = self.dir + filename
        pathToImage = self.dir + '000' + filename[:3] + '.dcm'  # govnokod, slomaetsa esli xml name ne ***.dcm
        print('Now processing Xml: ', pathToXml, 'and image: ', pathToImage)

        nodulesMatrix = self.parse_file(pathToXml)
        image, header = load(pathToImage)
        return nodulesMatrix, image, header

    def parse_file(self, path_to_file):
        calculatedNodulesMatrix = []
        tree = etree.parse(path_to_file)
        root = tree.getroot()
        # extract regions of interest
        for regionOfInterest in root.iter('roi'):
            extractedCenterOfRoi = self.extract_center_of_roi(regionOfInterest)
            calculatedNodulesMatrix.append(extractedCenterOfRoi)
        return calculatedNodulesMatrix

    def extract_center_of_roi(self, regionOfInterest):
        xCoordArray = []
        yCoordArray = []
        for edgeMap in regionOfInterest.iter('edgeMap'):
            xCoord = edgeMap.find('xCoord').text
            yCoord = edgeMap.find('yCoord').text

            xCoordArray.append(int(xCoord))
            yCoordArray.append(int(yCoord))

        xCoordAverageOfRoi = int(round(np.average(xCoordArray)))  # i know its a govnokod, sooooo sorry
        yCoordAverageOfRoi = int(round(np.average(yCoordArray)))

        return [xCoordAverageOfRoi, yCoordAverageOfRoi]
