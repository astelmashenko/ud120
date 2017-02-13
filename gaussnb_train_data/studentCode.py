from class_vis import prettyPicture
from prep_terrain_data import makeTerrainData
from classify import NBAccuracy

import matplotlib.pyplot as plt
import numpy as np
import pylab as pl


features_train, labels_train, features_test, labels_test = makeTerrainData()
# accuracy - no. of points classified correctly / all points (in the test set)
# method #1: write code that compares predictions to y_test , element by element
# OR
# method #2: google sklearn accuracy ang go from there
def submitAccuracy():
    accuracy = NBAccuracy(features_train, labels_train, features_test, labels_test)
    return accuracy