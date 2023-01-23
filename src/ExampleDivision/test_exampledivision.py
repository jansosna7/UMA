'''
*===========================================================================*
*				Authors: Jan Sosnowski & Marcin Latawiec                    *
*===========================================================================*
'''
from random import randrange

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from decision_tree_exampledivision import Examplesplitter
from datahandling import *
import os.path
from random import randrange
import sys
sys.path.append("/home/kali/PycharmProjects/UMA")


from src.ExampleDivision.decision_tree_fractional import OurTreeFractional
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from src.ProbabilisticPrediction.decision_tree import OurTree
from src.ProbabilisticPrediction.data_deleter import MissingValuesCreator
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix

def main():
    np.set_printoptions(precision=4)

    dirname = os.path.dirname(__file__)
    filename_cancer = os.path.join(dirname, '../../resources/breast-cancer.csv')
    filename_customerdata = os.path.join(dirname, '../../resources/customer_data.csv')
    filename_rice = os.path.join(dirname, '../../resources/rice.csv')

    '''csv file handler'''

    # columns_names_cancer =  ["radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean","compactness_mean","concavity_mean", "concave points_mean", "symmetry_mean", "fractal_dimension_mean","radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se", "compactness_se", "concavity_se", "concave points_se","symmetry_se", "fractal_dimension_se", "radius_worst", "texture_worst", "texture_worst", "area_worst", "smoothness_worst", "compactness_worst", "concavity_worst", "concave points_worst", "symmetry_worst""fractal_dimension_worst"]
    columns_names_customer_data = ["label", "id", "fea_1", "fea_2", "fea_3", "fea_4", "fea_5", "fea_6", "fea_7",
                                   "fea_8", "fea_9", "fea_10", "fea_11"]
    data = pd.read_csv(filename_customerdata, skiprows=1, header=None, names=columns_names_customer_data)
    data = data.sample(frac = 1)
    data = data.head(400)
    # print(data)
    data = data[
        ["fea_1", "fea_2", "fea_3", "fea_4", "fea_5", "fea_6", "fea_7", "fea_8", "fea_9", "fea_10", "fea_11", "label"]]
    # print(data)

    data['zzz'] = 1
    data = data.values
    data = replace_nans_with_fractionals(data)
    #print(data)

    Y = data[:,-2]
    X = np.delete(data, -2, axis=1)


    avg_accuracy = 0
    number_of_iterations = 3
    for n in range(number_of_iterations):
        seed = randrange(1, 1000)
        xz_train, xz_test, y_train, y_test = train_test_split(X, Y, test_size=.2, random_state=seed)
        z_train = xz_train[:, -1]
        x_train = xz_train[:, :-1]

        z_test = xz_test[:, -1]
        x_test = xz_test[:, :-1]

        """create_missing_data(x_train, malformed_rows_percentage=10, malformed_in_row=1)
        fill_missing_data(x_train, FillDataStrategy.MEAN)

        create_missing_data(x_test, malformed_rows_percentage=10, malformed_in_row=1)

        classifier = Examplesplitter(x_train, y_train)"""

        classifier = OurTreeFractional(8)


        classifier.fit(x_train,y_train,z_train)

        predicted_classes = []
        predicted_classes = classifier.predict(x_test)
        avg_accuracy += accuracy_score(y_test, predicted_classes)
        print(avg_accuracy/(n+1))

    print(avg_accuracy / number_of_iterations)


if __name__ == '__main__':
    main()
