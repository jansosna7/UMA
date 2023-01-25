'''
*===========================================================================*
*				Authors: Jan Sosnowski & Marcin Latawiec                    *
*===========================================================================*
'''
import sys


sys.path.append("/home/kali/PycharmProjects/UMA")

from random import randrange

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from src.ExampleDivision import *
import os.path
from random import randrange
from src.ExampleDivision.datahandling import replace_nans_with_fractionals



from decision_tree_fractional import OurTreeFractional
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from decision_tree import OurTree
from data_deleter import MissingValuesCreator
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix

def main():
    depth = 8 #max depth for all trees
    n = 100 #how many rows from data
    data_sets = ['breast-cancer.csv','customer_data.csv','rice.csv']
    dirname = os.path.dirname(__file__)
    data_chose_iterations = 3
    test_iterations = 3

    #init
    ourClf = OurTree(depth)
    clf = DecisionTreeClassifier(max_depth=depth)
    X_train = None
    Y_train = None

    for data_set in data_sets:
        label_column = 0
        if data_set == 'breast-cancer.csv':
            label_column = 1
        elif data_set == 'customer_data.csv':
            label_column = 0
        elif data_set == 'rice.csv':
            label_column = 11
        path = '../../resources/'+data_set
        filename = os.path.join(dirname, path)
        print(filename)
        raw_data = pd.read_csv(filename, skiprows=1, header=None, sep=',')
        if data_set == 'breast-cancer.csv':
            raw_data = raw_data.replace({'M':0, 'B':1})

        for choose_rows in range(0,data_chose_iterations):
            data = raw_data.sample(frac=1)
            data = data.head(n)
            data['z'] = 1
            data = data.values
            #print(data)
            seed = randrange(1, 100)
            train, test, tmp_train, tmp_test = train_test_split(data, data[:,0], test_size=.2, random_state=seed)

            #prepare train datasets accordingly to build method
            for build_method in ['skip','mean','median','most_frequent','fractionals']:
                ourClf = OurTree(depth)
                clf = DecisionTreeClassifier(max_depth=depth)
                if build_method == 'skip':
                    X_train = np.array(train, dtype=float)
                    nan_rows = np.isnan(X_train).any(axis=1)

                    # delete the rows that contain NaN values
                    X_train = np.delete(X_train, np.where(nan_rows)[0], axis=0)


                elif build_method in ['mean','median','most_frequent']:
                    imputer = SimpleImputer(strategy=build_method)
                    X_train = imputer.fit_transform(train)


                elif build_method == 'fractionals':
                    x_train = replace_nans_with_fractionals(train)
                    ourClf = OurTreeFractional(depth)

                Y_train = X_train[:,label_column]
                X_train = np.delete(X_train, label_column, axis=1)



                #prepare train datasets accordingly to build method
                for predict_method in ['skip', 'mean', 'median', 'most_frequent', 'fractionals']:
                    if predict_method == 'skip':
                        #problem solved in _predict
                        X_test = test

                    elif predict_method in ['mean', 'median', 'most_frequent']:
                        imputer = SimpleImputer(strategy=predict_method)
                        X_test = imputer.fit_transform(test)

                    elif predict_method == 'fractionals':
                        X_test = replace_nans_with_fractionals(test)

                    Y_test = X_test[:,label_column]
                    X_test = np.delete(X_test, label_column, axis=1)
                    print(build_method, " ", predict_method)
                    for iteration in range(0,test_iterations):
                        ourClf.fit(X_train,Y_train)
                        y_pred = ourClf.predict(X_test)
                        score = accuracy_score(Y_test, y_pred)
                        print(score, " ", end="")
                    clf.fit(X_train, Y_train)
                    y_pred = clf.predict(X_test)
                    score = accuracy_score(Y_test, y_pred)
                    print("library score: ",score)

if __name__ == '__main__':
    main()
