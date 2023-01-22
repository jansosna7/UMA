'''
*===========================================================================*
*				Authors: Jan Sosnowski & Marcin Latawiec                    *
*===========================================================================*
'''

#Script for experiments



import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from src.ProbabilisticPrediction.decision_tree import OurTree
import os

from src.ExampleDivision.tree_exampledivision import Examplesplitter
from src.ExampleDivision.datahandling import load_data

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree



'''
Files: 
- breast-cancer.csv
- customer_data.csv
- rice.csv
'''

dirname = os.path.dirname(__file__)
filename_cancer = os.path.join(dirname, '../resources/breast-cancer.csv')
filename_customerdata = os.path.join(dirname, '../resources/customer_data.csv')
filename_rice = os.path.join(dirname, '../../resources/rice.csv')

#data handlers

''' breast cancer data '''

#columns_names_cancer =  ["radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean","compactness_mean","concavity_mean", "concave points_mean", "symmetry_mean", "fractal_dimension_mean","radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se", "compactness_se", "concavity_se", "concave points_se","symmetry_se", "fractal_dimension_se", "radius_worst", "texture_worst", "texture_worst", "area_worst", "smoothness_worst", "compactness_worst", "concavity_worst", "concave points_worst", "symmetry_worst","fractal_dimension_worst"]
#data = pd.read_csv(filename_cancer, skiprows=1, header=None, names=columns_names_cancer)
#data_order = np.linspace(2, 31, num = 30, dtype = int).tolist() + [1]
#data = data[data_order]


''' customer_data '''
#columns_names_customer_data = ["fea_1", "fea_2", "fea_3", "fea_4", "fea_5", "fea_6", "fea_7", "fea_8", "fea_9", "fea_10", "fea_11"]
#data = pd.read_csv(filename_customerdata, skiprows=1, header=None, names=columns_names_customer_data)


'''rice dataset'''
columns_names_rice_data = ["id", "Area", "MajorAxisLength", "MinorAxisLength", "Eccentricity", "ConvexArea", "EquivDiameter", "Extent", "Perimeter", "Roundness", "AspectRation", "Class"]
data = pd.read_csv(filename_rice, skiprows=1, header=None, names=columns_names_rice_data)
data = data[["MajorAxisLength", "MinorAxisLength", "Eccentricity", "ConvexArea", "EquivDiameter", "Extent", "Perimeter", "Roundness", "AspectRation", "Class"]]
data.head(1000)

X = data.iloc[:, :-1].values
Y = data.iloc[:, -1].values.reshape(-1,1)


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=80)


''' sklearn algorithm '''
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
model = classifier.fit(X_train, Y_train)

text_representation = tree.export_text(classifier)
print(text_representation)
Y_pred_ref = model.predict(X_test)


print("Dokladnosc modelu sklearn : ", accuracy_score(Y_test, Y_pred_ref))


