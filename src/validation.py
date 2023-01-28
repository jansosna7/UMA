'''
*===========================================================================*
*				Authors: Jan Sosnowski & Marcin Latawiec                    *
*===========================================================================*
'''

#Script for experiments



import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from src.decision_tree_prob import OurTreeProb
from src.decision_tree import OurTree
import os
from sklearn.tree import DecisionTreeClassifier

'''
Files: 
- breast-cancer.csv
- customer_data.csv
- rice.csv
'''

dirname = os.path.dirname(__file__)
filename_cancer = os.path.join(dirname, '../resources/breast-cancer.csv')
filename_customerdata = os.path.join(dirname, '../resources/customer_data.csv')
filename_rice = os.path.join(dirname, '../resources/rice.csv')

#data handlers

''' breast cancer data ''' # 590 examples - full data set
'''
#columns_names_cancer =  ["radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean","compactness_mean","concavity_mean", "concave points_mean", "symmetry_mean", "fractal_dimension_mean","radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se", "compactness_se", "concavity_se", "concave points_se","symmetry_se", "fractal_dimension_se", "radius_worst", "texture_worst", "texture_worst", "area_worst", "smoothness_worst", "compactness_worst", "concavity_worst", "concave points_worst", "symmetry_worst","fractal_dimension_worst"]
data = pd.read_csv(filename_cancer, skiprows=1, header=None, sep= ',')
data_order = np.linspace(2, 31, num = 30, dtype = int).tolist() + [1]
data = data[data_order]
data = data.sample(frac=1)
data = data.replace({'M':0, 'B':1})
X = data.iloc[:, :-1].values
Y = data.iloc[:, -1].values
'''

''' customer_data ''' #1100 examples - missing data natively
#columns_names_customer_data = ["label", "id","fea_1", "fea_2", "fea_3", "fea_4", "fea_5", "fea_6", "fea_7", "fea_8", "fea_9", "fea_10", "fea_11"]
#data = pd.read_csv(filename_customerdata, skiprows=1, header=None, names=columns_names_customer_data)
#data = data[["fea_1","fea_2","fea_3","fea_4","fea_5","fea_6","fea_7","fea_8","fea_9", "fea_10", "fea_11", "label"]]
#data = data.sample(frac=1)
#data = data.head(1000)

#X = data.iloc[:, :-1].values
#Y = data.iloc[:, -1].values
#imputer = SimpleImputer(strategy='mean')
#X = imputer.fit_transform(X)

'''rice dataset''' # 20000 examples - full data set
columns_names_rice_data = ["id", "Area", "MajorAxisLength", "MinorAxisLength", "Eccentricity", "ConvexArea", "EquivDiameter", "Extent", "Perimeter", "Roundness", "AspectRation", "Class"]
data = pd.read_csv(filename_rice, skiprows=1, header=None, names=columns_names_rice_data)
data = data[["MajorAxisLength", "MinorAxisLength", "Eccentricity", "ConvexArea", "EquivDiameter", "Extent", "Perimeter", "Roundness", "AspectRation", "Class"]]
data = data.sample(frac=1)
data = data.head(1000)

X = data.iloc[:, :-1].values
Y = data.iloc[:, -1].values


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.25, random_state=80)


''' sklearn algorithm '''

classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
model = classifier.fit(X_train, Y_train)
Y_pred_ref = model.predict(X_test)


np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
titles_options = [
    ("Confusion matrix, without normalization", None),
    ("Normalized confusion matrix", "true"),
]

cm = confusion_matrix(Y_test, Y_pred_ref)
disp1 = ConfusionMatrixDisplay(cm)
disp1.plot()
plt.show()



''' implemented algorithm - '''
classifier = OurTree(16)
classifier.fit(X_train,Y_train)
Y_pred = classifier.predict(X_test)

np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
titles_options = [
    ("Confusion matrix, without normalization", None),
    ("Normalized confusion matrix", "true"),
]

cm = confusion_matrix(Y_test, Y_pred_ref)
disp1 = ConfusionMatrixDisplay(cm)
disp1.plot()
plt.show()

print("Dokladnosc zaimplementowanego klasyfikatora decision_tree: ", accuracy_score(Y_test, Y_pred))
print("Dokladnosc modelu sklearn : ", accuracy_score(Y_test, Y_pred_ref))


