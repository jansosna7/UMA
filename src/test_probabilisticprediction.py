import os.path
from random import randrange
import sys
sys.path.append("/home/kali/PycharmProjects/UMA")
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from src.decision_tree_prob import OurTreeProb
from src.decision_tree import OurTree
from src.data_deleter import MissingValuesCreator
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix

dirname = os.path.dirname(__file__)
filename_cancer = os.path.join(dirname, '../resources/breast-cancer.csv')
filename_customerdata = os.path.join(dirname, '../resources/customer_data.csv')
filename_rice = os.path.join(dirname, '../resources/rice.csv')

'''csv file handlers'''

'''rice data'''
'''
columns_names_rice_data = ["id", "Area", "MajorAxisLength", "MinorAxisLength", "Eccentricity", "ConvexArea", "EquivDiameter", "Extent", "Perimeter", "Roundness", "AspectRation", "Class"]
data = pd.read_csv(filename_rice, skiprows=1, header=None, names=columns_names_rice_data)
data = data[["MajorAxisLength", "MinorAxisLength", "Eccentricity", "ConvexArea", "EquivDiameter", "Extent", "Perimeter", "Roundness", "AspectRation", "Class"]]
data = data.sample(frac=1)
data = data.head(1000)
'''

''' customer_data '''
'''
columns_names_customer_data = ["label", "id","fea_1", "fea_2", "fea_3", "fea_4", "fea_5", "fea_6", "fea_7", "fea_8", "fea_9", "fea_10", "fea_11"]
data = pd.read_csv(filename_customerdata, skiprows=1, header=None, names=columns_names_customer_data)
data = data[["fea_1","fea_2","fea_3","fea_4","fea_5","fea_6","fea_7","fea_8","fea_9", "fea_10", "fea_11", "label"]]
data = data.sample(frac=1)
data = data.head(1000)
'''
''' breast-cancer '''

#columns_names_cancer =  ["radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean","compactness_mean","concavity_mean", "concave points_mean", "symmetry_mean", "fractal_dimension_mean","radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se", "compactness_se", "concavity_se", "concave points_se","symmetry_se", "fractal_dimension_se", "radius_worst", "texture_worst", "texture_worst", "area_worst", "smoothness_worst", "compactness_worst", "concavity_worst", "concave points_worst", "symmetry_worst","fractal_dimension_worst"]
data = pd.read_csv(filename_cancer, skiprows=1, header=None, sep= ',')
data_order = np.linspace(2, 31, num = 30, dtype = int).tolist() + [1]
data = data[data_order]
data = data.sample(frac=1)
data = data.replace({'M':0, 'B':1})

X = data.iloc[:, :-1].values
Y = data.iloc[:, -1].values.reshape(-1,1)

seed = randrange(1, 100)

#spliting to training and testing set
#x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=.25, random_state=seed)




#refererncyjna dokładność
'''
classifier = OurTree(min_samples_split=3, max_depth=np.inf, predictor= 1)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
print(accuracy_score(y_test, y_pred))
'''

number_of_iterations = 5
list_of_atributes_missing_values = [2, 3, 5, 6]
avg_accuracy_classifier = 0
avg_accuracy_classifier_without_missing_values = 0
percentage = 50
'''
for n in range(number_of_iterations):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.25, random_state=seed)

    classifier = OurTree(min_samples_split=3, max_depth=np.inf, predictor=1)
    classifier.fit(X_train, Y_train)

    missing_values_creator = MissingValuesCreator(percent=percentage) #removing data initializer
    X_test_missing = missing_values_creator.delete_random_values_from_given_columns(X_test, list_of_atributes_missing_values)
    Y_pred = classifier.predict(X_test_missing)

    avg_accuracy_classifier += accuracy_score(Y_test, Y_pred) # accuracy with probabilistic prediction
    seed = randrange(1, 1000)  

avg_accuracy_classifier = avg_accuracy_classifier/number_of_iterations
print("Dokładność dla zbiorów o brakach wynoszących:", percentage, "Liczba brakujących atrybutów:", 2)
print(avg_accuracy_classifier)
'''

#CONFUSION MATRIX EXPERIMENT

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.25, random_state=seed)

classifier = OurTreeProb(min_samples_split=3, max_depth=np.inf, predictor=1)
classifier.fit(X_train, Y_train)
missing_values_creator = MissingValuesCreator(percent=percentage) #removing data initializer
X_test_missing = missing_values_creator.delete_random_values_from_given_columns(X_test, list_of_atributes_missing_values)
Y_pred = classifier.predict(X_test_missing)

np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
titles_options = [
    ("Confusion matrix, without normalization", None),
    ("Normalized confusion matrix", "true"),
]
'''for title, normalize in titles_options:

    disp = ConfusionMatrixDisplay.from_estimator(
        full_classifier,
        X_test,
        Y_test,
        display_labels=[0,1],
        cmap=plt.cm.Blues,
        normalize=normalize,
    )
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)'''
cm = confusion_matrix(Y_test, Y_pred)
disp1 = ConfusionMatrixDisplay(cm)
disp1.plot()
plt.show()




#cm = confusion_matrix(Y_test, Y_pred)
#disp1 = ConfusionMatrixDisplay(cm)
#disp1.plot()


