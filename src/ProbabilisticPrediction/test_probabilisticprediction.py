import os.path
from random import randrange
import sys
sys.path.append("/home/kali/PycharmProjects/UMA")
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

dirname = os.path.dirname(__file__)
filename_cancer = os.path.join(dirname, '../../resources/breast-cancer.csv')
filename_customerdata = os.path.join(dirname, '../../resources/customer_data.csv')
filename_rice = os.path.join(dirname, '../../resources/rice.csv')

'''csv file handler'''

#columns_names_cancer =  ["radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean","compactness_mean","concavity_mean", "concave points_mean", "symmetry_mean", "fractal_dimension_mean","radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se", "compactness_se", "concavity_se", "concave points_se","symmetry_se", "fractal_dimension_se", "radius_worst", "texture_worst", "texture_worst", "area_worst", "smoothness_worst", "compactness_worst", "concavity_worst", "concave points_worst", "symmetry_worst""fractal_dimension_worst"]
columns_names_customer_data = ["label","id","fea_1", "fea_2", "fea_3", "fea_4", "fea_5", "fea_6", "fea_7", "fea_8", "fea_9", "fea_10", "fea_11"]
data = pd.read_csv(filename_customerdata, skiprows=1, header=None, names=columns_names_customer_data)
data = data.head(1000)
#print(data)
data = data[["fea_1", "fea_2", "fea_3", "fea_4", "fea_5", "fea_6", "fea_7", "fea_8", "fea_9", "fea_10", "fea_11", "label"]]
#print(data)

X = data.iloc[:, :-1].values
Y = data.iloc[:, -1].values

seed = randrange(1, 100)
number_of_iterations = 3
list_of_atributes_missing_values = [6]
avg_accuracy_full_classifier = 0
avg_accuracy_naive_classifier = 0
avg_accuracy_classifier_without_missing_values = 0


for n in range(number_of_iterations):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.25, random_state=seed)

    classifier = OurTree(16)
    classifier.fit(X_train,Y_train)

    full_classifier = OurTree(16)
    full_classifier.fit(X_train, Y_train)

    naive_classifier = OurTree(16)  # naive probabilistic approach to missing values
    naive_classifier.fit(X_train, Y_train)

    Y_pred_1 = classifier.predict(X_test) # Prediction without missing values


    accuracy = accuracy_score(Y_test, Y_pred_1)
    avg_accuracy_classifier_without_missing_values += accuracy

    missing_values_creator = MissingValuesCreator(percent=50) #removing data initializer
    X_test_missing = missing_values_creator.delete_random_values_from_given_columns(X_test, list_of_atributes_missing_values)
    Y_pred_2 = naive_classifier.predict(X_test_missing)
    Y_pred_3 = full_classifier.predict(X_test_missing)


    accuracyfull = accuracy_score(Y_test, Y_pred_2)
    avg_accuracy_full_classifier += accuracyfull

    accuracynaive = accuracy_score(Y_test, Y_pred_3)
    avg_accuracy_naive_classifier += accuracynaive

    seed = randrange(1, 1000) #new seed


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
    cm = confusion_matrix(Y_test, Y_pred_1)
    disp1 = ConfusionMatrixDisplay(cm)
    disp1.plot()
    cm = confusion_matrix(Y_test, Y_pred_2)
    disp2= ConfusionMatrixDisplay(cm)
    disp2.plot()
    cm = confusion_matrix(Y_test, Y_pred_3)
    disp3 = ConfusionMatrixDisplay(cm)
    disp3.plot()

    plt.show()

avg_accuracy_classifier_without_missing_values = avg_accuracy_classifier_without_missing_values/number_of_iterations
avg_accuracy_naive_classifier = avg_accuracy_naive_classifier/number_of_iterations
avg_accuracy_full_classifier = avg_accuracy_full_classifier/number_of_iterations

print("Classification without missing values: ",avg_accuracy_classifier_without_missing_values)
print("Classification with full probabilistic approach: ", avg_accuracy_full_classifier)
print("Classification with naive probabilistic approach: ", avg_accuracy_naive_classifier)
