import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

from decision_tree import OurTree

data = pd.read_csv('resources/customer_data.csv')
org_data = data

data = data.head(400)

clf = OurTree(5)


label = 'label'

X = data.drop(label, axis=1)
y = data[label]

X = X.values
y = y.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)#, random_state=42)




clf.fit(X_train,y_train,'skip')


accuracy = clf.score(X_test, y_test)

print("v1", accuracy)


X = org_data.drop(label, axis=1)
y = org_data[label]

imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

y = y.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)#, random_state=42)

clf.fit(X_train,y_train,'default')

accuracy = clf.score(X_test, y_test)

print("v2", accuracy)

clf0 = DecisionTreeClassifier(criterion='entropy')
clf0.fit(X_train, y_train)
y_pred = clf.predict(X_test)
#print(y_pred, y_test)
print("Accuracy basic: ", accuracy_score(y_test, y_pred))

X = org_data.drop(label, axis=1)
y = org_data[label]

imputer = SimpleImputer(strategy='median')
X = imputer.fit_transform(X)

y = y.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)#, random_state=42)

clf.fit(X_train,y_train,'default')

accuracy = clf.score(X_test, y_test)

print("v3", accuracy)

X = org_data.drop(label, axis=1)
y = org_data[label]

imputer = SimpleImputer(strategy='most_frequent')
X = imputer.fit_transform(X)

y = y.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)#, random_state=42)

clf.fit(X_train,y_train,'default')

accuracy = clf.score(X_test, y_test)

print("v4", accuracy)