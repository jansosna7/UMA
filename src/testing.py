import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from decision_tree import OurTree

data = pd.read_csv('resources/customer_data.csv')

data = data.head(100)

X = data.drop('label', axis=1)
y = data['label']

X = X.values
y = y.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)#, random_state=42)


clf = OurTree()

clf.fit(X_train,y_train)


accuracy = clf.score(X_test, y_test)

print(accuracy)