'''
*===========================================================================*
*				Authors: Jan Sosnowski & Marcin Latawiec                    *
*===========================================================================*
'''

import numpy as np
import pandas as pd
from collections import Counter


class OurTreeFractional:
    def __init__(self, max_depth=None):
        self.tree = None
        self.max_depth = max_depth

    def fit(self, X, y, z=None):
        """Choose dealing with NaN method and start building tree"""

        if z is None:
            z = X[:,:-1]
            X = np.delete(X, -1, axis=1)

        self.tree = self._create_tree(X, y, z, depth=0)

        # print(self.tree)

    def predict(self, X):
        return np.array([self._predict(x, self.tree) for x in X])

    def score(self, X_test, y_test):
        """Calculate the accuracy of the model on a test set."""
        predictions = self.predict(X_test)
        accuracy = (predictions == y_test).mean()
        return accuracy

    def _create_tree(self, X, y, z, depth):
        """Build the decision tree recursively."""

        # Check if reached max depth
        if self.max_depth is not None and depth >= self.max_depth:
            return self._majority_class(y)

        # If all the samples belong to the same class, return that class
        if len(np.unique(y)) == 1:
            return y[0]

        # If there are no more features to split on, return the most common class
        if X.shape[1] == 0:
            return Counter(y).most_common(1)[0][0]

        # Select the feature and value to split on
        feature, value = self._choose_split(X, y, z)

        # Split the data into subsets
        X_left, y_left, X_right, y_right, z_left, z_right = self._split_data(X, y, z, feature, value)

        # Build the left and right subtrees
        left_tree = self._create_tree(X_left, y_left, z_left, depth + 1)
        right_tree = self._create_tree(X_right, y_right, z_right, depth + 1)

        # Return the tree as a dictionary
        return {feature: {value: (left_tree, right_tree)}}

    # todo use fractianals here
    def _majority_class(self, y):
        count = Counter(y)
        return max(count, key=count.get)

    def _choose_split(self, X, y, z):
        """Select the feature and value to split on."""
        best_feature = None
        best_value = None
        max_gain = -float("inf")

        for feature in range(X.shape[1]):
            for value in set(X[:, feature]):
                gain = self._information_gain(X, y, z, feature, value)
                if gain > max_gain:
                    max_gain = gain
                    best_feature = feature
                    best_value = value

        return best_feature, best_value

    def _split_data(self, X, y, z, feature, value):
        """Split the data into subsets based on the selected feature and value."""
        X_left = []
        y_left = []
        X_right = []
        y_right = []
        z_left = []
        z_right = []

        for i in range(X.shape[0]):
            if X[i, feature] < value:
                X_left.append(X[i])
                y_left.append(y[i])
                z_left.append(z[i])

            else:
                X_right.append(X[i])
                y_right.append(y[i])
                z_right.append(z[i])

        return np.array(X_left), np.array(y_left), np.array(X_right), np.array(y_right), np.array(z_left), np.array(
            z_right)

    def _information_gain(self, X, y, z, feature, value):
        """Calculate the information gain for a given feature and value."""
        # Split the data into subsets based on the selected feature and value
        X_left, y_left, X_right, y_right, z_left, z_right = self._split_data(X, y, z, feature, value)

        # Calculate the entropy before the split
        entropy_before = self._entropy(y, z)

        # Calculate the weighted average entropy after the split
        entropy_left = self._entropy(y_left, z_left)
        entropy_right = self._entropy(y_right, z_right)
        weighted_entropy = (len(y_left) / len(y)) * entropy_left + (len(y_right) / len(y)) * entropy_right

        # Calculate the information gain
        gain = entropy_before - weighted_entropy

        return gain

    def _entropy(self, y, z):
        """Calculate the entropy of a given set of labels."""
        # Count the occurrences of each label
        count = Counter(y)
        weight = [sum(z[np.where(y == name)]) / len(np.where(y == name)) for name, value in count.items()]

        id_dict = {}
        i = 0
        for name, value in count.items():
            id_dict[name] = i
            i += 1

        # Calculate the probability of each label
        probabilities = [value / sum(z) for name, value in count.items()]

        # Calculate the entropy
        entropy = sum(-p * np.log2(p) for p in probabilities)

        return entropy

    def _predict(self, x, subtree):
        """Recursivly check what label are we assigning to x"""

        # If the subtree is a leaf node, return the class

        if isinstance(subtree, int) or isinstance(subtree, float) or isinstance(subtree, np.int64) or isinstance(subtree, np.float64):
            return subtree

        # Get the feature and value of the current node
        feature, value = list(subtree.keys())[0], list(subtree.values())[0]
        # If the feature value of the sample is less than the value of the current node, move to the left subtree
        if x[feature] < list(value)[0]:
            subtree = subtree[feature][list(value)[0]][0]
        else:
            # Otherwise, move to the right subtree
            subtree = subtree[feature][list(value)[0]][1]

        # Recursively call the predict function with the subtree
        return self._predict(x, subtree)
