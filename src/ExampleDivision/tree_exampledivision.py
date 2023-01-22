'''
*===========================================================================*
*				Authors: Jan Sosnowski & Marcin Latawiec                    *
*===========================================================================*
'''


import numpy as np

class Examplesplitter:
    def __init__(self, x, y):
        self.__n_classes = len(set(y))
        self.__n_features = x.shape[1]
        self.__tree = self.__build_tree(x, y)

    def predict(self, x):
        node = self.__tree
        while node.left:
            if x[node.feature_index] is not None:
                if x[node.feature_index] < node.threshold:
                    node = node.left
                else:
                    node = node.right
            else:
                surrogates = sorted(range(len(node.index_best_gini)), key=lambda k: node.index_best_gini[k])[1:]
                next_surr = 0
                while x[surrogates[next_surr]] is None:
                    next_surr += 1
                if x[surrogates[next_surr]] < node.index_best_thresholds[surrogates[next_surr]]:
                    node = node.left
                else:
                    node = node.right
        return node.predicted_class

    def __build_tree(self, x, y):
        num_samples_per_class = [np.sum(y == i) for i in range(self.__n_classes)]
        predicted_class = np.argmax(num_samples_per_class)
        node = Node(predicted_class=predicted_class)

        idx, thr, best, best_g = self.__best_split(x, y)
        if idx is not None:
            node.feature_index = idx
            node.threshold = thr
            node.index_best_gini = best_g
            node.index_best_thresholds = best

            indices_left = x[:, idx] < thr
            x_left, y_left = x[indices_left], y[indices_left]
            x_right, y_right = x[~indices_left], y[~indices_left]
            node.left = self.__build_tree(x_left, y_left)
            node.right = self.__build_tree(x_right, y_right)

        return node

    def __best_split(self, x, y):
        if y.size <= 1:
            return None, None, None, None

        num_parent = [np.sum(y == c) for c in range(self.__n_classes)]
        best_gini = 1.0 - sum((n / y.size) ** 2 for n in num_parent)
        best_idx, best_thr = None, None
        index_best_thresholds = {k: None for k in range(self.__n_features)}
        index_best_gini = [best_gini for _ in range(self.__n_features)]
        for idx in range(self.__n_features):
            thresholds, classes = zip(*sorted(zip(x[:, idx], y)))
            num_left = [0] * self.__n_classes
            num_right = num_parent.copy()
            for i in range(1, y.size):
                c = classes[i - 1]
                num_left[c] += 1
                num_right[c] -= 1

                gini_left = 1.0 - sum((num_left[x] / i) ** 2 for x in range(self.__n_classes))
                gini_right = 1.0 - sum((num_right[x] / (y.size - i)) ** 2 for x in range(self.__n_classes))
                gini = (i * gini_left + (y.size - i) * gini_right) / y.size

                if thresholds[i] == thresholds[i - 1]:
                    continue

                if gini < index_best_gini[idx]:
                    index_best_gini[idx] = gini
                    index_best_thresholds[idx] = (thresholds[i] + thresholds[i - 1]) / 2

                if gini < best_gini:
                    best_gini = gini
                    best_idx = idx
                    best_thr = (thresholds[i] + thresholds[i - 1]) / 2
        return best_idx, best_thr, index_best_thresholds, index_best_gini


class Node:
    def __init__(self, predicted_class):
        self.predicted_class = predicted_class
        self.feature_index = 0
        self.threshold = 0
        self.index_best_thresholds = {}
        self.index_best_gini = []
        self.left = None
        self.right = None