'''
*===========================================================================*
*				Author: Marcin Latawiec                                    *
*===========================================================================*
'''

''' class for handling missing values by filling it'''

from enum import Enum, auto
import numpy as np
from sklearn.datasets import load_breast_cancer
import copy
class FillDataMethod(Enum):
    MEAN = auto()
    MEDIAN = auto()
    MODE = auto()

def load_data(dataset_name):
    if dataset_name.lower() == 'cancer':
        return load_breast_cancer()
    raise TypeError

class Filler:
    def __int__(self):
        pass

    def fill_missing_data(self, list, method):
        list_filled = copy.copy(list)
        rows, cols = list_filled.shape
        for col_index in range(cols):
            column = list_filled[:, col_index]
            replacement = self.get_replacement_value(column, method)
            column[np. isnan(column)] = replacement
        return list_filled


    def get_replacement_value(self, column, method):
        if method is FillDataMethod.MEAN:
            return np.nanmean(column)
        if method is FillDataMethod.MEDIAN:
            return np.nanmedian(column)
        if method is FillDataMethod.MODE:
            unique, counts = np.unique(column, return_counts=True)
            values_frequency = dict(zip(unique, counts))
            return max(values_frequency, key=values_frequency.get)
        raise TypeError('Unsupported method')

