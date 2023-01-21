'''
*===========================================================================*
*				Author: Marcin Latawiec                                    *
*===========================================================================*
'''
from scipy.stats import rice

''' class for handling missing values by filling it'''

from enum import Enum, auto
import numpy as np

class FillDataMethod(Enum):
    MEAN = auto()
    MEDIAN = auto()
    MODE = auto()
class Filler:
    def __int__(self, unique_names_probability_dict, tree):
        self.probability_dict = unique_names_probability_dict
        self.tree = tree

    def fill_missing_data(self, dataset, method):
        rows, cols = dataset.shape
        for col_index in range(cols):
            column = dataset[:, col_index]
            replacement = self.get_replacement_value(column, method)
            column[np. isnan(column)] = replacement


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

