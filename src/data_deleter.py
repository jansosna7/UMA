# Created by: Marcin Latawiec, Jan Sosnowski


import numpy as np
import random
'''
*===========================================================================*
*				Authors: Jan Sosnowski & Marcin Latawiec                    *
*===========================================================================*
'''

import copy


class MissingValuesCreator:
    #percent - of data which is to be deleted
    def __init__(self, percent=100): #
        self.percent = percent

    def change_percent(self, percent):
        self.percent = percent

    def delete_whole_given_columns(self, list, list_of_indexes):
        list_with_missing_columns = copy.copy(list)  # making shallow copy of a given list

        for index in list_of_indexes:

            # loop until given percentage of missing values will be obtained
            for n in range(len(list_with_missing_columns)):
                # filling with missing value
                list_with_missing_columns[n][index] = 'NaN'
                # deleting index from list of indexes of elements without missing values

        return list_with_missing_columns

    ## Rest of the functions is for creating pseudo-realistic dataset with missing values

    def delete_random_values_from_given_column(self, list, index):
        '''function adding missing values to given data at given index'''
        # index - index of parameter of element which will be missing

        list_with_missing_values = copy.copy(list)  # making shallow copy of a given list
        indexes_of_not_missing_elements = np.linspace(0, len(list) - 1, len(list))

        # loop until given percentage of missing values will be obtained
        while (len(indexes_of_not_missing_elements) > (len(list) * ((100 - self.percent) / 100))):
            # randomly selecting element of list (element without missing values)
            index_of_missing_value = random.choice(indexes_of_not_missing_elements)
            # filling with missing value
            list_with_missing_values[int(index_of_missing_value)][index] = 'NaN'
            # deleting index from list of indexes of elements without missing values
            indices = np.where(indexes_of_not_missing_elements == index_of_missing_value)
            indexes_of_not_missing_elements = np.delete(indexes_of_not_missing_elements, indices)
            # indexes_of_not_missing_elements.remove(index_of_missing_value)

        return list_with_missing_values

    def delete_random_values_from_given_columns(self, list, list_of_indexes):
        '''function adding missing values to given data at given index'''
        # list_of_indexes - list of indexes of parameters which will be missing
        list_with_missing_values = copy.copy(list)  # making shallow copy of a given list

        for index in list_of_indexes:
            indexes_of_not_missing_elements = np.linspace(0, len(list) - 1, len(list))

            # loop until given percentage of missing values will be obtained
            while (len(indexes_of_not_missing_elements) > (len(list) * ((100 - self.percent) / 100))):
                # randomly selecting element of list (element without missing values)
                index_of_missing_value = random.choice(indexes_of_not_missing_elements)
                # filling with missing value
                list_with_missing_values[int(index_of_missing_value)][index] = np.nan
                # deleting index from list of indexes of elements without missing values
                indices = np.where(indexes_of_not_missing_elements == index_of_missing_value)
                indexes_of_not_missing_elements = np.delete(indexes_of_not_missing_elements, indices)
                # indexes_of_not_missing_elements.remove(index_of_missing_value)

        return list_with_missing_values

    def delete_random_value_from_random_column(self, list):
        list_with_missing_values = copy.copy(list)  # making shallow copy of a given list
        index = random.randint(0, len(list[0]))
        no = random.randint(0, len(list))
        b = False
        if list_with_missing_values[no][index] != 'Nan':
            b = True
            list_with_missing_values[no][index] = 'Nan'

        return b, list_with_missing_values

    def delete_random_values_from_random_columns(self, list, n):
        list_with_missing_values = copy.copy(list)  # making shallow copy of a given list
        for i in range(0, n):
            b, list_with_missing_values = self.delete_random_value_from_random_column(list_with_missing_values)
            i = 0
            while not b:
                b, list_with_missing_values = self.delete_random_value_from_random_column(list_with_missing_values)
                i += 1
                if i > 1000:
                    raise Exception("Can't delete that many values!")

        return list_with_missing_values

    def delete_random_values_from_list_percentage(self, list):
        list_with_missing_values = copy.copy(list)  # making shallow copy of a given list
        count_all = len(list_with_missing_values) * len(list_with_missing_values[0])
        count_nan = 0
        for example in list_with_missing_values:
            for x in example:
                if x == 'Nan':
                    count_nan += 1
        if (count_nan / count_all) * 100 < self.percent:
            list_with_missing_values = self.delete_random_values_from_random_columns(list_with_missing_values,
                                                                                     count_all * self.percent / 100 - count_nan)
        return list_with_missing_values
