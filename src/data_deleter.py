#Created by: Marcin Latawiec, Jan Sosnowski


import numpy as np
import random
import copy


class MissingValuesCreator:

    def __init__(self, percent=100):
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
                list_with_missing_values[int(index_of_missing_value)][index] = 'NaN'
                # deleting index from list of indexes of elements without missing values
                indices = np.where(indexes_of_not_missing_elements == index_of_missing_value)
                indexes_of_not_missing_elements = np.delete(indexes_of_not_missing_elements, indices)
                # indexes_of_not_missing_elements.remove(index_of_missing_value)

        return list_with_missing_values

    def delete_random_value_from_random_column(self,list):
        list_with_missing_values = copy.copy(list)  # making shallow copy of a given list
        index = random.randint(0, len(list[0]))
        no = random.randint(0, len(list))
        list_with_missing_values[no][index] = 'Nan'
        return list_with_missing_values
