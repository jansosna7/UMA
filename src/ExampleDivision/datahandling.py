import random
from copy import copy
from enum import Enum, auto

import numpy as np
from sklearn.datasets import load_breast_cancer


class FillDataStrategy(Enum):
    MEAN = auto()
    MEDIAN = auto()
    MOST_FREQUENT = auto()


def load_data(dataset_name):
    if dataset_name.lower() == 'cancer':
        return load_breast_cancer()
    raise TypeError


def create_missing_data(dataset, malformed_rows_percentage=10, malformed_in_row=1):
    rows, cols = dataset.shape
    assert 0 <= malformed_rows_percentage <= 100
    assert malformed_in_row < cols

    missing_limit = (rows * malformed_rows_percentage) // 100
    row_indices = random.sample(range(rows), missing_limit)
    for missing in row_indices:
        remove = random.sample(range(cols), malformed_in_row)
        for cell in range(malformed_in_row):
            dataset[missing][remove] = np.nan


def fill_missing_data(dataset, strategy):
    rows, cols = dataset.shape
    for col_index in range(cols):
        column = dataset[:, col_index]
        replacement = get_replacement_value(column, strategy)
        column[np.isnan(column)] = replacement


def get_replacement_value(column, strategy):
    if strategy is FillDataStrategy.MEAN:
        return np.nanmean(column)
    if strategy is FillDataStrategy.MEDIAN:
        return np.nanmedian(column)
    if strategy is FillDataStrategy.MOST_FREQUENT:
        unique, counts = np.unique(column, return_counts=True)
        values_frequency = dict(zip(unique, counts))
        return max(values_frequency, key=values_frequency.get)
    raise TypeError('Unsupported strategy')


def replace_nans_with_fractionals(dataset):
    rows, cols = dataset.shape
    fractions = [get_fractional_examples(dataset[:, id]) for id in range(cols)]
    for i in range(0, len(dataset)):
        row = dataset[i]
        if (np.isnan(row).any()):
            nan_col = np.isnan(row)
            id = 0
            for na in nan_col:
                if na:
                    nan_col = id
                    break
                id += 1

            tmp_row = row
            dataset = np.delete(dataset, i, axis=0)

            for value, weight in fractions[nan_col]:
                tmp_row = copy(row)
                tmp_row[nan_col] = value
                tmp_row[len(tmp_row) - 1] = weight

                dataset = np.append(dataset, [tmp_row], axis=0)

    return dataset

def get_fractional_examples(column):
    n = len(column)
    mask = np.logical_not(np.isnan(column))
    column = column[mask]
    unique, count = np.unique(column, return_counts=True)
    counts = []
    i = 0
    for x in unique:
        counts.append([x,count[i]/n])
        i += 1
    return counts
