
import random
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
