import numpy as np


def threshold_data(dataset: np.array,
                   threshold: float = None,
                   threshold_ratio_value: float = 0.5):
    if threshold is None:
        threshold = (dataset.max() - dataset.min()) * threshold_ratio_value + dataset.min()
    dataset[dataset < threshold] = 0
    dataset[dataset >= threshold] = 1
    return dataset
