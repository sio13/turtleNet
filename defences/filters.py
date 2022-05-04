import numpy as np
import scipy

def threshold_data(dataset_source: np.array,
                   threshold: float = None,
                   threshold_ratio_value: float = 0.5, **kwargs) -> np.array:
    dataset = np.array(dataset_source)
    if threshold is None:
        threshold = (dataset.max() - dataset.min()) * threshold_ratio_value + dataset.min()
    dataset[dataset < threshold] = 0
    dataset[dataset >= threshold] = 1
    return dataset


def mean_filter(input_array: np.array,
                size_of_filter: int = 4, **kwargs) -> np.array:
    return np.array(list(map(lambda x: scipy.ndimage.median_filter(x, size_of_filter), np.array(input_array))))
