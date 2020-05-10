import numpy as np

# pozor vazna chyba treba opravit
def threshold_data(dataset_source: np.array,
                   threshold: float = None,
                   threshold_ratio_value: float = 0.5):
    dataset = np.array(dataset_source)
    if threshold is None:
        threshold = (dataset.max() - dataset.min()) * threshold_ratio_value + dataset.min()
    dataset[dataset < threshold] = 0
    dataset[dataset >= threshold] = 1
    return dataset
