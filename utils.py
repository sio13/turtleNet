import numpy as np

from itertools import islice
from keras.datasets import mnist, cifar10
import matplotlib.pyplot as plt



def chunk(it, size: int):
    iter_list = iter(it)
    return list(iter(lambda: tuple(islice(iter_list, size)), ()))


def save_collage(filepath: str,
                 batch: np.array,
                 rows: int,
                 columns: int,
                 width: int = 28,
                 height: int = 28,
                 color: int = 1):
    batch = batch.reshape(batch.shape[0], width, height, color)
    fig, axs = plt.subplots(rows, columns)
    cnt = 0
    for i in range(rows):
        for j in range(columns):
            axs[i, j].imshow((batch[cnt]))
            axs[i, j].axis('off')
            cnt += 1
    fig.savefig(f"{filepath}.png")
    plt.close()


def get_keras_dataset(data: tuple, input_shape = (-1, 28, 28, 1)) -> tuple:
    (x_train, y_train), (x_test, y_test) = data
    x_train = x_train.reshape(input_shape)  / 255
    x_test = x_test.reshape(input_shape) / 255

    return x_train, y_train, x_test, y_test


