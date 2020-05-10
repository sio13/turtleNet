import numpy as np

from itertools import islice
from keras.datasets import mnist, cifar10
import matplotlib.pyplot as plt


def chunk(it, size: int):
    iter_list = iter(it)
    return iter(lambda: tuple(islice(iter_list, size)), ())


def save_collage(filepath: str,
                 array: np.array,
                 rows: int,
                 columns: int,
                 width: int = 28,
                 height: int = 28,
                 color: int = 1):
    array = array.reshape(array.shape[0], width, height, color)
    fig, axs = plt.subplots(rows, columns)
    cnt = 0
    for i in range(rows):
        for j in range(columns):
            axs[i, j].imshow(np.squeeze(array[cnt]))
            axs[i, j].axis('off')
            cnt += 1
    fig.savefig(f"{filepath}.png")
    plt.close()


def save_image(filepath: str,
               array: np.array):
    plt.imshow(np.squeeze(array))
    plt.axis('off')
    plt.savefig(f"{filepath}.png")


def save_image_and_collage(dir_path: str,
                           image_name: str,
                           array: np.array,
                           image_type: str,
                           rows: int,
                           columns: int,
                           sample_image_index: int = 0
                           ):
    save_image(f"{dir_path}/{image_name}_{image_type}_image",
               array[sample_image_index])

    save_collage(f"{dir_path}/{image_name}_{image_type}_collage",
                 array[:9],
                 rows,
                 columns,
                 array.shape[1],
                 array.shape[2],
                 array.shape[3])
    print(f"Saving image {dir_path}/{image_name}_{image_type}_image")
    print(f"Saving collage {dir_path}/{image_name}_{image_type}_collage")


def get_keras_dataset(data: tuple, input_shape=(-1, 28, 28, 1)) -> tuple:
    (x_train, y_train), (x_test, y_test) = data
    x_train = (x_train / 255).reshape(input_shape)
    x_test = (x_test / 255).reshape(input_shape)

    return x_train, y_train, x_test, y_test
