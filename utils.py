import numpy as np
import time

from itertools import islice
from keras.datasets import mnist, cifar10
import matplotlib.pyplot as plt

from keras.models import load_model


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
    """
    mean = np.mean(x_train,axis=(0,1,2,3))
    std = np.std(x_train,axis=(0,1,2,3))
    x_train = (x_train-mean)/(std+1e-7)
    x_test = (x_test-mean)/(std+1e-7)
    """
    (x_train, y_train), (x_test, y_test) = data
    x_train = (x_train / 255).reshape(input_shape)
    x_test = (x_test / 255).reshape(input_shape)

    return x_train, y_train, x_test, y_test


def print_evaluation(dataset_name: str, dataset_type: str, eval_tuple: tuple):
    print(f"Loss on {dataset_name} {dataset_type} data: {eval_tuple[0]}, accuracy: {eval_tuple[1]}")


def load_or_train_model(compiled_model,
                        dataset_name: str,
                        epochs: int = 5,
                        models_dir_name: str = 'models',
                        model_type: str = 'basic',
                        need_train=False):
    """
    loads model or trains model
    :param compiled_model: model for training
    :param dataset_name: name of target dataset
    :param epochs: number of epochs for training
    :param models_dir_name: str - directory for models
    :param model_type: custom specification for model (basic, advanced, etc)
    :param need_train: True for training False for just loading
    :return: keras model
    """
    network = compiled_model
    if need_train:
        start_time = time.time()
        network.train(epochs, save_model=True, target_name=f"{dataset_name}_{model_type}.h5")
        end_time = time.time()
        print(f"{dataset_name.capitalize()} training took {end_time - start_time} seconds")

    return network.model if need_train else load_model(f"{models_dir_name}/{dataset_name}_{model_type}.h5")
