import os

os.environ['KERAS_BACKEND'] = 'tensorflow'
from keras import backend as K

K.tensorflow_backend._get_available_gpus()
from keras.datasets import mnist
from keras.utils import to_categorical
from keras import layers, Model
from keras.models import *
from keras.layers import *
from regularization import InstanceNormalization
from keras.optimizers import Adam, SGD
from keras.metrics import binary_accuracy
from keras import backend as K
import numpy as np
from numpy.random import seed
from tensorflow import set_random_seed
import tensorflow as tf
from utils import get_keras_dataset


def eval_folder(dir_name: str,
                model_path: str,
                npy_files_list: list = None,
                custom_objects: dict = {'InstanceNormalization': InstanceNormalization},
                prefix: str = 'miss',
                suffix: str = '.npy'):
    target_model = load_model(model_path, custom_objects=custom_objects)
    model_names = filter(lambda x: x.startswith(prefix) and x.endswith(suffix),
                         os.listdir(folder_name)) if not npy_files_list else npy_files_list

    for file_name in model_names:
        iteration, start, end = map(int, re.search(f"{prefix}([0-9]*)_([0-9]*)_([0-9]*){suffix}", file_name).groups())
        print(f"Evaluating {dir_name}/{file_name}")
        print(f"Evaluating samples generated by {iteration} iteration")
        print(f"Samples starts at {start} position and ends at {end} position in training data")

        target_dataset_as_npy = np.load(f"{dir_name}/{file_name}")

        x_train, y_train, _, _ = get_keras_dataset(mnist.load_data())  # this needs to be parametrized

        print(f"Natural data... {target_model.evaluate(x_train[start: end], y_train[start, end])}")
        print(f"Malignant data... {target_model.evaluate(target_dataset_as_npy[start: end], y_train[start, end])}")


eval_folder(dir_name='np_debug2',
            model_path='../models/keras-stolen-model.h5',
            npy_files_list=['miss0_7200_9600.npy'])