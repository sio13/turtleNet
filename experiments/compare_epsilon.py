import os

os.environ['KERAS_BACKEND'] = 'tensorflow'

import sys

sys.path.append('../')

import attack
from defences.train import TurtleNet

from keras.datasets import mnist, cifar10
from keras.models import load_model
from keras.utils import to_categorical

from cleverhans.attacks import *
import cleverhans
import numpy as np
import time

from utils import get_keras_dataset, save_image_and_collage, print_evaluation
from defences.filters import threshold_data
from evaluation import eval_models

from architectures.target_model_mnist import CNNModel as MnistNetwork
from architectures.target_model_cifar_10_better import CNNModel as CifarNetwork

from keras import backend

sess = backend.get_session()


def compare_epsilon(dataset_name: str,
                    dataset: tuple,
                    compiled_model,
                    epsilons: list,
                    clip_min: float,
                    clip_max: float,
                    attack_type: cleverhans.attacks,
                    epochs: int = 5,
                    need_train: bool = False,
                    result_picture_image_dir: str = 'results',
                    sample_image_index: int = 2):
    x_train, y_train, x_test, y_test = dataset

    model = load_or_train_model(compiled_model=compiled_model,
                                dataset_name=dataset_name,
                                epochs=epochs,
                                models_dir_name='models',
                                model_type='compare_epsilon',
                                need_train=need_train
                                )
