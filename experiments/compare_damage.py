import os

os.environ['KERAS_BACKEND'] = 'tensorflow'

import sys

sys.path.append('../')

from attacks import attack
from defences.train import TurtleNet

from keras.datasets import mnist, cifar10
from keras.models import load_model
from keras.utils import to_categorical

from cleverhans.attacks import *
import cleverhans
import numpy as np
import time

from utils import get_keras_dataset, save_image_and_collage, print_evaluation, load_or_train_model
from defences.filters import threshold_data
from evaluation import eval_models

from architectures.target_model_mnist import CNNModel as MnistNetwork
from architectures.target_model_cifar_10_better import CNNModel as CifarNetwork

from keras import backend

sess = backend.get_session()


def compare_damage(dataset_name: str,
                   dataset: tuple,
                   compiled_model,
                   epsilon: float,
                   attack_types: list,
                   clip_min: float = None,
                   clip_max: float = None,
                   epochs: int = 5,
                   need_train: bool = False,
                   result_dir: str = 'results/json/compare_damage',
                   result_filename='natural_trained'):
    model = load_or_train_model(compiled_model=compiled_model,
                                dataset_name=dataset_name,
                                epochs=epochs,
                                models_dir_name='models',
                                model_type='compare_damage',
                                need_train=need_train
                                )

    eval_models(attack_types=attack_types,
                dataset=dataset,
                dataset_name=dataset_name,
                epsilon=epsilon,
                num_chunks=10,
                clip_min=clip_min,
                clip_max=clip_max,
                track_iteration=False,
                save_to_file=True,
                results_dir=result_dir,
                result_filename=result_filename,
                models_list=[model])


if __name__ == '__main__':
    cifar_model = CifarNetwork()
    mnist_model = MnistNetwork()

    compare_damage(dataset_name='mnist',
                   dataset=get_keras_dataset(mnist.load_data()),
                   compiled_model=mnist_model,
                   epsilon=0.3,
                   clip_min=0,
                   clip_max=1,
                   attack_types=[MomentumIterativeMethod,
                                 MadryEtAl,
                                 BasicIterativeMethod,
                                 ProjectedGradientDescent,
                                 FastGradientMethod],
                   need_train=True,
                   epochs=5,
                   )

    compare_damage(dataset_name='cifar',
                   dataset=get_keras_dataset(cifar10.load_data(), input_shape=(-1, 32, 32, 3)),
                   compiled_model=cifar_model,
                   epsilon=0.1,
                   clip_min=0,
                   clip_max=1,
                   attack_types=[MomentumIterativeMethod,
                                 MadryEtAl,
                                 BasicIterativeMethod,
                                 ProjectedGradientDescent,
                                 FastGradientMethod],
                   need_train=True,
                   epochs=10,
                   )
