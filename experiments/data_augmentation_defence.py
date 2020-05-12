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

from architectures.target_model_cifar_data_augmentation import CNNMCifarModelAugmentation

from keras import backend

sess = backend.get_session()


def evaluate_data_augmentation(dataset: tuple,
                               dataset_name: str,
                               attack_type: cleverhans.attacks,
                               epsilon: float,
                               num_chunks: int,
                               clip_min: int,
                               clip_max: int,
                               result_dir: str = 'results/json/data_augmentation'):
    model_without_augmentation = CNNMCifarModelAugmentation()
    model_with_augmentation = CNNMCifarModelAugmentation()

    model_not_augmented = load_or_train_model(compiled_model=model_without_augmentation,
                                              dataset_name='cifar',
                                              epochs=20,
                                              models_dir_name='models',
                                              model_type='data_not_augmentation',
                                              need_train=False)

    model_augmented = load_or_train_model(compiled_model=model_with_augmentation,
                                          dataset_name='cifar',
                                          epochs=20,
                                          models_dir_name='models',
                                          model_type='data_augmentation_20_iter',
                                          need_train=True)

    eval_models(attack_types=[attack_type],
                dataset=dataset,
                dataset_name=dataset_name,
                epsilon=epsilon,
                num_chunks=num_chunks,
                clip_min=clip_min,
                clip_max=clip_max,
                track_iteration=False,
                save_to_file=True,
                results_dir=result_dir,
                result_filename='not_augmented',
                models_list=[model_not_augmented])

    eval_models(attack_types=[attack_type],
                dataset=dataset,
                dataset_name=dataset_name,
                epsilon=epsilon,
                num_chunks=num_chunks,
                clip_min=clip_min,
                clip_max=clip_max,
                track_iteration=False,
                save_to_file=True,
                results_dir=result_dir,
                result_filename='augmented_20_iter',
                models_list=[model_augmented])




if __name__ == '__main__':
    evaluate_data_augmentation(
        dataset=get_keras_dataset(cifar10.load_data(), input_shape=(-1, 32, 32, 3)),
        dataset_name='cifar',
        attack_type=ProjectedGradientDescent,
        epsilon=0.1,
        num_chunks=10,
        clip_min=0,
        clip_max=1
    )
