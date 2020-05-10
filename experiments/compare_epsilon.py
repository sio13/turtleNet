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

from utils import get_keras_dataset, save_image_and_collage, print_evaluation, load_or_train_model
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
                    result_picture_image_dir: str = 'results/compare_epsilon',
                    sample_image_index: int = 2):
    x_train, y_train, x_test, y_test = dataset

    model = load_or_train_model(compiled_model=compiled_model,
                                dataset_name=dataset_name,
                                epochs=epochs,
                                models_dir_name='models',
                                model_type='compare_epsilon',
                                need_train=need_train
                                )

    rows = 3
    columns = 3

    save_image_and_collage(dir_path=result_picture_image_dir,
                           image_name=dataset_name,
                           array=x_test[:rows * columns],
                           image_type='natural',
                           rows=rows,
                           columns=columns,
                           sample_image_index=sample_image_index)

    for epsilon in epsilons:
        adv_attack = attack.Attack(attack_type, epsilon, clip_min, clip_max)
        start_time_attack = time.time()
        adv_samples = adv_attack.generate_perturbations(np.array(x_test[:rows * columns]), model, 1)
        end_time_attack = time.time()

        save_image_and_collage(dir_path=result_picture_image_dir,
                               image_name=dataset_name,
                               array=adv_samples,
                               image_type=f'adversarial_epsilon{epsilon}',
                               rows=rows,
                               columns=columns,
                               sample_image_index=sample_image_index)

        print(f"Attacks on {dataset_name} with epsilon {epsilon} lasted {end_time_attack - start_time_attack}")
        print(f"Using {attack_type}")


if __name__ == '__main__':
    cifar_model = CifarNetwork()
    mnist_model = MnistNetwork()

    compare_epsilon(dataset_name='mnist',
                    dataset=get_keras_dataset(mnist.load_data()),
                    compiled_model=mnist_model,
                    epsilons=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1, 1.2, 1.5, 2, 3],
                    clip_min=None,
                    clip_max=None,
                    epochs=5,
                    attack_type=ProjectedGradientDescent,
                    need_train=False)

    compare_epsilon(dataset_name='cifar10',
                    dataset=get_keras_dataset(cifar10.load_data(), input_shape=(-1, 32, 32, 3)),
                    compiled_model=cifar_model,
                    epsilons=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1, 1.2, 1.5, 2, 3],
                    clip_min=None,
                    clip_max=None,
                    epochs=10,
                    attack_type=ProjectedGradientDescent,
                    need_train=True)
