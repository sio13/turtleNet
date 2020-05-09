import os

os.environ['KERAS_BACKEND'] = 'tensorflow'

import sys

sys.path.append('../')

import attack
from defences.train import TurtleNet

from architectures.target_model_mnist import CNNModel
from keras.datasets import mnist, cifar10
from keras.models import load_model
from keras.utils import to_categorical

from cleverhans.attacks import *
import cleverhans
import numpy as np
import time

from utils import get_keras_dataset, save_collage
from evaluation import eval_models

from architectures.target_model_mnist import CNNModel as MnistNetwork
from architectures.target_model_cifar_10_better import CNNModel as CifarNetwork

from keras import backend

sess = backend.get_session()


def filters_experiment(dataset_name: str,
                       dataset: tuple,
                       compiled_model,
                       epsilon: float,
                       clip_min: float,
                       clip_max: float,
                       attack_type: cleverhans.attacks,
                       need_train: bool = False):
    x_train, y_train, x_test, y_test = dataset
    network = compiled_model

    print(f"Experiment with {str(attack_type)} attack.")

    if need_train:
        start_time = time.time()
        network.train(5, save_model=True, target_name=f"{dataset_name}_basic.h5")
        end_time = time.time()
        print(f"{dataset_name.capitalize()} training took {end_time - start_time} seconds")

    model = network.model if need_train else load_model(f"models/{dataset_name}_basic.h5")

    results = model.evaluate(x_test, to_categorical(y_test))
    print(f"Loss on {dataset_name} natural data: {results[0]} and accuracy: {results[1]}")

    adv_attack = attack.Attack(attack_type, epsilon, clip_min, clip_max)

    start_time_attack = time.time()
    adv_samples = adv_attack.generate_perturbations(x_test, model, 60)
    end_time_attack = time.time()

    results_adv = model.evaluate(adv_samples, to_categorical(y_test))
    print(f"Loss on {dataset_name} adversarial data: {results_adv[0]}, accuracy: {results_adv[1]}")
    print(f"{dataset_name} attack time: {end_time_attack - start_time_attack}")


if __name__ == '__main__':
    cifar_model = CifarNetwork()
    mnist_model = MnistNetwork()
    target_attack = ProjectedGradientDescent

    dataset_experiment(dataset_name='mnist',
                       dataset=get_keras_dataset(mnist.load_data()),
                       compiled_model=mnist_model,
                       epsilon=0.3,
                       clip_min=0,
                       clip_max=1,
                       attack_type=target_attack,
                       need_train=False)

    dataset_experiment(dataset_name='cifar',
                       dataset=get_keras_dataset(cifar10.load_data(), input_shape=(-1, 32, 32, 3)),
                       compiled_model=cifar_model,
                       epsilon=0.1,
                       clip_min=0,
                       clip_max=1,
                       attack_type=target_attack,
                       need_train=False)
