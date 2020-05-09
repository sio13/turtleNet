import os

os.environ['KERAS_BACKEND'] = 'tensorflow'

import sys

sys.path.append('../')

import attack
from train import TurtleNet

from architectures.target_model_mnist import CNNModel
from keras.datasets import mnist, cifar10
from keras.models import load_model
from keras.utils import to_categorical

from cleverhans.attacks import *
import cleverhans
import numpy as np

from utils import get_keras_dataset, save_collage
from evaluation import eval_models

from architectures.target_model_mnist import CNNModel as MnistNetwork
from architectures.target_model_cifar_10_better import CNNModel as CifarNetwork

from keras import backend

sess = backend.get_session()

if __name__ == '__main__':
    x_train_cifar, y_train_cifar, x_test_cifar, y_test_cifar = get_keras_dataset(
        cifar10.load_data(), input_shape=(-1, 32, 32, 3))

    x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist = get_keras_dataset(
        mnist.load_data())

    mnist_model = MnistNetwork()
    cifar_model = CifarNetwork()

    mnist_model.train_on_mnist(5)
    cifar_model.train_on_cifar10(10)

    print(mnist_model.model.evaluate(x_test_mnist, to_categorical(y_test_mnist)))
    print(cifar_model.model.evaluate(x_test_cifar, to_categorical(x_test_cifar)))

    mnist_attack = attack.Attack(ProjectedGradientDescent, 0.3, 0, 1)
    cifar_attack = attack.Attack(ProjectedGradientDescent, 0.1, 0, 1)

    mnist_adv = mnist_attack.generate_perturbations(x_test_mnist, mnist_model.model, 60)
    cifar_adv = cifar_attack.generate_perturbations(x_test_cifar, cifar_model.model, 60)

    print(mnist_model.model.evaluate(mnist_adv, to_categorical(y_test_mnist)))
    print(cifar_model.model.evaluate(cifar_adv, to_categorical(x_test_cifar)))


