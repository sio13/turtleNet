import config

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
from evaluation import eval_models, compare_damage

from architectures.target_model_mnist import CNNModelMnist as MnistNetwork
from architectures.target_model_cifar_10_better import CNNCifar10Model as CifarNetwork

from keras import backend

sess = backend.get_session()

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
