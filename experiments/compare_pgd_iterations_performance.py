import config

import numpy as np
from attacks import attack
from evaluation import eval_models
from keras.datasets import mnist, cifar10
from cleverhans.attacks import ProjectedGradientDescent
from keras.utils import to_categorical

from architectures.target_model_mnist import CNNModelMnist as MnistNetwork
from architectures.target_model_cifar_10_better import CNNCifar10Model as CifarNetwork
from utils import get_keras_dataset, save_image_and_collage, print_evaluation, load_or_train_model
from defences.filters import threshold_data
from evaluation import eval_models, compare_damage

if __name__ == '__main__':

    for iterations_number in (5, 10, 15, 20, 25, 30, 35):
        cifar_model = CifarNetwork()
        mnist_model = MnistNetwork()
        # change epsilon iter
        compare_damage(dataset_name='mnist',
                       dataset=get_keras_dataset(mnist.load_data()),
                       compiled_model=mnist_model,
                       epsilon=0.3,
                       clip_min=0,
                       clip_max=1,
                       attack_types=[ProjectedGradientDescent],
                       result_dir='results/json/compare_pgd_iterations',
                       need_train=False,
                       epochs=20,
                       model_type='compare_iterations_pgd',
                       nb_iter=iterations_number*2,
                       eps_iter=0.3/iterations_number
                       )
        # TODO relativize training steps
        compare_damage(dataset_name='cifar',
                       dataset=get_keras_dataset(cifar10.load_data(), input_shape=(-1, 32, 32, 3)),
                       compiled_model=cifar_model,
                       epsilon=0.1,
                       clip_min=0,
                       clip_max=1,
                       attack_types=[ProjectedGradientDescent],
                       result_dir='results/json/compare_pgd_iterations',
                       need_train=False,
                       epochs=20,
                       nb_iter=iterations_number*2,
                       eps_iter=0.1/iterations_number
                       )
