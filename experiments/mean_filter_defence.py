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
from defences.filters import mean_filter
from evaluation import eval_models, evaluate_filters

from architectures.target_model_mnist import CNNModelMnist as MnistNetwork
from architectures.target_model_cifar_10_better import CNNCifar10Model as CifarNetwork

from keras import backend

sess = backend.get_session()

if __name__ == '__main__':
    cifar_model = CifarNetwork()
    mnist_model = MnistNetwork()

    evaluate_filters(dataset_name='mnist',
                     dataset=get_keras_dataset(mnist.load_data()),
                     compiled_model=mnist_model,
                     epsilon=0.3,
                     clip_min=0,
                     clip_max=1,
                     attack_type=ProjectedGradientDescent,
                     result_picture_image_dir='results/json/mean_filter',
                     filter_function=mean_filter,
                     sample_image_index=1,
                     need_train=False)

    evaluate_filters(dataset_name='cifar',
                     dataset=get_keras_dataset(cifar10.load_data(), input_shape=(-1, 32, 32, 3)),
                     compiled_model=cifar_model,
                     epsilon=0.1,
                     clip_min=0,
                     clip_max=1,
                     attack_type=ProjectedGradientDescent,
                     result_picture_image_dir='results/json/mean_filter',
                     filter_function=mean_filter,
                     sample_image_index=1,
                     need_train=False)
