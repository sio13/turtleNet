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

from utils import get_keras_dataset, save_image_and_collage
from defences.filters import threshold_data
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
                       need_train: bool = False,
                       result_picture_image_dir: str = 'results',
                       sample_image_index: int = 2):
    x_train, y_train, x_test, y_test = dataset
    network = compiled_model

    print(f"Experiment with {str(attack_type)} attack on {dataset_name} dataset.")

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
    adv_samples = adv_attack.generate_perturbations(np.array(x_test), model, 60)
    end_time_attack = time.time()

    results_adv = model.evaluate(adv_samples, to_categorical(y_test))

    print(f"Loss on {dataset_name} adversarial data: {results_adv[0]}, accuracy: {results_adv[1]}")
    print(f"{dataset_name} attack time: {end_time_attack - start_time_attack}")

    filtered_adv_samples = threshold_data(adv_samples, threshold=0.5)  # pozot vazna chyba
    results_adv_filtered = model.evaluate(filtered_adv_samples, to_categorical(y_test))

    print(f"Loss on {dataset_name} filtered adversarial data: {results_adv_filtered[0]}")
    print(f"accuracy on {dataset_name} filtered adversarial data: {results_adv_filtered[1]}")

    rows = 3
    columns = 3

    save_image_and_collage(dir_path=result_picture_image_dir,
                           image_name=dataset_name,
                           array=x_test[:9],
                           image_type='natural',
                           rows=rows,
                           columns=columns,
                           sample_image_index=sample_image_index)

    save_image_and_collage(dir_path=result_picture_image_dir,
                           dataset_name,
                           adv_samples[:9],
                           'adversarial',
                           rows,
                           columns,
                           sample_image_index)

    save_image_and_collage(result_picture_image_dir,
                           dataset_name,
                           filtered_adv_samples[:9],
                           'adversarial_filtered',
                           rows,
                           columns,
                           sample_image_index)


if __name__ == '__main__':
    cifar_model = CifarNetwork()
    mnist_model = MnistNetwork()

    filters_experiment(dataset_name='mnist',
                       dataset=get_keras_dataset(mnist.load_data()),
                       compiled_model=mnist_model,
                       epsilon=0.3,
                       clip_min=0,
                       clip_max=1,
                       attack_type=ProjectedGradientDescent,
                       need_train=False)

    filters_experiment(dataset_name='cifar',
                       dataset=get_keras_dataset(cifar10.load_data(), input_shape=(-1, 32, 32, 3)),
                       compiled_model=cifar_model,
                       epsilon=0.1,
                       clip_min=0,
                       clip_max=1,
                       attack_type=ProjectedGradientDescent,
                       need_train=False)
