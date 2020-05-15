import config
from attacks import attack
from defences.train import TurtleNet

from architectures.target_model_mnist import CNNModelMnist
from keras.datasets import cifar10, mnist
from keras.models import load_model
from keras.utils import to_categorical

from cleverhans.attacks import *
import cleverhans
import numpy as np

from utils import get_keras_dataset, save_collage, save_image
from evaluation import eval_models

from architectures.target_model_cifar_10_better import CNNCifar10Model
from keras import backend

sess = backend.get_session()


def train_model(model,
                dataset: tuple,
                iteration_total: int,
                checkpoint_dir: str,
                epsilon: float,
                attack_type: cleverhans.attacks,
                eps_iter: float = 0.015,
                clip_min: float = 0,
                clip_max: float = 1,
                batch_size: int = 128,
                chunk_size: int = 128,
                checkpoint_frequency: int = 50,
                make_checkpoints: bool = True,
                iteration_so_far: int = 0):
    x_train, y_train, x_test, y_test = dataset

    net = TurtleNet(train_model=model,
                    attack_type=attack_type,
                    epsilon=epsilon,
                    clip_min=clip_min,
                    clip_max=clip_max,
                    eps_iter=eps_iter)

    net.adversarial_training(iterations=iteration_total,
                             x_train=x_train,
                             y_train=y_train,
                             chunk_size=chunk_size,
                             batch_size=batch_size,
                             checkpoint_dir=checkpoint_dir,
                             make_checkpoints=make_checkpoints,
                             checkpoint_frequency=checkpoint_frequency,
                             checkpoint_filename="checkpoint",
                             iteration_start=iteration_so_far + 1)


if __name__ == '__main__':
    target_model = CNNCifar10Model()
    target_model.train(1)

    train_model(model=target_model.model,
                dataset=get_keras_dataset(cifar10.load_data(), input_shape=(-1, 32, 32, 3)),
                iteration_total=15000,
                checkpoint_dir='../models_cifar_better_new',
                epsilon=0.1,
                iteration_so_far=0,
                attack_type=ProjectedGradientDescent
                )
