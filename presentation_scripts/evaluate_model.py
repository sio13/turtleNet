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

if __name__ == '__main__':
    model = load_model("models_mnist/robust_model.h5")

    x_train, y_train, x_test, y_test = get_keras_dataset(mnist.load_data())

    att = attack.Attack(ProjectedGradientDescent, 0.3, 0, 1)
    samples = att.generate_perturbations(np.array(x_test), model, 60, truth_labels=y_test)
    print("evaluating MNIST natural data...")
    print(model.evaluate(x_test, to_categorical(y_test)))

    print("evaluating MNIST adversarial data...")
    print(model.evaluate(samples, to_categorical(y_test)))

    # model = load_model("models_cifar/robust_model.h5")
    #
    # x_train, y_train, x_test, y_test = get_keras_dataset(cifar.load_data(), input_shape=(32,32,3))
    #
    # att = attack.Attack(ProjectedGradientDescent, 0.1, 0, 1)
    # samples = att.generate_perturbations(np.array(x_test), model, 60, truth_labels=y_test)
    # print("evaluating CIFAR10 natural data...")
    # print(model.evaluate(x_test, to_categorical(y_test)))
    #
    # print("evaluating CIFAR10 adversarial data...")
    # print(model.evaluate(samples, to_categorical(y_test)))
