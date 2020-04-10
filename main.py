from attack import Attack
from train import TurtleNet

from target_model import CNNModel
from keras.datasets import mnist
from keras.models import load_model
from keras.utils import to_categorical

from cleverhans.attacks import ProjectedGradientDescent, FastGradientMethod, BasicIterativeMethod
import cleverhans
import numpy as np


def main1():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = (x_train / 255).reshape((len(x_train), 28, 28, 1))

    attack = Attack(BasicIterativeMethod, 0.3, 0, 1)

    net = CNNModel()
    net.train_on_mnist()
    net.test_on_mnist()
    # model = load_model("models/conv_nn.h5")

    pert = attack.generate_perturbations(np.array(x_train), net.model, 6)
    print("adv data")
    print(net.model.evaluate(pert.reshape(-1, 28, 28, 1), to_categorical(y_train)))


def main2():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = (x_train / 255).reshape((len(x_train), 28, 28, 1))

    network = CNNModel()
    # network.train_on_mnist()
    network.test_on_mnist()

    net = TurtleNet(network.model, ProjectedGradientDescent, 0.3, 0, 1)

    net.adversarial_training(iterations=50, x_train=x_train, y_train=y_train, chunk_size=10_000,
                             epochs_per_iteration=3, batch_size=50)

    net.eval_on_attack(ProjectedGradientDescent, 0.3, 0, 1, x_test, y_test, chunk_size=10_000)
    net.eval_on_attack(FastGradientMethod, 0.3, 0, 1, x_test, y_test, chunk_size=10_000)

    print(net.model.evaluate(x_train, to_categorical(y_train)))


if __name__ == '__main__':
    main2()
