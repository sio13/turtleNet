from attack import Attack
from target_model import CNNModel
from keras.datasets import mnist

from cleverhans.attacks import FastGradientMethod
import cleverhans

def main():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = (x_train / 255).reshape((len(x_train), 28, 28, 1))


    attack = Attack(FastGradientMethod, 0.3, 0, 1)

    net = CNNModel()
    net.train_on_mnist()
    net.test_on_mnist()

    pert = attack.generate_perturbations(x_train, net.model)


if __name__ == '__main__':
    main()
