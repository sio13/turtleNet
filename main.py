from attack import Attack
from target_model import CNNModel
from keras.datasets import mnist
from keras.models import load_model
from keras.utils import to_categorical

from cleverhans.attacks import FastGradientMethod
import cleverhans
import numpy as np

def main():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = (x_train / 255).reshape((len(x_train), 28, 28, 1))


    attack = Attack(FastGradientMethod, 0.3, 0, 1)

    # net = CNNModel()
    # net.train_on_mnist()
    # net.test_on_mnist()
    model = load_model("models/conv_nn.h5")

    pert = attack.generate_perturbations(np.array(x_train), model, 6)
    print("adv data")
    print(model.evaluate(pert.reshape(-1,28,28,1), to_categorical(y_train)))




if __name__ == '__main__':
    main()
