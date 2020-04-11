from attack import Attack
from train import TurtleNet

from target_model import CNNModel
from keras.datasets import mnist
from keras.models import load_model
from keras.utils import to_categorical

from cleverhans.attacks import ProjectedGradientDescent, FastGradientMethod, BasicIterativeMethod
import cleverhans
import numpy as np

from utils import get_mnist_data
from evaluation import eval_models


def main1():
    x_train, y_train, x_test, y_test = get_mnist_data()

    attack = Attack(BasicIterativeMethod, 0.3, 0, 1)

    net = CNNModel()
    net.train_on_mnist()
    net.test_on_mnist()
    # model = load_model("models/conv_nn.h5")

    pert = attack.generate_perturbations(np.array(x_train), net.model, 6)
    print("adv data")
    print(net.model.evaluate(pert.reshape(-1, 28, 28, 1), to_categorical(y_train)))


def main2():
    x_train, y_train, x_test, y_test = get_mnist_data()

    network = CNNModel()
    # network.train_on_mnist()
    print(network.test_on_mnist())

    net = TurtleNet(network.model, ProjectedGradientDescent, 0.3, 0, 1)

    net.adversarial_training(iterations=3000, x_train=x_train, y_train=y_train, chunk_size=50,
                             epochs_per_iteration=3, batch_size=50, make_checkpoints=True)

    net.eval_on_attack(ProjectedGradientDescent, 0.3, 0, 1, x_test, y_test, chunk_size=50)
    net.eval_on_attack(FastGradientMethod, 0.3, 0, 1, x_test, y_test, chunk_size=50)

    print(net.model.evaluate(x_train, to_categorical(y_train)))

    net.save_model("models/robust_model.h5")

    # model_new = load_model("models/robust_model.h5")

    # print(model_new.evaluate(x_train, to_categorical(y_train)))


def main3():
    eval_models(attack_types=[
                              FastGradientMethod],
                epsilon=0.3,
                clip_min=0,
                clip_max=1,
                num_chunks=1,
                save_to_file=True,
                results_file_path="results/test.json",
                folder_name="models",
                folder_list=["checkpoint_1200"],
                prefix="checkpoint_")


if __name__ == '__main__':
    main3()
