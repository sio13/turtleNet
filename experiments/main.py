import global_config

from attacks import attack
from defences.train import TurtleNet

from architectures.target_model_mnist import CNNModelMnist
from keras.datasets import cifar10,mnist
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


def main1():
    x_train, y_train, x_test, y_test = get_keras_dataset(mnist.load_data())

    target_attack = attack.Attack(BasicIterativeMethod, 0.3, 0, 1)

    net = CNNCifar10Model()
    net.train()
    net.test()
    # model = load_model("models/conv_nn.h5")

    pert = target_attack.generate_perturbations(np.array(x_train), net.model, 6)
    print("adv data")
    print(net.model.evaluate(pert.reshape(-1, 28, 28, 1), to_categorical(y_train)))


def main2():
    x_train, y_train, x_test, y_test = get_keras_dataset(mnist.load_data())

    network = CNNCifar10Model()
    # network.train_on_mnist()
    print(network.test())

    net = TurtleNet(network.model, FastGradientMethod, 0.3, 0, 1)

    net.adversarial_training(iterations=3000, x_train=x_train, y_train=y_train, chunk_size=50,
                             batch_size=50, make_checkpoints=True,
                             checkpoint_dir='models_fgsm/')

    net.eval_on_attack(ProjectedGradientDescent, 0.3, 0, 1, x_test, y_test, chunk_size=50)
    net.eval_on_attack(FastGradientMethod, 0.3, 0, 1, x_test, y_test, chunk_size=50)

    print(net.model.evaluate(x_train, to_categorical(y_train)))

    net.save_model("models/robust_model.h5")

    # model_new = load_model("models/robust_model.h5")

    # print(model_new.evaluate(x_train, to_categorical(y_train)))


def main3():
    eval_models(attack_types=[MomentumIterativeMethod,
                              # MaxConfidence,
                              MadryEtAl,
                              BasicIterativeMethod,
                              ProjectedGradientDescent,
                              FastGradientMethod],
                epsilon=0.3,
                clip_min=0,
                clip_max=1,
                num_chunks=1,
                save_to_file=True,
                results_file_path="../results/json/models_fgsm.json",
                folder_name="models_fgsm",
                prefix="checkpoint_",
                suffix=".h5")


def main4():
    """
    train natural model
    """

    network = CNNCifar10Model()
    network.train()
    network.save_model("models_natural/conv_nn.h5")

    eval_models(attack_types=[MomentumIterativeMethod,
                              # MaxConfidence,
                              MadryEtAl,
                              BasicIterativeMethod,
                              ProjectedGradientDescent,
                              FastGradientMethod],
                epsilon=0.3,
                clip_min=0,
                clip_max=1,
                num_chunks=1,
                save_to_file=True,
                results_file_path="../results/json/model_natural.json",
                folder_name="models_natural",
                folder_list=["conv_nn_0.h5"],
                prefix="conv_nn_",
                suffix=".h5")


def main5():
    # network = CNNModel()
    # network.train_on_cifar10(epochs=10, batch_size=64)
    # print(network.test_on_cifar10())
    x_train, y_train, x_test, y_test = get_keras_dataset(cifar10.load_data(), input_shape=(-1, 32, 32, 3))

    model = load_model("models/conv_nn_cifar.h5")
    print(model.evaluate(x_test, to_categorical(y_test)))

    target_attack = attack.Attack(FastGradientMethod, 0.03, 0, 1)
    pert = target_attack.generate_perturbations(np.array(x_test), model, 6)
    save_collage("cifar_samples", pert[:9], 3, 3, 32, 32, 3)
    save_collage("cifar_original", x_test[:9], 3, 3, 32, 32, 3)
    print("adv data")
    print(model.evaluate(pert.reshape(-1, 32, 32, 3), to_categorical(y_test)))


def main6():
    # network = CNNModel()
    # network.train_on_mnist(epochs=10, batch_size=64)
    # print(network.test_on_mnist())
    x_train, y_train, x_test, y_test = get_keras_dataset(mnist.load_data())

    model = load_model("models/conv_nn.h5")
    print(model.evaluate(x_test, to_categorical(y_test)))

    target_attack = attack.Attack(ProjectedGradientDescent, 0.3, 0, 1)
    pert = target_attack.generate_perturbations(np.array(x_test), model, 6)
    print("adv data")
    print(model.evaluate(pert.reshape(-1, 28, 28, 1), to_categorical(y_test)))


def main7():
    # network = CNNModel()
    # network.train_on_cifar100(epochs=25, batch_size=64)
    # print(network.test_on_cifar100())
    x_train, y_train, x_test, y_test = get_keras_dataset(cifar100.load_data(), input_shape=(-1, 32, 32, 3))

    model = load_model("models/conv_nn_cifar100.h5")
    print(model.evaluate(x_test, to_categorical(y_test, num_classes=100)))

    target_attack = attack.Attack(ProjectedGradientDescent, 0.3, 0, 1)
    pert = target_attack.generate_perturbations(np.array(x_test), model, 6)
    print("adv data")
    print(model.evaluate(pert.reshape(-1, 32, 32, 3), to_categorical(y_test, num_classes=100)))


def train_cifar10_robust():
    network_better = CNNCifar10Model()
    network_better.train(1)
    # model_resnet = load_model("models/resnet_raw.h5")
    x_train, y_train, x_test, y_test = get_keras_dataset(
        cifar10.load_data(), input_shape=(-1, 32, 32, 3))

    net = TurtleNet(train_model=network_better.model,
                    attack_type=ProjectedGradientDescent,
                    epsilon=0.1, clip_min=0, clip_max=1)

    net.adversarial_training(iterations=15000,
                             x_train=x_train,
                             y_train=y_train,
                             chunk_size=128,
                             batch_size=128,
                             checkpoint_dir='models_cifar_better',
                             make_checkpoints=True,
                             checkpoint_frequency=50,
                             checkpoint_filename="checkpoint")


if __name__ == '__main__':
    train_cifar10_robust()
