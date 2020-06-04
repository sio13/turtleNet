import config

from experiments.train_robust_model import *


if __name__ == '__main__':

    target_model = CNNModelMnist()
    d = get_keras_dataset(mnist.load_data())
    x1, y1, _, _ = d
    target_model.model.train_on_batch(x1[:128], to_categorical(y1[:128]))

    train_model(model=target_model.model,
                dataset=get_keras_dataset(mnist.load_data()),
                iteration_total=15000,
                checkpoint_dir='models_mnist',
                epsilon=0.3,
                iteration_so_far=0,
                attack_type=ProjectedGradientDescent,
                use_natural=False,
                eps_iter=0.3 / 6,
                nb_iter=12
                )

    # for training cifar model
    # target_model = CNNCifar10Model()
    # d = get_keras_dataset(cifar10.load_data(), input_shape=(32,32,3))
    # x1, y1, _, _ = d
    # target_model.model.train_on_batch(x1[:128], to_categorical(y1[:128]))
    # train_model(model=target_model.model,
    #             dataset=get_keras_dataset(cifar10.load_data(), input_shape=(32, 32, 3)),
    #             iteration_total=80000,
    #             checkpoint_dir='models_cifar',
    #             epsilon=0.1,
    #             iteration_so_far=0,
    #             attack_type=ProjectedGradientDescent,
    #             use_natural=False,
    #             eps_iter=0.1 / 6,
    #             nb_iter=12
    #             )
