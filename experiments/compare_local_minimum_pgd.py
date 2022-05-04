import config

import numpy as np
from attacks import attack
from evaluation import eval_models
from keras.datasets import mnist, cifar10
from cleverhans.attacks import FastGradientMethod
from keras.utils import to_categorical

from architectures.target_model_mnist import CNNModelMnist as MnistNetwork
from architectures.target_model_cifar_10_better import CNNCifar10Model as CifarNetwork
from utils import get_keras_dataset, save_image_and_collage, print_evaluation, load_or_train_model
from defences.filters import threshold_data
from evaluation import eval_models


def restart_pgd(dataset: tuple,
                dataset_name: str,
                compiled_model,
                number_restarts: int,
                epsilon: float,
                eps_iter: float = 0.02,
                epochs: int = 10,
                need_train: bool = False,
                num_chunks: int = 60,
                ):
    _, _, x_test, y_test = dataset
    model = load_or_train_model(compiled_model=compiled_model,
                                dataset_name=dataset_name,
                                epochs=epochs,
                                models_dir_name='models',
                                model_type='compare_restarts_pgd',
                                need_train=need_train
                                )
    print(model.evaluate(x_test, to_categorical(y_test)))
    att = attack.Attack(attack_type=ProjectedGradientDescent,
                        epsilon=epsilon,
                        clip_min=0,
                        clip_max=1,
                        eps_iter=eps_iter,
                        rand_init=True)

    results = []
    for restart_number in range(number_restarts):
        samples = att.generate_perturbations(original_samples=x_test,
                                             model=model,
                                             num_chunks=num_chunks,
                                             ord=np.inf)
        print(model.evaluate(samples, to_categorical(y_test)))
        accuracy = model.evaluate(samples, to_categorical(y_test))[1]

        print(f"Accuracy for {restart_number}-th restart of PGD is {accuracy * 100} percent")
        results.append(accuracy)

    results_np = np.array(results)
    print(f"For {number_restarts} Mean accuracy: {np.mean(results_np)} and standard deviation: {np.std(results_np)}")


if __name__ == '__main__':
    cifar_model = CifarNetwork()
    mnist_model = MnistNetwork()

    restart_pgd(dataset=get_keras_dataset(mnist.load_data()),
                dataset_name='mnist',
                compiled_model=mnist_model,
                number_restarts=20,
                epochs=10,
                epsilon=0.3,
                need_train=False
                )

    restart_pgd(dataset=get_keras_dataset(cifar10.load_data(), input_shape=(-1, 32, 32, 3)),
                dataset_name='cifar10',
                compiled_model=cifar_model,
                number_restarts=20,
                epsilon=0.1,
                need_train=False
                )
