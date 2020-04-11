from itertools import islice
from keras.datasets import mnist
from keras.models import load_model

import cleverhans
import re
import time
import numpy as np
import json

from attack import Attack


def chunk(it, size: int):
    iter_list = iter(it)
    return iter(lambda: tuple(islice(iter_list, size)), ())


def save_collage(filepath: str,
                 batch: np.array,
                 rows: int,
                 columns: int,
                 width: int = 28,
                 height: int = 28,
                 interpolation: str = 'nearest',
                 cmap: str = 'grey'):
    batch = batch.reshape(batch.shape[0], width, height)
    fig, axs = plt.subplots(rows, columns)
    cnt = 0
    for i in range(rows):
        for j in range(columns):
            axs[i, j].imshow((batch[cnt] + 1) / 2., interpolation=interpolation, cmap=cmap)
            axs[i, j].axis('off')
            cnt += 1
    fig.savefig(f"{filepath}.png")
    plt.close()


def get_mnist_data() -> tuple:
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = (x_train / 255).reshape((len(x_train), 28, 28, 1))
    x_test = (x_test / 255).reshape((len(x_test), 28, 28, 1))

    return x_train, y_train, x_test, y_test


def eval_models(attack_types: list,
                epsilon: float,
                clip_min: float,
                clip_max: float,
                num_chunks: int,
                results_file_path: str = "",
                save_to_file: bool = False,
                folder_list: list = [],
                folder_name: str = "",
                prefix: str = "",
                suffix: str = "") -> dict:
    model_names = filter(lambda x: x.startswith(prefix) and x.endswith(suffix),
                         os.listdir(folder_name)) if not folder_list else folder_list

    _, _, x_test, y_test = get_mnist_data()

    json_test_results = {}

    for model_name in model_names:
        try:
            iteration_number = int(re.search(f"{prefix}(.*){suffix}", string).group(1))
        except Exception as e:
            print(f"An exception {e} occurred! with model {model_name}")
            return

        model = load_model(f"{folder_name}/{model_name}")

        model_results_json = {}

        for attack_type in attack_types:
            attack_str = str(attack_type).split("'")[1]
            print(f"Evaluating model '{model_name}' for attack '{attack_str}' ...")

            attack = Attack(attack_type, epsilon, clip_min, clip_max)

            start_attack = time.time()
            perturbations = attack.generate_perturbations(x_test, model, num_chunks)
            end_attack = time.time()
            total_attack_time = end_attack - start_attack
            print(f"Attack took {total_attack_time} seconds.")

            results = model.evaluate(perturbations, y_test)

            model_results_json[attack_str] = {"loss": results[0],
                                              "accuracy": results[1],
                                              "attack_time": total_attack_time}

        json_test_results[iteration_number] = model_results_json

    print(json_test_results)

    if save_to_file:
        json.dump(json_test_results, open(results_file_path))

    return json_test_results
