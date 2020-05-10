from attacks.attack import Attack
from utils import get_keras_dataset
from keras.models import load_model
from keras.utils import to_categorical
from keras.datasets import mnist

import os
import json
import re
import time


def eval_models(attack_types: list,
                dataset: tuple,
                dataset_name: str,
                epsilon: float,
                clip_min: float,
                clip_max: float,
                num_chunks: int,
                result_filename: str = '',
                track_iteration: bool = False,
                results_dir: str = "results/json",
                save_to_file: bool = False,
                models_list: list = [],
                folder_list: list = [],
                folder_name: str = "",
                prefix: str = "",
                suffix: str = "") -> dict:
    if len(models_list) == 0:
        model_names = filter(lambda x: x.startswith(prefix) and x.endswith(suffix),
                             os.listdir(folder_name)) if not folder_list else folder_list

    models = models_list if len(models_list) > 0 else map(
        lambda model_name: load_model(f"{folder_name}/{model_name}", model_names))

    _, _, x_test, y_test = dataset

    json_test_results = []

    for model in models:
        iteration_number = "None" # TODO

        for attack_type in attack_types:
            attack_str = str(attack_type).split("'")[1]
            print(f"Evaluating {dataset_name} model for attack '{attack_str}' ...")

            attack = Attack(attack_type, epsilon, clip_min, clip_max)

            start_attack = time.time()
            perturbations = attack.generate_perturbations(x_test, model, num_chunks)
            end_attack = time.time()

            total_attack_time = end_attack - start_attack
            print(f"Attack took {total_attack_time} seconds.")

            results = model.evaluate(perturbations, to_categorical(y_test, num_classes=10))
            loss, accuracy = results[0], results[1]

            model_results_json = {"iteration": iteration_number,
                                  "attack": attack_str.split('.')[-1],
                                  "loss": loss,
                                  "accuracy": accuracy,
                                  "attack_time": total_attack_time}
            print(f"{dataset_name} model was successfully evaluated on attack '{attack_str}'.")
            print(f"Loss: {loss} - - Accuracy: {accuracy}")

            json_test_results.append(model_results_json)

    print(json_test_results)

    if save_to_file:
        json.dump(json_test_results, open(f"{results_dir}/{dataset_name}_{result_filename}", "w"))

    return json_test_results
