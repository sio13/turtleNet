from attacks.attack import Attack
from utils import get_keras_dataset, load_or_train_model
from keras.models import load_model
from keras.utils import to_categorical
from keras.datasets import mnist

import os
import json
import re
import time


def eval_and_get_results(model,
                         dataset_name: str,
                         x_test,
                         y_test,
                         attack_type,
                         iteration_number,
                         total_attack_time: float,
                         epsilon: float):
    results = model.evaluate(x_test, to_categorical(y_test, num_classes=10))
    loss, accuracy = results[0], results[1]

    model_results_json = {"iteration": iteration_number,
                          "attack": attack_type,
                          "loss": loss,
                          "accuracy": accuracy,
                          "attack_time": total_attack_time,
                          "epsilon": epsilon}

    print(f"{dataset_name} model was successfully evaluated on attack '{attack_type}'.")
    print(f"Loss: {loss} - - Accuracy: {accuracy}")
    return model_results_json


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
        json_test_results.append(
            eval_and_get_results(
                model=model,
                dataset_name=dataset_name,
                x_test=x_test,
                y_test=y_test,
                attack_type='no attack',
                iteration_number='None',
                total_attack_time=0,
                epsilon=epsilon
            )
        )
        iteration_number = "None"  # TODO

        for attack_type in attack_types:
            attack_str = str(attack_type).split("'")[1]
            print(f"Evaluating {dataset_name} model for attack '{attack_str}' ...")

            attack = Attack(attack_type, epsilon, clip_min, clip_max)

            start_attack = time.time()
            perturbations = attack.generate_perturbations(x_test, model, num_chunks)
            end_attack = time.time()

            total_attack_time = end_attack - start_attack
            print(f"Attack took {total_attack_time} seconds.")
            json_test_results.append(
                eval_and_get_results(model=model,
                                     dataset_name=dataset_name,
                                     x_test=perturbations,
                                     y_test=y_test,
                                     attack_type=attack_str.split('.')[-1],
                                     iteration_number=iteration_number,
                                     total_attack_time=total_attack_time,
                                     epsilon=epsilon
                                     )
            )

    print(json_test_results)

    if save_to_file:
        json.dump(json_test_results, open(f"{results_dir}/{dataset_name}_{result_filename}.json", "w"))

    return json_test_results


def compare_damage(dataset_name: str,
                   dataset: tuple,
                   compiled_model,
                   epsilon: float,
                   attack_types: list,
                   clip_min: float = None,
                   clip_max: float = None,
                   epochs: int = 5,
                   need_train: bool = False,
                   result_dir: str = 'results/json/compare_damage',
                   result_filename='natural_trained'):
    """
    :param dataset_name: name of target dataset
    :param dataset: tuple of np arrays x_train, y_train, x_test and y_test of target dataset
    :param compiled_model: model for evaluate using attacks
    :param epsilon: epsilon for attack
    :param attack_types: list of attack types for evaluate
    :param clip_min: min for clip
    :param clip_max: max for clip
    :param epochs: number of epochs for training target model
    :param need_train: boolean - True for training model, False for loading
    :param result_dir: str - directory path for results in JSON
    :param result_filename: name of file with results in JSON format
    :return: None
    Compare different attacks against one model and prints results.
    """
    model = load_or_train_model(compiled_model=compiled_model,
                                dataset_name=dataset_name,
                                epochs=epochs,
                                models_dir_name='models',
                                model_type='compare_damage',
                                need_train=need_train
                                )

    eval_models(attack_types=attack_types,
                dataset=dataset,
                dataset_name=dataset_name,
                epsilon=epsilon,
                num_chunks=10,
                clip_min=clip_min,
                clip_max=clip_max,
                track_iteration=False,
                save_to_file=True,
                results_dir=result_dir,
                result_filename=f"{result_filename}_{epsilon}",
                models_list=[model])
