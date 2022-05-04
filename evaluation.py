from attacks.attack import Attack
from utils import get_keras_dataset, load_or_train_model
from keras.models import load_model
from keras.utils import to_categorical
from keras.datasets import mnist

from defences.filters import threshold_data
from utils import print_evaluation, save_image_and_collage

import cleverhans
from attacks import attack

import numpy as np
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
                results_dir: str = "results/json",
                save_to_file: bool = False,
                eps_iter: float = 0.02,
                rand_init: bool = False,
                models_list: list = [],
                folder_list: list = [],
                folder_name: str = "",
                prefix: str = "",
                suffix: str = "",
                nb_iter: int = 10) -> dict:
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
        iteration_number = "None"  # TODO this is not longer required

        for attack_type in attack_types:
            attack_str = str(attack_type).split("'")[1]
            print(f"Evaluating {dataset_name} model for attack '{attack_str}' ...")

            att = Attack(attack_type=attack_type,
                         epsilon=epsilon,
                         clip_min=clip_min,
                         clip_max=clip_max,
                         rand_init=rand_init,
                         eps_iter=eps_iter)

            start_attack = time.time()
            perturbations = att.generate_perturbations(original_samples=x_test,
                                                       model=model,
                                                       num_chunks=num_chunks,
                                                       nb_iter=nb_iter,
                                                       truth_labels=y_test)
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
                   result_filename='natural_trained',
                   model_type: str = 'compare_damage',
                   nb_iter: int = 15,
                   eps_iter: float = 0.01):
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
    :param model_type: string of model type
    :param nb_iter: number of iterations of attack method
    :param eps_iter: size of attacking step
    :return: None
    Compare different attacks against one model and prints results.
    """
    model = load_or_train_model(compiled_model=compiled_model,
                                dataset_name=dataset_name,
                                epochs=epochs,
                                models_dir_name='models',
                                model_type=f'{model_type}_{epochs}',
                                need_train=need_train
                                )

    eval_models(attack_types=attack_types,
                dataset=dataset,
                dataset_name=dataset_name,
                epsilon=epsilon,
                num_chunks=10,
                clip_min=clip_min,
                clip_max=clip_max,
                save_to_file=True,
                results_dir=result_dir,
                result_filename=f"{result_filename}_{str(epsilon).replace('.', '_')}_epoch_{epochs}_{eps_iter}_{nb_iter}",
                models_list=[model],
                nb_iter=nb_iter,
                eps_iter=eps_iter)


def evaluate_filters(dataset_name: str,
                     dataset: tuple,
                     compiled_model,
                     epsilon: float,
                     clip_min: float,
                     clip_max: float,
                     attack_type: cleverhans.attacks,
                     filter_function,
                     epochs: int = 5,
                     threshold: float = 0.5,
                     filter_size: int = 4,
                     need_train: bool = False,
                     result_picture_image_dir: str = 'results/filter_defences',
                     sample_image_index: int = 6):
    x_train, y_train, x_test, y_test = dataset

    print(f"Experiment with {str(attack_type)} attack on {dataset_name} dataset.")

    model = load_or_train_model(compiled_model=compiled_model,
                                dataset_name=dataset_name,
                                epochs=epochs,
                                models_dir_name='models',
                                model_type='basic',
                                need_train=need_train
                                )

    results = model.evaluate(x_test, to_categorical(y_test))
    print_evaluation(dataset_name=dataset_name,
                     dataset_type='adversarial',
                     eval_tuple=results)

    adv_attack = attack.Attack(attack_type, epsilon, clip_min, clip_max)

    start_time_attack = time.time()
    adv_samples = adv_attack.generate_perturbations(np.array(x_test), model, 60)
    end_time_attack = time.time()

    results_adv = model.evaluate(adv_samples, to_categorical(y_test))

    print_evaluation(dataset_name=dataset_name,
                     dataset_type='adversarial',
                     eval_tuple=results_adv)

    print(f"{dataset_name} attack time: {end_time_attack - start_time_attack}")

    filtered_adv_samples = filter_function(adv_samples,
                                           threshold=threshold,
                                           size_of_filter=filter_size)
    results_adv_filtered = model.evaluate(filtered_adv_samples, to_categorical(y_test))

    print_evaluation(dataset_name=dataset_name,
                     dataset_type='filtered_adversarial',
                     eval_tuple=results_adv_filtered)

    rows = 3
    columns = 3

    save_image_and_collage(dir_path=result_picture_image_dir,
                           image_name=dataset_name,
                           array=x_test[:9],
                           image_type='natural',
                           rows=rows,
                           columns=columns,
                           sample_image_index=sample_image_index)

    save_image_and_collage(dir_path=result_picture_image_dir,
                           image_name=dataset_name,
                           array=adv_samples[:9],
                           image_type='adversarial',
                           rows=rows,
                           columns=columns,
                           sample_image_index=sample_image_index)

    save_image_and_collage(dir_path=result_picture_image_dir,
                           image_name=dataset_name,
                           array=filtered_adv_samples[:9],
                           image_type='adversarial_filtered',
                           rows=rows,
                           columns=columns,
                           sample_image_index=sample_image_index)
