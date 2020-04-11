from attack import Attack
from utils import get_mnist_data
from keras.models import load_model

import json
import re
import time


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
            iteration_number = int(re.search(f"{prefix}(.*){suffix}", model_name).group(1))
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
