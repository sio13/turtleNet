import global_config

import cleverhans
import numpy as np

from keras.utils import to_categorical
from attacks.attack import Attack
from utils import save_collage


class TurtleNet:
    def __init__(self,
                 train_model,
                 attack_type: cleverhans.attacks,
                 epsilon: float,
                 clip_min: float,
                 clip_max: float,
                 target_model=None,
                 eps_iter: float = 0.05,
                 use_different_target=False):
        self.train_model = train_model
        self.eps_iter = eps_iter
        if use_different_target:
            self.target_model = target_model
        else:
            self.target_model = train_model

        self.use_different_target = use_different_target
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.attack = Attack(attack_type, epsilon, self.clip_min, self.clip_max)
        self.perturbed_data = None

    def adversarial_training(self,
                             iterations: int,
                             x_train: np.array,
                             y_train: np.array,
                             chunk_size: int,
                             batch_size: int,
                             checkpoint_dir: str = 'models',
                             make_checkpoints: bool = False,
                             checkpoint_frequency: int = 50,
                             checkpoint_filename: str = "checkpoint",
                             iteration_start: int = 0):
        """
        :param iterations: total number of iterations
        :param x_train: training dataset
        :param y_train: training labels
        :param chunk_size: size of chunk for generating adversarial examples -- affects memory power
        :param batch_size: training batch size
        :param checkpoint_dir: directory for models
        :param make_checkpoints: True for saving models, otherwise False
        :param checkpoint_frequency: number of iteration followed by checkpoint
        :param checkpoint_filename: filename of checkpoint -- automatically contains iteration number
        :param ord: order of reference attack
        :param iteration_start: name for the first iteration file
        :return:
        """
        for iteration in range(iteration_start, iterations):
            adv_size = batch_size
            batch_index_start = (adv_size * iteration) % len(x_train)
            batch_index_end = min(batch_index_start + adv_size, len(x_train))

            print(f"Generating samples for slice from {batch_index_start} to {batch_index_end}.")

            batch = np.array(x_train[batch_index_start:batch_index_end])
            labels = np.array(y_train[batch_index_start:batch_index_end])

            self.perturbed_data = self.attack.generate_perturbations(
                batch,
                self.train_model if self.use_different_target else self.train_model,
                max(len(batch) // chunk_size, 1))
            self.train_model.train_on_batch(self.perturbed_data,
                                            to_categorical(labels, num_classes=10))
            if self.use_different_target:
                self.target_model.train_on_batch(self.perturbed_data,
                                                 to_categorical(labels, num_classes=10))
            print(f"Iteration number {iteration}")
            if make_checkpoints and iteration % checkpoint_frequency == 0:
                checkpoint_full_path = f"{checkpoint_dir}/{checkpoint_filename}_{iteration}.h5"
                self.save_model(checkpoint_full_path)
                print(f"Saving checkpoint for iteration number {iteration} into {checkpoint_full_path}.")

    def eval_on_attack(self,
                       attack_type: cleverhans.attacks,
                       epsilon: float,
                       clip_min: float,
                       clip_max: float,
                       x_train: np.array,
                       y_train: np.array,
                       chunk_size: int):

        evaluation_attack = Attack(attack_type, epsilon, clip_min, clip_max)
        self.perturbed_data = evaluation_attack.generate_perturbations(np.array(x_train), self.target_model,
                                                                       len(x_train) // chunk_size)
        results = self.train_model.evaluate(self.perturbed_data, to_categorical(y_train))

        print(f"Total loss of target model is {results[0]} and its accuracy is {results[1]}")

    def save_perturbed_images(self,
                              file_path: str,
                              rows: int,
                              columns: int,
                              width: int,
                              height: int):
        if self.perturbed_data is None:
            print(f"You first need to generate adversarial examples using method `adversarial_train`.")
            return

        save_collage(file_path, self.perturbed_data, rows, columns, width, height)

    def save_model(self, model_path: str):
        print(f"Saving model into {model_path}")
        self.train_model.save(model_path)
