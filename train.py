from attack import Attack
from utils import save_collage
import cleverhans

from keras.utils import to_categorical
import numpy as np


class TurtleNet:
    def __init__(self,
                 model,
                 attack_type: cleverhans.attacks,
                 epsilon: float,
                 clip_min: float,
                 clip_max: float):
        self.model = model
        self.attack = Attack(attack_type, epsilon, clip_min, clip_max)
        self.perturbed_data = None

    def adversarial_training(self,
                             iterations: int,
                             x_train: np.array,
                             y_train: np.array,
                             chunk_size: int,
                             batch_size: int,
                             epochs_per_iteration: int,
                             checkpoint_dir: str = 'models',
                             make_checkpoints: bool = False,
                             checkpoint_frequency: int = 50,
                             checkpoint_filename: str = "checkpoint"):
        for iteration in range(iterations):
            batch_index_start = (batch_size * iteration) % len(x_train)
            batch_index_end = min(batch_index_start + batch_size, len(x_train))

            print(f"Generating samples for slice from {batch_index_start} to {batch_index_end}.")

            batch = np.array(x_train)[batch_index_start:batch_index_end]
            labels = np.array(y_train)[batch_index_start:batch_index_end]

            self.perturbed_data = self.attack.generate_perturbations(
                batch,
                self.model,
                len(batch) // chunk_size)
            self.model.fit(self.perturbed_data,
                           to_categorical(labels, num_classes=10),
                           epochs=epochs_per_iteration)
            print(f"Iteration number {iteration}")
            if make_checkpoints and iteration % checkpoint_frequency == 0:
                checkpoint_full_path = f"{checkpoint_dir}/{checkpoint_filename}_{iteration}"
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
        self.perturbed_data = evaluation_attack.generate_perturbations(np.array(x_train), self.model,
                                                                       len(x_train) // chunk_size)
        results = self.model.evaluate(self.perturbed_data, to_categorical(y_train))

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
        self.model.save(model_path)
