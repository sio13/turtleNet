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
                             epochs_per_iteration: int):
        for iteration in range(iterations):
            self.perturbed_data = self.attack.generate_perturbations(
                np.array(x_train)[batch_size * iteration:(batch_size + 1) * iteration],
                self.model,
                1)
            print(self.model.fit(self.perturbed_data, to_categorical(y_train[batch_size * iteration:(batch_size + 1) * iteration]), epochs=epochs_per_iteration))
            print(f"Iteration number {iteration}")

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
