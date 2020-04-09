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

    def adversarial_train(self,
                          num_chunks: int,
                          epochs: int,
                          x_train: np.array,
                          y_train: np.array):
        pass

    def eval_on_attack(self,
                       attack_type: cleverhans.attacks,
                       epsilon: float,
                       clip_min: float,
                       clip_max: float,
                       x_train: np.array,
                       y_train: np.array):
        evaluation_attack = Attack(attack_type, epsilon, clip_min, clip_max)
        self.perturbed_data = evaluation_attack.generate_perturbations(np.array(x_train), self.model,
                                                                       len(x_train) // 10_000)
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
