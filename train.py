from attack import Attack
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
        perturbed_data = evaluation_attack.generate_perturbations(np.array(x_train), self.model, len(x_train) // 10_000)
        results = self.model.evaluate(perturbed_data, to_categorical(y_train))

        print(f"Total loss of target model is {results[0]} and its accuracy is {results[1]}")
