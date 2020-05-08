import os
import pandas as pd
import numpy as np
import itertools


os.environ['KERAS_BACKEND'] = 'tensorflow'
from keras import backend as K

import keras
from keras import backend
from keras.datasets import mnist
from keras.utils import to_categorical
from keras import layers, Model
from keras.models import *
from keras.layers import *

import cleverhans
from cleverhans.attacks import FastGradientMethod
from cleverhans.utils_keras import KerasModelWrapper

from utils import chunk

sess = backend.get_session()


class Attack:
    def __init__(self, attack_type: cleverhans.attacks, epsilon: float):
        self.attack_type = attack_type
        self.epsilon = epsilon

    def generate_perturbations(self, original_samples, model, num_chunks: int):
        attack_params = {
            'eps': self.epsilon,
        }
        wrapped_model = KerasModelWrapper(model)
        attack = self.attack_type(model=wrapped_model, sess=sess)

        #chunks = chunk(original_samples, len(original_samples) // num_chunks)
        # print(chunks)
        perturbed_x_samples = itertools.chain.from_iterable(
            map(lambda x: attack.generate_np(np.array(x), **attack_params), original_samples))
        return np.array(list(perturbed_x_samples))
