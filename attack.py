import os

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

import pandas as pd
import numpy as np

sess = backend.get_session()

class Attack:
    def __init__(self, attack_type: cleverhans.attacks, epsilon: float, clip_min: float, clip_max: float):
        self.attack_type = attack_type
        self.epsilon = epsilon
        self.clip_min = clip_min
        self.clip_max = clip_max

    def generate_perturbations(self, x_train, model):
        attack_params = {'eps': self.epsilon, 'clip_min': self.clip_min, 'clip_max': self.clip_max}
        wrapped_model = KerasModelWrapper(model)
        attack = self.attack_type(model=wrapped_model, sess=sess)

        perturbed_x_samples = attack.generate(x_train, **attack_params)
        return perturbed_x_samples
