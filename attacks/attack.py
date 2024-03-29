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
from cleverhans.attacks import *
from cleverhans.utils_keras import KerasModelWrapper

from utils import chunk

sess = backend.get_session()


class Attack:
    def __init__(self,
                 attack_type: cleverhans.attacks,
                 epsilon: float,
                 clip_min: float,
                 clip_max: float,
                 eps_iter: float = 0.01,
                 rand_init: bool = False):
        self.eps_iter = eps_iter
        self.attack_type = attack_type
        self.epsilon = epsilon
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.rand_init = rand_init

    def generate_perturbations(self,
                               original_samples,
                               model,
                               num_chunks: int,
                               ord=np.inf,
                               nb_iter: int = 12,
                               truth_labels=None):
        attack_params = {
            'eps': self.epsilon,
            'clip_min': self.clip_min,
            'clip_max': self.clip_max,
            'ord': ord,

        }
        # TODO refactor methods to consume parameters
        if self.attack_type != FastGradientMethod:
            attack_params.update(
                {
                    'eps_iter': self.eps_iter,
                    'nb_iter': nb_iter,
                    # 'y': truth_labels
                }
            )
        elif self.attack_type not in (MomentumIterativeMethod, FastGradientMethod):
            attack_params.update(
                {
                    'rand_init': self.rand_init
                }
            )

        wrapped_model = KerasModelWrapper(model)
        attack = self.attack_type(model=wrapped_model, sess=sess)
        chunks = chunk(original_samples, len(original_samples) // num_chunks)
        chunks_truth = chunk(truth_labels, len(truth_labels) // num_chunks)
        perturbed_x_samples = itertools.chain.from_iterable(
            map(lambda x: attack.generate_np(
                x_val=np.array(x),
                **attack_params), chunks))
        return np.array(list(perturbed_x_samples))
