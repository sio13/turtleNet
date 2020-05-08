import numpy as np


class NoneAttack:
    def __init__(self, *args):
        pass

    def generate_np(self, x: np.array, **attack_params):
        return x
