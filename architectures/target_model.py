import os

os.environ['KERAS_BACKEND'] = 'tensorflow'
from keras import backend as K

K.tensorflow_backend._get_available_gpus()

import keras
from keras.models import Sequential
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.datasets import cifar10
from keras import regularizers
from keras.callbacks import LearningRateScheduler
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop, SGD
from keras.datasets import cifar10
from keras.utils import to_categorical


class CNNModel:
    def __init__(self, num_classes: int = 10, learning_rate: flaot = 0.001):
        self.x_test = None
        self.y_test = None
        self.num_classes = num_classes
        self.learning_rate = learning_rate

    def schedule(self, epoch):
        self.learning_rate = 0.001
        if epoch > 75:
            self.learning_rate = 0.0005
        if epoch > 100:
            self.learning_rate = 0.0003
        return self.learning_rate

    def test(self):
        return self.model.evaluate(self.x_test, to_categorical(self.y_test, num_classes=self.num_classes))
