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
    def __init__(self, num_classes: int = 10, learning_rate: float = 0.001):
        self.x_test = None
        self.y_test = None
        self.x_train = None
        self.y_train = None
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

    def train(self,
              epochs=20,
              batch_size=64,
              target_name="conv_nn_mnist.h5",
              save_model=False,
              with_augmentation=False,
              rotation_range: int = 15,
              width_shift_range: float = 0.1,
              height_shift_range: float = 0.1,
              horizontal_flip: bool = True):

        if with_augmentation:
            data_gen = ImageDataGenerator(
                rotation_range=rotation_range,
                width_shift_range=width_shift_range,
                height_shift_range=height_shift_range,
                horizontal_flip=horizontal_flip)
            data_gen.fit(x_train)

            model.fit_generator(data_gen.flow(self.x_train, self.y_train, batch_size=batch_size),
                                steps_per_epoch=len(self.x_train) // batch_size,
                                epochs=epochs,
                                verbose=1,
                                validation_data=(self.x_test, self.y_test),
                                callbacks=[LearningRateScheduler(self.schedule)])
        else:
            self.model.fit(self.x_train,
                           to_categorical(
                               self.y_train,
                               num_classes=self.num_classes),
                           epochs=epochs,
                           batch_size=batch_size)
        if save_model:
            self.model.save(f"models/{target_name}")

    def save_model(self, model_path: str):
        print(f"Saving model into {model_path}")
        self.model.save(model_path)
