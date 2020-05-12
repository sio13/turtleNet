import os
import numpy as np

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
from keras.optimizers import RMSprop, SGD
from keras.datasets import cifar10
from keras.utils import to_categorical

from architectures.target_model import CNNModel
from utils import get_keras_dataset


class CNNCifar10Model(CNNModel):
    def __init__(self, num_classes: int = 10, custom_optimizer=None, input_shape=(32, 32, 3)):
        super().__init__(num_classes)

        self.input_shape = input_shape
        x_train, y_train, x_test, y_test = get_keras_dataset(cifar10.load_data(), input_shape=self.input_shape)
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

        weight_decay = 1e-4
        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay),
                              input_shape=input_shape))
        self.model.add(Activation('elu'))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        self.model.add(Activation('elu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.2))

        self.model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        self.model.add(Activation('elu'))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        self.model.add(Activation('elu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.3))

        self.model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        self.model.add(Activation('elu'))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        self.model.add(Activation('elu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.4))

        self.model.add(Flatten())
        self.model.add(Dense(10, activation='softmax'))

        optimizer = custom_optimizer or SGD(lr=0.001, momentum=0.9)
        self.model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])



