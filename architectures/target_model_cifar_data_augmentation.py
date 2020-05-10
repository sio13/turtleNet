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

from architectures.target_model import CNNModel

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop, SGD
from keras.datasets import cifar10
from keras.utils import to_categorical

from utils import get_keras_dataset


class CNNMCifarModelAugmentation(CNNModel):
    def __init__(self, custom_optimizer=None, input_shape=(-1, 32, 32, 3), weight_decay: float = 1e-4,
                 learning_rate: float = 0.001, num_classes: int = 10):
        super().__init__(num_classes)

        x_train, y_train, x_test, y_test = get_keras_dataset(cifar10.load_data(), input_shape=self.input_shape)
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.input_shape = input_shape

        self.custom_optimizer = custom_optimizer

        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(self.weight_decay),
                              input_shape=input_shape))
        self.model.add(Activation('elu'))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(self.weight_decay)))
        self.model.add(Activation('elu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.2))

        self.model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(self.weight_decay)))
        self.model.add(Activation('elu'))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(self.weight_decay)))
        self.model.add(Activation('elu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.3))

        self.model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(self.weight_decay)))
        self.model.add(Activation('elu'))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(self.weight_decay)))
        self.model.add(Activation('elu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.4))

        self.model.add(Flatten())
        self.model.add(Dense(10, activation='softmax'))

        optimizer = custom_optimizer or keras.optimizers.rmsprop(lr=0.001, decay=1e-6)
        self.model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    def schedule(self, epoch):
        self.learning_rate = 0.001
        if epoch > 75:
            self.learning_rate = 0.0005
        if epoch > 100:
            self.learning_rate = 0.0003
        return self.learning_rate

    def train(self,
              epochs=20,
              batch_size=64,
              target_name="conv_nn_cifar.h5",
              save_model=False,
              with_augmentation=False):

        if with_augmentation:
            data_gen = ImageDataGenerator(
                rotation_range=15,
                width_shift_range=0.1,
                height_shift_range=0.1,
                horizontal_flip=True)
            data_gen.fit(x_train)

            model.fit_generator(data_gen.flow(x_train, y_train, batch_size=batch_size),
                                steps_per_epoch=len(x_train) // batch_size,
                                epochs=epochs,
                                verbose=1,
                                validation_data=(x_test, y_test),
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
