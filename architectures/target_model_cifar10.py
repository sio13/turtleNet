import os
import numpy as np

os.environ['KERAS_BACKEND'] = 'tensorflow'
from keras import backend as K

K.tensorflow_backend._get_available_gpus()

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop, SGD
from keras.datasets import cifar10
from keras.utils import to_categorical

from architectures.target_model import CNNModel
from utils import get_keras_dataset


class SCNNCifar10Model(CNNModel):
    def __init__(self, num_classes: int = 10, custom_optimizer=None, input_shape=(32, 32, 3)):
        super().__init__(num_classes)

        x_train, y_train, x_test, y_test = get_keras_dataset(cifar10.load_data(), input_shape=self.input_shape)
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
                              input_shape=input_shape))
        self.model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        self.model.add(MaxPool2D((2, 2)))
        self.model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        self.model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        self.model.add(MaxPool2D((2, 2)))
        self.model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        self.model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        self.model.add(MaxPool2D((2, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
        self.model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
        self.model.add(Dense(10, activation='softmax'))

        optimizer = custom_optimizer or SGD(lr=0.001, momentum=0.9)
        self.model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    def train(self, epochs=5, batch_size=64, target_name="conv_nn_cifar.h5", save_model=False):
        x_train, y_train, _, _ = get_keras_dataset(cifar10.load_data(), input_shape=(-1, 32, 32, 3))

        self.model.fit(x_train, to_categorical(y_train, num_classes=self.num_classes), epochs=epochs,
                       batch_size=batch_size)
        if save_model:
            self.model.save(f"models/{target_name}")

    def save_model(self, model_path: str):
        print(f"Saving model into {model_path}")
        self.model.save(model_path)
