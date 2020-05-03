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

from utils import get_keras_dataset


class CNNModel:
    def __init__(self, custom_optimizer=None, input_shape=(32, 32, 3)):
        self.model = Sequential()
        self.model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
        self.model.add(MaxPool2D(pool_size=(2, 2)))
        self.model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        self.model.add(MaxPool2D(pool_size=(2, 2)))
        self.model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        self.model.add(MaxPool2D(pool_size=(2, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(100, activation='softmax'))

        optimizer = custom_optimizer or SGD(lr=0.001, momentum=0.9)
        self.model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    def train_on_cifar100(self, epochs=5, batch_size=64, target_name="conv_nn_cifar100.h5"):
        x_train, y_train, _, _ = get_keras_dataset(cifar100.load_data(), input_shape=(-1, 32, 32, 3))

        self.model.fit(x_train, to_categorical(y_train, num_classes=100), epochs=epochs, batch_size=batch_size)
        self.model.save(f"models/{target_name}")

    def test_on_cifar100(self):
        _, _, x_test, y_test = get_keras_dataset(cifar100.load_data(), input_shape=(-1, 32, 32, 3))
        return self.model.evaluate(x_test, to_categorical(y_test, num_classes=100))

    def save_model(self, model_path: str):
        print(f"Saving model into {model_path}")
        self.model.save(model_path)
