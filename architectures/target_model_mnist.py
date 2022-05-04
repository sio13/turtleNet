import os
import numpy as np

os.environ['KERAS_BACKEND'] = 'tensorflow'
from keras import backend as K

K.tensorflow_backend._get_available_gpus()

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.datasets import mnist
from keras.utils import to_categorical

from architectures.target_model import CNNModel
from utils import get_keras_dataset


class CNNModelMnist(CNNModel):
    def __init__(self, num_classes: int = 10, custom_optimizer=None, input_shape=(28, 28, 1)):
        super().__init__(num_classes)

        self.input_shape = input_shape
        x_train, y_train, x_test, y_test = get_keras_dataset(mnist.load_data())
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

        self.model = Sequential()

        self.model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same',
                              activation='relu', input_shape=input_shape))
        self.model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same',
                              activation='relu'))
        self.model.add(MaxPool2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))
        self.model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same',
                              activation='relu'))
        self.model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same',
                              activation='relu'))
        self.model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(Dense(256, activation="relu"))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(10, activation="softmax"))

        optimizer = custom_optimizer or RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

        self.model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])


    def save_model(self, model_path: str):
        print(f"Saving model into {model_path}")
        self.model.save(model_path)
