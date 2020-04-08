import os

os.environ['KERAS_BACKEND'] = 'tensorflow'
from keras import backend as K

K.tensorflow_backend._get_available_gpus()

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.datasets import mnist
from keras.utils import to_categorical
import numpy as np


class CNNModel:
    def __init__(self, custom_optimizer=None, input_shape=(28, 28, 1)):
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

    def train_on_mnist(self, epochs=5, target_name="conv_nn.h5"):
        train, _ = mnist.load_data()
        Y_train = train[1]
        Y_train = to_categorical(Y_train, num_classes=10)
        X_train = train[0]

        # normalizing data
        X_train = np.divide(X_train, 255.0)
        X_train = X_train.reshape(-1, 28, 28, 1)

        self.model.fit(X_train, Y_train, epochs=epochs)
        self.model.save(f"models/{target_name}")

    def test_on_mnist(self):
        _, test = mnist.load_data()
        Y_test = test[1]
        Y_test = to_categorical(Y_test, num_classes=10)
        X_test = test[0]

        # normalizing data
        X_test = np.divide(X_test, 255.0)
        X_test = X_test.reshape(-1, 28, 28, 1)

        return self.model.evaluate(X_test, Y_test)
