import os

os.environ['KERAS_BACKEND'] = 'tensorflow'
from keras import backend as K

K.tensorflow_backend._get_available_gpus()

import time

import keras
from keras.datasets import mnist
from keras.utils import to_categorical
from keras import layers, Model
from keras.models import *
from keras.layers import *
from regularization import InstanceNormalization
from keras.optimizers import Adam, SGD
from keras.metrics import binary_accuracy
from keras import backend as K
import numpy as np
from numpy.random import seed
from tensorflow import set_random_seed
import tensorflow as tf
import sys

sys.path.append('../')
from utils import get_keras_dataset


class AdvGAN:
    def __init__(self):
        self.img_width = 28
        self.img_height = 28
        self.input_shape = (self.img_width, self.img_height, 1)  # 1 channel for grayscale

        optimizer_g = Adam(0.0002)
        optimizer_d = SGD(0.01)

        inputs = Input(shape=self.input_shape)
        outputs = self.build_generator(inputs)
        self.G = Model(inputs, outputs)
        self.G.summary()

        outputs = self.build_discriminator(self.G(inputs))
        self.D = Model(inputs, outputs)
        self.D.compile(loss=keras.losses.binary_crossentropy, optimizer=optimizer_d, metrics=[self.custom_acc])
        self.D.summary()

        outputs = self.build_target(self.G(inputs))
        self.target = Model(inputs, outputs)
        self.target.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),
                            metrics=['accuracy'])
        self.target.summary()

        self.stacked = Model(inputs=inputs,
                             outputs=[self.G(inputs), self.D(self.G(inputs)), self.target(self.G(inputs))])
        self.stacked.compile(
            loss=[self.generator_loss, keras.losses.binary_crossentropy, keras.losses.binary_crossentropy],
            optimizer=optimizer_g)
        self.stacked.summary()

    @staticmethod
    def perturb_loss(preds, thresh=0.3):
        zeros = tf.zeros((tf.shape(preds)[0]))
        return tf.reduce_mean(tf.maximum(zeros, tf.norm(tf.reshape(preds, (tf.shape(preds)[0], -1)), axis=1) - thresh))

    @staticmethod
    def generator_loss(y_true, y_pred):
        return K.mean(K.maximum(K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1)) - 0.3, 0), axis=-1)
        # ||G(x) - x||_2 - c, where c is user-defined. Here it is set to 0.3

    @staticmethod
    def custom_acc(y_true, y_pred):
        return binary_accuracy(K.round(y_true), K.round(y_pred))

    @staticmethod
    def build_discriminator(inputs):

        D = Conv2D(32, 4, strides=(2, 2))(inputs)
        D = LeakyReLU()(D)
        D = Dropout(0.4)(D)
        D = Conv2D(64, 4, strides=(2, 2))(D)
        D = BatchNormalization()(D)
        D = LeakyReLU()(D)
        D = Dropout(0.4)(D)
        D = Flatten()(D)
        D = Dense(64)(D)
        D = BatchNormalization()(D)
        D = LeakyReLU()(D)
        D = Dense(1, activation='sigmoid')(D)
        return D

    @staticmethod
    def build_generator(inputs):
        # c3s1-8
        G = Conv2D(8, 3, padding='same')(inputs)
        G = InstanceNormalization()(G)
        G = Activation('relu')(G)

        # d16
        G = Conv2D(16, 3, strides=(2, 2), padding='same')(G)
        G = InstanceNormalization()(G)
        G = Activation('relu')(G)

        # d32
        G = Conv2D(32, 3, strides=(2, 2), padding='same')(G)
        G = InstanceNormalization()(G)
        G = Activation('relu')(G)

        residual = G
        # four r32 blocks
        for _ in range(4):
            G = Conv2D(32, 3, padding='same')(G)
            G = BatchNormalization()(G)
            G = Activation('relu')(G)
            G = Conv2D(32, 3, padding='same')(G)
            G = BatchNormalization()(G)
            G = layers.add([G, residual])
            residual = G

        # u16
        G = Conv2DTranspose(16, 3, strides=(2, 2), padding='same')(G)
        G = InstanceNormalization()(G)
        G = Activation('relu')(G)

        # u8
        G = Conv2DTranspose(8, 3, strides=(2, 2), padding='same')(G)
        G = InstanceNormalization()(G)
        G = Activation('relu')(G)

        # c3s1-3
        G = Conv2D(1, 3, padding='same')(G)
        G = InstanceNormalization()(G)
        G = Activation('relu')(G)
        G = layers.add([G, inputs])

        return G

    @staticmethod
    def build_target(inputs):
        f = Conv2D(filters=32, kernel_size=(5, 5), padding='Same', activation='relu', input_shape=(28, 28, 1))(inputs)
        f = Conv2D(filters=32, kernel_size=(5, 5), padding='Same', activation='relu')(f)
        f = MaxPool2D(pool_size=(2, 2))(f)
        f = Dropout(0.25)(f)
        f = Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu')(f)
        f = Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu')(f)
        f = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(f)
        f = Dropout(0.25)(f)
        f = Flatten()(f)
        f = Dense(256, activation="relu")(f)
        f = Dropout(0.5)(f)
        f = Dense(10, activation="softmax")(f)
        return f

    def get_batches(self, start, end, x_train, y_train):
        x_batch = x_train[start:end]
        x_batch_perturbed = self.G.predict_on_batch(x_batch)
        y_batch = y_train[start:end]
        return x_batch, x_batch_perturbed, y_batch

    def train_discriminator_on_batch(self, batches):
        x_batch, x_batch_perturbed, _ = batches

        # for each batch:
        # predict noise on generator: G(z) = batch of fake images
        # train fake images on discriminator: D(G(z)) = update D params per D's classification for fake images
        # train real images on disciminator: D(x) = update D params per classification for real images

        # Update D params
        self.D.trainable = True
        d_loss_real = self.D.train_on_batch(x_batch, np.random.uniform(0.9, 1.0, size=(
            len(x_batch), 1)))  # real=1, positive label smoothing
        d_loss_fake = self.D.train_on_batch(x_batch_perturbed, np.zeros((len(x_batch_perturbed), 1)))  # fake=0
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        return d_loss  # (loss, accuracy) tuple

    def train_stacked_on_batch(self, batches):
        x_batch, _, y_batch = batches
        flipped_y_batch = 9 - y_batch

        # for each batch:
        # train fake images on discriminator: D(G(z)) = update G params per D's classification for fake images

        # Update only G params
        self.D.trainable = False
        self.target.trainable = False
        stacked_loss = self.stacked.train_on_batch(x_batch, [x_batch, np.ones((len(x_batch), 1)),
                                                             to_categorical(flipped_y_batch)])
        # input to full GAN is original image
        # output 1 label for generated image is original image
        # output 2 label for discriminator classification is real/fake; G wants D to mark these as real=1
        # output 3 label for target classification is 1/3; g wants to flip these so 1=1 and 3=0
        return stacked_loss  # (total loss, hinge loss, gan loss, adv loss) tuple

    def train_and_generate(self,
                           fit_epochs=5,
                           train_network=True,
                           model_name='keras-stolen-model.h5',
                           model_dir='models'
                           ):
        x_train, y_train, x_test, y_test = get_keras_dataset(mnist.load_data())

        if train_network:
            self.target.fit(x_train, to_categorical(y_train), epochs=fit_epochs)
            self.target.save(f"{model_dir}/{model_name}")
        else:
            self.target = load_model(f'{model_dir}/{model_name}',
                                     custom_objects={'InstanceNormalization': InstanceNormalization})

        self.generate_perturbations(x_train, y_train, x_test, y_test)  # change name

    def generate_perturbations(self,
                               x_train,
                               y_train,
                               x_test,
                               y_test,
                               epochs=50,
                               batch_size=128,
                               dir_name="np_adversarial"
                               ):

        num_batches = len(x_train) // batch_size

        for epoch in range(epochs):
            print("Epoch " + str(epoch))
            start_time = time.time()

            for batch_index in range(num_batches - 1):
                batches = self.get_batches(batch_size * batch_index,
                                           batch_size * (batch_index + 1),
                                           x_train,
                                           y_train)

                self.train_discriminator_on_batch(batches)
                self.train_stacked_on_batch(batches)

            x_batch, x_batch_perturbed, y_batch = self.get_batches(batch_size * batch_index,
                                                                   batch_size * (batch_index + 1),
                                                                   x_train,
                                                                   y_train)

            d_loss, d_acc = self.train_discriminator_on_batch((x_batch, x_batch_perturbed, y_batch))
            g_loss, hinge_loss, gan_loss, adv_loss = self.train_stacked_on_batch(
                (x_batch, x_batch_perturbed, y_batch))

            target_acc = self.target.test_on_batch(x_batch_perturbed, to_categorical(y_batch))[1]
            # target_predictions = self.target.predict_on_batch(x_batch_perturbed)

            # misclassified = np.where(y_batch.reshape((end - start,)) != np.argmax(target_predictions, axis=1))[0]
            end_time = time.time()
            print(
                f"Discriminator -- Loss:{d_loss} Accuracy:{d_acc * 100}\n"
                f"Generator -- Loss:{gan_loss} Hinge Loss: {hinge_loss}\n"
                f"Target Loss: {adv_loss} Accuracy:{target_acc * 100.}\n"
                f"Time per epoch: {end_time - start_time} seconds")
            if epoch % 10 == 0:
                x_test_perturbed = self.G.predict_on_batch(x_test)
                np.save(f"{dir_name}/miss{epoch}_test", x_test_perturbed)
                self.G.save(f"models/generator_{epoch}")


if __name__ == '__main__':
    seed(5)
    set_random_seed(1)
    gan = AdvGAN()
    gan.train_and_generate(train_network=False)
