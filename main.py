from attack import Attack
from keras.datasets import mnist


def main():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = (x_train / 255).reshape((len(x_train), 28, 28, 1))


if __name__ == '__main__':
    main()