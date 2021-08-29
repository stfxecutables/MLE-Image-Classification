from typing import Union

import tensorflow.keras
from tensorflow.keras import Sequential
from tensorflow.keras.activations import relu
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    MaxPool2D,
    Dropout,
    Dense,
    Flatten,
)


def get_architecture(
    dataset: str, units: int = 10
) -> Union[tensorflow.keras.Sequential, None]:
    """
    utility function to get base-learner architecture for ensemble based on the dataset name specified
    :param dataset: string representing dataset in use
    :param units: number of classes in training labels (or number of units in final softmax-layer)
    :return: Keras sequential model or None (if dataset name is not recognized)
    """
    if dataset == "MNIST" or dataset == "MNIST_BIN":
        return mnist_architecture(units)
    elif dataset == "F_MNIST" or dataset == "F_MNIST_BIN":
        return fashion_mnist_architecture(units)
    elif dataset == "CIFAR10" or dataset == "CIFAR10_BIN":
        return cifar10_architecture(units)
    return None


def mnist_architecture(units: int = 10) -> tensorflow.keras.Sequential:
    """
    gets base-learner architecture for MNIST ensemble
    :param units: number of classes in training labels (or number of units in final softmax-layer)
    :return: Keras sequential model
    """
    model = Sequential()
    model.add(BatchNormalization())

    model.add(
        Conv2D(
            128,
            kernel_size=7,
            input_shape=(28, 28, 1),
            activation=relu,
            data_format="channels_last",
            padding="same",
        )
    )
    model.add(Dropout(0.2))

    model.add(Conv2D(64, kernel_size=3, activation=relu, padding="same"))
    model.add(MaxPool2D())
    model.add(Dropout(0.2))

    model.add(Conv2D(64, kernel_size=3, activation=relu))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, kernel_size=3, activation=relu))
    model.add(Conv2D(64, kernel_size=3, activation=relu))
    model.add(Dropout(0.1))

    model.add(Flatten())
    for size in [128]:
        model.add(Dense(size))

    model.add(Dense(units=units, activation="softmax"))

    return model


def fashion_mnist_architecture(units: int = 10) -> tensorflow.keras.Sequential:
    """
    gets base-learner architecture for Fashion-MNIST ensemble
    :param units: number of classes in training labels (or number of units in final softmax-layer)
    :return: Keras sequential model
    """
    model = Sequential()
    model.add(BatchNormalization())

    model.add(
        Conv2D(
            128,
            kernel_size=7,
            input_shape=(28, 28, 1),
            activation=relu,
            data_format="channels_last",
            padding="same",
        )
    )
    model.add(Dropout(0.2))

    model.add(Conv2D(64, kernel_size=3, activation=relu, padding="same"))
    model.add(MaxPool2D())
    model.add(Dropout(0.2))

    model.add(Conv2D(64, kernel_size=3, activation=relu))
    model.add(Conv2D(64, kernel_size=3, activation=relu))
    model.add(Conv2D(64, kernel_size=3, activation=relu))
    model.add(Dropout(0.1))

    model.add(Flatten())
    for size in [128, 64]:
        model.add(Dense(size))

    model.add(Dense(units=units, activation="softmax"))

    return model


def cifar10_architecture(units: int = 10) -> tensorflow.keras.Sequential:
    """
    gets base-learner architecture for CIFAR-10 ensemble
    :param units: number of classes in training labels (or number of units in final softmax-layer)
    :return: Keras sequential model
    """
    model = Sequential()

    model.add(
        Conv2D(
            64,
            kernel_size=3,
            activation=relu,
            data_format="channels_last",
            padding="same",
        )
    )
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=3, activation=relu, padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(3, 3)))
    model.add(Dropout(0.3))

    model.add(Conv2D(128, kernel_size=3, activation=relu, padding="same"))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Conv2D(128, kernel_size=3, activation=relu, padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(64, kernel_size=3, activation=relu, padding="same"))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=3, activation=relu, padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))

    model.add(Flatten())
    for size in [128]:
        model.add(Dense(size))
    model.add(Dropout(0.3))

    model.add(Dense(units=units, activation="softmax"))

    return model
