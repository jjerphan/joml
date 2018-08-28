import os

import numpy as np
from functions import ReLu
from logger import CSVLogger
from network import Network
from layer import Layer
from output_layer import SoftMaxCrossEntropy
from utils import load_sets, one_hot, transpose_weights
from time import gmtime, strftime

def first_architecture():
    """
    Return an initialised 14-100-40-4 architecture.

    :return: a Network
    """

    network = Network(input_size=14)

    network.stack(Layer(size=100, activation_function=ReLu()))
    network.stack(Layer(size=40, activation_function=ReLu()))

    network.output(SoftMaxCrossEntropy(size=4))

    return network


def second_architecture():
    """
    Return an initialised 14-28x6-4 architecture.

    :return: a Network
    """
    network = Network(input_size=14)

    for _ in range(6):
        network.stack(Layer(size=28))

    network.output(SoftMaxCrossEntropy(size=4))

    return network


def third_architecture():
    """
    Return an initialised 14-14x28-4 architecture.

    :return: a Network
    """
    network = Network(input_size=14)

    for _ in range(28):
        network.stack(Layer(size=14))

    network.output(SoftMaxCrossEntropy(size=4))

    return network


if __name__ == "__main__":
    data_folder = os.path.join("data")
    out_folder = os.path.join("logs")

    x_train, y_train, x_test, y_test = load_sets(data_folder)
    y_train = one_hot(y_train)

    if transpose_weights:
        y_train = y_train.T

    # n_train = 29
    # x_train = x_train[:, 0:n_train]
    # y_train = y_train[:, 0:n_train]

    current_datetime = strftime("%Y-%m-%d-%H:%M", gmtime())
    log_file = os.path.join(out_folder, f"log_arch1-{current_datetime}.csv")

    network = first_architecture().with_logger(CSVLogger(log_file))

    network.train(x_train, y_train, num_epochs=1)

    accuracy, values, preds = network.test(x_train, y_train)

    print(values.sum(axis=0))
    print(accuracy)
