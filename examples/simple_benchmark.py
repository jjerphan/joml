import numpy as np
import os

from joml.network import Network
from joml.layer import Layer, SoftMaxCrossEntropyOutputLayer
from joml.functions import ReLu
from joml.utils import one_hot

np.random.seed(1337)


def load_sets(folder, dtype=np.float32, delimiter=","):
    print(f"Loading dataset from {folder}")
    x_train = np.loadtxt(os.path.join(folder, "x_train.csv"), dtype=dtype, delimiter=delimiter)
    y_train = np.loadtxt(os.path.join(folder, "y_train.csv"), dtype=dtype, delimiter=delimiter)
    x_test = np.loadtxt(os.path.join(folder, "x_test.csv"), dtype=dtype, delimiter=delimiter)
    y_test = np.loadtxt(os.path.join(folder, "y_test.csv"), dtype=dtype, delimiter=delimiter)

    x_train = x_train.T
    x_test = x_test.T

    # Conversion to integers
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    return x_train, y_train, x_test, y_test


if __name__ == "__main__":
    # Loading data
    data_folder = "../data"  # set your own value
    x_train, y_train, x_test, y_test = load_sets(data_folder)
    y_train = one_hot(y_train)
    y_test = one_hot(y_test)

    # Defining the network
    network = Network(input_size=14, name="14-100-40-4 Arch")

    network.stack(Layer(size=100, activation_function=ReLu()))
    network.stack(Layer(size=40, activation_function=ReLu()))

    network.output(SoftMaxCrossEntropyOutputLayer(size=4))

    # Printing information
    print(network)

    # Training and testing
    network.train(x_train, y_train)
    y_pred, y_hat, acc = network.test(x_test, y_test)
    print("Testing accuracy :", acc)

    # logger1 = network.benchmark(x_train, y_train, x_test, y_test, csv_file_name=log_file(n1.name),
    #                             num_epochs=10,
    #                             learning_rate=0.002)
    # logger1.dump_results()
    # logger1.plot_benchmark()
