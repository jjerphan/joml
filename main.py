import os

from functions import SoftMax, CrossEntropy
from neurons import Network, Layer
from utils import load_sets, one_hot, transpose_weights


def first_architecture():
    """
    Return an initialised 14-100-40-4 architecture.

    :return: a Network
    """

    network = Network(input_size=14)

    network.stack(Layer(size=100))
    network.stack(Layer(size=40))

    network.output(output_size=4, output_function=SoftMax(), cost_function=CrossEntropy())

    return network


def second_architecture():
    """
    Return an initialised 14-28x6-4 architecture.

    :return: a Network
    """
    network = Network(input_size=14)

    for _ in range(6):
        network.stack(Layer(size=28))

    network.output(output_size=4, output_function=SoftMax(), cost_function=CrossEntropy())

    return network


def third_architecture():
    """
    Return an initialised 14-14x28-4 architecture.

    :return: a Network
    """
    network = Network(input_size=14)

    for _ in range(28):
        network.stack(Layer(size=14))

    network.output(output_size=4, output_function=SoftMax(), cost_function=CrossEntropy())

    return network


if __name__ == "__main__":
    data_folder = os.path.join("data")

    x_train, y_train, x_test, y_test = load_sets(data_folder)
    y_train = one_hot(y_train)

    if transpose_weights:
        y_train = y_train.T

    n_train = 12

    x_train = x_train[:, 0:n_train]
    y_train = y_train[:, 0:n_train]

    network = first_architecture()

    network.train(x_train,y_train)

    accuracy, values, preds = network.test(x_train, y_train)
    print(accuracy)
