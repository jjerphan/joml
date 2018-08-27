import numpy as np
import os

# The default float type we are using
float_type = np.float32

formatter = "%.16f"

# Let N_l be the size of the l-th layer.
# The weights are by default represented by a (N_l,N_{l+1}) matrix
# Here we transpose it to have a (N_{l+1}, N_l) matrix so that we can use
# canonical matrix-vector operations.
transpose_weights = True


def load_sets(folder, dtype=np.float32, delimiter=","):
    print(f"Loading dataset from {folder}")
    x_train = np.loadtxt(os.path.join(folder, "x_train.csv"), dtype=dtype, delimiter=delimiter)
    y_train = np.loadtxt(os.path.join(folder, "y_train.csv"), dtype=dtype, delimiter=delimiter)
    x_test = np.loadtxt(os.path.join(folder, "x_test.csv"), dtype=dtype, delimiter=delimiter)
    y_test = np.loadtxt(os.path.join(folder, "y_test.csv"), dtype=dtype, delimiter=delimiter)

    if transpose_weights:
        x_train = x_train.T
        x_test = x_test.T

    # Conversion to integers
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    return x_train, y_train, x_test, y_test


def one_hot(y_vect: np.ndarray, num_classes=None):
    """
    Returns the one representation of a flat vector of labels.

    :param y_vect: a flat (n_sample,) vector of labels
    :param num_classes: the number of classes (by default inferred from y_vect)
    :return: a one-hot representation of y_vect as a (n_samples, num_classes) np.ndarray
    """
    if num_classes is None:
        num_classes = int(np.max(y_vect) + 1)

    one_hot_y = np.squeeze(np.eye(num_classes)[y_vect.reshape(-1)])

    return one_hot_y
