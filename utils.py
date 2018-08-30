import numpy as np


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