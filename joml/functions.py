import numpy as np


class ActivationFunction:
    """
    `ActivationFunction` are used in `Layer` to perform a non-linear mapping.

    `ActivationFunction` define the value they return based on a input.
    They also define the value of their derivative.
    """
    def __init__(self, value, derivative):
        self._value = value
        self._derivative = derivative

    def value(self, x_array):
        return np.apply_along_axis(self._value, axis=0, arr=x_array)

    def der(self, x_array):
        return np.apply_along_axis(self._derivative, axis=0, arr=x_array)


class ReLu(ActivationFunction):

    @staticmethod
    def _relu_value(x_array): return x_array.clip(0.0)

    @staticmethod
    def _relu_derivative(x_array): return 1. * (x_array.clip(0.0) != 0.0)

    def __init__(self):
        value = ReLu._relu_value
        derivative = ReLu._relu_derivative
        super().__init__(value, derivative)

    def __str__(self):
        return "ReLu"


class SoftMax(ActivationFunction):

    @staticmethod
    def _softmax_value(x_array):
        C = np.max(x_array)
        shifted = x_array - C
        exps = np.exp(shifted)
        return exps / np.sum(exps)

    @staticmethod
    def _softmax_derivative(x_array):
        x_array_vert = x_array.reshape(-1, 1)
        return np.diagflat(x_array_vert) - np.dot(x_array_vert, x_array_vert.T)

    def __init__(self):
        value = SoftMax._softmax_value
        derivative = SoftMax._softmax_derivative
        super().__init__(value, derivative)

    def __str__(self):
        return "Softmax"


class Identity(ActivationFunction):

    @staticmethod
    def _identity_value(x_array): return x_array

    @staticmethod
    def _identity_derivative(x_array): return (0 * x_array) + 1.0

    def __init__(self):
        value = Identity._identity_value
        derivative = Identity._identity_derivative
        super().__init__(value, derivative)

    def __str__(self):
        return "Identity"
