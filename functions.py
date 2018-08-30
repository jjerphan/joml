import numpy as np


class ActivationFunction:

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
        exps = np.exp(x_array - np.max(x_array))
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