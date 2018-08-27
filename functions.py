import numpy as np


class ActivationFunction:

    def __init__(self, value, derivative):
        self._value = value
        self._derivative = derivative

    def value(self, x_array):
        return np.apply_along_axis(self._value, axis=0, arr=x_array)

    def der(self, x_array):
        return np.apply_along_axis(self._derivative, axis=0, arr=x_array)


class CostFunction:

    def __init__(self, value, derivative):
        self._value = value
        self._derivative = derivative

    def value(self, y_hat, y):
        return self._value(y_hat, y)

    def der(self, y_hat, y):
        return self._derivative(y_hat, y)


class ReLu(ActivationFunction):

    def __init__(self):
        value = lambda x_array: x_array * (x_array > 0)
        derivative = lambda x_array: 1 * (x_array > 0)
        super().__init__(value, derivative)

    def __str__(self):
        return "ReLu"


class Sigmoid(ActivationFunction):

    @staticmethod
    def _sigmoid_val(x_array):
        return 1 / (1+np.exp(-x_array))

    @staticmethod
    def _sigmoid_der(x_array):
        val = Sigmoid._sigmoid_val(x_array)

        return val * (1 - val)

    def __init__(self):
        value = Sigmoid._sigmoid_val
        derivative = Sigmoid._sigmoid_der
        super().__init__(value, derivative)

    def __str__(self):
        return "Sigmoid"


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


class CrossEntropy(CostFunction):

    eps = 10e-9

    @staticmethod
    def _cross_entropy_value(y, y_hat):
        n_sample = y.shape[1]
        value = - np.sum(np.sum(np.log(y_hat + CrossEntropy.eps) * y)) / n_sample

        return value

    @staticmethod
    def _cross_entropy_derivative(y, y_hat):
        value = - np.mean(y/(y_hat + CrossEntropy.eps), axis=1)

        assert(value.shape[0] == y.shape[0])
        return value

    def __init__(self):
        value = CrossEntropy._cross_entropy_value
        derivative = CrossEntropy._cross_entropy_derivative
        super().__init__(value, derivative)

    def __str__(self):
        return "Cross Entropy"
