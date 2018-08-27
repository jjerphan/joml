import numpy as np


class ActivationFunction:

    def __init__(self, value, derivative):
        self._value = value
        self._derivative = derivative

    def value(self, x_array):
        return np.apply_along_axis(self._value, 0, x_array)

    def der(self, x_array):
        return self._derivative(x_array)


class CostFunction:

    def __init__(self, value, derivative):
        self.value = value
        self.derivative = derivative

    def value(self, y_hat, y):
        return self.value(y_hat, y)

    def der(self, y_hat, y):
        return self.derivative(y_hat, y)


class ReLu(ActivationFunction):

    def __init__(self):
        value = lambda x_array: x_array * (x_array > 0)
        derivative = lambda x_array: 1 * (x_array > 0)
        super().__init__(value, derivative)

    def __str__(self):
        return "ReLu"


class Sigmoid(ActivationFunction):

    def __init__(self):
        value = lambda x_array: 1 / x_array * (x_array > 0)
        derivative = lambda x_array: 1 * (x_array > 0)
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
        pass

    def __init__(self):
        value = SoftMax._softmax_value
        derivative = SoftMax._softmax_derivative
        super().__init__(value, derivative)

    def __str__(self):
        return "Softmax"


class CrossEntropy(CostFunction):

    @staticmethod
    def _cross_entropy_value(y, y_hat):
        eps = 10e-9
        n_sample = y.shape[1]
        value = - np.sum(np.sum(np.log(y_hat + eps) * y)) / n_sample

        return value

    @staticmethod
    def _cross_entropy_derivative(y, y_hat):
        n_sample = y.shape[1]
        # TODO : replace by the real value
        value = - np.sum(np.sum(np.log(y_hat) * y)) / n_sample

        return value

    def __init__(self):
        value = CrossEntropy._cross_entropy_value
        derivative = CrossEntropy._cross_entropy_derivative
        super().__init__(value, derivative)

    def __str__(self):
        return "Cross Entropy"
