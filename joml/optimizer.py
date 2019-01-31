import numpy as np

from abc import ABC


class Optimizer(ABC):
    """
    Optimize parameters using a specific schema.
    Their are used by `Layers` to

    It can persist information about previous updates to
    have an adaptive behaviour (this is up to the implementation).

    """
    def __init__(self, eps: float = 10 ** (-8), lr: float=0.01):
        """
        :param eps: numerical epsilon
        :param lr: learning rate
        """
        self._eps = eps
        self._lr = lr
        pass

    def optimize(self, param, grad):
        """
        Optimize given parameters using a gradient.

        :param param: the current parameters
        :param grad: the gradient of the parameters
        :return: the updated parameters
        """
        pass


class GradientDescent(Optimizer):
    """
    The world simplest, yet useful, optimizer.

    """
    def optimize(self, param, grad): param - self._lr * grad


class Adam(Optimizer):
    """
    Implementation of the Adam Optimizer, "an algorithm for first-order
    gradient-based optimization of stochastic objective functions, based
    on adaptive estimates of lower-order moments".

    See original paper: https://arxiv.org/pdf/1412.6980.pdf.

    Default values for parameters are the one recommended by the authors.

    """
    def __init__(self, eps: float = 10 ** (-8), lr: float=0.01, beta1: float = 0.9, beta2: float = 0.999):
        """
        :param eps: numerical epsilon
        :param lr: learning rate
        :param beta1: exponential decay rate for the first moment vector
        :param beta2: exponential decay rate for the second moment vector
        """
        super().__init__(eps, lr)
        self._beta1 = beta1
        self._beta2 = beta2

        # Timestep
        self._t = 1
        # First moment vector estimate
        self._m = 0
        # Second moment vector estimate
        self._v = 0

    def optimize(self, param, grad):
        self._m = self._beta1 * self._m + (1 - self._beta1) * grad
        self._v = self._beta2 * self._v + (1 - self._beta2) * (
                grad * grad)

        # Unbiased moment vector estimates
        m_hat = self._m / (1 - (self._beta1 ** self._t))
        v_hat = self._v / (1 - (self._beta2 ** self._t))

        self._t += 1

        return param - self._lr * m_hat / (np.sqrt(v_hat) + self._eps)
