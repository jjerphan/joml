import numpy as np

from functions import ActivationFunction, ReLu


class SoftMaxCrossEntropy:

    @staticmethod
    def _softmax_value(x_array: np.ndarray):
        # Shifted for numerical stability
        exps = np.exp(x_array - np.max(x_array))
        return exps / np.sum(exps)

    # To apply this on a matrices but column wise
    @staticmethod
    def value(x_array: np.ndarray):
        return np.apply_along_axis(SoftMaxCrossEntropy._softmax_value, axis=0, arr=x_array)

    @staticmethod
    def derivative(y_hat: np.ndarray, y: np.ndarray):
        n_sample = y_hat.shape[0]
        return (y_hat - y) / n_sample

    eps = 10e-9

    @staticmethod
    def cost(y, y_hat):
        n_sample = y.shape[1]
        value = - np.sum(np.sum(np.log(y_hat + SoftMaxCrossEntropy.eps) * y)) / n_sample

        return value

    def __init__(self, size: int):
        self.size = size

        # Initialization with a small magnitude
        self._biases = np.zeros(size) + 0.001

        # Unknown for now, gets updated then when self.initialize get called.
        self._previous_layer_input_size = 0
        self._weights = None
        self._initialised = False

        self._input_last_value = None
        self._activation_last_value = None
        self.delta_L = None  # layer l (this layer)
        self._error_last_value_last_layer = None  # layer l - 1 (previous layer)

    def forward_propagate(self, inputs: np.ndarray) -> np.ndarray:
        """
        Calculate the outputs of the layer based on inputs

        :param inputs: inputs as a (n_{l-1}, n_samples) np.ndarray
        :return: outputs as a (n_l, n_samples) np.ndarray
        """
        assert self._initialised

        # For now, let's take the mean
        self._input_last_value = inputs#np.mean(inputs, axis=1)
        assert self._input_last_value.shape[0] == self._previous_layer_input_size

        z_array = self._weights.dot(inputs)
        z_array = np.add(z_array.T, self._biases).T  # not nice for now

        outputs = self.value(z_array)

        self._activation_last_value = outputs#np.mean(outputs, axis=1)
        assert self._activation_last_value.shape[0] == self.size

        return outputs

    def initialise(self, previous_layer_size: int) -> (int, int):
        """
        Initialise the Layer internals using the previous layer info.

        :param previous_layer_size: the size of the previous layer
        :return: dimensions of the weights matrix as a tuple
        """
        self._previous_layer_input_size = previous_layer_size
        # He-et-al initialization
        self._weights = np.random.randn(self.dims[0], self.dims[1]) * 2 / np.sqrt(self._previous_layer_input_size)
        self._initialised = True
        return self.dims

    def back_propagate(self, y: np.ndarray):
        """
        Propagate the error signals.

        :param y: the incoming error signals as a (n_l, 1) np.ndarray
        :return: the outgoing error signals as a (n_{l-1}, 1) np.ndarray
        """
        y_hat = self._activation_last_value
        # Value of the derivative for the Softmax/CrossEntropy combo
        delta_L = y_hat - y
        assert(delta_L.shape[0] == self.size)

        W_T_L = self._weights.T.dot(delta_L)

        self.delta_L = delta_L

        return W_T_L, delta_L

    def optimize(self, learning_rate: float):
        """
        Perform a gradient descent on the weights

        :param learning_rate:
        :return:
        """
        # print("Weights", self._weights.shape)
        a_lminus1 = self._input_last_value.mean(axis=1).reshape(-1, 1)
        delta_L = self.delta_L.mean(axis=1).reshape(-1, 1)
        # print("a_lminus1", a_lminus1.shape)
        # print("delta_L", delta_L.shape)
        gradient = delta_L.dot(a_lminus1.T)
        # print("gradient", gradient.shape)

        old = self._weights
        # print(gradient)
        self._weights -= learning_rate * gradient
        # print("Biases", self._biases.shape)
        # print("Error", delta_L.shape)
        self._biases -= learning_rate * np.ndarray.flatten(delta_L)

    @property
    def dims(self) -> (int, int):
        """
        Dimensions of the layer weights matrix.

        :rtype: (int,int)
        """
        return self.size, self._previous_layer_input_size

    def __str__(self):
        string = " - Output Layer\n"
        string += f"  - Size : {self.size}\n"
        string += f"  - Activation Function : SoftMax\n"
        string += f"  - Cost Function : Cross Entropy\n"
        string += f"  - W : {self.dims}\n"
        return string
