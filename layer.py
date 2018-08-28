import numpy as np

from functions import ActivationFunction, ReLu


class Layer:
    """`Layers` are building blocks of a `Network`.

    `Layer` is defined by a `size` (i.e. number of neurons in the Layer)
    and an `ActivationFunction`.

    `Layers` can be stacked in a `Network` using `network.stack(theLayer)`

    Note:
        Layer is for now implicitly a Fully-Connected Layer.

    """

    def __init__(self, size: int, activation_function: ActivationFunction = ReLu()):
        self.size = size
        self._activation_function = activation_function

        # Initialization with a small magnitude
        self._biases = np.zeros(size) + 0.001

        # Unknown for now, gets updated then when self.initialize get called.
        self._previous_layer_input_size = 0
        self._weights = None
        self._initialised = False
        self._input_last_value = None
        self._activation_last_value = None
        self.delta_l = None # layer l (this layer)
        self.delta_lp1 = None # layer l - 1 (previous layer)

    def forward_propagate(self, inputs: np.ndarray) -> np.ndarray:
        """
        Calculate the outputs of the layer based on inputs

        :param inputs: inputs as a (n_{l-1}, n_samples) np.ndarray
        :return: outputs as a (n_l, n_samples) np.ndarray
        """
        assert self._initialised

        # For now, let's take the mean
        self._input_last_value = inputs# np.mean(inputs, axis=1)
        assert self._input_last_value.shape[0] == self._previous_layer_input_size

        z_array = self._weights.dot(inputs)
        z_array = np.add(z_array.T, self._biases).T  # not nice for now

        outputs = self._activation_function.value(z_array)

        self._activation_last_value = outputs #np.mean(outputs, axis=1)
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

    def back_propagate(self, W_T_lp1: np.ndarray, delta_lp1):
        """
        Propagate the error signals.

        :param errors: the incoming error signals as a (n_l, 1) np.ndarray
        :return: the outgoing error signals as a (n_{l-1}, 1) np.ndarray
        """
        der = self._activation_function.der(self._activation_last_value)
        delta_l = W_T_lp1 * der
        assert(delta_l.shape[0] == self.size)

        W_T_l = self._weights.T.dot(delta_l)

        self.delta_lp1 = delta_lp1
        self.delta_l = delta_l

        return W_T_l, delta_l

    def _optimize(self, learning_rate: float):
        """
        Perform a gradient descent on the weights

        :param learning_rate:
        :return:
        """
        # print("Weights", self._weights.shape)

        a_lminus1 = self._input_last_value.mean(axis=1).reshape(-1, 1)
        delta_l = self.delta_l.mean(axis=1).reshape(-1, 1)
        # print("a_lminus1", a_lminus1.shape)
        # print("delta_l", delta_l.shape)
        gradient = delta_l.dot(a_lminus1.T)
        # print("gradient", gradient.shape)

        old = self._weights
        self._weights -= learning_rate * gradient
        # print("Biases", self._biases.shape)
        # print("Error", delta_l.shape)
        self._biases -= learning_rate * np.ndarray.flatten(delta_l)

    @property
    def dims(self) -> (int, int):
        """
        Dimensions of the layer weights matrix.

        :rtype: (int,int)
        """
        return self.size, self._previous_layer_input_size

    def __str__(self):
        string = " - Simple Layer\n"
        string += f"  - Size : {self.size}\n"
        string += f"  - Activation Function : {self._activation_function}\n"
        string += f"  - W : {self.dims}\n"
        return string