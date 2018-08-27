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
        self._biases = np.random.random(size)

        # Unknown for now, gets updated then when self.initialize get called.
        self._previous_layer_input_size = 0
        self._weights = None
        self._initialised = False
        self._input_last_value = None
        self._activation_last_value = None
        self._error_signal_last_value = None

    def forward_propagate(self, inputs: np.ndarray) -> np.ndarray:
        """
        Calculate the outputs of the layer based on inputs

        :param inputs: inputs as a (n_{l-1}, n_samples) np.ndarray
        :return: outputs as a (n_l, n_samples) np.ndarray
        """
        assert self._initialised

        # For now, let's take the mean
        self._input_last_value = np.mean(inputs, axis=1)
        assert self._input_last_value.shape[0] == self._previous_layer_input_size

        z_array = self._weights.dot(inputs)
        z_array = np.add(z_array.T, self._biases).T  # not nice for now

        outputs = self._activation_function.value(z_array)

        self._activation_last_value = np.mean(outputs, axis=1)
        assert self._activation_last_value.shape[0] == self.size


        return outputs

    def initialise(self, previous_layer_size: int) -> (int, int):
        """
        Initialise the Layer internals using the previous layer info.

        :param previous_layer_size: the size of the previous layer
        :return: dimensions of the weights matrix as a tuple
        """
        self._previous_layer_input_size = previous_layer_size
        self._weights = np.random.random(self.dims)
        self._initialised = True
        return self.dims

    def _back_propagate(self, error_signals: np.ndarray) -> np.ndarray:
        """
        Propagate the error signals.

        :param error_signals: the incoming error signals as a (n_l, 1) np.ndarray
        :return: the outgoing error signals as a (n_{l-1}, 1) np.ndarray
        """
        der = self._activation_function.der(self._input_last_value)
        print("Der", der.shape)
        yo = self._weights.T.dot(error_signals)
        print("yo",yo.shape)
        out_put_errors = yo * der
        self._error_signal_last_value = out_put_errors

        return out_put_errors

    def _optimize(self, learning_rate: float):
        """
        Perform a gradient descent on the weights

        :param learning_rate:
        :return:
        """
        # print("Weights", self._weights.shape)

        a_lminus1 = self._activation_last_value.reshape(-1, 1)
        error_signal_l = self._error_signal_last_value.reshape(-1, 1)
        # print("a_lminus1", a_lminus1.shape)
        # print("error_signal_l", error_signal_l.shape)
        gradient = a_lminus1.dot(error_signal_l.T)
        # print("gradient", gradient.shape)

        old = self._weights
        # print(gradient)
        self._weights -= learning_rate * gradient
        print("Biases", self._biases.shape)
        print("Error", error_signal_l.shape)
        self._biases -= learning_rate * error_signal_l

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