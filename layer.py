import numpy as np

from joml.functions import ActivationFunction, ReLu, SoftMax


class Layer:

    def __init__(self, size: int, activation_function: ActivationFunction = ReLu(), name="Simple Layer"):
        self.name = name
        self.size = size
        self._activation_function = activation_function
        self._initialised = False

        # Unknown for now, gets updated then when self.initialize get called.
        self._previous_layer_size = 0

        # Parameters for the affine transformation from layer l-1 to layer l
        self.W = None
        self.b = np.zeros(size) + 0.001  # Initialization with a small magnitude

        # Last activation from previous layer
        self.a_lm1 = None

        # Last affine transformation of combination of inputs on this layer
        self.z_l = None

        # Error on this layer l
        self.delta_l = None

    def __str__(self):
        string = f" - {self.name}\n"
        string += f"  - Size : {self.size}\n"
        string += f"  - Activation Function : {self._activation_function}\n"
        string += f"  - Dims : {self.dims}\n"
        string += f"  - W shape : {self.W.shape}\n"
        return string

    @property
    def dims(self) -> (int, int):
        return self.size, self._previous_layer_size

    @staticmethod
    def from_W_b(previous_layer_size: int, W: np.ndarray, b: np.ndarray):
        size = b.shape[0]
        layer = Layer(size)
        layer._previous_layer_size = previous_layer_size
        layer.W = W
        layer.b = b
        layer._initialised = True

        return layer

    def forward_propagate(self, inputs: np.ndarray) -> np.ndarray:
        assert self._initialised

        assert inputs.shape[0] == self._previous_layer_size
        self.a_lm1 = inputs

        # Affine transform
        z_array = self.W.dot(inputs)
        # NOTE : Double transposition, works for now but not nice
        z_array = np.add(z_array.T, self.b).T

        self.z_l = z_array

        outputs = self._activation_function.value(z_array)

        assert self.z_l.shape[0] == self.size

        return outputs

    def initialise(self, previous_layer_size: int) -> (int, int):
        self._previous_layer_size = previous_layer_size
        # He-et-al initialization for the weights
        self.W = np.random.randn(self.dims[0], self.dims[1]) * 2 / np.sqrt(self._previous_layer_size)
        self._initialised = True
        return self.dims

    def back_propagate(self, W_T_lp1: np.ndarray, delta_lp1: np.ndarray):
        # General error with activation functions
        der = self._activation_function.der(self.z_l)
        delta_l = W_T_lp1 * der
        assert (delta_l.shape[0] == self.size)

        # We need to pass this so that the previous layer
        # can calculate its errors
        W_T_delta_l = self.W.T.dot(delta_l)

        self.delta_l = delta_l

        return W_T_delta_l, delta_l

    def get_d_W(self) -> np.ndarray:
        # NOTE : taking the means and then the outer product isn't the same as taking
        # outer products and then taking the mean of the result
        # Tedious boilerplate bellow: could be improved
        a_lm1_T = self.a_lm1.T
        delta_l_T = self.delta_l.T
        d_W = self.W * 0
        for a_T, d_T in zip(a_lm1_T, delta_l_T):
            delta_l = d_T.reshape(-1, 1)
            a = a_T.reshape(-1, 1).T
            d_W += delta_l.dot(a)

        n_sample = self.a_lm1.shape[1]
        d_W /= n_sample

        return d_W

    def get_d_b(self) -> np.ndarray:
        return self.delta_l.mean(axis=1)

    def optimize(self, learning_rate: float):
        # Gettings gradients
        d_W = self.get_d_W()
        d_b = self.get_d_b()

        self.W -= learning_rate * d_W
        self.b -= learning_rate * d_b

class SoftMaxCrossEntropyOutputLayer(Layer):

    def __init__(self, size: int):
        super().__init__(size, activation_function=SoftMax(), name="OutputLayer")

    def __str__(self):
        string = super().__str__()
        string += f"  - Cost Function : Cross Entropy\n"
        return string

    @staticmethod
    def from_W_b(previous_layer_size: int, W: np.ndarray, b: np.ndarray):
        size = b.shape[0]
        layer = SoftMaxCrossEntropyOutputLayer(size)
        layer._previous_layer_size = previous_layer_size
        layer.b = b
        layer.W = W
        layer._initialised = True

        return layer

    @staticmethod
    def cost(y, y_hat) -> float:
        eps = 10e-9
        n_sample = y.shape[1]
        value = - np.sum(np.sum(np.log(y_hat + eps) * y)) / n_sample

        return value

    def back_propagate(self, y: np.ndarray, **kwargs) -> (np.ndarray, np.ndarray):
        # Value of the error with the Softmax/CrossEntropy combo
        y_hat = self._activation_function.value(self.z_l)
        delta_l = y_hat - y
        assert (delta_l.shape[0] == self.size)

        W_T_delta_l = self.W.T.dot(delta_l)

        self.delta_l = delta_l

        return W_T_delta_l, delta_l


