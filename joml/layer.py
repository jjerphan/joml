import numpy as np

from joml.functions import ActivationFunction, ReLu, SoftMax


class Layer:
    """
    A `Layer` is a building block of a `Network`.

    A `Layer` has a specific `size` the is the number of outputs it yields.

    A `Layer` numbered l (of size N_l) applies a non-linear transformation on inputs a_{l-1} of size N_{l-1}
    using an `ActivationFunction` and some parameters :

        z_l = W_{l-1,l} a_{l-1} + b_l
        a_l = σ_l(z_l)

    W_{l-1,l} and b_l  are the parameters (respectively the weights and are biases)
    whilst σ_l is the `ActivationFunction`.

    `Layers` can be stacked (added) to `Network` using `Network.stack`.
    """

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

        # Last gradient
        self._last_d_W = 0
        self._last_d_b = 0

        # Adam parameters
        self.eps = 10 ** (-8)
        self.t = 1
        self.beta1_w = 0.9
        self.beta2_w = 0.999
        self.m_w = 0
        self.v_w = 0

        self.beta1_b = 0.9
        self.beta2_b = 0.999
        self.m_b = 0
        self.v_b = 0

    def __str__(self):
        string = f" - {self.name}\n"
        string += f"  - Size : {self.size}\n"
        string += f"  - # Parameters : {self.get_num_parameters()}\n"
        string += f"  - Activation Function : {self._activation_function}\n"
        string += f"  - Dims : {self.dims}\n"
        string += f"  - W shape : {self.W.shape}\n"
        return string

    @property
    def dims(self) -> (int, int):
        """
        :return: the shape of W
        """
        return self.size, self._previous_layer_size

    @staticmethod
    def from_W_b(previous_layer_size: int, W: np.ndarray, b: np.ndarray):
        """
        Factory of layer : create a Layer from given array of weights and bias.

        :param previous_layer_size: the size of the previous layer.
        :param W: the matrix of weights to use
        :param b: the vector of biases to use
        :return: a `Layer` using this parameters
        """
        size = b.shape[0]
        layer = Layer(size)
        layer._previous_layer_size = previous_layer_size
        layer.W = W
        layer.b = b
        layer.delta_l = layer.b * 0
        layer.a_lm1 = np.zeros((previous_layer_size, 1))
        layer._initialised = True

        return layer

    def forward_propagate(self, inputs: np.ndarray, persist: bool) -> np.ndarray:
        """
        Returns to output of an inputs.

        Persists the inputs and the affine transformation results.

        :param inputs: the considered inputs of size (self._previous_layer_size, n_sample)
        :return: the associated outputs of size (self.size, n_sample)
        """
        assert self._initialised

        assert inputs.shape[0] == self._previous_layer_size
        if persist:
            self.a_lm1 = inputs

        # Affine transform
        z_array = self.W.dot(inputs)
        # NOTE : Double transposition, works for now but not nice
        z_array = np.add(z_array.T, self.b).T

        if persist:
            self.z_l = z_array

        outputs = self._activation_function.value(z_array)

        assert self.z_l.shape[0] == self.size

        return outputs

    def initialise(self, previous_layer_size: int) -> (int, int):
        """
        Initialise the `Layer` (in a network) with respect to the surrounding layers

        :param previous_layer_size: size of the previous layer considered
        :return: the dimensions of W
        """
        self._previous_layer_size = previous_layer_size
        # He-et-al initialization for the weights
        self.W = np.random.randn(self.dims[0], self.dims[1]) * np.sqrt(2 / self._previous_layer_size)
        self.delta_l = self.b * 0
        self.a_lm1 = np.zeros((previous_layer_size, 1))
        self._initialised = True
        return self.dims

    def back_propagate(self, W_T_lp1: np.ndarray, delta_lp1: np.ndarray):
        """
        Compute the error `delta_l` at the inputs based on the error in the outputs.

        Persist the error `delta_l`.

        :param W_T_lp1: the matrix of weights of the next layer
        :param delta_lp1: the error at the inputs.
        :return:
        """
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
        """
        Computes the gradients for the matrix of weights using the persisted
        information during forward and backpropagation.

        More precisely the mean of the gradients is actually returned as there
        can be more than one inputs

        :return: the (average) gradient of W
        """
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
        """
        Computes the gradients for the vector of biases using the persisted
        information backpropagation.

        More precisely the mean of the gradients is actually returned as there
        can be more than one inputs

        :return: the (average) gradient of b
        """
        return self.delta_l.mean(axis=1)

    def optimize(self, learning_rate: float, momentum: float):
        """
        Optimisation routine: Updates the parameters using gradient
        descent with momentum.

        Persists the gradient computed for the next iteration.

        :param learning_rate: the learning rate to use
        :param momentum: the proportion of the last gradient to add
        """
        # Getting gradients
        d_W = self.get_d_W()
        d_b = self.get_d_b()

        # Adam update (working but not nice for now)
        self.m_w = self.beta1_w * self.m_w + (1 - self.beta1_w) * d_W
        self.v_w = self.beta2_w * self.v_w + (1 - self.beta2_w) * (
                    d_W * d_W)
        m_w_cap = self.m_w / (1 - (self.beta1_w ** self.t))
        v_w_cap = self.v_w / (1 - (self.beta2_w ** self.t))

        self.m_b = self.beta1_b * self.m_b + (1 - self.beta1_b) * d_b
        self.v_b = self.beta2_b * self.v_b + (1 - self.beta2_b) * (
                    d_b * d_b)
        m_b_cap = self.m_b / (1 - (self.beta1_b ** self.t))
        v_b_cap = self.v_b / (1 - (self.beta2_b ** self.t))

        self.t += 1

        # Updating parameters
        self.W -= learning_rate * m_w_cap / (np.sqrt(v_w_cap) + self.eps)
        self.b -= learning_rate * m_b_cap / (np.sqrt(v_b_cap) + self.eps)

    def get_num_parameters(self) -> int:
        """
        :return: the number of parameters in the layer
        """
        num_parameter = 0
        num_parameter += self.b.size
        num_parameter += self.W.shape[0] * self.W.shape[1]
        return num_parameter


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


class ReLuMSEOutputLayer(Layer):

    def __init__(self, size: int):
        super().__init__(size, activation_function=ReLu(), name="SimpleMSEOutputLayer")

    def __str__(self):
        string = super().__str__()
        string += f"  - Cost Function : Mean Squared Error\n"
        return string

    @staticmethod
    def from_W_b(previous_layer_size: int, W: np.ndarray, b: np.ndarray):
        size = b.shape[0]
        layer = ReLuMSEOutputLayer(size)

        if W.shape[1] != previous_layer_size:
            raise RuntimeError("The dimension are non-consistent:\n"
                               f"W.shape[1] = {W.shape[1]} != previous_layer_size = {previous_layer_size}")

        layer._previous_layer_size = previous_layer_size
        layer.b = b
        layer.W = W
        layer._initialised = True

        return layer

    @staticmethod
    def cost(y, y_hat) -> float:
        n_sample = y.shape[1]
        value = - np.linalg.norm(y - y_hat) ** 2 / (2 * n_sample)

        return value

    def back_propagate(self, y: np.ndarray, **kwargs) -> (np.ndarray, np.ndarray):
        # Value of the error with the Identity/MSE combo
        y_hat = self._activation_function.value(self.z_l)
        der = self._activation_function.der(self.z_l)
        delta_l = (y_hat - y) * der
        assert (delta_l.shape[0] == self.size)

        W_T_delta_l = self.W.T.dot(delta_l)

        self.delta_l = delta_l

        return W_T_delta_l, delta_l
