import numpy as np
import warnings

from joml.layer import Layer, SoftMaxCrossEntropyOutputLayer
from joml.logger import StdOutLogger, Logger
from joml.utils import one_hot


class Network:
    """A `Network` is a collections of `Layers` that learns a mapping
    from an input space to an output space.

    A `Network` is defined using an input_size.
    `Layers` can be then stacked in the network using `network.stack`.

    After `Layers` being stacked, the output can be defined using `network.output`.
    Finally, the `Network` can be trained and test using `network.train` and `network.test`
    on given datasets.

    A `Network` can also embedded a specific `Logger` to output result in the standard output
    or in a csv file for example. See `network.with_logger` and `Loggers`

    A `Network` can also be constructed from given weights matrices and bias vectors
    using the static method `Network.create_from_W_b`

    Weights and biases of a network can also be extracted using `network.get_Ws_bs`.
    """

    def __init__(self, input_size, name="Simple Network"):
        self.layers = []
        self.input_size = input_size
        self._output_layer = SoftMaxCrossEntropyOutputLayer(size=2)
        self.done_constructing = False
        self.times_trained = 0
        self.batch_size = 32

        self.name = name
        self.logger = StdOutLogger()

    def __str__(self):
        string = "=========================\n"
        string += f"{self.name}\n"
        string += f" - Input size: {self.input_size}\n"
        string += f" - Times trained: {self.times_trained}\n"
        string += f" - Batch size: {self.batch_size}\n"
        string += "\nLayers:\n"
        for (i, layer) in enumerate(self.layers):
            string += f" - Layer #{i+1}"
            string += str(layer)
            string += "\n"

        string += str(self._output_layer)

        string += "\n=========================\n"
        return string

    @staticmethod
    def create_from_Ws_and_bs(weights_matrices: list, biases_vectors: list):

        input_size = weights_matrices[0].shape[1]
        network = Network(input_size=input_size)

        last_W = weights_matrices.pop()
        last_b = biases_vectors.pop()

        previous_layer_size = input_size
        packs = zip(weights_matrices, biases_vectors)
        for W, b in packs:
            layer = Layer.from_W_b(previous_layer_size, W, b)
            previous_layer_size = layer.size
            network.stack(layer)

        network._output_layer = SoftMaxCrossEntropyOutputLayer.from_W_b(previous_layer_size, last_W, last_b)
        network.done_constructing = True

        return network

    def with_logger(self, logger: Logger):
        self.logger = logger
        return self

    def stack(self, layer: Layer):
        self.layers.append(layer)
        return self

    def output(self, output_layer: SoftMaxCrossEntropyOutputLayer):
        if self.done_constructing:
            raise RuntimeError("Network already set: output() called twice")

        self._output_layer = output_layer

        previous_layer_size = self.input_size
        dims = []
        for layer in self.layers:
            dims.append(layer.initialise(previous_layer_size))
            previous_layer_size = layer.size

        self._output_layer.initialise(previous_layer_size)

        self.done_constructing = True

        return self

    def train(self, x_train: np.ndarray, y_train: np.ndarray, num_epochs=10, verbose=True):
        self._prepropagation_check(x_train, y_train)

        def printv(t):
            if(verbose):
                print(t)

        n_sample = x_train.shape[1]

        printv(f"Training the network for the {self.times_trained+1} time")

        for epoch in range(num_epochs):
            printv(f"| Epoch {epoch+1} / {num_epochs}")

            for n_b, batch_indices in enumerate(self._batcher(self.batch_size, n_sample)):
                x_batch = x_train[:, batch_indices]
                y_batch = y_train[:, batch_indices]

                y_hat = self._forward_propagation(x_batch)

                y_pred = one_hot(y_hat.argmax(axis=0)).T
                accuracy = np.mean(1 * (y_pred == y_batch))

                cost = self._output_layer.cost(y_hat, y_batch)

                assert y_hat.shape[0] == self._output_layer.size
                assert y_batch.shape[0] == self._output_layer.size

                self._back_propagation(y_batch)
                self._optimize()

            self.logger.log_cost_accuracy(n_b, cost, accuracy)

        self.times_trained += 1

        return self

    def test(self, x_test: np.ndarray, y_test: np.ndarray):
        self._prepropagation_check(x_test, y_test)

        if self.times_trained == 0:
            warnings.warn("The network has not been trained yet: results will be fuzy!")

        n_sample = x_test.shape[1]
        y_hat = y_test * 0

        for batch_indices in self._batcher(n_sample):
            x_batch = x_test[:, batch_indices]
            y_hat[:, batch_indices] = self._forward_propagation(x_batch)

        y_pred = one_hot(y_hat.argmax(axis=0)).T
        prec = np.mean(1 * (y_pred == y_test))

        return y_pred, y_hat, prec

    def get_Ws_bs(self) -> (list,list):
        Ws = []
        bs = []
        for l in self.layers:
            Ws.append(l.W)
            bs.append(l.b)

        Ws.append(self._output_layer.W)
        bs.append(self._output_layer.b)

        return Ws, bs

    # =============== #
    # Private methods #
    # =============== #

    @staticmethod
    def _batcher(batch_size, n_samples, shuffle=True):
        n_batches = n_samples // batch_size
        n_batches += 1 * (n_samples % batch_size != 0)
        indices = np.arange(n_samples)
        if shuffle:
            np.random.shuffle(indices)
        for i in range(n_batches):
            start_batch = i * batch_size
            end_batch = min(n_samples, (i + 1) * batch_size)
            yield indices[start_batch:end_batch]

    def _prepropagation_check(self, x_array: np.ndarray, y_array: np.ndarray):
        if not self.done_constructing:
            raise RuntimeError("Network not yet initialised : define output layer using output()")

        # Checking samples consistency
        assert (x_array.shape[1] == y_array.shape[1])

        # Checking dimensions consistency
        assert (x_array.shape[0] == self.input_size)
        assert (y_array.shape[0] == self._output_layer.size)

    def _forward_propagation(self, inputs: np.ndarray) -> (np.ndarray, np.ndarray):
        x_array = inputs

        for layer in self.layers:
            x_array = layer.forward_propagate(x_array)

        y_hat = self._output_layer.forward_propagate(x_array)

        # Test if same shape, boilerplate here for vector vs matrices
        have_one_sample = (len(y_hat.shape) == 1 and len(inputs.shape) == 1)
        have_same_number_samples = have_one_sample or y_hat.shape[1] == inputs.shape[1]
        assert have_same_number_samples
        assert (y_hat.shape[0] == self._output_layer.size)

        return y_hat

    def _back_propagation(self, y: np.ndarray):
        W_T_l, delta_l = self._output_layer.back_propagate(y)
        for layer in reversed(self.layers):
            assert (W_T_l.shape[0] == layer.size)
            W_T_l, delta_l = layer.back_propagate(W_T_l, delta_l)

    def _optimize(self):
        learning_rate = 0.01

        self._output_layer.optimize(learning_rate)
        for layer in self.layers:
            layer.optimize(learning_rate)
