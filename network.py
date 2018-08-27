import numpy as np
import warnings
from functions import ActivationFunction, SoftMax, CrossEntropy, CostFunction
from layer import Layer


class Network:
    """A `Network` is a collections of `Layers` that learns a mapping
    from an input space to an output space.

    A `Network` is defined using an input_size.
    `Layers` can be then stacked in the network using `network.stack`.
    After `Layers` being stacked, the output can be defined using `network.output`.
    Finally, the `Network` can be trained and test using `network.train` and `network.test`
    on given datasets.

    """

    def __init__(self, input_size):
        self.layers = []
        self.input_size = input_size
        self._output_size = 0 # to be determined then
        self.done_constructing = False
        self.times_trained = 0
        self.batch_size = 32

        self.cost_function = CrossEntropy()

    def stack(self, layer: Layer):
        """
        Add a layer to the network
        :param layer: a new Layer to add

        :return: the same network
        """
        self.layers.append(layer)
        return self

    def output(self, output_size, output_function: ActivationFunction = SoftMax(),
               cost_function: CostFunction = CrossEntropy()):
        """
        Specify the output layer of the network and its property.

        From there on, the network can be trained and used.

        :param output_size: the output size of the network
        :param output_function: the ActivationFunction of the last layer (defaulted to SoftMax)
        :param cost_function: the CostFunction to use (defaulted to CrossEntropy)

        :return: the same network (now initialized)
        """
        if self.done_constructing:
            raise RuntimeError("Network already set: output() called twice")

        self._output_size = output_size
        output_layer = Layer(size=output_size, activation_function=output_function)
        self.stack(output_layer)
        self.cost_function = cost_function

        previous_layer_size = self.input_size
        dims = []
        for layer in self.layers:
            dims.append(layer.initialise(previous_layer_size))
            previous_layer_size = layer.size

        self.done_constructing = True

        return self

    def _batcher(self, n_samples):
        """
        A simple generator of indices to access batches of the data.

        :param n_samples: the total number of the data given.
        """
        n_batches = n_samples // self.batch_size
        n_batches += 1 * (n_samples % self.batch_size != 0)
        for i in range(n_batches):
            yield range(i * self.batch_size, min(n_samples, (i + 1) * self.batch_size) - 1)

    def _prepropagation_check(self, x_array: np.ndarray, y_array: np.ndarray):
        """
        Some consistency verifications before propagation on the dimensions of inputs
        and on the their compatibility with respect to the network properties.

        :param x_array:
        :param y_array:
        """
        if not self.done_constructing:
            raise RuntimeError("Network not yet initialised : define output layer using output()")

        # Checking samples consistency
        assert (x_array.shape[1] == y_array.shape[1])

        # Checking dimensions consistency
        assert (x_array.shape[0] == self.input_size)
        assert (y_array.shape[0] == self.layers[-1].size)

    def train(self, x_train: np.ndarray, y_train: np.ndarray, num_epochs=10):
        """
        Train the network using the provided data.

        :param x_train: a np.ndarray of size (input_size, n_samples)
        :param y_train: a np.ndarray of size (output_size, n_samples
        :param num_epochs: the number of epoches used to train the network

        :return: the same network trained one more time
        """

        self._prepropagation_check(x_train, y_train)

        n_sample = x_train.shape[1]

        print(f"Training the network for the {self.times_trained+1} time")

        for epoch in range(num_epochs):
            print(f" - Epoch {epoch+1} / {num_epochs}", end=" | ")

            for batch_indices in self._batcher(n_sample):
                x_batch = x_train[:, batch_indices]
                y_batch = y_train[:, batch_indices]
                y_value, y_pred = self._forward_propagation(x_batch)
                cost = self.cost_function.value(y_value, y_batch)
                print("Cost : ",cost)
                error_signal = self.cost_function.der(y_value, y_batch)
                assert error_signal.shape[0] == self._output_size
                self._back_propagation(error_signal)
                self._optimize()

        self.times_trained += 1

        return self

    def test(self, x_test: np.ndarray, y_test: np.ndarray):
        """
        Test the network using the provided data.

        :param x_test: a np.ndarray of size (input_size, n_samples)
        :param y_test: a np.ndarray of size (output_size, n_samples

        :return: the precision of predictions, outputs values and their associated predictions.
        """

        self._prepropagation_check(x_test, y_test)

        if self.times_trained == 0:
            warnings.warn("The network has not been trained yet: results will be fuzy!")

        n_sample = x_test.shape[1]
        y_value = y_test * 0
        y_pred = np.ndarray.astype(y_test * 0, int)

        for batch_indices in self._batcher(n_sample):
            x_batch = x_test[:, batch_indices]
            y_value[:, batch_indices], y_pred[:, batch_indices] = self._forward_propagation(x_batch)

        prec = np.mean(1 * (y_pred == y_test))

        return prec, y_value, y_pred

    def _forward_propagation(self, inputs: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Computes the values and predictions for given inputs.

        :param inputs: inputs as a (input_size, n_samples) np.ndarray
        :return: outputs values and associated predictions
        """
        x_array = inputs

        for layer in self.layers:
            x_array = layer.forward_propagate(x_array)

        y_values = x_array
        y_preds = np.rint(y_values)

        return y_values, y_preds

    def _back_propagation(self, error_signals: np.ndarray):
        """
        Propagate the error signals in the network

        :param error_signals: error signals as a (output_size, n_samples) np.ndarray
        """
        x_array = error_signals

        for layer in reversed(self.layers):
            x_array = layer._back_propagate(x_array)

    def __str__(self):
        string = "=========================\n"
        string += "Basic simple network\n"
        string += "Layers:\n"
        for (i, layer) in enumerate(self.layers):
            string += f" - Layer #{i}"
            string += str(layer)
            string += "\n"

        string += f"Cost Function : {self.cost_function}"
        string += "\n=========================\n"
        return string

    def _optimize(self):
        learning_rate = 0.01
        for layer in self.layers:
            layer._optimize(learning_rate)

