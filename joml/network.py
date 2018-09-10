import numpy as np
import warnings

from joml.layer import Layer, SoftMaxCrossEntropyOutputLayer
from joml.logger import StdOutLogger, Logger, BenchmarkLogger
from joml.utils import one_hot


class Network:
    """
    A `Network` is a collections of `Layers` that learns a mapping
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
        self._times_trained = 0
        self.batch_size = 32

        self.name = name
        self.logger = StdOutLogger()

    def __str__(self):
        string = "=========================\n"
        string += f"{self.name}\n"
        string += f" - Input size: {self.input_size}\n"
        string += f" - # Parameters : {self.get_num_parameters()}\n"
        string += f" - Times trained: {self._times_trained}\n"
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
    def create_from_Ws_and_bs(Ws: list, bs: list):
        """
        Creates a network from a list on weights and biases.
        Each `Layer` gets created (in the order) according to a pair of zip(W, b).
        For now, the output layer is a `SoftMaxCrossEntropyOutputLayer`.

        :param Ws: list of weights to use in layers
        :param bs: list of biases to use in networks
        :return: a `Network` with the given architecture
        """
        if len(Ws) != len(bs):
            raise RuntimeError("Ws and bs don't have the same number of elements:\n"
                               f"len(Ws) = {len(Ws)} != len(bs)={len(bs)}")

        input_size = Ws[0].shape[1]
        network = Network(input_size=input_size)

        # We keep the last parameters for the output layer
        last_W = Ws.pop()
        last_b = bs.pop()

        previous_layer_size = input_size
        packs = zip(Ws, bs)
        for W, b in packs:
            layer = Layer.from_W_b(previous_layer_size, W, b)
            previous_layer_size = layer.size
            network.stack(layer)

        network._output_layer = SoftMaxCrossEntropyOutputLayer.from_W_b(previous_layer_size, last_W, last_b)
        network.done_constructing = True

        return network

    def with_logger(self, logger: Logger):
        """
        Specify a `Logger` to use for the network.

        :param logger: the logger to use
        :return: the same but modified `Network`
        """
        self.logger = logger
        return self

    def stack(self, layer: Layer):
        """
        Stack (append) a layer to the Network

        :param layer:
        :return: the same `Network` but with this new layer
        """
        self.layers.append(layer)
        return self

    def output(self, output_layer: SoftMaxCrossEntropyOutputLayer):
        """
        Specify the output layer to use and initialise the `Network`.

        The `Network` can now be trained and test.

        :param output_layer: the OutputLayer to use.
        :return:
        """
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

    def train(self, x_train: np.ndarray, y_train: np.ndarray, num_epochs=10, verbose=True,
              learning_rate=0.01, momentum=0.9):
        """
        Train a `Network` using the data provided for a given number of epochs.

        An epoch correspond to a full-pass on the data-set for a training.

        :param x_train: the inputs to use to train
        :param y_train: the labels to use to train
        :param num_epochs: the number of epochs for the training
        :param verbose: if true, logs progress
        :param learning_rate: parameter for the gradient descent
        :param momentum: parameter to add momentum to gradient
        :return: the same `Network` but trained one more time
        """
        self._prepropagation_check(x_train, y_train)

        def printv(t): not verbose or print(t)

        # If the dataset only consists of one example, it is represented as a vector
        # If it is the case, we change it to be a matrix so that the processing is the same
        if len(x_train.shape) == 1:
            x_train = x_train[:, np.newaxis]
            y_train = y_train[:, np.newaxis]

        n_sample = x_train.shape[1]

        printv(f"Training the network for the {self._times_trained+1} time")

        for n_epoch in range(1, num_epochs + 1):
            printv(f"| Epoch {n_epoch} / {num_epochs}")

            accuracy, cost = 0., 0.

            for n_b, batch_indices in enumerate(self._batcher(self.batch_size, n_sample)):
                x_batch = x_train[:, batch_indices]
                y_batch = y_train[:, batch_indices]

                y_hat = self._forward_propagation(x_batch)

                y_pred = one_hot(y_hat.argmax(axis=0))
                accuracy = np.mean(1 * (y_pred == y_batch))

                cost = self._output_layer.cost(y_hat, y_batch)

                assert y_hat.shape[0] == self._output_layer.size
                assert y_batch.shape[0] == self._output_layer.size

                self._back_propagation(y_batch)
                self._optimize(learning_rate=learning_rate, momentum=momentum)

            self.logger.log_cost_accuracy(n_epoch, cost, accuracy)

        self._times_trained += 1

        return self

    def test(self, x_test: np.ndarray, y_test: np.ndarray, warn=True):
        """
        Test a `Network` using the data provided for a given number of epochs.

        An epoch correspond to a full-pass on the data-set for a training.
        :param x_test: the inputs used to test
        :param y_test: the labels used to test
        :param warn: if true, warn in case of a `Network` not having been trained.
        :return: the predictions, the outputs and associated accuracy
        """
        self._prepropagation_check(x_test, y_test)

        if warn and self._times_trained == 0:
            warnings.warn("The network has not been trained yet: results will be fuzzy!")

        # If the dataset only consists of one example, it is represented as a vector
        # If it is the case, we change it to be a matrix so that the processing is the same
        if len(x_test.shape) == 1:
            x_test = x_test[:, np.newaxis]
            y_test = y_test[:, np.newaxis]

        n_sample = x_test.shape[1]

        # Outputs of the networks
        y_hat = y_test * 0

        for batch_indices in self._batcher(self.batch_size, n_sample):
            x_batch = x_test[:, batch_indices]
            # Here, we don't persist the results calculate during the forward
            # propagation because results are persisted uniquely for training
            y_hat[:, batch_indices] = self._forward_propagation(x_batch, persist=False)

        # Doing an hard max on the output to find the prediction
        y_pred = one_hot(y_hat.argmax(axis=0), num_classes=self._output_layer.size)
        accuracy = np.mean(1 * (y_pred == y_test))

        return y_pred, y_hat, accuracy

    def benchmark(self, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray,
                  num_epochs=10, verbose=True, csv_file_name=None, learning_rate=0.01, momentum=0.9, warn=True):
        """
        Benchmark a network. This consist of training a network with dataset (x_train,_train)
        from scratch and testing it at each iteration with dataset (x_test, y_test)

        An iteration correspond to the processing of a mini-batch.

        This routine can be slow has testing is done at each iteration.

        A `BenchmarkLogger` is updated with the logs of the benchmark and can be then used to plot
        the results.

        :param x_train: the inputs to use for the training
        :param y_train: the labels to use for the training
        :param x_test: the inputs to use for the training
        :param y_test: the labels to use for the training
        :param num_epochs: the number of epochs to perform
        :param verbose: if true, logs progress
        :param csv_file_name: the csv file to use to persist the log
        :param learning_rate: parameter for the optimisation routine
        :param momentum: parameter to add momentum to gradients
        :param warn: if true, warns about the network being already trained
        :return: a `BenchmarkLogger` containing logs of the benchmark
        """
        self._prepropagation_check(x_train, y_train)

        def printv(t): not verbose or print(t)

        if warn and self._times_trained > 0:
            warnings.warn("The network has already been trained: results might not be representative for the benchmark")

        # If the dataset only consists of one example, it is represented as a vector
        # If it is the case, we change it to be a matrix so that the processing is the same
        if len(x_train.shape) == 1:
            x_train = x_train[:, np.newaxis]
            y_train = y_train[:, np.newaxis]

        n_sample = x_train.shape[1]

        printv(f"Training the network for the {self._times_trained+1} time")

        logger = BenchmarkLogger(csv_file_name=csv_file_name)
        for n_epoch in range(1, num_epochs + 1):
            printv(f"| Epoch {n_epoch} / {num_epochs}")

            train_cost, test_cost, train_acc, test_acc = 0,0,0,0
            for n_b, batch_indices in enumerate(self._batcher(self.batch_size, n_sample)):
                x_batch = x_train[:, batch_indices]
                y_batch = y_train[:, batch_indices]

                y_hat = self._forward_propagation(x_batch)

                y_pred = one_hot(y_hat.argmax(axis=0))
                train_acc = np.mean(1 * (y_pred == y_batch))

                train_cost = self._output_layer.cost(y_hat, y_batch)

                assert y_hat.shape[0] == self._output_layer.size
                assert y_batch.shape[0] == self._output_layer.size

                self._back_propagation(y_batch)
                self._optimize(learning_rate=learning_rate, momentum=momentum)

                y_pred_test, y_hat_test, test_acc = self.test(x_test, y_test, warn=False)

                test_cost = self._output_layer.cost(y_hat_test, y_test)

                logger.benchmark_log(train_cost=train_cost, train_acc=train_acc, test_cost=test_cost, test_acc=test_acc)
            print("Training Cost:", train_cost)
            print("Testing Cost:", test_cost)
            print("Training Accuracy:", train_acc)
            print("Testing Accuracy:", test_acc)
        return logger

    def get_Ws_bs(self) -> (list, list):
        """
        :return: a tuple of the list of weights and the list of bias to the `Network`.
        """
        if not self.done_constructing:
            raise RuntimeError("The Network has not been complety constructed yet.")

        Ws = []
        bs = []
        for l in self.layers:
            Ws.append(l.W)
            bs.append(l.b)

        Ws.append(self._output_layer.W)
        bs.append(self._output_layer.b)

        return Ws, bs

    def get_dWs_dbs(self) -> (list, list):
        """
        :return: a tuple of two lists : one of the last gradients of weights, the other for the biases
        """
        if not self.done_constructing:
            raise RuntimeError("The Network has not been complety constructed yet.")

        d_Ws = []
        d_bs = []
        for l in self.layers:
            d_Ws.append(l.get_d_W())
            d_bs.append(l.get_d_b())

        d_Ws.append(self._output_layer.get_d_W())
        d_bs.append(self._output_layer.get_d_b())

        return d_Ws, d_bs

    def get_num_parameters(self)->int:
        """
        :return: the total number of parameters in the network
        """
        num_parameters = 0
        for layer in self.layers:
            num_parameters += layer.get_num_parameters()

        num_parameters += self._output_layer.get_num_parameters()

        return num_parameters

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
        have_one_sample = (len(x_array.shape) == 1 and len(y_array.shape) == 1)
        have_same_number_samples = have_one_sample or x_array.shape[1] == y_array.shape[1]
        assert have_same_number_samples

        # Checking dimensions consistency
        assert (x_array.shape[0] == self.input_size)
        assert (y_array.shape[0] == self._output_layer.size)

    def _forward_propagation(self, inputs: np.ndarray, persist=True) -> np.ndarray:
        x_array = inputs

        for layer in self.layers:
            x_array = layer.forward_propagate(x_array, persist=persist)

        y_hat = self._output_layer.forward_propagate(x_array, persist=persist)

        # Test the consistency w.r.t samples
        # Some boilerplate code here as we need to check both the case of
        # a single vector (only one sample) and the case of a matrix (multiple samples)
        have_one_sample = (len(y_hat.shape) == 1 and len(inputs.shape) == 1)
        have_same_number_samples = have_one_sample or y_hat.shape[1] == inputs.shape[1]
        assert have_same_number_samples
        assert (y_hat.shape[0] == self._output_layer.size)

        return y_hat

    def _back_propagation(self, y: np.ndarray):

        # TODO : for now, the transposed matrix of weights is passed
        # from one layer to another
        W_T_l, delta_l = self._output_layer.back_propagate(y)

        for layer in reversed(self.layers):
            assert (W_T_l.shape[0] == layer.size)
            W_T_l, delta_l = layer.back_propagate(W_T_l, delta_l)

    def _optimize(self, learning_rate=0.01, momentum=0.9):
        self._output_layer.optimize(learning_rate, momentum)
        for layer in self.layers:
            layer.optimize(learning_rate, momentum)
