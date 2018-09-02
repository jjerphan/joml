from abc import ABC, abstractmethod
import csv
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter


class Logger(ABC):
    @abstractmethod
    def log_cost_accuracy(self, n_epoch, cost, accuracy):
        pass


class SilentLogger(Logger):

    def log_cost_accuracy(self, n_epoch, cost, accuracy):
        pass


class StdOutLogger(Logger):

    def log_cost_accuracy(self, n_epoch, cost, accuracy):
        print(" - Epoch :", n_epoch)
        print(" - Cost : ", cost)
        print(" - Accuracy", accuracy)
        print()


class CSVLogger(Logger):

    def __init__(self, csv_file):
        self.csv_file = csv_file
        fd = open(self.csv_file, "a+")
        writer = csv.writer(fd)
        writer.writerow(["n_epoch", "cost", "accuracy"])
        fd.close()

    def log_cost_accuracy(self, n_epoch, cost, accuracy):
        with open(self.csv_file, 'a') as fd:
            writer = csv.writer(fd)
            writer.writerow([n_epoch, cost, accuracy])


class MatplotlibLogger(Logger):

    def __init__(self):
        self.cost_vals = []
        self.accuracy_vals = []

    @staticmethod
    def _plot_metric(metric_vals, metric_name, export_file=None):
        t = np.arange(len(metric_vals))

        fig, ax = plt.subplots()
        ax.plot(t, metric_vals)

        ax.set(xlabel="Epochs", ylabel=metric_name, title='Training pork')
        ax.grid()

        if export_file is not None:
            fig.savefig(export_file)
        plt.show()

    def log_cost_accuracy(self, n_epoch, cost, accuracy):
        self.cost_vals.append(cost)
        self.accuracy_vals.append(accuracy)

    def plot_cost(self, ): self._plot_metric(self.cost_vals, "Cost")

    def plot_accuracy(self, ): self._plot_metric(self.accuracy_vals, "Accuracy")

    def reset(self):
        self.cost_vals = []
        self.accuracy_vals = []


class BenchmarkLogger:

    def __init__(self, csv_file_name: str):
        self.csv_file_name = csv_file_name
        self.train_cost_vals = []
        self.test_cost_vals = []
        self.train_acc_vals = []
        self.test_acc_vals = []

    def reset(self):
        self.train_cost_vals = []
        self.test_cost_vals = []
        self.test_acc_vals = []
        self.test_acc_vals = []

    def dump_results(self):
        if self.csv_file_name is None:
            pass

        fd = open(self.csv_file_name, "a+")
        writer = csv.writer(fd)
        writer.writerow(["n_iter", "train_cost", "test_cost", "train_acc", "test_acc"])
        for i, res in enumerate(
                zip(self.train_cost_vals, self.test_cost_vals, self.train_acc_vals, self.test_acc_vals)):
            writer.writerow([i] + list(res))

        fd.close()

    def benchmark_log(self, train_cost, train_acc, test_cost, test_acc):
        self.train_cost_vals.append(train_cost)
        self.train_acc_vals.append(train_acc)
        self.test_cost_vals.append(test_cost)
        self.test_acc_vals.append(test_acc)

    def plot_benchmark(self):
        x = np.arange(len(self.train_cost_vals))

        # Creating 2 subplots
        plt.figure(1)

        # Plotting Costs
        plt.subplot(211)
        plt.plot(x, self.train_cost_vals)
        plt.plot(x, self.test_cost_vals)
        plt.yscale('linear')
        plt.title('Cost')
        plt.legend(['Training Cost', 'Testing Cost'], loc='upper left')
        plt.grid(True)

        # Plotting Accuracies
        plt.subplot(212)
        plt.plot(x, self.train_acc_vals)
        plt.plot(x, self.test_acc_vals)
        plt.yscale('linear')
        plt.title('Accuracy')
        plt.legend(['Training Accuracy', 'Testing Accuracy'], loc='upper left')
        plt.grid(True)

        # Format the minor tick labels of the y-axis into empty strings with
        # `NullFormatter`, to avoid cumbering the axis with too many labels.
        plt.gca().yaxis.set_minor_formatter(NullFormatter())

        plt.show()
