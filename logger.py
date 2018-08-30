from abc import ABC, abstractmethod
import csv
import matplotlib.pyplot as plt
import numpy as np


class Logger(ABC):
    @abstractmethod
    def log_cost_accuracy(self, n_batch, cost, accuracy):
        pass


class SilentLogger(Logger):

    def log_cost_accuracy(self, n_batch, cost, accuracy):
        pass


class StdOutLogger(Logger):

    def log_cost_accuracy(self, n_batch, cost, accuracy):
        print(" - Batch :", n_batch)
        print(" - Cost : ", cost)
        print(" - Accuracy", accuracy)
        print()


class CSVLogger(Logger):

    def __init__(self, csv_file):
        self.csv_file = csv_file
        fd = open(self.csv_file, "a+")
        writer = csv.writer(fd)
        writer.writerow(["n_batch", "cost", "accuracy"])
        fd.close()

    def log_cost_accuracy(self, n_batch, cost, accuracy):
        with open(self.csv_file, 'a') as fd:
            writer = csv.writer(fd)
            writer.writerow([n_batch, cost, accuracy])


class MatplotlibLogger(Logger):

    def __init__(self):
        self.cost_vals = []
        self.accuracy_vals = []

    def log_cost_accuracy(self, n_batch, cost, accuracy):
        self.cost_vals.append(cost)
        self.accuracy_vals.append(accuracy)

    def plot_cost(self, ): self._plot_metric(self.cost_vals, "Cost")

    def plot_accuracy(self, ): self._plot_metric(self.accuracy_vals, "Accuracy")

    def _plot_metric(self, metric_vals, metric_name, export_file=None):
        t = np.arange(len(metric_vals))

        fig, ax = plt.subplots()
        ax.plot(t, metric_vals)

        ax.set(xlabel="Epochs", ylabel=metric_name, title='Training pork')
        ax.grid()

        if export_file is not None:
            fig.savefig(export_file)
        plt.show()

    def reset(self):
        self.cost_vals = []
        self.accuracy_vals = []
