from abc import ABC, abstractmethod
import csv


class Logger(ABC):

    @abstractmethod
    def log_cost_accuracy(self, n_batch, cost, accuracy):
        pass


class SimpleLogger(Logger):

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
