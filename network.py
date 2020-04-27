import random
import logging
from train import train_and_score

class Network():
    def __init__(self, nn_param_population=None):
        self.accuracy = 0.
        self.nn_param_population = nn_param_population
        self.network = {}  # (dic): represents MLP network parameters

    def create_random(self):
        for key in self.nn_param_population:
            self.network[key] = random.choice(self.nn_param_population[key])

    def create_set(self, network):
        self.network = network

    def train(self, dataset):
        if self.accuracy == 0.:
            self.accuracy = train_and_score(self.network, dataset)

    def print_network(self):
        logging.info(self.network)
        logging.info("Network accuracy: %.2f%%" % (self.accuracy * 100))
